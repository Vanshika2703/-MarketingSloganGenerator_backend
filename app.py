from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F
from tqdm import trange


app = Flask(__name__)

tokenizer = GPT2Tokenizer.from_pretrained("sloganGenerator")
model = GPT2LMHeadModel.from_pretrained("sloganGenerator")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, length, context, segments_tokens=None, num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0,
                    device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context

    with torch.no_grad():
        for _ in trange(length):

            inputs = {'input_ids': generated}
            if segments_tokens != None:
              inputs['token_type_ids'] = torch.tensor(segments_tokens[:generated.shape[1]]).unsqueeze(0).repeat(num_samples, 1)

            outputs = model(**inputs)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
    return generated


@app.route("/predict", methods=["POST"])
def main():
    try:
        product = request.form.get('product')
        description = int(request.form.get('description'))

    except Exception as e:
        return jsonify({'message': 'Invalid request'}), 500
    
    context = product+", "+description
    context_tkn = tokenizer.additional_special_tokens_ids[0]
    slogan_tkn = tokenizer.additional_special_tokens_ids[1]

    input_ids = [context_tkn] + tokenizer.encode(context)

    segments = [slogan_tkn] * 64
    segments[:len(input_ids)] = [context_tkn] * len(input_ids)

    input_ids += [slogan_tkn]

    # Move the model back to the CPU for inference:
    model.to(torch.device('cpu'))

    generated = sample_sequence(model, length=30, context=input_ids, segments_tokens=segments, num_samples=20)
    
    slogans = ""

    for g in generated:
        slogan = tokenizer.decode(g)
        slogan = slogan.split('<|endoftext|>')[0].split('<slogan>')[1]
        slogans = slogans + slogan + "\r\n"


    return slogans


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000")