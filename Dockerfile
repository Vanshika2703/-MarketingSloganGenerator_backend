FROM vanshika27/slogan_generator:tag

WORKDIR /workspace

RUN pip install flask transformers torch tqdm

COPY . .

CMD ["python", "app.py"]
