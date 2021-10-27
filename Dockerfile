FROM vanshika27/slogan_generator:latest

WORKDIR /app

RUN pip install flask transformers torch tqdm

COPY . .

CMD ["python", "app.py"]
