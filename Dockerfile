FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# To run training instead, you can override this command. Example:
# > docker run --rm -it --gpus all -v $(pwd)/lora_output:/app/lora_output --env-file .env <image-name> accelerate launch lora_train.py
CMD ["python", "inference.py"]
