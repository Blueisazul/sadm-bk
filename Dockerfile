FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV TF_USE_LEGACY_KERAS=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p models
RUN wget -O models/breast_ultrasound_model.keras "https://www.dropbox.com/scl/fi/7wnka80fx7c5ud4fm2lbn/breast_ultrasound_model.keras?rlkey=uwrly99nyv9q54bw5csgqipyp&st=v70v5j2s&dl=1"

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]