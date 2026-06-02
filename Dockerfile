FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias esenciales del sistema para procesamiento de imágenes y descargas
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copiar el archivo de requerimientos
COPY requirements.txt ./

# Actualizar pip e instalar las dependencias con versiones clásicas fijadas
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar todo el contenido del repositorio al contenedor
COPY . .

# Asegurar la existencia de la carpeta models
RUN mkdir -p models

# Descargar tu modelo desde Dropbox de forma limpia durante la compilación
RUN wget -O models/breast_ultrasound_model.keras "https://www.dropbox.com/scl/fi/7wnka80fx7c5ud4fm2lbn/breast_ultrasound_model.keras?rlkey=uwrly99nyv9q54bw5csgqipyp&st=v70v5j2s&dl=1"

# Puerto obligatorio para Hugging Face Spaces
EXPOSE 7860

# Arrancar la API apuntando al puerto de Hugging Face
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]