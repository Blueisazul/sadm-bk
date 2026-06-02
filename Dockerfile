FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias de procesamiento de imágenes y herramientas de red (wget)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copiar el archivo de requerimientos
COPY requirements.txt ./

# Actualizar pip e instalar dependencias
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar todo el contenido de tu repositorio al contenedor (excepto carpetas grandes locales)
COPY . .

# Asegurar que la carpeta models exista físicamente en el contenedor
RUN mkdir -p models

# TRUCO MAESTRO: Descarga directa del modelo desde Dropbox al contenedor usando dl=1
RUN wget -O models/breast_ultrasound_model.keras "https://www.dropbox.com/scl/fi/7wnka80fx7c5ud4fm2lbn/breast_ultrasound_model.keras?rlkey=uwrly99nyv9q54bw5csgqipyp&st=v70v5j2s&dl=1"

# Exponer el puerto
EXPOSE 8000

# Arrancar con un único worker para proteger la RAM
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]