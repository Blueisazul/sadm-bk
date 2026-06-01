FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias del sistema necesarias para procesamiento de imágenes (OpenCV/Pillow)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar los requerimientos desde la carpeta backend
COPY backend/requirements.txt ./

# Actualizar pip e instalar dependencias optimizadas
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar todo el contenido del proyecto al contenedor
COPY . .

# Render asigna dinámicamente un puerto, pero internamente Docker suele escuchar en el 10000 o el 8000
EXPOSE 8000

# CAMBIO CRÍTICO: Eliminamos Gunicorn y forzamos un único worker con Uvicorn directo
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]