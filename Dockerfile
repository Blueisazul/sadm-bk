FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias esenciales del sistema para procesamiento de imágenes
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar el archivo de requerimientos (que está en la raíz de tu repositorio)
COPY requirements.txt ./

# Actualizar pip e instalar dependencias
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar todo el contenido de tu repositorio al contenedor
COPY . .

# Render asignará dinámicamente un puerto en la variable $PORT (suele ser 10000 o 8000)
EXPOSE 8000

# CORRECCIÓN DE RUTA Y TRABAJADORES: 
# Cambiamos "backend.app.main" por "app.main:app" y limitamos a 1 worker para ahorrar RAM
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]