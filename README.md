# SADM Backend: API de Asistencia Diagnóstica Mamaria 🧬🩺

Este repositorio contiene el backend y el motor de inferencia de **SADM** (Sistema de Asistencia Diagnóstica Mamaria), una solución de Inteligencia Artificial orientada a la clasificación automatizada de lesiones en ecografías mamarias. 

La API está desarrollada con **FastAPI** y ejecuta una Red Neuronal Convolucional (CNN) optimizada, devolviendo predicciones de diagnóstico junto con mapas de calor explicativos (**Grad-CAM**) para sustentar la toma de decisiones médicas.

---

## 🚀 Arquitectura y Despliegue

Debido al alto consumo de memoria RAM de las librerías de Deep Learning en arquitecturas CPU (`tensorflow-cpu`), el backend está optimizado para entornos de contenedores Docker y se encuentra desplegado de forma gratuita en:

* **Servidor de Producción:** [Hugging Face Spaces](https://huggingface.co/spaces/Sant-4/sadm-backend) 
* **Infraestructura de Inferencia:** Entorno aislado de 16 GB RAM / 2 vCPU.

---

## 🛠️ Tecnologías Utilizadas

* **Python 3.10**
* **FastAPI:** Framework de alto rendimiento para la construcción de la API REST.
* **TensorFlow CPU & tf_keras (v2.15.0):** Motor clásico de ejecución para redes neuronales y compatibilidad de serialización nativa `.keras`.
* **OpenCV (Headless):** Procesamiento adaptativo de matrices de imágenes ecográficas.
* **Uvicorn:** Servidor ASGI para la gestión de solicitudes concurrentes.
* **Docker:** Orquestación de dependencias y aislamiento del entorno de producción.

---

## 🔧 Parche Adaptativo de Inferencia (Compatibilidad Keras 2/3)

El modelo fue entrenado originalmente bajo una estructura de **Keras 2**. Para evitar errores de deserialización al montarse en contenedores modernos, este backend implementa un algoritmo de desinfección en memoria que abre dinámicamente el archivo comprimido `.keras` en el arranque, examina recursivamente el árbol de configuración `config.json` y realiza las siguientes correcciones en pleno vuelo:
1. Convierte los diccionarios complejos `DTypePolicy` de precisión mixta (`mixed_float16`) a estructuras de strings planos legibles por entornos heredados.
2. Reemplaza el parámetro obsoleto `batch_shape` en las capas de entrada (`InputLayer`) por `input_shape`.
3. Elimina argumentos desconocidos como `data_format` en las capas de augmentación de imágenes (`RandomFlip`, `RandomRotation`, `RandomZoom`).
4. Desactiva la carga del optimizador (`compile=False`) optimizando el consumo de RAM, ya que el contenedor se limita estrictamente a tareas de inferencia y no de re-entrenamiento.

---

## 📂 Estructura del Proyecto

```text
├── app/
│   ├── models/
│   │   └── breast_ultrasound_model.keras  # Modelo CNN descargado dinámicamente
│   ├── main.py                            # Rutas FastAPI y políticas CORS
│   ├── core_model.py                      # Parche de desinfección JSON y lógica de inferencia
│   └── utils.py                           # Preprocesamiento de imágenes y generador Grad-CAM
├── .Dockerfile                             # Configuración del contenedor e inyección legacy
├── requirements.txt                       # Dependencias del sistema con versiones fijadas
└── runtime.txt                            # Declaración del entorno de ejecución de Python
```

# 📡 Endpoints de la API
## 1. Verificar Estado del Servidor
* Ruta: `GET /` o `GET /health`
* Respuesta:

```
{
  "status": "ok",
  "service": "API de Asistencia Diagnóstica Mamaria activa"
}
```
## 2. Clasificación de Ecografía
* Ruta: `POST /api/clasificar`
* Tipo de Entrada: `multipart/form-data` (Clave: `file`, Valor: Archivo de imagen `.png`, `.jpg` o `.jpeg`)
* Respuesta de Inferencia:

```
{
  "clase": "Maligno",
  "probabilidad": 0.8745,
  "gradcam": "iVBORw0KGgoAAAANSUhEUgAA...", // String Base64 de la imagen superpuesta
  "tiempo_proceso": 340 // Latencia interna del servidor en milisegundos
}
```

# 💻 Instalación y Ejecución Local
Si deseas clonar este backend y ejecutarlo de manera local en tu computadora, sigue estos pasos:

## Clonar el repositorio:

```
git clone [https://github.com/Blueisazul/sadm-bk.git](https://github.com/Blueisazul/sadm-bk.git)
cd sadm-bk
```

## Crear y activar un entorno virtual:

```
python -m venv env
source env/Scripts/activate  # En Windows (Git Bash)
```

## Instalar dependencias:

```
pip install --upgrade pip
pip install -r requirements.txt
```
## Descargar el modelo:

Asegúrate de colocar tu archivo `breast_ultrasound_model.keras` dentro de la ruta `app/models/.`

## Lanzar el servidor de desarrollo:

```
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
