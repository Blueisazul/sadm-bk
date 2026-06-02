import os
import time
import numpy as np

# 1. Forzar la importación base de TensorFlow
import tensorflow as tf

# === PARCHE INTEGRAL DE DESERIALIZACIÓN PARA TF_KERAS CLÁSICO ===
# Interceptamos la capa de entrada del paquete que está usando el servidor
import tf_keras
from tf_keras.layers import InputLayer

class CompatibleInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        # Si el deserializador encuentra 'batch_shape', lo adaptamos al formato moderno en vivo
        if 'batch_shape' in kwargs:
            kwargs['input_shape'] = kwargs.pop('batch_shape')[1:]
        super().__init__(*args, **kwargs)

# Inyectamos el parche en los objetos personalizados globales de tf_keras
tf_keras.utils.get_custom_objects()['InputLayer'] = CompatibleInputLayer
# ================================================================

from app.utils import procesar_bytes_imagen, generar_gradcam_base64

# --- CONFIGURACIÓN DE RUTAS INTERNAS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "models", "breast_ultrasound_model.keras")

print(f"\n--> [SADM EN ARRANQUE] Buscando el modelo de IA en: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    BASE_DIR = os.path.dirname(CURRENT_DIR)
    ALT_PATH = os.path.join(BASE_DIR, "models", "breast_ultrasound_model.keras")
    if os.path.exists(ALT_PATH):
        MODEL_PATH = ALT_PATH
    else:
        raise FileNotFoundError(f"El archivo del modelo no existe en: {MODEL_PATH}")

print("--> [SADM] Cargando estructura de red neuronal con el parche adaptativo...")

# CARGA CRÍTICA: Forzamos el uso de tf_keras.models en lugar de tf.keras para asegurar el aislamiento
model = tf_keras.models.load_model(MODEL_PATH, custom_objects={'InputLayer': CompatibleInputLayer})

print("✅ [SADM EN ARRANQUE] Modelo Keras cargado con éxito en memoria.\n")


def predecir_lesion(imagen_bytes):
    t_inicio = time.time()
    
    try:
        expected_channels = None
        if model.inputs and len(model.input_shape) >= 4:
            expected_channels = int(model.input_shape[-1])
    except Exception:
        expected_channels = None

    img_tensor, img_original = procesar_bytes_imagen(imagen_bytes, channels=expected_channels)
    
    raw_prediction = model.predict(img_tensor)
    
    prediccion_plana = np.array(raw_prediction).flatten()
    prediccion_base = float(prediccion_plana[0])
    
    if len(prediccion_plana) > 1:
        probabilidad_maligno = float(prediccion_plana[1])
        clase_resultado = "Maligno" if probabilidad_maligno > 0.5 else "Benigno"
        probabilidad_final = probabilidad_maligno if clase_resultado == "Maligno" else float(prediccion_plana[0])
    else:
        clase_resultado = "Maligno" if prediccion_base > 0.5 else "Benigno"
        probabilidad_final = prediccion_base if clase_resultado == "Maligno" else float(1.0 - prediccion_base)
    
    mapa_gradcam = generar_gradcam_base64(model, img_tensor, img_original)
    
    t_fin = time.time()
    latencia_ms = int((t_fin - t_inicio) * 1000)
    
    return {
        "clase": clase_resultado,
        "probabilidad": probabilidad_final,
        "gradcam": mapa_gradcam,
        "tiempo_proceso": latencia_ms
    }