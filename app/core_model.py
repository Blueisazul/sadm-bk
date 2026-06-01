import tensorflow as tf
import time
import numpy as np
from app.utils import procesar_bytes_imagen, generar_gradcam_base64

# Cargar el modelo serializado en formato nativo .keras
MODEL_PATH = "models/breast_ultrasound_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

def predecir_lesion(imagen_bytes):
    # Iniciar cronómetro interno para la métrica de sistemas (Latencia operativa)
    t_inicio = time.time()
    
    # Determinar cuántos canales espera el modelo (ej. 1 para grayscale, 3 para RGB)
    try:
        expected_channels = None
        if model.inputs and len(model.input_shape) >= 4:
            expected_channels = int(model.input_shape[-1])
    except Exception:
        expected_channels = None

    # Preprocesar los bytes entrantes adaptando el número de canales al modelo
    img_tensor, img_original = procesar_bytes_imagen(imagen_bytes, channels=expected_channels)
    
    # Ejecutar la inferencia en la CNN
    raw_prediction = model.predict(img_tensor)
    
    # --- PROCESAMIENTO ADAPTATIVO DE ENTRADAS/SALIDAS ---
    # Aplana la estructura y fuerza la conversión de float16 (Mixed Precision) a float de CPU
    prediccion_plana = np.array(raw_prediction).flatten()
    prediccion_base = float(prediccion_plana[0])
    
    # Si la red tiene múltiples neuronas de salida (ej. Softmax con 2 o más clases)
    if len(prediccion_plana) > 1:
        probabilidad_maligno = float(prediccion_plana[1])
        clase_resultado = "Maligno" if probabilidad_maligno > 0.5 else "Benigno"
        probabilidad_final = probabilidad_maligno if clase_resultado == "Maligno" else float(prediccion_plana[0])
    else:
        # Salida Sigmoidal estándar (1 sola neurona de salida entre 0 y 1)
        clase_resultado = "Maligno" if prediccion_base > 0.5 else "Benigno"
        probabilidad_final = prediccion_base if clase_resultado == "Maligno" else float(1.0 - prediccion_base)
    
    # Generar el componente visual explicativo (Grad-CAM)
    mapa_gradcam = generar_gradcam_base64(model, img_tensor, img_original)
    
    t_fin = time.time()
    latencia_ms = int((t_fin - t_inicio) * 1000)
    
    return {
        "clase": clase_resultado,
        "probabilidad": probabilidad_final,
        "gradcam": mapa_gradcam,
        "tiempo_proceso": latencia_ms
    }