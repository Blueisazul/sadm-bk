import os
import time
import json
import zipfile
import io
import tempfile
import numpy as np
import tensorflow as tf
import tf_keras

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

print("--> [SADM] Iniciando desempaquetado preventivo del formato .keras...")

# === PARCHE QUIRÚRGICO DE CONFIGURACIÓN JSON (Solución Definitiva) ===
try:
    # Creamos un archivo temporal para guardar el modelo sanitizado
    temp_dir = tempfile.gettempdir()
    SANUTIZED_MODEL_PATH = os.path.join(temp_dir, "sanitized_model.keras")
    
    # Abrimos el archivo original como un ZIP
    with zipfile.ZipFile(MODEL_PATH, 'r') as zin:
        with zipfile.ZipFile(SANUTIZED_MODEL_PATH, 'w') as zout:
            for item in zin.infolist():
                buffer = zin.read(item.filename)
                
                # Si encontramos el JSON de configuración de la arquitectura, lo modificamos
                if item.filename == "config.json":
                    config_data = json.loads(buffer.decode('utf-8'))
                    
                    # Buscamos la capa de entrada en la estructura del secuencial
                    if "config" in config_data and "layers" in config_data["config"]:
                        for layer in config_data["config"]["layers"]:
                            if layer.get("class_name") == "InputLayer" and "batch_shape" in layer.get("config", {}):
                                # Extraemos la tupla de dimensiones espaciales removiendo el lote (None)
                                batch_shape = layer["config"]["batch_shape"]
                                # Convertimos [None, 224, 224, 1] -> [224, 224, 1]
                                layer["config"]["input_shape"] = batch_shape[1:]
                                # Removemos la clave que rompe Keras
                                layer["config"].pop("batch_shape", None)
                                print("--> [PARCHE SUCCESS] Argumento 'batch_shape' eliminado de la estructura JSON.")
                    
                    buffer = json.dumps(config_data).encode('utf-8')
                
                # Escribimos el componente en el nuevo archivo temporal
                zout.writestr(item, buffer)
                
    # Reemplazamos la ruta de carga por la del modelo corregido
    LOAD_PATH = SANUTIZED_MODEL_PATH
    print("--> [SADM] Estructura JSON parchada con éxito en espacio temporal.")

except Exception as e:
    print(f"⚠️ [AVISO PARCHE] No se pudo alterar el ZIP interno: {str(e)}. Intentando carga directa.")
    LOAD_PATH = MODEL_PATH

# =====================================================================

print("--> [SADM] Cargando estructura de red neuronal en tf_keras...")
model = tf_keras.models.load_model(LOAD_PATH)
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