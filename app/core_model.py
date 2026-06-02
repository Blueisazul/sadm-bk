import os
import time
import json
import zipfile
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

print("--> [SADM] Iniciando desempaquetado preventivo profundo del formato .keras...")

# === PARCHE RECURSIVO MULTI-CAPA DE CONFIGURACIÓN JSON ===
def sanitizar_estructura_json(data):
    """Rastrea de forma recursiva todo el JSON para corregir incompatibilidades de Keras 2/3"""
    if isinstance(data, dict):
        class_name = data.get("class_name", "")
        config = data.get("config", {})
        
        # 1. Parche para capas de Entrada
        if class_name == "InputLayer" and "batch_shape" in config:
            batch_shape = config["batch_shape"]
            config["input_shape"] = batch_shape[1:]
            config.pop("batch_shape", None)
            print(f"--> [PARCHE CRÍTICO] 'batch_shape' eliminado con éxito de: {config.get('name')}")
        
        # 2. Parche para capas de Preprocesamiento / Augmentación
        if "Random" in class_name and "data_format" in config:
            config.pop("data_format", None)
            print(f"--> [PARCHE CRÍTICO] 'data_format' obsoleto eliminado de la capa: {config.get('name')} ({class_name})")
        
        # 3. Parche adaptativo para DTypePolicy (Conversión de objeto de Keras 3 a string clásico)
        if "dtype" in data and isinstance(data["dtype"], dict):
            dtype_dict = data["dtype"]
            # Extraer el nombre de la política (ej: 'mixed_float16' o 'float32')
            policy_name = dtype_dict.get("config", {}).get("name") or dtype_dict.get("class_name")
            if policy_name:
                data["dtype"] = policy_name
                print(f"--> [PARCHE CRÍTICO] DTypePolicy convertido a string plano: '{policy_name}'")

        # Algunas capas guardan el dtype directamente dentro de su propio bloque 'config'
        if isinstance(config, dict) and "dtype" in config and isinstance(config["dtype"], dict):
            dtype_dict = config["dtype"]
            policy_name = dtype_dict.get("config", {}).get("name") or dtype_dict.get("class_name")
            if policy_name:
                config["dtype"] = policy_name
                print(f"--> [PARCHE CRÍTICO] DTypePolicy interno convertido a string plano: '{policy_name}'")

        # Continuar rastreando recursivamente dentro del diccionario
        for clave, valor in data.items():
            sanitizar_estructura_json(valor)
            
    elif isinstance(data, list):
        # Continuar rastreando si nos topamos con listas de capas u objetos anidados
        for elemento in data:
            sanitizar_estructura_json(elemento)

try:
    temp_dir = tempfile.gettempdir()
    SANUTIZED_MODEL_PATH = os.path.join(temp_dir, "sanitized_model.keras")
    
    with zipfile.ZipFile(MODEL_PATH, 'r') as zin:
        with zipfile.ZipFile(SANUTIZED_MODEL_PATH, 'w') as zout:
            for item in zin.infolist():
                buffer = zin.read(item.filename)
                
                if item.filename == "config.json":
                    config_data = json.loads(buffer.decode('utf-8'))
                    
                    # Ejecutamos la desinfección profunda multi-capa y multi-política
                    sanitizar_estructura_json(config_data)
                    
                    buffer = json.dumps(config_data).encode('utf-8')
                
                zout.writestr(item, buffer)
                
    LOAD_PATH = SANUTIZED_MODEL_PATH
    print("--> [SADM] Estructura JSON completamente sanitizada de forma profunda.")

except Exception as e:
    print(f"⚠️ [AVISO PARCHE] Error en la desinfección profunda: {str(e)}. Intentando carga directa.")
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