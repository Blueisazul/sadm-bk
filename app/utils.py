import cv2
import numpy as np
import tensorflow as tf
import base64

def procesar_bytes_imagen(imagen_bytes, target_size=(224, 224), channels=None):
    """
    Preprocesamiento adaptativo que soporta entrada RGB o grayscale según el modelo.
    - `channels`: Si es `1` fuerza salida con un canal; si es `3` produce 3 canales; si es None intenta mantener 3.
    Devuelve: (img_tensor, img_original_bgr)
    """
    nparr = np.frombuffer(imagen_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("Formato de imagen inválido.")

    # Mantener una copia BGR para visualizaciones (Grad-CAM)
    img_original = img_bgr.copy()

    # Redimensionar a target_size
    img_resized_bgr = cv2.resize(img_bgr, target_size)

    # Decidir número de canales de entrada
    if channels is None:
        channels = 3

    if int(channels) == 1:
        # Convertir a gris y mantener un canal adicional para la dimensión de canales
        img_gray = cv2.cvtColor(img_resized_bgr, cv2.COLOR_BGR2GRAY)
        img_normalized = img_gray.astype(np.float32) / 255.0
        img_tensor = np.expand_dims(img_normalized, axis=(0, -1))  # (1, H, W, 1)
    else:
        # Convertir a RGB porque Keras suele entrenar con RGB
        img_rgb = cv2.cvtColor(img_resized_bgr, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_tensor = np.expand_dims(img_normalized, axis=0)  # (1, H, W, 3)

    return img_tensor, img_original

def generar_gradcam_base64(model, img_tensor, img_original):
    """
    Generador adaptativo y tolerante a fallos de mapas de calor Grad-CAM
    """
    try:
        # Buscar la última capa convolucional por tipo (más robusto que por nombre)
        capas_conv = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]

        # Si no se detectan capas Conv2D, intentar seleccionar la última capa con salida 4D
        if not capas_conv:
            for l in reversed(model.layers):
                try:
                    if hasattr(l, 'output_shape') and l.output_shape is not None and len(l.output_shape) == 4:
                        capas_conv = [l]
                        break
                except Exception:
                    continue

        if not capas_conv:
            print("Aviso operativo: no se encontró una capa convolucional para Grad-CAM; devolviendo imagen original.")
            _, buffer = cv2.imencode('.png', img_original)
            return base64.b64encode(buffer).decode('utf-8')

        last_conv_layer = capas_conv[-1]

        # Construir modelo para obtener salidas intermedias
        grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])

        # Asegurar tipo float32
        inputs = tf.cast(img_tensor, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(inputs)
            conv_outputs, predictions = grad_model(inputs)
            # Si la salida tiene más de una unidad, tomar la predicción de la clase con mayor probabilidad
            if predictions.shape[-1] > 1:
                class_idx = tf.argmax(predictions[0])
                loss = predictions[:, class_idx]
            else:
                loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            raise RuntimeError("Gradientes nulos al calcular Grad-CAM.")

        # Promediar gradientes sobre las dimensiones espaciales
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()

        conv_outputs = conv_outputs[0].numpy()

        # Multiplicar características por gradientes medios y sumar canales
        heatmap = np.mean(conv_outputs * pooled_grads[np.newaxis, np.newaxis, :], axis=-1)

        # Normalización del mapa de calor
        heatmap = np.maximum(heatmap, 0)
        max_val = np.max(heatmap) if np.max(heatmap) != 0 else 1e-10
        heatmap = heatmap / max_val

        # Redimensionar al tamaño de la imagen original y aplicar colormap
        heatmap = cv2.resize(heatmap, (img_original.shape[1], img_original.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Asegurar que la imagen original esté en BGR uint8
        img_vis = img_original
        if img_vis.dtype != np.uint8:
            img_vis = np.uint8(255 * (img_vis.astype(np.float32) / np.max(img_vis)))

        # Si la imagen de entrada es grayscale (1 canal) expandir a 3 canales para overlay
        if len(img_vis.shape) == 2 or img_vis.shape[-1] == 1:
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)

        # Mezclar heatmap con la imagen (peso ajustable)
        alpha = 0.4
        img_superpuesta = cv2.addWeighted(img_vis, 1 - alpha, heatmap_color, alpha, 0)

        _, buffer = cv2.imencode('.png', img_superpuesta)
        return base64.b64encode(buffer).decode('utf-8')

    except Exception as e:
        detalle = str(e)
        print(f"Aviso operativo: Grad-CAM omitido. Detalle: {detalle}")
        # Fallback: calcular mapa de saliencia simple (gradientes respecto a la entrada)
        try:
            inputs = tf.cast(img_tensor, tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                preds = model(inputs)
                if preds.shape[-1] > 1:
                    class_idx = tf.argmax(preds[0])
                    loss = preds[:, class_idx]
                else:
                    loss = preds[:, 0]

            grads = tape.gradient(loss, inputs)
            if grads is None:
                raise RuntimeError("Gradientes nulos en el fallback de saliencia.")

            # Reducir canales y normalizar
            saliency = tf.reduce_mean(tf.abs(grads), axis=-1)[0].numpy()
            saliency = np.maximum(saliency, 0)
            max_val = np.max(saliency) if np.max(saliency) != 0 else 1e-10
            saliency = saliency / max_val

            heatmap = cv2.resize(saliency, (img_original.shape[1], img_original.shape[0]))
            heatmap_uint8 = np.uint8(255 * heatmap)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            img_vis = img_original
            if img_vis.dtype != np.uint8:
                img_vis = np.uint8(255 * (img_vis.astype(np.float32) / np.max(img_vis)))
            if len(img_vis.shape) == 2 or img_vis.shape[-1] == 1:
                img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)

            alpha = 0.5
            img_superpuesta = cv2.addWeighted(img_vis, 1 - alpha, heatmap_color, alpha, 0)
            _, buffer = cv2.imencode('.png', img_superpuesta)
            return base64.b64encode(buffer).decode('utf-8')

        except Exception as e2:
            print(f"Aviso operativo: fallback de saliencia falló. Detalle: {str(e2)}")
            _, buffer = cv2.imencode('.png', img_original)
            return base64.b64encode(buffer).decode('utf-8')