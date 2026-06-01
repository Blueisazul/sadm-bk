async function enviarEcografia() {
    const inputVisual = document.getElementById('input-file').files[0];
    if (!inputVisual) return alert("Por favor, cargue una imagen ecográfica.");

    const formData = new FormData();
    formData.append("file", inputVisual);

    // Iniciar contador de tiempo para medir la latencia localmente
    const inicio = performance.now();

    try {
        const base = (window.API_BASE_URL && window.API_BASE_URL.replace(/\/$/, '')) || 'https://sadm-bk.onrender.com';
        const response = await fetch(`${base}/api/clasificar`, {
            method: "POST",
            body: formData
        });
        const data = await response.json();
        const fin = performance.now();

        // Manejar errores del servidor (status >= 400)
        if (!response.ok) {
            console.error("Error del servidor:", data);
            const mensaje = data.detail && (data.detail.error || data.detail) ? (data.detail.error || JSON.stringify(data.detail)) : JSON.stringify(data);
            alert(`Error en el servidor: ${mensaje}`);
            return;
        }

        // Renderizar resultados en la interfaz web (usar las claves devueltas por el backend)
        document.getElementById("txt-resultado").innerText = `Clasificación: ${data.clase}`;
        document.getElementById("txt-confianza").innerText = `Confianza: ${(data.probabilidad * 100).toFixed(2)}%`;

        // Mostrar Grad-CAM en el contenedor y ocultar el placeholder
        const imgGradcam = document.getElementById("img-gradcam");
        const placeholder = document.getElementById("placeholder-gradcam");
        if (data.gradcam) {
            imgGradcam.src = `data:image/png;base64,${data.gradcam}`;
            imgGradcam.classList.remove('hidden');
            placeholder.classList.add('hidden');
        } else {
            imgGradcam.classList.add('hidden');
            placeholder.classList.remove('hidden');
        }

        // Actualizar métricas de latencia en la UI
        const latenciaTotal = (fin - inicio).toFixed(2);
        const latenciaInterna = data.tiempo_proceso !== undefined ? data.tiempo_proceso : 'N/A';
        document.getElementById("txt-latencia-red").innerText = `${latenciaTotal} ms`;
        document.getElementById("txt-latencia-server").innerText = `${latenciaInterna} ms`;

        // Registrar la métrica operativa en consola
        console.log(`Latencia total de red: ${latenciaTotal} ms`);
        console.log(`Latencia interna del servidor: ${latenciaInterna} ms`);

    } catch (error) {
        console.error("Error en la conexión con la API:", error);
        alert('No se pudo conectar con la API. Revisa que el backend esté desplegado en https://sadm-bk.onrender.com y consulta los logs en Render.');
    }
}