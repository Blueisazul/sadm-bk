from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import traceback
from app.core_model import predecir_lesion 

app = FastAPI(title="API de Asistencia Diagnóstica Mamaria")

# Configuración explícita de CORS para producción y desarrollo local
origins = [
    "https://sadm-fr.onrender.com",  # Tu frontend desplegado en Render
    "http://localhost:3000",         # Entornos locales comunes
    "http://127.0.0.1:5500",         # Live Server de VS Code
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,           # Solo permite los orígenes de la lista
    allow_credentials=True,
    allow_methods=["*"],             # Permite POST, GET, OPTIONS, etc.
    allow_headers=["*"], 
)

@app.post("/api/clasificar")
async def clasificar_imagen(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Archivo vacío.")
            
        # Ejecución del modelo Keras
        resultado = predecir_lesion(contents)
        return resultado
        
    except Exception as e:
        # Imprime el rastro completo del error en los logs de Render
        print("\n" + "="*50)
        print("TRACEBACK CRÍTICO EN PRODUCCIÓN (SADM):")
        traceback.print_exc()
        print("="*50 + "\n")
        
        # Devolvemos un error JSON limpio para que el frontend pueda leer el string
        raise HTTPException(
            status_code=500, 
            detail=f"Error en el motor de inferencia: {str(e)}"
        )

# Endpoint de verificación para Render
@app.get("/")
async def root():
    return {"status": "ok", "service": "API de Asistencia Diagnóstica Mamaria activa"}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)