from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import traceback
from app.core_model import predecir_lesion 

app = FastAPI(title="API de Asistencia Diagnóstica Mamaria")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/clasificar")
async def clasificar_imagen(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Archivo vacío.")
            
        resultado = predecir_lesion(contents)
        return resultado
        
    except Exception as e:
        # Imprime el rastro completo del error en la consola Git Bash
        print("\n" + "="*50)
        print("TRACEBACK CRÍTICO DEL BACKEND:")
        traceback.print_exc()
        print("="*50 + "\n")
        
        raise HTTPException(
            status_code=500, 
            detail={"error": str(e), "contexto": "Error interno en el motor de inferencia Keras"}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)