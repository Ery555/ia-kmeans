import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas import QuantizationResponse, ElbowResponse
from quantizer import ImageQuantizerEngine

app = FastAPI(
    title="K-Means Color Quantization Lab API",
    description="API de Aprendizaje No Supervisado para Cuantización Vectorial de Imágenes",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = ImageQuantizerEngine(max_dimension=800)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

def validate_image_file(file: UploadFile):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Formato no soportado ({file.content_type}). Usa JPEG, PNG o WEBP."
        )

@app.get("/api/health")
def health_check():
    return {"status": "ok", "algorithm": "K-Means Vector Quantization"}

@app.post("/api/quantize", response_model=QuantizationResponse)
async def quantize_image_endpoint(
    file: UploadFile = File(...),
    k: int = Form(8, ge=2, le=64),
    init_method: str = Form("k-means++"),
    use_minibatch: bool = Form(True)
):
    validate_image_file(file)
    if init_method not in {"k-means++", "random"}:
        raise HTTPException(status_code=400, detail="init_method debe ser 'k-means++' o 'random'")

    image_bytes = await file.read()
    try:
        result = engine.quantize(
            image_bytes=image_bytes,
            k=k,
            init_method=init_method,
            use_minibatch=use_minibatch
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando la imagen: {str(e)}")

@app.post("/api/elbow", response_model=ElbowResponse)
async def elbow_curve_endpoint(file: UploadFile = File(...)):
    validate_image_file(file)
    image_bytes = await file.read()
    try:
        return engine.compute_elbow_curve(image_bytes=image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculando curva del codo: {str(e)}")

@app.get("/api/demo-sample", response_model=QuantizationResponse)
def run_synthetic_demo(k: int = 6):
    """
    Genera un gradiente sintético con figuras geométricas de colores puros 
    para probar el backend sin necesidad de subir archivos externos.
    """
    w, h = 400, 300
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Gradiente de fondo + bloques cromáticos definidos
    for y in range(h):
        for x in range(w):
            img[y, x] = [int((x / w) * 255), int((y / h) * 255), 120]
            
    # Círculo rojo y recuadro azul brillante (clusters bien diferenciados)
    img[50:150, 50:150] = [230, 40, 50]
    img[160:260, 220:350] = [30, 140, 250]

    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=95)
    
    return engine.quantize(image_bytes=buf.getvalue(), k=k, init_method="k-means++")