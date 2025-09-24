from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import httpx
import base64
import io
import json
import uuid
from PIL import Image
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import os
from concurrent.futures import ThreadPoolExecutor
import asyncio

app = FastAPI(title="Computer Vision API", version="1.0.0")
security = HTTPBearer()

# Configuración (usa variables de entorno)
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_PROJECT = os.getenv("ROBOFLOW_PROJECT", "sku-110k")
ROBOFLOW_VERSION = os.getenv("ROBOFLOW_VERSION", "4")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MAX_IMAGE_SIZE = 15 * 1024 * 1024  # 15MB
MAX_CONCURRENT_OCR = 4

# Cliente HTTP reutilizable
http_client = httpx.AsyncClient(timeout=30.0)

# Modelos Pydantic
class BoundingBox(BaseModel):
    xMin: float
    yMin: float
    xMax: float
    yMax: float

class Product(BaseModel):
    bbox: BoundingBox
    confidence: float
    class_name: str
    brand: str
    variety: str
    volume: str
    product_name: str
    ocr_confidence: Optional[float] = None
    runpod_candidate: Optional[str] = None

class AnalysisResponse(BaseModel):
    imageId: str
    width: int
    height: int
    products: List[Product]
    processing: Dict[str, int]

# Autenticación simple (opcional)
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implementar validación de token aquí
    # Por ahora, acepta cualquier token
    return credentials.credentials

# Función para llamar a Roboflow
async def call_roboflow(image_bytes: bytes) -> Dict[str, Any]:
    """Llama a Roboflow para detectar productos"""
    if not ROBOFLOW_API_KEY:
        raise HTTPException(status_code=500, detail="Roboflow configuration error: missing ROBOFLOW_API_KEY env var")

    # Roboflow Serverless expects the API key as a query parameter
    url = f"https://serverless.roboflow.com/{ROBOFLOW_PROJECT}/{ROBOFLOW_VERSION}?api_key={ROBOFLOW_API_KEY}"

    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
    headers = {}

    try:
        response = await http_client.post(url, files=files, headers=headers)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        # Superficial parse for 401 to provide clearer guidance
        status = getattr(e.response, "status_code", None) if hasattr(e, "response") and e.response is not None else None
        if status == 401:
            raise HTTPException(status_code=500, detail="Roboflow error: 401 Unauthorized. Verify ROBOFLOW_API_KEY and project/version access.")
        raise HTTPException(status_code=500, detail=f"Roboflow error: {str(e)}")

# Función para llamar a Gemini OCR
async def call_gemini_ocr(image_bytes: bytes) -> Dict[str, Any]:
    """Llama a Gemini para OCR de un producto recortado"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    # Convertir imagen a base64
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    
    prompt = """
Eres un experto en analizar imágenes de productos en estanterías de supermercado.
Analiza la siguiente imagen de un producto y extrae sus detalles clave.

GUÍA DE VOLÚMENES TÍPICOS:
- Head & Shoulders: 180ml, 250ml, 300ml, 375ml, 650ml, 850ml, 1L
- Savital: 510ml

INSTRUCCIONES:
1. Identifica la marca (Head & Shoulders, Savital, etc.)
2. Extrae la variedad específica (ej: "Control Caída", "Zero Caspa", "Suavidad Increíble")
3. Para el volumen:
   - Lee EXACTAMENTE lo que ves en la imagen
   - Si el número leído es muy similar a un volumen típico (ej: 658→650, 374→375), usa el típico
   - Si no estás seguro o hay ambigüedad, usa "N/A"
4. Normaliza formatos: "1L" = "1l", "375ML" = "375ml"

Responde únicamente con un objeto JSON que contenga las siguientes claves:
- "product_name": El nombre completo del producto.
- "brand": La marca del producto.
- "variety": La variedad específica (ej: "Control Caída", "Restauración").
- "volume": El contenido o volumen (ej: "375ml", "1l").
Si un detalle no es visible o legible, usa el valor "N/A".
"""
    
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_b64
                    }
                }
            ]
        }]
    }
    
    try:
        response = await http_client.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Extraer texto de la respuesta
        text = result["candidates"][0]["content"]["parts"][0]["text"]
        
        # Limpiar y parsear JSON
        cleaned_text = text.strip().replace('```json', '').replace('```', '').strip()
        return json.loads(cleaned_text)
        
    except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
        return {
            "product_name": "N/A",
            "brand": "N/A", 
            "variety": "N/A",
            "volume": "N/A",
            "error": str(e)
        }

# Función para recortar imagen
def crop_image(image_bytes: bytes, bbox: Dict[str, float], padding: int = 10) -> bytes:
    """Recorta una imagen basándose en el bounding box"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        width, height = image.size
        
        # Convertir coordenadas relativas a absolutas si es necesario
        x_min = max(0, int(bbox["x"] - bbox["width"]/2 - padding))
        y_min = max(0, int(bbox["y"] - bbox["height"]/2 - padding))
        x_max = min(width, int(bbox["x"] + bbox["width"]/2 + padding))
        y_max = min(height, int(bbox["y"] + bbox["height"]/2 + padding))
        
        # Recortar imagen
        cropped = image.crop((x_min, y_min, x_max, y_max))
        
        # Convertir a bytes
        output = io.BytesIO()
        cropped.save(output, format="JPEG", quality=85)
        return output.getvalue()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image cropping error: {str(e)}")

# Función para normalizar volumen
def normalize_volume(volume: str) -> str:
    """Normaliza el formato del volumen"""
    if not volume or volume == "N/A":
        return "N/A"
    
    # Convertir a minúsculas y quitar espacios
    normalized = volume.lower().strip()
    
    # Normalizar formatos comunes
    normalized = normalized.replace("ml", "ml").replace("l", "l")
    normalized = normalized.replace(" ", "")
    
    return normalized

# Función para mapear a RunPod candidate
def map_to_runpod_candidate(brand: str, variety: str, volume: str) -> Optional[str]:
    """Mapea producto a candidato RunPod"""
    if brand == "N/A" or variety == "N/A" or volume == "N/A":
        return None
    
    # Mapeo simple basado en reglas
    brand_lower = brand.lower()
    variety_lower = variety.lower()
    volume_lower = volume.lower()
    
    if "head" in brand_lower and "shoulders" in brand_lower:
        if "anticomezon" in variety_lower or "antipicazón" in variety_lower:
            return f"hs_anticomezon_{volume_lower}"
        elif "hidratacion" in variety_lower or "hidratación" in variety_lower:
            return f"hs_hidratacion_{volume_lower}"
        # Agregar más mapeos según necesites
    
    return None

# Endpoint principal
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    image: UploadFile = File(...),
    token: str = Depends(verify_token)
):
    """Analiza una imagen de góndola y devuelve productos detectados"""
    
    start_time = time.time()
    image_id = str(uuid.uuid4())
    
    # Validar archivo
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Leer imagen
    image_bytes = await image.read()
    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail="Image too large")
    
    try:
        # Obtener dimensiones de la imagen
        image_obj = Image.open(io.BytesIO(image_bytes))
        width, height = image_obj.size
        
        # Llamar a Roboflow
        roboflow_start = time.time()
        roboflow_result = await call_roboflow(image_bytes)
        roboflow_time = int((time.time() - roboflow_start) * 1000)
        
        # Procesar detecciones
        products = []
        predictions = roboflow_result.get("predictions", [])
        
        if predictions:
            # Procesar OCR en paralelo (limitado)
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_OCR)
            
            async def process_detection(prediction):
                async with semaphore:
                    # Recortar imagen
                    cropped_bytes = crop_image(image_bytes, prediction)
                    
                    # OCR con Gemini
                    ocr_result = await call_gemini_ocr(cropped_bytes)
                    
                    # Normalizar volumen
                    volume = normalize_volume(ocr_result.get("volume", "N/A"))
                    
                    # Mapear a RunPod candidate
                    runpod_candidate = map_to_runpod_candidate(
                        ocr_result.get("brand", "N/A"),
                        ocr_result.get("variety", "N/A"),
                        volume
                    )
                    
                    return {
                        "bbox": {
                            "xMin": prediction["x"] - prediction["width"]/2,
                            "yMin": prediction["y"] - prediction["height"]/2,
                            "xMax": prediction["x"] + prediction["width"]/2,
                            "yMax": prediction["y"] + prediction["height"]/2
                        },
                        "confidence": prediction["confidence"],
                        "class_name": prediction["class"],
                        "brand": ocr_result.get("brand", "N/A"),
                        "variety": ocr_result.get("variety", "N/A"),
                        "volume": volume,
                        "product_name": ocr_result.get("product_name", "N/A"),
                        "ocr_confidence": 0.8,  # Podrías calcular esto
                        "runpod_candidate": runpod_candidate
                    }
            
            # Procesar todas las detecciones en paralelo
            tasks = [process_detection(pred) for pred in predictions]
            products_data = await asyncio.gather(*tasks)
            
            # Convertir a objetos Product
            products = [Product(**product_data) for product_data in products_data]
        
        # Calcular tiempos
        total_time = int((time.time() - start_time) * 1000)
        gemini_time = total_time - roboflow_time
        
        return AnalysisResponse(
            imageId=image_id,
            width=width,
            height=height,
            products=products,
            processing={
                "roboflow_ms": roboflow_time,
                "gemini_ms": gemini_time,
                "total_ms": total_time
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

# Endpoint de salud
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

# Cerrar cliente HTTP al shutdown
@app.on_event("shutdown")
async def shutdown():
    await http_client.aclose()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
