from pydantic import BaseModel, Field
from typing import List

class ColorCluster(BaseModel):
    cluster_id: int
    hex: str
    rgb: List[int]
    percentage: float = Field(..., description="Porcentaje de píxeles asignados a este centroide")
    pixel_count: int
    mask_base64: str = Field(..., description="Imagen con solo este cluster resaltado y el resto en escala de grises oscura")

class QuantizationMetrics(BaseModel):
    k: int
    init_method: str
    inertia: float
    mse: float
    psnr_db: float
    execution_time_ms: float
    original_dimensions: List[int]
    total_pixels: int
    unique_colors_original: int
    bits_per_pixel_original: int = 24
    bits_per_pixel_compressed: int
    theoretical_compression_ratio_pct: float

class PixelPoint3D(BaseModel):
    r: int
    g: int
    b: int
    cluster: int
    hex: str

class QuantizationResponse(BaseModel):
    original_image_base64: str
    quantized_image_base64: str
    palette: List[ColorCluster]
    metrics: QuantizationMetrics
    rgb_scatter_sample: List[PixelPoint3D]

class ElbowPoint(BaseModel):
    k: int
    inertia: float
    mse: float
    execution_time_ms: float

class ElbowResponse(BaseModel):
    elbow_curve: List[ElbowPoint]
    recommended_k: int