import io
import base64
import time
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans, KMeans
from typing import Tuple, Dict, Any, List

class ImageQuantizerEngine:
    def __init__(self, max_dimension: int = 750):
        self.max_dimension = max_dimension

    @staticmethod
    def _rgb_to_hex(rgb: np.ndarray) -> str:
        return "#{:02x}{:02x}{:02x}".format(
            int(np.clip(rgb[0], 0, 255)),
            int(np.clip(rgb[1], 0, 255)),
            int(np.clip(rgb[2], 0, 255))
        )

    @staticmethod
    def _pil_to_base64(img: Image.Image, quality: int = 85) -> str:
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=quality)
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def _preprocess_image(self, image_bytes: bytes, max_dim: int = None) -> Tuple[Image.Image, np.ndarray]:
        limit = max_dim or self.max_dimension
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.thumbnail((limit, limit), Image.Resampling.LANCZOS)
        img_np = np.array(img, dtype=np.float64)
        return img, img_np

    def _generate_cluster_mask(
        self,
        original_pixels: np.ndarray,
        labels: np.ndarray,
        target_cluster: int,
        centroid_rgb: np.ndarray,
        h: int,
        w: int
    ) -> str:
        """
        Crea una visualización de segmentación: el fondo que no pertenece al cluster 
        se convierte a escala de grises al 25% de intensidad, mientras que los píxeles 
        del cluster seleccionado resaltan con el color de su centroide.
        """
        # Luminancia estándar ITU-R BT.601 atenuada al 25% para dar contraste al cluster activo
        luminance = (
            0.299 * original_pixels[:, 0] +
            0.587 * original_pixels[:, 1] +
            0.114 * original_pixels[:, 2]
        ) * 0.25

        bg_dimmed = np.stack([luminance, luminance, luminance], axis=-1)

        # Máscara booleana de píxeles pertenecientes al cluster k
        mask = (labels == target_cluster)[:, np.newaxis]
        
        # Donde mask es True ponemos el color del centroide, donde es False dejamos el fondo oscuro
        highlighted = np.where(mask, centroid_rgb, bg_dimmed)
        highlighted_img = np.clip(highlighted, 0, 255).astype(np.uint8).reshape(h, w, 3)

        return self._pil_to_base64(Image.fromarray(highlighted_img), quality=80)

    def quantize(
        self,
        image_bytes: bytes,
        k: int = 8,
        init_method: str = "k-means++",
        use_minibatch: bool = True
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()

        # 1. Preprocesar imagen
        pil_original, img_np = self._preprocess_image(image_bytes)
        h, w, c = img_np.shape
        total_pixels = h * w

        # 2. Aplanar tensor 3D (H, W, 3) -> Matriz 2D (N, 3)
        pixels = img_np.reshape(-1, c)
        unique_colors_orig = len(np.unique(pixels.astype(np.uint8), axis=0))

        # 3. Entrenar modelo K-Means
        if use_minibatch:
            model = MiniBatchKMeans(
                n_clusters=k,
                init=init_method,
                random_state=42,
                batch_size=4096,
                n_init=3,
                max_iter=100
            )
        else:
            model = KMeans(
                n_clusters=k,
                init=init_method,
                random_state=42,
                n_init=3,
                max_iter=150
            )

        labels = model.fit_predict(pixels)
        centroids = model.cluster_centers_

        # 4. Reconstruir imagen cuantizada completa
        quantized_pixels = centroids[labels]
        quantized_img_np = np.clip(quantized_pixels, 0, 255).astype(np.uint8).reshape(h, w, c)
        pil_quantized = Image.fromarray(quantized_img_np)

        # 5. Calcular MSE y PSNR
        mse = float(np.mean((img_np - quantized_pixels.reshape(h, w, c)) ** 2))
        psnr = float(10 * np.log10((255.0 ** 2) / mse)) if mse > 0 else 100.0

        # 6. Construir paleta + máscaras individuales de segmentación por cada cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        palette = []
        for cluster_id, count in zip(unique_labels, counts):
            centroid_rgb = centroids[cluster_id]
            mask_b64 = self._generate_cluster_mask(
                original_pixels=pixels,
                labels=labels,
                target_cluster=int(cluster_id),
                centroid_rgb=centroid_rgb,
                h=h,
                w=w
            )
            palette.append({
                "cluster_id": int(cluster_id),
                "hex": self._rgb_to_hex(centroid_rgb),
                "rgb": [int(round(x)) for x in centroid_rgb],
                "percentage": round((int(count) / total_pixels) * 100, 2),
                "pixel_count": int(count),
                "mask_base64": mask_b64
            })
        palette.sort(key=lambda x: x["percentage"], reverse=True)

        # 7. Submuestra de 500 píxeles para graficar distribución en el espacio RGB
        sample_size = min(500, total_pixels)
        rng = np.random.default_rng(seed=42)
        sample_indices = rng.choice(total_pixels, size=sample_size, replace=False)
        scatter_sample = []
        for idx in sample_indices:
            px = pixels[idx]
            cl_id = int(labels[idx])
            scatter_sample.append({
                "r": int(px[0]),
                "g": int(px[1]),
                "b": int(px[2]),
                "cluster": cl_id,
                "hex": self._rgb_to_hex(centroids[cl_id])
            })

        # 8. Compresión teórica en bits
        bits_original = 24
        bits_compressed = max(1, int(np.ceil(np.log2(k))))
        compression_ratio = round((1.0 - (bits_compressed / bits_original)) * 100, 1)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)

        return {
            "original_image_base64": self._pil_to_base64(pil_original),
            "quantized_image_base64": self._pil_to_base64(pil_quantized),
            "palette": palette,
            "metrics": {
                "k": k,
                "init_method": init_method,
                "inertia": round(float(model.inertia_), 2),
                "mse": round(mse, 2),
                "psnr_db": round(psnr, 2),
                "execution_time_ms": elapsed_ms,
                "original_dimensions": [w, h],
                "total_pixels": total_pixels,
                "unique_colors_original": unique_colors_orig,
                "bits_per_pixel_original": bits_original,
                "bits_per_pixel_compressed": bits_compressed,
                "theoretical_compression_ratio_pct": compression_ratio
            },
            "rgb_scatter_sample": scatter_sample
        }

    def compute_elbow_curve(self, image_bytes: bytes, k_values: List[int] = None) -> Dict[str, Any]:
        if k_values is None:
            k_values = [2, 4, 6, 8, 12, 16, 24, 32]

        _, img_np = self._preprocess_image(image_bytes, max_dim=250)
        pixels = img_np.reshape(-1, 3)

        elbow_points = []
        inertias = []

        for k in k_values:
            t0 = time.perf_counter()
            km = MiniBatchKMeans(
                n_clusters=k,
                init="k-means++",
                random_state=42,
                batch_size=2048,
                n_init=2
            )
            labels = km.fit_predict(pixels)
            reconstructed = km.cluster_centers_[labels]
            mse = float(np.mean((pixels - reconstructed) ** 2))
            inertia_scaled = round(float(km.inertia_) / 1e6, 3)

            inertias.append(inertia_scaled)
            elbow_points.append({
                "k": k,
                "inertia": inertia_scaled,
                "mse": round(mse, 2),
                "execution_time_ms": round((time.perf_counter() - t0) * 1000, 1)
            })

        recommended_k = self._find_optimal_k(k_values, inertias)

        return {
            "elbow_curve": elbow_points,
            "recommended_k": recommended_k
        }

    @staticmethod
    def _find_optimal_k(k_values: List[int], inertias: List[float]) -> int:
        x = np.array(k_values, dtype=np.float64)
        y = np.array(inertias, dtype=np.float64)

        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-9)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-9)

        p1 = np.array([x_norm[0], y_norm[0]])
        p2 = np.array([x_norm[-1], y_norm[-1]])
        line_vec = p2 - p1
        line_vec_norm = line_vec / np.linalg.norm(line_vec)

        distances = []
        for i in range(len(x_norm)):
            p = np.array([x_norm[i], y_norm[i]])
            vec_from_p1 = p - p1
            proj = np.dot(vec_from_p1, line_vec_norm) * line_vec_norm
            orth = vec_from_p1 - proj
            distances.append(np.linalg.norm(orth))

        best_idx = int(np.argmax(distances))
        return int(k_values[best_idx])