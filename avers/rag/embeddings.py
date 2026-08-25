"""
Embeddings для Vision RAG - CLIP и fallback.

Использует CLIP ViT-B/32 для кодирования ROI в 512d векторы.
Fallback: HOG + цветовые гистограммы.
"""

from typing import Optional, List, Tuple
import numpy as np
import cv2

from avers.core.logger import get_logger

logger = get_logger("avers.rag.embeddings")


class EmbeddingEngine:
    """Базовый интерфейс для эмбеддингов."""
    
    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """Закодировать изображение в вектор."""
        raise NotImplementedError
    
    def encode_text(self, text: str) -> np.ndarray:
        """Закодировать текст в вектор."""
        raise NotImplementedError
    
    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Косинусная близость."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))


class CLIPEmbedding(EmbeddingEngine):
    """CLIP эмбеддинги для vision RAG."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._loaded = False
    
    def load(self) -> bool:
        """Загрузить CLIP модель."""
        if self._loaded:
            return True
        
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
            
            logger.info(f"Loading CLIP: {self.model_name}")
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
            
            self._loaded = True
            logger.info("CLIP loaded")
            return True
        
        except ImportError:
            logger.warning("transformers not installed, using fallback embeddings")
            return self._load_fallback()
        except Exception as e:
            logger.warning(f"CLIP load failed: {e}, using fallback")
            return self._load_fallback()
    
    def _load_fallback(self) -> bool:
        """Fallback - используем HOG."""
        self._loaded = True
        self.model = None
        logger.info("Using HOG fallback embeddings")
        return True
    
    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """Закодировать ROI."""
        if not self._loaded:
            self.load()
        
        if self.model is not None:
            return self._clip_encode_image(image)
        else:
            return self._hog_encode_image(image)
    
    def _clip_encode_image(self, image: np.ndarray) -> np.ndarray:
        """CLIP image encoding."""
        try:
            import torch
            
            # Preprocess
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            inputs = self.processor(images=image, return_tensors="pt")
            
            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
                # Normalize
                features = features / features.norm(dim=-1, keepdim=True)
                return features.cpu().numpy()[0]
        
        except Exception as e:
            logger.warning(f"CLIP encode failed: {e}, fallback to HOG")
            return self._hog_encode_image(image)
    
    def _hog_encode_image(self, image: np.ndarray) -> np.ndarray:
        """HOG fallback encoding - 512d. Works without cv2.HOGDescriptor."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Resize to standard
        gray = cv2.resize(gray, (128, 128))
        
        # Try HOG descriptor, fallback to simple gradients
        try:
            if hasattr(cv2, 'HOGDescriptor'):
                hog = cv2.HOGDescriptor((128,128), (16,16), (8,8), (8,8), 9)
                hog_features = hog.compute(gray).flatten()
            else:
                raise AttributeError("No HOGDescriptor")
        except Exception:
            # Fallback: use gradient histograms manually
            # Sobel gradients
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            angle = np.arctan2(grad_y, grad_x) * 180 / np.pi % 180
            
            # Simple 8x8 cells, 9 orientation bins
            hog_features = []
            cell_size = 16
            for y in range(0, 128, cell_size):
                for x in range(0, 128, cell_size):
                    cell_mag = magnitude[y:y+cell_size, x:x+cell_size]
                    cell_ang = angle[y:y+cell_size, x:x+cell_size]
                    # 9 bins histogram
                    hist, _ = np.histogram(cell_ang, bins=9, range=(0,180), weights=cell_mag)
                    hog_features.extend(hist)
            hog_features = np.array(hog_features, dtype=np.float32)
        
        # Color histogram
        if len(image.shape) == 3:
            hist_b = cv2.calcHist([image], [0], None, [16], [0,256]).flatten()
            hist_g = cv2.calcHist([image], [1], None, [16], [0,256]).flatten()
            hist_r = cv2.calcHist([image], [2], None, [16], [0,256]).flatten()
            color_features = np.concatenate([hist_b, hist_g, hist_r])
            color_features = color_features / (np.linalg.norm(color_features) + 1e-6)
        else:
            color_features = np.zeros(48, dtype=np.float32)
        
        # Combine and pad to 512
        combined = np.concatenate([hog_features[:400], color_features])
        if len(combined) < 512:
            combined = np.pad(combined, (0, 512 - len(combined)))
        else:
            combined = combined[:512]
        
        # Normalize
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        
        return combined.astype(np.float32)
    
    @property
    def backend(self) -> str:
        """Какой движок реально работает: 'clip' или 'hog_fallback'."""
        return "clip" if self.model is not None else "hog_fallback"

    def encode_text(self, text: str) -> np.ndarray:
        """Закодировать текст."""
        if not self._loaded:
            self.load()
        
        if self.model is not None:
            try:
                import torch
                inputs = self.processor(text=[text], return_tensors="pt", padding=True)
                if self.device == "cuda":
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                with torch.no_grad():
                    features = self.model.get_text_features(**inputs)
                    features = features / features.norm(dim=-1, keepdim=True)
                    return features.cpu().numpy()[0]
            except Exception as e:
                logger.warning(f"CLIP text encode failed: {e}")
        
        # Fallback (без CLIP): честный нулевой вектор - текстовой семантики нет.
        # Раньше здесь был случайный вектор от hash(text): hash() в Python солёный,
        # вектор менялся между запусками процесса (ломал сохранённую базу),
        # а как шум он ещё и портил мультимодальный запрос.
        # Текстовый поиск в fallback-режиме делает VisionRAG.query по label/description.
        return np.zeros(512, dtype=np.float32)


class MultiModalEmbedding:
    """Мультимодальные эмбеддинги - объединяет image + text."""
    
    def __init__(self, embedding_engine: Optional[EmbeddingEngine] = None):
        self.engine = embedding_engine or CLIPEmbedding()
    
    def encode(self, image: Optional[np.ndarray] = None, text: Optional[str] = None, weight: float = 0.7) -> np.ndarray:
        """
        Кодировать мультимодальный запрос.
        
        Args:
            image: ROI изображение
            text: текстовый запрос
            weight: вес изображения vs текста (0-1)
        
        Returns:
            512d вектор
        """
        img_emb = None
        text_emb = None
        
        if image is not None:
            img_emb = self.engine.encode_image(image)
        
        if text is not None:
            text_emb = self.engine.encode_text(text)
        
        # Нулевые (недоступные) части исключаем, веса оставшихся нормируем
        parts = []
        if img_emb is not None and np.linalg.norm(img_emb) > 1e-8:
            parts.append((img_emb, weight))
        if text_emb is not None and np.linalg.norm(text_emb) > 1e-8:
            parts.append((text_emb, 1 - weight))
        
        if not parts:
            return np.zeros(512, dtype=np.float32)
        total_w = sum(w for _, w in parts)
        combined = sum(e * (w / total_w) for e, w in parts)
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        return combined
