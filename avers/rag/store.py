"""
Vector Store для Vision RAG - хранение и поиск эмбеддингов.

Поддерживает:
  - FAISS (если установлен)
  - In-memory brute force (fallback)
  - Qdrant (опционально)
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import uuid
from datetime import datetime
import numpy as np

from avers.core.logger import get_logger

logger = get_logger("avers.rag.store")


@dataclass
class VectorEntry:
    """Запись в векторной БД."""
    id: str
    embedding: np.ndarray
    label: str
    bbox: Tuple[int, int, int, int]
    description: Optional[str] = None
    image_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "label": self.label,
            "bbox": self.bbox,
            "description": self.description,
            "image_path": self.image_path,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


class VectorStore:
    """Базовый интерфейс векторного хранилища."""
    
    def add(self, entry: VectorEntry) -> str:
        raise NotImplementedError
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[VectorEntry, float]]:
        raise NotImplementedError
    
    def delete(self, entry_id: str) -> bool:
        raise NotImplementedError
    
    def clear(self):
        raise NotImplementedError
    
    def save(self, path: Path):
        raise NotImplementedError
    
    def load(self, path: Path):
        raise NotImplementedError


class FAISSStore(VectorStore):
    """FAISS-based векторное хранилище с fallback на brute force."""
    
    def __init__(self, dim: int = 512, use_faiss: bool = True):
        self.dim = dim
        self.entries: Dict[str, VectorEntry] = {}
        self.embeddings: List[np.ndarray] = []
        self.ids: List[str] = []
        
        self.faiss_index = None
        self.use_faiss = use_faiss
        
        if use_faiss:
            try:
                import faiss
                self.faiss_index = faiss.IndexFlatIP(dim)  # Inner product (cosine if normalized)
                logger.info("FAISS index created")
            except ImportError:
                logger.warning("FAISS not installed, using brute force")
                self.use_faiss = False
    
    def add(self, entry: VectorEntry) -> str:
        """Добавить запись."""
        if entry.id in self.entries:
            # Update
            idx = self.ids.index(entry.id)
            self.embeddings[idx] = entry.embedding
            self.entries[entry.id] = entry
            
            if self.use_faiss and self.faiss_index:
                # Rebuild FAISS (simple approach)
                self._rebuild_faiss()
        else:
            self.entries[entry.id] = entry
            self.embeddings.append(entry.embedding)
            self.ids.append(entry.id)
            
            if self.use_faiss and self.faiss_index:
                self.faiss_index.add(entry.embedding.reshape(1, -1).astype(np.float32))
        
        return entry.id
    
    def _rebuild_faiss(self):
        """Пересобрать FAISS индекс."""
        if not self.use_faiss:
            return
        
        try:
            import faiss
            self.faiss_index = faiss.IndexFlatIP(self.dim)
            if self.embeddings:
                matrix = np.vstack(self.embeddings).astype(np.float32)
                self.faiss_index.add(matrix)
        except Exception as e:
            logger.warning(f"FAISS rebuild failed: {e}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[VectorEntry, float]]:
        """Поиск ближайших."""
        if not self.entries:
            return []
        
        query_embedding = query_embedding.astype(np.float32)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        if self.use_faiss and self.faiss_index and self.faiss_index.ntotal > 0:
            try:
                scores, indices = self.faiss_index.search(query_embedding, min(top_k, self.faiss_index.ntotal))
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < 0:
                        continue
                    entry_id = self.ids[idx]
                    entry = self.entries[entry_id]
                    results.append((entry, float(score)))
                return results
            except Exception as e:
                logger.warning(f"FAISS search failed: {e}, fallback to brute force")
        
        # Brute force
        query_norm = query_embedding[0] / (np.linalg.norm(query_embedding[0]) + 1e-6)
        results = []
        
        for emb, entry_id in zip(self.embeddings, self.ids):
            emb_norm = emb / (np.linalg.norm(emb) + 1e-6)
            score = float(np.dot(query_norm, emb_norm))
            results.append((self.entries[entry_id], score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def delete(self, entry_id: str) -> bool:
        """Удалить запись."""
        if entry_id not in self.entries:
            return False
        
        idx = self.ids.index(entry_id)
        del self.embeddings[idx]
        del self.ids[idx]
        del self.entries[entry_id]
        
        if self.use_faiss:
            self._rebuild_faiss()
        
        return True
    
    def clear(self):
        """Очистить хранилище."""
        self.entries.clear()
        self.embeddings.clear()
        self.ids.clear()
        
        if self.use_faiss and self.faiss_index:
            try:
                import faiss
                self.faiss_index = faiss.IndexFlatIP(self.dim)
            except:
                pass
    
    def save(self, path: Path):
        """Сохранить на диск."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        meta = {
            "entries": [e.to_dict() for e in self.entries.values()],
            "dim": self.dim,
            "count": len(self.entries),
        }
        
        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        
        # Save embeddings
        if self.embeddings:
            matrix = np.vstack(self.embeddings)
            np.save(path / "embeddings.npy", matrix)
            with open(path / "ids.json", "w") as f:
                json.dump(self.ids, f)
        
        logger.info(f"Vector store saved to {path}")
    
    def load(self, path: Path):
        """Загрузить с диска."""
        path = Path(path)
        
        if not (path / "meta.json").exists():
            logger.warning(f"No meta.json in {path}")
            return
        
        with open(path / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        # Load embeddings
        if (path / "embeddings.npy").exists():
            matrix = np.load(path / "embeddings.npy")
            with open(path / "ids.json", "r") as f:
                ids = json.load(f)
            
            # Reconstruct entries (without embeddings in meta, need to re-add)
            self.embeddings = [matrix[i] for i in range(len(ids))]
            self.ids = ids
            
            # Entries need to be reconstructed from meta
            for entry_data in meta["entries"]:
                # Find embedding
                if entry_data["id"] in self.ids:
                    idx = self.ids.index(entry_data["id"])
                    emb = self.embeddings[idx]
                    entry = VectorEntry(
                        id=entry_data["id"],
                        embedding=emb,
                        label=entry_data["label"],
                        bbox=tuple(entry_data["bbox"]),
                        description=entry_data.get("description"),
                        metadata=entry_data.get("metadata", {}),
                    )
                    self.entries[entry.id] = entry
            
            if self.use_faiss:
                self._rebuild_faiss()
        
        logger.info(f"Vector store loaded from {path}: {len(self.entries)} entries")
    
    def stats(self) -> Dict:
        """Статистика."""
        labels = {}
        for entry in self.entries.values():
            labels[entry.label] = labels.get(entry.label, 0) + 1
        
        return {
            "total": len(self.entries),
            "dim": self.dim,
            "labels": labels,
            "use_faiss": self.use_faiss,
        }


# Алиас для обратной совместимости
VectorStore = FAISSStore
