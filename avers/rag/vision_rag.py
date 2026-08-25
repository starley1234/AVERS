"""
Vision RAG - Retrieval Augmented Generation для схем.

Архитектура:
  1. Индексация: ROI (256x256) -> CLIP embedding -> Vector DB
  2. Запрос: ROI + текст -> embedding -> Top-K похожих примеров
  3. Генерация: Примеры + ROI + промпт -> VLM -> ответ

Используется в Stage 6 для разрешения коллизий:
  - Вместо чистого VLM: VLM + few-shot примеры из базы
  - База пополняется из валидатора (human-in-the-loop)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import uuid
from datetime import datetime
import numpy as np
import cv2

from avers.rag.embeddings import CLIPEmbedding, MultiModalEmbedding, EmbeddingEngine
from avers.rag.store import FAISSStore, VectorEntry
from avers.core.logger import get_logger

logger = get_logger("avers.rag")


@dataclass
class RAGEntry:
    """Запись для RAG."""
    id: str
    label: str  # например: "junction_dot_connected", "ground", "diode"
    bbox: Tuple[int, int, int, int]
    description: str
    image: Optional[np.ndarray] = None  # ROI image
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class RAGQuery:
    """Запрос к RAG."""
    text: Optional[str] = None
    image: Optional[np.ndarray] = None  # ROI
    bbox: Optional[Tuple[int, int, int, int]] = None
    top_k: int = 5


@dataclass
class RAGResult:
    """Результат RAG."""
    entry: RAGEntry
    score: float
    embedding: Optional[np.ndarray] = None


@dataclass
class RAGResponse:
    """Ответ RAG + VLM."""
    query: RAGQuery
    results: List[RAGResult]
    vlm_answer: Optional[Dict[str, Any]] = None
    prompt_used: Optional[str] = None


class VisionRAG:
    """
    Vision RAG система для АВЕРС.
    
    Использование:
      rag = VisionRAG()
      rag.add_example(roi_image, label="junction_dot", description="Connected junction")
      
      # Запрос
      response = rag.query(image=roi, text="Is there a junction dot?", top_k=5)
      # response.results - похожие примеры
      # response.vlm_answer - ответ VLM с учетом примеров
    """
    
    def __init__(
        self,
        embedding_engine: Optional[EmbeddingEngine] = None,
        vector_store: Optional[FAISSStore] = None,
        vlm_model: Optional[Any] = None,
        storage_path: Path = Path("/tmp/avers_rag"),
    ):
        self.embedding_engine = embedding_engine or CLIPEmbedding()
        self.multi_modal = MultiModalEmbedding(self.embedding_engine)
        self.vector_store = vector_store or FAISSStore(dim=512)
        self.vlm_model = vlm_model  # Optional VLM wrapper
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache of RAGEntry
        self.entries: Dict[str, RAGEntry] = {}
    
    def add_example(
        self,
        image: np.ndarray,
        label: str,
        bbox: Tuple[int, int, int, int] = (0, 0, 256, 256),
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Добавить пример в базу.
        
        Args:
            image: ROI изображение (256x256 или любое)
            label: метка класса
            bbox: bbox в оригинальном изображении
            description: описание
            metadata: доп. метаданные
        
        Returns:
            ID записи
        """
        entry_id = uuid.uuid4().hex[:12]
        
        # Resize to standard
        if image.shape[0] != 256 or image.shape[1] != 256:
            image_resized = cv2.resize(image, (256, 256))
        else:
            image_resized = image
        
        # Embedding
        embedding = self.multi_modal.encode(image=image_resized, text=description or label)
        
        # Save image
        img_path = self.storage_path / f"{entry_id}.jpg"
        cv2.imwrite(str(img_path), image_resized)
        
        # Vector store entry
        vec_entry = VectorEntry(
            id=entry_id,
            embedding=embedding,
            label=label,
            bbox=bbox,
            description=description,
            image_path=str(img_path),
            metadata=metadata or {},
        )
        
        self.vector_store.add(vec_entry)
        
        # RAG entry
        rag_entry = RAGEntry(
            id=entry_id,
            label=label,
            bbox=bbox,
            description=description or "",
            image=image_resized,
            metadata=metadata or {},
        )
        self.entries[entry_id] = rag_entry
        
        # Автосохранение: иначе всё проиндексированное теряется при перезапуске
        try:
            self.save()
        except Exception as e:
            logger.warning(f"RAG autosave failed: {e}")
        
        logger.info(f"Added RAG example: {entry_id} - {label}")
        return entry_id
    
    def query(
        self,
        image: Optional[np.ndarray] = None,
        text: Optional[str] = None,
        top_k: int = 5,
        use_vlm: bool = True,
        vlm_prompt: Optional[str] = None,
    ) -> RAGResponse:
        """
        Запрос к RAG.
        
        Args:
            image: ROI изображение
            text: текстовый запрос
            top_k: количество примеров
            use_vlm: использовать VLM для генерации ответа
            vlm_prompt: кастомный промпт для VLM
        
        Returns:
            RAGResponse с результатами и ответом VLM
        """
        query = RAGQuery(text=text, image=image, top_k=top_k)
        
        # Encode query
        query_embedding = self.multi_modal.encode(image=image, text=text)
        
        # Текстовый запрос без картинки (или в fallback-режиме без CLIP):
        # векторное поиск бесполезен - ищем по label/description подстрокой
        engine_backend = getattr(self.embedding_engine, "backend", "clip")
        text_only = image is None or (np.linalg.norm(query_embedding) < 1e-8) or engine_backend != "clip"
        if text_only and text:
            norm_text = text.lower().strip()
            scored = []
            for entry in self.entries.values():
                hay_label = entry.label.lower()
                hay_desc = (entry.description or "").lower()
                if norm_text == hay_label:
                    score = 0.95
                elif norm_text in hay_label:
                    score = 0.8
                elif norm_text in hay_desc:
                    score = 0.6
                else:
                    # совпадение по отдельным словам
                    words = [w for w in norm_text.split() if len(w) > 2]
                    hits = sum(1 for wd in words if wd in hay_label or wd in hay_desc)
                    score = min(0.5, 0.15 * hits) if hits else 0.0
                if score > 0:
                    scored.append((RAGResult(entry=entry, score=score), score))
            scored.sort(key=lambda x: x[1], reverse=True)
            rag_results = [r for r, _ in scored[:top_k]]
            
            vlm_answer = None
            if use_vlm and rag_results:
                _, vlm_answer = self._generate_vlm_answer(query, rag_results, custom_prompt=vlm_prompt)
            return RAGResponse(query=query, results=rag_results, vlm_answer=vlm_answer, prompt_used=None)
        
        # Search
        search_results = self.vector_store.search(query_embedding, top_k=top_k)
        
        rag_results = []
        for vec_entry, score in search_results:
            # Get RAG entry
            rag_entry = self.entries.get(vec_entry.id)
            if rag_entry is None:
                # Reconstruct from vector entry
                rag_entry = RAGEntry(
                    id=vec_entry.id,
                    label=vec_entry.label,
                    bbox=vec_entry.bbox,
                    description=vec_entry.description or "",
                    metadata=vec_entry.metadata,
                )
            
            rag_results.append(RAGResult(
                entry=rag_entry,
                score=score,
                embedding=vec_entry.embedding,
            ))
        
        # VLM answer with few-shot examples
        vlm_answer = None
        prompt_used = None
        
        if use_vlm and self.vlm_model and rag_results:
            prompt_used, vlm_answer = self._generate_vlm_answer(
                query=query,
                examples=rag_results,
                custom_prompt=vlm_prompt,
            )
        elif use_vlm and rag_results:
            # Mock VLM answer based on examples
            vlm_answer = self._mock_vlm_answer(query, rag_results)
        
        return RAGResponse(
            query=query,
            results=rag_results,
            vlm_answer=vlm_answer,
            prompt_used=prompt_used,
        )
    
    def _generate_vlm_answer(
        self,
        query: RAGQuery,
        examples: List[RAGResult],
        custom_prompt: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Генерация ответа VLM с few-shot примерами."""
        
        # Build few-shot prompt
        prompt = custom_prompt or self._build_few_shot_prompt(query, examples)
        
        try:
            if self.vlm_model and hasattr(self.vlm_model, 'query'):
                # Use VLM wrapper
                response = self.vlm_model.query(
                    image=query.image if query.image is not None else np.zeros((256,256,3), dtype=np.uint8),
                    prompt=prompt
                )
                return prompt, response
            else:
                return prompt, self._mock_vlm_answer(query, examples)
        except Exception as e:
            logger.warning(f"VLM query failed: {e}")
            return prompt, self._mock_vlm_answer(query, examples)
    
    def _build_few_shot_prompt(self, query: RAGQuery, examples: List[RAGResult]) -> str:
        """Построить few-shot промпт для VLM."""
        prompt = """You are a CAD schematic QA inspector. Analyze the schematic junction.

Examples of similar junctions:
"""
        for i, result in enumerate(examples[:3]):
            prompt += f"\nExample {i+1}: {result.entry.label} - {result.entry.description} (similarity: {result.score:.2f})"
        
        prompt += f"""

Now analyze this junction:
Question: {query.text or 'Is there an electrical connection?'}

Look for:
- Junction dot (filled circle) at intersection = connected
- No dot = not connected (just crossing)
- Consider the examples above

Respond in JSON: {{"connected": true/false, "confidence": 0.0-1.0, "reasoning": "brief"}}
"""
        return prompt
    
    def _mock_vlm_answer(self, query: RAGQuery, examples: List[RAGResult]) -> Dict[str, Any]:
        """Mock VLM answer на основе примеров."""
        if not examples:
            return {"connected": False, "confidence": 0.5, "reasoning": "No examples found"}
        
        # Majority vote from examples
        connected_votes = sum(1 for r in examples if "connected" in r.entry.label.lower() or "junction" in r.entry.label.lower())
        total = len(examples)
        
        # Check query text
        query_text = (query.text or "").lower()
        if "junction" in query_text or "dot" in query_text:
            # Likely asking about junction
            avg_score = sum(r.score for r in examples) / total if total > 0 else 0.5
            return {
                "connected": connected_votes > total/2,
                "confidence": min(0.95, avg_score + 0.2),
                "reasoning": f"Based on {total} similar examples, {connected_votes} indicate connection",
                "examples_used": total,
                "method": "mock_rag_voting"
            }
        
        return {
            "connected": False,
            "confidence": 0.6,
            "reasoning": f"Mock answer based on {total} examples",
            "examples_used": total,
        }
    
    def save(self, path: Optional[Path] = None):
        """Сохранить базу."""
        save_path = path or self.storage_path
        self.vector_store.save(save_path)
        
        # Save entries meta
        meta = {
            "entries": [e.to_dict() if hasattr(e, 'to_dict') else {
                "id": e.id,
                "label": e.label,
                "bbox": e.bbox,
                "description": e.description,
            } for e in self.entries.values()]
        }
        
        with open(save_path / "rag_meta.json", "w", encoding="utf-8") as f:
            import json
            json.dump(meta, f, indent=2, ensure_ascii=False)
        
        logger.info(f"RAG saved to {save_path}")
    
    def load(self, path: Optional[Path] = None):
        """Загрузить базу."""
        load_path = path or self.storage_path
        self.vector_store.load(load_path)
        
        # Load entries
        meta_path = load_path / "rag_meta.json"
        if meta_path.exists():
            import json
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                for entry_data in meta.get("entries", []):
                    entry = RAGEntry(
                        id=entry_data["id"],
                        label=entry_data["label"],
                        bbox=tuple(entry_data["bbox"]),
                        description=entry_data.get("description", ""),
                    )
                    self.entries[entry.id] = entry
        
        logger.info(f"RAG loaded from {load_path}: {len(self.entries)} entries")
    
    def stats(self) -> Dict[str, Any]:
        """Статистика."""
        store_stats = self.vector_store.stats()
        return {
            **store_stats,
            "rag_entries": len(self.entries),
            "storage_path": str(self.storage_path),
            "embedding_backend": getattr(self.embedding_engine, "backend", "unknown"),
        }
    
    def clear(self):
        """Очистить базу."""
        self.vector_store.clear()
        self.entries.clear()
        try:
            self.save()
        except Exception as e:
            logger.warning(f"RAG save after clear failed: {e}")
        logger.info("RAG cleared")


# Глобальный инстанс
_global_rag: Optional[VisionRAG] = None

def get_rag(storage_path: Path = Path("/tmp/avers_rag")) -> VisionRAG:
    """Получить глобальный RAG инстанс."""
    global _global_rag
    if _global_rag is None:
        _global_rag = VisionRAG(storage_path=storage_path)
        # Загружаем ранее сохранённую базу, если есть
        try:
            _global_rag.load()
        except Exception as e:
            logger.warning(f"RAG load skipped: {e}")
    return _global_rag
