"""Tests for Vision RAG module."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from avers.rag.embeddings import CLIPEmbedding, MultiModalEmbedding, EmbeddingEngine
from avers.rag.store import FAISSStore, VectorEntry
from avers.rag.vision_rag import VisionRAG, RAGEntry


class TestEmbeddings:
    """Test embedding engines."""
    
    def test_clip_embedding_init(self):
        emb = CLIPEmbedding()
        assert emb.model_name == "openai/clip-vit-base-patch32"
        assert emb._loaded is False
    
    def test_clip_embedding_load_fallback(self):
        emb = CLIPEmbedding()
        result = emb.load()
        assert result is True
        assert emb._loaded is True
    
    def test_encode_image(self):
        emb = CLIPEmbedding()
        emb.load()
        
        # Random image
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        vec = emb.encode_image(img)
        
        assert vec.shape == (512,)
        # Should be normalized
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 0.1 or norm == 0
    
    def test_encode_text(self):
        emb = CLIPEmbedding()
        emb.load()
        
        vec = emb.encode_text("junction dot connected")
        assert vec.shape == (512,)
    
    def test_similarity(self):
        emb = CLIPEmbedding()
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        sim = emb.similarity(v1, v2)
        assert abs(sim - 1.0) < 0.01
        
        v3 = np.array([0.0, 1.0, 0.0])
        sim2 = emb.similarity(v1, v3)
        assert abs(sim2 - 0.0) < 0.01
    
    def test_multimodal(self):
        engine = CLIPEmbedding()
        engine.load()
        mm = MultiModalEmbedding(engine)
        
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        vec = mm.encode(image=img, text="ground symbol", weight=0.7)
        assert vec.shape == (512,)


class TestVectorStore:
    """Test vector store."""
    
    def test_faiss_store_init(self):
        store = FAISSStore(dim=512, use_faiss=False)
        assert store.dim == 512
        assert len(store.entries) == 0
    
    def test_add_and_search(self):
        store = FAISSStore(dim=4, use_faiss=False)
        
        # Add entries
        emb1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        entry1 = VectorEntry(id="test1", embedding=emb1, label="junction_dot", bbox=(0,0,10,10))
        store.add(entry1)
        
        emb2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        entry2 = VectorEntry(id="test2", embedding=emb2, label="ground", bbox=(0,0,10,10))
        store.add(entry2)
        
        assert len(store.entries) == 2
        
        # Search - should find emb1 as closest to itself
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = store.search(query, top_k=2)
        
        assert len(results) == 2
        assert results[0][0].id == "test1"
        assert results[0][1] > 0.9  # High similarity
    
    def test_delete(self):
        store = FAISSStore(dim=4, use_faiss=False)
        emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        entry = VectorEntry(id="del_test", embedding=emb, label="test", bbox=(0,0,10,10))
        store.add(entry)
        assert len(store.entries) == 1
        
        result = store.delete("del_test")
        assert result is True
        assert len(store.entries) == 0
    
    def test_clear(self):
        store = FAISSStore(dim=4, use_faiss=False)
        for i in range(3):
            emb = np.random.randn(4).astype(np.float32)
            entry = VectorEntry(id=f"id_{i}", embedding=emb, label="test", bbox=(0,0,10,10))
            store.add(entry)
        
        assert len(store.entries) == 3
        store.clear()
        assert len(store.entries) == 0
    
    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            store = FAISSStore(dim=4, use_faiss=False)
            
            emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            entry = VectorEntry(id="save_test", embedding=emb, label="junction", bbox=(0,0,10,10), description="test")
            store.add(entry)
            
            store.save(tmpdir)
            assert (tmpdir / "meta.json").exists()
            assert (tmpdir / "embeddings.npy").exists()
            
            # Load into new store
            new_store = FAISSStore(dim=4, use_faiss=False)
            new_store.load(tmpdir)
            assert len(new_store.entries) == 1
            assert "save_test" in new_store.entries


class TestVisionRAG:
    """Test Vision RAG."""
    
    def test_rag_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = VisionRAG(storage_path=Path(tmpdir))
            assert rag.storage_path == Path(tmpdir)
            assert len(rag.entries) == 0
    
    def test_add_example(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = VisionRAG(storage_path=Path(tmpdir))
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            
            entry_id = rag.add_example(img, label="junction_dot", description="connected")
            assert entry_id is not None
            assert len(rag.entries) == 1
            assert entry_id in rag.entries
    
    def test_query(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = VisionRAG(storage_path=Path(tmpdir))
            
            # Add examples
            img1 = np.ones((256, 256, 3), dtype=np.uint8) * 255
            import cv2
            cv2.circle(img1, (128, 128), 10, (0, 0, 0), -1)
            
            rag.add_example(img1, label="junction_dot_connected", description="junction with dot")
            
            img2 = np.ones((256, 256, 3), dtype=np.uint8) * 255
            cv2.line(img2, (0, 128), (256, 128), (0, 0, 0), 2)
            rag.add_example(img2, label="wire_crossing", description="crossing without dot")
            
            # Query
            query_img = np.ones((256, 256, 3), dtype=np.uint8) * 255
            cv2.circle(query_img, (128, 128), 8, (0, 0, 0), -1)
            
            response = rag.query(image=query_img, text="junction dot", top_k=2)
            
            assert len(response.results) >= 1
            assert response.query is not None
            # Should have VLM answer (mock)
            assert response.vlm_answer is not None
    
    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            rag = VisionRAG(storage_path=tmpdir)
            
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            rag.add_example(img, label="test_label", description="test")
            
            rag.save()
            assert (tmpdir / "meta.json").exists()
            
            # Load
            new_rag = VisionRAG(storage_path=tmpdir)
            new_rag.load()
            assert len(new_rag.entries) == 1
    
    def test_stats_and_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = VisionRAG(storage_path=Path(tmpdir))
            
            for i in range(3):
                img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                rag.add_example(img, label=f"label_{i%2}", description=f"desc {i}")
            
            stats = rag.stats()
            assert stats["total"] == 3
            assert stats["rag_entries"] == 3
            
            rag.clear()
            assert len(rag.entries) == 0
            assert len(rag.vector_store.entries) == 0
