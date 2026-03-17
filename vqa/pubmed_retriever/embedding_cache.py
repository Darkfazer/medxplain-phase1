import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

class PubMedEmbeddingCache:
    def __init__(self, use_gpu=True):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        
        # Simple local FAISS/Chroma mock: dictionary mapping ID to embedding
        self.cache_file = "pubmed_embeddings.npy"
        self.embedding_db = {}
        if os.path.exists(self.cache_file):
            item = np.load(self.cache_file, allow_pickle=True)
            if item.size > 0:
                self.embedding_db = item.item()
                
    def _load_model(self):
        if self.model is None:
            print("Loading sentence-transformers/all-MiniLM-L6-v2 directly into huggingface pipeline...")
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(self.device)
            self.model.eval()
            
    def get_embedding(self, text):
        self._load_model()
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Take mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
        return embeddings
        
    def add_to_cache(self, pmid, abstract):
        if pmid not in self.embedding_db:
            emb = self.get_embedding(abstract)
            self.embedding_db[pmid] = emb
            
    def save_cache(self):
        np.save(self.cache_file, self.embedding_db)
        
    def search_similar(self, query, abstracts_dicts, top_k=3):
        """Re-rank retrieved abstracts using semantic similarity to query."""
        if not abstracts_dicts:
            return []
            
        query_emb = self.get_embedding(query)
        
        scores = []
        for d in abstracts_dicts:
            pmid = d['pmid']
            abstract = d['abstract']
            if pmid not in self.embedding_db:
                self.add_to_cache(pmid, abstract)
            
            doc_emb = self.embedding_db[pmid]
            # Cosine similarity
            sim = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb) + 1e-9)
            scores.append((sim, d))
            
        self.save_cache()
        
        # Sort by similarity descending
        scores.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scores[:top_k]]
