"""
RAG retrieval and generation core classes.
"""
from abc import ABC, abstractmethod
from sklearn.neighbors import NearestNeighbors
from random import choices

class ChunkRetriever(ABC):
    def __init__(self, chunks):
        self.chunks = chunks

    @abstractmethod
    def get_n_closest_chunks(self, query, n=3):
        pass

class RandomChunkRetriever(ChunkRetriever):
    def get_n_closest_chunks(self, query, n=3):
        return choices(self.chunks, k=n)

class BiEncoderChunkRetriever(ChunkRetriever):
    def __init__(self, chunks, model_emb):
        super().__init__(chunks)
        self.model_emb = model_emb
        self.embeddings_chunks = model_emb.encode(["passage: " + c for c in chunks])
        self.nn = NearestNeighbors(n_neighbors=5, metric="cosine").fit(self.embeddings_chunks)

    def get_n_closest_chunks(self, query, n=3):
        emb = self.model_emb.encode(["query: " + query])
        indices = list(self.nn.kneighbors(emb, return_distance=False))[0]
        return [self.chunks[i] for i in indices[:n]]

class BiEncoderInstructChunkRetriever(ChunkRetriever):
    def __init__(self, chunks, model_emb, task_prompt):
        super().__init__(chunks)
        self.model_emb = model_emb
        self.task_prompt = task_prompt
        self.embeddings_chunks = model_emb.encode(["passage: " + c for c in chunks])
        self.nn = NearestNeighbors(n_neighbors=5, metric="cosine").fit(self.embeddings_chunks)

    def get_n_closest_chunks(self, query, n=3):
        emb = self.model_emb.encode([f"Instruct:{self.task_prompt}\nQuery:{query}"])
        indices = list(self.nn.kneighbors(emb, return_distance=False))[0]
        return [self.chunks[i] for i in indices[:n]]
