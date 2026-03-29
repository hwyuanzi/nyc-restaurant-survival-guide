import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class SemanticSearchModel:
    """
    Week 4 concept: Transformers for embedding & Nearest neighbor search.
    Encodes text into continuous vectors to create a shared space for queries and restaurants.
    """
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def embed_texts(self, texts):
        """
        Generates dense vector embeddings for a list of strings
        """
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Mean Pooling - Take attention mask into account for correct averaging
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
        
    def search(self, query, dataset_embeddings, top_k=5):
        """
        Performs vector similarity search given a query string and precomputed embeddings.
        Returns the indices of the most similar items and their scores.
        """
        query_embedding = self.embed_texts([query])
        
        # Compute cosine similarities via dot product (since vectors are L2 normalized)
        cosine_scores = torch.mm(query_embedding, dataset_embeddings.transpose(0, 1))[0]
        
        # Get top k
        top_k = min(top_k, dataset_embeddings.size(0))
        top_results = torch.topk(cosine_scores, k=top_k)
        
        return top_results.indices.tolist(), top_results.values.tolist()
