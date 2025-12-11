"""
Module for handling word embeddings using llama.cpp API
"""
import requests
import numpy as np
from typing import List, Dict, Optional
import pandas as pd


class EmbeddingGenerator:
    """
    Class to handle communication with llama.cpp embedding API
    """
    
    def __init__(self, server_url: str = "http://localhost:8080"):
        self.server_url = server_url.rstrip('/')
    
    def _extract_embedding(self, data):
        """
        Extract embedding from various possible response formats
        """
        if isinstance(data, dict):
            if "embedding" in data:
                embedding = data["embedding"]
                # Handle nested embedding case (double array [[...]])
                if isinstance(embedding, list) and len(embedding) == 1 and isinstance(embedding[0], list):
                    return embedding[0]  # Extract from the nested array
                return embedding
            elif "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
                # Some llama.cpp versions return embeddings in a "data" array
                embedding = data["data"][0] if isinstance(data["data"][0], list) else data["data"]
                # Handle nested embedding case (double array [[...]])
                if isinstance(embedding, list) and len(embedding) == 1 and isinstance(embedding[0], list):
                    return embedding[0]  # Extract from the nested array
                return embedding
            else:
                # If we can't find the embedding in expected locations, try to find it
                # Look for any list-like value in the response that could be an embedding
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0:
                        # Handle nested embedding case (double array [[...]])
                        if len(value) == 1 and isinstance(value[0], list):
                            # This is likely the nested embedding format
                            return value[0]
                        elif all(isinstance(x, (int, float)) for x in value):
                            # This is a flat embedding vector
                            return value
        elif isinstance(data, list) and len(data) > 0:
            # Response is a list, possibly containing embedding objects
            # Check if it's a list with one item that has an embedding
            first_item = data[0]
            if isinstance(first_item, dict) and "embedding" in first_item:
                embedding = first_item["embedding"]
                # Handle nested embedding case (double array [[...]])
                if isinstance(embedding, list) and len(embedding) == 1 and isinstance(embedding[0], list):
                    return embedding[0]  # Extract from the nested array
                return embedding
            # Otherwise, check if it's a direct list of numbers
            elif all(isinstance(x, (int, float)) for x in data):
                return data
            # Or if it's a list of lists (first being the embedding)
            elif all(isinstance(x, list) for x in data) and len(data) > 0:
                return data[0]
        return None
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text using llama.cpp API
        
        Args:
            text: Input text to generate embedding for
            
        Returns:
            Embedding as a list of floats, or None if failed
        """
        try:
            response = requests.post(
                f"{self.server_url}/embedding",
                json={"content": text},
                timeout=30
            )
            
            if response.status_code == 200:
                response_data = response.json()
                return self._extract_embedding(response_data)
            else:
                print(f"Error: API returned status code {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to embedding API: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error generating embedding: {e}")
            return None
    
    def generate_embeddings_batch(self, texts: List[str], 
                                progress_callback=None) -> List[Dict[str, any]]:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of input texts
            progress_callback: Optional callback function to report progress
            
        Returns:
            List of dictionaries with 'text' and 'embedding' keys
        """
        embeddings = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            embedding = self.generate_embedding(text)
            
            if embedding is not None:
                embeddings.append({
                    "text": text,
                    "embedding": embedding
                })
            else:
                print(f"Failed to generate embedding for text: {text[:50]}...")
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return embeddings


class EmbeddingProcessor:
    """
    Class for processing and analyzing embeddings
    """
    
    @staticmethod
    def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity value between -1 and 1
        """
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    @staticmethod
    def find_most_similar(embedding: List[float], 
                         embeddings_list: List[Dict[str, any]], 
                         top_k: int = 5) -> List[Dict[str, any]]:
        """
        Find the most similar embeddings to a given embedding
        
        Args:
            embedding: Reference embedding to compare against
            embeddings_list: List of embeddings to search through
            top_k: Number of most similar items to return
            
        Returns:
            List of dictionaries with 'text', 'embedding', and 'similarity' keys
        """
        similarities = []
        
        for item in embeddings_list:
            similarity = EmbeddingProcessor.cosine_similarity(embedding, item['embedding'])
            similarities.append({
                'text': item['text'],
                'similarity': similarity
            })
        
        # Sort by similarity in descending order and return top_k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]


def load_embeddings_from_csv(file_path: str, text_column: str) -> List[str]:
    """
    Load text chunks from a CSV file
    
    Args:
        file_path: Path to the CSV file
        text_column: Name of the column containing text chunks
        
    Returns:
        List of text chunks
    """
    df = pd.read_csv(file_path)
    return df[text_column].dropna().tolist()


def save_embeddings_to_json(embeddings_data: List[Dict[str, any]], 
                          file_path: str) -> None:
    """
    Save embeddings data to a JSON file
    
    Args:
        embeddings_data: List of dictionaries with 'text' and 'embedding' keys
        file_path: Path to save the JSON file
    """
    import json
    
    # Ensure all entries have a cluster field before saving
    export_embeddings = []
    for item in embeddings_data:
        item_copy = item.copy()  # Make a copy to avoid modifying original
        # Ensure cluster field exists
        if "cluster" not in item_copy:
            item_copy["cluster"] = -1  # Default to unassigned cluster
        export_embeddings.append(item_copy)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(export_embeddings, f, ensure_ascii=False, indent=2)


def load_embeddings_from_json(file_path: str) -> List[Dict[str, any]]:
    """
    Load embeddings data from a JSON file
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of dictionaries with 'text' and 'embedding' keys
    """
    import json
    
    with open(file_path, 'r', encoding='utf-8') as f:
        loaded_embeddings = json.load(f)
        
        # Ensure all items have a 'cluster' field (default to -1 if not present)
        for item in loaded_embeddings:
            if "cluster" not in item:
                item["cluster"] = -1  # Default to unassigned cluster
        
        return loaded_embeddings