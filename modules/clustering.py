"""
Module for clustering embeddings using HDBSCAN and related algorithms
"""
import numpy as np
import hdbscan
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from typing import List, Dict, Tuple, Optional
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer


class EmbeddingClusterer:
    """
    Class for clustering embeddings using various algorithms
    """
    
    def __init__(self):
        self.clustering_algorithm = None
        self.labels_ = None
        self.n_clusters_ = 0
        
    def fit_hdbscan(self, embeddings: np.ndarray, 
                   min_cluster_size: int = 5,
                   min_samples: Optional[int] = None,
                   cluster_selection_epsilon: float = 0.0,
                   metric: str = 'euclidean',
                   cluster_selection_method: str = 'eom') -> np.ndarray:
        """
        Fit HDBSCAN clustering algorithm to embeddings
        
        Args:
            embeddings: Array of embeddings (n_samples, n_features)
            min_cluster_size: Minimum cluster size
            min_samples: Minimum samples for core points (if None, uses min_cluster_size)
            cluster_selection_epsilon: Noise threshold for clustering
            metric: Distance metric to use
            cluster_selection_method: 'eom' or 'leaf'
            
        Returns:
            Cluster labels for each embedding
        """
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric=metric,
            cluster_selection_method=cluster_selection_method
        )
        
        cluster_labels = clusterer.fit_predict(embeddings)
        
        self.clustering_algorithm = clusterer
        self.labels_ = cluster_labels
        self.n_clusters_ = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
        return cluster_labels
    
    def fit_kmeans(self, embeddings: np.ndarray, n_clusters: int = 8) -> np.ndarray:
        """
        Fit K-Means clustering algorithm to embeddings
        
        Args:
            embeddings: Array of embeddings (n_samples, n_features)
            n_clusters: Number of clusters to create
            
        Returns:
            Cluster labels for each embedding
        """
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = clusterer.fit_predict(embeddings)
        
        self.clustering_algorithm = clusterer
        self.labels_ = cluster_labels
        self.n_clusters_ = n_clusters
        
        return cluster_labels
    
    def fit_dbscan(self, embeddings: np.ndarray, 
                  eps: float = 0.5,
                  min_samples: int = 5) -> np.ndarray:
        """
        Fit DBSCAN clustering algorithm to embeddings
        
        Args:
            embeddings: Array of embeddings (n_samples, n_features)
            eps: Maximum distance between samples for clustering
            min_samples: Minimum number of samples for a core point
            
        Returns:
            Cluster labels for each embedding
        """
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = clusterer.fit_predict(embeddings)
        
        self.clustering_algorithm = clusterer
        self.labels_ = cluster_labels
        self.n_clusters_ = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
        return cluster_labels
    
    def get_cluster_statistics(self) -> Dict:
        """
        Get statistics about the clustering results
        
        Returns:
            Dictionary with cluster statistics
        """
        if self.labels_ is None:
            return {}
        
        unique_labels, counts = np.unique(self.labels_, return_counts=True)
        
        stats = {
            'n_clusters': self.n_clusters_,
            'n_noise_points': np.sum(self.labels_ == -1) if -1 in self.labels_ else 0,
            'cluster_sizes': dict(zip(unique_labels, counts)),
            'largest_cluster_size': max(counts) if len(counts) > 0 else 0,
            'smallest_cluster_size': min(counts) if len(counts) > 0 else 0,
        }
        
        return stats


def evaluate_clustering(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Evaluate clustering quality using various metrics
    
    Args:
        embeddings: Array of embeddings (n_samples, n_features)
        labels: Cluster labels for each embedding
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Only calculate silhouette score if we have more than one cluster and not all points are noise
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    metrics = {
        'n_clusters': n_clusters,
        'n_noise_points': np.sum(labels == -1) if -1 in labels else 0
    }
    
    if n_clusters >= 2 and n_clusters < len(embeddings):
        try:
            silhouette_avg = silhouette_score(embeddings, labels)
            metrics['silhouette_score'] = silhouette_avg
        except ValueError:
            # This can happen if all points are in one cluster or all are noise
            metrics['silhouette_score'] = float('nan')
    else:
        metrics['silhouette_score'] = float('nan')
    
    return metrics


class ClusterAnalyzer:
    """
    Class for analyzing cluster results and extracting insights
    """
    
    @staticmethod
    def find_cluster_representatives(embeddings: np.ndarray, 
                                   labels: np.ndarray, 
                                   texts: List[str],
                                   n_representatives: int = 3) -> Dict[int, List[str]]:
        """
        Find representative texts for each cluster
        
        Args:
            embeddings: Array of embeddings (n_samples, n_features)
            labels: Cluster labels for each embedding
            texts: List of text chunks corresponding to embeddings
            n_representatives: Number of representative texts per cluster
            
        Returns:
            Dictionary mapping cluster IDs to representative texts
        """
        representatives = {}
        
        for cluster_id in set(labels):
            if cluster_id == -1:  # Skip noise points
                continue
                
            # Get embeddings and texts for this cluster
            cluster_mask = labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            cluster_texts = np.array(texts)[cluster_mask]
            
            # Calculate centroid of the cluster
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Calculate distances from centroid
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            
            # Get indices of n closest points to centroid
            closest_indices = np.argsort(distances)[:n_representatives]
            
            representatives[cluster_id] = cluster_texts[closest_indices].tolist()
        
        return representatives
    
    @staticmethod
    def get_cluster_descriptions(embeddings: np.ndarray, 
                               labels: np.ndarray, 
                               texts: List[str],
                               n_keywords: int = 5) -> Dict[int, List[str]]:
        """
        Generate simple descriptions for each cluster based on common keywords
        
        Args:
            embeddings: Array of embeddings (n_samples, n_features)
            labels: Cluster labels for each embedding
            texts: List of text chunks corresponding to embeddings
            n_keywords: Number of keywords per cluster
            
        Returns:
            Dictionary mapping cluster IDs to keyword lists
        """
        descriptions = {}
        
        for cluster_id in set(labels):
            if cluster_id == -1:  # Skip noise points
                continue
                
            # Get texts for this cluster
            cluster_mask = labels == cluster_id
            cluster_texts = np.array(texts)[cluster_mask]
            
            # Simple approach: find common words (this is a simplified implementation)
            # In a real implementation, you might want to use more sophisticated NLP
            all_words = []
            for text in cluster_texts:
                # Simple tokenization: split by whitespace and remove punctuation
                words = text.lower().replace(',', ' ').replace('.', ' ').replace('!', ' ').replace('?', ' ').split()
                # Remove short words
                words = [w for w in words if len(w) > 2]
                all_words.extend(words)
            
            # Count word frequencies
            from collections import Counter
            word_counts = Counter(all_words)
            most_common = [word for word, count in word_counts.most_common(n_keywords)]
            
            descriptions[cluster_id] = most_common[:n_keywords]
        
        return descriptions


def optimize_hdbscan_params(embeddings: np.ndarray,
                          min_cluster_size_range: Tuple[int, int] = (5, 20),
                          sample_size: int = 1000) -> Dict[str, any]:
    """
    Find optimal HDBSCAN parameters using a subset of data
    
    Args:
        embeddings: Array of embeddings (n_samples, n_features)
        min_cluster_size_range: Range for min_cluster_size parameter
        sample_size: Size of sample to use for optimization (for performance)
        
    Returns:
        Dictionary with optimal parameters and evaluation metrics
    """
    if len(embeddings) <= sample_size:
        sample_embeddings = embeddings
    else:
        # Randomly sample embeddings
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample_embeddings = embeddings[indices]
    
    best_score = float('-inf')
    best_params = {}
    best_labels = None
    
    # Try different min_cluster_size values
    for min_size in range(min_cluster_size_range[0], min_cluster_size_range[1] + 1):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size)
        labels = clusterer.fit_predict(sample_embeddings)
        
        # Evaluate clustering (skip if only one cluster type exists)
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        if n_clusters >= 2:  # Need at least 2 clusters to evaluate
            try:
                score = silhouette_score(sample_embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_params = {'min_cluster_size': min_size}
                    best_labels = labels
            except ValueError:
                # This can happen in edge cases
                continue
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'n_clusters_with_best_params': len(set(best_labels)) - (1 if -1 in best_labels else 0) if best_labels is not None else 0
    }


class BERTopicKeywordExtractor:
    """
    Class for extracting keywords from text clusters using BERTopic's c-TF-IDF approach
    """

    def __init__(self,
                 language: str = "english",
                 top_n_words: int = 10,
                 ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 1,
                 max_df: float = 1.0,
                 stop_words: str = "english"):
        """
        Initialize BERTopic keyword extractor

        Args:
            language: Language for the embeddings model
            top_n_words: Number of keywords to extract per cluster
            ngram_range: Range of n-grams to consider (min, max)
            min_df: Minimum document frequency for a term
            max_df: Maximum document frequency for a term (proportion)
            stop_words: Stop words to use ('english' or custom list)
        """
        self.language = language
        self.top_n_words = top_n_words
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.stop_words = stop_words
        self.topic_model = None

    def extract_keywords(self,
                        texts: List[str],
                        cluster_labels: np.ndarray,
                        embeddings: Optional[np.ndarray] = None,
                        text_ids: Optional[List] = None) -> Dict[int, Dict[str, any]]:
        """
        Extract keywords from text clusters using c-TF-IDF

        Args:
            texts: List of text documents
            cluster_labels: Cluster labels for each text
            embeddings: Pre-computed embeddings (optional)
            text_ids: Original text IDs from the input data (optional)

        Returns:
            Dictionary with cluster information and keywords
        """
        # Group texts by cluster (don't combine them!)
        cluster_to_docs = {}
        cluster_to_indices = {}

        for i, (text, label) in enumerate(zip(texts, cluster_labels)):
            if label != -1:  # Skip noise points
                if label not in cluster_to_docs:
                    cluster_to_docs[label] = []
                    cluster_to_indices[label] = []
                cluster_to_docs[label].append(text)
                cluster_to_indices[label].append(i)

        if not cluster_to_docs:
            return {}

        # Initialize vectorizer properly for c-TF-IDF
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words=self.stop_words,
            lowercase=True
        )

        # Create proper document structure for c-TF-IDF
        # Each text remains as a separate document
        all_docs = texts
        valid_indices = [i for i, label in enumerate(cluster_labels) if label != -1]
        valid_docs = [texts[i] for i in valid_indices]
        valid_labels = [cluster_labels[i] for i in valid_indices]

        if len(valid_docs) == 0:
            return {}

        # Fit vectorizer on all documents
        try:
            doc_term_matrix = vectorizer.fit_transform(valid_docs)
            feature_names = vectorizer.get_feature_names_out()
        except ValueError as e:
            # Handle case where there are no valid features
            print(f"Vectorization failed: {e}")
            return {}

        # Calculate c-TF-IDF manually
        c_tf_idf_matrix, tf_scores = self._c_tf_idf(doc_term_matrix, valid_labels)

        # Extract keywords for each cluster
        results = {}

        for cluster_id in cluster_to_docs.keys():
            # Get indices of documents belonging to this cluster
            cluster_doc_indices = [i for i, label in enumerate(valid_labels) if label == cluster_id]

            if not cluster_doc_indices:
                continue

            # Calculate average c-TF-IDF scores for this cluster
            cluster_scores = np.mean(c_tf_idf_matrix[cluster_doc_indices], axis=0).A1

            # Get top N words with highest scores
            top_indices = cluster_scores.argsort()[-self.top_n_words:][::-1]

            # Format keywords with their scores
            keywords = []
            for idx in top_indices:
                if cluster_scores[idx] > 0:  # Only include words with non-zero scores
                    keywords.append({
                        "word": feature_names[idx],
                        "score": float(cluster_scores[idx])
                    })

            # Get all texts in this cluster with their original indices and text IDs
            cluster_texts_info = []
            for doc_idx in cluster_doc_indices:
                original_idx = valid_indices[doc_idx]  # Map back to original indices
                # Use original text_id if provided, otherwise use array index
                original_text_id = text_ids[original_idx] if text_ids else original_idx
                cluster_texts_info.append({
                    "text_id": original_text_id,
                    "text": texts[original_idx]
                })

            results[cluster_id] = {
                "cluster_id": int(cluster_id),
                "keywords": keywords,
                "n_texts": len(cluster_texts_info),
                "texts": cluster_texts_info
            }

        return results

    def _c_tf_idf(self, doc_term_matrix, cluster_labels):
        """
        Calculate class-based TF-IDF (c-TF-IDF) matrix

        Args:
            doc_term_matrix: Document-term matrix from CountVectorizer
            cluster_labels: Cluster labels for each document

        Returns:
            c-TF-IDF matrix and TF scores
        """
        from scipy.sparse import csr_matrix
        import numpy as np

        # Convert to dense if needed for calculations
        if not isinstance(doc_term_matrix, np.ndarray):
            dt_matrix = doc_term_matrix.toarray()
        else:
            dt_matrix = doc_term_matrix

        # Calculate term frequency per document
        tf = np.sum(dt_matrix, axis=1, keepdims=True)
        tf = np.where(tf == 0, 1, tf)  # Avoid division by zero

        # Normalize by document length to get TF
        tf_normalized = dt_matrix / tf

        # Calculate document frequency for each term
        df = np.sum(dt_matrix > 0, axis=0)

        # Calculate IDF
        n_docs = len(cluster_labels)
        idf = np.log(n_docs / df)

        # Calculate c-TF-IDF: TF * IDF
        c_tf_idf = tf_normalized * idf

        return csr_matrix(c_tf_idf), tf_normalized

    def get_topic_info(self) -> pd.DataFrame:
        """
        Get detailed topic information (placeholder for compatibility)
        """
        return pd.DataFrame()

    def update_params(self,
                     top_n_words: Optional[int] = None,
                     ngram_range: Optional[Tuple[int, int]] = None,
                     min_df: Optional[int] = None,
                     max_df: Optional[float] = None):
        """
        Update c-TF-IDF parameters
        """
        if top_n_words is not None:
            self.top_n_words = top_n_words
        if ngram_range is not None:
            self.ngram_range = ngram_range
        if min_df is not None:
            self.min_df = min_df
        if max_df is not None:
            self.max_df = max_df


class DocumentTopicRelevanceAnalyzer:
    """
    Class for analyzing document-topic relevance within clusters
    """

    def __init__(self):
        self.relevance_scores = {}

    def calculate_document_topic_relevance(self,
                                         texts: List[str],
                                         cluster_labels: np.ndarray,
                                         topic_keywords: Dict[int, List[str]],
                                         method: str = "tfidf") -> Dict[int, Dict[str, any]]:
        """
        Calculate relevance of each topic/keyword to each document within its cluster

        Args:
            texts: List of text documents
            cluster_labels: Cluster labels for each text
            topic_keywords: Dictionary mapping cluster_id to list of keywords
            method: Method for calculating relevance ("tfidf", "frequency", "jaccard")

        Returns:
            Dictionary with document-topic relevance scores
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from collections import Counter
        import re

        results = {}

        for cluster_id, keywords in topic_keywords.items():
            if cluster_id == -1:  # Skip noise points
                continue

            # Get documents in this cluster
            cluster_mask = np.array(cluster_labels) == cluster_id
            cluster_texts = [texts[i] for i in range(len(texts)) if cluster_mask[i]]
            cluster_indices = [i for i in range(len(texts)) if cluster_mask[i]]

            if not cluster_texts or not keywords:
                continue

            # Preprocess keywords
            processed_keywords = [kw.lower().strip() for kw in keywords]

            if method == "tfidf":
                # Use TF-IDF to calculate relevance
                vectorizer = TfidfVectorizer(
                    ngram_range=(1, 2),  # Handle both unigrams and bigrams
                    stop_words='english',
                    min_df=1,
                    vocabulary=processed_keywords
                )

                try:
                    tfidf_matrix = vectorizer.fit_transform(cluster_texts)
                    feature_names = vectorizer.get_feature_names_out()

                    cluster_results = []
                    for doc_idx, (original_idx, text) in enumerate(zip(cluster_indices, cluster_texts)):
                        doc_scores = []
                        tfidf_scores = tfidf_matrix[doc_idx].toarray()[0]

                        for keyword in processed_keywords:
                            if keyword in feature_names:
                                keyword_idx = list(feature_names).index(keyword)
                                score = tfidf_scores[keyword_idx]
                            else:
                                score = 0.0
                            doc_scores.append({
                                "keyword": keyword,
                                "relevance_score": float(score),
                                "normalized_score": float(score / max(tfidf_scores) if max(tfidf_scores) > 0 else 0)
                            })

                        # Sort by relevance score
                        doc_scores.sort(key=lambda x: x["relevance_score"], reverse=True)

                        cluster_results.append({
                            "document_index": original_idx,
                            "text": text,
                            "topic_relevance_scores": doc_scores,
                            "top_keywords": doc_scores[:5]  # Top 5 most relevant keywords
                        })

                    results[cluster_id] = {
                        "cluster_id": cluster_id,
                        "method": "tfidf",
                        "documents": cluster_results,
                        "cluster_keywords": processed_keywords
                    }

                except ValueError:
                    # Handle case where keywords are not found in documents
                    cluster_results = []
                    for original_idx, text in zip(cluster_indices, cluster_texts):
                        doc_scores = [{"keyword": kw, "relevance_score": 0.0, "normalized_score": 0.0} for kw in processed_keywords]
                        cluster_results.append({
                            "document_index": original_idx,
                            "text": text,
                            "topic_relevance_scores": doc_scores,
                            "top_keywords": []
                        })

                    results[cluster_id] = {
                        "cluster_id": cluster_id,
                        "method": "tfidf",
                        "documents": cluster_results,
                        "cluster_keywords": processed_keywords
                    }

            elif method == "frequency":
                # Use term frequency to calculate relevance
                cluster_results = []

                for original_idx, text in zip(cluster_indices, cluster_texts):
                    # Simple tokenization
                    words = re.findall(r'\b\w+\b', text.lower())
                    word_count = len(words)
                    word_freq = Counter(words)

                    doc_scores = []
                    for keyword in processed_keywords:
                        keyword_words = keyword.split()
                        # Count occurrences of keyword (handle multi-word keywords)
                        keyword_count = sum(1 for i in range(len(words) - len(keyword_words) + 1)
                                          if ' '.join(words[i:i+len(keyword_words)]) == keyword)

                        frequency = keyword_count / word_count if word_count > 0 else 0.0
                        doc_scores.append({
                            "keyword": keyword,
                            "relevance_score": frequency,
                            "normalized_score": frequency
                        })

                    # Sort by relevance score
                    doc_scores.sort(key=lambda x: x["relevance_score"], reverse=True)

                    cluster_results.append({
                        "document_index": original_idx,
                        "text": text,
                        "topic_relevance_scores": doc_scores,
                        "top_keywords": doc_scores[:5]
                    })

                results[cluster_id] = {
                    "cluster_id": cluster_id,
                    "method": "frequency",
                    "documents": cluster_results,
                    "cluster_keywords": processed_keywords
                }

        return results

    def get_cluster_topic_summary(self, relevance_results: Dict) -> Dict[int, Dict[str, any]]:
        """
        Get summary statistics for each cluster's topic relevance

        Args:
            relevance_results: Results from calculate_document_topic_relevance

        Returns:
            Summary statistics for each cluster
        """
        summary = {}

        for cluster_id, cluster_data in relevance_results.items():
            documents = cluster_data["documents"]

            if not documents:
                continue

            # Calculate average relevance scores for each keyword across all documents
            keyword_scores = {}
            for doc in documents:
                for score_info in doc["topic_relevance_scores"]:
                    keyword = score_info["keyword"]
                    score = score_info["relevance_score"]

                    if keyword not in keyword_scores:
                        keyword_scores[keyword] = []
                    keyword_scores[keyword].append(score)

            # Calculate statistics for each keyword
            keyword_stats = {}
            for keyword, scores in keyword_scores.items():
                keyword_stats[keyword] = {
                    "avg_relevance": float(np.mean(scores)),
                    "max_relevance": float(np.max(scores)),
                    "min_relevance": float(np.min(scores)),
                    "std_relevance": float(np.std(scores)),
                    "documents_with_keyword": sum(1 for s in scores if s > 0),
                    "total_documents": len(scores)
                }

            # Sort keywords by average relevance
            sorted_keywords = sorted(keyword_stats.items(), key=lambda x: x[1]["avg_relevance"], reverse=True)

            summary[cluster_id] = {
                "cluster_id": cluster_id,
                "total_documents": len(documents),
                "method": cluster_data["method"],
                "keyword_statistics": dict(sorted_keywords),
                "top_keywords_by_avg_relevance": [kw for kw, _ in sorted_keywords[:5]],
                "top_keywords_by_max_relevance": sorted(keyword_stats.items(), key=lambda x: x[1]["max_relevance"], reverse=True)[:5]
            }

        return summary