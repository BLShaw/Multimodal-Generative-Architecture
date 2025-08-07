# modules/multimodal_clustering.py
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import os
import pickle

class MultimodalClustering:
    def __init__(self, config):
        self.config = config
        self.model_config = config['model']['clustering']
        self.image_clusters = None
        self.audio_clusters = None
        self.joint_clusters = None
        self.relationships = {
            'one_to_one': {},
            'one_to_many': {}
        }
    
    def cluster_image_data(self, image_latent):
        print("Clustering image data...")
        
        if self.model_config['algorithm'] == 'kmeans':
            kmeans = KMeans(
                n_clusters=self.model_config['n_clusters'],
                random_state=42,
                max_iter=1000
            )
            self.image_clusters = kmeans.fit_predict(image_latent)
            
            # Calculate silhouette score
            silhouette = silhouette_score(image_latent, self.image_clusters)
            print(f"Image clustering silhouette score: {silhouette:.4f}")
            
            return self.image_clusters, silhouette
        
        elif self.model_config['algorithm'] == 'dbscan':
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            self.image_clusters = dbscan.fit_predict(image_latent)
            
            # Calculate silhouette score (excluding noise points)
            if len(set(self.image_clusters)) > 1:  # More than one cluster (excluding noise)
                mask = self.image_clusters != -1
                silhouette = silhouette_score(image_latent[mask], self.image_clusters[mask])
                print(f"Image clustering silhouette score: {silhouette:.4f}")
            else:
                silhouette = -1  # Not enough clusters
                print("Not enough clusters for silhouette score calculation")
            
            return self.image_clusters, silhouette
        
        elif self.model_config['algorithm'] == 'agglomerative':
            agg = AgglomerativeClustering(
                n_clusters=self.model_config['n_clusters']
            )
            self.image_clusters = agg.fit_predict(image_latent)
            
            # Calculate silhouette score
            silhouette = silhouette_score(image_latent, self.image_clusters)
            print(f"Image clustering silhouette score: {silhouette:.4f}")
            
            return self.image_clusters, silhouette
    
    def cluster_audio_data(self, audio_latent):
        print("Clustering audio data...")
        
        if self.model_config['algorithm'] == 'kmeans':
            kmeans = KMeans(
                n_clusters=self.model_config['n_clusters'],
                random_state=42,
                max_iter=1000
            )
            self.audio_clusters = kmeans.fit_predict(audio_latent)
            
            # Calculate silhouette score
            silhouette = silhouette_score(audio_latent, self.audio_clusters)
            print(f"Audio clustering silhouette score: {silhouette:.4f}")
            
            return self.audio_clusters, silhouette
        
        elif self.model_config['algorithm'] == 'dbscan':
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            self.audio_clusters = dbscan.fit_predict(audio_latent)
            
            # Calculate silhouette score (excluding noise points)
            if len(set(self.audio_clusters)) > 1:  # More than one cluster (excluding noise)
                mask = self.audio_clusters != -1
                silhouette = silhouette_score(audio_latent[mask], self.audio_clusters[mask])
                print(f"Audio clustering silhouette score: {silhouette:.4f}")
            else:
                silhouette = -1  # Not enough clusters
                print("Not enough clusters for silhouette score calculation")
            
            return self.audio_clusters, silhouette
        
        elif self.model_config['algorithm'] == 'agglomerative':
            agg = AgglomerativeClustering(
                n_clusters=self.model_config['n_clusters']
            )
            self.audio_clusters = agg.fit_predict(audio_latent)
            
            # Calculate silhouette score
            silhouette = silhouette_score(audio_latent, self.audio_clusters)
            print(f"Audio clustering silhouette score: {silhouette:.4f}")
            
            return self.audio_clusters, silhouette
    
    def cluster_joint_data(self, image_latent, audio_latent):
        print("Clustering joint multimodal data...")
        
        # Combine latent representations
        joint_latent = np.concatenate([image_latent, audio_latent], axis=1)
        
        if self.model_config['algorithm'] == 'kmeans':
            kmeans = KMeans(
                n_clusters=self.model_config['n_clusters'],
                random_state=42,
                max_iter=1000
            )
            self.joint_clusters = kmeans.fit_predict(joint_latent)
            
            # Calculate silhouette score
            silhouette = silhouette_score(joint_latent, self.joint_clusters)
            print(f"Joint clustering silhouette score: {silhouette:.4f}")
            
            return self.joint_clusters, silhouette
        
        elif self.model_config['algorithm'] == 'dbscan':
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            self.joint_clusters = dbscan.fit_predict(joint_latent)
            
            # Calculate silhouette score (excluding noise points)
            if len(set(self.joint_clusters)) > 1:  # More than one cluster (excluding noise)
                mask = self.joint_clusters != -1
                silhouette = silhouette_score(joint_latent[mask], self.joint_clusters[mask])
                print(f"Joint clustering silhouette score: {silhouette:.4f}")
            else:
                silhouette = -1  # Not enough clusters
                print("Not enough clusters for silhouette score calculation")
            
            return self.joint_clusters, silhouette
        
        elif self.model_config['algorithm'] == 'agglomerative':
            agg = AgglomerativeClustering(
                n_clusters=self.model_config['n_clusters']
            )
            self.joint_clusters = agg.fit_predict(joint_latent)
            
            # Calculate silhouette score
            silhouette = silhouette_score(joint_latent, self.joint_clusters)
            print(f"Joint clustering silhouette score: {silhouette:.4f}")
            
            return self.joint_clusters, silhouette
    
    def establish_relationships(self, image_labels, audio_labels, image_indices=None, audio_indices=None):
        """Establish relationships between image and audio clusters"""
        print("Establishing relationships between modalities...")
        
        # If indices are provided, use only those samples
        if image_indices is not None and audio_indices is not None:
            image_clusters_subset = self.image_clusters[image_indices]
            audio_clusters_subset = self.audio_clusters[audio_indices]
            image_labels_subset = image_labels[image_indices]
            audio_labels_subset = audio_labels[audio_indices]
        else:
            image_clusters_subset = self.image_clusters
            audio_clusters_subset = self.audio_clusters
            image_labels_subset = image_labels
            audio_labels_subset = audio_labels
        
        # One-to-one relationships
        for img_cluster in np.unique(image_clusters_subset):
            for aud_cluster in np.unique(audio_clusters_subset):
                # Find samples that belong to both clusters
                img_mask = image_clusters_subset == img_cluster
                aud_mask = audio_clusters_subset == aud_cluster
                intersection = np.sum(img_mask & aud_mask)
                
                if intersection > 0:
                    self.relationships['one_to_one'][(img_cluster, aud_cluster)] = intersection
        
        # One-to-many relationships
        for img_cluster in np.unique(image_clusters_subset):
            related_audio = []
            for aud_cluster in np.unique(audio_clusters_subset):
                img_mask = image_clusters_subset == img_cluster
                aud_mask = audio_clusters_subset == aud_cluster
                intersection = np.sum(img_mask & aud_mask)
                
                if intersection > 0:
                    related_audio.append((aud_cluster, intersection))
            
            if related_audio:
                self.relationships['one_to_many'][img_cluster] = related_audio
        
        for aud_cluster in np.unique(audio_clusters_subset):
            related_images = []
            for img_cluster in np.unique(image_clusters_subset):
                img_mask = image_clusters_subset == img_cluster
                aud_mask = audio_clusters_subset == aud_cluster
                intersection = np.sum(img_mask & aud_mask)
                
                if intersection > 0:
                    related_images.append((img_cluster, intersection))
            
            if related_images:
                self.relationships['one_to_many'][aud_cluster] = related_images
        
        return self.relationships
    
    def analyze_convergence_divergence(self, image_latent, audio_latent, image_clusters_subset=None, audio_clusters_subset=None):
        """Analyze convergence and divergence zones between modalities"""
        print("Analyzing convergence and divergence zones...")
        
        # Calculate distances between image and audio clusters
        convergence_zones = []
        divergence_zones = []
        
        # Get cluster centers
        if image_clusters_subset is None:
            unique_img_clusters = np.unique(self.image_clusters)
        else:
            unique_img_clusters = np.unique(image_clusters_subset)
            
        if audio_clusters_subset is None:
            unique_aud_clusters = np.unique(self.audio_clusters)
        else:
            unique_aud_clusters = np.unique(audio_clusters_subset)
        
        img_centers = []
        for cluster in unique_img_clusters:
            if image_clusters_subset is None:
                mask = self.image_clusters == cluster
                center = np.mean(image_latent[mask], axis=0)
            else:
                mask = image_clusters_subset == cluster
                center = np.mean(image_latent[mask], axis=0)
            img_centers.append(center)
        
        aud_centers = []
        for cluster in unique_aud_clusters:
            if audio_clusters_subset is None:
                mask = self.audio_clusters == cluster
                center = np.mean(audio_latent[mask], axis=0)
            else:
                mask = audio_clusters_subset == cluster
                center = np.mean(audio_latent[mask], axis=0)
            aud_centers.append(center)
        
        # Calculate distances between cluster centers
        distances = np.zeros((len(unique_img_clusters), len(unique_aud_clusters)))
        for i, img_center in enumerate(img_centers):
            for j, aud_center in enumerate(aud_centers):
                # Pad the smaller representation to match dimensions
                max_dim = max(len(img_center), len(aud_center))
                img_padded = np.pad(img_center, (0, max_dim - len(img_center)))
                aud_padded = np.pad(aud_center, (0, max_dim - len(aud_center)))
                
                # Calculate cosine similarity
                dot_product = np.dot(img_padded, aud_padded)
                norm_img = np.linalg.norm(img_padded)
                norm_aud = np.linalg.norm(aud_padded)
                
                if norm_img > 0 and norm_aud > 0:
                    similarity = dot_product / (norm_img * norm_aud)
                    distances[i, j] = 1 - similarity  # Convert to distance
                else:
                    distances[i, j] = 1.0  # Maximum distance
        
        # Identify convergence and divergence zones
        threshold = self.model_config['convergence_threshold']
        
        for i, img_cluster in enumerate(unique_img_clusters):
            for j, aud_cluster in enumerate(unique_aud_clusters):
                distance = distances[i, j]
                
                if distance < threshold:
                    convergence_zones.append((img_cluster, aud_cluster, distance))
                else:
                    divergence_zones.append((img_cluster, aud_cluster, distance))
        
        return {
            'convergence_zones': convergence_zones,
            'divergence_zones': divergence_zones,
            'distance_matrix': distances
        }
    
    def save_clusters(self, path):
        """Save clustering results to disk"""
        if not os.path.exists(path):
            os.makedirs(path)
        
        with open(os.path.join(path, 'image_clusters.pkl'), 'wb') as f:
            pickle.dump(self.image_clusters, f)
        
        with open(os.path.join(path, 'audio_clusters.pkl'), 'wb') as f:
            pickle.dump(self.audio_clusters, f)
        
        with open(os.path.join(path, 'joint_clusters.pkl'), 'wb') as f:
            pickle.dump(self.joint_clusters, f)
        
        with open(os.path.join(path, 'relationships.pkl'), 'wb') as f:
            pickle.dump(self.relationships, f)
    
    def load_clusters(self, path):
        """Load clustering results from disk"""
        with open(os.path.join(path, 'image_clusters.pkl'), 'rb') as f:
            self.image_clusters = pickle.load(f)
        
        with open(os.path.join(path, 'audio_clusters.pkl'), 'rb') as f:
            self.audio_clusters = pickle.load(f)
        
        with open(os.path.join(path, 'joint_clusters.pkl'), 'rb') as f:
            self.joint_clusters = pickle.load(f)
        
        with open(os.path.join(path, 'relationships.pkl'), 'rb') as f:
            self.relationships = pickle.load(f)