# modules/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib import gridspec
import librosa
import librosa.display

class Visualization:
    def __init__(self, config):
        self.config = config
        self.plot_dir = config['paths']['plot_dir']
        os.makedirs(self.plot_dir, exist_ok=True)
    
    def plot_image_reconstruction(self, original, reconstructed, n=10):
        """Plot original and reconstructed images"""
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # Original images
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(original[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
            # Reconstructed images
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(reconstructed[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        
        plt.savefig(os.path.join(self.plot_dir, 'image_reconstruction.png'))
        plt.close()
    
    def plot_audio_reconstruction(self, original, reconstructed, n=3, sample_rate=8000):
        """Plot original and reconstructed audio waveforms"""
        plt.figure(figsize=(15, 8))
        for i in range(n):
            # Original audio
            plt.subplot(n, 2, i*2 + 1)
            librosa.display.waveshow(original[i], sr=sample_rate)
            plt.title(f'Original Audio {i+1}')
            
            # Reconstructed audio
            plt.subplot(n, 2, i*2 + 2)
            librosa.display.waveshow(reconstructed[i], sr=sample_rate)
            plt.title(f'Reconstructed Audio {i+1}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'audio_reconstruction.png'))
        plt.close()
    
    def plot_latent_space(self, latent, labels=None, title='Latent Space'):
        """Plot latent space using t-SNE"""
        # Reduce dimensionality to 2D using t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        latent_2d = tsne.fit_transform(latent)
        
        plt.figure(figsize=(10, 8))
        if labels is not None:
            scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, label='Class')
        else:
            plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6)
        
        plt.title(title)
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        plt.savefig(os.path.join(self.plot_dir, f'{title.lower().replace(" ", "_")}.png'))
        plt.close()
    
    def plot_clusters(self, latent, clusters, title='Cluster Visualization'):
        """Plot clustering results"""
        # Reduce dimensionality to 2D using t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        latent_2d = tsne.fit_transform(latent)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Cluster')
        
        plt.title(title)
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        plt.savefig(os.path.join(self.plot_dir, f'{title.lower().replace(" ", "_")}.png'))
        plt.close()
    
    def plot_relationship_heatmap(self, relationships, title='Modality Relationships'):
        """Plot heatmap of relationships between modalities"""
        if not relationships['one_to_one']:
            print("No relationships to plot")
            return
        
        # Prepare data for heatmap
        img_clusters = set()
        aud_clusters = set()
        for (img, aud), count in relationships['one_to_one'].items():
            img_clusters.add(img)
            aud_clusters.add(aud)
        
        img_clusters = sorted(img_clusters)
        aud_clusters = sorted(aud_clusters)
        
        # Create matrix
        matrix = np.zeros((len(img_clusters), len(aud_clusters)))
        for i, img in enumerate(img_clusters):
            for j, aud in enumerate(aud_clusters):
                matrix[i, j] = relationships['one_to_one'].get((img, aud), 0)
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='g', cmap='viridis',
                   xticklabels=aud_clusters, yticklabels=img_clusters)
        plt.title(title)
        plt.xlabel('Audio Clusters')
        plt.ylabel('Image Clusters')
        plt.savefig(os.path.join(self.plot_dir, f'{title.lower().replace(" ", "_")}.png'))
        plt.close()
    
    def plot_convergence_divergence(self, analysis_results, title='Convergence and Divergence Analysis'):
        """Plot convergence and divergence zones"""
        distance_matrix = analysis_results['distance_matrix']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(distance_matrix, annot=True, fmt='.3f', cmap='coolwarm')
        plt.title(title)
        plt.xlabel('Audio Clusters')
        plt.ylabel('Image Clusters')
        plt.savefig(os.path.join(self.plot_dir, f'{title.lower().replace(" ", "_")}.png'))
        plt.close()
    
    def plot_attention_weights(self, attention_weights, title='Modality Attention Weights'):
        """Plot attention weights for modalities"""
        plt.figure(figsize=(10, 6))
        
        # Create histogram of attention weights
        plt.hist(attention_weights[:, 0], alpha=0.5, label='Image Attention')
        plt.hist(attention_weights[:, 1], alpha=0.5, label='Audio Attention')
        
        plt.title(title)
        plt.xlabel('Attention Weight')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(self.plot_dir, f'{title.lower().replace(" ", "_")}.png'))
        plt.close()
    
    def plot_training_history(self, history, title='Training History'):
        """Plot training history"""
        plt.figure(figsize=(12, 6))
        
        # Plot training & validation accuracy values
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        # Plot training & validation loss values
        if 'accuracy' in history.history:
            plt.subplot(1, 2, 2)
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='lower right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f'{title.lower().replace(" ", "_")}.png'))
        plt.close()
    
    def generate_synthetic_samples(self, generator, n_samples=5, title='Synthetic Samples'):
        """Generate and plot synthetic samples from a generator"""
        # Generate random latent vectors
        latent_dim = generator.input_shape[1]
        random_latent = np.random.normal(0, 1, (n_samples, latent_dim))
        
        # Generate samples
        generated = generator.predict(random_latent)
        
        # Plot generated samples
        plt.figure(figsize=(15, 3))
        for i in range(n_samples):
            plt.subplot(1, n_samples, i + 1)
            plt.imshow(generated[i].reshape(28, 28))
            plt.gray()
            plt.axis('off')
        
        plt.suptitle(title)
        plt.savefig(os.path.join(self.plot_dir, f'{title.lower().replace(" ", "_")}.png'))
        plt.close()
        
        return generated