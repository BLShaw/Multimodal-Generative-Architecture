# evaluate.py
import os
import sys
import argparse
import numpy as np
import tensorflow as tf

# Add modules to path
sys.path.append('modules')

from config_controller import ConfigController
from image_autoencoder import ImageAutoencoder
from audio_autoencoder import AudioAutoencoder
from multimodal_clustering import MultimodalClustering
from brain_module import BrainModule
from utils import configure_gpu

def convert_numpy_to_python(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                         np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.void)):
        return None
    else:
        return obj

def main(args):
    # Configure GPU before doing anything else
    configure_gpu()
    
    # Load configuration
    config_controller = ConfigController(args.config)
    config = config_controller.config
    
    # Initialize autoencoders
    image_ae = ImageAutoencoder(config)
    audio_ae = AudioAutoencoder(config)
    
    # Load trained models
    print("Loading trained models...")
    image_ae.load_model(os.path.join(config['paths']['model_dir'], 'image_autoencoder'))
    audio_ae.load_model(os.path.join(config['paths']['model_dir'], 'audio_autoencoder'))
    print("Models loaded successfully.")
    
    # Load latent representations and labels
    print("Loading latent representations...")
    image_latent = np.load(os.path.join(config['paths']['output_dir'], 'image_latent.npy'))
    audio_latent = np.load(os.path.join(config['paths']['output_dir'], 'audio_latent.npy'))
    y_test_img = np.load(os.path.join(config['paths']['output_dir'], 'y_test_img.npy'))
    y_test_aud = np.load(os.path.join(config['paths']['output_dir'], 'y_test_aud.npy'))
    
    print(f"Image latent shape: {image_latent.shape}")
    print(f"Audio latent shape: {audio_latent.shape}")
    
    # Initialize clustering module
    clustering = MultimodalClustering(config)
    
    # Cluster data
    print("Clustering data...")
    img_clusters, img_silhouette = clustering.cluster_image_data(image_latent)
    aud_clusters, aud_silhouette = clustering.cluster_audio_data(audio_latent)
    
    # For joint clustering, we need to have the same number of samples for both modalities
    min_samples = min(len(image_latent), len(audio_latent))
    print(f"Using {min_samples} samples for joint clustering")
    
    # Take the first min_samples from each modality
    image_latent_subset = image_latent[:min_samples]
    audio_latent_subset = audio_latent[:min_samples]
    y_test_img_subset = y_test_img[:min_samples]
    y_test_aud_subset = y_test_aud[:min_samples]
    
    # Create subset of clusters
    img_clusters_subset = img_clusters[:min_samples]
    aud_clusters_subset = aud_clusters[:min_samples]
    
    # Create indices arrays for the subset
    image_indices = np.arange(min_samples)
    audio_indices = np.arange(min_samples)
    
    joint_clusters, joint_silhouette = clustering.cluster_joint_data(image_latent_subset, audio_latent_subset)
    
    # Establish relationships between modalities using the subset indices
    relationships = clustering.establish_relationships(
        y_test_img_subset, y_test_aud_subset, image_indices, audio_indices
    )
    
    # Analyze convergence and divergence using the subset of clusters
    analysis_results = clustering.analyze_convergence_divergence(
        image_latent_subset, audio_latent_subset, img_clusters_subset, aud_clusters_subset
    )
    
    # Save clustering results
    clustering.save_clusters(config['paths']['cluster_dir'])
    
    # Save additional results for visualization
    np.save(os.path.join(config['paths']['cluster_dir'], 'img_clusters.npy'), img_clusters)
    np.save(os.path.join(config['paths']['cluster_dir'], 'aud_clusters.npy'), aud_clusters)
    np.save(os.path.join(config['paths']['cluster_dir'], 'joint_clusters.npy'), joint_clusters)
    
    # Save metrics
    metrics = {
        'img_silhouette': float(img_silhouette),
        'aud_silhouette': float(aud_silhouette),
        'joint_silhouette': float(joint_silhouette)
    }
    
    import json
    with open(os.path.join(config['paths']['cluster_dir'], 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Convert analysis results to JSON-serializable format
    analysis_results_json = convert_numpy_to_python(analysis_results)
    
    # Save analysis results
    with open(os.path.join(config['paths']['cluster_dir'], 'analysis_results.json'), 'w') as f:
        json.dump(analysis_results_json, f, indent=4)
    
    print("Evaluation completed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Multimodal Generative Architecture')
    parser.add_argument('--config', type=str, default='config/config.json', help='Path to configuration file')
    args = parser.parse_args()
    
    main(args)