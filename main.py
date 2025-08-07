# main.py
import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf

# Add modules to path
sys.path.append('modules')

from config_controller import ConfigController
from data_loader import DataLoader
from image_autoencoder import ImageAutoencoder
from audio_autoencoder import AudioAutoencoder
from multimodal_clustering import MultimodalClustering
from brain_module import BrainModule
from visualization import Visualization
from utils import configure_gpu

def main(args):
    # Configure GPU before doing anything else
    configure_gpu()
    
    # Load configuration
    config_controller = ConfigController(args.config)
    config = config_controller.config
    
    # Create output directories
    for path in config['paths'].values():
        os.makedirs(path, exist_ok=True)
    
    # Initialize data loader
    data_loader = DataLoader(config)
    
    # Load datasets
    (x_train_img, y_train_img), (x_test_img, y_test_img) = data_loader.load_mnist()
    (x_train_aud, y_train_aud), (x_test_aud, y_test_aud) = data_loader.load_fsdd()
    
    print(f"Image train shape: {x_train_img.shape}, Image test shape: {x_test_img.shape}")
    print(f"Audio train shape: {x_train_aud.shape}, Audio test shape: {x_test_aud.shape}")
    
    # Initialize autoencoders
    image_ae = ImageAutoencoder(config)
    audio_ae = AudioAutoencoder(config)
    
    # Train or load models
    if args.mode == 'train' or not os.path.exists(os.path.join(config['paths']['model_dir'], 'image_autoencoder', 'image_autoencoder.h5')):
        print("Training image autoencoder...")
        img_history = image_ae.train(x_train_img, x_test_img)
        image_ae.save_model(os.path.join(config['paths']['model_dir'], 'image_autoencoder'))
        
        # Plot training history
        viz = Visualization(config)
        viz.plot_training_history(img_history, 'Image Autoencoder Training History')
    else:
        print("Loading image autoencoder...")
        image_ae.load_model(os.path.join(config['paths']['model_dir'], 'image_autoencoder'))
    
    if args.mode == 'train' or not os.path.exists(os.path.join(config['paths']['model_dir'], 'audio_autoencoder', 'audio_autoencoder.h5')):
        print("Training audio autoencoder...")
        aud_history = audio_ae.train(x_train_aud, x_test_aud)
        audio_ae.save_model(os.path.join(config['paths']['model_dir'], 'audio_autoencoder'))
        
        # Plot training history
        viz = Visualization(config)
        viz.plot_training_history(aud_history, 'Audio Autoencoder Training History')
    else:
        print("Loading audio autoencoder...")
        audio_ae.load_model(os.path.join(config['paths']['model_dir'], 'audio_autoencoder'))
    
    # Encode data to latent space
    print("Encoding data to latent space...")
    image_latent = image_ae.encode(x_test_img)
    audio_latent = audio_ae.encode(x_test_aud)
    
    print(f"Image latent shape: {image_latent.shape}")
    print(f"Audio latent shape: {audio_latent.shape}")
    
    # Visualize reconstructions
    print("Visualizing reconstructions...")
    viz = Visualization(config)
    
    # Image reconstructions
    reconstructed_img = image_ae.decode(image_latent)
    viz.plot_image_reconstruction(x_test_img, reconstructed_img)
    
    # Audio reconstructions
    reconstructed_aud = audio_ae.decode(audio_latent)
    viz.plot_audio_reconstruction(x_test_aud, reconstructed_aud, sample_rate=config['data']['fsdd']['sample_rate'])
    
    # Visualize latent spaces
    viz.plot_latent_space(image_latent, y_test_img, 'Image Latent Space')
    viz.plot_latent_space(audio_latent, y_test_aud, 'Audio Latent Space')
    
    # Initialize clustering module
    clustering = MultimodalClustering(config)
    
    # Cluster data
    print("Clustering data...")
    img_clusters, img_silhouette = clustering.cluster_image_data(image_latent)
    aud_clusters, aud_silhouette = clustering.cluster_audio_data(audio_latent)
    
    # For joint clustering, we need to have the same number of samples for both modalities
    # We'll use the minimum number of samples available
    min_samples = min(len(image_latent), len(audio_latent))
    print(f"Using {min_samples} samples for joint clustering")
    
    # Take the first min_samples from each modality
    image_latent_subset = image_latent[:min_samples]
    audio_latent_subset = audio_latent[:min_samples]
    y_test_img_subset = y_test_img[:min_samples]
    y_test_aud_subset = y_test_aud[:min_samples]
    
    joint_clusters, joint_silhouette = clustering.cluster_joint_data(image_latent_subset, audio_latent_subset)
    
    # Visualize clusters
    viz.plot_clusters(image_latent, img_clusters, 'Image Clusters')
    viz.plot_clusters(audio_latent, aud_clusters, 'Audio Clusters')
    viz.plot_clusters(np.concatenate([image_latent_subset, audio_latent_subset], axis=1), 
                     joint_clusters, 'Joint Clusters')
    
    # Establish relationships between modalities
    relationships = clustering.establish_relationships(y_test_img_subset, y_test_aud_subset)
    
    # Visualize relationships
    viz.plot_relationship_heatmap(relationships)
    
    # Analyze convergence and divergence
    analysis_results = clustering.analyze_convergence_divergence(image_latent_subset, audio_latent_subset)
    
    # Visualize convergence and divergence
    viz.plot_convergence_divergence(analysis_results)
    
    # Save clustering results
    clustering.save_clusters(config['paths']['cluster_dir'])
    
    # Initialize brain module
    image_latent_dim = config['model']['image_autoencoder']['latent_dim']
    audio_latent_dim = config['model']['audio_autoencoder']['latent_dim']
    brain = BrainModule(config, image_latent_dim, audio_latent_dim)
    
    # Train or load brain model
    if args.mode == 'train' or not os.path.exists(os.path.join(config['paths']['model_dir'], 'brain_model.h5')):
        print("Training brain module...")
        brain_history = brain.train(image_latent_subset, audio_latent_subset)
        brain.save_model(config['paths']['model_dir'])
        
        # Plot training history
        viz.plot_training_history(brain_history, 'Brain Module Training History')
    else:
        print("Loading brain module...")
        brain.load_model(config['paths']['model_dir'])
    
    # Process data through brain module
    print("Processing data through brain module...")
    img_recon, aud_recon, joint_rep, attention = brain.process(image_latent_subset, audio_latent_subset)
    
    # Visualize attention weights
    viz.plot_attention_weights(attention)
    
    # Visualize joint representation
    viz.plot_latent_space(joint_rep, title='Joint Representation')
    
    # Generate synthetic samples
    print("Generating synthetic samples...")
    synthetic_img = viz.generate_synthetic_samples(image_ae.decoder, title='Synthetic Images')
    
    # Save synthetic samples
    np.save(os.path.join(config['paths']['synthetic_dir'], 'synthetic_images.npy'), synthetic_img)
    
    print("Process completed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multimodal Generative Architecture')
    parser.add_argument('--config', type=str, default='config/config.json', help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'], help='Mode to run the script in')
    args = parser.parse_args()
    
    main(args)