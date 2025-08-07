# visualize.py
import os
import sys
import argparse
import numpy as np
import tensorflow as tf

# Add modules to path
sys.path.append('modules')

from config_controller import ConfigController
from data_loader import DataLoader
from image_autoencoder import ImageAutoencoder
from audio_autoencoder import AudioAutoencoder
from brain_module import BrainModule
from visualization import Visualization
from utils import configure_gpu

def main(args):
    # Configure GPU before doing anything else
    configure_gpu()
    
    # Load configuration
    config_controller = ConfigController(args.config)
    config = config_controller.config
    
    # Initialize data loader
    data_loader = DataLoader(config)
    
    # Load datasets
    (x_train_img, y_train_img), (x_test_img, y_test_img) = data_loader.load_mnist()
    (x_train_aud, y_train_aud), (x_test_aud, y_test_aud) = data_loader.load_fsdd()
    
    # Initialize autoencoders
    image_ae = ImageAutoencoder(config)
    audio_ae = AudioAutoencoder(config)
    
    # Load trained models
    print("Loading trained models...")
    image_ae.load_model(os.path.join(config['paths']['model_dir'], 'image_autoencoder'))
    audio_ae.load_model(os.path.join(config['paths']['model_dir'], 'audio_autoencoder'))
    
    # Load brain module
    image_latent_dim = config['model']['image_autoencoder']['latent_dim']
    audio_latent_dim = config['model']['audio_autoencoder']['latent_dim']
    brain = BrainModule(config, image_latent_dim, audio_latent_dim)
    brain.load_model(config['paths']['model_dir'])
    
    # Load latent representations and labels
    print("Loading latent representations...")
    image_latent = np.load(os.path.join(config['paths']['output_dir'], 'image_latent.npy'))
    audio_latent = np.load(os.path.join(config['paths']['output_dir'], 'audio_latent.npy'))
    y_test_img = np.load(os.path.join(config['paths']['output_dir'], 'y_test_img.npy'))
    y_test_aud = np.load(os.path.join(config['paths']['output_dir'], 'y_test_aud.npy'))
    
    # Load brain module outputs
    joint_rep = np.load(os.path.join(config['paths']['output_dir'], 'joint_rep.npy'))
    attention = np.load(os.path.join(config['paths']['output_dir'], 'attention.npy'))
    
    # Load clustering results
    img_clusters = np.load(os.path.join(config['paths']['cluster_dir'], 'img_clusters.npy'))
    aud_clusters = np.load(os.path.join(config['paths']['cluster_dir'], 'aud_clusters.npy'))
    joint_clusters = np.load(os.path.join(config['paths']['cluster_dir'], 'joint_clusters.npy'))
    
    # Initialize visualization
    viz = Visualization(config)
    
    # Visualize reconstructions
    print("Visualizing reconstructions...")
    reconstructed_img = image_ae.decode(image_latent)
    reconstructed_aud = audio_ae.decode(audio_latent)
    
    viz.plot_image_reconstruction(x_test_img, reconstructed_img)
    viz.plot_audio_reconstruction(x_test_aud, reconstructed_aud, sample_rate=config['data']['fsdd']['sample_rate'])
    
    # Visualize latent spaces
    print("Visualizing latent spaces...")
    viz.plot_latent_space(image_latent, y_test_img, 'Image Latent Space')
    viz.plot_latent_space(audio_latent, y_test_aud, 'Audio Latent Space')
    viz.plot_latent_space(joint_rep, title='Joint Representation')
    
    # Visualize clusters
    print("Visualizing clusters...")
    viz.plot_clusters(image_latent, img_clusters, 'Image Clusters')
    viz.plot_clusters(audio_latent, aud_clusters, 'Audio Clusters')
    
    # For joint clustering visualization
    min_samples = min(len(image_latent), len(audio_latent))
    image_latent_subset = image_latent[:min_samples]
    audio_latent_subset = audio_latent[:min_samples]
    viz.plot_clusters(np.concatenate([image_latent_subset, audio_latent_subset], axis=1), 
                     joint_clusters, 'Joint Clusters')
    
    # Visualize attention weights
    print("Visualizing attention weights...")
    viz.plot_attention_weights(attention)
    
    # Generate synthetic samples
    print("Generating synthetic samples...")
    synthetic_img = viz.generate_synthetic_samples(image_ae.decoder, title='Synthetic Images')
    
    # Save synthetic samples
    np.save(os.path.join(config['paths']['synthetic_dir'], 'synthetic_images.npy'), synthetic_img)
    
    print("Visualization completed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Multimodal Generative Architecture Results')
    parser.add_argument('--config', type=str, default='config/config.json', help='Path to configuration file')
    args = parser.parse_args()
    
    main(args)