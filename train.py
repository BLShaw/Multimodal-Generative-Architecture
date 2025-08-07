# train.py
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
    
    # Train image autoencoder
    print("Training image autoencoder...")
    img_history = image_ae.train(x_train_img, x_test_img)
    image_ae.save_model(os.path.join(config['paths']['model_dir'], 'image_autoencoder'))
    print("Image autoencoder trained and saved.")
    
    # Train audio autoencoder
    print("Training audio autoencoder...")
    aud_history = audio_ae.train(x_train_aud, x_test_aud)
    audio_ae.save_model(os.path.join(config['paths']['model_dir'], 'audio_autoencoder'))
    print("Audio autoencoder trained and saved.")
    
    # Encode data to latent space
    print("Encoding data to latent space...")
    image_latent = image_ae.encode(x_test_img)
    audio_latent = audio_ae.encode(x_test_aud)
    
    print(f"Image latent shape: {image_latent.shape}")
    print(f"Audio latent shape: {audio_latent.shape}")
    
    # Save latent representations
    np.save(os.path.join(config['paths']['output_dir'], 'image_latent.npy'), image_latent)
    np.save(os.path.join(config['paths']['output_dir'], 'audio_latent.npy'), audio_latent)
    np.save(os.path.join(config['paths']['output_dir'], 'y_test_img.npy'), y_test_img)
    np.save(os.path.join(config['paths']['output_dir'], 'y_test_aud.npy'), y_test_aud)
    
    # For joint processing, we need to have the same number of samples for both modalities
    min_samples = min(len(image_latent), len(audio_latent))
    print(f"Using {min_samples} samples for joint processing")
    
    # Take the first min_samples from each modality
    image_latent_subset = image_latent[:min_samples]
    audio_latent_subset = audio_latent[:min_samples]
    y_test_img_subset = y_test_img[:min_samples]
    y_test_aud_subset = y_test_aud[:min_samples]
    
    # Initialize brain module
    image_latent_dim = config['model']['image_autoencoder']['latent_dim']
    audio_latent_dim = config['model']['audio_autoencoder']['latent_dim']
    brain = BrainModule(config, image_latent_dim, audio_latent_dim)
    
    # Train brain module
    print("Training brain module...")
    brain_history = brain.train(image_latent_subset, audio_latent_subset)
    brain.save_model(config['paths']['model_dir'])
    print("Brain module trained and saved.")
    
    # Process data through brain module
    print("Processing data through brain module...")
    img_recon, aud_recon, joint_rep, attention = brain.process(image_latent_subset, audio_latent_subset)
    
    # Save brain module outputs
    np.save(os.path.join(config['paths']['output_dir'], 'joint_rep.npy'), joint_rep)
    np.save(os.path.join(config['paths']['output_dir'], 'attention.npy'), attention)
    
    print("Training completed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Multimodal Generative Architecture')
    parser.add_argument('--config', type=str, default='config/config.json', help='Path to configuration file')
    args = parser.parse_args()
    
    main(args)