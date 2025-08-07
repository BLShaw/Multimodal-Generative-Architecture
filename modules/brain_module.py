# modules/brain_module.py
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

class BrainModule:
    def __init__(self, config, image_latent_dim, audio_latent_dim):
        self.config = config
        self.model = self._build_model(image_latent_dim, audio_latent_dim)
    
    def _build_model(self, image_latent_dim, audio_latent_dim):
        # Inputs from both modalities
        image_input = layers.Input(shape=(image_latent_dim,), name='image_input')
        audio_input = layers.Input(shape=(audio_latent_dim,), name='audio_input')
        
        # Process each modality separately
        x_img = layers.Dense(64, activation='relu')(image_input)
        x_img = layers.Dense(32, activation='relu')(x_img)
        
        x_aud = layers.Dense(64, activation='relu')(audio_input)
        x_aud = layers.Dense(32, activation='relu')(x_aud)
        
        # Combine modalities
        combined = layers.Concatenate()([x_img, x_aud])
        
        # Joint processing
        x = layers.Dense(64, activation='relu')(combined)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # Output different types of information
        # 1. Cross-modal reconstruction
        img_reconstruction = layers.Dense(image_latent_dim, activation='linear', name='img_recon')(x)
        aud_reconstruction = layers.Dense(audio_latent_dim, activation='linear', name='aud_recon')(x)
        
        # 2. Joint representation
        joint_representation = layers.Dense(16, activation='relu', name='joint_rep')(x)
        
        # 3. Attention weights
        attention = layers.Dense(2, activation='softmax', name='attention')(x)
        
        # Create model
        brain_model = models.Model(
            inputs=[image_input, audio_input],
            outputs=[
                img_reconstruction, 
                aud_reconstruction, 
                joint_representation,
                attention
            ]
        )
        
        # Compile with different losses for different outputs
        brain_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'img_recon': 'mse',
                'aud_recon': 'mse',
                'joint_rep': 'mse',
                'attention': 'categorical_crossentropy'
            },
            loss_weights={
                'img_recon': 0.3,
                'aud_recon': 0.3,
                'joint_rep': 0.2,
                'attention': 0.2
            }
        )
        
        return brain_model
    
    def train(self, image_latent, audio_latent, epochs=20, batch_size=32):
        # Create dummy targets for training
        img_recon_target = image_latent
        aud_recon_target = audio_latent
        joint_rep_target = np.random.rand(len(image_latent), 16)  # Dummy target
        attention_target = np.array([[0.5, 0.5]] * len(image_latent))  # Balanced attention
        
        history = self.model.fit(
            [image_latent, audio_latent],
            {
                'img_recon': img_recon_target,
                'aud_recon': aud_recon_target,
                'joint_rep': joint_rep_target,
                'attention': attention_target
            },
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )
        
        return history
    
    def process(self, image_latent, audio_latent):
        return self.model.predict([image_latent, audio_latent])
    
    def extract_joint_representation(self, image_latent, audio_latent):
        _, _, joint_rep, _ = self.process(image_latent, audio_latent)
        return joint_rep
    
    def get_attention_weights(self, image_latent, audio_latent):
        _, _, _, attention = self.process(image_latent, audio_latent)
        return attention
    
    def cross_modal_reconstruction(self, image_latent, audio_latent):
        img_recon, aud_recon, _, _ = self.process(image_latent, audio_latent)
        return img_recon, aud_recon
    
    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save(os.path.join(path, 'brain_model.h5'))
    
    def load_model(self, path):
        # Define custom objects to handle the loss function
        custom_objects = {
            'mse': tf.keras.losses.MeanSquaredError(),
            'categorical_crossentropy': tf.keras.losses.CategoricalCrossentropy()
        }
        self.model = tf.keras.models.load_model(
            os.path.join(path, 'brain_model.h5'),
            custom_objects=custom_objects
        )