# modules/image_autoencoder.py
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from utils import configure_gpu

# Configure GPU before creating models
configure_gpu()

class ImageAutoencoder:
    def __init__(self, config):
        self.config = config
        self.model_config = config['model']['image_autoencoder']
        self.model = self._build_model()
        self.encoder = self._extract_encoder()
        self.decoder = self._build_decoder()
    
    def _build_model(self):
        input_shape = self.model_config['input_shape']
        latent_dim = self.model_config['latent_dim']
        
        # Encoder with convolutional layers
        inputs = layers.Input(shape=input_shape, name='encoder_input')
        
        # Convolutional layers
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Flatten and dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        latent = layers.Dense(latent_dim, activation='relu', name='latent_layer')(x)
        
        # Decoder
        x = layers.Dense(256, activation='relu')(latent)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Calculate the shape after the last conv layer in the encoder
        # For 28x28 input, after 3 max pooling layers with stride 2, we get 4x4x128
        x = layers.Dense(4 * 4 * 128)(x)
        x = layers.Reshape((4, 4, 128))(x)
        
        # Use upsampling instead of transposed convolutions for better control over output size
        x = layers.UpSampling2D((2, 2))(x)  # 4x4 -> 8x8
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.UpSampling2D((2, 2))(x)  # 8x8 -> 16x16
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.UpSampling2D((2, 2))(x)  # 16x16 -> 32x32
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Crop to get back to 28x28
        x = layers.Cropping2D(((2, 2), (2, 2)))(x)  # 32x32 -> 28x28
        
        outputs = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        
        # Complete model
        autoencoder = models.Model(inputs, outputs, name='image_autoencoder')
        autoencoder.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.model_config['learning_rate']),
            loss='mse'
        )
        
        return autoencoder
    
    def _extract_encoder(self):
        # Create encoder model by connecting input to latent layer
        encoder_input = self.model.input
        encoder_output = self.model.get_layer('latent_layer').output
        return models.Model(encoder_input, encoder_output, name='image_encoder')
    
    def _build_decoder(self):
        # Build decoder from scratch
        latent_input = layers.Input(shape=(self.model_config['latent_dim'],), name='decoder_input')
        
        # Get the input shape
        input_shape = self.model_config['input_shape']
        
        # Dense layers
        x = layers.Dense(256, activation='relu')(latent_input)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Calculate the shape after the last conv layer in the encoder
        # For 28x28 input, after 3 max pooling layers with stride 2, we get 4x4x128
        x = layers.Dense(4 * 4 * 128)(x)
        x = layers.Reshape((4, 4, 128))(x)
        
        # Use upsampling instead of transposed convolutions for better control over output size
        x = layers.UpSampling2D((2, 2))(x)  # 4x4 -> 8x8
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.UpSampling2D((2, 2))(x)  # 8x8 -> 16x16
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.UpSampling2D((2, 2))(x)  # 16x16 -> 32x32
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Crop to get back to 28x28
        x = layers.Cropping2D(((2, 2), (2, 2)))(x)  # 32x32 -> 28x28
        
        outputs = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        
        return models.Model(latent_input, outputs, name='image_decoder')
    
    def train(self, x_train, x_test):
        # Reduce batch size to save memory
        batch_size = min(self.model_config['batch_size'], 16)
        
        # Add callbacks for better training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.00001)
        ]
        
        history = self.model.fit(
            x_train, x_train,
            epochs=self.model_config['epochs'],
            batch_size=batch_size,
            shuffle=True,
            validation_data=(x_test, x_test),
            callbacks=callbacks
        )
        return history
    
    def encode(self, data):
        return self.encoder.predict(data, batch_size=16)
    
    def decode(self, latent):
        return self.decoder.predict(latent, batch_size=16)
    
    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save(os.path.join(path, 'image_autoencoder.h5'))
        self.encoder.save(os.path.join(path, 'image_encoder.h5'))
        self.decoder.save(os.path.join(path, 'image_decoder.h5'))
    
    def load_model(self, path):
        # Define custom objects to handle the loss function
        custom_objects = {'mse': tf.keras.losses.MeanSquaredError()}
        self.model = tf.keras.models.load_model(
            os.path.join(path, 'image_autoencoder.h5'),
            custom_objects=custom_objects
        )
        self.encoder = tf.keras.models.load_model(os.path.join(path, 'image_encoder.h5'))
        self.decoder = tf.keras.models.load_model(os.path.join(path, 'image_decoder.h5'))