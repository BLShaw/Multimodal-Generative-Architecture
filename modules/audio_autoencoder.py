# modules/audio_autoencoder.py
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from utils import configure_gpu

# Configure GPU before creating models
configure_gpu()

class AudioAutoencoder:
    def __init__(self, config):
        self.config = config
        self.model_config = config['model']['audio_autoencoder']
        self.model = self._build_model()
        self.encoder = self._extract_encoder()
        self.decoder = self._build_decoder()
    
    def _build_model(self):
        input_shape = self.model_config['input_shape']
        latent_dim = self.model_config['latent_dim']
        
        # Encoder
        inputs = layers.Input(shape=input_shape, name='encoder_input')
        
        # Reshape for 1D convolutions
        x = layers.Reshape((input_shape[0], 1))(inputs)
        
        # 1D Convolutional layers with reduced complexity
        x = layers.Conv1D(32, 5, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Conv1D(128, 5, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Flatten()(x)
        
        # Dense layers with reduced size
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        latent = layers.Dense(latent_dim, activation='relu', name='latent_layer')(x)
        
        # Decoder
        x = layers.Dense(256, activation='relu')(latent)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Calculate the reshape dimensions based on the input shape
        # After 3 max pooling layers with stride 2, the sequence length is reduced by factor of 8
        reduced_length = input_shape[0] // 8
        x = layers.Dense(reduced_length * 128)(x)
        x = layers.Reshape((reduced_length, 128))(x)
        
        # Transposed convolutions
        x = layers.Conv1DTranspose(128, 5, strides=2, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Conv1DTranspose(64, 5, strides=2, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Conv1DTranspose(32, 5, strides=2, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Conv1DTranspose(1, 5, activation='linear', padding='same')(x)
        
        outputs = layers.Flatten()(x)
        
        # Complete model
        autoencoder = models.Model(inputs, outputs, name='audio_autoencoder')
        autoencoder.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.model_config['learning_rate']),
            loss='mse'
        )
        
        return autoencoder
    
    def _extract_encoder(self):
        # Create encoder model by connecting input to latent layer
        encoder_input = self.model.input
        encoder_output = self.model.get_layer('latent_layer').output
        return models.Model(encoder_input, encoder_output, name='audio_encoder')
    
    def _build_decoder(self):
        # Build decoder from scratch
        latent_input = layers.Input(shape=(self.model_config['latent_dim'],), name='decoder_input')
        
        # Get the input shape
        input_shape = self.model_config['input_shape']
        
        # Dense layers
        x = layers.Dense(256, activation='relu')(latent_input)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Calculate the reshape dimensions based on the input shape
        # After 3 max pooling layers with stride 2, the sequence length is reduced by factor of 8
        reduced_length = input_shape[0] // 8
        x = layers.Dense(reduced_length * 128)(x)
        x = layers.Reshape((reduced_length, 128))(x)
        
        # Transposed convolutions
        x = layers.Conv1DTranspose(128, 5, strides=2, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Conv1DTranspose(64, 5, strides=2, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Conv1DTranspose(32, 5, strides=2, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Conv1DTranspose(1, 5, activation='linear', padding='same')(x)
        
        outputs = layers.Flatten()(x)
        
        return models.Model(latent_input, outputs, name='audio_decoder')
    
    def train(self, x_train, x_test):
        # Reduce batch size to save memory
        batch_size = min(self.model_config['batch_size'], 8)
        
        # Add callbacks for better training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.00001)
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
        return self.encoder.predict(data, batch_size=8)
    
    def decode(self, latent):
        return self.decoder.predict(latent, batch_size=8)
    
    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save(os.path.join(path, 'audio_autoencoder.h5'))
        self.encoder.save(os.path.join(path, 'audio_encoder.h5'))
        self.decoder.save(os.path.join(path, 'audio_decoder.h5'))
    
    def load_model(self, path):
        # Define custom objects to handle the loss function
        custom_objects = {'mse': tf.keras.losses.MeanSquaredError()}
        self.model = tf.keras.models.load_model(
            os.path.join(path, 'audio_autoencoder.h5'),
            custom_objects=custom_objects
        )
        self.encoder = tf.keras.models.load_model(os.path.join(path, 'audio_encoder.h5'))
        self.decoder = tf.keras.models.load_model(os.path.join(path, 'audio_decoder.h5'))