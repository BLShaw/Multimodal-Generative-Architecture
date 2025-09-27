# modules/data_loader.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
import requests
import zipfile
import io

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.data_dir = config['paths']['data_dir']
        self.mnist_dir = os.path.join(self.data_dir, 'mnist')
        self.fsdd_dir = os.path.join(self.data_dir, 'fsdd')
        
        os.makedirs(self.mnist_dir, exist_ok=True)
        os.makedirs(self.fsdd_dir, exist_ok=True)
    
    def load_mnist(self):
        print("Loading MNIST dataset...")
        
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        x_train = self._preprocess_images(x_train)
        x_test = self._preprocess_images(x_test)
        
        resize_to = self.config['data']['mnist']['resize_to']
        if resize_to and resize_to != [28, 28]:
            x_train = self._resize_images(x_train, resize_to)
            x_test = self._resize_images(x_test, resize_to)
        
        return (x_train, y_train), (x_test, y_test)
    
    def load_fsdd(self):
        print("Loading FSDD dataset...")
        
        if not os.listdir(self.fsdd_dir):
            self._download_fsdd()
        
        audio_files = []
        labels = []
        
        for digit in range(10):
            digit_dir = os.path.join(self.fsdd_dir, str(digit))
            if os.path.exists(digit_dir):
                for filename in os.listdir(digit_dir):
                    if filename.endswith('.wav'):
                        filepath = os.path.join(digit_dir, filename)
                        audio, _ = librosa.load(filepath, sr=self.config['data']['fsdd']['sample_rate'])
                        audio_files.append(audio)
                        labels.append(digit)
        
        # Use the model's expected input shape instead of calculating from sample rate and duration
        target_length = self.config['model']['audio_autoencoder']['input_shape'][0]
        audio_files = self._preprocess_audio(audio_files, target_length)
        labels = np.array(labels)
        
        x_train, x_test, y_train, y_test = train_test_split(
            audio_files, labels, test_size=0.2, random_state=42
        )
        
        return (x_train, y_train), (x_test, y_test)
    
    def _preprocess_images(self, images):
        # Normalize to [0,1] range
        images = images.astype('float32') / 255.0
        images = np.expand_dims(images, axis=-1)
        return images
    
    def _resize_images(self, images, target_size):
        resized_images = np.zeros((len(images), target_size[0], target_size[1], 1))
        for i, img in enumerate(images):
            resized_images[i] = tf.image.resize(img, target_size).numpy()
        return resized_images
    
    def _preprocess_audio(self, audio_files, target_length=None):
        if target_length is None:
            target_length = int(self.config['data']['fsdd']['sample_rate'] * 
                               self.config['data']['fsdd']['duration'])
        
        processed_audio = []
        
        for audio in audio_files:
            # Pad or truncate to target length
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
            else:
                audio = audio[:target_length]
            
            # Normalize to have zero mean and unit variance
            audio = (audio - np.mean(audio)) / (np.std(audio) + 1e-8)
            
            # Scale to [-1, 1] range to help with training
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            
            processed_audio.append(audio)
        
        return np.array(processed_audio)
    
    def _download_fsdd(self):
        print("Downloading FSDD dataset...")
        url = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/master.zip"
        r = requests.get(url, timeout=30)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(self.data_dir)
        
        extracted_dir = os.path.join(self.data_dir, "free-spoken-digit-dataset-master", "recordings")
        for digit in range(10):
            os.makedirs(os.path.join(self.fsdd_dir, str(digit)), exist_ok=True)
        
        for filename in os.listdir(extracted_dir):
            if filename.endswith('.wav'):
                digit = filename.split('_')[0]
                src = os.path.join(extracted_dir, filename)
                dst = os.path.join(self.fsdd_dir, digit, filename)
                os.rename(src, dst)