# modules/config_controller.py
import json
import os

class ConfigController:
    def __init__(self, config_path='config/config.json'):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self):
        if not os.path.exists(self.config_path):
            self._create_default_config()
            return self.config
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config
        except json.JSONDecodeError:
            self._create_default_config()
            return self.config
    
    def _create_default_config(self):
        default_config = {
            "model": {
                "image_autoencoder": {
                    "input_shape": [28, 28, 1],
                    "encoder_layers": [128, 64, 32],
                    "latent_dim": 16,
                    "decoder_layers": [32, 64, 128],
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "epochs": 50
                },
                "audio_autoencoder": {
                    "input_shape": [8192],
                    "encoder_layers": [1024, 512, 256],
                    "latent_dim": 32,
                    "decoder_layers": [256, 512, 1024],
                    "learning_rate": 0.001,
                    "batch_size": 16,
                    "epochs": 50
                },
                "clustering": {
                    "n_clusters": 10,
                    "algorithm": "kmeans",
                    "convergence_threshold": 0.001
                }
            },
            "data": {
                "mnist": {
                    "resize_to": [28, 28],
                    "normalize": True
                },
                "fsdd": {
                    "sample_rate": 8000,
                    "duration": 1.0,
                    "normalize": True
                }
            },
            "paths": {
                "data_dir": "data",
                "output_dir": "outputs",
                "model_dir": "outputs/models",
                "cluster_dir": "outputs/clusters",
                "synthetic_dir": "outputs/synthetic",
                "plot_dir": "outputs/plots"
            }
        }
        
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        self.config = default_config
        return default_config
    
    def save_config(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key, value):
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update_model_hyperparameters(self, model_type, params):
        if model_type not in self.config['model']:
            self.config['model'][model_type] = {}
        
        for param, value in params.items():
            self.config['model'][model_type][param] = value
        
        self.save_config()
    
    def update_data_parameters(self, data_type, params):
        if data_type not in self.config['data']:
            self.config['data'][data_type] = {}
        
        for param, value in params.items():
            self.config['data'][data_type][param] = value
        
        self.save_config()
    
    def update_paths(self, paths):
        for path, value in paths.items():
            self.config['paths'][path] = value
        
        self.save_config()
    
    def get_model_config(self, model_type):
        return self.config['model'].get(model_type, {})
    
    def get_data_config(self, data_type):
        return self.config['data'].get(data_type, {})
    
    def get_all_paths(self):
        return self.config['paths']