# Multimodal Generative Architecture

## Overview

This project implements a sophisticated system that processes both image (MNIST) and audio (FSDD) data using separate autoencoders, performs clustering on the latent representations, and analyzes relationships between modalities. The architecture is fully decoupled, allowing for independent processing of each modality while still enabling joint analysis.

## Features

- **Separate Processing Modules**: Independent modules for image and audio data processing
- **Unsupervised Learning**: No labeled data required for training
- **Advanced Encoding**: Autoencoder architectures for both image and audio data
- **Multimodal Clustering**: Clustering techniques for individual and joint modalities
- **Relationship Analysis**: Establishes one-to-one and one-to-many relationships between modalities
- **Convergence/Divergence Analysis**: Analyzes zones of similarity and difference between modalities
- **Visualization**: Comprehensive visualization of results including reconstructions, latent spaces, and clustering

## Architecture

The architecture consists of the following main components:

1. **Data Preprocessing and Encoding**
   - MNIST image dataset processing
   - FSDD audio dataset processing
   - Robust encoding generators for both datasets
   - Autoencoder architectures for each data type

2. **Clustering and Analysis Modules**
   - Visual clustering mechanism
   - One-to-one and one-to-many relationship tables
   - Brain module for intelligent processing
   - Config controller for dynamic parameter management
   - Convergence and divergence zone analysis

3. **Visualization**
   - Image and audio reconstruction visualizations
   - Latent space visualizations using t-SNE
   - Clustering result visualizations
   - Relationship heatmaps
   - Convergence and divergence analysis plots

```mermaid
flowchart TD
    A[Raw Data] --> B{Data Preprocessing}
    B --> C["Image Data (MNIST)"]
    B --> D["Audio Data (FSDD)"]
    C --> E[Image Autoencoder]
    D --> F[Audio Autoencoder]
    E --> G[Image Latent Space]
    F --> H[Audio Latent Space]
    G --> I[Multimodal Clustering]
    H --> I
    I --> J[Relationship Analysis]
    J --> K[Convergence/Divergence Analysis]
    K --> L[Visualization Module]
    L --> M[Outputs]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style M fill:#bbf,stroke:#333,stroke-width:2px
```

## Directory Structure

```
Multimodal-Generative-Architecture/
├── config/
│   └── config.json
├── data/
│   ├── mnist/
│   └── fsdd/
├── modules/
│   ├── __init__.py
│   ├── brain_module.py
│   ├── config_controller.py
│   ├── data_loader.py
│   ├── image_autoencoder.py
│   ├── audio_autoencoder.py
│   ├── multimodal_clustering.py
│   ├── utils.py
│   └── visualization.py
├── outputs/
│   ├── models/
│   │   ├── image_autoencoder/
│   │   └── audio_autoencoder/
│   ├── clusters/
│   ├── synthetic/
│   └── plots/
├── main.py
├── train.py
├── evaluate.py
└── visualize.py
```
## Full Architecture Diagram

```mermaid
flowchart TD
%% Global entities
ConfigFile["Configuration File"]:::config
Train["Training Workflow"]:::workflow
Eval["Evaluation Workflow"]:::workflow
Visual["Visualization Workflow"]:::workflow
%% Input Layer
subgraph "Input Layer"
DataLoader["Data Loader"]:::input
MNIST["MNIST Images"]:::data
FSDD["FSDD Audio"]:::data
end
%% Processing Layer
subgraph "Processing Layer"
ImgAE["Image Autoencoder"]:::image
AudioAE["Audio Autoencoder"]:::audio
ImgLatent["Image Latent Space"]:::latent
AudioLatent["Audio Latent Space"]:::latent
end
%% Analysis Layer
subgraph "Analysis Layer"
MultiCluster["Multimodal Clustering"]:::analysis
Brain["Brain Module"]:::analysis
RelAnalysis["Relationship Analysis"]:::analysis
ConvDiv["Convergence/Divergence"]:::analysis
end
%% Output Layer
subgraph "Output Layer"
Visualize["Visualization"]:::output
Models["Model Outputs"]:::output
Plots["Plots"]:::output
Synth["Synthetic Data"]:::output
end
%% Data Flow
MNIST -->|"image data"| DataLoader
FSDD -->|"audio data"| DataLoader
DataLoader -->|"images"| ImgAE
DataLoader -->|"audio"| AudioAE
ImgAE -->|"latent rep"| ImgLatent
AudioAE -->|"latent rep"| AudioLatent
ImgLatent --> MultiCluster
AudioLatent --> MultiCluster
MultiCluster --> Brain
Brain --> RelAnalysis
RelAnalysis --> ConvDiv
ConvDiv --> Visualize
Visualize --> Models
Visualize --> Plots
Visualize --> Synth
%% Configuration Flow
ConfigFile -->|"config"| DataLoader
ConfigFile -->|"config"| ImgAE
ConfigFile -->|"config"| AudioAE
ConfigFile -->|"config"| MultiCluster
ConfigFile -->|"config"| Brain
ConfigFile -->|"config"| Visualize
%% Workflow Connections
Train --> DataLoader
Eval --> MultiCluster
Visual --> Visualize
%% Click Events
click DataLoader "https://github.com/blshaw/multimodal-generative-architecture/blob/main/modules/data_loader.py"
click ImgAE "https://github.com/blshaw/multimodal-generative-architecture/blob/main/modules/image_autoencoder.py"
click AudioAE "https://github.com/blshaw/multimodal-generative-architecture/blob/main/modules/audio_autoencoder.py"
click MultiCluster "https://github.com/blshaw/multimodal-generative-architecture/blob/main/modules/multimodal_clustering.py"
click Brain "https://github.com/blshaw/multimodal-generative-architecture/blob/main/modules/brain_module.py"
click ConfigFile "https://github.com/blshaw/multimodal-generative-architecture/blob/main/config/config.json"
click Visualize "https://github.com/blshaw/multimodal-generative-architecture/blob/main/modules/visualization.py"
click Train "https://github.com/blshaw/multimodal-generative-architecture/blob/main/train.py"
click Eval "https://github.com/blshaw/multimodal-generative-architecture/blob/main/evaluate.py"
click Visual "https://github.com/blshaw/multimodal-generative-architecture/blob/main/visualize.py"
%% Styles
classDef input fill:#D5E8D4,stroke:#82B366,color:#000
classDef data fill:#F8CECC,stroke:#B85450,color:#000
classDef image fill:#DAE8FC,stroke:#6C8EBF,color:#000
classDef audio fill:#E1D5E7,stroke:#9673A6,color:#000
classDef latent fill:#FFF2CC,stroke:#D6B656,color:#000
classDef analysis fill:#F5F5F5,stroke:#666666,color:#000
classDef output fill:#D4EDF7,stroke:#6DA9CF,color:#000
classDef config fill:#FCE4D6,stroke:#D79B00,color:#000
classDef workflow fill:#E2F0D9,stroke:#70AD47,color:#000
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/BLShaw/Multimodal-Generative-Architecture.git
cd Multimodal-Generative-Architecture
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

The project is designed to be run in three separate stages:

### 1. Training

Train the autoencoders and brain module:

```bash
python train.py --config config/config.json
```

This will:

- Download and preprocess the MNIST and FSDD datasets
- Train the image and audio autoencoders
- Train the brain module for multimodal processing
- Save all trained models and latent representations

### 2. Evaluation

Evaluate the trained models and perform clustering:

```bash
python evaluate.py --config config/config.json
```

This will:

- Load the trained models
- Perform clustering on individual modalities
- Perform joint clustering on multimodal data
- Establish relationships between modalities
- Analyze convergence and divergence zones
- Save clustering results and metrics

### 3. Visualization

Generate visualizations of the results:

```bash
python visualize.py --config config/config.json
```

This will:

- Load the trained models and results
- Generate reconstruction visualizations
- Visualize latent spaces using t-SNE
- Plot clustering results
- Visualize attention weights
- Generate synthetic samples

## Configuration

The system behavior can be customized through the `config/config.json` file:

```json
{
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
      "normalize": true
    },
    "fsdd": {
      "sample_rate": 8000,
      "duration": 1.0,
      "normalize": true
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
```

## Results

The system generates several outputs:

1. **Trained Models**: Saved in `outputs/models/`
2. **Clustering Results**: Saved in `outputs/clusters/`
3. **Visualizations**: Saved in `outputs/plots/`
4. **Synthetic Samples**: Saved in `outputs/synthetic/`

Key metrics include:

- Silhouette scores for clustering quality
- Reconstruction losses for autoencoder performance
- Relationship matrices between modalities
- Convergence and divergence zone analysis

## Dependencies

- TensorFlow
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Librosa
- Soundfile
- Requests

See `requirements.txt` for the complete list.

## Contributing

1. Fork the repository
2. Create a feature branch:

```bash
git checkout -b feature/new-feature
```

3. Commit your changes:

```bash
git commit -am 'Add some new feature'
```

4. Push to the branch:

```bash
git push origin feature/new-feature
```

5. Open a Pull Request


## Acknowledgments

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) - Yann LeCun
- [FSDD Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset) - Jakobovski
- TensorFlow Team
- Keras Team
