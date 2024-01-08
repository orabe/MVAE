# Multimodal Variational Autoencoder (MVAE)

## Overview

This repository contains an implementation of the Multimodal Variational Autoencoder (MVAE), a generative model introduced in the paper "Multimodal Generative Models for Scalable Weakly-Supervised Learning" by Mike Wu and Noah Goodman. This implementation is part of a machine learning class project conducted at TU Berlin.

The project focuses on exploring the MVAE's capability to efficiently learn joint representations from multiple modalities using a variational autoencoder approach.

## Getting Started
The MVAE is implemented currently with only the MNIST dataset. You can use the already trained model in order to generate images conditioned on certain labels and vice verca.

### Prerequisites

- Python (>=3.6)
- torch (>= 2.0.1)
- torchvision (>= 0.15.2)
- numpy (>= 1.24.3)
- matplotlib (>= 3.7.1)


### Installation and usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/MVAE.git
   cd MVAE
   ```

2. Sampling data

   Pre-trained model weights are located at:

   ```bash
   src/<dataset_name>/trained_models
   ```

   Each model has a corresponding best-performing set of weights, saved as:

   ```bash
   src/<dataset_name>/trained_models/final_best_epoch.pth.tar
   ```
   
   The default model in `sample.py`uses these weights. Adjust the `condition_on_image` and `condition_on_text` variables as needed. 

   To generate new data, execute:

   ```bash
   python sample.py
   ```

3. <i>(optional)</i> Train model

   If you wish to train the model on a specific device, modify the hyperparameters in:

   ```bash
   src/<dataset_name>/utils.py
   ```

   Specify a location to store the resulting weights in:

   ```
   src/<dataset_name>/train.py
   ```

   Run the following command to initiate the training
   ```
   python train.py
   ```



### TODO