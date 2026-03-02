# Generative Semantic Communication with Enhanced ToDayGAN

This repository contains the implementation of a novel framework combining the **GESCO (Generative Semantic Communication) diffusion model** with an enhanced **ToDayGAN (ComboGAN)** architecture. 

Our research focuses on improving Image-to-Image translation (Night-to-Day / Day-to-Night) under semantic communication frameworks. We introduced several structural optimizations to the baseline GAN model to achieve highly accurate, feature-aligned, and semantic-consistent image synthesis.

## ✨ Key Contributions & Modifications

Our customized `ComboGANModel` features three major architectural improvements:

1. **Center Weight Function (`보완 1-1`)**: We introduced a spatial center-weighted penalty to the Identity Loss. This preserves critical central features of the image during translation while allowing flexibility at the edges.
2. **Feature Discriminator (`보완 2-1`)**: Beyond the standard 3-way discriminator, we integrated an additional `Feature Discriminator` using MSE loss. This enforces strict feature-level alignment between the real and generated domains, significantly improving synthesis quality.
3. **Segmentation Map Conditioning (`보완 3-1, 3-2`)**: We modified the Generator's input pipeline to concatenate real images with their corresponding One-Hot Encoded Segmentation Maps (`real_A` + `seg_A`). This explicitly guides the generator to maintain structural and semantic consistency during domain translation.

## 📂 Repository Structure

* `/GESCO`: Contains the diffusion-based Generative Semantic Communication models and sampling scripts.
* `/ToDayGAN`: Contains the modified ComboGAN architecture, data loaders, and training/testing scripts.
    * `models/combogan_model.py`: Core logic containing our customized Generator, feature Discriminators, and loss functions.

## 🚀 Getting Started

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
Training the Enhanced ToDayGAN
To train the model with your custom configuration:

Bash
cd ToDayGAN
python second_revision_train.py --dataroot ./dataset/path --name my_experiment --batchSize 4 --seg_nc 35
Testing the Model
To generate translated images from the trained checkpoints:

Bash
cd ToDayGAN
python review_test.py --dataroot ./dataset/path --name my_experiment --how_many 100