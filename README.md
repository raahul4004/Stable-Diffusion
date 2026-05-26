# Stable Diffusion

A Python implementation of Stable Diffusion, a latent text-to-image diffusion model capable of generating photo-realistic images from text prompts.

## Overview

This repository contains an implementation of the Stable Diffusion model, including custom attention mechanisms and supporting utilities for image generation using diffusion-based techniques.

## Repository Structure

```
Stable-Diffusion/
├── sd/                 # Core Stable Diffusion module
├── attention.py        # Custom attention mechanism implementation
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Features

- **Text-to-Image Generation** — Generate images from natural language text prompts using latent diffusion
- **Custom Attention Mechanism** — Includes a dedicated `attention.py` module implementing attention layers used in the diffusion model
- **Modular Architecture** — Core model components organized in the `sd/` package for easy extension and experimentation

## Tech Stack

- **Python** (97.9%) — Primary language for model implementation
- **Jupyter Notebook** (2.1%) — For experimentation and demonstrations

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- NumPy
- Pillow

### Installation

```bash
git clone https://github.com/raahul4004/Stable-Diffusion.git
cd Stable-Diffusion
pip install -r requirements.txt
```

### Usage

```python
from sd import StableDiffusion

model = StableDiffusion()
image = model.generate("A beautiful sunset over the ocean")
image.save("output.png")
```

## How It Works

Stable Diffusion operates in a compressed latent space rather than pixel space, making it computationally efficient while maintaining high-quality image generation. The key components include:

- **VAE (Variational Autoencoder)** — Encodes images into latent space and decodes latents back to images
- **U-Net** — Predicts noise in the latent space, guided by text embeddings
- **Text Encoder** — Converts text prompts into embeddings that guide the diffusion process
- **Attention Layers** — Cross-attention and self-attention mechanisms that enable text-conditioned image generation

## Contributing

Contributions are welcome. Feel free to open issues or submit pull requests.

## Author

**Raahul Muthukrishnan** — [@raahul4004](https://github.com/raahul4004)

## License

This project is open source. Please check the repository for license details.

## Acknowledgments

- Based on the Stable Diffusion paper and architecture by Rombach et al.
- Inspired by the open-source AI image generation community
