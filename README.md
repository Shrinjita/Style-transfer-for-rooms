### Contributors: 
- Shrinjita Paul
- Dharanya C
- Ancy B John
- Akshayaharshini

## Overview

This project aims to apply photorealistic style transfer to interior rooms, allowing homeowners to visualize various design styles (modern, minimalistic, contemporary, etc.) before embarking on costly remodels. The model uses an autoencoder architecture with block training, high-frequency residual skip connections, and bottleneck feature aggregation to generate realistic images that incorporate new room styles.

---

## Project Structure

```bash
|-- data/          # Contains datasets (MSCOCO, ADE20K)
|-- models/        # Autoencoder model and VGG19 feature extractor
|-- experiments/   # Results and model comparisons
|-- notebooks/     # Jupyter Notebooks for training and analysis
|-- scripts/       # Scripts to run the model and generate stylized images
|-- README.md      # This file
```

## Methodology

The project builds upon the autoencoder-based Neural Style Transfer (NST) method with several improvements:
1. **Autoencoder Architecture:** We utilize a VGG-19 based autoencoder for content and style extraction.
2. **Whitening and Coloring Transforms (WCT):** We apply WCT to transform content image features to match the covariance of the style image.
3. **Bottleneck Feature Aggregation (BFA):** Concatenates multi-scale features for better detail preservation.
4. **High-Frequency Skip Connections:** Improves the photorealism of the output by adding blockwise residual training.

## Datasets

- **MSCOCO:** Contains 118,288 images used for training and validation.
- **ADE20K:** Used for initial semantic segmentation; contains 25,574 training images.

## Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/mariemchu/PhotoNetWCT2.git
   cd PhotoNetWCT2
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the datasets:**
   - [MSCOCO](http://cocodataset.org/#download)
   - [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)

## Training the Model

To train the model using the MSCOCO dataset:

```bash
python scripts/train_autoencoder.py --dataset MSCOCO --epochs 50
```

## Running Inference

To apply style transfer on a room image:

```bash
python scripts/style_transfer.py --content path_to_content_image --style path_to_style_image --output output_image.png
```

## Key Features

- **Photorealistic Image Style Transfer:** Preserves structural details while changing the style of the room.
- **Efficient Architecture:** Incorporates feature aggregation and high-frequency residuals for improved results.
- **Dataset Compatibility:** Utilizes popular datasets like MSCOCO and ADE20K for training and validation.

## Future Work

- **GAN-Based Models:** We plan to explore Generative Adversarial Networks (GANs) to enable generation, modification, or removal of furniture styles.
- **More Training Time:** With additional computational resources, further model tuning and larger training epochs will enhance the results.

## References
