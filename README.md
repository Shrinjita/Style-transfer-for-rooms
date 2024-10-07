### Contributors: 
- Shrinjita Paul
- Dharanya C
- Ancy B John
- Akshayaharshini
  
# Style Transfer for Interior Rooms

This repository contains the implementation of a photorealistic style transfer method for interior rooms using a custom autoencoder architecture. The model leverages Bottleneck Feature Aggregation (BFA), high-frequency residual skip connections, and ZCA transforms to achieve high-quality style transfer, preserving the fine details of the room images.

## Features

- **Autoencoder Architecture**: Uses a VGG-19 encoder and a symmetric decoder.
- **Bottleneck Feature Aggregation (BFA)**: Enhances the style transfer by combining multi-scale features.
- **High-Frequency Residual Skip Connections**: Preserves fine details like edges and textures during style transfer.
- **ZCA Transform**: A specialized whitening and coloring transformation to match content and style covariance.
- **Blockwise Training**: Trains the model in stages to capture the coarse and fine features of the image.

## Datasets

The model is trained on the following datasets:
- **MSCOCO**: Used for initial training to minimize pixel reconstruction and feature losses.
- **ADE20K**: Used for semantic segmentation.
- Filtered subsets of interior room images were used, consisting of 6118 training images and 523 validation images.

## Model Architecture

The architecture combines:
- **VGG-19 Encoder** for extracting features.
- **Symmetric Decoder** for high-quality image reconstruction.
- **BFA** for feature aggregation.
- **Residual Skip Connections** for preserving high-frequency details.
- **ZCA Transform** for matching content features to style covariance.

## Training

- The model was trained with pixel reconstruction and feature losses.
- Hyperparameters such as the number of VGG layers concatenated in BFA and learning rate were fine-tuned.
- Blockwise training was used to refine features from coarse to fine detail.

## Future Improvements

Possible enhancements include:
- Using **Adaptive Instance Normalization (AdaIN)** to replace ZCA transform.
- Trying alternative architectures like **ResNet + DenseNet** or **EfficientNet** for improved feature extraction.
- Experimenting with additional datasets such as **Interiornet** and **IIW**.

## Usage

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/StyleTransferForRooms.git
cd StyleTransferForRooms
pip install -r requirements.txt
```

Run the Jupyter Notebook (`v1.ipynb`):

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open and run the cells in `v1.ipynb` to train the model and perform style transfer.
