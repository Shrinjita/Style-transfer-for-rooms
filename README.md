### Contributors: 
- Shrinjita Paul
- Dharanya C
- Ancy B John
- Akshayaharshini
  
## Project Overview
This project demonstrates two main functionalities:
1. **Loading and Displaying Images**: The code streams images from a dataset and displays them. It includes methods to load images from a directory, split the dataset into training and testing sets, and visualize random samples.
2. **Style Transfer Autoencoder**: A PyTorch-based autoencoder designed for photorealistic style transfer. The model uses pre-trained VGG-19 as an encoder and a custom decoder to reconstruct images. The model also implements bottleneck feature aggregation and ZCA whitening for enhanced style transfer.

---

## Features

### 1. **Loading and Displaying Images**:
- **Streaming Dataset**: Loads images using `datasets.load_dataset()` without downloading the entire dataset at once, avoiding memory overload.
- **Split Folders**: The dataset is split into 80% training and 20% testing using `splitfolders.ratio()`.
- **Displaying Images**: Images from the training dataset are visualized using `matplotlib` and `PIL`.

### 2. **Style Transfer Autoencoder**:
- **VGG-19 Based Encoder**: Pre-trained VGG-19 is used to extract features from the input images.
- **Bottleneck Feature Aggregation**: Multi-scale features are aggregated in the bottleneck layer to enhance style transfer.
- **Symmetric Decoder**: A custom decoder reconstructs the styled image from the bottlenecked features.
- **ZCA Whitening**: This ensures style features are transferred in a more refined manner by transforming content features into the style domain.

---

## Installation and Requirements

### Dependencies:
- `datasets`
- `matplotlib`
- `numpy`
- `splitfolders`
- `torch`
- `torchvision`
- `Pillow` (PIL)

To install all the necessary packages, run:
```bash
pip install datasets matplotlib numpy splitfolders torch torchvision Pillow
```

### Dataset Setup:
1. Create an input folder named `dataset` and store images in subfolders (categories).
2. Run the script to split the dataset into training and testing sets:
   ```bash
   splitfolders.ratio('dataset', output='style images', seed=1337, ratio=(.8, .2))
   ```

---

## Running the Code

### 1. Display Random Images from the Dataset:
- Use the code to stream and display images from the `lsun-bedrooms` dataset:
```python
ds = load_dataset("pcuenq/lsun-bedrooms", split="train", streaming=True)
```
- It displays random images using `matplotlib`.

### 2. Visualizing Local Images:
- Display images from the training directory:
```python
train_dir = 'style images/train'
image_files = get_image_files(train_dir, num_images=5)
display_images(image_files)
```

### 3. Training the Style Transfer Autoencoder:
- Train the autoencoder with a custom dataset:
```python
autoencoder = StyleTransferAutoencoder()
train_model(autoencoder, dataloader, epochs=10, lr=1e-4)
```
- The model is trained on low-to-high resolution images in a block-wise manner.

---

## Customization

- Modify `num_images` in `get_image_files()` to change the number of images displayed.
- Adjust the dataset resolution in `train_model()` for low-to-high resolution training.

---

## Future Improvements
- **Perceptual Loss**: Implement a more advanced perceptual loss for better results.
- **Dataset Augmentation**: Add image augmentation techniques for better generalization.
- **Interactive GUI**: Create an interactive interface using `Streamlit` for easier model usage and visualization.
