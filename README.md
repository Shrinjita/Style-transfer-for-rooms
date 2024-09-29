### Contributors: 
- Shrinjita Paul
- Dharanya C
- Ancy B John
- Akshayaharshini

### Photorealistic Style Transfer for Interior Rooms

## Overview

In this project, I implemented a photorealistic style transfer model that transforms the style of interior room images using neural networks. The model utilizes an autoencoder architecture combined with VGG-19 as the feature extractor to apply styles from one image onto another while preserving the content structure.

## Features

- I used a pre-trained VGG-19 model for style transfer.
- The project leverages the MSCOCO dataset for both content and style images.
- I visualized the styled output in real-time, allowing for immediate feedback on the transformations.

## Requirements

To run this project in Google Colab, I ensured that the following libraries are installed:

- `torch`
- `torchvision`
- `pillow`
- `matplotlib`
- `tensorflow-datasets`

These libraries are crucial for building the model and processing images.

## Cloning the Notebook

To get started, I copied my Jupyter notebook from my GitHub repository. You can do this by running the following command in a new Colab notebook:

```bash
!git clone Shrinjita/Style-transfer-for-rooms
```

## Code Description

1. **Importing Libraries**: 
   In the first section, I imported all the necessary libraries. This includes PyTorch for building the neural network, torchvision for model components, and other utilities like PIL for image processing and matplotlib for visualization.

2. **Defining the VGG Encoder**: 
   I created a class named `VGGEncoder` that uses a pre-trained VGG-19 model to extract features from input images. This allows the model to learn styles effectively by capturing high-level features.

3. **Creating the Decoder**: 
   I defined a `Decoder` class that reconstructs images from the encoded features. This step is crucial for generating the final stylized images.

4. **Style Transfer Model**: 
   The `StyleTransferModel` class combines the encoder and decoder. It implements the forward pass, where the content and style images are processed, and the whitening and coloring transform (WCT) is applied to combine features appropriately.

5. **Image Preprocessing**: 
   I included functions to preprocess images, resizing and normalizing them for input into the model. This step ensures that the images are in the right format for processing.

6. **Loading the MSCOCO Dataset**: 
   The project fetches images directly from the MSCOCO dataset. I wrote code to automatically select one image as the content and another as the style, eliminating the need for manual uploads.

7. **Training the Model**: 
   The training function executes the style transfer process. During training, I optimized the model by minimizing the reconstruction loss between the stylized output and the content image.

8. **Visualizing Results**: 
   Finally, I added functionality to visualize the output images, displaying the original content image, the style image, and the resulting stylized image.

## Conclusion

Through this project, I demonstrated how to apply neural style transfer to interior room images using deep learning techniques. By leveraging the MSCOCO dataset, I accessed a diverse set of images for both content and style, resulting in impressive transformations. I also experimented with various parameters to enhance the results further.

## Notes for Users

- I recommend setting the Colab runtime to GPU for optimal performance. This can be done by navigating to **Runtime > Change runtime type > Hardware accelerator > T4 GPU**.
- Users can also modify the training parameters, such as the number of epochs or learning rates, to explore different configurations and achieve better results.
