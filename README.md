# Data Preparation Repository

This repository contains scripts for data preparation and augmentation, primarily focusing on image data. Here is a brief description of the files in the repository:

## Files Description

### 1. `data_prepration.py`

This Python script contains code for preparing image data for machine learning models. It includes functionalities such as:

- Reading and converting images to arrays.
- Resizing images to a specified size (256x256 in this case).
- Normalizing image data.
- Converting images from RGB to LAB color space using both skimage and OpenCV libraries.
- Adjusting the a and b channels in the LAB color space to enhance the image data.

### 2. `data_aug.cpp`

This C++ script contains code for augmenting image data. The functionalities include:

- Applying random color changes (gamma and brightness adjustments) to the images.
- Aligning the original and altered images side by side.
- Resizing images to a specified size (512x512 in this case).
- Saving the augmented images to a specified path.

The script uses the OpenCV library to manipulate images.

## Usage

For the Python script, specify the paths where your images are located in the respective variables (`path1`, `path`) in the script.

For the C++ script, use the following command to run the script:
```
./data_aug <source_image_path> <destination_image_path>
```

## Author
Milad

---

Please note that the paths in the Python script are placeholders and need to be replaced with actual paths where your images are located.
