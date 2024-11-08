# Masked Image Style Transfer using Neural Networks and YOLO

> Transform your images with artistic styles using selective masking powered by Neural Networks and YOLO.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Modes](#modes)
- [Examples](#examples)
- [Implementation Details](#implementation-details)
- [References](#references)

---

## Overview
This project performs **style transfer** on images, allowing selective styling of certain parts, such as people, clothing, or specific regions, using **YOLO for segmentation** and **Neural Networks** for style synthesis. The model leverages the YOLO segmentation model to apply styles with precision and flexibility across different image parts.

## Features
- **Full Image Style Transfer**: Apply the style to the entire image.
- **Selective Transfer**: Apply styles selectively to the background, person, or specific clothing regions.
- **Multiple Modes**: Choose from five modes (`full`, `person`, `fg`, `upper`, `lower`) for different masking options.
- **Optimized Performance**: Uses VGG19 and YOLO models, with efficient loss computations for content and style preservation.

## Installation
To install the required packages and set up the project:
1. Clone this repository:
   ```bash
   git clone https://github.com/saubhagyapandey27/masked_style_transfer.git
   cd masked_style_transfer
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Ensure you have a suitable environment for TensorFlow, YOLO, OpenCV, and other dependencies.

## Usage
To run the style transfer, use the following command:
```bash
python style_transfer.py <content_image> <style_image> <mode> [--num_iter <iterations>] [--save_seg]
```

### Arguments
- **content_image**: Name of the content image file in the `inputs` folder.
- **style_image**: Name of the style image file in the `inputs` folder.
- **mode**: Mode of style transfer. Options: `full`, `person`, `fg`, `upper`, `lower`.
- **--num_iter** *(optional)*: Number of iterations for the style transfer (default: 300).
- **--save_seg** *(optional)*: Flag to save segmentation results.

Example:
```bash
python style_transfer.py image.jpg style.jpg person --num_iter 1000 --save_seg
```

## Modes
This project offers five distinct modes for flexible style transfer.  
- **Full**: Style is applied to the entire image.
- **Person**: Style is applied to the image leaving the parts where humans are present.
- **Foreground (fg)**: Style is applied to the background, excluding the foreground .
- **Upper**: Style is applied to the upper part of clothing.
- **Lower**: Style is applied to the lower part of clothing.

Each mode leverages segmentation models to ensure precise mask application for localized style transfer.

## Examples
Here are some examples showcasing the various modes.

### Full Image Style Transfer ("full" mode)
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/a7b51773-6d44-49c0-94a2-aea7ede58328" width="250" height="250"></td>
    <td style="font-size: 36px; text-align: center; vertical-align: middle;">+</td>
    <td><img src="https://github.com/user-attachments/assets/aafbd691-dc36-4f2d-9c4d-ca0b5b03d7ec" width="250" height="250"></td>
    <td style="font-size: 36px; text-align: center; vertical-align: middle;">=</td>
    <td><img src="https://github.com/user-attachments/assets/e7851abc-b0b4-443c-b466-ea5125b8bdbe" width="250" height="250"></td>
  </tr>
</table>

### Clothing Style Transfer (using "upper" mode)
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/ed4278e1-8ed1-44dc-a8cf-7c0f1cb23364" width="250" height="250"></td>
    <td style="font-size: 36px; text-align: center; vertical-align: middle;">+</td>
    <td><img src="https://github.com/user-attachments/assets/5ff49770-6d4f-476c-9949-d191f9788bc2" width="250" height="250"></td>
    <td style="font-size: 36px; text-align: center; vertical-align: middle;">=</td>
    <td><img src="https://github.com/user-attachments/assets/ae8de79b-1e05-4e26-ae72-d7f17fd2e1d3" width="250" height="250"></td>
  </tr>
</table>

### Person-Background Style Transfer ("person" mode)
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/56e64cd9-49ed-404b-b00e-fe934602137c" width="250" height="250"></td>
    <td style="font-size: 36px; text-align: center; vertical-align: middle;">+</td>
    <td><img src="https://github.com/user-attachments/assets/3583cda2-184f-44a3-954f-ae81b34d19fb" width="250" height="250"></td>
    <td style="font-size: 36px; text-align: center; vertical-align: middle;">=</td>
    <td><img src="https://github.com/user-attachments/assets/330a7efc-d5cf-42b8-a917-354a9fb3ac4a" width="250" height="250"></td>
  </tr>
</table>

### Background Style Transfer ("fg" mode)
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/0ce7dd80-be26-4493-a476-2dee0c40d430" width="250" height="250"></td>
    <td style="font-size: 36px; text-align: center; vertical-align: middle;">+</td>
    <td><img src="https://github.com/user-attachments/assets/5ff49770-6d4f-476c-9949-d191f9788bc2" width="250" height="250"></td>
    <td style="font-size: 36px; text-align: center; vertical-align: middle;">=</td>
    <td><img src="https://github.com/user-attachments/assets/b8ef8366-4c85-4a6e-ae90-6d8148f4995d" width="250" height="250"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/37e11da5-5075-432c-9b7b-422f9575c603" width="250" height="250"></td>
    <td style="font-size: 36px; text-align: center; vertical-align: middle;">+</td>
    <td><img src="https://github.com/user-attachments/assets/26fe9ef2-5450-430e-b430-d7ddbc2e6e00" width="250" height="250"></td>
    <td style="font-size: 36px; text-align: center; vertical-align: middle;">=</td>
    <td><img src="https://github.com/user-attachments/assets/0e1834c4-a3d2-40c8-a61b-72e00dbe86d5" width="250" height="250"></td>
  </tr>
</table>

## Implementation Details
The project utilizes the **VGG19** model for feature extraction and **YOLO segmentation models** for generating masks. Here’s a breakdown:

1. **Content & Style Feature Extraction**: Uses specific layers of the VGG19 model to compute content and style features.
2. **Masking**: YOLO segmentation generates binary masks to isolate selected regions.
3. **Combination**: Combines styled and original images based on the binary mask to achieve the selective style transfer.

### Directory Structure
- **inputs/**: Store content and style images here.
- **outputs/**: Resulting images with applied styles will be saved here.

## References
- For Style Transfer Part Implemented the paper [Image Style Transfer Using Convolutional Neural Networks, Gatys et al](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
- Took help from this [Github Repository](https://github.com/superb20/Image-Style-Transfer-Using-Convolutional-Neural-Networks?tab=readme-ov-file) for assistance in coding implementation of style transfer algorithm.
- In Segmentation part, took help from [YOLO v11 documentation](https://github.com/ultralytics/ultralytics) for training on custom dataset to adapt segmentation on Clothing.
---
