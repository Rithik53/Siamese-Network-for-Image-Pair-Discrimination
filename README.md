# Siamese Network for Image Pair Discrimination

## Project Overview
This project involves the development of a Siamese neural network designed to assess the similarity between pairs of images. Leveraging TensorFlow and Keras, the network utilizes convolutional neural layers to extract features from each image in a pair, computes the euclidean distance between these feature sets, and classifies the pair as either similar or dissimilar based on this distance.

## Key Features
- **Image Preprocessing**: Load and normalize grayscale images for model input.
- **Siamese Model Architecture**: Build and compile a Siamese network with a custom distance layer to compare image features.
- **Training and Validation**: Train the model on a dataset of image pairs and validate its performance.
- **Result Visualization**: Plot training and validation loss and accuracy to evaluate the model's learning.

## Installation
To set up the project environment, ensure you have Python installed, then run the following commands to install the necessary packages:
```bash
pip install tensorflow==2.3.0 opencv-python matplotlib
```
### Prerequisites
- Python 3.8
- TensorFlow 2.3
- OpenCV
- Matplotlib
- Numpy

### Setup
Clone this repository to your local machine:
```bash
git clone https://github.com/Rithik53/Siamese-Network-for-Image-Pair-Discrimination.git
cd siamese-image-pair-discrimination
```
## Usage
To run this project, navigate to the project directory and execute the script by running:
```bash
python siamese_network.py
```
## Example Output

![Example Output](https://github.com/Rithik53/Siamese-Network-for-Image-Pair-Discrimination/blob/main/output/plot.png)