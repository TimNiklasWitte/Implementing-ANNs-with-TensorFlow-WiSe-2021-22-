# IANNWTF - Homework 08

## Task

An autoencoder is developed which is able to denoise MNIST data.

## Usage

Run the `Training.py`. 

```bash
python3 Training.py
```

Thereafter, for each embedding size definied in `Training.py` a autoencoder is trainied for the task mentioned above.
For each trained autoencoder are 5 of each digit examplarently plotted (orginal, noised and denoised by the autoencoder). 
The plots are saved into `/ConvolutionalAutoencoder/Plots/Denoising`.
Besides, the corresponding loss for each embedding size will be plotted.
This plot will be saved in the `/ConvolutionalAutoencoder/Plots/Loss.png` file. 
Furthermore, for each defined embedding size, the corresponding embedding of 1000 digits are plotted: The dimension is reduced to two by using the t-SNE algorithm.
The resulting plot is saved in `/ConvolutionalAutoencoder/Plots/LatentSpace.png`.