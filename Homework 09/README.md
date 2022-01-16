# IANNWTF - Homework 09

A GAN and a WGAN are implemented for generated drawn candle images based on the quickdraw dataset.


## Usage

### Start training

Run the `Training.py`. Before starting the training, the training data (including testing data) will be downloaded 
and stored/saved in the `candle.npy` file (~100MB).
After training, a `GeneratedCandles.png` file will be created displaying 5 generated candle images per epoch.

```bash
python Training.py
```

### See live results of training

Run `tensorboard` to inspect current loss of the generator and discriminator (WGAN: critic) in the epoch including previous onces.
Furthermore, you can see the images created by the generator per epoch.
In case of the GAN you can see the discriminator accuracy detecting fake and real images. 

```bash
tensorboard --logdir ./test_logs/
```

## Results

### GAN

![alt text](./WGAN/GeneratedCandles.png)

### WGAN

![alt text](./GAN/GeneratedCandles.png)

## Tricks and take home messages
- The generator must be strong than the discriminator (critic).
- Sometimes train generator more than discriminator (critic).
- No bias for generator