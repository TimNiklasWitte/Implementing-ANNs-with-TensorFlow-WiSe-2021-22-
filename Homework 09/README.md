# IANNWTF - Homework 09

Neural networks based on the ResNet-architecture and on the DenseNet-architecture are trained on the
cifar10 dataset.

## Usage

Run the `Training.py`. Thereafter, a window opens containing a plot which represents the performance of the trainined models.
This plot will be saved in the `result.png` file. 

```bash
python Training.py
```

Run `tensorboard --logdir ./test_logs/` the current (live) loss of the generator and discriminator (WGAN: critic) per epoch. 
Furthermore, you can see the images created by the generator per epoch.
In case of the GAN you can see the discriminator accuracy detecting fake and real images. 