# Calculate the receptive field size 

Assume we have the following CNN structure:

          Layer type: Conv
         Kernel size: (4, 4)
        Padding size: 0
         Stride size: (1, 1)

          Layer type: Conv
         Kernel size: (3, 3)
        Padding size: 0
         Stride size: (1, 1)

          Layer type: MaxPool
         Kernel size: (3, 3)
        Padding size: 0
         Stride size: (3, 3)

The CNN structure in MyModel.py is a bad example for calculating the receptive field size.
So we take a very simple CNN structure shown above :)
The calculation steps are shown in the two pictures with a lot of colors!

**Note**: The output size of each layer is not needed for getting the receptive field size (Calc_steps_1.jpg)
These calculations were done for better understanding.

# CalcReceptiveField.py

In the script CalcReceptiveField.py calculates size of the receptive field and the corresponding output size for each layer.
Feel free to change to the dict cnn_layers considering the kernel, padding and stride size.

## Usage

```
python3 CalcReceptiveField.py
```