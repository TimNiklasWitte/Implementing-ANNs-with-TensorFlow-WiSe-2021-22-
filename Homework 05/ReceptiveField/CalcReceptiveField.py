import math;

def main():

    # (kernel size, padding size, stride size)
    # -> SQUARE SHAPE OF KERNEL AND STRIDE <- !!!
    cnn_layers = {
        'conv1': [4, 0, 1], # kernel = (3,3) , padding = 0 , stride = (1,1)
        'conv2': [3, 0, 1],
        'pool1': [3, 0, 3],
    }

    # Calc layer output size
    # Not needed for receptive field size calcucation but it's nice to know ;)
    previous_layer_output_size = 28
    for layer_name in cnn_layers:
        kernel_size = cnn_layers[layer_name][0]
        padding_size = cnn_layers[layer_name][1]
        stride_size = cnn_layers[layer_name][2]

        # Formula: see 1st picture in GitHub
        previous_layer_output_size = ( (previous_layer_output_size - kernel_size + 2*padding_size) / stride_size) + 1
        output_size = int(previous_layer_output_size)

        cnn_layers[layer_name].append(output_size)

    # Calc receptive field size
    receptiveField_size = 1
    for layer_name in reversed(list(cnn_layers)):
        kernel_size = cnn_layers[layer_name][0]
        padding_size = cnn_layers[layer_name][1]
        stride_size = cnn_layers[layer_name][2]

        # Formula: see 2st picture in GitHub
        receptiveField_size = kernel_size + (receptiveField_size - 1) * stride_size
        cnn_layers[layer_name].append(receptiveField_size)

    # Print results
    for layer_name in cnn_layers:
        kernel_size = cnn_layers[layer_name][0]
        padding_size = cnn_layers[layer_name][1]
        stride_size = cnn_layers[layer_name][2]
        output_size = cnn_layers[layer_name][3]
        receptiveField_size = cnn_layers[layer_name][4]

        print(f"          Layer name: {layer_name}")
        print(f"         Kernel size: ({kernel_size}, {kernel_size})")
        print(f"         Stride size: ({stride_size}, {stride_size})")
        print()
        print(f"         Output size: ({output_size}, {output_size})")
        print(f"Receptive field size: ({receptiveField_size}, {receptiveField_size})")
        print("------------------------------")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
