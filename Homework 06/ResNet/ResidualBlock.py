import tensorflow as tf


class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, n_filters=64, out_filters=256, mode="normal"):

        """
         ==> Conv2D ==> Conv2D ==> Conv2D ==> +
        |                                     |
        |_____________________________________|
              this path is called shortcut (aka skipped connection)

        ==> = BatchNormalization and Activation
        + = add output of 3th ConvLayer with input of 1st ConvLayer (before BatchNormalization and Activation are applied)
        
        Args:

        n_filters (int) : number of filters applied in the 1st and 2nd ConvLayer

        out_filters (int) : number of filters used in the 3th ConvLayer

        mode (str) : Either "normal", "strided" or "constant".
            normal: Not change size of feature maps, change number of channels (out_filters)
                    ConvLayer is applied to shortcut -> adjust shape of shortcut -> add required two identical shapes (number of feature maps!!)

            strided: Not change number of feature maps, change size (reduce) number of feature maps
                    normal path: (2nd Conv2D) strides=(2,2)
                    shortcut:  MaxPool2D strides=(2,2)
            
            constant: Keep size and number of channels constant
        """

        super(ResidualBlock, self).__init__()

        self.mode = mode
        self.out_filters = out_filters

        self.layer_list = [

            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(1, 1)),

            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu)

        ]

        if self.mode == "normal":

            self.layer_list += [
                tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(3, 3), padding="same"),
            ]

            self.shortcut = tf.keras.layers.Conv2D(filters=self.out_filters, kernel_size=(1, 1))

        elif self.mode == "strided":

            # self.out_filters =

            self.layer_list += [
                tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(3, 3), padding="same", strides=(2, 2))
            ]

            self.shortcut = tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=(2, 2))

        elif mode == "constant":

            self.layer_list += [
                tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(3, 3), padding="same")
            ]

        self.layer_list += [
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv2D(filters=out_filters, kernel_size=(1, 1))
        ]

    @tf.function
    def call(self, x: tf.Tensor, train: bool) -> tf.Tensor:

        """
        Propagate the input towards all layers

        Args:
            x input
            train flag set if we train
        """

        input = x

        if self.mode == "normal":
            input = self.shortcut(input)

        elif self.mode == "strided":
            self.out_filters = input.shape[-1]
            input = self.shortcut(input)

        # Propagte input towards all layers
        for layer in self.layer_list:

            # Replay train parameter to BatchNormalization layers during call
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                x = layer(x, training=train)
            else:
                x = layer(x)

        return tf.keras.layers.Add()([x, input])
