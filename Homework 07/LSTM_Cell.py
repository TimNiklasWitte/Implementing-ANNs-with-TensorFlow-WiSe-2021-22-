import tensorflow as tf

class LSTM_Cell(tf.keras.layers.Layer): # <-- Needed to make parameters trainable and to be callable

    """
    LSTM cell
    Calc on single time steps
    """

    # units = hidden size = size of hidden state vector 
    def __init__(self, units):
        super(LSTM_Cell, self).__init__()
        self.units = units

        # Forget Gate
        self.dense_layer_forget = tf.keras.layers.Dense(units, activation="sigmoid", bias_initializer=tf.keras.initializers.Ones())

        # Input Gate
        self.dense_layer_input = tf.keras.layers.Dense(units, activation="sigmoid")

        # Cell-state Candiates
        self.dense_layer_candiates = tf.keras.layers.Dense(units, activation="tanh") # <- Not sigmoid

        # Output Gate
        self.dense_layer_output = tf.keras.layers.Dense(units, activation="sigmoid")
        
    @tf.function
    def call(self, x, states):
        
        """
        Propagate the input towards the cell -> SINGLE TIME STEP
        Same symbols used as in Courseware

        Args:
            x input of the cell (batch size, input size)
            states of the cell: tuple (hidden_state, cell_state)

        Return:
            Output of the cell: tuple (hidden_state, cell_state)
        """

        hidden_state, cell_state = states

        concated = tf.concat((hidden_state, x), axis= 1)

        # 1. Preparing

        # Forget Gate
        f_t = self.dense_layer_forget(concated)

        # Input Gate
        i_t = self.dense_layer_input(concated)

        # Cell-state Candiates
        ĉ_t = self.dense_layer_candiates(concated)
        
        # 2. Update Cell State
        cell_state = tf.math.multiply(f_t, cell_state) + tf.math.multiply(i_t, ĉ_t)

        # 3. Determining hidden state/output
        # Output gate
        o_t = self.dense_layer_output(concated)

        # New hidden state
        regularized_cell_state = tf.math.tanh(cell_state)
        hidden_state = tf.math.multiply(o_t, regularized_cell_state)

        return hidden_state, cell_state 