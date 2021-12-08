import tensorflow as tf

class LSTM_Cell:

    # units = hidden size = size of hidden state vector #= The length of the resulting vector (similar to the units argument in Dense layers)
    # 
    def __init__(self,units):

        # Forget Gate
        self.dense_layer_forget = tf.keras.layers.Dense(units, activation="sigmoid", bias_initializer=tf.keras.initializers.Ones())

        # Input Gate
        self.dense_layer_input = tf.keras.layers.Dense(units, activation="sigmoid")

        # Cell-state Candiates
        self.dense_layer_candiates = tf.keras.layers.Dense(units, activation="tanh") # <- Not sigmoid

        # Output Gate
        self.dense_layer_output = tf.keras.layers.Dense(units, activation="sigmoid")
        

    def call(self, x, states):

        cell_state, hidden_state = states

        concated = tf.concat((states, x), axis= 1)

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

        print(concated)

        return cell_state, hidden_state