# Interpretation 

- After one epoch: Accuracy reached approximately 82%, the loss goes down from over 0.8 to under 0.1 -> Problem simple to learn for the RNN
- Futher epoches: slow reaching (converge) to 90% accuracy -> fine tuning of weights 

## Recurrent Neural Network Architecture

```console
Model: "my_model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               multiple                  64        
                                                                 
 lstm__layer (LSTM_Layer)    multiple                  53200     
                                                                 
 dense_5 (Dense)             multiple                  101       
                                                                 
=================================================================
Total params: 53,365
Trainable params: 53,365
Non-trainable params: 0
_________________________________________________________________

```

## Results

![alt text](result.png)

## Questions

### Can / should you use truncated BPTT here?

### Should you rather take this as a regression, or a classification problem?

- Regression:
- Classification
