# Interpretation of the results

- All variants of the models has the same performance -> Same minimum reached
- SGD and Momentum has similar train and test loss
- Adagrad lower loss on train data -> More overfitted
- RMSprop lower test and train loss than Adagrad -> More overfitted
- Adam lower test and train loss than Adagrad -> Less overfitted/more generalized
- L1 and L2 Regularization similar train and test loss (similar to SGD and Momentum)
- Dropout similar performance and train and test loss than RMSprop
- Label smoothing higher train and test loss -> Prevent overfitting
- Adam + L2 + Dropout: similar train and test loss to RMSprop. Dropout shall prevent overfitting but Adam prevents to generalization.
- Combining three optimizers but the same minimum reached -> high chance to be a global minimum
