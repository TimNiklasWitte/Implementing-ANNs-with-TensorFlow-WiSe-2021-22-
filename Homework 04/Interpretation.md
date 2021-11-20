# Interpretation of the results

- SGD and Momentum lead to same results
- Adagrad and RMSprop same performance, higher test and train loss -> more generlized (see threshold)
- Adam has same performance but lower test and train error -> more overfitted
- L1 Regularization useless in performance but low loss in comparison to other optimizers (threshold)
- L2 Regularization leads to similar results than Adagrad and RMSprop
- Dropout similar results than Adam
- Label smoothing leads to similar results than L1 Regularization

result = performance and loss (train and test)
