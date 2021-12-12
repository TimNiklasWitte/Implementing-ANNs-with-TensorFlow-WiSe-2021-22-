# IANNWTF - Homework 07

## Task

**Problem**: 
An input (aka input sequence) of length 25 shall be classified as 0 (target) if the sum of that sequence is greater than 1.
Otherwise, it shall be shall be classified as 1.

To solve this problem an Recurrent Neural Network (RNN) is used which contains an LSTM layer.
We archived an accuracy of about 90%.

## Usage

Run the `Training.py`. Thereafter, a window opens containing a plot which represents the performance of the trainined RNN.
This plot will be saved in the `result.png` file. 

```bash
python3 Training.py
```