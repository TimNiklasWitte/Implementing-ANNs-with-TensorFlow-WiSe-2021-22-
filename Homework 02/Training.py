from MultiLayerPerceptron import *
from LogicOperators import *

import matplotlib.pyplot as plt
import random

def generatorLogicFunction(operator):
    bools = [True, False]
    while True:
        x1 = bools[random.randint(0,1)]
        x2 = bools[random.randint(0,1)]
        target = operator.result(x1,x2)

        result = [x1, x2, target]

        # Bool to [0,1]
        for i, value in enumerate(result):
            if value:
                result[i] = 1
            else:
                result[i] = 0

        yield result

def getPerformance(mlp, operator,  testSize = 50):

    THRESHOLD = 0.5
    loss = 0
    cntCorrect = 0
    for i in range(testSize):
        data = next(generatorLogicFunction(operator))
        [x1, x2, target] = data
        y = mlp.forward_step([data[0], data[1]])
        loss += (y - target)**2

        # MLP said "this is true"
        if y > THRESHOLD:
            # correct?
            if target == 1:
                cntCorrect += 1

        # MLP said "this is false"
        else:
            # correct?
            if target == 0:
                cntCorrect += 1

    avgPerformance = cntCorrect/testSize
    avgLoss = loss/testSize
    return (avgPerformance, avgLoss)


def main():

    epsilon = 1 # Learning rate
    NUM_EPOCHS = 1000
    operators = [AND(), OR(), NOT(), NAND(), NOR(), XOR()]

    fig, axs = plt.subplots(2, 3)
    row_idx = 0
    column_idx = 0
    for operator in operators:
        mlp = MultiLayerPerceptron([2,4,1])
        performance = []
        loss = []
        for i in range(NUM_EPOCHS):

            # Measure
            avgPerformance, avgLoss = getPerformance(mlp, operator)
            performance.append(avgPerformance)
            loss.append(avgLoss)

            # Train
            data = next(generatorLogicFunction(operator))
            [x1, x2, target] = data
            mlp.backprop_step([x1, x2], target, epsilon)

        # Plot data
        axs[row_idx, column_idx].set_title(operator.name)
        axs[row_idx, column_idx].plot(np.arange(0,NUM_EPOCHS), performance, color= 'b')
        ax2=axs[row_idx, column_idx].twinx()
        ax2.plot(np.arange(0,NUM_EPOCHS), loss, color= 'r')

        # Subplot logic
        if column_idx == 0:
            axs[row_idx, column_idx].set_ylabel("Average performance", color= 'b')

        if column_idx == 2:
            column_idx = 0
            row_idx += 1
            ax2.set_ylabel("Average loss", color= 'r')
        else:
            column_idx +=1

        print(f"Current evaluation: {operator.name}")

    # Format plot
    for ax in axs.flat:
        ax.label_outer()

    fig.set_size_inches(14.5, 7.5)
    fig.savefig("result.png")
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received.")
