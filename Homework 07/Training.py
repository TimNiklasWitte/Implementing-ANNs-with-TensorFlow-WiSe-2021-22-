import numpy as np

RANGE_MAX = 2
RANGE_MIN = -RANGE_MAX


def main():
    data = next(integration_task(10, 100))
    print(data)

    

def integration_task(seq_len, num_samples):

    for _ in range(num_samples):
        data = np.random.uniform(RANGE_MIN, RANGE_MAX, seq_len)
        data = np.expand_dims(data,-1)
        sum = np.sum(data)

        if sum >= 1:
            yield (data, 1)
        else:
            yield (data, 0)



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received.")