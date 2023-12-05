from io import TextIOWrapper
import math
import numpy as np
import random
import tensorflow as tf


ModelT = tf.keras.Model

INPUT_SIZE = 5
def benchmark_1_model() -> ModelT:
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=1, input_dim=INPUT_SIZE, activation='linear'))
    model.compile(optimizer='sgd', loss='mean_squared_error')
    return model

def rand_input() -> np.ndarray:
    return np.random.rand(INPUT_SIZE)

def function(t: np.ndarray) -> float:
    return 5 * (2 * t[0] / max(0.5, t[4]) + t[1]  * math.sqrt(t[3]) - t[2] * 0.5 - 0.1 * t[3])

training_data = []
quick_losses = []

def calc_loss(model: ModelT, num_val: int) -> float:
    loss = 0
    for _ in range(num_val):
        input_tensor = rand_input()   
        output = model(input_tensor.unsqueeze(0).float())
        value = function(input_tensor)
        # f_str = "%0.2f" % output.item()
        # print(f"{f_str}", end=", ")
        loss += nn.MSELoss()(output, torch.tensor([[value]]).float())
    return loss / num_val

def train_model(model: ModelT, num_trains: int) -> None:
    print(f"Loss at start: {calc_loss(model, 100)}")
    random.shuffle(training_data)
    model.train()

    for _ in range(num_trains):
        input_tensor = rand_input()   
        optimizer.zero_grad()
        value = function(input_tensor)
        output = model(input_tensor.unsqueeze(0).float())
        loss = nn.MSELoss()(output, torch.tensor([[value]]).float())

        loss.backward()
        optimizer.step()
    print(f"Loss at end: {calc_loss(model, 100)}")

def run_benchmark() -> None:
    np.random.seed(0)
    # torch.random.manual_seed(1)
    # model = Benchmark1Model()
    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    # print(model.fc1.state_dict()['weight'].numpy())
    # print(model.fc1.state_dict()['bias'].numpy())
    # train_model(model, optimizer, 1000)
    X_train = np.random.rand(1000, 5)  # 1000 samples, 5 features
    y_train = 2 * X_train.sum(axis=1) + 1 + 0.1 * np.random.randn(1000)
    model = benchmark_1_model()
    i = rand_input()
    print(f"{X_train.shape = }, {y_train.shape = }")
    model.compute_loss(X_train, y_train)
    # model.compute_loss(i, np.array([function(i)]))
    # print(model(example))

if __name__ == "__main__":
    run_benchmark()
