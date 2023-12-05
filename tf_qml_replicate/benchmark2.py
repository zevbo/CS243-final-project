from io import TextIOWrapper
import math
import numpy as np
import random
import tensorflow as tf
import tensorflow_model_optimization as tfmot



ModelT = tf.keras.Model

QUANTIZE = True
INPUT_SIZE = 5
L1_SIZE = 10
def benchmark_2_model() -> ModelT:
    model = tf.keras.models.Sequential()
    d1 = tf.keras.layers.Dense(units=L1_SIZE, input_dim=INPUT_SIZE, activation='relu')
    d2 = tf.keras.layers.Dense(units=1, input_dim=L1_SIZE, activation='linear')
    if QUANTIZE: 
        d1 = tfmot.quantization.keras.quantize_annotate_layer(d1)
        d2 = tfmot.quantization.keras.quantize_annotate_layer(d2)
    model.add(d1)
    model.add(d2)
    op = tf.keras.optimizers.legacy.SGD(learning_rate=0.001)
    if QUANTIZE:
        model = tfmot.quantization.keras.quantize_apply(model)
    model.compile(optimizer=op, loss='mean_squared_error')
    return model

def rand_input() -> np.ndarray:
    return np.random.rand(1, INPUT_SIZE) * 5

def function(t: np.ndarray) -> float:
    return (2 * t[0] / max(0.5, t[4]) + t[1]  * math.sqrt(t[3]) - t[2] * 0.5 - 0.1 * t[3])

training_data = []
quick_losses = []

def test_loss(model: ModelT) -> float:
    input_tensor = np.array([[0.3971, 0.7544, 0.5695, 0.4388, 0.6387]])
    output = model(input_tensor)[0].numpy()[0]
    value = function(input_tensor[0])
    loss = (value - output) * (value - output)
    print(f"Test loss {loss}: {value = }, {output = }")

def calc_loss(model: ModelT, num_val: int) -> float:
    loss = 0
    for _ in range(num_val):
        input_tensor = rand_input()   
        output = model(input_tensor)[0].numpy()[0]
        value = function(input_tensor[0])
        loss += (value - output) * (value - output)
    return loss / num_val

def train_model(model: ModelT, num_trains: int) -> None:
    print(f"Loss at start: {calc_loss(model, 1000)}")
    random.shuffle(training_data)
    # model.train()

    Xs = []
    Ys = []

    for _ in range(num_trains):
        input_tensor = rand_input()[0]   
        value = function(input_tensor)
        # output = model(input_tensor.unsqueeze(0).float())
        # model.sess()
        Xs.append(input_tensor)
        Ys.append(value)
        # loss = nn.MSELoss()(output, torch.tensor([[value]]).float())

        # loss.backward()
        # optimizer.step()
    
    Xs = np.array(Xs)
    Ys = np.array(Ys)

    model.fit(Xs, Ys, batch_size= 1, epochs= 1)
    print(f"Loss at end: {calc_loss(model, 1000)}")

def run_benchmark() -> None:
    np.random.seed(0)
    tf.random.set_seed(0)
    tf.keras.utils.set_random_seed(0)
    # tf.random.set_gl(0)
    # torch.random.manual_seed(1)
    # model = Benchmark1Model()
    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    # print(model.fc1.state_dict()['weight'].numpy())
    # print(model.fc1.state_dict()['bias'].numpy())
    # train_model(model, optimizer, 1000)
    model = benchmark_2_model()
    d : tf.keras.layers.Dense = model.layers[1 if QUANTIZE else 0]
    # print(f"{d.weights = }")
    # d.set_weights([w,  np.array([0.2682017])])
    test_loss(model)
    # d.set_weights([[0.23043269, -0.19739035, -0.08669749,  0.20990819, -0.42102337]])
    # [0.2682017]
    # l = calc_loss(model, 100)
    # print(f"{l = }")
    train_model(model, 10000)
    # model.compute_loss(i, np.array([function(i)]))
    # print(model(example))

if __name__ == "__main__":
    run_benchmark()
