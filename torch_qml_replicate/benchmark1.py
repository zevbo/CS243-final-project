from io import TextIOWrapper
import math
from torch import nn , Tensor, optim
import torch
import numpy as np
import random



INPUT_SIZE = 5
class Benchmark1Model(torch.nn.Module):
    
    def __init__(self) -> None:
        super(Benchmark1Model, self).__init__()

        self.fc1 = nn.Linear(INPUT_SIZE, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        return x
    
def rand_input() -> Tensor:
    return torch.rand(INPUT_SIZE)

def function(t_: Tensor) -> float:
    t = t_.numpy()
    return 5 * (2 * t[0] / max(0.5, t[4]) + t[1]  * math.sqrt(t[3]) - t[2] * 0.5 - 0.1 * t[3])

training_data = []
quick_losses = []

def calc_loss(model: nn.Module, num_val: int) -> float:
    loss = 0
    for _ in range(num_val):
        input_tensor = rand_input()   
        output = model(input_tensor.unsqueeze(0).float())
        value = function(input_tensor)
        # f_str = "%0.2f" % output.item()
        # print(f"{f_str}", end=", ")
        loss += nn.MSELoss()(output, torch.tensor([[value]]).float())
    return loss / num_val

def train_model(model: nn.Module, optimizer: optim.Optimizer, num_trains: int) -> None:
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
    torch.random.manual_seed(1)
    model = Benchmark1Model()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    print(model.fc1.state_dict()['weight'].numpy())
    print(model.fc1.state_dict()['bias'].numpy())
    train_model(model, optimizer, 1000)
    # print(model(example))

if __name__ == "__main__":
    run_benchmark()
