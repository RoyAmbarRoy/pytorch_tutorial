import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# Use the model (example)
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Softmax dist: {logits}")
print(f"Predicted class: {y_pred}")

# Flatten
input_img = torch.rand(3, 28, 28)
print(input_img.size())

flatten = nn.Flatten()
flat_img = flatten(input_img)
print(flat_img.size())

flat_flat_img = flatten(flat_img)
print(flat_flat_img.size())

# nn.Linear
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_img)
print(hidden1.size())

# nn.ReLU
print(f"Before ReLU: {hidden1}")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# nn.Sequential
seq_moduls = nn.Sequential(
    nn.Flatten(),
    layer1,
    nn.ReLU(),
    nn.Linear(20, 2)
)

input_img = torch.rand(3, 28, 28)
logits = seq_moduls(input_img)
print(logits)

# nn.Softmax
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
print(pred_probab)


# Model parameters
print("Model structure: ", model, "\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]}")