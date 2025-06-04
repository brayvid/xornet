import torch
import torch.nn as nn
import torch.optim as optim

# 1. XOR dataset
X = torch.tensor([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]], dtype=torch.float32)
Y = torch.tensor([[0.],
                  [1.],
                  [1.],
                  [0.]], dtype=torch.float32)

# 2. Define the neural network
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # input to hidden
        self.fc2 = nn.Linear(2, 1)  # hidden to output
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return x

# 3. Instantiate model, loss, and optimizer
model = XORNet()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. Train the model
for epoch in range(10000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# 5. Print learned weights and biases
print("\nTrained Weights and Biases:")
print("fc1 weights:\n", model.fc1.weight.data)
print("fc1 biases:\n", model.fc1.bias.data)
print("fc2 weights:\n", model.fc2.weight.data)
print("fc2 biases:\n", model.fc2.bias.data)

# 6. Evaluate predictions
with torch.no_grad():
    predictions = model(X)
    print("\nPredictions (rounded):")
    print(predictions.round())
