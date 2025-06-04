import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define XOR dataset
X = torch.tensor([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]], dtype=torch.float32)
Y = torch.tensor([[0.],
                  [1.],
                  [1.],
                  [0.]], dtype=torch.float32)

# 2. Define the network architecture
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # Input to hidden layer
        self.fc2 = nn.Linear(2, 1)  # Hidden to output layer
        self.activation = nn.Sigmoid()  # Nonlinearity

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return x

# 3. Instantiate model, loss, and optimizer
model = XORNet()
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
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

# 5. Evaluate the model
with torch.no_grad():
    predictions = model(X)
    print("\nPredictions:")
    print(predictions.round())  # Rounded to 0 or 1
