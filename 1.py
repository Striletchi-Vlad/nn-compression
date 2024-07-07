import torch


# Create a fully-connected feed-forward neural network that takes a number as input and outputs a number from 0 to 256.
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(1, 100)
        self.fc2 = torch.nn.Linear(100, 100)
        self.fc3 = torch.nn.Linear(100, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


# Train the network on the following dataset:
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

# Create the network
net = Net()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

# Train the network
for epoch in range(1000):
    for i in range(len(x)):
        x_i = torch.tensor([x[i]], dtype=torch.float32)
        y_i = torch.tensor([y[i]], dtype=torch.float32)
        optimizer.zero_grad()
        y_hat = net(x_i)
        loss = criterion(y_hat, y_i)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Test the network
for i in range(len(x)):
    x_i = torch.tensor([x[i]], dtype=torch.float32)
    y_i = torch.tensor([y[i]], dtype=torch.float32)
    print(y_i.shape)
    print(x_i.shape)
    y_hat = net(x_i)
    print(y_hat.shape)
    print(f"Input: {x_i.item()}, Target: {y_i.item()}, Prediction: {y_hat.item()}")
