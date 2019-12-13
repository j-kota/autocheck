import torch
import torch.nn as nn
import torch as tr

num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001
board_size = 8
input_size = board_size**4


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 36, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
    # self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        x = torch.FloatTensor(x).reshape((16, 1, 4, 4))
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        # out = self.fc(out)
        return out


inputs = [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0,
          1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0,
          1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0,
          1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0,
          1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0,
          1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0,
          1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0,
          1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0,
          1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0,
          1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0,
          1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0,
          1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0,
          1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0,
          1, 1, 0, ]


def to_tensor(inputs): return torch.FloatTensor(inputs)


inputs = to_tensor(inputs)

inputs = inputs.reshape((16, 1, 5, 5))
print(inputs)
# Loss and optimizer
# model = ConvNet()
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# num_epochs = 10000
# for epoch in range(num_epochs):
#     # Start with zero gradient
#     optimizer.zero_grad()
#
#     # Calculate network output and sum of squared loss for each datapoint
#     y = model(inputs)
#     # loss = ((y.squeeze() - targets.squeeze())**2).sum()
#     # loss = criterion(outputs, labels)
#
#     # Calculate gradients and take a descent step
#     # loss.backward()
#     optimizer.step()
#
# # Monitor optimization progress
# # num_errors = (y.squeeze().round().detach().numpy() != targets.numpy()).sum()
# # if epoch % (num_epochs/10) == 0: print(loss, num_errors)
#
# # Print predicted and target outputs for the first 10 datapoints
# print(y.squeeze()[:])
# print(y.squeeze()[:].round())
# # print(targets[:].squeeze())
# print((y.squeeze().detach().numpy() > .5).any())