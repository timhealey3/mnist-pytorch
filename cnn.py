import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # in channels starts at 1 because the image is in black and white
            # out channels is the number of filters being applied
            # start out with a filter size of 3 x 3
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),
            # non linearity
            nn.ReLU(),
            # max pooling
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            nn.Linear(2704, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        return self.model(x)