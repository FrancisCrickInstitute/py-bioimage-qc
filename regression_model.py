import torch
import torch.nn as nn


class CrossTalkRegressionModel(nn.Module):
    def __init__(self, input_channels=2, initial_filters=64, num_conv_blocks=5):
        super(CrossTalkRegressionModel, self).__init__()

        layers = []
        in_c = input_channels
        out_c = initial_filters

        # Initial Convolutional Block
        layers.append(nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.01))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # Halves dimensions

        # Subsequent Convolutional Blocks
        for i in range(1, num_conv_blocks):
            in_c = out_c
            out_c = min(out_c * 2, 512)  # Double filters, up to a max (e.g., 512 or 1024)
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # Halves dimensions

        self.conv_layers = nn.Sequential(*layers)

        # Calculate the size of the flattened features dynamically
        conv_output_size = self._get_conv_output((256, 256))

        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 512),  # First FC layer, larger output
            nn.BatchNorm1d(512),  # BatchNorm for FC layers too
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),  # Regularization

            nn.Linear(512, 128),  # Second FC layer
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),

            nn.Linear(128, 1),  # Output layer for scalar regression
            # No activation here if your target is unbounded, or Sigmoid for [0,1] range
            # For alpha values (0 to 1), a Sigmoid might be considered, but direct prediction often works.
            #nn.Sigmoid()
        )

    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.zeros(1, self.conv_layers[0].in_channels, *shape)
            output = self.conv_layers(input)
            return int(torch.prod(torch.tensor(output.size()[1:])))

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
