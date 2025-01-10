import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------
# ResNet Building Blocks
# ------------------------

class BasicBlock(nn.Module):
    expansion = 1  # For BasicBlock, expansion is 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=3, stride=stride, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # To match dimensions
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        # First layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second layer
        out = self.conv2(out)
        out = self.bn2(out)

        # Adjust identity if dimensions change
        if self.downsample is not None:
            identity = self.downsample(x)

        # Residual connection
        out += identity
        out = self.relu(out)

        return out

# ------------------------
# Customized ResNet
# ------------------------

class ResNet(nn.Module):
    def __init__(self, layers, block=BasicBlock, n_classes=94, input_channels=1):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(input_channels, 64, 
                               kernel_size=7, stride=2, 
                               padding=3, bias=False)  # Adjust kernel size and stride as needed
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # Max pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])   # e.g., 2 blocks
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Adaptive average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Linear(512 * block.expansion, n_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        # If the input and output dimensions differ, adjust the identity
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # First block may need downsampling
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        # Initialize weights as in the original ResNet paper
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.unsqueeze(dim=1)

        # Input x: (batch_size, 1, 128, 64)
        x = self.conv1(x)      # (batch_size, 64, H/2, W/2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # (batch_size, 64, H/4, W/4)

        x = self.layer1(x)     # (batch_size, 64, H/4, W/4)
        x = self.layer2(x)     # (batch_size, 128, H/8, W/8)
        x = self.layer3(x)     # (batch_size, 256, H/16, W/16)
        x = self.layer4(x)     # (batch_size, 512, H/32, W/32)

        x = self.avgpool(x)    # (batch_size, 512, 1, 1)
        x = torch.flatten(x, 1)  # (batch_size, 512)
        x = self.fc(x)         # (batch_size, n_classes)
        
        return x
