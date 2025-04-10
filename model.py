import torch
import torch.nn as nn

class CNN(nn.Module):
        def __init__(self,num_classes):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            self.fc1 = nn.Linear(32 * 25 * 25, 512)
            self.fc2 = nn.Linear(512, num_classes)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.pool(x)

            x = self.conv2(x)
            x = self.relu(x)
            x = self.pool(x)

            x = x.view(x.size(0), -1)  # Flatten the tensor

            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)

            x = self.fc2(x)
            
            return x