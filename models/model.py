import torch.nn as nn

class RiskClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(RiskClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.fc(x)

# class CategoryClassifier(nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super(CategoryClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, num_classes)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.3)
        
#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)  # Output logits
#         return x

class CategoryClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CategoryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
