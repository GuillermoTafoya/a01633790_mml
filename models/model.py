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

class LargeCategoryClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LargeCategoryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.leaky_relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
