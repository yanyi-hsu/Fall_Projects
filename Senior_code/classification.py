import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from tqdm import tqdm

df = pd.read_csv('./keypoint.csv')

# class NeuralNet(torch.nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(NeuralNet, self).__init__()
#         self.l1 = torch.nn.Linear(input_size, hidden_size) 
#         self.l2 = torch.nn.Linear(hidden_size, num_classes)
#         self.relu = torch.nn.ReLU()
        
#     def forward(self, x):
#         out = self.l1(x)
#         out = self.relu(out)
#         out = self.l2(out)
#         return out

class NeuralNet(nn.Module):
    def __init__(self, input_size=34, hidden_size1=64, hidden_size2=32, num_classes=2):
        super(NeuralNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(34, 64),
            # nn.BatchNorm1d(64),  
            nn.ReLU(), 
            nn.Dropout(0.2), 
            nn.Linear(64, 32), 
            # nn.BatchNorm1d(32),
            nn.ReLU(), 
            nn.Dropout(0.2), 
            nn.Linear(32, 2)
        )
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.network(x)

class DataKeypointClassification(Dataset):
    def __init__(self, X, y):
        self.x = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.n_samples = X.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples
    
# Encoder label
encoder = LabelEncoder()
y_label = df['label']
y = encoder.fit_transform(y_label)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)#num_classes
X = df.iloc[:,1:]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, stratify=y, 
        shuffle=True, random_state=2022)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
train_dataset = DataKeypointClassification(X_train, y_train)
test_dataset = DataKeypointClassification(X_test, y_test)
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Build model
# hidden_size = 512
# model = NeuralNet(X_train.shape[1], hidden_size, len(class_weights))
model = NeuralNet()
# Loss & Optimizer
criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights.astype(np.float32)))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Training
num_epoch = 500
for epoch in range(num_epoch):
    train_acc = 0
    train_loss = 0
    loop = tqdm(train_loader)
    for idx, (features, labels) in enumerate(loop):
        outputs = model(features)
        loss = criterion(outputs, labels)

        predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
        correct = (predictions == labels).sum().item()
        accuracy = correct / batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch [{epoch}/{num_epoch}]")
        loop.set_postfix(loss=loss.item(), acc=accuracy)

# Test model
test_features = torch.from_numpy(X_test.astype(np.float32))
test_labels = y_test
with torch.no_grad():
    outputs = model(test_features)
    _, predictions = torch.max(outputs, 1)
print(classification_report(test_labels, predictions, target_names=encoder.classes_))

torch.save(model.state_dict(), './classification.pth')