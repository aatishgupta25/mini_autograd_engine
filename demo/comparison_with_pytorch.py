# digits_demo_torch.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1) Load & normalize (same as before)
digits = datasets.load_digits()
X, y = digits.data, digits.target.reshape(-1,1)
X = StandardScaler().fit_transform(X)

# 2) One-hot encode
enc = OneHotEncoder(sparse_output=False)
Y = enc.fit_transform(y)

# 3) Train/val split + subsample 500
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.2, random_state=0, stratify=y
)
X_train, Y_train = X_train[:500], Y_train[:500]

# 4) Create PyTorch datasets and loaders (vs. manual Python loops)
train_ds = TensorDataset(
    torch.from_numpy(X_train).float(),
    torch.from_numpy(Y_train).float()
)
val_ds   = TensorDataset(
    torch.from_numpy(X_val).float(),
    torch.from_numpy(Y_val).float()
)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=32)

# 5) Define model: 64→32→10 (vs. custom MLP)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))        # built-in ReLU
        x = self.fc2(x)                # logits
        return x

model = Net()

# 6) Optimizer & loss (vs. custom SGD and cross_entropy)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()  # expects integer labels

# convert one-hot to class indices for CrossEntropyLoss
y_train_idx = torch.from_numpy(Y_train).argmax(dim=1)
y_val_idx   = torch.from_numpy(Y_val).argmax(dim=1)

# 7) Training loop (uses DataLoader, zero_grad, backward, step)
for epoch in range(1, 11):
    model.train()
    total_loss = 0.0

    for xb, yb in train_loader:
        optimizer.zero_grad()                          # same concept
        logits = model(xb)                             # forward pass
        loss   = criterion(logits, yb.argmax(dim=1))   # built-in CE
        loss.backward()                                # built-in autograd
        optimizer.step()                               # built-in step
        total_loss += loss.item() * xb.size(0)

    # validation
    model.eval()
    correct = 0
    with torch.no_grad():                              # no grad in eval
        for xb, yb in val_loader:
            logits = model(xb)
            preds  = logits.argmax(dim=1)
            correct += (preds == yb.argmax(dim=1)).sum().item()

    avg_loss = total_loss / len(train_ds)
    val_acc  = correct / len(val_ds)
    print(f"Epoch {epoch:2d}  TrainLoss {avg_loss:.4f}  ValAcc {val_acc:.2%}")
