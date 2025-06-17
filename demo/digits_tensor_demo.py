# demo/digits_demo_fast.py
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from engine.tensor_engine import Tensor
from engine.tensor_modules import MLP
from engine.tensor_optimizer import SGD

# 1) Load & normalize
digits = datasets.load_digits()
X, y = digits.data, digits.target.reshape(-1,1)
X = StandardScaler().fit_transform(X)

# 2) One-hot encode
Y = OneHotEncoder(sparse_output=False).fit_transform(y)

# 3) Train/val split, then subsample 500 for speed
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.2, random_state=0, stratify=y
)
X_train, Y_train = X_train[:500], Y_train[:500]

# 4) Build Tensor datasets
train_data = [([Tensor(f) for f in x], Tensor(label))
              for x, label in zip(X_train, Y_train)]
val_data   = [([Tensor(f) for f in x], Tensor(label))
              for x, label in zip(X_val,   Y_val)]

# 5) Model & loss
model = MLP(nin=64, nouts=[32,10])
def cross_entropy(logits, labels):
    p = logits.softmax(axis=-1)
    return - (p*labels).sum(axis=-1).log().mean()

# 6) Training loop
def train_sgd(epochs=10, lr=0.01, batch_size=32):
    N = len(train_data)
    opt = SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(1, epochs+1):
        perm = np.random.permutation(N)
        total_loss = 0.0

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            opt.zero_grad()
            for j in idx:
                x_list, y = train_data[j]
                loss = cross_entropy(model(x_list), y)
                total_loss += loss.data
                loss.backward()
            for p in model.parameters():
                p.grad /= len(idx)
            opt.step()

        # End of epoch: report
        avg_loss = total_loss / N
        correct = sum(
            np.argmax(model(x_list).softmax().data) == np.argmax(y.data)
            for x_list, y in val_data
        )
        acc = correct / len(val_data)
        print(f"Epoch {epoch:2d}  TrainLoss {avg_loss:.4f}  ValAcc {acc:.2%}", flush=True)

if __name__ == "__main__":
    print("=== Quick Digits smoke-test ===")
    train_sgd(epochs=10, lr=0.01, batch_size=32)

# Sample Run Results
# === Quick Digits smoke-test ===
# Epoch  1  TrainLoss 9.9027  ValAcc 40.83%
# Epoch  2  TrainLoss 2.2454  ValAcc 72.78%
# Epoch  3  TrainLoss 0.7751  ValAcc 80.28%
# Epoch  4  TrainLoss 0.4117  ValAcc 83.06%
# Epoch  5  TrainLoss 0.2691  ValAcc 83.33%
# Epoch  6  TrainLoss 0.2060  ValAcc 84.17%
# Epoch  7  TrainLoss 0.1568  ValAcc 83.89%
# Epoch  8  TrainLoss 0.1258  ValAcc 83.61%
# Epoch  9  TrainLoss 0.1114  ValAcc 84.44%
# Epoch 10  TrainLoss 0.0926  ValAcc 85.28%