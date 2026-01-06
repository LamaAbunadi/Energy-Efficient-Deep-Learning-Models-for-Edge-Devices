**1.Load & Preprocess Data**

```
import pandas as pd
import time
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
df = pd.read_csv("/kaggle/input/environmental-sensor-data-132k/iot_telemetry_data.csv")

features = ['co', 'humidity', 'light', 'lpg', 'motion', 'smoke', 'temp']
X = df[features].values

# Example rule-based labeling (can be replaced)
df['label'] = (
    (df['co'] > 9) |
    (df['smoke'] > 300) |
    (df['temp'] > 50)
).astype(int)

y = df['label'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test, dtype=torch.long)
```
**2.Dataset & DataLoader**

```
from torch.utils.data import Dataset, DataLoader

class SensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(SensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader  = DataLoader(SensorDataset(X_test, y_test), batch_size=32, shuffle=False)
```

**3.Baseline Model (Edge-Friendly MLP)**

```
import torch.nn as nn

class MultiSensorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiSensorNet().to(device)

```

**4.Training**

```
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(5):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

```

**5.Evaluation**

**First Standard: is Accuracy**

```
def evaluate(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100 * correct / total

print("Baseline Accuracy:", evaluate(model), "%")
```

```
Baseline Accuracy: 100.0 %
```
**Second Standard: is Model Size (KB)**
```
import torch
import os

torch.save(model.state_dict(), "model.pth")

model_size_kb = os.path.getsize("model.pth") / 1024
print(f"Model Size: {model_size_kb:.2f} KB")
```

```
Model Size ≈ 10.24 KB
```
**Third Standard: Inference Time (ms)**

```


if device.type == 'cuda':
    torch.cuda.synchronize()
start_time = time.time()

with torch.no_grad():
    outputs = model(X_test)

if device.type == 'cuda':
    torch.cuda.synchronize()
end_time = time.time()

num_samples = X_test.shape[0]
total_time = end_time - start_time
inference_time_per_sample = (total_time / num_samples) * 1000  # ms per sample

print(f"Inference time per sample: {inference_time_per_sample:.4f} ms")
```


```
Inference time per sample: 0.0001 ms
```

**Fourth Standard: Estimated Energy Consumption**

```
# -----------------------------
# 1. Set approximate Power (Watts)
# -----------------------------
# Adjust based on your device
if device.type == 'cuda':
    power_watts = 120  # typical GPU power
else:
    power_watts = 65   # typical CPU power

# -----------------------------
# 2. Estimated energy per sample
# -----------------------------
energy_per_sample_joules = (total_time / num_samples) * power_watts
print(f"Estimated energy per sample: {energy_per_sample_joules:.4f} J")

# -----------------------------
# 3. Estimated total energy for the entire test set
# -----------------------------
energy_total_joules = total_time * power_watts
print(f"Estimated total energy for all samples: {energy_total_joules:.4f} J")


```


```
Estimated energy per sample: 0.0000 J
Estimated total energy for all samples: 0.5718 J
```


**6.Structured Pruning**



```
import torch.nn.utils.prune as prune

for module in model.modules():
    if isinstance(module, nn.Linear):
        prune.ln_structured(module, name='weight', amount=0.3, n=2, dim=0)

for module in model.modules():
    if isinstance(module, nn.Linear):
        prune.remove(module, 'weight')

```

**First Standard: is Accuracy**
```

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for features, labels in test_loader:
        features = features.to(device)   # شكلها (batch_size, 9)
        labels = labels.to(device)

        outputs = model(features)         # (batch_size, num_classes)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy after pruning: {accuracy:.2f}%")


```

```
Accuracy after pruning: 100.00%
```
**Second Standard: is Model Size (KB)**

```
Model Size: 4.28 KB
```

**Third Standard: Inference Time (ms)**
```

```

**Fourth Standard: Estimated Energy Consumption**

```

```


**7.INT8 Quantization**

```
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8
)
```
**8.Model Size & Latency**

```
import os, time

torch.save(model.state_dict(), "fp32.pth")
torch.save(quantized_model.state_dict(), "int8.pth")

print("FP32 size (MB):", os.path.getsize("fp32.pth") / 1024**2)
print("INT8 size (MB):", os.path.getsize("int8.pth") / 1024**2)

```
```

def latency(model):
    model.eval()
    x = torch.randn(1, 7)
    start = time.time()
    for _ in range(100):
        model(x)
    return (time.time() - start) / 100

print("FP32 latency:", latency(model))
print("INT8 latency:", latency(quantized_model))
```
