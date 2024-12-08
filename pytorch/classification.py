import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


RANDOM_STATE = 42

# Dataset prepration
X = pd.read_excel(open("..\Datasets\data2.xlsx", "rb"), sheet_name="Export")
y = X.Creatinine
X.drop(["Creatinine"], axis=1, inplace=True) #axis=0 for rows and axis=1 for cols
y = torch.tensor(y, dtype=float)
for i in range(len(y)):
    if y[i]>1.2:
        y[i]=1
    else:
        y[i]=0


X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, shuffle=True, random_state=RANDOM_STATE)

X_train = torch.from_numpy(np.array(X_train).astype(np.float32))
X_test = torch.from_numpy(np.array(X_test).astype(np.float32))
y_train = torch.from_numpy(np.array(y_train).astype(np.float32))
y_test = torch.from_numpy(np.array(y_test).astype(np.float32))

print(f"Train Data Len:{len(X_train)}, Test Data Len:{len(X_test)}")

# Define Model
class BinaryClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Sequential(
            nn.Linear(7, 24),
            nn.ReLU(),
            nn.Linear(24, 48),
            nn.ReLU(),
            nn.Linear(48, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.linear_layer(x)
model1 = BinaryClassificationModel()

# Accuracy funtion
def accuracy_fn(y_true, y_pred):
  currect = torch.eq(y_true, y_pred).sum().item()
  acc = (currect/len(y_pred))*100
  return acc

# Function selection
torch.manual_seed(RANDOM_STATE)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model1.parameters(),
                            lr=0.01)

train_loss_list=[]
train_acc_list=[]
test_loss_list=[]
test_acc_list=[]

epochs = 20
for epoch in tqdm(range(epochs)):
    model1.train()
    y_logits = model1(X_train).squeeze()
    y_pred = torch.round(y_logits)
    train_loss = loss_fn(y_logits,
                         y_train)
    train_loss_list.append(float(train_loss))
    train_acc = accuracy_fn(y_true=y_train,                            
                            y_pred=y_pred)
    train_acc_list.append(float(train_acc))
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    model1.eval()
    with torch.inference_mode():
        test_logits = model1(X_test).squeeze()
        test_pred = torch.round(test_logits)
        
        test_loss = loss_fn(test_logits,
                            y_test)
        test_loss_list.append(float(test_loss))
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)
        test_acc_list.append(float(test_acc))

# Plot
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.plot(range(len(train_loss_list)), train_loss_list, label='train_loss')
plt.plot(range(len(test_loss_list)), test_loss_list, label='test_loss')
plt.title('Loss')
plt.xlabel('Epochs')

plt.subplot(1, 2, 2)
plt.plot(range(len(train_acc_list)), train_acc_list, label='train_accuracy')
plt.plot(range(len(test_acc_list)), test_acc_list, label='test_accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')

plt.tight_layout()
plt.show()