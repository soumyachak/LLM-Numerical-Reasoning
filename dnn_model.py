import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import notebook
import matplotlib.pyplot as plt

'''Generate data'''
np.random.seed(32)
X1 = np.random.randint(1, 100, size=1000)
X2 = np.random.randint(1, 100, size=1000)
Y = np.log(X1) + np.sin(X2)
df = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})

'''Define model'''
class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DeepNeuralNetwork, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(7)])
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_layer(x)
        for idx, layer in enumerate(self.hidden_layers):
            x = torch.tanh(layer(x))
        x = self.output_layer(x)
        return x
    
'''Create training and evaluation dataset'''
input_size = 2
hidden_size = 16
model = DeepNeuralNetwork(input_size, hidden_size).to('cuda')

X_train_tensor = torch.tensor(df.iloc[:800,:-1].values, dtype=torch.float32).to('cuda')
y_train_tensor = torch.tensor(df.iloc[:800,-1].values, dtype=torch.float32).to('cuda')
X_test_tensor = torch.tensor(df.iloc[800:,:-1].values, dtype=torch.float32).to('cuda')
y_test_tensor = torch.tensor(df.iloc[800:,-1].values, dtype=torch.float32).to('cuda')

batch_size = 4
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

'''Define optimizer and loss function'''
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

'''Model training'''
num_epochs = 1000
train_loss_per_epoch = []
eval_loss_per_epoch= []
for epoch in notebook.tqdm(range(num_epochs)):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_dataloader:
        y_batch = y_batch.unsqueeze(1)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss
    train_loss_per_epoch.append(train_loss/(800/4))

    model.eval()
    with torch.no_grad():
        val_loss = 0
        for X_val_batch, y_val_batch in test_dataloader:
            outputs = model(X_val_batch)
            loss = criterion(outputs, y_val_batch)
            val_loss += loss
        eval_loss_per_epoch.append(val_loss/200)

    print("Train loss", train_loss/(800/4), " Test loss", val_loss/200)

'''Save model'''
model_save_path = "dnn.pt"
torch.save(model.state_dict(), model_save_path)

'''Load model'''
input_size = 2
hidden_size = 16
model_save_path = "dnn.pt"
model = DeepNeuralNetwork(input_size, hidden_size)
model.load_state_dict(torch.load(model_save_path))
model.eval()
model.to('cuda')

'''Model evaluation'''
np.random.seed(32)
X1 = np.random.randint(1, 100, size=1000)
X2 = np.random.randint(1, 100, size=1000)
Y = np.log(X1) + np.sin(X2)

lst = []
lst1 = []
for x in X1:
    inp = torch.tensor([x, 12], dtype=torch.float32).to('cuda')
    with torch.no_grad():
        cat = model(inp)
    result = cat.cpu().numpy()[0]
    lst.append([x, result])
    lst1.append([x, np.log(x) + np.sin(12)])

lst = sorted(lst, key=lambda x: x[0])
lst1 = sorted(lst1, key=lambda x: x[0])
plt.plot([x[0] for x in lst], [x[1] for x in lst])
plt.plot([x[0] for x in lst1], [x[1] for x in lst1])