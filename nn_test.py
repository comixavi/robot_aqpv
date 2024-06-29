import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import random


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu_(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')


def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.eq(predicted, labels).sum().item()  # Using torch.eq for equality check
    print(f'Accuracy: {100 * correct / total}%')


def test_with_random_photos(model, test_dataset, num_photos=5):
    model.eval()
    indices = random.sample(range(len(test_dataset)), num_photos)
    fig, axs = plt.subplots(1, num_photos, figsize=(15, 5))
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, label = test_dataset[idx]
            output = model(image.unsqueeze(0))
            _, predicted = torch.max(output.data, 1)
            axs[i].imshow(image.squeeze(), cmap='gray')
            axs[i].set_title(f'True: {label}, Pred: {predicted.item()}')
            axs[i].axis('off')
    plt.show()


def save_model_to_excel(model, filename):
    model_params = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    with pd.ExcelWriter(filename) as writer:
        for param_name, param_values in model_params.items():
            if param_values.ndim == 1:
                df = pd.DataFrame(param_values.reshape(-1, 1))  # Ensure 1D tensors are saved as columns
            else:
                df = pd.DataFrame(param_values)
            df.to_excel(writer, sheet_name=param_name)


def load_model_from_excel(model, filename):
    state_dict = {}
    xls = pd.ExcelFile(filename)
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(filename, sheet_name=sheet_name, index_col=0)
        param_values = df.values
        if param_values.shape[1] == 1:
            param_values = param_values.flatten()
        state_dict[sheet_name] = torch.tensor(param_values)
    model.load_state_dict(state_dict)


train = True

if train:
    train_model(model, train_loader, criterion, optimizer, epochs=10)
    test_model(model, test_loader)

save_model_to_excel(model, 'model_params.xlsx')

loaded_model = SimpleNN()
load_model_from_excel(loaded_model, 'model_params.xlsx')

test_model(loaded_model, test_loader)

test_with_random_photos(loaded_model, test_dataset, num_photos=5)
