import torch
import torch.nn as nn
import torch.optim as optim
import importlib.util
import sys

# 동적으로 250610_model_lenet와 250610_data_mnist import
spec_model = importlib.util.spec_from_file_location("model_lenet", "250610_model_lenet.py")
model_lenet = importlib.util.module_from_spec(spec_model)
sys.modules["model_lenet"] = model_lenet
spec_model.loader.exec_module(model_lenet)

spec_data = importlib.util.spec_from_file_location("data_mnist", "250610_data_mnist.py")
data_mnist = importlib.util.module_from_spec(spec_data)
sys.modules["data_mnist"] = data_mnist
spec_data.loader.exec_module(data_mnist)


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch}, Train Loss: {avg_loss:.4f}')


def evaluate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(val_loader)
    accuracy = 100. * correct / (len(val_loader.dataset))
    print(f'Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return val_loss, accuracy


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = data_mnist.get_mnist_dataloaders()
    model = model_lenet.LeNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    epochs = 10
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        evaluate(model, device, val_loader, criterion)
    torch.save(model.state_dict(), '250610_lenet_mnist.pth')
    print('Model saved as 250610_lenet_mnist.pth')

if __name__ == '__main__':
    main() 