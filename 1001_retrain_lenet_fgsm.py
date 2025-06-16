import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import importlib.util
import sys
from tqdm import tqdm
import os

# 로그 저장 함수
class Logger:
    def __init__(self, filename):
        self.logfile = open(filename, 'w', encoding='utf-8')
    def logprint(self, *args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=self.logfile)
    def close(self):
        self.logfile.close()

# 1. LeNet 모델 불러오기
def load_lenet():
    spec_model = importlib.util.spec_from_file_location("model_lenet", "02_250610_model_lenet.py")
    model_lenet = importlib.util.module_from_spec(spec_model)
    sys.modules["model_lenet"] = model_lenet
    spec_model.loader.exec_module(model_lenet)
    return model_lenet.LeNet

LeNet = load_lenet()

def run_retrain(num_epochs, pth_path, txt_path):
    logger = Logger(txt_path)
    logprint = logger.logprint
    # 2. 데이터셋 준비
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    N = len(mnist_full)
    val_size = N // 6
    train_size = N - val_size
    mnist_train, mnist_val = torch.utils.data.random_split(
        mnist_full, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    adv_train = torch.load('06-1_250610_fgsm_lenet_mnist_train.pt')
    adv_train_imgs = adv_train['images']
    adv_train_labels = adv_train['labels']
    adv_train_dataset = TensorDataset(adv_train_imgs, adv_train_labels)
    adv_N = len(adv_train_dataset)
    adv_val_size = adv_N // 6
    adv_train_size = adv_N - adv_val_size
    adv_train_split, adv_val_split = torch.utils.data.random_split(
        adv_train_dataset, [adv_train_size, adv_val_size], generator=torch.Generator().manual_seed(42)
    )
    final_train = ConcatDataset([mnist_train, adv_train_split])
    final_val = ConcatDataset([mnist_val, adv_val_split])
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    adv_test = torch.load('06-2_250610_fgsm_lenet_mnist.pt')
    adv_test_imgs = adv_test['images']
    adv_test_labels = adv_test['labels']
    adv_test_dataset = TensorDataset(adv_test_imgs, adv_test_labels)
    final_test = ConcatDataset([mnist_test, adv_test_dataset])
    batch_size = 64
    train_loader = DataLoader(final_train, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(final_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(final_test, batch_size=batch_size, shuffle=False)
    # 3. 모델 준비
    model = LeNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    # 4. 학습/평가 함수
    def train_epoch(model, loader, criterion, optimizer, device):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        return total_loss / total, correct / total
    def eval_epoch(model, loader, criterion, device):
        model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                total_loss += loss.item() * x.size(0)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return total_loss / total, correct / total
    # 5. 학습 루프
    best_val_acc = 0
    logprint(f"Starting retraining for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        logprint(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        logprint(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        logprint(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc*100:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), pth_path)
            logprint(f"Model saved to {pth_path} (Val Acc: {best_val_acc*100:.2f}%)")
    logprint("\nRetraining completed!")
    # 6. 최종 test 평가
    model.load_state_dict(torch.load(pth_path, map_location=device))
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
    logprint(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc*100:.2f}%")
    logger.close()

if __name__ == "__main__":
    run_retrain(5, '1001-1_250616_lenet_mnist_retrained_5epoch.pth', '1001-2_250616_lenet_mnist_retrained_5epoch.txt')
    run_retrain(10, '1001-3_250616_lenet_mnist_retrained_10epoch.pth', '1001-4_250616_lenet_mnist_retrained_10epoch.txt') 