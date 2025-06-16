import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from torchvision import datasets, transforms
import os
import time
import logging
import importlib.util
import sys

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# MNIST label을 Tensor로 변환하는 래퍼
class MNISTTensorLabelWrapper(torch.utils.data.Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
    def __len__(self):
        return len(self.mnist_dataset)
    def __getitem__(self, idx):
        img, label = self.mnist_dataset[idx]
        return img, torch.tensor(label, dtype=torch.long)

# 데이터셋 로드 함수
def load_tensor_dataset(dataset_path):
    logger.info(f"Loading dataset from {dataset_path}")
    data = torch.load(dataset_path)
    logger.info(f"Loaded keys: {list(data.keys())}")
    images = data['images']
    labels = data['labels']
    return TensorDataset(images, labels)

# 학습 함수
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / len(train_loader), 100. * correct / total

# 검증 함수
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return val_loss / len(val_loader), 100. * correct / total

def run_retrain(num_epochs, pth_path, txt_path):
    logger = Logger(txt_path)
    logprint = logger.logprint
    # 2. 데이터셋 준비
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # MNIST
    mnist_full = MNISTTensorLabelWrapper(datasets.MNIST(root='./data', train=True, download=True, transform=transform))
    N = len(mnist_full)
    val_size = N // 6
    train_size = N - val_size
    mnist_train, mnist_val = torch.utils.data.random_split(
        mnist_full, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    # FGSM train 데이터 로드
    fgsm_train = torch.load('06-2_250610_fgsm_lenet_mnist.pt')
    fgsm_train_imgs = fgsm_train['images']
    fgsm_train_labels = fgsm_train['labels']
    fgsm_train_dataset = TensorDataset(fgsm_train_imgs, fgsm_train_labels)
    # CW train 데이터 로드
    cw_train = torch.load('08c-3_250614_cw_lenet_mnist_train.pt')
    cw_train_imgs = cw_train['images'][:50000]
    cw_train_labels = cw_train['labels'][:50000]
    cw_train_dataset = TensorDataset(cw_train_imgs, cw_train_labels)
    # CW train split
    cw_N = len(cw_train_dataset)
    cw_val_size = cw_N // 6
    cw_train_size = cw_N - cw_val_size
    cw_train_split, cw_val_split = torch.utils.data.random_split(
        cw_train_dataset, [cw_train_size, cw_val_size], generator=torch.Generator().manual_seed(42)
    )
    # FGSM split (동일 비율)
    fgsm_N = len(fgsm_train_dataset)
    fgsm_val_size = fgsm_N // 6
    fgsm_train_size = fgsm_N - fgsm_val_size
    fgsm_train_split, fgsm_val_split = torch.utils.data.random_split(
        fgsm_train_dataset, [fgsm_train_size, fgsm_val_size], generator=torch.Generator().manual_seed(42)
    )
    # 최종 train/val dataset 합치기 (순서대로, shuffle 없이)
    final_train = ConcatDataset([mnist_train, fgsm_train_split, cw_train_split])
    final_val = ConcatDataset([mnist_val, fgsm_val_split, cw_val_split])
    # test set: MNIST test + FGSM test + CW test
    mnist_test = MNISTTensorLabelWrapper(datasets.MNIST(root='./data', train=False, download=True, transform=transform))
    fgsm_test = torch.load('06-2_250610_fgsm_lenet_mnist.pt')
    fgsm_test_imgs = fgsm_test['images']
    fgsm_test_labels = fgsm_test['labels']
    fgsm_test_dataset = TensorDataset(fgsm_test_imgs, fgsm_test_labels)
    cw_test = torch.load('08c-2_250610_cw_lenet_mnist.pt')
    cw_test_imgs = cw_test['images']
    cw_test_labels = cw_test['labels']
    cw_test_dataset = TensorDataset(cw_test_imgs, cw_test_labels)
    final_test = ConcatDataset([mnist_test, fgsm_test_dataset, cw_test_dataset])
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
    run_retrain(5, '1003-1_250616_lenet_mnist_all_retrained_5epoch.pth', '1003-2_250616_lenet_mnist_all_retrained_5epoch.txt')
    run_retrain(10, '1003-3_250616_lenet_mnist_all_retrained_10epoch.pth', '1003-4_250616_lenet_mnist_all_retrained_10epoch.txt') 