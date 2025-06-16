import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import importlib.util
import sys

# 동적 import: 02_250610_model_lenet.py에서 LeNet 불러오기
spec_model = importlib.util.spec_from_file_location("model_lenet", "02_250610_model_lenet.py")
model_lenet = importlib.util.module_from_spec(spec_model)
sys.modules["model_lenet"] = model_lenet
spec_model.loader.exec_module(model_lenet)
LeNet = model_lenet.LeNet

# -------------------------------
# 2. FGSM 공격 함수
# -------------------------------
def fgsm_attack(model, x, y, epsilon, loss_fn):
    x_adv = x.clone().detach().requires_grad_(True)
    output = model(x_adv)
    loss = loss_fn(output, y)
    model.zero_grad()
    loss.backward()
    grad_sign = x_adv.grad.data.sign()
    x_adv = x_adv + epsilon * grad_sign
    x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv.detach()

def evaluate_and_collect(model, data_loader, device, epsilon, loss_fn):
    model.eval()
    adv_images, adv_labels, adv_preds = [], [], []
    remain_images, remain_labels = [], []
    for x, y in tqdm(data_loader, desc=f"FGSM eps={epsilon}"):
        x, y = x.to(device), y.to(device)
        x_adv = fgsm_attack(model, x, y, epsilon, loss_fn)
        with torch.no_grad():
            pred = model(x_adv).argmax(dim=1)
            # 오분류(공격 성공)만 저장
            mask = (pred != y)
            if mask.any():
                adv_images.append(x_adv[mask].cpu())
                adv_labels.append(y[mask].cpu())
                adv_preds.append(pred[mask].cpu())
            # 아직 분류 성공한(공격 실패) 데이터만 다음 단계로 넘김
            if (~mask).any():
                remain_images.append(x[~mask].cpu())
                remain_labels.append(y[~mask].cpu())
    if adv_images:
        adv_images = torch.cat(adv_images)
        adv_labels = torch.cat(adv_labels)
        adv_preds = torch.cat(adv_preds)
    else:
        adv_images = torch.empty(0)
        adv_labels = torch.empty(0, dtype=torch.long)
        adv_preds = torch.empty(0, dtype=torch.long)
    if remain_images:
        remain_images = torch.cat(remain_images)
        remain_labels = torch.cat(remain_labels)
    else:
        remain_images = torch.empty(0)
        remain_labels = torch.empty(0, dtype=torch.long)
    return adv_images, adv_labels, adv_preds, remain_images, remain_labels

# -------------------------------
# 4. 메인 실행
# -------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터셋
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 모델 로드 (이미 학습된 모델이 있다고 가정)
    model = LeNet().to(device)
    model.load_state_dict(torch.load('03-1_250610_lenet_mnist.pth', map_location=device))
    model.eval()

    # 손실 함수
    loss_fn = nn.CrossEntropyLoss()

    # 1. epsilon=0.2 (test)
    adv_imgs_all, adv_labels_all, adv_preds_all = [], [], []
    adv_imgs, adv_labels, adv_preds, remain_imgs, remain_labels = evaluate_and_collect(model, test_loader, device, 0.2, loss_fn)
    adv_imgs_all.append(adv_imgs)
    adv_labels_all.append(adv_labels)
    adv_preds_all.append(adv_preds)
    print(f"[테스트셋] epsilon=0.2: 공격 성공 {len(adv_imgs)}개, 남은 {len(remain_imgs)}개")

    # 2. epsilon=0.3 (test)
    if len(remain_imgs) > 0:
        remain_loader = DataLoader(TensorDataset(remain_imgs, remain_labels), batch_size=128, shuffle=False)
        adv_imgs, adv_labels, adv_preds, remain_imgs, remain_labels = evaluate_and_collect(model, remain_loader, device, 0.3, loss_fn)
        adv_imgs_all.append(adv_imgs)
        adv_labels_all.append(adv_labels)
        adv_preds_all.append(adv_preds)
        print(f"[테스트셋] epsilon=0.3: 공격 성공 {len(adv_imgs)}개, 남은 {len(remain_imgs)}개")

    # 3. epsilon=0.4 (test)
    if len(remain_imgs) > 0:
        remain_loader = DataLoader(TensorDataset(remain_imgs, remain_labels), batch_size=128, shuffle=False)
        adv_imgs, adv_labels, adv_preds, remain_imgs, remain_labels = evaluate_and_collect(model, remain_loader, device, 0.4, loss_fn)
        adv_imgs_all.append(adv_imgs)
        adv_labels_all.append(adv_labels)
        adv_preds_all.append(adv_preds)
        print(f"[테스트셋] epsilon=0.4: 공격 성공 {len(adv_imgs)}개, 남은 {len(remain_imgs)}개")

    # 누적된 공격 성공 데이터 저장 (test)
    adv_imgs_all = [x for x in adv_imgs_all if x.numel() > 0]
    adv_labels_all = [x for x in adv_labels_all if x.numel() > 0]
    adv_preds_all = [x for x in adv_preds_all if x.numel() > 0]
    if adv_imgs_all:
        adv_imgs_all = torch.cat(adv_imgs_all)
        adv_labels_all = torch.cat(adv_labels_all)
        adv_preds_all = torch.cat(adv_preds_all)
        torch.save({'images': adv_imgs_all, 'labels': adv_labels_all, 'preds': adv_preds_all}, '06-1_250610_fgsm_lenet_mnist.pt')
        print(f"[테스트셋] 공격 성공 데이터 {len(adv_imgs_all)}개를 06-1_250610_fgsm_lenet_mnist.pt에 저장했습니다.")
    else:
        print("[테스트셋] 공격 성공 데이터가 없습니다.")

    # (2) 학습셋 공격 (추가)
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    adv_imgs_all, adv_labels_all, adv_preds_all = [], [], []
    adv_imgs, adv_labels, adv_preds, remain_imgs, remain_labels = evaluate_and_collect(model, train_loader, device, 0.2, loss_fn)
    adv_imgs_all.append(adv_imgs)
    adv_labels_all.append(adv_labels)
    adv_preds_all.append(adv_preds)
    print(f"[학습셋] epsilon=0.2: 공격 성공 {len(adv_imgs)}개, 남은 {len(remain_imgs)}개")

    if len(remain_imgs) > 0:
        remain_loader = DataLoader(TensorDataset(remain_imgs, remain_labels), batch_size=128, shuffle=False)
        adv_imgs, adv_labels, adv_preds, remain_imgs, remain_labels = evaluate_and_collect(model, remain_loader, device, 0.3, loss_fn)
        adv_imgs_all.append(adv_imgs)
        adv_labels_all.append(adv_labels)
        adv_preds_all.append(adv_preds)
        print(f"[학습셋] epsilon=0.3: 공격 성공 {len(adv_imgs)}개, 남은 {len(remain_imgs)}개")

    if len(remain_imgs) > 0:
        remain_loader = DataLoader(TensorDataset(remain_imgs, remain_labels), batch_size=128, shuffle=False)
        adv_imgs, adv_labels, adv_preds, remain_imgs, remain_labels = evaluate_and_collect(model, remain_loader, device, 0.4, loss_fn)
        adv_imgs_all.append(adv_imgs)
        adv_labels_all.append(adv_labels)
        adv_preds_all.append(adv_preds)
        print(f"[학습셋] epsilon=0.4: 공격 성공 {len(adv_imgs)}개, 남은 {len(remain_imgs)}개")

    adv_imgs_all = [x for x in adv_imgs_all if x.numel() > 0]
    adv_labels_all = [x for x in adv_labels_all if x.numel() > 0]
    adv_preds_all = [x for x in adv_preds_all if x.numel() > 0]
    if adv_imgs_all:
        adv_imgs_all = torch.cat(adv_imgs_all)
        adv_labels_all = torch.cat(adv_labels_all)
        adv_preds_all = torch.cat(adv_preds_all)
        torch.save({'images': adv_imgs_all, 'labels': adv_labels_all, 'preds': adv_preds_all}, '06-1_250610_fgsm_lenet_mnist_train.pt')
        print(f"[학습셋] 공격 성공 데이터 {len(adv_imgs_all)}개를 06-1_250610_fgsm_lenet_mnist_train.pt에 저장했습니다.")
    else:
        print("[학습셋] 공격 성공 데이터가 없습니다.")

if __name__ == '__main__':
    main()
