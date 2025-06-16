import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import importlib.util
import sys

# 1. 동적 import: LeNet 불러오기
spec_model = importlib.util.spec_from_file_location("model_lenet", "02_250610_model_lenet.py")
model_lenet = importlib.util.module_from_spec(spec_model)
sys.modules["model_lenet"] = model_lenet
spec_model.loader.exec_module(model_lenet)
LeNet = model_lenet.LeNet

# -------------------------------
# 2. CW-L2 공격 함수
# -------------------------------
def cw_attack(model, x, y, c, loss_fn, max_iter=1000, lr=0.01, targeted=False):
    # x: (B, 1, 28, 28), y: (B,)
    device = x.device
    x_adv = x.clone().detach()
    # CW 논문 방식: 최적화 변수 w로 변환 (arctanh)
    x_var = torch.clamp(x_adv, 1e-6, 1-1e-6)
    w = torch.atanh(2 * x_var - 1)
    w = w.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([w], lr=lr)
    num_classes = 10

    # mask: 이미 성공한 샘플은 그대로 두기
    mask = torch.ones(x.size(0), dtype=torch.bool, device=device)
    best_adv = x_adv.clone().detach()

    for _ in range(max_iter):
        x_adv_iter = (torch.tanh(w) + 1) / 2  # [0,1]로 변환
        output = model(x_adv_iter)
        loss1 = F.mse_loss(x_adv_iter, x, reduction='sum')  # L2 거리
        real = output.gather(1, y.unsqueeze(1)).squeeze(1)
        other, _ = torch.max(output + torch.eye(num_classes, device=device)[y]*-1e4, dim=1)
        f_term = torch.clamp(real - other + 0, min=0)  # 다른 클래스가 더 크도록
        loss2 = (c * f_term).sum()
        loss = loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 중간 결과에서 성공한 샘플 기록
        pred = output.argmax(dim=1)
        success = (pred != y)
        # best_adv 갱신
        best_adv[success & mask] = x_adv_iter[success & mask].detach()
        mask[success] = False
        if not mask.any():  # 모두 성공하면 중단
            break

        # 다음 iter에서는 실패한 샘플만 남김 (효율 최적화, 선택)
        # 아래 주석을 풀면, 실패 샘플만 계속 갱신
        # if mask.any():
        #     w = w[mask].detach().clone().requires_grad_(True)
        #     x = x[mask]
        #     y = y[mask]

    x_adv_final = best_adv
    return x_adv_final.detach()

# -------------------------------
# 3. 평가 및 데이터 수집 함수
# -------------------------------
def evaluate_and_collect_cw(model, data_loader, device, c, loss_fn, max_iter=1000):
    model.eval()
    adv_images, adv_labels, adv_preds = [], [], []
    remain_images, remain_labels = [], []
    for x, y in tqdm(data_loader, desc=f"CW c={c}"):
        x, y = x.to(device), y.to(device)
        x_adv = cw_attack(model, x, y, c, loss_fn, max_iter=max_iter)
        with torch.no_grad():
            pred = model(x_adv).argmax(dim=1)
            mask = (pred != y)
            if mask.any():
                adv_images.append(x_adv[mask].cpu())
                adv_labels.append(y[mask].cpu())
                adv_preds.append(pred[mask].cpu())
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
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model = LeNet().to(device)
    model.load_state_dict(torch.load('03-1_250610_lenet_mnist.pth', map_location=device))
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    # 1. c=1
    adv_imgs_all, adv_labels_all, adv_preds_all = [], [], []
    adv_imgs, adv_labels, adv_preds, remain_imgs, remain_labels = evaluate_and_collect_cw(model, test_loader, device, c=1, loss_fn=loss_fn, max_iter=300)
    adv_imgs_all.append(adv_imgs)
    adv_labels_all.append(adv_labels)
    adv_preds_all.append(adv_preds)
    print(f"c=1: 공격 성공 {len(adv_imgs)}개, 남은 {len(remain_imgs)}개")

    # 2. c=5
    if len(remain_imgs) > 0:
        remain_loader = DataLoader(TensorDataset(remain_imgs, remain_labels), batch_size=64, shuffle=False)
        adv_imgs, adv_labels, adv_preds, remain_imgs, remain_labels = evaluate_and_collect_cw(model, remain_loader, device, c=5, loss_fn=loss_fn, max_iter=300)
        adv_imgs_all.append(adv_imgs)
        adv_labels_all.append(adv_labels)
        adv_preds_all.append(adv_preds)
        print(f"c=5: 공격 성공 {len(adv_imgs)}개, 남은 {len(remain_imgs)}개")

    # 3. c=10
    if len(remain_imgs) > 0:
        remain_loader = DataLoader(TensorDataset(remain_imgs, remain_labels), batch_size=64, shuffle=False)
        adv_imgs, adv_labels, adv_preds, remain_imgs, remain_labels = evaluate_and_collect_cw(model, remain_loader, device, c=10, loss_fn=loss_fn, max_iter=300)
        adv_imgs_all.append(adv_imgs)
        adv_labels_all.append(adv_labels)
        adv_preds_all.append(adv_preds)
        print(f"c=10: 공격 성공 {len(adv_imgs)}개, 남은 {len(remain_imgs)}개")

    # 누적된 공격 성공 데이터 저장 (중복 없이)
    adv_imgs_all = [x for x in adv_imgs_all if x.numel() > 0]
    adv_labels_all = [x for x in adv_labels_all if x.numel() > 0]
    adv_preds_all = [x for x in adv_preds_all if x.numel() > 0]
    if adv_imgs_all:
        adv_imgs_all = torch.cat(adv_imgs_all)
        adv_labels_all = torch.cat(adv_labels_all)
        adv_preds_all = torch.cat(adv_preds_all)
        torch.save({'images': adv_imgs_all, 'labels': adv_labels_all, 'preds': adv_preds_all}, 'cw_lenet_mnist.pt')
        print(f"공격 성공 데이터 {len(adv_imgs_all)}개를 cw_lenet_mnist.pt에 저장했습니다.")
    else:
        print("공격 성공 데이터가 없습니다.")

if __name__ == '__main__':
    main()
