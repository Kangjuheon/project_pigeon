import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import importlib.util
import sys

# 모델-데이터셋 매핑
model_dataset_map = {
    "03-1_250610_lenet_mnist.pth": "mnist_train.pt",
    "1001-1_250616_lenet_mnist_retrained_fgsm_5epoch.pth": "mnist_fgsm_train.pt",
    "1001-3_250616_lenet_mnist_retrained_fgsm_10epoch.pth": "mnist_fgsm_train.pt",
    "1002-1_250616_lenet_mnist_retrained_cw_5epoch.pth": "mnist_cw_train.pt",
    "1002-3_250616_lenet_mnist_retrained_cw_10epoch.pth": "mnist_cw_train.pt",
    "1003-1_250616_lenet_mnist_all_retrained_5epoch.pth": "mnist_fgsm_cw_train.pt",
    "1003-3_250616_lenet_mnist_all_retrained_10epoch.pth": "mnist_fgsm_cw_train.pt"
}

# LeNet 모델 불러오기
spec_model = importlib.util.spec_from_file_location("model_lenet", "02_250610_model_lenet.py")
model_lenet = importlib.util.module_from_spec(spec_model)
sys.modules["model_lenet"] = model_lenet
spec_model.loader.exec_module(model_lenet)
LeNet = model_lenet.LeNet

os.makedirs("neuron_stats", exist_ok=True)

for model_path, dataset_path in model_dataset_map.items():
    print(f"[INFO] {model_path} 기준: {dataset_path}")
    # 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 데이터 로드
    data = torch.load(dataset_path)
    images = data['images']
    batch_size = 128
    loader = DataLoader(images, batch_size=batch_size, shuffle=False)

    # 뉴런별 activation 저장용
    activations = {layer: [] for layer in ['conv1', 'conv2', 'fc1', 'fc2', 'fc3']}
    
    def get_hook(name):
        def hook(module, input, output):
            # (batch, ...)
            out = output.detach().cpu()
            # (batch, ...) -> (batch, 뉴런)
            if out.dim() > 2:
                out = out.view(out.size(0), -1)
            activations[name].append(out)
        return hook
    
    hooks = []
    hooks.append(model.conv1.register_forward_hook(get_hook('conv1')))
    hooks.append(model.conv2.register_forward_hook(get_hook('conv2')))
    hooks.append(model.fc1.register_forward_hook(get_hook('fc1')))
    hooks.append(model.fc2.register_forward_hook(get_hook('fc2')))
    hooks.append(model.fc3.register_forward_hook(get_hook('fc3')))

    # Forward pass
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{model_path} activations"):
            batch = batch.to(device)
            _ = model(batch)
    
    # 훅 제거
    for h in hooks:
        h.remove()
    
    # 뉴런별 평균, 분산, 신뢰구간 계산
    stats = {}
    confidence_boundaries = {}
    activation_maxs = {}
    for layer, acts in activations.items():
        acts_cat = torch.cat(acts, dim=0)  # (N, 뉴런)
        mean = acts_cat.mean(dim=0)
        std = acts_cat.std(dim=0)
        stats[layer] = {
            'mean': mean,
            'std': std
        }
        # 신뢰구간: 평균 + z*표준편차
        confidence_boundaries[layer] = {
            '95': mean + 1.645 * std,
            '99': mean + 2.326 * std,
            '99.99': mean + 3.891 * std
        }
        # max 활성도
        activation_maxs[layer] = acts_cat.max(dim=0)[0]
    
    # 전체 저장
    save_dict = {
        'mean_std': stats,
        'confidence_boundaries': confidence_boundaries,
        'activation_maxs': activation_maxs
    }
    out_path = os.path.join("neuron_stats", f"neuron_stats_{os.path.splitext(model_path)[0]}.pt")
    torch.save(save_dict, out_path)
    print(f"[SAVE] {out_path}") 