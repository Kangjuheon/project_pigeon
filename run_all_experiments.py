import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from torchvision import datasets, transforms
import os
import torch
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import concurrent.futures
from tqdm import tqdm
import importlib.util
import sys
import numpy as np
from art.estimators.classification import PyTorchClassifier
from art.metrics import clever_u
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from multiprocessing import Pool, cpu_count

# LeNet 모델 불러오기
spec_model = importlib.util.spec_from_file_location("model_lenet", "02_model_lenet.py")
model_lenet = importlib.util.module_from_spec(spec_model)
sys.modules["model_lenet"] = model_lenet
spec_model.loader.exec_module(model_lenet)
LeNet = model_lenet.LeNet

def get_spectral_norm(module):
    W = module.weight.data
    W_mat = W.reshape(W.size(0), -1)
    sigma = torch.linalg.norm(W_mat, ord=2)
    return sigma.item()

def get_lenet_lipschitz(model):
    layers = [model.conv1, model.conv2, model.fc1, model.fc2, model.fc3]
    L = 1.0
    for layer in layers:
        L *= get_spectral_norm(layer)
    return L

def snac_hook_fn(name, snac_hits_95, snac_hits_99, snac_hits_9999, snac_hits_origin, conf95, conf99, conf9999, max_origin):
    def hook(module, input, output):
        out = output.detach().cpu()
        # 기준값 텐서의 크기를 출력 텐서에 맞게 조정
        if name.startswith('conv'):
            conf95_tensor = conf95[name].view(-1).to(out.device)
            conf99_tensor = conf99[name].view(-1).to(out.device)
            conf9999_tensor = conf9999[name].view(-1).to(out.device)
            max_origin_tensor = max_origin[name].view(-1).to(out.device)
        else:
            conf95_tensor = conf95[name].to(out.device)
            conf99_tensor = conf99[name].to(out.device)
            conf9999_tensor = conf9999[name].to(out.device)
            max_origin_tensor = max_origin[name].to(out.device)
        
        # 출력 텐서를 2D로 변환 (batch_size x features)
        out_reshaped = out.view(out.size(0), -1)
        
        # SNAC 계산
        over_95 = out_reshaped > conf95_tensor
        snac_hits_95[name] |= torch.any(over_95, dim=0)
        over_99 = out_reshaped > conf99_tensor
        snac_hits_99[name] |= torch.any(over_99, dim=0)
        over_9999 = out_reshaped > conf9999_tensor
        snac_hits_9999[name] |= torch.any(over_9999, dim=0)
        over_origin = out_reshaped > max_origin_tensor
        snac_hits_origin[name] |= torch.any(over_origin, dim=0)
    return hook

def collect_snac(snac_hits, label, snac_results):
    snac_results[label] = {}
    total = 0
    covered = 0
    for layer_name, hit in snac_hits.items():
        n_total = hit.numel()
        n_covered = hit.sum().item()
        total += n_total
        covered += n_covered
        snac_results[label][layer_name] = 100 * n_covered / n_total if n_total > 0 else 0
    snac_results[label]['[Total SNAC]'] = 100 * covered / total if total > 0 else 0

# 1. 모델 리스트
models = [
    "03-1_250610_lenet_mnist.pth",
    "1001-1_250616_lenet_mnist_retrained_fgsm_5epoch.pth",
    "1001-3_250616_lenet_mnist_retrained_fgsm_10epoch.pth",
    "1002-1_250616_lenet_mnist_retrained_cw_5epoch.pth",
    "1002-3_250616_lenet_mnist_retrained_cw_10epoch.pth",
    "1003-1_250616_lenet_mnist_all_retrained_5epoch.pth",
    "1003-3_250616_lenet_mnist_all_retrained_10epoch.pth"
]

# 2. 데이터셋 리스트 (train/test/all 세트, 조합 포함)
dataset_files = {
    "mnist_train": "mnist_train.pt",
    "mnist_test": "mnist_test.pt",
    "mnist_all": "mnist_all.pt",
    "fgsm_train": "06-1_250610_fgsm_lenet_mnist_train.pt",
    "fgsm_test": "06-2_250610_fgsm_lenet_mnist.pt",
    "fgsm_all": "fgsm_all.pt",
    "cw_train": "08c-3_250614_cw_lenet_mnist_train.pt",
    "cw_test": "08c-2_250610_cw_lenet_mnist.pt",
    "cw_all": "cw_all.pt",
    "mnist+fgsm_train": "mnist_fgsm_train.pt",
    "mnist+fgsm_test": "mnist_fgsm_test.pt",
    "mnist+fgsm_all": "mnist_fgsm_all.pt",
    "mnist+cw_train": "mnist_cw_train.pt",
    "mnist+cw_test": "mnist_cw_test.pt",
    "mnist+cw_all": "mnist_cw_all.pt",
    "fgsm+cw_train": "fgsm_cw_train.pt",
    "fgsm+cw_test": "fgsm_cw_test.pt",
    "fgsm+cw_all": "fgsm_cw_all.pt",
    "mnist+fgsm+cw_train": "mnist_fgsm_cw_train.pt",
    "mnist+fgsm+cw_test": "mnist_fgsm_cw_test.pt",
    "mnist+fgsm+cw_all": "mnist_fgsm_cw_all.pt"
}

os.makedirs("results", exist_ok=True)

def save_pt(data, path):
    torch.save(data, path)
    print(f"[CREATE] {path}")

def load_mnist_pt(split):
    # split: 'train' or 'test'
    ds = datasets.MNIST(root='./data', train=(split=='train'), download=True, transform=transforms.ToTensor())
    imgs = torch.stack([img for img, _ in ds])
    labels = torch.tensor([label for _, label in ds])
    return {'images': imgs, 'labels': labels}

def load_pt(path):
    return torch.load(path)

def combine_and_save(paths, out_path):
    datas = [load_pt(p) if isinstance(p, str) else p for p in paths]
    imgs = torch.cat([d['images'] for d in datas], dim=0)
    labels = torch.cat([d['labels'] for d in datas], dim=0)
    save_pt({'images': imgs, 'labels': labels}, out_path)

def ensure_dataset(ds_name, ds_path):
    if os.path.exists(ds_path):
        return
    # MNIST 단일
    if ds_name == "mnist_train":
        save_pt(load_mnist_pt('train'), ds_path)
    elif ds_name == "mnist_test":
        save_pt(load_mnist_pt('test'), ds_path)
    elif ds_name == "mnist_all":
        combine_and_save(["mnist_train.pt", "mnist_test.pt"], ds_path)
    elif ds_name == "fgsm_all":
        combine_and_save(["06-1_250610_fgsm_lenet_mnist_train.pt", "06-2_250610_fgsm_lenet_mnist.pt"], ds_path)
    elif ds_name == "cw_all":
        combine_and_save(["08c-3_250614_cw_lenet_mnist_train.pt", "08c-2_250610_cw_lenet_mnist.pt"], ds_path)
    # 조합
    elif ds_name == "mnist+fgsm_train":
        combine_and_save(["mnist_train.pt", "06-1_250610_fgsm_lenet_mnist_train.pt"], ds_path)
    elif ds_name == "mnist+fgsm_test":
        combine_and_save(["mnist_test.pt", "06-2_250610_fgsm_lenet_mnist.pt"], ds_path)
    elif ds_name == "mnist+fgsm_all":
        combine_and_save(["mnist_all.pt", "fgsm_all.pt"], ds_path)
    elif ds_name == "mnist+cw_train":
        combine_and_save(["mnist_train.pt", "08c-3_250614_cw_lenet_mnist_train.pt"], ds_path)
    elif ds_name == "mnist+cw_test":
        combine_and_save(["mnist_test.pt", "08c-2_250610_cw_lenet_mnist.pt"], ds_path)
    elif ds_name == "mnist+cw_all":
        combine_and_save(["mnist_all.pt", "cw_all.pt"], ds_path)
    elif ds_name == "fgsm+cw_train":
        combine_and_save(["06-1_250610_fgsm_lenet_mnist_train.pt", "08c-3_250614_cw_lenet_mnist_train.pt"], ds_path)
    elif ds_name == "fgsm+cw_test":
        combine_and_save(["06-2_250610_fgsm_lenet_mnist.pt", "08c-2_250610_cw_lenet_mnist.pt"], ds_path)
    elif ds_name == "fgsm+cw_all":
        combine_and_save(["fgsm_all.pt", "cw_all.pt"], ds_path)
    elif ds_name == "mnist+fgsm+cw_train":
        combine_and_save(["mnist_train.pt", "06-1_250610_fgsm_lenet_mnist_train.pt", "08c-3_250614_cw_lenet_mnist_train.pt"], ds_path)
    elif ds_name == "mnist+fgsm+cw_test":
        combine_and_save(["mnist_test.pt", "06-2_250610_fgsm_lenet_mnist.pt", "08c-2_250610_cw_lenet_mnist.pt"], ds_path)
    elif ds_name == "mnist+fgsm+cw_all":
        combine_and_save(["mnist_all.pt", "fgsm_all.pt", "cw_all.pt"], ds_path)
    else:
        print(f"[SKIP] 알 수 없는 데이터셋 조합: {ds_name}")

class DictTensorDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.images = data['images']
        self.labels = data['labels']
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def get_activations(model, data_loader, device):
    activations = []
    
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            activations.append(output.detach().cpu())
    
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn))
    
    model.eval()
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
    
    for hook in hooks:
        hook.remove()
    
    return activations

def calculate_snac(activations, neuron_stats, device):
    # 모든 레이어의 활성도를 하나의 벡터로 합침
    all_activations = torch.cat([act.view(act.size(0), -1) for act in activations], dim=1)
    
    # neuron_stats에서 모든 레이어의 통계를 하나의 벡터로 합침
    mean = torch.cat([stats['mean'] for stats in neuron_stats.values()])
    std = torch.cat([stats['std'] for stats in neuron_stats.values()])
    max_activations = torch.cat([stats['max_activations'] for stats in neuron_stats.values()])
    conf_95 = torch.cat([stats['conf_95'] for stats in neuron_stats.values()])
    conf_99 = torch.cat([stats['conf_99'] for stats in neuron_stats.values()])
    conf_9999 = torch.cat([stats['conf_9999'] for stats in neuron_stats.values()])
    
    # 각 기준점을 초과하는 뉴런의 비율 계산
    snac_origin = (all_activations > max_activations).float().mean().item()
    snac_95 = (all_activations > conf_95).float().mean().item()
    snac_99 = (all_activations > conf_99).float().mean().item()
    snac_9999 = (all_activations > conf_9999).float().mean().item()
    
    return {
        'SNAC_origin': snac_origin,
        'SNAC_95': snac_95,
        'SNAC_99': snac_99,
        'SNAC_9999': snac_9999
    }

def calculate_clever(model, data_loader, device):
    # CLEVER 계산 로직
    # Lipschitz constant, CL1, CL2, CLi 계산
    # 실제 구현은 CLEVER 논문의 방법론을 따르도록 해야 함
    return {
        'Lipschitz_constant': 0.0,  # 임시값
        'CL1': 0.0,  # 임시값
        'CL2': 0.0,  # 임시값
        'CLi': 0.0   # 임시값
    }

def run_metric(model_path, dataset_path, neuron_stats_path, device_str):
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    print(f"\n[Processing] Model: {model_name} | Dataset: {dataset_name}")
    
    device = torch.device(device_str)
    model = LeNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 데이터셋 로드
    data = torch.load(dataset_path)
    images = data['images']
    labels = data['labels']
    
    # Load neuron statistics
    neuron_stats = torch.load(neuron_stats_path)
    
    # Get confidence boundaries
    conf95 = {k: v['95'] for k, v in neuron_stats['confidence_boundaries'].items()}
    conf99 = {k: v['99'] for k, v in neuron_stats['confidence_boundaries'].items()}
    conf9999 = {k: v['99.99'] for k, v in neuron_stats['confidence_boundaries'].items()}
    max_activations = neuron_stats['activation_maxs']
    
    # SNAC 계산
    snac_hits_95 = {k: torch.zeros_like(v, dtype=torch.bool) for k, v in conf95.items()}
    snac_hits_99 = {k: torch.zeros_like(v, dtype=torch.bool) for k, v in conf99.items()}
    snac_hits_9999 = {k: torch.zeros_like(v, dtype=torch.bool) for k, v in conf9999.items()}
    snac_hits_origin = {k: torch.zeros_like(v, dtype=torch.bool) for k, v in max_activations.items()}
    
    # 훅 등록
    hooks = []
    hooks.append(model.conv1.register_forward_hook(snac_hook_fn('conv1', snac_hits_95, snac_hits_99, snac_hits_9999, snac_hits_origin, conf95, conf99, conf9999, max_activations)))
    hooks.append(model.conv2.register_forward_hook(snac_hook_fn('conv2', snac_hits_95, snac_hits_99, snac_hits_9999, snac_hits_origin, conf95, conf99, conf9999, max_activations)))
    hooks.append(model.fc1.register_forward_hook(snac_hook_fn('fc1', snac_hits_95, snac_hits_99, snac_hits_9999, snac_hits_origin, conf95, conf99, conf9999, max_activations)))
    hooks.append(model.fc2.register_forward_hook(snac_hook_fn('fc2', snac_hits_95, snac_hits_99, snac_hits_9999, snac_hits_origin, conf95, conf99, conf9999, max_activations)))
    hooks.append(model.fc3.register_forward_hook(snac_hook_fn('fc3', snac_hits_95, snac_hits_99, snac_hits_9999, snac_hits_origin, conf95, conf99, conf9999, max_activations)))
    
    # 데이터셋 순회하며 SNAC 계산
    batch_size = 128
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            _ = model(batch)
    
    # 훅 제거
    for h in hooks:
        h.remove()
    
    # SNAC coverage 계산
    snac_results = {}
    collect_snac(snac_hits_95, '95', snac_results)
    collect_snac(snac_hits_99, '99', snac_results)
    collect_snac(snac_hits_9999, '99.99', snac_results)
    collect_snac(snac_hits_origin, 'origin', snac_results)
    
    # Lipschitz constant 계산
    lipschitz_val = get_lenet_lipschitz(model)
    
    # CLEVER score 계산 (10개 샘플 평균)
    clever_scores_l1 = []
    clever_scores_l2 = []
    clever_scores_linf = []
    clever_success = False
    try:
        classifier = PyTorchClassifier(
            model=model,
            loss=torch.nn.CrossEntropyLoss(),
            input_shape=(1, 28, 28),
            nb_classes=10,
            clip_values=(0.0, 1.0),
        )
        for i in range(10):
            if i >= len(images):
                break
            x = images[i].cpu().numpy()
            score_l1 = clever_u(classifier, x, nb_batches=10, batch_size=10, radius=10.0, norm=1, pool_factor=20)
            score_l2 = clever_u(classifier, x, nb_batches=10, batch_size=10, radius=10.0, norm=2, pool_factor=20)
            score_linf = clever_u(classifier, x, nb_batches=10, batch_size=10, radius=10.0, norm=np.inf, pool_factor=20)
            clever_scores_l1.append(score_l1)
            clever_scores_l2.append(score_l2)
            clever_scores_linf.append(score_linf)
        clever_success = True
    except Exception as e:
        clever_err_msg = str(e)
    
    # 결과 저장
    results = {
        'SNAC_95': snac_results['95']['[Total SNAC]'],
        'SNAC_99': snac_results['99']['[Total SNAC]'],
        'SNAC_9999': snac_results['99.99']['[Total SNAC]'],
        'SNAC_origin': snac_results['origin']['[Total SNAC]'],
        'Lipschitz_constant': lipschitz_val,
        'CL1': np.mean(clever_scores_l1) if clever_success else 0.0,
        'CL2': np.mean(clever_scores_l2) if clever_success else 0.0,
        'CLi': np.mean(clever_scores_linf) if clever_success else 0.0
    }
    
    return results

def run_metric_wrapper(args):
    model_path, dataset_path, neuron_stats_path, device_str = args
    return run_metric(model_path, dataset_path, neuron_stats_path, device_str)

if __name__ == '__main__':
    # 3. metric 결과 생성 (302 실행) - 병렬화 + 진행상황 표시
    tasks = []
    valid_pairs = 0
    total_pairs = 0
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\n=== Preparing Model-Dataset Pairs ===")
    
    for model in models:
        model_path = model
        for ds_name, ds_path in dataset_files.items():
            dataset_path = ds_path
            total_pairs += 1
            
            # 데이터셋 파일이 없으면 생성
            if not os.path.exists(dataset_path):
                print(f"[Creating Dataset] {ds_name}")
                try:
                    ensure_dataset(ds_name, dataset_path)
                except Exception as e:
                    print(f"[ERROR] Failed to create dataset {ds_name}: {e}")
                    continue
            
            # 모델 파일이 없으면 스킵
            if not os.path.exists(model_path):
                print(f"[SKIP] Model file not found: {model_path}")
                continue
                
            valid_pairs += 1
            tasks.append((model_path, dataset_path, os.path.join("neuron_stats", f"neuron_stats_{os.path.splitext(model)[0]}.pt"), device_str))
    
    print(f"\n=== Starting Metric Calculations ===")
    print(f"Total possible pairs: {total_pairs}")
    print(f"Valid pairs: {valid_pairs}")
    print(f"Skipped pairs: {total_pairs - valid_pairs}")
    print("\nProcessing pairs (this may take a while)...")
    
    # 결과를 저장할 리스트
    all_results = []
    completed_pairs = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        for result in tqdm(executor.map(run_metric_wrapper, tasks), 
                         total=len(tasks), 
                         desc="Overall Progress",
                         unit="pair"):
            if result is not None:
                all_results.append(result)
                completed_pairs += 1
                print(f"\nCompleted {completed_pairs}/{valid_pairs} pairs")
    
    print("\n=== Metric Calculations Complete ===")
    print(f"Successfully processed: {completed_pairs}/{valid_pairs} pairs")
    
    # 결과를 DataFrame으로 변환하고 저장
    if all_results:
        os.makedirs("results", exist_ok=True)
        results_df = pd.DataFrame(all_results)
        results_df.to_csv('results/all_results.csv', index=False)
        print("\n결과가 results/all_results.csv 파일에 저장되었습니다.")
    
    # 4. 상관계수 분석 (310 실행)
    subprocess.run(["python", "310_250616_correlation_batch_runner.py"], check=True)

    # 5. 시각화 (각 csv에 대해 heatmap)
    csv_files = glob.glob("results/*.csv")
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if df.shape[1] > 1:
                corr = df.corr()
                plt.figure(figsize=(8, 6))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
                plt.title(f"Correlation Heatmap: {os.path.basename(csv_file)}")
                plt.tight_layout()
                plt.savefig(f"{csv_file.replace('.csv', '_heatmap.png')}")
                plt.close()
        except Exception as e:
            print(f"[ERROR] {csv_file}: {e}")
    print("모든 실험, 상관분석, 시각화 완료!") 