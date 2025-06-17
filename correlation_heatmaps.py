import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# 결과를 저장할 디렉토리 생성
os.makedirs("correlation_heatmaps", exist_ok=True)

# CSV 파일 읽기
df = pd.read_csv("results/all_results.csv")

# 각 상관계수 타입에 대해 히트맵 생성
correlation_types = {
    'pearson': 'Pearson Correlation',
    'spearman': 'Spearman Correlation',
    'kendall': "Kendall's Tau Correlation"
}

for corr_type, title in correlation_types.items():
    # 상관계수 계산
    corr = df.corr(method=corr_type)
    
    # 일반 상관계수 히트맵
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, 
                annot=True, 
                fmt=".2f", 
                cmap="coolwarm",
                vmin=-1, 
                vmax=1,
                center=0)
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"correlation_heatmaps/{corr_type}_correlation_heatmap.png")
    plt.close()
    
    # 절대값 상관계수 히트맵
    plt.figure(figsize=(10, 8))
    sns.heatmap(np.abs(corr), 
                annot=True, 
                fmt=".2f", 
                cmap="YlOrRd",  # 절대값은 양수만 있으므로 다른 컬러맵 사용
                vmin=0, 
                vmax=1)
    
    plt.title(f"Absolute {title}")
    plt.tight_layout()
    plt.savefig(f"correlation_heatmaps/abs_{corr_type}_correlation_heatmap.png")
    plt.close()

print("모든 상관계수 히트맵이 correlation_heatmaps 폴더에 저장되었습니다.") 