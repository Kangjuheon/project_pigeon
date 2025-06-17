import os
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
import itertools

# 분석할 csv 파일들이 있는 폴더
csv_dir = 'results'  # 또는 원하는 경로로 수정
out_dir = 'correlation_results'
os.makedirs(out_dir, exist_ok=True)

# 상관계수 계산 함수 (2001_250616_correlation_metrics.py와 동일)
def compute_correlations(x, y):
    pearson_corr, pearson_p = pearsonr(x, y)
    spearman_corr, spearman_p = spearmanr(x, y)
    kendall_corr, kendall_p = kendalltau(x, y)
    return {
        'pearson': (pearson_corr, pearson_p),
        'spearman': (spearman_corr, spearman_p),
        'kendall': (kendall_corr, kendall_p)
    }

def load_metrics_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    if df.columns[0].lower() in ['model', 'name', 'id']:
        df = df.set_index(df.columns[0])
    return df

# results 폴더 내 모든 csv 파일에 대해 실행
def main():
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    for csv_file in csv_files:
        csv_path = os.path.join(csv_dir, csv_file)
        df = load_metrics_from_csv(csv_path)
        metrics = df.columns.tolist()
        out_txt = os.path.join(out_dir, csv_file.replace('.csv', '_correlation.txt'))
        with open(out_txt, 'w', encoding='utf-8') as f:
            f.write(f"[INFO] Metrics: {metrics}\n")
            f.write("\n[상관계수 결과표]\n")
            for m1, m2 in itertools.combinations(metrics, 2):
                x, y = df[m1].values, df[m2].values
                results = compute_correlations(x, y)
                f.write(f"\n{m1} vs {m2}\n")
                for name, (corr, p) in results.items():
                    f.write(f"  {name.capitalize():8}: {corr:.4f} (p={p:.4g})\n")
        print(f"[DONE] {csv_file} -> {out_txt}")

if __name__ == "__main__":
    main() 