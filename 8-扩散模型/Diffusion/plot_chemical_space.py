"""
plot_chemical_space.py - 绘制化学空间t-SNE可视化
对比Diffusion无条件生成、CDiffusion条件生成、真实分子的化学空间分布
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from sklearn.manifold import TSNE
from tqdm import tqdm
import random
import os

# 禁用RDKit警告
RDLogger.DisableLog('rdApp.*')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def compute_ecfp_fingerprints(smiles_list, radius=2, n_bits=2048):
    """计算ECFP指纹（Morgan指纹）"""
    fingerprints = []
    valid_smiles = []

    for smi in tqdm(smiles_list, desc="  计算指纹"):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                # ECFP4 = radius 2
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                fingerprints.append(np.array(fp))
                valid_smiles.append(smi)
            except:
                continue

    return np.array(fingerprints), valid_smiles


def load_molecules(file_path, sample_size=None):
    """加载分子数据"""
    print(f"\n加载: {file_path}")

    if not os.path.exists(file_path):
        print(f"  警告: 文件不存在")
        return []

    df = pd.read_csv(file_path)

    smiles_col = 'SMILES' if 'SMILES' in df.columns else 'smiles'
    smiles_list = df[smiles_col].tolist()

    # 随机采样
    if sample_size and len(smiles_list) > sample_size:
        random.seed(42)
        smiles_list = random.sample(smiles_list, sample_size)
        print(f"  随机采样: {sample_size} / {len(df)}")
    else:
        print(f"  分子数: {len(smiles_list)}")

    return smiles_list


def plot_tsne_chemical_space(fps_dict, output_path='chemical_space_tsne.png'):
    """
    绘制t-SNE化学空间

    Args:
        fps_dict: {label: fingerprints} 字典
        output_path: 输出路径
    """
    print("\n准备t-SNE降维...")

    # 合并所有指纹
    all_fps = []
    all_labels = []

    for label, fps in fps_dict.items():
        if len(fps) > 0:
            all_fps.append(fps)
            all_labels.extend([label] * len(fps))

    if len(all_fps) == 0:
        print("错误: 没有有效的指纹数据")
        return

    all_fps = np.vstack(all_fps)
    print(f"  总指纹数: {len(all_fps)}")

    # t-SNE降维
    print("\n运行t-SNE（可能需要几分钟）...")
    perplexity = min(30, len(all_fps) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000, verbose=1)
    coords = tsne.fit_transform(all_fps)

    # 绘制化学空间
    print("\n绘制化学空间...")
    fig, ax = plt.subplots(figsize=(12, 10))

    # 定义颜色（对比明显）
    colors = {
        'Diffusion (Unconditional)': '#1E90FF',  # 道奇蓝
        'CDiffusion (Conditional)': '#FF4500',   # 橙红色
        'Real (ZINC)': '#32CD32'                 # 酸橙绿
    }

    # 绘制每个数据集
    start_idx = 0
    for label, fps in fps_dict.items():
        if len(fps) == 0:
            continue

        end_idx = start_idx + len(fps)
        coords_subset = coords[start_idx:end_idx]

        ax.scatter(
            coords_subset[:, 0],
            coords_subset[:, 1],
            c=colors.get(label, '#808080'),
            alpha=0.6,
            s=30,
            label=f'{label} (n={len(fps)})',
            edgecolors='black',
            linewidth=0.3
        )

        start_idx = end_idx

    ax.set_xlabel('t-SNE Dimension 1', fontsize=13, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=13, fontweight='bold')
    ax.set_title('Chemical Space Distribution (ECFP4 + t-SNE)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'\n化学空间图已保存: {output_path}')
    plt.close()


def main():
    print("=" * 60)
    print("化学空间t-SNE可视化")
    print("=" * 60)

    # 文件路径
    diffusion_file = 'generated/generated_diffusion.csv'
    cdiffusion_file = 'generated/generated_cdiffusion.csv'
    real_file = 'zinc_250k.csv'

    # 加载分子（真实数据采样1000个）
    diffusion_smiles = load_molecules(diffusion_file)
    cdiffusion_smiles = load_molecules(cdiffusion_file)
    real_smiles = load_molecules(real_file, sample_size=1000)

    # 检查是否有足够的数据
    if not diffusion_smiles and not cdiffusion_smiles:
        print("\n错误: 没有找到生成的分子数据")
        print("请先运行 generate.py 生成分子")
        return

    if not real_smiles:
        print("\n错误: 没有找到真实分子数据")
        return

    # 计算ECFP指纹
    print("\n" + "=" * 60)
    print("计算ECFP4指纹")
    print("=" * 60)

    fps_dict = {}

    if diffusion_smiles:
        print("\nDiffusion无条件生成:")
        diffusion_fps, diffusion_valid = compute_ecfp_fingerprints(diffusion_smiles, radius=2, n_bits=2048)
        print(f"  有效: {len(diffusion_fps)}/{len(diffusion_smiles)}")
        if len(diffusion_fps) > 0:
            fps_dict['Diffusion (Unconditional)'] = diffusion_fps

    if cdiffusion_smiles:
        print("\nCDiffusion条件生成:")
        cdiffusion_fps, cdiffusion_valid = compute_ecfp_fingerprints(cdiffusion_smiles, radius=2, n_bits=2048)
        print(f"  有效: {len(cdiffusion_fps)}/{len(cdiffusion_smiles)}")
        if len(cdiffusion_fps) > 0:
            fps_dict['CDiffusion (Conditional)'] = cdiffusion_fps

    print("\n真实分子:")
    real_fps, real_valid = compute_ecfp_fingerprints(real_smiles, radius=2, n_bits=2048)
    print(f"  有效: {len(real_fps)}/{len(real_smiles)}")
    if len(real_fps) > 0:
        fps_dict['Real (ZINC)'] = real_fps

    # 绘制t-SNE图
    print("\n" + "=" * 60)
    plot_tsne_chemical_space(fps_dict, output_path='chemical_space_tsne.png')

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
