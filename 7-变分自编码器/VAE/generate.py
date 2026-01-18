"""
generate.py - Fragment-VAE/CVAE 分子生成脚本
VAE: 无条件生成（随机采样）
CVAE: 有条件生成（多种条件对比）
"""

import torch
import argparse
import pickle
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random

from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.QED import qed
from rdkit import RDLogger

from model import FragmentVAE, FragmentCVAE

# 禁用RDKit警告
RDLogger.DisableLog('rdApp.*')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_real_molecules(dataset_file, n_samples=1000):
    """加载真实数据集并计算性质"""
    print(f"\n加载真实数据集: {dataset_file}")
    df = pd.read_csv(dataset_file)

    smiles_col = 'SMILES' if 'SMILES' in df.columns else 'smiles'

    # 随机采样
    if len(df) > n_samples:
        df_sample = df.sample(n=n_samples, random_state=42)
    else:
        df_sample = df

    smiles_list = df_sample[smiles_col].tolist()

    # 计算性质
    results = []
    for smi in tqdm(smiles_list, desc="  计算真实分子性质"):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                results.append({
                    'SMILES': smi,
                    'logP': MolLogP(mol),
                    'QED': qed(mol)
                })
            except:
                continue

    df_result = pd.DataFrame(results)
    print(f"  - 有效分子: {len(df_result)}/{len(smiles_list)}")
    return df_result


def generate_molecules(model, dataset, num_samples, device, mode='vae',
                       logp=None, qed_target=None):
    """
    生成分子

    Args:
        model: VAE或CVAE模型
        dataset: 数据集（用于decode）
        num_samples: 目标生成的独特分子数量
        device: 设备
        mode: 'vae' 或 'cvae'
        logp: 目标logP值（CVAE模式）
        qed_target: 目标QED值（CVAE模式）
    """
    model.eval()
    results = []
    seen_smiles = set()
    total_attempts = 0
    success_count = 0
    fail_count = 0
    batch = 100

    if mode == 'cvae':
        print(f'条件: logP={logp}, QED={qed_target}')

    pbar = tqdm(total=num_samples, desc='生成分子', unit='个')

    with torch.no_grad():
        while len(results) < num_samples:
            # 生成
            if mode == 'vae':
                template_idx, sub_idx = model.sample(batch, device)
            else:
                template_idx, sub_idx = model.sample(batch, logp, qed_target, device)

            template_idx = template_idx.cpu()
            sub_idx = sub_idx.cpu()

            for i in range(batch):
                total_attempts += 1

                # 解码为SMILES
                smiles = dataset.decode(template_idx[i], sub_idx[i])

                # 验证分子
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    # 规范化SMILES
                    canonical_smi = Chem.MolToSmiles(mol)
                    if canonical_smi not in seen_smiles:
                        seen_smiles.add(canonical_smi)
                        success_count += 1
                        results.append({
                            'SMILES': canonical_smi,
                            'logP': MolLogP(mol),
                            'QED': qed(mol),
                            'NumAtoms': mol.GetNumAtoms()
                        })
                        pbar.update(1)
                        if len(results) >= num_samples:
                            break
                    else:
                        fail_count += 1
                else:
                    fail_count += 1

            # 更新状态
            rate = success_count / total_attempts * 100 if total_attempts > 0 else 0
            pbar.set_postfix({
                '尝试': total_attempts,
                '成功': success_count,
                '失败': fail_count,
                '成功率': f'{rate:.1f}%'
            })

    pbar.close()

    # 最终统计
    final_rate = success_count / total_attempts * 100 if total_attempts > 0 else 0
    print(f'\n生成完成: {success_count} 个独特分子')
    print(f'总尝试: {total_attempts}, 成功: {success_count}, 失败: {fail_count}, 成功率: {final_rate:.2f}%')

    return pd.DataFrame(results)


def visualize_svg(df, path, n=100, mols_per_row=10):
    """保存分子网格图为SVG格式"""
    from rdkit.Chem.Draw import rdMolDraw2D

    df_subset = df.head(n)
    mols = []
    legends = []

    for _, row in df_subset.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol:
            mols.append(mol)
            legends.append(f"logP:{row['logP']:.2f} QED:{row['QED']:.2f}")

    if not mols:
        print('警告: 没有有效分子可绘制')
        return

    n_mols = len(mols)
    n_cols = min(mols_per_row, n_mols)
    n_rows = (n_mols + n_cols - 1) // n_cols

    mol_size = (250, 280)
    img_width = n_cols * mol_size[0]
    img_height = n_rows * mol_size[1]

    drawer = rdMolDraw2D.MolDraw2DSVG(img_width, img_height, mol_size[0], mol_size[1])

    opts = drawer.drawOptions()
    opts.legendFontSize = 20
    opts.annotationFontScale = 0.8

    drawer.DrawMolecules(mols, legends=legends)
    drawer.FinishDrawing()

    svg_text = drawer.GetDrawingText()
    with open(path, 'w') as f:
        f.write(svg_text)

    print(f'可视化: {path} ({len(mols)}个分子, SVG高清格式)')


def plot_cvae_comparison(df_real, generated_results, output_path='cvae_comparison.png'):
    """
    绘制CVAE条件生成对比图
    3个对比图放在同一画布，每个图显示QED和logP分布对比
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    colors = {
        'Real': '#32CD32',  # 绿色
        'Generated': '#DC143C'  # 红色
    }

    alpha = 0.6
    bins = 30

    for idx, (condition_name, df_gen) in enumerate(generated_results):
        # 左侧：QED分布
        ax_qed = axes[idx, 0]
        ax_qed.hist(df_real['QED'], bins=bins, alpha=alpha, color=colors['Real'],
                    label=f'Real (n={len(df_real)})', density=True, edgecolor='black', linewidth=0.5)
        ax_qed.hist(df_gen['QED'], bins=bins, alpha=alpha, color=colors['Generated'],
                    label=f'Generated (n={len(df_gen)})', density=True, edgecolor='black', linewidth=0.5)

        ax_qed.set_xlabel('QED', fontsize=11, fontweight='bold')
        ax_qed.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax_qed.set_title(f'{condition_name} - QED Distribution', fontsize=12, fontweight='bold')
        ax_qed.legend(fontsize=9)
        ax_qed.grid(True, alpha=0.3)
        ax_qed.set_xlim(0, 1)

        # 统计信息
        stats_text = f"Mean:\nReal: {df_real['QED'].mean():.3f}\nGen: {df_gen['QED'].mean():.3f}"
        ax_qed.text(0.02, 0.98, stats_text, transform=ax_qed.transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 右侧：logP分布
        ax_logp = axes[idx, 1]
        ax_logp.hist(df_real['logP'], bins=bins, alpha=alpha, color=colors['Real'],
                     label=f'Real (n={len(df_real)})', density=True, edgecolor='black', linewidth=0.5)
        ax_logp.hist(df_gen['logP'], bins=bins, alpha=alpha, color=colors['Generated'],
                     label=f'Generated (n={len(df_gen)})', density=True, edgecolor='black', linewidth=0.5)

        ax_logp.set_xlabel('logP', fontsize=11, fontweight='bold')
        ax_logp.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax_logp.set_title(f'{condition_name} - logP Distribution', fontsize=12, fontweight='bold')
        ax_logp.legend(fontsize=9)
        ax_logp.grid(True, alpha=0.3)

        # 统计信息
        stats_text = f"Mean:\nReal: {df_real['logP'].mean():.2f}\nGen: {df_gen['logP'].mean():.2f}"
        ax_logp.text(0.02, 0.98, stats_text, transform=ax_logp.transAxes,
                     fontsize=8, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'\n对比图已保存: {output_path}')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Fragment-VAE/CVAE 分子生成')
    parser.add_argument('--model_path', type=str, default='checkpoints/vae_final.pt',
                        help='模型文件路径（包含配置）')
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--dataset', type=str, default='zinc_250k.csv', help='真实数据集用于对比')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'mps', 'cpu'])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device != 'cuda' else 'cpu')
    print(f'设备: {device}')

    # 从模型文件加载配置
    print(f'\n加载模型: {args.model_path}')
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint['config']

    mode = config['mode']
    print(f'模式: {mode.upper()}')

    print(f"\n片段库配置:")
    print(f"  - 模板数量: {config['num_templates']}")
    print(f"  - 取代基数量: {config['num_substituents']}")
    print(f"  - 最大取代位点: {config['max_r_count']}")

    # 恢复片段库
    import data
    data.TEMPLATES = config['templates']
    data.SUBSTITUENTS = config['substituents']
    data.TEMPLATE_R_COUNTS = config['template_r_counts']
    data.NUM_TEMPLATES = config['num_templates']
    data.NUM_SUBSTITUENTS = config['num_substituents']
    data.MAX_R_COUNT = config['max_r_count']

    # 创建一个简单的解码器
    class SimpleDecoder:
        def decode(self, template_idx, substituent_indices):
            if isinstance(template_idx, torch.Tensor):
                template_idx = template_idx.item()
            if isinstance(substituent_indices, torch.Tensor):
                substituent_indices = substituent_indices.cpu().numpy().tolist()

            template = data.TEMPLATES[template_idx]
            smiles = template
            r_count = data.TEMPLATE_R_COUNTS[template_idx]

            for i in range(r_count):
                placeholder = f"{{R{i+1}}}"
                sub_idx = substituent_indices[i]
                substituent = data.SUBSTITUENTS[sub_idx]
                smiles = smiles.replace(placeholder, substituent)

            return smiles

    dataset = SimpleDecoder()

    # 根据配置中的mode创建模型
    print(f'\n创建{mode.upper()}模型...')
    if mode == 'vae':
        model = FragmentVAE(
            config['num_templates'],
            config['num_substituents'],
            config['max_r_count'],
            config.get('template_emb_dim', 64),
            config.get('substituent_emb_dim', 32),
            config.get('hidden_dim', 256),
            config.get('latent_dim', 128)
        ).to(device)
    else:
        model = FragmentCVAE(
            config['num_templates'],
            config['num_substituents'],
            config['max_r_count'],
            config.get('template_emb_dim', 64),
            config.get('substituent_emb_dim', 32),
            config.get('hidden_dim', 256),
            config.get('latent_dim', 128),
            config.get('condition_emb_dim', 32)
        ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print('模型加载成功')

    if mode == 'vae':
        # VAE模式：只生成一次，保存CSV和SVG
        print('\n===== VAE无条件生成 =====')
        df = generate_molecules(model, dataset, args.num_samples, device, mode)

        # 保存CSV
        output_csv = 'generated_vae.csv'
        df.to_csv(output_csv, index=False)
        print(f'\n保存CSV: {output_csv}')

        # 统计
        print(f'\n统计:')
        print(f'  总数: {len(df)}')
        print(f'  logP: [{df["logP"].min():.2f}, {df["logP"].max():.2f}], 均值={df["logP"].mean():.2f}')
        print(f'  QED: [{df["QED"].min():.3f}, {df["QED"].max():.3f}], 均值={df["QED"].mean():.3f}')

        # 绘制分子结构图
        svg_path = 'generated_vae.svg'
        visualize_svg(df, svg_path)
        print(f'\n完成！')

    else:
        # CVAE模式：生成3种条件，对比真实数据集
        print('\n===== CVAE条件生成 =====')

        # 定义3种条件
        conditions = [
            (0, -2, 'QED=0_logP=-2'),
            (0.5, -1, 'QED=0.5_logP=-1'),
            (1, 0, 'QED=1_logP=0')
        ]

        generated_results = []

        # 生成每种条件的分子
        for qed_target, logp_target, name in conditions:
            print(f'\n生成条件: QED={qed_target}, logP={logp_target}')
            df_gen = generate_molecules(
                model, dataset, args.num_samples, device, mode,
                logp_target, qed_target
            )

            # 保存CSV
            output_csv = f'generated_{name}.csv'
            df_gen.to_csv(output_csv, index=False)
            print(f'  保存CSV: {output_csv}')

            # 统计
            print(f'  统计:')
            print(f'    总数: {len(df_gen)}')
            print(f'    logP: 均值={df_gen["logP"].mean():.2f} (目标={logp_target})')
            print(f'    QED: 均值={df_gen["QED"].mean():.3f} (目标={qed_target})')

            # 绘制分子结构图
            svg_path = f'generated_{name}.svg'
            visualize_svg(df_gen, svg_path, n=100)
            print(f'  保存SVG: {svg_path}')

            # 记录用于对比
            generated_results.append((name, df_gen))

        # 加载真实数据集
        df_real = load_real_molecules(args.dataset, n_samples=1000)

        # 绘制对比图
        print('\n===== 绘制对比图 =====')
        plot_cvae_comparison(df_real, generated_results, output_path='cvae_comparison.png')

        print(f'\n完成！')


if __name__ == '__main__':
    main()
