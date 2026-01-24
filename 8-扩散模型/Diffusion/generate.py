"""
generate.py - 使用训练好的扩散模型生成分子

从训练好的模型生成新的分子片段组合，并转换为SMILES
"""

import torch
import argparse
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, QED
from tqdm import tqdm

import data
from model import FragmentDiffusion, FragmentCDiffusion


def load_model(model_path, device):
    """加载训练好的模型"""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    model_config = checkpoint['model_config']

    print(f"加载模型: {model_path}")
    print(f"配置: {model_config}")

    # 恢复全局片段库
    import data
    data.TEMPLATES = config['templates']
    data.SUBSTITUENTS = config['substituents']
    data.TEMPLATE_R_COUNTS = config['template_r_counts']
    data.MAX_R_COUNT = config['max_r_count']
    data.NUM_TEMPLATES = config['num_templates']
    data.NUM_SUBSTITUENTS = config['num_substituents']

    if model_config['mode'] == 'diffusion':
        model = FragmentDiffusion(
            num_templates=config['num_templates'],
            num_substituents=config['num_substituents'],
            max_r_count=config['max_r_count'],
            emb_dim=model_config['emb_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_timesteps=model_config['num_timesteps'],
            beta_schedule=model_config['beta_schedule']
        )
    else:
        model = FragmentCDiffusion(
            num_templates=config['num_templates'],
            num_substituents=config['num_substituents'],
            max_r_count=config['max_r_count'],
            emb_dim=model_config['emb_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_timesteps=model_config['num_timesteps'],
            beta_schedule=model_config['beta_schedule']
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, config, model_config


def decode_to_smiles(template_idx, sub_idx):
    """将片段索引解码为SMILES"""
    if isinstance(template_idx, torch.Tensor):
        template_idx = template_idx.item()
    if isinstance(sub_idx, torch.Tensor):
        sub_idx = sub_idx.cpu().numpy()

    template = data.TEMPLATES[template_idx]
    smiles = template
    r_count = data.TEMPLATE_R_COUNTS[template_idx]

    for i in range(r_count):
        placeholder = f"{{R{i+1}}}"
        sub_index = int(sub_idx[i])
        substituent = data.SUBSTITUENTS[sub_index]
        smiles = smiles.replace(placeholder, substituent)

    return smiles


def validate_smiles(smiles):
    """验证SMILES有效性"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, None, None, None

        # 标准化
        smiles = Chem.MolToSmiles(mol)

        # 计算性质
        logp = Descriptors.MolLogP(mol)
        qed = QED.qed(mol)

        return True, smiles, logp, qed
    except:
        return False, None, None, None


def visualize_trajectory_gif(trajectory, sample_idx, output_path, mode):
    """
    将单个分子的去噪轨迹保存为GIF动画

    Args:
        trajectory: 轨迹列表 [(t, template_idx, sub_idx), ...]
        sample_idx: 要可视化的样本索引
        output_path: 输出GIF文件路径
        mode: 模式（用于标题）
    """
    from PIL import Image, ImageDraw, ImageFont
    from rdkit.Chem import Draw

    frames = []

    print(f"    处理 {len(trajectory)} 帧...")

    for t, template_idx, sub_idx in trajectory:
        # 解码为SMILES
        smiles = decode_to_smiles(template_idx[sample_idx], sub_idx[sample_idx])
        valid, std_smiles, logp, qed = validate_smiles(smiles)

        if valid:
            mol = Chem.MolFromSmiles(std_smiles)

            # 使用RDKit绘制单个分子
            img = Draw.MolToImage(mol, size=(400, 400))

            # 添加文字标注
            draw = ImageDraw.Draw(img)
            text = f"Step: {t}\nlogP: {logp:.2f}\nQED: {qed:.2f}"

            # 使用默认字体
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
            except:
                font = ImageFont.load_default()

            draw.text((10, 10), text, fill='black', font=font)
            frames.append(img)
        else:
            # 无效结构，创建空白帧
            img = Image.new('RGB', (400, 400), color='white')
            draw = ImageDraw.Draw(img)
            text = f"Step: {t}\n[Invalid Structure]"

            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
            except:
                font = ImageFont.load_default()

            draw.text((150, 180), text, fill='red', font=font)
            frames.append(img)

    if len(frames) > 0:
        # 保存为GIF
        # 第一帧和最后一帧停留更久
        durations = [500] + [100] * (len(frames) - 2) + [1000] if len(frames) > 2 else [500] * len(frames)

        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0
        )
        print(f"    保存: {output_path} ({len(frames)} 帧)")
    else:
        print(f"    警告: 样本 {sample_idx} 没有有效帧")


def visualize_fingerprint_trajectory_gif(trajectory, sample_idx, output_path, reference_smiles, mode):
    """
    绘制摩根指纹t-SNE轨迹图GIF动画

    Args:
        trajectory: 轨迹列表 [(t, template_idx, sub_idx), ...]
        sample_idx: 要可视化的样本索引
        output_path: 输出GIF文件路径
        reference_smiles: 参考分子的SMILES列表（用于t-SNE背景）
        mode: 模式
    """
    from rdkit.Chem import AllChem
    from sklearn.manifold import TSNE
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    print(f"    计算摩根指纹并进行t-SNE降维...")

    # 1. 计算参考分子的指纹
    ref_fps = []
    valid_ref_smiles = []
    for smi in reference_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            ref_fps.append(list(fp))
            valid_ref_smiles.append(smi)

    # 2. 计算轨迹中每一步的指纹（只保留有效的）
    valid_traj_fps = []
    valid_traj_indices = []
    all_traj_info = []

    for step_idx, (t, template_idx, sub_idx) in enumerate(trajectory):
        smiles = decode_to_smiles(template_idx[sample_idx], sub_idx[sample_idx])
        valid, std_smiles, logp, qed = validate_smiles(smiles)

        if valid:
            mol = Chem.MolFromSmiles(std_smiles)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            valid_traj_fps.append(list(fp))
            valid_traj_indices.append(step_idx)
            all_traj_info.append((t, std_smiles, logp, qed, True, step_idx))
        else:
            # 无效结构，不参与t-SNE，但记录信息用于GIF显示
            all_traj_info.append((t, None, None, None, False, step_idx))

    # 3. 合并所有有效指纹进行t-SNE（只包含参考分子和有效的轨迹步骤）
    all_fps = np.array(ref_fps + valid_traj_fps)

    print(f"    t-SNE降维 ({len(all_fps)} 个分子，其中轨迹有效步骤: {len(valid_traj_fps)}/{len(trajectory)})...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_fps)-1))
    coords_2d = tsne.fit_transform(all_fps)

    # 分离参考点和有效轨迹点
    ref_coords = coords_2d[:len(ref_fps)]
    valid_traj_coords = coords_2d[len(ref_fps):]

    # 创建完整轨迹的坐标映射（无效步骤用None标记）
    traj_coords_map = {}
    for i, step_idx in enumerate(valid_traj_indices):
        traj_coords_map[step_idx] = valid_traj_coords[i]

    # 4. 生成GIF帧（包括所有步骤，无效步骤显示问号）
    frames = []
    print(f"    生成 {len(all_traj_info)} 帧...")

    # 获取有效轨迹坐标的范围（用于固定坐标轴）
    if len(valid_traj_coords) > 0:
        all_valid_coords = np.vstack([ref_coords, valid_traj_coords])
    else:
        all_valid_coords = ref_coords
    x_min, x_max = all_valid_coords[:, 0].min() - 5, all_valid_coords[:, 0].max() + 5
    y_min, y_max = all_valid_coords[:, 1].min() - 5, all_valid_coords[:, 1].max() + 5

    # 收集有效步骤的坐标用于绘制轨迹线
    valid_coords_for_plot = []
    for i in range(len(all_traj_info)):
        if i in traj_coords_map:
            valid_coords_for_plot.append(traj_coords_map[i])

    for i, (t, smiles, logp, qed, valid, step_idx) in enumerate(all_traj_info):
        fig, ax = plt.subplots(figsize=(8, 8))

        # 绘制参考分子（背景，灰色点，增大半径，提高透明度）
        ax.scatter(ref_coords[:, 0], ref_coords[:, 1],
                  c='lightgray', s=50, alpha=0.8, label='Reference molecules', edgecolors='none')

        # 绘制已经走过的有效轨迹路径
        if len(valid_coords_for_plot) > 1:
            valid_coords_array = np.array(valid_coords_for_plot)
            # 只绘制到当前步骤之前的有效点
            current_valid_idx = sum(1 for j in range(i+1) if j in traj_coords_map)
            if current_valid_idx > 1:
                ax.plot(valid_coords_array[:current_valid_idx, 0], valid_coords_array[:current_valid_idx, 1],
                       'b-', linewidth=2, alpha=0.6, label='Trajectory')

        # 标记起点（第一个有效步骤，圆形，绿色）
        if len(valid_coords_for_plot) > 0:
            ax.scatter(valid_coords_for_plot[0][0], valid_coords_for_plot[0][1],
                      c='green', s=200, marker='o', edgecolors='black', linewidth=2,
                      label='Start', zorder=5)

        # 标记当前点
        if valid:
            # 有效结构：星星，红色
            current_coord = traj_coords_map[step_idx]
            ax.scatter(current_coord[0], current_coord[1],
                      c='red', s=300, marker='*', edgecolors='black', linewidth=2,
                      label='Current (valid)', zorder=10)
            title = f"Step {i+1}/{len(all_traj_info)} (t={t}) | logP={logp:.2f}, QED={qed:.2f}"
        else:
            # 无效结构：问号，橙色，显示在图的左上角
            ax.text(0.05, 0.95, '?', transform=ax.transAxes,
                   fontsize=100, color='orange', weight='bold',
                   ha='left', va='top', alpha=0.7, zorder=10)
            title = f"Step {i+1}/{len(all_traj_info)} (t={t}) | Invalid Structure"

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        # 保持坐标轴范围固定
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # 转换为PIL图像
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = Image.fromarray(img_array)
        frames.append(img)

        plt.close(fig)

    # 5. 保存为GIF
    if len(frames) > 0:
        durations = [500] + [100] * (len(frames) - 2) + [1000] if len(frames) > 2 else [500] * len(frames)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0
        )
        print(f"    保存: {output_path} ({len(frames)} 帧)")
    else:
        print(f"    警告: 没有生成任何帧")



def main():
    parser = argparse.ArgumentParser(description='生成分子')
    parser.add_argument('--model_path', type=str, default='checkpoints/cdiffusion_final.pt', help='模型文件路径')
    parser.add_argument('--num_samples', type=int, default=300, help='生成样本数')
    parser.add_argument('--sampling_steps', type=int, default=100, help='采样去噪步数（默认100步，训练时为1000步）')
    parser.add_argument('--output_dir', type=str, default='generated', help='输出目录')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'mps', 'cpu'])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # 加载模型
    model, config, model_config = load_model(args.model_path, device)
    mode = model_config['mode']

    print(f"\n训练步数: {model_config['num_timesteps']}")
    print(f"采样步数: {args.sampling_steps}")

    # 加载参考数据集（用于指纹t-SNE背景）
    print(f"\n===== 加载参考数据集 =====")
    dataset_file = 'zinc_250k.csv'
    if os.path.exists(dataset_file):
        ref_df = pd.read_csv(dataset_file)
        smiles_col = 'SMILES' if 'SMILES' in ref_df.columns else 'smiles'
        # 随机采样500个分子作为背景
        if len(ref_df) > 1500:
            ref_df_sample = ref_df.sample(n=1500, random_state=42)
        else:
            ref_df_sample = ref_df
        reference_smiles = ref_df_sample[smiles_col].tolist()
        print(f"加载 {len(reference_smiles)} 个参考分子用于t-SNE背景")
    else:
        print(f"警告: 未找到数据集文件 {dataset_file}，将跳过指纹轨迹可视化")
        reference_smiles = []

    print(f"\n===== 生成 {args.num_samples} 个有效分子 =====")

    # 循环生成直到达到目标数量
    results = []
    unique_smiles = set()
    total_attempts = 0
    batch_size = 100  # 每次生成100个候选

    pbar = tqdm(total=args.num_samples, desc='生成有效分子', unit='个')

    while len(results) < args.num_samples:
        # 生成一批候选
        if mode == 'diffusion':
            template_idx, sub_idx = model.sample(batch_size, device, sampling_steps=args.sampling_steps)
        else:
            # 条件生成：均匀分配3种条件
            n_per_condition = batch_size // 3
            template_1, sub_1 = model.sample(n_per_condition, logp=2, qed=0.0, device=device, sampling_steps=args.sampling_steps)
            template_2, sub_2 = model.sample(n_per_condition, logp=2, qed=0.5, device=device, sampling_steps=args.sampling_steps)
            template_3, sub_3 = model.sample(batch_size - 2 * n_per_condition, logp=2, qed=1.0, device=device, sampling_steps=args.sampling_steps)
            template_idx = torch.cat([template_1, template_2, template_3], dim=0)
            sub_idx = torch.cat([sub_1, sub_2, sub_3], dim=0)

        # 解码并验证
        for i in range(len(template_idx)):
            total_attempts += 1
            smiles = decode_to_smiles(template_idx[i], sub_idx[i])
            valid, std_smiles, logp, qed = validate_smiles(smiles)

            if valid and std_smiles not in unique_smiles:
                unique_smiles.add(std_smiles)
                results.append({
                    'Index': len(results),
                    'SMILES': std_smiles,
                    'logP': logp,
                    'QED': qed,
                    'Template_idx': template_idx[i].item(),
                    'Sub_idx': sub_idx[i].cpu().numpy().tolist()
                })
                pbar.update(1)

                if len(results) >= args.num_samples:
                    break

    pbar.close()
    success_rate = len(results) / total_attempts * 100
    print(f"\n生成完成: {len(results)} 个有效分子 (总尝试: {total_attempts}, 成功率: {success_rate:.1f}%)")

    # 保存CSV
    df = pd.DataFrame(results)
    csv_path = f'{args.output_dir}/generated_{mode}.csv'
    df.to_csv(csv_path, index=False)
    print(f"保存结果: {csv_path}")

    # 可视化分子（最多100个）
    if len(results) > 0:
        print("\n生成分子可视化...")
        n_show = min(100, len(results))
        mols = []
        legends = []

        for i in range(n_show):
            mol = Chem.MolFromSmiles(results[i]['SMILES'])
            if mol:
                mols.append(mol)
                legends.append(f"#{i}\nlogP={results[i]['logP']:.2f}\nQED={results[i]['QED']:.3f}")

        if mols:
            # 使用RDKit原生SVG绘制
            from rdkit.Chem.Draw import rdMolDraw2D

            n_mols = len(mols)
            n_cols = 10  # 每行10个分子
            n_rows = (n_mols + n_cols - 1) // n_cols

            mol_size = (200, 200)  # 稍微缩小以适应更多分子
            img_width = n_cols * mol_size[0]
            img_height = n_rows * mol_size[1]

            drawer = rdMolDraw2D.MolDraw2DSVG(img_width, img_height, mol_size[0], mol_size[1])

            # 调整绘图选项：细化化学键
            opts = drawer.drawOptions()
            opts.bondLineWidth = 1.0  # 默认是2，降低到1使键更细

            drawer.DrawMolecules(mols, legends=legends)
            drawer.FinishDrawing()

            svg_path = f'{args.output_dir}/generated_{mode}.svg'
            with open(svg_path, 'w') as f:
                f.write(drawer.GetDrawingText())
            print(f"保存可视化: {svg_path} (展示 {len(mols)} 个分子)")

    # 生成轨迹可视化（2个成功生成的分子的GIF动画）
    if len(results) >= 2:
        print("\n===== 生成去噪轨迹GIF动画 =====")
        print("生成2个有效分子的完整去噪轨迹...")

        import random
        random.seed(42)

        num_trajectory_samples = 2
        num_steps = args.sampling_steps  # 使用采样步数

        print(f"采样步数: {num_steps} 步，每步都会记录")
        print("循环生成直到得到2个有效轨迹...")

        successful_trajectories = 0
        attempts = 0
        max_attempts = 10  # 最多尝试10次

        while successful_trajectories < num_trajectory_samples and attempts < max_attempts:
            attempts += 1
            print(f"\n尝试 {attempts}: 生成轨迹候选...")

            # 生成单个样本（带完整轨迹，每步都记录）
            if mode == 'diffusion':
                template_idx, sub_idx, trajectory = model.sample(
                    1, device, return_trajectory=True, trajectory_interval=1, sampling_steps=args.sampling_steps
                )
            else:
                # 随机选择一个条件
                conditions = [(1.0, 0.5), (2.5, 0.7), (4.0, 0.9)]
                logp_val, qed_val = random.choice(conditions)
                template_idx, sub_idx, trajectory = model.sample(
                    1, logp_val, qed_val, device,
                    return_trajectory=True, trajectory_interval=1, sampling_steps=args.sampling_steps
                )

            # 检查最终结果是否有效
            final_smiles = decode_to_smiles(template_idx[0], sub_idx[0])
            valid, std_smiles, logp, qed = validate_smiles(final_smiles)

            if valid:
                successful_trajectories += 1
                print(f"  ✓ 成功! 最终分子: {std_smiles[:50]}... (logP={logp:.2f}, QED={qed:.2f})")

                # 1. 可视化分子结构轨迹并保存为GIF
                gif_path = f'{args.output_dir}/trajectory_{mode}_{successful_trajectories}.gif'
                visualize_trajectory_gif(trajectory, 0, gif_path, mode)

                # 2. 可视化摩根指纹t-SNE轨迹并保存为GIF
                if len(reference_smiles) > 0:
                    fp_gif_path = f'{args.output_dir}/trajectory_fingerprint_{mode}_{successful_trajectories}.gif'
                    visualize_fingerprint_trajectory_gif(trajectory, 0, fp_gif_path, reference_smiles, mode)
            else:
                print(f"  ✗ 失败: 最终结构无效，重新生成")

        if successful_trajectories == num_trajectory_samples:
            print(f"\n✓ 轨迹GIF动画完成！保存在 {args.output_dir}/trajectory_{mode}_*.gif")
        else:
            print(f"\n警告: 只成功生成了 {successful_trajectories}/{num_trajectory_samples} 个有效轨迹")

    # 统计
    print(f"\n===== 统计信息 =====")
    print(f"目标数量: {args.num_samples}")
    print(f"成功生成: {len(results)}")
    print(f"总尝试次数: {total_attempts}")
    print(f"有效率: {success_rate:.1f}%")

    if len(results) > 0:
        logp_values = [r['logP'] for r in results]
        qed_values = [r['QED'] for r in results]

        print(f"\nlogP 统计:")
        print(f"  - 均值: {np.mean(logp_values):.2f}")
        print(f"  - 标准差: {np.std(logp_values):.2f}")
        print(f"  - 范围: [{np.min(logp_values):.2f}, {np.max(logp_values):.2f}]")

        print(f"\nQED 统计:")
        print(f"  - 均值: {np.mean(qed_values):.3f}")
        print(f"  - 标准差: {np.std(qed_values):.3f}")
        print(f"  - 范围: [{np.min(qed_values):.3f}, {np.max(qed_values):.3f}]")

        print(f"\n前5个样例:")
        print(df.head()[['SMILES', 'logP', 'QED']])

    # 绘制分布对比图（仅cdiffusion模式）
    if mode == 'cdiffusion':
        print(f"\n===== CDiffusion 3种条件对比生成 =====")

        # 定义3种条件
        conditions = [
            (2, 0.0, 'QED=0_logP=2'),
            (2, 0.5, 'QED=0.5_logP=2'),
            (2, 1.0, 'QED=1_logP=2')
        ]

        generated_results = []

        # 为每种条件生成分子
        for logp_target, qed_target, name in conditions:
            print(f"\n生成条件: logP={logp_target}, QED={qed_target}")

            condition_results = []
            condition_unique_smiles = set()
            condition_attempts = 0
            target_count = args.num_samples // 3  # 每种条件生成1/3的分子

            pbar = tqdm(total=target_count, desc=f'  生成 {name}', unit='个')

            while len(condition_results) < target_count:
                # 生成一批
                template_idx, sub_idx = model.sample(
                    100, logp_target, qed_target, device, sampling_steps=args.sampling_steps
                )

                # 解码验证
                for i in range(len(template_idx)):
                    condition_attempts += 1
                    smiles = decode_to_smiles(template_idx[i], sub_idx[i])
                    valid, std_smiles, logp, qed = validate_smiles(smiles)

                    if valid and std_smiles not in condition_unique_smiles:
                        condition_unique_smiles.add(std_smiles)
                        condition_results.append({
                            'SMILES': std_smiles,
                            'logP': logp,
                            'QED': qed
                        })
                        pbar.update(1)

                        if len(condition_results) >= target_count:
                            break

            pbar.close()

            df_condition = pd.DataFrame(condition_results)
            success_rate = len(condition_results) / condition_attempts * 100
            print(f"  完成: {len(condition_results)} 个 (尝试 {condition_attempts} 次, 成功率 {success_rate:.1f}%)")
            print(f"  logP: 均值={df_condition['logP'].mean():.2f} (目标={logp_target})")
            print(f"  QED: 均值={df_condition['QED'].mean():.3f} (目标={qed_target})")

            # 保存CSV
            condition_csv = f'{args.output_dir}/generated_{name}.csv'
            df_condition.to_csv(condition_csv, index=False)
            print(f"  保存: {condition_csv}")

            # 记录用于对比
            generated_results.append((name, df_condition))

        # 加载真实数据集
        print(f"\n加载真实数据集...")
        dataset_file = 'zinc_250k.csv'
        if os.path.exists(dataset_file):
            df_real_full = pd.read_csv(dataset_file)
            smiles_col = 'SMILES' if 'SMILES' in df_real_full.columns else 'smiles'

            # 随机采样1000个分子
            if len(df_real_full) > 1000:
                df_real_sample = df_real_full.sample(n=1000, random_state=42)
            else:
                df_real_sample = df_real_full

            # 计算真实分子的性质
            real_data = []
            for smi in tqdm(df_real_sample[smiles_col].tolist(), desc='计算真实分子性质'):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    try:
                        real_data.append({
                            'SMILES': smi,
                            'logP': Descriptors.MolLogP(mol),
                            'QED': QED.qed(mol)
                        })
                    except:
                        continue

            df_real = pd.DataFrame(real_data)
            print(f"有效真实分子: {len(df_real)}")

            # 绘制3行2列对比图
            print(f"\n绘制分布对比图...")
            import matplotlib.pyplot as plt

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
            comparison_path = f'{args.output_dir}/distribution_comparison.png'
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            print(f"对比图已保存: {comparison_path}")
            plt.close()
        else:
            print(f"警告: 未找到数据集文件 {dataset_file}")

    print(f"\n===== 完成 =====")



if __name__ == '__main__':
    main()
