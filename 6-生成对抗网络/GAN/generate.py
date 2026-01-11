"""
生成脚本 - 使用训练好的GAN模型生成新分子
"""

import torch
import pandas as pd
import numpy as np
from data import NUM_TEMPLATES, NUM_SUBSTITUENTS, MAX_R_COUNT
from model import MoleculeGAN

# 导入RDKit用于分子验证和可视化
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    print("警告: RDKit未安装，无法验证和可视化分子")
    RDKIT_AVAILABLE = False

# 导入t-SNE和可视化库
try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    TSNE_AVAILABLE = True
except ImportError:
    print("警告: sklearn未安装，无法进行t-SNE可视化")
    TSNE_AVAILABLE = False


def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    """
    将SMILES转换为Morgan指纹

    Args:
        smiles: SMILES字符串
        radius: Morgan指纹半径
        n_bits: 指纹位数

    Returns:
        fingerprint: numpy array或None
    """
    if not RDKIT_AVAILABLE:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except:
        return None


def visualize_chemical_space(generated_smiles, real_smiles, save_path='tsne_chemical_space.png'):
    """
    使用t-SNE可视化生成分子和真实分子的药物化学空间

    Args:
        generated_smiles: 生成的SMILES列表
        real_smiles: 真实的SMILES列表
        save_path: 保存路径
    """
    if not RDKIT_AVAILABLE or not TSNE_AVAILABLE:
        print("跳过t-SNE可视化（需要RDKit和sklearn）")
        return

    print("\n" + "=" * 60)
    print("计算分子指纹...")

    # 1. 计算分子指纹
    generated_fps = []
    generated_valid = []
    for smi in generated_smiles:
        fp = smiles_to_fingerprint(smi)
        if fp is not None:
            generated_fps.append(fp)
            generated_valid.append(smi)

    real_fps = []
    real_valid = []
    for smi in real_smiles:
        fp = smiles_to_fingerprint(smi)
        if fp is not None:
            real_fps.append(fp)
            real_valid.append(smi)

    print(f"  - Generated valid fingerprints: {len(generated_fps)}/{len(generated_smiles)}")
    print(f"  - Real valid fingerprints: {len(real_fps)}/{len(real_smiles)}")

    if len(generated_fps) == 0 or len(real_fps) == 0:
        print("没有足够的有效分子，跳过t-SNE可视化")
        return

    # 1.5 分析生成分子多样性
    print("\n分析生成分子多样性...")
    generated_fps_array = np.array(generated_fps)

    # 计算生成分子之间的平均相似度
    from sklearn.metrics.pairwise import cosine_similarity
    if len(generated_fps) > 1:
        gen_similarity = cosine_similarity(generated_fps_array)
        # 排除对角线（自己和自己）
        mask = np.ones(gen_similarity.shape, dtype=bool)
        np.fill_diagonal(mask, False)
        avg_gen_similarity = gen_similarity[mask].mean()
        print(f"  - 生成分子平均指纹相似度: {avg_gen_similarity:.3f}")

        if avg_gen_similarity > 0.95:
            print(f"  ⚠️ 警告: 生成分子相似度极高 ({avg_gen_similarity:.3f})，存在严重的模式崩塌!")
        elif avg_gen_similarity > 0.85:
            print(f"  ⚠️ 警告: 生成分子相似度较高 ({avg_gen_similarity:.3f})，多样性不足")

    # 2. 合并指纹
    all_fps = np.array(generated_fps + real_fps)
    labels = ['Generated'] * len(generated_fps) + ['Real'] * len(real_fps)

    # 3. t-SNE降维
    print("\n执行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_fps) - 1))
    embeddings = tsne.fit_transform(all_fps)

    # 4. 绘制散点图
    print("绘制化学空间图...")
    plt.figure(figsize=(10, 8))

    # 分别绘制生成和真实分子
    gen_embeddings = embeddings[:len(generated_fps)]
    real_embeddings = embeddings[len(generated_fps):]

    plt.scatter(real_embeddings[:, 0], real_embeddings[:, 1],
                c='blue', alpha=0.5, s=30, label=f'Real (n={len(real_fps)})', edgecolors='none')
    plt.scatter(gen_embeddings[:, 0], gen_embeddings[:, 1],
                c='red', alpha=0.5, s=30, label=f'Generated (n={len(generated_fps)})', edgecolors='none')

    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title('Chemical Space Visualization (Morgan Fingerprints + t-SNE)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # 添加多样性信息
    if len(generated_fps) > 1:
        info_text = f"Generated avg similarity: {avg_gen_similarity:.3f}"
        plt.text(0.02, 0.02, info_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"\n  - t-SNE visualization saved: {save_path}")

    # 诊断信息
    print("\n" + "=" * 60)
    print("t-SNE诊断信息:")
    print(f"  - 如果生成分子聚集成圆圈/团状，说明:")
    print(f"    1. 生成分子多样性不足（指纹相似度高）")
    print(f"    2. 可能存在模式崩塌(mode collapse)")
    print(f"    3. GAN训练可能需要:")
    print(f"       - 降低判别器学习率")
    print(f"       - 增加噪声维度")
    print(f"       - 使用更多样的训练数据")
    print("=" * 60)


def generate_molecules(
    checkpoint_path,
    num_molecules=20,
    device='cpu',
    save_image=True,
    image_path='generated_molecules.png'
):
    """
    使用训练好的模型生成分子

    Args:
        checkpoint_path: 模型checkpoint路径
        num_molecules: 要生成的分子数量
        device: 设备
        save_image: 是否保存为图片
        image_path: 图片保存路径

    Returns:
        smiles_list: 生成的SMILES列表
    """

    # 1. 加载checkpoint
    print(f"加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 提取模型参数
    num_templates = checkpoint.get('num_templates', NUM_TEMPLATES)
    num_substituents = checkpoint.get('num_substituents', NUM_SUBSTITUENTS)
    max_r_count = checkpoint.get('max_r_count', MAX_R_COUNT)
    noise_dim = checkpoint.get('noise_dim', 128)

    # 恢复片段库（关键！必须使用训练时的片段库）
    import data
    if 'templates' in checkpoint and 'substituents' in checkpoint:
        data.TEMPLATES = checkpoint['templates']
        data.SUBSTITUENTS = checkpoint['substituents']
        data.TEMPLATE_R_COUNTS = checkpoint['template_r_counts']
        data.NUM_TEMPLATES = len(data.TEMPLATES)
        data.NUM_SUBSTITUENTS = len(data.SUBSTITUENTS)
        data.MAX_R_COUNT = max_r_count
        print(f"✓ 已从checkpoint恢复片段库:")
        print(f"  - 模板数量: {data.NUM_TEMPLATES}")
        print(f"  - 取代基数量: {data.NUM_SUBSTITUENTS}")
        print(f"  - 最大取代位点: {data.MAX_R_COUNT}")
    else:
        print("⚠️ 警告: checkpoint中没有片段库信息，请重新训练模型")
        raise RuntimeError("旧版本checkpoint不包含片段库，请重新训练")

    # 2. 创建模型
    print("创建模型...")
    gan = MoleculeGAN(
        num_templates=num_templates,
        num_substituents=num_substituents,
        max_r_count=max_r_count,
        noise_dim=noise_dim,
        device=device
    )

    # 加载权重
    gan.generator.load_state_dict(checkpoint['generator_state_dict'])
    gan.generator.eval()

    print(f"\n生成 {num_molecules} 个分子...\n")

    # 3. 生成分子
    smiles_list = []
    template_list = []
    substituent_list = []

    # 定义decode函数（直接使用恢复的片段库）
    def decode(template_idx, substituent_indices):
        """将(模板, 取代基)解码为SMILES字符串"""
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

    with torch.no_grad():
        # 生成噪声
        noise = gan.sample_noise(num_molecules)

        # 生成模板和取代基索引
        template_indices, sub_indices = gan.generator.generate(noise)

        # 解码为SMILES
        for i in range(num_molecules):
            template_idx = template_indices[i]
            sub_idx = sub_indices[i]

            smiles = decode(template_idx, sub_idx)
            smiles_list.append(smiles)
            template_list.append(template_idx.item())
            substituent_list.append(sub_idx.cpu().tolist())

            print(f"{i+1:2d}. 模板{template_idx.item():2d} | 取代基{str(sub_idx.tolist()[:3]):20s} | {smiles}")

    # 4. 验证分子有效性
    if RDKIT_AVAILABLE:
        print("\n" + "=" * 60)
        print("验证分子有效性...")

        valid_mols = []
        valid_smiles = []
        valid_count = 0

        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_mols.append(mol)
                valid_smiles.append(smiles)
                valid_count += 1
                print(f"  ✓ 分子 {i+1:2d} 有效: {smiles}")
            else:
                print(f"  ✗ 分子 {i+1:2d} 无效: {smiles}")

        validity_rate = (valid_count / num_molecules) * 100
        print(f"\n有效率: {valid_count}/{num_molecules} ({validity_rate:.1f}%)")

        # 5. 可视化有效分子
        if save_image and valid_count > 0:
            print("\n" + "=" * 60)
            print(f"保存分子图片到: {image_path}")

            try:
                img = Draw.MolsToGridImage(
                    valid_mols,
                    molsPerRow=5,
                    subImgSize=(300, 300),
                    legends=valid_smiles,
                    returnPNG=False
                )
                img.save(image_path)
                print(f"✓ 图片已保存 ({valid_count} 个有效分子)")
            except Exception as e:
                print(f"✗ 保存图片失败: {e}")
        elif valid_count == 0:
            print("\n没有有效分子，跳过图片保存")

    return smiles_list


if __name__ == '__main__':
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}\n")

    # ================================
    # 配置区
    # ================================

    # 模型checkpoint路径
    CHECKPOINT_PATH = 'checkpoints/gan_final.pt'  # 或使用 'checkpoints/gan_epoch_XXX.pt'

    # 生成参数
    NUM_MOLECULES = 1000  # 生成分子数量
    SAVE_IMAGE = True   # 是否保存图片
    IMAGE_PATH = 'generated_molecules.png'

    # 真实数据集路径（用于t-SNE对比）
    DATASET_FILE = 'dataset.txt'
    NUM_REAL_MOLECULES = 1000  # 从真实数据集采样的分子数量

    # 生成分子
    smiles_list = generate_molecules(
        checkpoint_path=CHECKPOINT_PATH,
        num_molecules=NUM_MOLECULES,
        device=device,
        save_image=SAVE_IMAGE,
        image_path=IMAGE_PATH
    )

    print("\n" + "=" * 60)
    print("生成完成！")

    # ================================
    # 保存生成的分子为CSV
    # ================================

    print("\n" + "=" * 60)
    print("保存生成分子为CSV...")

    df_generated = pd.DataFrame({'smiles': smiles_list})
    df_generated.to_csv('generated_molecules.csv', index=False)
    print(f"  - 已保存 {len(smiles_list)} 个生成分子到: generated_molecules.csv")

    # 统计唯一分子数量
    unique_smiles = set(smiles_list)
    print(f"  - 唯一分子数: {len(unique_smiles)}/{len(smiles_list)} ({len(unique_smiles)/len(smiles_list)*100:.1f}%)")

    if len(unique_smiles) < len(smiles_list) * 0.5:
        print(f"  ⚠️ 警告: 生成分子重复率高 ({100 - len(unique_smiles)/len(smiles_list)*100:.1f}%)，可能存在模式崩塌(mode collapse)")

    # ================================
    # 可视化随机100个分子
    # ================================

    if RDKIT_AVAILABLE and len(smiles_list) > 0:
        print("\n" + "=" * 60)
        print("可视化随机100个生成分子...")

        # 验证分子有效性
        valid_mols = []
        valid_smiles_viz = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_mols.append(mol)
                valid_smiles_viz.append(smi)

        # 随机取100个
        if len(valid_mols) > 100:
            import random
            random.seed(42)
            indices = random.sample(range(len(valid_mols)), 100)
            sample_mols = [valid_mols[i] for i in indices]
            sample_smiles = [valid_smiles_viz[i] for i in indices]
        else:
            sample_mols = valid_mols
            sample_smiles = valid_smiles_viz

        print(f"  - 选择了 {len(sample_mols)} 个有效分子进行可视化")

        try:
            img = Draw.MolsToGridImage(
                sample_mols,
                molsPerRow=10,
                subImgSize=(200, 200),
                legends=sample_smiles,
                returnPNG=False
            )
            img.save('generated_molecules_sample.png')
            print(f"  - 分子结构图已保存: generated_molecules_sample.png")
        except Exception as e:
            print(f"  ✗ 保存图片失败: {e}")

    # ================================
    # t-SNE可视化化学空间
    # ================================

    print("\n" + "=" * 60)
    print("准备t-SNE可视化...")

    # 读取真实分子
    try:
        with open(DATASET_FILE, 'r') as f:
            lines = f.readlines()

        # 每行一个SMILES
        real_smiles_all = [line.strip() for line in lines if line.strip()]

        # 随机采样指定数量的真实分子
        if len(real_smiles_all) > NUM_REAL_MOLECULES:
            import random
            random.seed(42)
            real_smiles = random.sample(real_smiles_all, NUM_REAL_MOLECULES)
        else:
            real_smiles = real_smiles_all

        print(f"  - 从数据集读取了 {len(real_smiles)} 个真实分子")

        # t-SNE可视化
        visualize_chemical_space(
            generated_smiles=smiles_list,
            real_smiles=real_smiles,
            save_path='tsne_chemical_space.png'
        )

    except Exception as e:
        print(f"读取数据集失败: {e}")
        print("跳过t-SNE可视化")

    print("\n" + "=" * 60)
    print("全部完成！")
