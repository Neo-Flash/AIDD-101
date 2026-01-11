"""
训练脚本 - 训练基于模板填空的GAN模型
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

import data
from data import get_dataloader
from model import MoleculeGAN

# 导入RDKit用于分子可视化和QED计算
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, QED
    from rdkit import RDLogger
    import numpy as np
    # 关闭RDKit的警告和错误日志
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    print("警告: RDKit未安装，无法生成分子图片。可以运行: pip install rdkit")
    RDKIT_AVAILABLE = False


def visualize_molecules(smiles_list, save_path, mols_per_row=5):
    """
    将SMILES列表可视化为分子结构图并保存

    Args:
        smiles_list: SMILES字符串列表
        save_path: 保存路径
        mols_per_row: 每行显示的分子数

    Returns:
        valid_count: 有效分子数量
    """
    if not RDKIT_AVAILABLE:
        return 0

    # 将SMILES转换为分子对象
    mols = []
    valid_smiles = []

    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mols.append(mol)
                valid_smiles.append(smiles)
        except:
            pass

    valid_count = len(mols)

    # 如果没有有效分子，直接返回
    if valid_count == 0:
        print(f"  - 有效分子: 0/{len(smiles_list)} (0.0%) - 跳过绘制图片")
        return 0

    # 绘制分子网格图
    try:
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=mols_per_row,
            subImgSize=(300, 300),
            legends=valid_smiles,
            returnPNG=False
        )

        # 保存图片
        img.save(save_path)

        # 统计有效性
        print(f"  - 分子图片已保存: {save_path}")
        print(f"  - 有效分子: {valid_count}/{len(smiles_list)} ({valid_count/len(smiles_list)*100:.1f}%)")

    except Exception as e:
        print(f"  错误: 绘制分子失败 - {e}")
        return valid_count

    return valid_count


def calculate_qed_scores(smiles_list):
    """
    计算SMILES列表的QED分数

    Args:
        smiles_list: SMILES字符串列表

    Returns:
        qed_scores: QED分数列表（只包含有效分子的QED）
    """
    if not RDKIT_AVAILABLE:
        return []

    qed_scores = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                qed = QED.qed(mol)
                qed_scores.append(qed)
        except:
            pass

    return qed_scores


def plot_qed_distribution(generated_qeds, real_qeds, save_path):
    """
    绘制生成分子和真实分子的QED分布对比图

    Args:
        generated_qeds: 生成分子的QED列表
        real_qeds: 真实分子的QED列表
        save_path: 保存路径
    """
    plt.figure(figsize=(8, 6))

    # 绘制直方图（红色在下，蓝色在上，使用频率而不是频数）
    if real_qeds:
        plt.hist(real_qeds, bins=20, alpha=0.7, label=f'Real (n={len(real_qeds)})',
                color='red', edgecolor='black', density=True)
    if generated_qeds:
        plt.hist(generated_qeds, bins=20, alpha=0.7, label=f'Generated (n={len(generated_qeds)})',
                color='blue', edgecolor='black', density=True)

    plt.xlabel('QED Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('QED Distribution Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # 添加统计信息
    stats_text = ""
    if generated_qeds:
        stats_text += f"Generated: μ={np.mean(generated_qeds):.3f}, σ={np.std(generated_qeds):.3f}\n"
    if real_qeds:
        stats_text += f"Real: μ={np.mean(real_qeds):.3f}, σ={np.std(real_qeds):.3f}"

    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"  - QED distribution plot saved: {save_path}")
    if generated_qeds:
        print(f"  - Generated QED: mean={np.mean(generated_qeds):.3f}, std={np.std(generated_qeds):.3f}")
    if real_qeds:
        print(f"  - Real QED: mean={np.mean(real_qeds):.3f}, std={np.std(real_qeds):.3f}")


def train_gan(
    # 训练参数
    epochs=200,
    batch_size=64,
    lr_g=0.0002,
    lr_d=0.0001,
    device='cpu',
    save_dir='checkpoints',
    sample_interval=10,
    dataset_file='dataset.txt',
    max_molecules=None,
    # 模型架构参数
    noise_dim=128,
    g_hidden_dims=[256, 512, 256],
    g_dropout=0.3,
    d_template_emb=32,
    d_substituent_emb=16,
    d_hidden_dims=[128, 64],
    d_dropout=0.5,
    # 训练技巧参数
    temperature=0.5
):
    """
    训练GAN模型

    Args:
        # 训练参数
        epochs: 训练轮数
        batch_size: 批次大小
        lr_g: 生成器学习率
        lr_d: 判别器学习率
        device: 训练设备
        save_dir: 模型保存目录
        sample_interval: 采样间隔
        dataset_file: 真实分子数据集文件路径
        max_molecules: 最多使用的分子数量（None表示全部）

        # 模型架构参数
        noise_dim: 噪声维度
        g_hidden_dims: 生成器隐藏层维度列表
        g_dropout: 生成器Dropout比例
        d_template_emb: 判别器模板Embedding维度
        d_substituent_emb: 判别器取代基Embedding维度
        d_hidden_dims: 判别器隐藏层维度列表
        d_dropout: 判别器Dropout比例

        # 训练技巧参数
        temperature: Softmax温度参数
    """

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 1. 加载数据
    print("=" * 60)
    print("加载真实分子数据...")
    dataloader, dataset = get_dataloader(
        batch_size=batch_size,
        shuffle=True,
        dataset_file=dataset_file,
        max_molecules=max_molecules
    )

    # 2. 获取片段库信息（必须在加载数据后获取，因为全局变量在此时才被初始化）
    NUM_TEMPLATES = data.NUM_TEMPLATES
    NUM_SUBSTITUENTS = data.NUM_SUBSTITUENTS
    MAX_R_COUNT = data.MAX_R_COUNT

    # 2. 创建模型
    print("\n" + "=" * 60)
    print("创建模型...")
    gan = MoleculeGAN(
        num_templates=NUM_TEMPLATES,
        num_substituents=NUM_SUBSTITUENTS,
        max_r_count=MAX_R_COUNT,
        noise_dim=noise_dim,
        device=device,
        g_hidden_dims=g_hidden_dims,
        g_dropout=g_dropout,
        d_template_emb=d_template_emb,
        d_substituent_emb=d_substituent_emb,
        d_hidden_dims=d_hidden_dims,
        d_dropout=d_dropout
    )

    # 3. 定义优化器
    optimizer_g = optim.Adam(gan.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(gan.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

    # 4. 定义损失函数
    criterion_bce = nn.BCELoss()  # 用于判别器的真假判断
    criterion_ce = nn.CrossEntropyLoss()  # 用于分类损失

    # 5. 计算真实分子的QED分布（用于对比）
    print("\n" + "=" * 60)
    print("计算真实分子的QED分布...")
    if RDKIT_AVAILABLE:
        real_smiles_sample = []
        # 直接从训练数据集中解码真实分子样本
        sample_size = min(2000, len(dataset))
        for i in range(sample_size):
            template_idx, sub_idx = dataset.samples[i]
            smiles = dataset.decode(template_idx, sub_idx)
            real_smiles_sample.append(smiles)
        real_qeds = calculate_qed_scores(real_smiles_sample)
        print(f"  - Valid molecules: {len(real_qeds)}/{len(real_smiles_sample)}")
        if real_qeds:
            print(f"  - Real molecule QED: mean={np.mean(real_qeds):.3f}, std={np.std(real_qeds):.3f}")
    else:
        real_qeds = []

    # 6. 训练循环
    print("\n" + "=" * 60)
    print("开始训练...")

    # 用于记录损失
    g_losses = []
    d_losses = []

    for epoch in range(epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        num_batches = 0

        # 使用tqdm显示进度条
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')

        for real_template_idx, real_sub_idx in pbar:
            batch_size_actual = real_template_idx.size(0)
            real_template_idx = real_template_idx.to(device)
            real_sub_idx = real_sub_idx.to(device)

            # ===============================
            # 训练判别器 Discriminator
            # ===============================
            optimizer_d.zero_grad()

            # Label Smoothing：软化标签，避免判别器过度自信
            # 真实样本标签（0.9而不是1.0）
            real_labels = torch.ones(batch_size_actual, 1, device=device) * 0.9
            # 生成样本标签（0.1而不是0.0）
            fake_labels = torch.zeros(batch_size_actual, 1, device=device) + 0.1

            # 判别器判断真实样本
            real_scores = gan.discriminator(real_template_idx, real_sub_idx, use_soft=False)
            d_loss_real = criterion_bce(real_scores, real_labels)

            # 生成假样本
            noise = gan.sample_noise(batch_size_actual)
            fake_template_idx, fake_sub_idx = gan.generator.generate(noise)

            # 判别器判断生成样本
            fake_scores = gan.discriminator(fake_template_idx.detach(), fake_sub_idx.detach(), use_soft=False)
            d_loss_fake = criterion_bce(fake_scores, fake_labels)

            # 判别器总损失
            d_loss = d_loss_real + d_loss_fake

            # 反向传播和优化
            d_loss.backward()
            optimizer_d.step()

            # ===============================
            # 训练生成器 Generator
            # ===============================
            optimizer_g.zero_grad()

            # 生成新的假样本
            noise = gan.sample_noise(batch_size_actual)
            fake_template_logits, fake_sub_logits = gan.generator(noise)

            # 使用Softmax获得软概率分布（用于梯度回传）
            fake_template_probs = torch.softmax(fake_template_logits / temperature, dim=-1)
            fake_sub_probs = torch.softmax(fake_sub_logits / temperature, dim=-1)

            # 判别器判断生成样本（使用软概率）
            fake_scores = gan.discriminator(fake_template_probs, fake_sub_probs, use_soft=True)

            # 生成器损失：希望判别器认为生成的是真的（标签为1）
            g_loss = criterion_bce(fake_scores, real_labels)

            # 反向传播和优化
            g_loss.backward()
            optimizer_g.step()

            # 记录损失
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({
                'D_loss': f'{d_loss.item():.4f}',
                'G_loss': f'{g_loss.item():.4f}',
                'D(real)': f'{real_scores.mean().item():.2f}',
                'D(fake)': f'{fake_scores.mean().item():.2f}'
            })

        # 计算epoch平均损失
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)

        # 打印epoch总结
        print(f'Epoch [{epoch+1}/{epochs}] - D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}')

        # 定期保存模型和生成样本
        if (epoch + 1) % sample_interval == 0:
            # 保存模型
            checkpoint_path = os.path.join(save_dir, f'gan_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': gan.generator.state_dict(),
                'discriminator_state_dict': gan.discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'num_templates': NUM_TEMPLATES,
                'num_substituents': NUM_SUBSTITUENTS,
                'max_r_count': MAX_R_COUNT,
                'noise_dim': noise_dim,
                # 保存片段库（关键！生成时需要用相同的片段库）
                'templates': data.TEMPLATES,
                'substituents': data.SUBSTITUENTS,
                'template_r_counts': data.TEMPLATE_R_COUNTS
            }, checkpoint_path)
            print(f'  - 模型已保存: {checkpoint_path}')

            # 生成并打印一些样本
            print("  - 生成样本示例:")
            gan.generator.eval()
            generated_smiles = []
            with torch.no_grad():
                # 生成100个样本用于可视化和QED计算
                sample_noise = gan.sample_noise(100)
                sample_template_idx, sample_sub_idx = gan.generator.generate(sample_noise)
                for i in range(100):
                    smiles = dataset.decode(sample_template_idx[i], sample_sub_idx[i])
                    generated_smiles.append(smiles)
                    # 只打印前5个
                    if i < 5:
                        print(f"    {i+1}. {smiles}")
            gan.generator.train()

            # 将生成的分子可视化并保存为图片
            if RDKIT_AVAILABLE:
                img_path = os.path.join(save_dir, f'molecules_epoch_{epoch+1}.png')
                visualize_molecules(generated_smiles, img_path, mols_per_row=10)

                # 计算生成分子的QED并绘制分布对比图
                print("  - 计算QED分布...")
                generated_qeds = calculate_qed_scores(generated_smiles)
                qed_plot_path = os.path.join(save_dir, f'qed_distribution_epoch_{epoch+1}.png')
                plot_qed_distribution(generated_qeds, real_qeds, qed_plot_path)

    # 训练完成后保存最终模型
    final_path = os.path.join(save_dir, 'gan_final.pt')
    torch.save({
        'epoch': epochs,
        'generator_state_dict': gan.generator.state_dict(),
        'discriminator_state_dict': gan.discriminator.state_dict(),
        'num_templates': NUM_TEMPLATES,
        'num_substituents': NUM_SUBSTITUENTS,
        'max_r_count': MAX_R_COUNT,
        'noise_dim': noise_dim,
        # 保存片段库（关键！生成时需要用相同的片段库）
        'templates': data.TEMPLATES,
        'substituents': data.SUBSTITUENTS,
        'template_r_counts': data.TEMPLATE_R_COUNTS
    }, final_path)
    print(f"\n最终模型已保存: {final_path}")

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(save_dir, 'loss_curve.png')
    plt.savefig(loss_plot_path)
    print(f"Loss curve saved: {loss_plot_path}")

    print("\nTraining completed!")


if __name__ == '__main__':
    # 设置随机种子，保证可复现性
    torch.manual_seed(42)

    # 检查是否有GPU可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # ================================
    # 超参数配置区 - 所有参数都在这里调整
    # ================================

    # 训练参数
    EPOCHS = 100
    BATCH_SIZE = 64
    LR_G = 0.0002  # 降低生成器学习率（防止过度优化）
    LR_D = 0.0001  # 保持判别器学习率（增强判别能力）
    SAMPLE_INTERVAL = 10
    DATASET_FILE = 'dataset.txt'  # 真实分子数据集文件
    MAX_MOLECULES = 10000  # 随机采样1万个分子用于训练（None=全部，但会很慢）

    # 模型架构参数
    NOISE_DIM = 512  # 增加噪声维度，提升多样性（原256）
    G_HIDDEN_DIMS = [256, 512, 256]
    G_DROPOUT = 0.3
    D_TEMPLATE_EMB = 32
    D_SUBSTITUENT_EMB = 16
    D_HIDDEN_DIMS = [256, 128, 64]
    D_DROPOUT = 0.3  # 降低dropout，增强判别器能力（原0.1）

    # 训练技巧参数
    TEMPERATURE = 0.8  # 提高温度，增加采样随机性（原0.3）

    # 开始训练
    train_gan(
        # 训练参数
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr_g=LR_G,
        lr_d=LR_D,
        device=device,
        save_dir='checkpoints',
        sample_interval=SAMPLE_INTERVAL,
        dataset_file=DATASET_FILE,
        max_molecules=MAX_MOLECULES,
        # 模型架构参数
        noise_dim=NOISE_DIM,
        g_hidden_dims=G_HIDDEN_DIMS,
        g_dropout=G_DROPOUT,
        d_template_emb=D_TEMPLATE_EMB,
        d_substituent_emb=D_SUBSTITUENT_EMB,
        d_hidden_dims=D_HIDDEN_DIMS,
        d_dropout=D_DROPOUT,
        # 训练技巧参数
        temperature=TEMPERATURE
    )
