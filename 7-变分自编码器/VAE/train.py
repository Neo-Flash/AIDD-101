"""
train.py - 基于片段的VAE/CVAE训练脚本
--mode vae  : 无条件生成
--mode cvae : 有条件生成（使用logP和QED作为条件）
"""

import torch
import torch.optim as optim
import argparse
import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

from data import get_dataloader, get_config
from model import FragmentVAE, FragmentCVAE, vae_loss


def train_epoch(model, dataloader, optimizer, device, mode='vae', kl_weight=0.1, epoch=0):
    """训练一个epoch"""
    model.train()
    total_loss, total_recon, total_kl = 0, 0, 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch:3d} [Train]', leave=False)

    for batch_idx, batch in enumerate(pbar):
        try:
            template_idx, sub_idx, logp, qed_score = batch
            template_idx = template_idx.to(device)
            sub_idx = sub_idx.to(device)
            logp = logp.to(device)
            qed_score = qed_score.to(device)

            optimizer.zero_grad()

            if mode == 'vae':
                template_logits, sub_logits, mu, logvar = model(template_idx, sub_idx)
                loss_tuple = vae_loss(
                    template_logits, sub_logits, template_idx, sub_idx, mu, logvar, kl_weight
                )
                loss, recon, kl = loss_tuple
                prop_loss = 0.0
            else:
                # CVAE模式：返回5个值（加入predicted_properties）
                template_logits, sub_logits, mu, logvar, pred_props = model(
                    template_idx, sub_idx, logp, qed_score
                )
                # 准备目标属性（归一化后的logP和QED）
                target_props = torch.stack([
                    ((logp - (-2)) / 10).clamp(0, 1),
                    qed_score.clamp(0, 1)
                ], dim=-1)

                loss_tuple = vae_loss(
                    template_logits, sub_logits, template_idx, sub_idx, mu, logvar, kl_weight,
                    predicted_properties=pred_props, target_properties=target_props, property_weight=1.0
                )
                loss, recon, kl, prop_loss = loss_tuple

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            num_batches += 1

            if mode == 'cvae':
                pbar.set_postfix({
                    'loss': f'{loss.item():.3f}',
                    'recon': f'{recon.item():.3f}',
                    'kl': f'{kl.item():.3f}',
                    'prop': f'{prop_loss:.3f}'
                })
            else:
                pbar.set_postfix({
                    'loss': f'{loss.item():.3f}',
                    'recon': f'{recon.item():.3f}',
                    'kl': f'{kl.item():.3f}'
                })
        except Exception as e:
            print(f"\n错误在batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            raise

    return total_loss / num_batches, total_recon / num_batches, total_kl / num_batches


def test_epoch(model, dataloader, device, mode='vae', kl_weight=0.1):
    """测试一个epoch"""
    model.eval()
    total_loss, total_recon, total_kl = 0, 0, 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            template_idx, sub_idx, logp, qed_score = batch
            template_idx = template_idx.to(device)
            sub_idx = sub_idx.to(device)
            logp = logp.to(device)
            qed_score = qed_score.to(device)

            if mode == 'vae':
                template_logits, sub_logits, mu, logvar = model(template_idx, sub_idx)
                loss_tuple = vae_loss(
                    template_logits, sub_logits, template_idx, sub_idx, mu, logvar, kl_weight
                )
                loss, recon, kl = loss_tuple
            else:
                template_logits, sub_logits, mu, logvar, pred_props = model(
                    template_idx, sub_idx, logp, qed_score
                )
                target_props = torch.stack([
                    ((logp - (-2)) / 10).clamp(0, 1),
                    qed_score.clamp(0, 1)
                ], dim=-1)
                loss_tuple = vae_loss(
                    template_logits, sub_logits, template_idx, sub_idx, mu, logvar, kl_weight,
                    predicted_properties=pred_props, target_properties=target_props, property_weight=1.0
                )
                loss, recon, kl, prop_loss = loss_tuple

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            num_batches += 1

    return total_loss / max(1, num_batches), total_recon / max(1, num_batches), total_kl / max(1, num_batches)


def evaluate_reconstruction(model, dataset, device, mode='vae', num_samples=100):
    """评估重建准确率"""
    model.eval()
    correct_template = 0
    correct_sub = 0
    total = 0

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            template_idx = sample[0].unsqueeze(0).to(device)
            sub_idx = sample[1].unsqueeze(0).to(device)
            logp = sample[2].unsqueeze(0).to(device)
            qed_score = sample[3].unsqueeze(0).to(device)

            if mode == 'vae':
                template_logits, sub_logits, _, _ = model(template_idx, sub_idx)
            else:
                template_logits, sub_logits, _, _, _ = model(
                    template_idx, sub_idx, logp, qed_score
                )

            pred_template = torch.argmax(template_logits, dim=-1)
            pred_sub = torch.argmax(sub_logits, dim=-1)

            if pred_template.item() == template_idx.item():
                correct_template += 1
            if torch.all(pred_sub == sub_idx):
                correct_sub += 1
            total += 1

    return correct_template / total, correct_sub / total


def plot_loss_curves(history, output_path):
    """绘制训练损失曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    # 总损失
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['test_loss'], 'r-', label='Test')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True)

    # 重建损失
    axes[1].plot(epochs, history['train_recon'], 'b-', label='Train')
    axes[1].plot(epochs, history['test_recon'], 'r-', label='Test')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Reconstruction Loss')
    axes[1].legend()
    axes[1].grid(True)

    # KL散度
    axes[2].plot(epochs, history['train_kl'], 'b-', label='Train')
    axes[2].plot(epochs, history['test_kl'], 'r-', label='Test')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KL Divergence')
    axes[2].set_title('KL Divergence')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'损失曲线: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Fragment-VAE/CVAE 训练')
    parser.add_argument('--mode', type=str, default='vae', choices=['vae', 'cvae'],
                        help='vae=无条件, cvae=有条件')
    parser.add_argument('--dataset', type=str, default='zinc_250k.csv')
    parser.add_argument('--max_molecules', type=int, default=200000)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--template_emb_dim', type=int, default=64)
    parser.add_argument('--substituent_emb_dim', type=int, default=64)
    parser.add_argument('--condition_emb_dim', type=int, default=32, help='条件嵌入维度（CVAE）')
    parser.add_argument('--kl_weight', type=float, default=0.3, help='KL损失权重(最终值)')
    parser.add_argument('--kl_start', type=float, default=0.0, help='KL退火起始权重')
    parser.add_argument('--kl_anneal', type=int, default=1500, help='KL退火epoch数')
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'mps', 'cpu'])
    parser.add_argument('--patience', type=int, default=1500, help='Early stopping patience')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device != 'cuda' else 'cpu')
    print(f'设备: {device}')
    print(f'模式: {args.mode.upper()}')

    # 加载数据
    print('\n===== 加载数据 =====')
    train_loader, test_loader, dataset = get_dataloader(
        batch_size=args.batch_size,
        dataset_file=args.dataset,
        max_molecules=args.max_molecules,
        cache_dir=args.output_dir,
        test_ratio=0.0  # 不划分测试集
    )

    # 获取配置
    config = get_config()
    print(f"\n片段库配置:")
    print(f"  - 模板数量: {config['num_templates']}")
    print(f"  - 取代基数量: {config['num_substituents']}")
    print(f"  - 最大取代位点: {config['max_r_count']}")

    # 创建模型
    print('\n===== 创建模型 =====')
    print(f'hidden_dim={args.hidden_dim}, latent_dim={args.latent_dim}')

    if args.mode == 'vae':
        model = FragmentVAE(
            config['num_templates'],
            config['num_substituents'],
            config['max_r_count'],
            args.template_emb_dim,
            args.substituent_emb_dim,
            args.hidden_dim,
            args.latent_dim
        ).to(device)
    else:
        model = FragmentCVAE(
            config['num_templates'],
            config['num_substituents'],
            config['max_r_count'],
            args.template_emb_dim,
            args.substituent_emb_dim,
            args.hidden_dim,
            args.latent_dim,
            args.condition_emb_dim
        ).to(device)

    print(f'参数量: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # 准备配置（将保存在模型文件中）
    config['mode'] = args.mode
    config['hidden_dim'] = args.hidden_dim
    config['latent_dim'] = args.latent_dim
    config['template_emb_dim'] = args.template_emb_dim
    config['substituent_emb_dim'] = args.substituent_emb_dim
    config['condition_emb_dim'] = args.condition_emb_dim

    print(f'配置准备完成，将随模型一起保存')

    # 训练
    print('\n===== 开始训练 =====')
    print(f'KL权重: {args.kl_weight}' + (f', 退火{args.kl_anneal}个epoch' if args.kl_anneal > 0 else ''))
    print('-' * 80)

    best_loss = float('inf')
    patience = args.patience  # Early stopping patience
    patience_counter = 0
    history = {
        'train_loss': [], 'train_recon': [], 'train_kl': [],
        'test_loss': [], 'test_recon': [], 'test_kl': []
    }

    for epoch in range(1, args.epochs + 1):
        # KL退火
        if args.kl_anneal > 0 and epoch <= args.kl_anneal:
            t = epoch / args.kl_anneal
            current_kl_weight = args.kl_start + (args.kl_weight - args.kl_start) * t
        else:
            current_kl_weight = args.kl_weight

        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, device, args.mode, current_kl_weight, epoch
        )
        scheduler.step()

        test_loss, test_recon, test_kl = test_epoch(
            model, test_loader, device, args.mode, current_kl_weight
        )

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_recon'].append(train_recon)
        history['train_kl'].append(train_kl)
        history['test_loss'].append(test_loss)
        history['test_recon'].append(test_recon)
        history['test_kl'].append(test_kl)

        kl_info = f' [kl_w={current_kl_weight:.3f}]' if epoch <= args.kl_anneal else ''
        print(f'Epoch {epoch:3d}{kl_info} | '
              f'Train: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f}) | '
              f'Test: {test_loss:.4f} (Recon: {test_recon:.4f}, KL: {test_kl:.4f})')

        # 每10个epoch评估重建准确率
        if epoch % 10 == 0:
            template_acc, sub_acc = evaluate_reconstruction(
                model, dataset, device, args.mode
            )
            print(f'         重建准确率: 模板={template_acc:.1%}, 取代基={sub_acc:.1%}')

        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            # 保存模型和配置
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config
            }, f'{args.output_dir}/{args.mode}_best.pt')
            print(f'         ✓ 保存最佳模型 (test_loss={test_loss:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping: 测试loss连续{patience}个epoch未改善')
                break

    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, f'{args.output_dir}/{args.mode}_final.pt')

    print(f'\n===== 完成 =====')
    print(f'最佳模型: {args.output_dir}/{args.mode}_best.pt')
    print(f'最终模型: {args.output_dir}/{args.mode}_final.pt')

    # 绘制损失曲线
    plot_loss_curves(history, f'{args.output_dir}/{args.mode}_loss.png')

    # 最终评估
    checkpoint = torch.load(f'{args.output_dir}/{args.mode}_best.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    template_acc, sub_acc = evaluate_reconstruction(model, dataset, device, args.mode, num_samples=500)
    print(f'\n最终重建准确率: 模板={template_acc:.1%}, 取代基={sub_acc:.1%}')


if __name__ == '__main__':
    main()
