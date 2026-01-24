"""
train.py - 基于片段的扩散模型训练脚本
--mode diffusion  : 无条件生成
--mode cdiffusion : 有条件生成（使用logP和QED作为条件）
"""

import torch
import torch.optim as optim
import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from data import get_dataloader, get_config
from model import FragmentDiffusion, FragmentCDiffusion


def train_epoch(model, dataloader, optimizer, device, mode='diffusion', epoch=0):
    """训练一个epoch"""
    model.train()
    total_loss = 0
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

            if mode == 'diffusion':
                loss = model(template_idx, sub_idx)
            else:
                loss = model(template_idx, sub_idx, logp, qed_score)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        except Exception as e:
            print(f"\n错误在batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            raise

    return total_loss / num_batches


def test_epoch(model, dataloader, device, mode='diffusion'):
    """测试一个epoch"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            template_idx, sub_idx, logp, qed_score = batch
            template_idx = template_idx.to(device)
            sub_idx = sub_idx.to(device)
            logp = logp.to(device)
            qed_score = qed_score.to(device)

            if mode == 'diffusion':
                loss = model(template_idx, sub_idx)
            else:
                loss = model(template_idx, sub_idx, logp, qed_score)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(1, num_batches)


def plot_loss_curves(history, output_path):
    """绘制训练损失曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(history['train_loss']) + 1)

    ax.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['test_loss'], 'r-', label='Test', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Training Loss Curve', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'损失曲线: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='片段扩散模型训练')
    parser.add_argument('--mode', type=str, default='cdiffusion', choices=['diffusion', 'cdiffusion'],
                        help='diffusion=无条件, cdiffusion=有条件')
    parser.add_argument('--dataset', type=str, default='zinc_250k.csv')
    parser.add_argument('--max_molecules', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_timesteps', type=int, default=1000, help='扩散步数')
    parser.add_argument('--beta_schedule', type=str, default='cosine', choices=['linear', 'cosine'])
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'mps', 'cpu'])
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
        shuffle=True,
        dataset_file=args.dataset,
        max_molecules=args.max_molecules,
        cache_dir=args.output_dir,
        test_ratio=0.1
    )

    # 获取配置
    config = get_config()
    print(f"\n片段库配置:")
    print(f"  - 模板数量: {config['num_templates']}")
    print(f"  - 取代基数量: {config['num_substituents']}")
    print(f"  - 最大取代位点: {config['max_r_count']}")

    # 确保片段库数据保存到config中（用于生成时恢复）
    import data
    config['templates'] = data.TEMPLATES
    config['substituents'] = data.SUBSTITUENTS
    config['template_r_counts'] = data.TEMPLATE_R_COUNTS

    # 创建模型
    print('\n===== 创建模型 =====')
    print(f'emb_dim={args.emb_dim}, hidden_dim={args.hidden_dim}')
    print(f'num_timesteps={args.num_timesteps}, beta_schedule={args.beta_schedule}')

    if args.mode == 'diffusion':
        model = FragmentDiffusion(
            num_templates=config['num_templates'],
            num_substituents=config['num_substituents'],
            max_r_count=config['max_r_count'],
            emb_dim=args.emb_dim,
            hidden_dim=args.hidden_dim,
            num_timesteps=args.num_timesteps,
            beta_schedule=args.beta_schedule
        ).to(device)
    else:
        model = FragmentCDiffusion(
            num_templates=config['num_templates'],
            num_substituents=config['num_substituents'],
            max_r_count=config['max_r_count'],
            emb_dim=args.emb_dim,
            hidden_dim=args.hidden_dim,
            num_timesteps=args.num_timesteps,
            beta_schedule=args.beta_schedule
        ).to(device)

    print(f'参数量: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 训练
    print('\n===== 开始训练 =====')
    print('-' * 80)

    best_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'test_loss': []}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, args.mode, epoch)
        scheduler.step()

        test_loss = test_epoch(model, test_loader, device, args.mode)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)

        print(f'Epoch {epoch:3d} | Train: {train_loss:.4f} | Test: {test_loss:.4f}')

        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            # 保存模型和配置
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'model_config': {
                    'mode': args.mode,
                    'emb_dim': args.emb_dim,
                    'hidden_dim': args.hidden_dim,
                    'num_timesteps': args.num_timesteps,
                    'beta_schedule': args.beta_schedule
                }
            }, f'{args.output_dir}/{args.mode}_best.pt')
            print(f'         ✓ 保存最佳模型 (test_loss={test_loss:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'\nEarly stopping: 测试loss连续{args.patience}个epoch未改善')
                break

    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'model_config': {
            'mode': args.mode,
            'emb_dim': args.emb_dim,
            'hidden_dim': args.hidden_dim,
            'num_timesteps': args.num_timesteps,
            'beta_schedule': args.beta_schedule
        }
    }, f'{args.output_dir}/{args.mode}_final.pt')

    print(f'\n===== 完成 =====')
    print(f'最佳模型: {args.output_dir}/{args.mode}_best.pt')
    print(f'最终模型: {args.output_dir}/{args.mode}_final.pt')

    # 绘制损失曲线
    plot_loss_curves(history, f'{args.output_dir}/{args.mode}_loss.png')


if __name__ == '__main__':
    main()
