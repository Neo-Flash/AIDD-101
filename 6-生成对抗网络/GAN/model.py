"""
GAN模型定义 - 基于模板填空的分子生成
生成器：噪声 → (模板索引, 取代基索引列表)
判别器：(模板索引, 取代基索引列表) → 真/假概率
"""

import torch
import torch.nn as nn
from data import NUM_TEMPLATES, NUM_SUBSTITUENTS, MAX_R_COUNT


class Generator(nn.Module):
    """
    生成器：将随机噪声转换为分子模板和取代基的选择

    输入：随机噪声向量 [batch_size, noise_dim]
    输出：
        - 模板概率分布 [batch_size, num_templates]
        - 取代基概率分布 [batch_size, max_r_count, num_substituents]
    """

    def __init__(self, noise_dim, num_templates, num_substituents, max_r_count,
                 hidden_dims=[256, 512, 256], dropout=0.3):
        """
        初始化生成器

        Args:
            noise_dim: 输入噪声的维度（如128）
            num_templates: 模板数量
            num_substituents: 取代基数量
            max_r_count: 最大取代位点数
            hidden_dims: 隐藏层维度列表
            dropout: Dropout比例
        """
        super(Generator, self).__init__()

        self.noise_dim = noise_dim
        self.num_templates = num_templates
        self.num_substituents = num_substituents
        self.max_r_count = max_r_count

        # 共享的MLP主干
        layers = []
        input_dim = noise_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self.shared_mlp = nn.Sequential(*layers)

        # 模板分支：输出模板选择
        self.template_head = nn.Linear(input_dim, num_templates)

        # 取代基分支：输出每个位置的取代基选择
        self.substituent_head = nn.Linear(input_dim, max_r_count * num_substituents)

    def forward(self, noise):
        """
        前向传播

        Args:
            noise: 随机噪声 [batch_size, noise_dim]

        Returns:
            template_logits: 模板概率分布 [batch_size, num_templates]
            substituent_logits: 取代基概率分布 [batch_size, max_r_count, num_substituents]
        """
        # 共享特征提取
        features = self.shared_mlp(noise)  # [batch_size, hidden_dim]

        # 模板分支
        template_logits = self.template_head(features)  # [batch_size, num_templates]

        # 取代基分支
        sub_logits = self.substituent_head(features)  # [batch_size, max_r_count * num_substituents]
        substituent_logits = sub_logits.view(-1, self.max_r_count, self.num_substituents)

        return template_logits, substituent_logits

    def generate(self, noise):
        """
        生成分子（用于推理）

        Args:
            noise: 随机噪声 [batch_size, noise_dim]

        Returns:
            template_indices: 模板索引 [batch_size]
            substituent_indices: 取代基索引 [batch_size, max_r_count]
        """
        with torch.no_grad():
            template_logits, substituent_logits = self.forward(noise)

            # 选择概率最高的模板
            template_indices = torch.argmax(template_logits, dim=-1)  # [batch_size]

            # 选择概率最高的取代基
            substituent_indices = torch.argmax(substituent_logits, dim=-1)  # [batch_size, max_r_count]

        return template_indices, substituent_indices


class Discriminator(nn.Module):
    """
    判别器：判断(模板, 取代基)的组合是真实的还是生成的

    输入：
        - 模板索引 [batch_size]
        - 取代基索引列表 [batch_size, max_r_count]
    输出：
        - 真假概率 [batch_size, 1]
    """

    def __init__(self, num_templates, num_substituents, max_r_count,
                 template_emb_dim=32, substituent_emb_dim=16,
                 hidden_dims=[128, 64], dropout=0.5):
        """
        初始化判别器

        Args:
            num_templates: 模板数量
            num_substituents: 取代基数量
            max_r_count: 最大取代位点数
            template_emb_dim: 模板Embedding维度
            substituent_emb_dim: 取代基Embedding维度
            hidden_dims: 隐藏层维度列表
            dropout: Dropout比例
        """
        super(Discriminator, self).__init__()

        self.num_templates = num_templates
        self.num_substituents = num_substituents
        self.max_r_count = max_r_count

        # Embedding层
        self.template_embedding = nn.Embedding(num_templates, template_emb_dim)
        self.substituent_embedding = nn.Embedding(num_substituents, substituent_emb_dim)

        # MLP网络
        layers = []
        input_dim = template_emb_dim + max_r_count * substituent_emb_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*layers)

    def forward(self, template_input, substituent_input, use_soft=False):
        """
        前向传播

        Args:
            template_input:
                - 如果use_soft=False: 模板索引 [batch_size]
                - 如果use_soft=True: 模板概率分布 [batch_size, num_templates]
            substituent_input:
                - 如果use_soft=False: 取代基索引 [batch_size, max_r_count]
                - 如果use_soft=True: 取代基概率分布 [batch_size, max_r_count, num_substituents]
            use_soft: 是否使用软概率（用于生成器训练时的梯度回传）

        Returns:
            output: 真假概率 [batch_size, 1]
        """
        if use_soft:
            # 软Embedding（用于生成器训练）
            # 模板embedding = 概率分布 × embedding矩阵
            template_emb = torch.matmul(template_input, self.template_embedding.weight)
            # [batch_size, template_emb_dim]

            # 取代基embedding = 概率分布 × embedding矩阵
            sub_emb = torch.matmul(substituent_input, self.substituent_embedding.weight)
            # [batch_size, max_r_count, substituent_emb_dim]
        else:
            # 硬Embedding（用于判别器训练真实/生成样本）
            template_emb = self.template_embedding(template_input)
            # [batch_size, template_emb_dim]

            sub_emb = self.substituent_embedding(substituent_input)
            # [batch_size, max_r_count, substituent_emb_dim]

        # 拼接特征
        sub_emb_flat = sub_emb.view(sub_emb.size(0), -1)  # [batch_size, max_r_count * substituent_emb_dim]
        combined = torch.cat([template_emb, sub_emb_flat], dim=1)  # [batch_size, total_dim]

        # 通过MLP
        output = self.mlp(combined)  # [batch_size, 1]

        return output


class MoleculeGAN:
    """
    完整的GAN模型包装类
    """

    def __init__(self, num_templates, num_substituents, max_r_count, noise_dim=128, device='cpu',
                 g_hidden_dims=[256, 512, 256], g_dropout=0.3,
                 d_template_emb=32, d_substituent_emb=16, d_hidden_dims=[128, 64], d_dropout=0.5):
        """
        初始化GAN模型

        Args:
            num_templates: 模板数量
            num_substituents: 取代基数量
            max_r_count: 最大取代位点数
            noise_dim: 噪声维度
            device: 设备
            g_hidden_dims: 生成器隐藏层
            g_dropout: 生成器Dropout
            d_template_emb: 判别器模板Embedding维度
            d_substituent_emb: 判别器取代基Embedding维度
            d_hidden_dims: 判别器隐藏层
            d_dropout: 判别器Dropout
        """
        self.num_templates = num_templates
        self.num_substituents = num_substituents
        self.max_r_count = max_r_count
        self.noise_dim = noise_dim
        self.device = device

        # 创建生成器和判别器
        self.generator = Generator(
            noise_dim=noise_dim,
            num_templates=num_templates,
            num_substituents=num_substituents,
            max_r_count=max_r_count,
            hidden_dims=g_hidden_dims,
            dropout=g_dropout
        ).to(device)

        self.discriminator = Discriminator(
            num_templates=num_templates,
            num_substituents=num_substituents,
            max_r_count=max_r_count,
            template_emb_dim=d_template_emb,
            substituent_emb_dim=d_substituent_emb,
            hidden_dims=d_hidden_dims,
            dropout=d_dropout
        ).to(device)

        print(f"GAN模型已创建：")
        print(f"  - 生成器参数量: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"  - 判别器参数量: {sum(p.numel() for p in self.discriminator.parameters()):,}")

    def sample_noise(self, batch_size):
        """
        采样随机噪声

        Args:
            batch_size: 批次大小

        Returns:
            noise: 随机噪声 [batch_size, noise_dim]
        """
        return torch.randn(batch_size, self.noise_dim, device=self.device)


if __name__ == '__main__':
    # 测试代码
    print("测试GAN模型...\n")

    batch_size = 4

    # 创建模型
    gan = MoleculeGAN(
        num_templates=NUM_TEMPLATES,
        num_substituents=NUM_SUBSTITUENTS,
        max_r_count=MAX_R_COUNT,
        noise_dim=128,
        device='cpu'
    )

    # 测试生成器
    noise = gan.sample_noise(batch_size)
    template_logits, sub_logits = gan.generator(noise)
    print(f"\n生成器输出:")
    print(f"  - 模板logits shape: {template_logits.shape}")  # [4, num_templates]
    print(f"  - 取代基logits shape: {sub_logits.shape}")  # [4, max_r_count, num_substituents]

    template_indices, sub_indices = gan.generator.generate(noise)
    print(f"\n生成的索引:")
    print(f"  - 模板索引 shape: {template_indices.shape}")  # [4]
    print(f"  - 取代基索引 shape: {sub_indices.shape}")  # [4, max_r_count]
    print(f"  - 示例: 模板{template_indices[0].item()}, 取代基{sub_indices[0].tolist()}")

    # 测试判别器
    real_scores = gan.discriminator(template_indices, sub_indices, use_soft=False)
    print(f"\n判别器输出:")
    print(f"  - 分数 shape: {real_scores.shape}")  # [4, 1]
    print(f"  - 分数范围: [{real_scores.min().item():.3f}, {real_scores.max().item():.3f}]")
