"""
model.py - 基于片段的VAE和CVAE模型
VAE: 无条件分子生成
CVAE: 有条件分子生成（指定logP和QED）

输入: (模板索引, 取代基索引列表)
输出: (模板概率分布, 取代基概率分布)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FragmentEncoder(nn.Module):
    """
    片段编码器：将(模板, 取代基)编码为潜在向量

    输入：
        - 模板索引 [batch]
        - 取代基索引 [batch, max_r_count]
        - 条件 [batch, 2]（可选，logP和QED）

    输出：
        - mu [batch, latent_dim]
        - logvar [batch, latent_dim]
    """

    def __init__(self, num_templates, num_substituents, max_r_count,
                 template_emb_dim=64, substituent_emb_dim=32,
                 hidden_dim=256, latent_dim=128, use_condition=False, condition_dim=2):
        super().__init__()

        self.use_condition = use_condition

        # Embedding层
        self.template_embedding = nn.Embedding(num_templates, template_emb_dim)
        self.substituent_embedding = nn.Embedding(num_substituents, substituent_emb_dim)

        # 计算输入维度
        input_dim = template_emb_dim + max_r_count * substituent_emb_dim
        if use_condition:
            input_dim += condition_dim  # 使用可变的条件维度

        # MLP编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # 均值和方差
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, template_idx, substituent_idx, condition=None):
        # Embedding
        template_emb = self.template_embedding(template_idx)  # [B, template_emb_dim]
        sub_emb = self.substituent_embedding(substituent_idx)  # [B, max_r_count, sub_emb_dim]
        sub_emb_flat = sub_emb.view(sub_emb.size(0), -1)  # [B, max_r_count * sub_emb_dim]

        # 拼接
        x = torch.cat([template_emb, sub_emb_flat], dim=-1)

        if self.use_condition and condition is not None:
            x = torch.cat([x, condition], dim=-1)

        # 编码
        h = self.encoder(x)

        return self.fc_mu(h), self.fc_logvar(h)


class FragmentDecoder(nn.Module):
    """
    片段解码器：从潜在向量重建(模板, 取代基)

    输入：
        - z [batch, latent_dim]
        - 条件 [batch, 2]（可选）

    输出：
        - 模板logits [batch, num_templates]
        - 取代基logits [batch, max_r_count, num_substituents]
    """

    def __init__(self, latent_dim, num_templates, num_substituents, max_r_count,
                 hidden_dim=256, use_condition=False, condition_dim=2):
        super().__init__()

        self.num_templates = num_templates
        self.num_substituents = num_substituents
        self.max_r_count = max_r_count

        input_dim = latent_dim + condition_dim if use_condition else latent_dim

        # 共享解码器
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # 模板输出头
        self.template_head = nn.Linear(hidden_dim, num_templates)

        # 取代基输出头
        self.substituent_head = nn.Linear(hidden_dim, max_r_count * num_substituents)

    def forward(self, z, condition=None):
        if condition is not None:
            z = torch.cat([z, condition], dim=-1)

        h = self.decoder(z)

        # 模板logits
        template_logits = self.template_head(h)

        # 取代基logits
        sub_logits = self.substituent_head(h)
        substituent_logits = sub_logits.view(-1, self.max_r_count, self.num_substituents)

        return template_logits, substituent_logits


class FragmentVAE(nn.Module):
    """基于片段的VAE：无条件分子生成"""

    def __init__(self, num_templates, num_substituents, max_r_count,
                 template_emb_dim=64, substituent_emb_dim=32,
                 hidden_dim=256, latent_dim=128):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_templates = num_templates
        self.num_substituents = num_substituents
        self.max_r_count = max_r_count

        self.encoder = FragmentEncoder(
            num_templates, num_substituents, max_r_count,
            template_emb_dim, substituent_emb_dim,
            hidden_dim, latent_dim, use_condition=False
        )

        self.decoder = FragmentDecoder(
            latent_dim, num_templates, num_substituents, max_r_count,
            hidden_dim, use_condition=False
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, template_idx, substituent_idx):
        # 编码
        mu, logvar = self.encoder(template_idx, substituent_idx)

        # 重参数化
        z = self.reparameterize(mu, logvar)

        # 解码
        template_logits, substituent_logits = self.decoder(z)

        return template_logits, substituent_logits, mu, logvar

    def generate(self, num_samples, device):
        """从随机噪声生成分子"""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        template_logits, substituent_logits = self.decoder(z)
        return template_logits, substituent_logits

    def sample(self, num_samples, device):
        """生成分子索引（用于推理）"""
        template_logits, substituent_logits = self.generate(num_samples, device)

        # 取argmax
        template_idx = torch.argmax(template_logits, dim=-1)
        substituent_idx = torch.argmax(substituent_logits, dim=-1)

        return template_idx, substituent_idx


class FragmentCVAE(nn.Module):
    """基于片段的CVAE：有条件分子生成（指定logP和QED）

    改进版：
    1. 条件嵌入层：将2维条件映射到32维，增强条件信号
    2. 属性预测器：显式预测logP和QED，强制模型学习条件-性质关系
    """

    def __init__(self, num_templates, num_substituents, max_r_count,
                 template_emb_dim=64, substituent_emb_dim=32,
                 hidden_dim=256, latent_dim=128, condition_emb_dim=32):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_templates = num_templates
        self.num_substituents = num_substituents
        self.max_r_count = max_r_count
        self.condition_emb_dim = condition_emb_dim

        # 方案1：条件嵌入层 - 将2维条件映射到高维空间
        self.condition_embedding = nn.Sequential(
            nn.Linear(2, condition_emb_dim),
            nn.ReLU(),
            nn.Linear(condition_emb_dim, condition_emb_dim),
            nn.ReLU()
        )

        self.encoder = FragmentEncoder(
            num_templates, num_substituents, max_r_count,
            template_emb_dim, substituent_emb_dim,
            hidden_dim, latent_dim, use_condition=True, condition_dim=condition_emb_dim
        )

        self.decoder = FragmentDecoder(
            latent_dim, num_templates, num_substituents, max_r_count,
            hidden_dim, use_condition=True, condition_dim=condition_emb_dim
        )

        # 方案2：属性预测器 - 从潜在向量预测logP和QED
        self.property_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # 输出logP和QED的预测值
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _normalize_condition(self, logp, qed_score):
        """归一化条件"""
        # logP范围约 -2 ~ 8，归一化到 0~1
        logp_norm = ((logp - (-2)) / 10).clamp(0, 1)
        # QED范围 0 ~ 1，直接使用
        qed_norm = qed_score.clamp(0, 1)
        return torch.stack([logp_norm, qed_norm], dim=-1)

    def forward(self, template_idx, substituent_idx, logp, qed_score):
        # 归一化条件
        condition_raw = self._normalize_condition(logp, qed_score)

        # 方案1：嵌入条件，增强信号强度
        condition_emb = self.condition_embedding(condition_raw)

        # 编码
        mu, logvar = self.encoder(template_idx, substituent_idx, condition_emb)

        # 重参数化
        z = self.reparameterize(mu, logvar)

        # 解码
        template_logits, substituent_logits = self.decoder(z, condition_emb)

        # 方案2：从潜在向量预测属性
        predicted_properties = self.property_predictor(z)

        return template_logits, substituent_logits, mu, logvar, predicted_properties

    def generate(self, num_samples, logp, qed_score, device):
        """从随机噪声和条件生成分子"""
        z = torch.randn(num_samples, self.latent_dim, device=device)

        # 处理条件
        if not isinstance(logp, torch.Tensor):
            logp = torch.full((num_samples,), logp, device=device)
        if not isinstance(qed_score, torch.Tensor):
            qed_score = torch.full((num_samples,), qed_score, device=device)

        condition_raw = self._normalize_condition(logp, qed_score)
        condition_emb = self.condition_embedding(condition_raw)

        template_logits, substituent_logits = self.decoder(z, condition_emb)
        return template_logits, substituent_logits

    def sample(self, num_samples, logp, qed_score, device):
        """生成分子索引（用于推理）"""
        template_logits, substituent_logits = self.generate(
            num_samples, logp, qed_score, device
        )

        # 取argmax
        template_idx = torch.argmax(template_logits, dim=-1)
        substituent_idx = torch.argmax(substituent_logits, dim=-1)

        return template_idx, substituent_idx


def vae_loss(template_logits, substituent_logits,
             target_template, target_substituent,
             mu, logvar, kl_weight=0.1, predicted_properties=None, target_properties=None, property_weight=1.0):
    """
    VAE损失 = 重建损失 + KL散度 + 属性预测损失（CVAE）

    Args:
        template_logits: 模板预测 [B, num_templates]
        substituent_logits: 取代基预测 [B, max_r_count, num_substituents]
        target_template: 目标模板索引 [B]
        target_substituent: 目标取代基索引 [B, max_r_count]
        mu, logvar: 潜在分布参数
        kl_weight: KL散度权重
        predicted_properties: 预测的属性 [B, 2] (logP, QED) - CVAE专用
        target_properties: 目标属性 [B, 2] (logP, QED) - CVAE专用
        property_weight: 属性预测损失权重

    Returns:
        total_loss, recon_loss, kl_loss, property_loss (如果是CVAE)
    """
    batch_size = template_logits.shape[0]

    # 模板损失（交叉熵）
    template_loss = F.cross_entropy(template_logits, target_template, reduction='mean')

    # 取代基损失（每个位置分别计算交叉熵）
    # substituent_logits: [B, max_r_count, num_substituents]
    # target_substituent: [B, max_r_count]
    sub_logits_flat = substituent_logits.view(-1, substituent_logits.shape[-1])
    sub_target_flat = target_substituent.view(-1)
    substituent_loss = F.cross_entropy(sub_logits_flat, sub_target_flat, reduction='mean')

    # 重建损失（模板权重更高）
    recon_loss = 2.0 * template_loss + substituent_loss

    # KL散度
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # 属性预测损失（方案2：辅助损失）
    if predicted_properties is not None and target_properties is not None:
        # MSE损失：预测的logP和QED vs 真实值
        property_loss = F.mse_loss(predicted_properties, target_properties)
        total_loss = recon_loss + kl_weight * kl_loss + property_weight * property_loss
        return total_loss, recon_loss, kl_loss, property_loss
    else:
        # VAE模式，无属性损失
        total_loss = recon_loss + kl_weight * kl_loss
        return total_loss, recon_loss, kl_loss


if __name__ == '__main__':
    # 测试代码
    print("测试Fragment VAE模型...\n")

    batch_size = 4
    num_templates = 100
    num_substituents = 50
    max_r_count = 3

    # 创建VAE
    vae = FragmentVAE(num_templates, num_substituents, max_r_count)
    print(f"VAE参数量: {sum(p.numel() for p in vae.parameters()):,}")

    # 创建CVAE
    cvae = FragmentCVAE(num_templates, num_substituents, max_r_count)
    print(f"CVAE参数量: {sum(p.numel() for p in cvae.parameters()):,}")

    # 测试数据
    template_idx = torch.randint(0, num_templates, (batch_size,))
    sub_idx = torch.randint(0, num_substituents, (batch_size, max_r_count))
    logp = torch.randn(batch_size) * 2 + 2
    qed = torch.rand(batch_size)

    # VAE前向传播
    template_logits, sub_logits, mu, logvar = vae(template_idx, sub_idx)
    print(f"\nVAE输出:")
    print(f"  - 模板logits: {template_logits.shape}")
    print(f"  - 取代基logits: {sub_logits.shape}")
    print(f"  - mu: {mu.shape}")

    # CVAE前向传播
    template_logits, sub_logits, mu, logvar = cvae(template_idx, sub_idx, logp, qed)
    print(f"\nCVAE输出:")
    print(f"  - 模板logits: {template_logits.shape}")
    print(f"  - 取代基logits: {sub_logits.shape}")

    # 测试生成
    gen_template, gen_sub = vae.sample(10, 'cpu')
    print(f"\nVAE生成:")
    print(f"  - 模板索引: {gen_template.shape}")
    print(f"  - 取代基索引: {gen_sub.shape}")

    # 测试损失
    loss, recon, kl = vae_loss(template_logits, sub_logits, template_idx, sub_idx, mu, logvar)
    print(f"\n损失: total={loss:.4f}, recon={recon:.4f}, kl={kl:.4f}")
