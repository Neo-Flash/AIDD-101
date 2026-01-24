"""
model.py - 基于Embedding的扩散模型（教学版）

核心思想：保留模板+取代基表示，但用连续扩散避免巨大矩阵

流程：
1. 模板ID/取代基ID → Embedding（连续向量）
2. 在Embedding空间做高斯扩散
3. 生成时：噪声 → Embedding → 找最近邻ID

优点：
- 保留模板+取代基的清晰结构
- 连续扩散，速度快，无需巨大转移矩阵
- 可直接生成有效分子
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==================== 扩散调度 ====================

def get_beta_schedule(schedule, num_timesteps, beta_start=1e-4, beta_end=0.02):
    """生成噪声调度表"""
    if schedule == 'linear':
        return np.linspace(beta_start, beta_end, num_timesteps)
    elif schedule == 'cosine':
        s = 0.008
        steps = num_timesteps + 1
        x = np.linspace(0, num_timesteps, steps)
        alphas_cumprod = np.cos(((x / num_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0, 0.999)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


# ==================== 去噪网络 ====================

class FragmentDenoiser(nn.Module):
    """
    片段去噪网络：预测噪声

    输入：加噪的embeddings + 时间步 t + 条件（可选）
    输出：预测的噪声
    """

    def __init__(self, emb_dim, hidden_dim, max_r_count, use_condition=False):
        super().__init__()

        self.emb_dim = emb_dim
        self.max_r_count = max_r_count
        self.use_condition = use_condition

        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.SiLU(),
            nn.Linear(emb_dim * 2, emb_dim)
        )

        # 条件嵌入（可选）
        if use_condition:
            self.condition_mlp = nn.Sequential(
                nn.Linear(2, 64),
                nn.SiLU(),
                nn.Linear(64, emb_dim)
            )

        # 输入：template_emb + sub_emb_flattened
        input_dim = emb_dim * (1 + max_r_count)

        # 去噪网络
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)  # 输出噪声
        )

    def get_time_embedding(self, timesteps):
        """Sinusoidal 时间嵌入"""
        half_dim = self.emb_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.emb_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, x, t, condition=None):
        """
        Args:
            x: [B, (1+max_r_count)*emb_dim] 加噪的embeddings（拼接后）
            t: [B] 时间步
            condition: [B, 2] 条件（logP, QED）

        Returns:
            noise_pred: [B, (1+max_r_count)*emb_dim] 预测的噪声
        """
        # 时间嵌入
        t_emb = self.get_time_embedding(t)
        t_emb = self.time_mlp(t_emb)  # [B, emb_dim]

        # 广播到所有位置
        t_emb_expanded = t_emb.unsqueeze(1).expand(-1, 1 + self.max_r_count, -1)  # [B, 1+max_r, emb_dim]
        t_emb_flat = t_emb_expanded.reshape(x.size(0), -1)  # [B, (1+max_r)*emb_dim]

        h = x + t_emb_flat

        # 加入条件（可选）
        if self.use_condition and condition is not None:
            c_emb = self.condition_mlp(condition)  # [B, emb_dim]
            c_emb_expanded = c_emb.unsqueeze(1).expand(-1, 1 + self.max_r_count, -1)
            c_emb_flat = c_emb_expanded.reshape(x.size(0), -1)
            h = h + c_emb_flat

        # 预测噪声
        noise_pred = self.net(h)

        return noise_pred


# ==================== 扩散模型 ====================

class FragmentDiffusion(nn.Module):
    """片段扩散模型：无条件生成"""

    def __init__(self, num_templates, num_substituents, max_r_count,
                 emb_dim=64, hidden_dim=256, num_timesteps=500, beta_schedule='cosine'):
        super().__init__()

        self.num_templates = num_templates
        self.num_substituents = num_substituents
        self.max_r_count = max_r_count
        self.emb_dim = emb_dim
        self.num_timesteps = num_timesteps

        # Embeddings（共享）
        self.template_embedding = nn.Embedding(num_templates, emb_dim)
        self.substituent_embedding = nn.Embedding(num_substituents, emb_dim)

        # 去噪网络
        self.denoiser = FragmentDenoiser(
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            max_r_count=max_r_count,
            use_condition=False
        )

        # 扩散参数
        betas = get_beta_schedule(beta_schedule, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        self.register_buffer('betas', torch.from_numpy(betas).float())
        self.register_buffer('alphas', torch.from_numpy(alphas).float())
        self.register_buffer('alphas_cumprod', torch.from_numpy(alphas_cumprod).float())
        self.register_buffer('sqrt_alphas_cumprod', torch.from_numpy(np.sqrt(alphas_cumprod)).float())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.from_numpy(np.sqrt(1 - alphas_cumprod)).float())

    def q_sample(self, x_emb, t, noise=None):
        """
        前向加噪：q(x_t | x_0)

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        if noise is None:
            noise = torch.randn_like(x_emb)

        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]

        x_t = sqrt_alpha_bar_t * x_emb + sqrt_one_minus_alpha_bar_t * noise

        return x_t

    def forward(self, template_idx, sub_idx):
        """
        训练：计算损失

        Args:
            template_idx: [B] 模板索引
            sub_idx: [B, max_r_count] 取代基索引

        Returns:
            loss: 标量
        """
        batch_size = template_idx.size(0)
        device = template_idx.device

        # 1. ID → Embedding
        template_emb = self.template_embedding(template_idx)  # [B, emb_dim]
        sub_emb = self.substituent_embedding(sub_idx)  # [B, max_r_count, emb_dim]

        # 拼接为一个向量
        x_0 = torch.cat([template_emb.unsqueeze(1), sub_emb], dim=1)  # [B, 1+max_r_count, emb_dim]
        x_0 = x_0.reshape(batch_size, -1)  # [B, (1+max_r_count)*emb_dim]

        # 2. 随机采样时间步
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)

        # 3. 采样噪声
        noise = torch.randn_like(x_0)

        # 4. 前向加噪
        x_t = self.q_sample(x_0, t, noise)

        # 5. 预测噪声
        noise_pred = self.denoiser(x_t, t)

        # 6. 计算损失（MSE）
        loss = F.mse_loss(noise_pred, noise)

        return loss

    @torch.no_grad()
    def p_sample(self, x_t, t):
        """反向去噪一步：p(x_{t-1} | x_t)"""
        noise_pred = self.denoiser(x_t, t)

        alpha_t = self.alphas[t][:, None]
        alpha_bar_t = self.alphas_cumprod[t][:, None]
        beta_t = self.betas[t][:, None]

        mean = (x_t - beta_t / torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_t)

        if t[0] > 0:
            noise = torch.randn_like(x_t)
            sigma_t = torch.sqrt(beta_t)
            x_prev = mean + sigma_t * noise
        else:
            x_prev = mean

        return x_prev

    @torch.no_grad()
    def sample(self, num_samples, device, return_trajectory=False, trajectory_interval=100, sampling_steps=None):
        """
        采样：从噪声生成分子

        Args:
            num_samples: 生成样本数
            device: 设备
            return_trajectory: 是否返回去噪轨迹
            trajectory_interval: 轨迹采样间隔（每隔多少步记录一次）
            sampling_steps: 采样步数（None表示使用全部训练步数）

        Returns:
            template_idx: [num_samples] 生成的模板ID
            sub_idx: [num_samples, max_r_count] 生成的取代基ID
            trajectory: (可选) 轨迹列表，每个元素是 (t, template_idx, sub_idx)
        """
        from tqdm import tqdm

        # 确定采样步数
        if sampling_steps is None:
            sampling_steps = self.num_timesteps
            time_steps = list(reversed(range(self.num_timesteps)))
        else:
            # 使用DDIM式跳步采样
            step_size = self.num_timesteps // sampling_steps
            time_steps = list(reversed(range(0, self.num_timesteps, step_size)))
            if time_steps[0] != self.num_timesteps - 1:
                time_steps = [self.num_timesteps - 1] + time_steps
            if time_steps[-1] != 0:
                time_steps.append(0)

        # 1. 从标准正态分布采样
        dim = (1 + self.max_r_count) * self.emb_dim
        x_t = torch.randn(num_samples, dim, device=device)

        trajectory = [] if return_trajectory else None

        # 2. 反向去噪
        for t_idx in tqdm(time_steps, desc='采样', disable=return_trajectory):
            t = torch.full((num_samples,), t_idx, device=device, dtype=torch.long)
            x_t = self.p_sample(x_t, t)

            # 记录轨迹
            if return_trajectory and (t_idx % trajectory_interval == 0 or t_idx == 0):
                # 解码当前状态 - 使用概率采样而非最近邻，增加变化性
                x_tmp = x_t.reshape(num_samples, 1 + self.max_r_count, self.emb_dim)
                template_emb = x_tmp[:, 0, :]
                sub_emb = x_tmp[:, 1:, :]

                # 计算距离并转换为概率分布（温度系数控制随机性）
                temperature = max(0.1, t_idx / self.num_timesteps)  # 早期更随机，后期更确定

                template_dist = torch.cdist(template_emb, self.template_embedding.weight)
                template_logits = -template_dist / temperature
                template_probs = F.softmax(template_logits, dim=-1)
                template_idx_tmp = torch.multinomial(template_probs, 1).squeeze(-1)

                sub_emb_flat = sub_emb.reshape(-1, self.emb_dim)
                sub_dist = torch.cdist(sub_emb_flat, self.substituent_embedding.weight)
                sub_logits = -sub_dist / temperature
                sub_probs = F.softmax(sub_logits, dim=-1)
                sub_idx_tmp = torch.multinomial(sub_probs, 1).squeeze(-1).reshape(num_samples, self.max_r_count)

                trajectory.append((t_idx, template_idx_tmp.cpu(), sub_idx_tmp.cpu()))

        # 3. Embedding → ID（找最近邻）
        # 拆分模板和取代基
        x_t = x_t.reshape(num_samples, 1 + self.max_r_count, self.emb_dim)
        template_emb = x_t[:, 0, :]  # [B, emb_dim]
        sub_emb = x_t[:, 1:, :]  # [B, max_r_count, emb_dim]

        # 找最近邻模板
        template_dist = torch.cdist(template_emb, self.template_embedding.weight)  # [B, num_templates]
        template_idx = template_dist.argmin(dim=-1)  # [B]

        # 找最近邻取代基
        sub_emb_flat = sub_emb.reshape(-1, self.emb_dim)  # [B*max_r_count, emb_dim]
        sub_dist = torch.cdist(sub_emb_flat, self.substituent_embedding.weight)  # [B*max_r, num_subs]
        sub_idx = sub_dist.argmin(dim=-1)  # [B*max_r]
        sub_idx = sub_idx.reshape(num_samples, self.max_r_count)  # [B, max_r]

        if return_trajectory:
            return template_idx, sub_idx, trajectory
        else:
            return template_idx, sub_idx


class FragmentCDiffusion(FragmentDiffusion):
    """片段条件扩散模型"""

    def __init__(self, num_templates, num_substituents, max_r_count,
                 emb_dim=64, hidden_dim=256, num_timesteps=500, beta_schedule='cosine'):
        # 调用父类初始化大部分组件
        super().__init__(num_templates, num_substituents, max_r_count,
                        emb_dim, hidden_dim, num_timesteps, beta_schedule)

        # 替换为条件去噪网络
        self.denoiser = FragmentDenoiser(
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            max_r_count=max_r_count,
            use_condition=True
        )

    def _normalize_condition(self, logp, qed):
        """归一化条件"""
        logp_norm = (logp + 2.0) / 6.0
        qed_norm = qed
        return torch.stack([logp_norm, qed_norm], dim=-1)

    def forward(self, template_idx, sub_idx, logp, qed):
        """训练（带条件）"""
        batch_size = template_idx.size(0)
        device = template_idx.device

        # 处理条件
        condition = self._normalize_condition(logp, qed)

        # ID → Embedding
        template_emb = self.template_embedding(template_idx)
        sub_emb = self.substituent_embedding(sub_idx)

        x_0 = torch.cat([template_emb.unsqueeze(1), sub_emb], dim=1)
        x_0 = x_0.reshape(batch_size, -1)

        # 随机采样时间步
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)

        # 采样噪声
        noise = torch.randn_like(x_0)

        # 前向加噪
        x_t = self.q_sample(x_0, t, noise)

        # 预测噪声（带条件）
        noise_pred = self.denoiser(x_t, t, condition)

        # 计算损失
        loss = F.mse_loss(noise_pred, noise)

        return loss

    @torch.no_grad()
    def p_sample(self, x_t, t, condition):
        """反向去噪一步（带条件）"""
        noise_pred = self.denoiser(x_t, t, condition)

        alpha_t = self.alphas[t][:, None]
        alpha_bar_t = self.alphas_cumprod[t][:, None]
        beta_t = self.betas[t][:, None]

        mean = (x_t - beta_t / torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_t)

        if t[0] > 0:
            noise = torch.randn_like(x_t)
            sigma_t = torch.sqrt(beta_t)
            x_prev = mean + sigma_t * noise
        else:
            x_prev = mean

        return x_prev

    @torch.no_grad()
    def sample(self, num_samples, logp, qed, device, return_trajectory=False, trajectory_interval=100, sampling_steps=None):
        """采样（带条件）"""
        from tqdm import tqdm

        # 处理条件
        if not isinstance(logp, torch.Tensor):
            logp = torch.full((num_samples,), logp, device=device)
        if not isinstance(qed, torch.Tensor):
            qed = torch.full((num_samples,), qed, device=device)

        condition = self._normalize_condition(logp, qed)

        # 确定采样步数
        if sampling_steps is None:
            sampling_steps = self.num_timesteps
            time_steps = list(reversed(range(self.num_timesteps)))
        else:
            # 使用DDIM式跳步采样
            step_size = self.num_timesteps // sampling_steps
            time_steps = list(reversed(range(0, self.num_timesteps, step_size)))
            if time_steps[0] != self.num_timesteps - 1:
                time_steps = [self.num_timesteps - 1] + time_steps
            if time_steps[-1] != 0:
                time_steps.append(0)

        # 从标准正态分布采样
        dim = (1 + self.max_r_count) * self.emb_dim
        x_t = torch.randn(num_samples, dim, device=device)

        trajectory = [] if return_trajectory else None

        # 反向去噪
        for t_idx in tqdm(time_steps, desc='条件采样', disable=return_trajectory):
            t = torch.full((num_samples,), t_idx, device=device, dtype=torch.long)
            x_t = self.p_sample(x_t, t, condition)

            # 记录轨迹
            if return_trajectory and (t_idx % trajectory_interval == 0 or t_idx == 0):
                # 解码当前状态 - 使用概率采样而非最近邻，增加变化性
                x_tmp = x_t.reshape(num_samples, 1 + self.max_r_count, self.emb_dim)
                template_emb = x_tmp[:, 0, :]
                sub_emb = x_tmp[:, 1:, :]

                # 计算距离并转换为概率分布（温度系数控制随机性）
                temperature = max(0.1, t_idx / self.num_timesteps)  # 早期更随机，后期更确定

                template_dist = torch.cdist(template_emb, self.template_embedding.weight)
                template_logits = -template_dist / temperature
                template_probs = F.softmax(template_logits, dim=-1)
                template_idx_tmp = torch.multinomial(template_probs, 1).squeeze(-1)

                sub_emb_flat = sub_emb.reshape(-1, self.emb_dim)
                sub_dist = torch.cdist(sub_emb_flat, self.substituent_embedding.weight)
                sub_logits = -sub_dist / temperature
                sub_probs = F.softmax(sub_logits, dim=-1)
                sub_idx_tmp = torch.multinomial(sub_probs, 1).squeeze(-1).reshape(num_samples, self.max_r_count)

                trajectory.append((t_idx, template_idx_tmp.cpu(), sub_idx_tmp.cpu()))

        # Embedding → ID
        x_t = x_t.reshape(num_samples, 1 + self.max_r_count, self.emb_dim)
        template_emb = x_t[:, 0, :]
        sub_emb = x_t[:, 1:, :]

        template_dist = torch.cdist(template_emb, self.template_embedding.weight)
        template_idx = template_dist.argmin(dim=-1)

        sub_emb_flat = sub_emb.reshape(-1, self.emb_dim)
        sub_dist = torch.cdist(sub_emb_flat, self.substituent_embedding.weight)
        sub_idx = sub_dist.argmin(dim=-1)
        sub_idx = sub_idx.reshape(num_samples, self.max_r_count)

        if return_trajectory:
            return template_idx, sub_idx, trajectory
        else:
            return template_idx, sub_idx


if __name__ == '__main__':
    # 测试
    print("测试片段扩散模型")

    device = torch.device('cpu')
    num_templates = 100
    num_substituents = 300
    max_r_count = 2
    batch_size = 4

    model = FragmentDiffusion(
        num_templates=num_templates,
        num_substituents=num_substituents,
        max_r_count=max_r_count,
        emb_dim=64,
        hidden_dim=256,
        num_timesteps=100
    )
    model.to(device)

    # 测试训练
    template_idx = torch.randint(0, num_templates, (batch_size,), device=device)
    sub_idx = torch.randint(0, num_substituents, (batch_size, max_r_count), device=device)
    loss = model(template_idx, sub_idx)
    print(f"训练损失: {loss.item():.4f}")

    # 测试采样
    template_gen, sub_gen = model.sample(2, device)
    print(f"生成样本形状: template={template_gen.shape}, sub={sub_gen.shape}")

    print("\n测试条件扩散模型")
    cond_model = FragmentCDiffusion(
        num_templates=num_templates,
        num_substituents=num_substituents,
        max_r_count=max_r_count,
        emb_dim=64,
        hidden_dim=256,
        num_timesteps=100
    )
    cond_model.to(device)

    # 测试条件训练
    logp = torch.randn(batch_size, device=device)
    qed = torch.rand(batch_size, device=device)
    loss = cond_model(template_idx, sub_idx, logp, qed)
    print(f"条件训练损失: {loss.item():.4f}")

    # 测试条件采样
    template_gen, sub_gen = cond_model.sample(2, logp=2.0, qed=0.8, device=device)
    print(f"条件生成样本形状: template={template_gen.shape}, sub={sub_gen.shape}")
