"""
数据处理模块 - 基于模板填空的分子生成
从真实数据集自动提取分子片段库
"""

import torch
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd
from collections import Counter

# 这些变量会在load时自动填充
TEMPLATES = []
SUBSTITUENTS = []
TEMPLATE_R_COUNTS = []
MAX_R_COUNT = 0
NUM_TEMPLATES = 0
NUM_SUBSTITUENTS = 0


def decompose_molecule_with_brics(smiles):
    """
    使用BRICS算法分解分子为片段

    Args:
        smiles: 分子的SMILES字符串

    Returns:
        fragments: 片段列表，如果分解失败返回None
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import BRICS
        import re

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # BRICS分解
        fragments = list(BRICS.BRICSDecompose(mol))
        # 清理dummy原子标记
        cleaned_frags = []
        for frag in fragments:
            # 移除BRICS的所有dummy原子标记 [*]、[1*]、[2*] ... [99*]
            cleaned = re.sub(r'\[\d*\*\]', '', frag)
            # 验证清理后的SMILES是否有效
            if cleaned:
                test_mol = Chem.MolFromSmiles(cleaned)
                if test_mol and cleaned not in cleaned_frags:
                    cleaned_frags.append(cleaned)

        return cleaned_frags if cleaned_frags else None
    except Exception as e:
        return None


def build_fragment_library_from_dataset(dataset_file='dataset.txt', max_molecules=None):
    """
    从真实数据集中提取片段库并自动分类为模板和取代基

    Args:
        dataset_file: 数据集TXT文件路径（每行一个SMILES）
        max_molecules: 最多处理的分子数量（None表示全部）

    Returns:
        templates: 模板列表（大片段，添加{R1}占位符）
        substituents: 取代基列表（小片段）
        molecule_fragments: 每个分子对应的片段列表（用于后续重建训练样本）
    """
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
    except ImportError:
        print("错误: RDKit未安装，无法从真实数据集提取片段")
        print("请运行: pip install rdkit")
        return None, None, None

    print("\n" + "=" * 60)
    print(f"从真实数据集提取片段库: {dataset_file}")

    # 1. 读取数据集
    try:
        with open(dataset_file, 'r') as f:
            lines = f.readlines()

        # 每行一个SMILES，去除空行和空格
        all_smiles = [line.strip() for line in lines if line.strip()]

        print(f"  - 数据集总分子数: {len(all_smiles)}")

        # 随机采样指定数量的分子
        if max_molecules and len(all_smiles) > max_molecules:
            import random
            random.seed(42)  # 固定随机种子，保证可复现
            smiles_list = random.sample(all_smiles, max_molecules)
            print(f"  - 随机采样: {max_molecules} 个分子用于训练")
        else:
            smiles_list = all_smiles
            print(f"  - 使用全部分子: {len(smiles_list)} 个")
    except Exception as e:
        print(f"错误: 无法读取数据集 - {e}")
        return None, None, None

    # 2. 收集所有片段
    all_fragments = set()
    molecule_fragments = []  # 记录每个分子的片段
    successful_molecules = 0

    print("  - 分解分子中...")
    for i, smi in enumerate(smiles_list):
        if (i + 1) % 1000 == 0:
            print(f"    已处理 {i+1}/{len(smiles_list)} 个分子...")

        frags = decompose_molecule_with_brics(smi)
        if frags and len(frags) > 0:
            molecule_fragments.append({
                'original_smiles': smi,
                'fragments': frags
            })
            all_fragments.update(frags)
            successful_molecules += 1

    print(f"  - 成功分解: {successful_molecules}/{len(smiles_list)} 个分子")
    print(f"  - 提取到 {len(all_fragments)} 个唯一片段")

    # 3. 根据分子大小分类片段
    templates = []
    substituents = [""]  # 空取代基（H）

    for frag in all_fragments:
        try:
            mol = Chem.MolFromSmiles(frag)
            if mol is None:
                continue

            num_heavy = mol.GetNumHeavyAtoms()

            # 分类策略：
            # 小片段（<5个重原子）→ 取代基
            # 大片段（≥5个重原子）→ 模板（添加{R1}占位符）
            if num_heavy < 5:
                if frag not in substituents:
                    substituents.append(frag)
            else:
                # 模板（添加一个连接点）
                template_smi = frag + "{R1}"
                if template_smi not in templates:
                    templates.append(template_smi)
        except:
            continue

    print(f"  - 模板数量: {len(templates)} (重原子≥5)")
    print(f"  - 取代基数量: {len(substituents)} (重原子<5)")

    # 4. 打印一些样本
    print(f"\n  - 模板样本 (前5个):")
    for i, t in enumerate(templates[:5]):
        print(f"    {i+1}. {t}")

    print(f"\n  - 取代基样本 (前10个):")
    for i, s in enumerate(substituents[:10]):
        if s == "":
            print(f"    {i+1}. (H)")
        else:
            print(f"    {i+1}. {s}")

    print("=" * 60)

    return templates, substituents, molecule_fragments


def _get_default_fragments():
    """
    获取默认片段库（作为后备）
    """
    templates = [
        # 苯环类
        "c1ccc({R1})cc1",
        "c1cccc({R1})c1",
        "c1cc({R1})cc({R2})c1",

        # 杂环类
        "c1cc{R1}ncc1",
        "c1nc{R1}ncc1",

        # 链烷烃类
        "C{R1}C{R2}C",
        "CC{R1}C{R2}",

        # 环烷烃类
        "C1CC{R1}CC1",

        # 联苯
        "c1ccc(-c2ccc({R1})cc2)cc1",
    ]

    substituents = [
        "", "C", "CC", "CCC", "O", "N", "F", "Cl", "Br",
        "C(=O)", "C(=O)O", "C#N", "S", "C=C"
    ]

    return templates, substituents


def count_substitution_sites(template):
    """计算模板中有多少个{R}占位符"""
    import re
    return len(re.findall(r'\{R\d+\}', template))


# ============================================
# 数据集类
# ============================================

class TemplateDataset(Dataset):
    """
    模板数据集 - 基于真实分子片段化
    每个样本是从真实分子分解得到的(模板索引, 取代基索引列表)组合
    """

    def __init__(self, dataset_file='dataset.txt', max_molecules=None):
        """
        初始化数据集

        Args:
            dataset_file: 真实分子数据集文件路径（每行一个SMILES）
            max_molecules: 最多使用的分子数量（None表示全部）
        """
        global TEMPLATES, SUBSTITUENTS, TEMPLATE_R_COUNTS, MAX_R_COUNT
        global NUM_TEMPLATES, NUM_SUBSTITUENTS

        # 从真实数据集提取片段库
        print("\n" + "=" * 60)
        print("正在从真实数据集构建训练数据...")

        result = build_fragment_library_from_dataset(dataset_file, max_molecules)
        if result[0] is None:
            raise RuntimeError("无法从数据集提取片段库，请检查dataset.txt文件")

        TEMPLATES, SUBSTITUENTS, molecule_fragments = result

        TEMPLATE_R_COUNTS = [count_substitution_sites(t) for t in TEMPLATES]
        MAX_R_COUNT = max(TEMPLATE_R_COUNTS) if TEMPLATE_R_COUNTS else 1
        NUM_TEMPLATES = len(TEMPLATES)
        NUM_SUBSTITUENTS = len(SUBSTITUENTS)

        # 构建训练样本：从真实分子的片段重建
        self.samples, self.original_smiles_list = self._build_samples_from_real_molecules(molecule_fragments)

        self.max_r_count = MAX_R_COUNT
        self.num_templates = NUM_TEMPLATES
        self.num_substituents = NUM_SUBSTITUENTS

        print(f"\n训练数据集信息：")
        print(f"  - 真实分子样本数: {len(self.samples)}")
        print(f"  - 模板数量: {NUM_TEMPLATES}")
        print(f"  - 取代基数量: {NUM_SUBSTITUENTS}")
        print(f"  - 最大取代位点: {MAX_R_COUNT}")
        print(f"  ✓ 数据来源: {dataset_file} 的真实分子")
        print("=" * 60)

    def _build_samples_from_real_molecules(self, molecule_fragments):
        """
        直接使用真实分子作为训练样本
        保存原始SMILES和片段表示的对应关系
        """
        samples = []
        original_smiles_list = []

        print("  - 构建训练样本（使用dataset.txt的真实分子）...")

        # 创建片段到索引的映射
        template_to_idx = {}
        for i, t in enumerate(TEMPLATES):
            base = t.replace('{R1}', '')
            template_to_idx[base] = i

        substituent_to_idx = {s: i for i, s in enumerate(SUBSTITUENTS)}

        # 直接使用所有真实分子
        for mol_data in molecule_fragments:
            original_smiles = mol_data['original_smiles']
            frags = mol_data['fragments']

            if not frags:
                continue

            # 按大小排序，最大的作为模板
            try:
                from rdkit import Chem
                frag_sizes = []
                for frag in frags:
                    mol = Chem.MolFromSmiles(frag)
                    if mol:
                        frag_sizes.append((frag, mol.GetNumHeavyAtoms()))
                    else:
                        frag_sizes.append((frag, 0))

                frag_sizes.sort(key=lambda x: x[1], reverse=True)

                # 找到第一个可以作为模板的片段
                template_idx = None
                for frag, size in frag_sizes:
                    if size >= 5 and frag in template_to_idx:
                        template_idx = template_to_idx[frag]
                        remaining_frags = [f for f, s in frag_sizes if f != frag]
                        break

                if template_idx is None:
                    continue

                # 获取该模板需要的取代基数量
                r_count = TEMPLATE_R_COUNTS[template_idx]

                # 映射取代基
                substituent_indices = []
                for frag in remaining_frags[:r_count]:
                    if frag in substituent_to_idx:
                        substituent_indices.append(substituent_to_idx[frag])
                    else:
                        substituent_indices.append(0)

                # 填充到MAX_R_COUNT
                while len(substituent_indices) < MAX_R_COUNT:
                    substituent_indices.append(0)

                samples.append((template_idx, substituent_indices))
                original_smiles_list.append(original_smiles)

            except Exception as e:
                continue

        print(f"  - 成功构建: {len(samples)} 个真实分子样本")
        print(f"  ✓ 所有训练数据来自dataset.txt，无数据增强")

        return samples, original_smiles_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取一个样本

        Returns:
            template_idx: 模板索引 (标量)
            substituent_indices: 取代基索引列表 [max_r_count]
        """
        template_idx, substituent_indices = self.samples[idx]

        return (
            torch.tensor(template_idx, dtype=torch.long),
            torch.tensor(substituent_indices, dtype=torch.long)
        )

    def decode(self, template_idx, substituent_indices):
        """
        将(模板, 取代基)解码为SMILES字符串

        Args:
            template_idx: 模板索引（可以是tensor或int）
            substituent_indices: 取代基索引列表（可以是tensor或list）

        Returns:
            SMILES字符串
        """
        # 转换为Python原生类型
        if isinstance(template_idx, torch.Tensor):
            template_idx = template_idx.item()
        if isinstance(substituent_indices, torch.Tensor):
            substituent_indices = substituent_indices.cpu().numpy().tolist()

        # 获取模板
        template = TEMPLATES[template_idx]
        smiles = template

        # 替换每个{R}占位符
        r_count = TEMPLATE_R_COUNTS[template_idx]
        for i in range(r_count):
            placeholder = f"{{R{i+1}}}"
            sub_idx = substituent_indices[i]
            substituent = SUBSTITUENTS[sub_idx]
            smiles = smiles.replace(placeholder, substituent)

        return smiles


def get_dataloader(batch_size=32, shuffle=True, dataset_file='dataset.txt', max_molecules=None):
    """
    创建数据加载器的便捷函数

    Args:
        batch_size: 批次大小
        shuffle: 是否打乱数据
        dataset_file: 真实分子数据集文件路径
        max_molecules: 最多使用的分子数量（None表示全部）

    Returns:
        dataloader: DataLoader对象
        dataset: Dataset对象
    """
    dataset = TemplateDataset(dataset_file=dataset_file, max_molecules=max_molecules)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )

    # 可视化片段库（只在首次加载时）
    _visualize_fragment_library()

    return dataloader, dataset


def _visualize_fragment_library():
    """
    可视化片段库：模板和取代基
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
    except ImportError:
        return

    import os

    # 检查是否已经生成过
    if os.path.exists('fragment_library.png'):
        return

    print("\n" + "=" * 60)
    print("可视化片段库...")

    # 1. 可视化模板
    template_mols = []
    template_legends = []

    for i, template in enumerate(TEMPLATES[:18]):  # 最多显示18个模板
        # 用空取代基填充所有位置
        smiles = template
        r_count = TEMPLATE_R_COUNTS[i]
        for j in range(r_count):
            placeholder = f"{{R{j+1}}}"
            smiles = smiles.replace(placeholder, "")

        mol = Chem.MolFromSmiles(smiles)
        if mol:
            template_mols.append(mol)
            template_legends.append(f"Template {i}")

    # 2. 可视化取代基
    substituent_mols = []
    substituent_legends = []

    for i, sub in enumerate(SUBSTITUENTS[:15]):  # 最多显示15个取代基
        # 附加到苯环上以便可视化
        if sub == "":
            smiles = "c1ccccc1"
        else:
            smiles = f"c1ccc({sub})cc1"

        mol = Chem.MolFromSmiles(smiles)
        if mol:
            substituent_mols.append(mol)
            if sub == "":
                substituent_legends.append(f"Substituent {i}: (H)")
            else:
                substituent_legends.append(f"Substituent {i}: {sub}")

    # 3. 绘制组合图
    all_mols = template_mols + substituent_mols
    all_legends = template_legends + substituent_legends

    if all_mols:
        img = Draw.MolsToGridImage(
            all_mols,
            molsPerRow=6,
            subImgSize=(300, 300),
            legends=all_legends,
            returnPNG=False
        )
        img.save('fragment_library.png')
        print(f"  - Fragment library saved: fragment_library.png")
        print(f"  - Templates: {len(template_mols)}, Substituents: {len(substituent_mols)}")

    print("=" * 60)


if __name__ == '__main__':
    # 测试代码
    print("测试数据加载...\n")
    dataloader, dataset = get_dataloader(
        batch_size=4,
        dataset_file='dataset.txt',
        max_molecules=100  # 测试时只用100个分子
    )

    # 获取一个batch
    template_batch, substituent_batch = next(iter(dataloader))
    print(f"\nBatch shapes:")
    print(f"  - 模板索引: {template_batch.shape}")  # [batch_size]
    print(f"  - 取代基索引: {substituent_batch.shape}")  # [batch_size, max_r_count]

    # 解码第一个样本
    print(f"\n前4个样本解码:")
    for i in range(4):
        smiles = dataset.decode(template_batch[i], substituent_batch[i])
        print(f"  {i+1}. 模板{template_batch[i].item()}, 取代基{substituent_batch[i].tolist()[:3]} -> {smiles}")

    # 验证SMILES有效性
    print("\n验证SMILES有效性:")
    try:
        from rdkit import Chem
        valid_count = 0
        for i in range(4):
            smiles = dataset.decode(template_batch[i], substituent_batch[i])
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                valid_count += 1
                print(f"  ✓ 样本{i+1}有效")
            else:
                print(f"  ✗ 样本{i+1}无效: {smiles}")
        print(f"\n有效率: {valid_count}/4 ({valid_count/4*100:.1f}%)")
    except ImportError:
        print("  (需要RDKit验证)")
