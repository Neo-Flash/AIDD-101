"""
data.py - 基于BRICS片段的分子数据处理
从真实数据集自动提取分子片段库，用于VAE/CVAE训练
"""

import torch
from torch.utils.data import Dataset, DataLoader
import random
import pickle
import os
from collections import Counter

# 全局片段库（在load时自动填充）
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
    except Exception:
        return None


def build_fragment_library_from_dataset(dataset_file, max_molecules=None):
    """
    从真实数据集中提取片段库并自动分类为模板和取代基

    Args:
        dataset_file: 数据集文件路径（CSV或TXT）
        max_molecules: 最多处理的分子数量

    Returns:
        templates: 模板列表
        substituents: 取代基列表
        molecule_data: 每个分子的数据（片段、logP、QED）
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Crippen import MolLogP
        from rdkit.Chem.QED import qed
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
    except ImportError:
        print("错误: RDKit未安装")
        return None, None, None

    print("\n" + "=" * 60)
    print(f"从数据集提取片段库: {dataset_file}")

    # 1. 读取数据集
    try:
        if dataset_file.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(dataset_file)
            smiles_col = 'SMILES' if 'SMILES' in df.columns else 'smiles'
            all_smiles = df[smiles_col].tolist()
        else:
            with open(dataset_file, 'r') as f:
                lines = f.readlines()
            all_smiles = [line.strip() for line in lines if line.strip()]

        print(f"  - 数据集总分子数: {len(all_smiles)}")

        if max_molecules and len(all_smiles) > max_molecules:
            random.seed(42)
            smiles_list = random.sample(all_smiles, max_molecules)
            print(f"  - 随机采样: {max_molecules} 个分子")
        else:
            smiles_list = all_smiles

    except Exception as e:
        print(f"错误: 无法读取数据集 - {e}")
        return None, None, None

    # 2. 收集所有片段并记录分子属性
    all_fragments = set()
    molecule_data = []
    successful_molecules = 0

    print("  - 分解分子中...")
    from tqdm import tqdm

    for smi in tqdm(smiles_list, desc="    BRICS分解", unit="分子"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        frags = decompose_molecule_with_brics(smi)
        if frags and len(frags) > 0:
            try:
                logp = MolLogP(mol)
                qed_score = qed(mol)

                molecule_data.append({
                    'original_smiles': smi,
                    'fragments': frags,
                    'logp': logp,
                    'qed': qed_score
                })
                all_fragments.update(frags)
                successful_molecules += 1
            except Exception:
                continue

    print(f"  - 成功分解: {successful_molecules}/{len(smiles_list)} 个分子")
    print(f"  - 提取到 {len(all_fragments)} 个唯一片段")

    # 3. 根据分子大小分类片段
    templates = []
    substituents = [""]  # 空取代基（H）

    print("  - 分类片段中...")
    from tqdm import tqdm

    for frag in tqdm(all_fragments, desc="    片段分类", unit="片段"):
        try:
            mol = Chem.MolFromSmiles(frag)
            if mol is None:
                continue

            num_heavy = mol.GetNumHeavyAtoms()

            # 分类策略：
            # 小片段（<5个重原子）→ 取代基
            # 大片段（≥5个重原子）→ 模板
            if num_heavy < 15:
                if frag not in substituents:
                    substituents.append(frag)
            else:
                template_smi = frag + "{R1}"
                if template_smi not in templates:
                    templates.append(template_smi)
        except Exception:
            continue

    print(f"  - 模板数量: {len(templates)} (重原子≥15)")
    print(f"  - 取代基数量: {len(substituents)} (重原子<15)")
    print("=" * 60)

    return templates, substituents, molecule_data


def count_substitution_sites(template):
    """计算模板中有多少个{R}占位符"""
    import re
    return len(re.findall(r'\{R\d+\}', template))


class FragmentDataset(Dataset):
    """
    片段数据集 - 基于真实分子BRICS分解
    每个样本: (模板索引, 取代基索引列表, logP, QED)
    """

    def __init__(self, dataset_file='zinc_250k.csv', max_molecules=None, cache_dir='checkpoints'):
        global TEMPLATES, SUBSTITUENTS, TEMPLATE_R_COUNTS, MAX_R_COUNT
        global NUM_TEMPLATES, NUM_SUBSTITUENTS

        # 生成缓存文件名
        dataset_name = os.path.splitext(os.path.basename(dataset_file))[0]
        cache_file = f'{cache_dir}/fragment_data_{dataset_name}_n{max_molecules}.pkl'

        # 尝试加载缓存
        if os.path.exists(cache_file):
            print(f'加载缓存: {cache_file}')
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)

            TEMPLATES = cached['templates']
            SUBSTITUENTS = cached['substituents']
            TEMPLATE_R_COUNTS = cached['template_r_counts']
            MAX_R_COUNT = cached['max_r_count']
            NUM_TEMPLATES = len(TEMPLATES)
            NUM_SUBSTITUENTS = len(SUBSTITUENTS)

            self.samples = cached['samples']
            self.max_r_count = MAX_R_COUNT
            self.num_templates = NUM_TEMPLATES
            self.num_substituents = NUM_SUBSTITUENTS

            print(f"  - 样本数: {len(self.samples)}")
            print(f"  - 模板: {NUM_TEMPLATES}, 取代基: {NUM_SUBSTITUENTS}")
            return

        # 无缓存，重新处理
        print("\n" + "=" * 60)
        print("正在从数据集构建训练数据...")

        result = build_fragment_library_from_dataset(dataset_file, max_molecules)
        if result[0] is None:
            raise RuntimeError("无法从数据集提取片段库")

        TEMPLATES, SUBSTITUENTS, molecule_data = result

        TEMPLATE_R_COUNTS = [count_substitution_sites(t) for t in TEMPLATES]
        MAX_R_COUNT = max(TEMPLATE_R_COUNTS) if TEMPLATE_R_COUNTS else 1
        NUM_TEMPLATES = len(TEMPLATES)
        NUM_SUBSTITUENTS = len(SUBSTITUENTS)

        # 构建训练样本
        self.samples = self._build_samples(molecule_data)

        self.max_r_count = MAX_R_COUNT
        self.num_templates = NUM_TEMPLATES
        self.num_substituents = NUM_SUBSTITUENTS

        print(f"\n训练数据集信息：")
        print(f"  - 样本数: {len(self.samples)}")
        print(f"  - 模板数量: {NUM_TEMPLATES}")
        print(f"  - 取代基数量: {NUM_SUBSTITUENTS}")
        print(f"  - 最大取代位点: {MAX_R_COUNT}")

        # 保存缓存
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'templates': TEMPLATES,
                'substituents': SUBSTITUENTS,
                'template_r_counts': TEMPLATE_R_COUNTS,
                'max_r_count': MAX_R_COUNT,
                'samples': self.samples
            }, f)
        print(f"缓存已保存: {cache_file}")

    def _build_samples(self, molecule_data):
        """从真实分子构建训练样本"""
        from rdkit import Chem
        from tqdm import tqdm

        samples = []

        # 创建片段到索引的映射
        template_to_idx = {}
        for i, t in enumerate(TEMPLATES):
            base = t.replace('{R1}', '')
            template_to_idx[base] = i

        substituent_to_idx = {s: i for i, s in enumerate(SUBSTITUENTS)}

        print("  - 构建训练样本...")

        for mol_data in tqdm(molecule_data, desc="    样本构建", unit="样本"):
            frags = mol_data['fragments']
            logp = mol_data['logp']
            qed_score = mol_data['qed']

            if not frags:
                continue

            # 按大小排序，最大的作为模板
            try:
                frag_sizes = []
                for frag in frags:
                    mol = Chem.MolFromSmiles(frag)
                    if mol:
                        frag_sizes.append((frag, mol.GetNumHeavyAtoms()))
                    else:
                        frag_sizes.append((frag, 0))

                frag_sizes.sort(key=lambda x: x[1], reverse=True)

                # 找到可以作为模板的片段
                template_idx = None
                for frag, size in frag_sizes:
                    if size >= 15 and frag in template_to_idx:
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

                samples.append({
                    'template_idx': template_idx,
                    'substituent_indices': substituent_indices,
                    'logp': logp,
                    'qed': qed_score
                })

            except Exception:
                continue

        print(f"  - 成功构建: {len(samples)} 个样本")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.tensor(sample['template_idx'], dtype=torch.long),
            torch.tensor(sample['substituent_indices'], dtype=torch.long),
            torch.tensor(sample['logp'], dtype=torch.float),
            torch.tensor(sample['qed'], dtype=torch.float)
        )

    def decode(self, template_idx, substituent_indices):
        """将(模板, 取代基)解码为SMILES"""
        if isinstance(template_idx, torch.Tensor):
            template_idx = template_idx.item()
        if isinstance(substituent_indices, torch.Tensor):
            substituent_indices = substituent_indices.cpu().numpy().tolist()

        template = TEMPLATES[template_idx]
        smiles = template
        r_count = TEMPLATE_R_COUNTS[template_idx]

        for i in range(r_count):
            placeholder = f"{{R{i+1}}}"
            sub_idx = substituent_indices[i]
            substituent = SUBSTITUENTS[sub_idx]
            smiles = smiles.replace(placeholder, substituent)

        return smiles


def get_dataloader(batch_size=64, shuffle=True, dataset_file='zinc_250k.csv',
                   max_molecules=None, cache_dir='checkpoints', test_ratio=0.1):
    """创建数据加载器"""
    dataset = FragmentDataset(
        dataset_file=dataset_file,
        max_molecules=max_molecules,
        cache_dir=cache_dir
    )

    # 划分训练/测试
    n_total = len(dataset)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_test

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"训练集: {n_train}, 测试集: {n_test}")

    return train_loader, test_loader, dataset


def get_config():
    """获取片段库配置"""
    return {
        'num_templates': NUM_TEMPLATES,
        'num_substituents': NUM_SUBSTITUENTS,
        'max_r_count': MAX_R_COUNT,
        'templates': TEMPLATES,
        'substituents': SUBSTITUENTS,
        'template_r_counts': TEMPLATE_R_COUNTS
    }


if __name__ == '__main__':
    print("测试数据加载...\n")
    train_loader, test_loader, dataset = get_dataloader(
        batch_size=4,
        dataset_file='zinc_250k.csv',
        max_molecules=1000
    )

    # 获取一个batch
    template_batch, sub_batch, logp_batch, qed_batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  - 模板索引: {template_batch.shape}")
    print(f"  - 取代基索引: {sub_batch.shape}")
    print(f"  - logP: {logp_batch.shape}")
    print(f"  - QED: {qed_batch.shape}")

    # 解码
    print(f"\n前4个样本:")
    for i in range(4):
        smiles = dataset.decode(template_batch[i], sub_batch[i])
        print(f"  {i+1}. logP={logp_batch[i]:.2f}, QED={qed_batch[i]:.3f} | {smiles}")
