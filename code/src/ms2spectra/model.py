import torch as th
import torch.nn as nn
import numpy as np
from pprint import pprint
from pyteomics.mass import Composition

from ms2spectra.utils.frag_utils import (
	get_node_feats,
	get_edge_feats,
	th_long_to_mask,
	CUT_CHEM_EDGE_FEAT_SIZE,
)
from ms2spectra.utils.misc_utils import scatter_logsumexp, scatter_logsoftmax, scatter_reduce, scatter_masked_softmax, scatter_masked_logsumexp
from ms2spectra.utils.feat_utils import get_mol_feats_sizes, get_mol_fp_size
from ms2spectra.utils.nn_utils import *
from ms2spectra.components.formula_features import build_formula_embedder
from ms2spectra.utils.misc_utils import check_pyg_compile, LOG_ZERO, check_pyg_full_compile
from ms2spectra.utils.spec_utils import transform_ce, batched_bin_func
from ms2spectra.utils.data_utils import combine_formulae
from ms2spectra.utils.formula_utils import PREC_TYPE_TO_FORMULA_DIFF
from ms2spectra.components.collision_energy_modulation import CEFragmentFiLM
from ms2spectra.components.collision_energy_transition import CELocalTransitionPrior, CEHChannelTransitionPrior

"""
这是一个基于 **PyTorch** and **PyTorch Geometric (PyG)** 的深度学习模型代码文件，主要用于 **质谱（Mass Spectrometry）预测**，特别是针对 **小分子的碎片谱图（Fragmentation Spectra / MS/MS）** 的预测。

这个文件定义了一个名为 `FragGNNModel` 的核心模型，以及几个辅助模型（`NeimsModel`, `GNNModel`）和特征处理模块（Mixin classes）。

核心思想是：通过 **图神经网络 (GNN)** 学习分子的结构特征，并结合 **碎片化树（Fragmentation Tree / DAG）** 的逻辑来预测分子在质谱仪中被打碎后产生的各种碎片离子的概率（即谱图中的峰强度）。

下面我将分模块非常详细地解释这段代码：

---

### 1. 辅助特征处理类 (Mixin Classes)

这三个类主要用于处理实验条件（元数据），并将它们嵌入（Embed）到神经网络中。

#### `CEModel` (Collision Energy Model)
*   **功能**: 处理 **碰撞能量 (Collision Energy, CE)**。碰撞能量决定了分子被打碎的程度。
*   **初始化 (`_ce_init`)**:
    *   支持多种插入位置 (`ce_insert_location`): 分子层 (`mol`)、MLP层 (`mlp`)。
    *   支持多种编码方式 (`ce_insert_type`):
        *   `id`: 恒等映射（标准化后直接使用数值）。
        *   `lin`: 线性层投影。
        *   `embed`: 将连续的能量值离散化（取整）后使用 `nn.Embedding`。
        *   `bin`: 分桶（Binning）后使用 One-hot 编码再线性投影。
*   **前向传播 (`embed_ce`)**: 计算 CE 的 Embedding 向量。

#### `PrecModel` (Precursor Model)
*   **功能**: 处理 **前体离子类型 (Precursor Type)**（例如 `[M+H]+`, `[M-H]-` 等）。
*   **实现**: 使用 `nn.Embedding` 将离散的类型 ID 映射为向量。

#### `InstModel` (Instrument Model)
*   **功能**: 处理 **仪器类型 (Instrument Type)**（例如 Orbitrap, Q-TOF 等）。不同仪器产生的谱图特征不同。
*   **实现**: 同样使用 `nn.Embedding`。

---

### 2. 核心模型: `FragGNNModel`

这是文件中最复杂、最重要的类。它是一个 **双层图神经网络架构**。

#### **架构概览**
1.  **Molecule GNN**: 编码原始分子的原子和键的特征。
2.  **Interstage (中间层)**: 将原始分子的特征映射到“碎片图”的节点上。
3.  **Fragment GNN**: 在“碎片图”上进行消息传递，学习碎片之间的父子关系。
4.  **Prediction Head**: 预测每个碎片生成的概率。

#### **详细流程 (`forward` 函数解析)**

1.  **输入处理**:
    *   `mol_pyg`: 原始分子的图数据（节点=原子，边=化学键）。
    *   `frag_pyg`: **碎片图 (Fragmentation Graph)**。这通常是一个有向无环图 (DAG)，节点代表可能的碎片（子结构），边代表碎裂路径。
    *   `spec_ce`, `spec_prec_type` 等: 实验条件。

2.  **条件嵌入**:
    *   调用 `embed_ce`, `embed_prec`, `embed_inst` 获取实验条件的向量，并将它们拼接到分子节点特征 (`mol_x`) 或 MLP 输入中。

3.  **Molecule GNN (分子编码)**:
    *   `self.mol_embedder(mol_x, ...)`: 运行 GNN 提取原子级别的特征。
    *   `self.mol_pool`: 对原子特征进行池化（如 sum/mean），得到整个分子的全局特征向量。

4.  **构建碎片图特征 (Interstage)**:
    *   这是代码中最巧妙的部分。碎片图中的节点（碎片）是由原分子的原子组成的。
    *   **Masking & Scattering**: 代码使用 `scatter_reduce` 操作，根据碎片包含哪些原子（`cc` - connected components），将 **Molecule GNN** 计算出的原子特征聚合起来，作为 **Fragment GNN** 节点的初始特征。
    *   `self.cc_interstage`: 定义了聚合方式（相加 `add`、相减 `sub` 或线性变换 `linear`）。
    *   此外，还处理了化学式 (`base_formula`)、碎裂深度 (`depth`) 等特征。

5.  **Fragment GNN (碎片传播)**:
    *   `self.frag_embedder(...)`: 在碎片图上运行 GNN。这模拟了能量在碎片路径上的传递过程。
    *   `frag_embed`: 结合了 GNN 输出、节点自身投影以及全局条件（CE, Precursor 等）的最终特征向量。

6.  **概率预测 (MLP Heads)**:
    *   **氢重排处理**: 质谱中常见的现象是氢原子的得失。代码通过 `2*self.num_hs+1` 预测不同氢状态的概率。
    *   **`formula_module`**: 预测给定碎片节点生成特定化学式（Formula）的 Logits。
    *   **`node_module`** (可选): 预测该节点本身存在的概率。
    *   **联合概率 $P(f, n)$**: 计算“既是该碎片节点 $n$ 又是该化学式 $f$”的联合概率。

7.  **Out of Scope (OOS) 预测**:
    *   `self.oos_module`: 预测谱图中是否存在模型无法解释的峰（即不在输入的碎片图中的碎片）。

8.  **谱图生成 (Aggregation)**:
    *   利用 `scatter_logsumexp` 将所有预测出的相同化学式/质量的碎片概率聚合。
    *   **`spec_logprobs`**: 最终生成的预测谱图（m/z vs Intensity）。
    *   **`bin_output`**: 如果开启，会对 m/z 轴进行分桶（Binning），将连续的 m/z 离散化为直方图形式。

9.  **输出**:
    *   返回一个字典 `out_d`，包含预测的 m/z (`pred_mzs`)、强度对数 (`pred_logprobs`) 以及中间过程的各种概率分布（用于解释性分析或更细粒度的损失函数计算）。

---

### 3. 对比模型 (Baseline / Alternatives)

#### `NeimsModel` (Neural Electron-Ionization MS)
*   **原理**: 这是一个经典的基于 **指纹 (Fingerprint)** 的模型，类似于 NEIMS 论文中的方法。
*   **输入**: 不使用 GNN，而是使用分子的指纹（Morgan, MACCS, RDKit）。
*   **结构**: 指纹向量 + 实验条件 -> 多层感知机 (MLP / `SpecFFN`) -> 预测谱图。
*   **用途**: 通常作为基准模型 (Baseline) 来评估 FragGNN 的性能。

#### `PrecursorModel`
*   **原理**: 一个“哑”模型 (Dummy Model)。
*   **功能**: 只预测前体离子的 m/z，强度设为 0 或常数。用于调试或作为最基本的对比。

#### `GNNModel`
*   **原理**: 直接从分子图预测谱图，**跳过了显式的碎片图 (Frag Graph) 构建过程**。
*   **流程**: Molecule GNN -> Global Pooling -> MLP (`SpecFFN`) -> 预测谱图。
*   **区别**: `FragGNNModel` 试图模拟碎裂过程（可解释性强，能知道哪个碎片来自哪），而 `GNNModel` 是端到端的黑盒预测。

---

### 4. 关键技术点与工具函数

*   **`scatter_*` 函数 (来自 `ms2spectra.utils.misc_utils`)**:
    *   代码大量使用了 `scatter_reduce`, `scatter_logsoftmax`, `scatter_logsumexp`。这是图神经网络处理变长数据（例如不同分子有不同数量的原子，不同碎片有不同数量的父节点）的核心操作。它们用于在索引指导下进行聚合（求和、求最大值、LogSumExp等）。
*   **PyTorch Geometric (PyG)**:
    *   使用 `pyg.data.Data` 对象存储图数据。
    *   `batch`: 处理图数据的一个关键属性，用于区分一个大 Batch 中哪些节点属于哪个图。
*   **动态编译 (`th.compile`)**:
    *   代码中包含 `get_compile` 和 `compile_submodules`，利用 PyTorch 2.0 的 `torch.compile` 来加速模型推理和训练。
*   **化学式与氢**:
    *   代码非常注重化学式的精确计算（`Composition`），特别是氢原子的得失（Hydrogen shift），这是质谱预测准确性的关键。

### 总结
这段代码是一个 **SOTA (State-of-the-Art) 级别** 的质谱预测模型实现。它不仅仅是一个简单的回归模型，而是结合了领域知识（碎片化路径、化学式守恒、碰撞能量影响）的结构化深度学习模型。`FragGNNModel` 显式地对碎裂过程建模，使其比传统的指纹方法 (`NeimsModel`) 或直接 GNN 方法 (`GNNModel`) 具有更好的潜在准确性和可解释性。
"""

class CEModel:
	""" class for handling collision engery embedding
	"""
	def _ce_init(
		self,
		int_embedder,
		ce_insert_location: str,
		ce_insert_type: str,
		ce_insert_merge: bool,
		ce_insert_size: int,
		ce_mean: float,
  		ce_std: float,
    	ce_max: float,):

		# ce stuff
		assert ce_insert_type in ["id","lin","embed","bin"]
		assert ce_insert_location in ["none","mol","frag","mlp"]
		self.ce_insert_type = ce_insert_type
		self.ce_insert_location = ce_insert_location
		self.ce_insert_merge = ce_insert_merge
		self.ce_insert_size = ce_insert_size
		self.int_embedder = int_embedder
		self.ce_max = ce_max
		self.ce_mean = ce_mean
		self.ce_std = ce_std
		self._ce_location_check()
		self._setup_ce()

	def _ce_location_check(self):

		raise NotImplementedError

	def _setup_ce(self):

		# embedding type
		if self.ce_insert_type == "id":
			def ce_transform(ce):
				ce = transform_ce(ce, self.ce_mean, self.ce_std)
				ce = ce.reshape(-1,1)
				ce = th.repeat_interleave(ce, self.ce_insert_size, dim=1)
				return ce
			ce_embedder = nn.Identity()
			ce_input_dim = self.ce_insert_size
		elif self.ce_insert_type == "lin":
			def ce_transform(ce):
				ce = transform_ce(ce, self.ce_mean, self.ce_std)
				ce = ce.reshape(-1,1)
				return ce
			ce_embedder = nn.Linear(1,self.ce_insert_size)
			ce_input_dim = self.ce_insert_size
		elif self.ce_insert_type == "embed":
			def ce_transform(ce):
				ce = th.clamp(ce, min=0, max=int(self.ce_max)-1)
				ce = th.round(ce, decimals=0).long()
				ce = ce.reshape(-1,1)
				return ce
			embedder = build_formula_embedder(self.int_embedder, max_count_int=int(self.ce_max))
			ce_embedder = nn.Sequential(
				embedder,
				nn.Linear(embedder.num_dim,self.ce_insert_size)
			)	
			ce_input_dim = self.ce_insert_size
		elif self.ce_insert_type == "bin":
			def ce_transform(ce):
				ce = th.clamp(ce, min=0, max=int(self.ce_max)-10)
				ce = th.round(ce, decimals=-1).long() // 10
				ce = F.one_hot(ce, num_classes=int(self.ce_max)//10).float()
				return ce
			ce_embedder = nn.Linear(int(self.ce_max)//10,self.ce_insert_size)
			ce_input_dim = self.ce_insert_size
		# location
		if self.ce_insert_location == "mol":
			ce_mol_input_dim = ce_input_dim
			ce_mlp_input_dim = 0
		elif self.ce_insert_location == "mlp":
			ce_mol_input_dim = 0
			ce_mlp_input_dim = ce_input_dim
		else:
			assert self.ce_insert_location == "none"
			ce_mol_input_dim = 0
			ce_mlp_input_dim = 0
		self.ce_transform = ce_transform
		self.ce_embedder = ce_embedder
		self.ce_mol_input_dim = ce_mol_input_dim
		self.ce_mlp_input_dim = ce_mlp_input_dim

	def embed_ce(self, ce, ce_batch_idxs, batch_size):

		if self.ce_insert_location != "none":
			ce_embed = self.ce_transform(ce)
			ce_embed = self.ce_embedder(ce_embed)
			# possibly merge the embeddings
			if self.ce_insert_merge:
				ce_embed = scatter_reduce(
					src=ce_embed,
					index=ce_batch_idxs.unsqueeze(1).expand_as(ce_embed),
					reduce="mean",
					dim_size=batch_size,
					include_self=False
				)
		else:
			ce_embed = None
		return ce_embed

class PrecModel:

	def _prec_init(
		self,
		prec_insert_location: str,
		prec_insert_size: int,
		prec_num_types: int):

		self.prec_insert_location = prec_insert_location
		self.prec_embedder = nn.Embedding(prec_num_types+1, prec_insert_size)	
		prec_dim = prec_insert_size

		self._prec_location_check()

		if self.prec_insert_location == "mol":
			prec_mol_input_dim = prec_dim
			prec_mlp_input_dim = 0
		elif self.prec_insert_location == "mlp":
			prec_mol_input_dim = 0
			prec_mlp_input_dim = prec_dim
		else:
			assert self.prec_insert_location == "none"
			prec_mol_input_dim = 0
			prec_mlp_input_dim = 0
		self.prec_mol_input_dim = prec_mol_input_dim
		self.prec_mlp_input_dim = prec_mlp_input_dim

	def _prec_location_check(self):

		raise NotImplementedError

	def embed_prec(self, prec_type):

		if self.prec_insert_location != "none":
			prec_embed = self.prec_embedder(prec_type)
		else:
			prec_embed = None
		return prec_embed

class InstModel:

	def _inst_init(
		self,
		inst_insert_location: str,
		inst_insert_size: int,
		inst_num_types: int):

		self.inst_insert_location = inst_insert_location
		self.inst_embedder = nn.Embedding(inst_num_types+1, inst_insert_size)	
		inst_dim = inst_insert_size

		self._inst_location_check()

		if self.inst_insert_location == "mol":
			inst_mol_input_dim = inst_dim
			inst_mlp_input_dim = 0
		elif self.inst_insert_location == "mlp":
			inst_mol_input_dim = 0
			inst_mlp_input_dim = inst_dim
		else:
			assert self.inst_insert_location == "none"
			inst_mol_input_dim = 0
			inst_mlp_input_dim = 0
		self.inst_mol_input_dim = inst_mol_input_dim
		self.inst_mlp_input_dim = inst_mlp_input_dim

	def _inst_location_check(self):

		raise NotImplementedError

	def embed_inst(self, inst_type):

		if self.inst_insert_location != "none":
			inst_embed = self.inst_embedder(inst_type)
		else:
			inst_embed = None
		return inst_embed

class SpectrumCandidateRefiner(nn.Module):
	"""
	R12: Spectrum-level candidate interaction refiner.

	It refines top-K node/formula/H candidate logits inside each spectrum.
	This is not ensemble/retrieval. It is an internal single-model residual
	before scatter_logsoftmax.
	"""
	def __init__(
		self,
		input_size: int,
		hidden_size: int = 128,
		num_layers: int = 2,
		num_heads: int = 4,
		dropout: float = 0.1,
		delta_scale: float = 0.15,
		topk: int = 384,
		center_per_spectrum: bool = True,
		use_logit_feature: bool = True,
	):
		super().__init__()

		assert hidden_size % num_heads == 0, (hidden_size, num_heads)
		self.topk = int(topk)
		self.delta_scale = float(delta_scale)
		self.center_per_spectrum = bool(center_per_spectrum)
		self.use_logit_feature = bool(use_logit_feature)

		self.input_proj = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.LayerNorm(hidden_size),
			nn.SiLU(),
			nn.Dropout(dropout),
		)

		enc_layer = nn.TransformerEncoderLayer(
			d_model=hidden_size,
			nhead=num_heads,
			dim_feedforward=4 * hidden_size,
			dropout=dropout,
			activation="gelu",
			batch_first=True,
			norm_first=True,
		)

		self.encoder = nn.TransformerEncoder(
			enc_layer,
			num_layers=num_layers,
		)

		self.out = nn.Sequential(
			nn.LayerNorm(hidden_size),
			nn.Linear(hidden_size, hidden_size),
			nn.SiLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_size, 1),
		)

		# zero-init: at step 0 this module is exactly no-op.
		last = self.out[-1]
		nn.init.zeros_(last.weight)
		nn.init.zeros_(last.bias)

	def forward(
		self,
		cand_feats: th.Tensor,
		base_logits_flat: th.Tensor,
		cand_batch_idxs: th.Tensor,
		valid_mask: th.Tensor,
		batch_size: int,
	):
		"""
		cand_feats:       [num_candidates, input_size]
		base_logits_flat: [num_candidates]
		cand_batch_idxs:  [num_candidates]
		valid_mask:       [num_candidates], non-null candidate mask
		"""
		device = cand_feats.device
		dtype = cand_feats.dtype

		delta_flat = th.zeros_like(base_logits_flat)

		for b in range(int(batch_size)):
			idx = th.nonzero(
				(cand_batch_idxs == b) & valid_mask,
				as_tuple=False,
			).reshape(-1)

			if idx.numel() == 0:
				continue

			k = min(int(self.topk), int(idx.numel()))

			# Select candidates by the current R3/G2 score.
			# This top-k is an internal routing decision, not ensemble/retrieval.
			score = base_logits_flat[idx].detach()
			top_rel = th.topk(score, k=k, largest=True, sorted=True).indices
			top_idx = idx[top_rel]

			x = cand_feats[top_idx]
			h = self.input_proj(x).unsqueeze(0)
			h = self.encoder(h).squeeze(0)

			d = self.out(h).squeeze(-1)
			d = self.delta_scale * th.tanh(d)

			if self.center_per_spectrum:
				d = d - d.mean()

			delta_flat[top_idx] = d.to(dtype=dtype)

		return delta_flat

class FragGNNModel(nn.Module, CEModel, PrecModel, InstModel):

	def __init__(
		self,
		num_depth: int,
		num_hs: int,
		num_elements: int,
		int_embedder: str,
		int_embedder_tight: bool,
		mol_node_feats: list[str],
		mol_edge_feats: list[str],
		mol_pe_embed_k: int,
		mol_hidden_size: int,
		mol_num_layers: int,
		mol_gnn_type: str,
		mol_normalization: str,
		mol_dropout: float,
		mol_pool_type: str,
		frag_node_feats: list[str],
		frag_edge_feats: list[str],
		frag_hidden_size: int,
		frag_num_layers: int,
		frag_gnn_type: str,
		frag_normalization: str,
		frag_dropout: float,
		frag_pool_type: str,
		frag_embed_combine: str,
		frag_pool_combine: str,
		mlp_output_format: str,
		mlp_hidden_size: int,
		mlp_normalization: str,
		mlp_dropout: float,
		mlp_num_layers: int,
		mlp_use_residuals: bool,
		cc_interstage_type: str,
		nb_iso: bool,
		skip_edge_loss: bool,
		mask_null_formula: bool,
		predict_oos: bool,
		bin_output: bool,
		mz_bin_res: float,
		mz_max: float,
		ce_insert_location: str,
		ce_insert_type: str,
		ce_insert_merge: bool,
		ce_insert_size: int,
  		ce_max: float,
		ce_mean: float,
		ce_std: float,
		prec_insert_location: str,
		prec_insert_size: int,
		prec_types: list[str],
		inst_insert_location: str,
		inst_insert_size: int,
		inst_types: list[str],
		output_formula_str: bool,
		use_ce_fragment_gate: bool = False,
        ce_fragment_gate_hidden_size: int = 128,
        ce_fragment_gate_dropout: float = 0.1,
        ce_fragment_gate_gamma_scale: float = 0.2,
        ce_fragment_gate_use_depth: bool = True,
        use_ce_oos_head: bool = False,
        use_ce_local_transition_prior: bool = False,
        ce_local_transition_hidden_size: int = 256,
        ce_local_transition_dropout: float = 0.1,
        ce_local_transition_delta_scale: float = 0.5,
        use_ce_hchannel_transition_prior: bool = False,
        ce_hchannel_transition_hidden_size: int = 256,
        ce_hchannel_transition_dropout: float = 0.1,
        ce_hchannel_transition_delta_scale: float = 0.05,
        ce_hchannel_preserve_node_score: bool = False,
        use_ce_path_energy: bool = False,
		ce_path_energy_hidden_size: int = 128,
		ce_path_energy_dropout: float = 0.1,
		ce_path_energy_delta_scale: float = 0.3,
		ce_path_energy_max_depth: int = 4,
        use_ce_peak_channel_allocator: bool = False,
        ce_peak_channel_hidden_size: int = 128,
        ce_peak_channel_dropout: float = 0.1,
        ce_peak_channel_delta_scale: float = 0.5,
        ce_peak_channel_max_channels: int = 8,
        ce_peak_channel_allocator_mode: str = "ce_only",
		use_rendered_peak_drop_gate: bool = False,
		rendered_peak_gate_hidden_size: int = 128,
		rendered_peak_gate_dropout: float = 0.1,
		rendered_peak_gate_delta_scale: float = 4.0,
		rendered_peak_gate_init_bias: float = 8.0,
		rendered_peak_gate_max_channels: int = 8,
		rendered_peak_gate_use_extra_features: bool = False,
		use_mz_offset_peak_expansion: bool = False,
		mz_offset_peak_steps: list = None,
		mz_offset_peak_prior_sigma: float = 0.00055,
		use_ce_depth_mixture_head: bool = False,
		ce_depth_mixture_hidden_size: int = 128,
		ce_depth_mixture_dropout: float = 0.1,
		ce_depth_mixture_delta_scale: float = 0.5,
		ce_depth_mixture_num_channels: int = 3,
		use_ce_formula_node_allocator: bool = False,
		ce_formula_node_hidden_size: int = 128,
		ce_formula_node_dropout: float = 0.1,
		ce_formula_node_delta_scale: float = 0.25,
		ce_formula_node_center_per_spectrum: bool = True,
		ce_formula_node_use_depth: bool = True,
  		ce_formula_node_mode: str = "node",
		use_formula_vocab_residual: bool = False,
		formula_vocab_size: int = 4096,
		formula_vocab_hidden_size: int = 256,
		formula_vocab_dropout: float = 0.1,
		formula_vocab_delta_scale: float = 0.02,
		formula_vocab_center_per_spectrum: bool = True,
		formula_vocab_oov_id: int = 0,
		use_formula_comp_residual: bool = False,
		formula_comp_feat_size: int = 18,
		formula_comp_hidden_size: int = 128,
		formula_comp_dropout: float = 0.2,
		formula_comp_delta_scale: float = 0.01,
		formula_comp_center_per_spectrum: bool = True,
		use_ce_response_scorer: bool = False,
		ce_response_hidden_size: int = 128,
		ce_response_dropout: float = 0.1,
		ce_response_delta_scale: float = 0.05,
		ce_response_center_per_spectrum: bool = True,
		ce_response_use_formula_comp: bool = True,
		ce_response_use_depth: bool = True,
		ce_response_use_h: bool = True,
		use_cutchem_node_residual: bool = False,
		cutchem_node_hidden_size: int = 64,
		cutchem_node_dropout: float = 0.1,
		cutchem_node_delta_scale: float = 0.02,
		cutchem_node_center_per_spectrum: bool = True,
		cutchem_node_use_ce: bool = True,
		cutchem_node_use_depth: bool = True,
		cutchem_node_use_h: bool = True,
		use_ce_flowfrag: bool = False,
		ce_flowfrag_hidden_size: int = 128,
		ce_flowfrag_dropout: float = 0.1,
		ce_flowfrag_max_depth: int = 4,
		ce_flowfrag_lambda_max: float = 0.0,
		ce_flowfrag_mixture_hidden_size: int = 128,
		ce_flowfrag_mixture_dropout: float = 0.1,
		ce_flowfrag_mixture_init_bias: float = -6.0,
		ce_flowfrag_delta_clip: float = 5.0,
		ce_flowfrag_use_direct_node: bool = True,
		ce_flowfrag_direct_mix: float = 0.3,
		use_spectrum_candidate_refiner: bool = False,
		spectrum_refiner_hidden_size: int = 128,
		spectrum_refiner_num_layers: int = 2,
		spectrum_refiner_num_heads: int = 4,
		spectrum_refiner_dropout: float = 0.1,
		spectrum_refiner_delta_scale: float = 0.15,
		spectrum_refiner_topk: int = 384,
		spectrum_refiner_center_per_spectrum: bool = True,
		spectrum_refiner_use_logit_feature: bool = True,
		spectrum_refiner_use_mz_features: bool = False,
		spectrum_refiner_use_peak_prior: bool = False,
		use_pre_r54_peak_entry_gate: bool = False,
		pre_r54_peak_entry_hidden_size: int = 128,
		pre_r54_peak_entry_dropout: float = 0.1,
		pre_r54_peak_entry_delta_scale: float = 0.05,
		pre_r54_peak_entry_max_channels: int = 16):

		# nn.Module init
		super().__init__()
		
		self._ce_init(
			int_embedder=int_embedder,
			ce_insert_location=ce_insert_location,
			ce_insert_type=ce_insert_type,
			ce_insert_merge=ce_insert_merge,
			ce_insert_size=ce_insert_size,
			ce_max=ce_max,
			ce_mean=ce_mean,
			ce_std=ce_std
		)

		self._prec_init(
			prec_insert_location=prec_insert_location,
			prec_insert_size=prec_insert_size,
			prec_num_types=len(prec_types)
		)

		self._inst_init(
			inst_insert_location=inst_insert_location,
			inst_insert_size=inst_insert_size,
			inst_num_types=len(inst_types)
		)

		self.num_depth = num_depth
		self.num_hs = num_hs
		self.num_elements = num_elements

		# calculate node/edge feats sizes
		self.mol_node_feats = mol_node_feats
		self.mol_edge_feats = mol_edge_feats
		self.mol_pe_embed_k = mol_pe_embed_k
		self._compute_mol_feats_sizes()

		# setup mol gnn
		self.mol_node_feats_size += self.ce_mol_input_dim + self.prec_mol_input_dim + self.inst_mol_input_dim
		mol_kwargs = {
			"node_feats_size": self.mol_node_feats_size,
			"edge_feats_size": self.mol_edge_feats_size,
			"hidden_size": mol_hidden_size,
			"num_layers": mol_num_layers,
			"gnn_type": mol_gnn_type,
			"dropout": mol_dropout,
			"normalization": mol_normalization,
		}
		
		# Mol GNN
		self.mol_embedder = GNN(**mol_kwargs)
		self.mol_pool_type = mol_pool_type
		self.mol_pool = build_pool_module(mol_pool_type,mol_hidden_size)
		if int_embedder_tight:
			formula_d = {"max_count_int": 255}
			depth_d = {"max_count_int": num_depth+1}
			complement_d = {"max_count_int": 2}
		else:
			formula_d = depth_d = complement_d = {}
		self.formula_embedder = build_formula_embedder(int_embedder,**formula_d)
		self.depth_embedder = build_formula_embedder(int_embedder,**depth_d)
		self.complement_embedder = build_formula_embedder(int_embedder,**complement_d)
		self.frag_node_feats = frag_node_feats
		self.frag_edge_feats = frag_edge_feats
		self._compute_frag_feats_sizes()

		# define interstage
		assert cc_interstage_type in ["add","sub","linear","direct"]
		self.cc_interstage_type = cc_interstage_type
		if self.cc_interstage_type == "linear":
			self.cc_interstage = nn.Linear(mol_hidden_size * 2, mol_hidden_size)

		frag_kwargs = {
			"node_feats_size": self.frag_node_feats_size,
			"edge_feats_size": self.frag_edge_feats_size,
			"hidden_size": frag_hidden_size,
			"num_layers": frag_num_layers,
			"gnn_type": frag_gnn_type,
			"dropout": frag_dropout,
			"normalization": frag_normalization
		}

		self.frag_embedder = GNN(**frag_kwargs)
		self.frag_pool_type = frag_pool_type
		self.frag_pool = build_pool_module(frag_pool_type,frag_hidden_size)
		self.frag_embed_combine = frag_embed_combine
		self.frag_pool_combine = frag_pool_combine
		self.mlp_output_format = mlp_output_format
		# ===== Our CE-Gated Fragment Activation =====
		self.use_ce_fragment_gate = use_ce_fragment_gate
		self.ce_fragment_gate_use_depth = ce_fragment_gate_use_depth

		if self.use_ce_fragment_gate:
			# 第一版要求 CE embedding 存在；保持 ce_insert_location=mlp 即可复用 ce_embedder
			assert self.ce_insert_location != "none", \
				"use_ce_fragment_gate=True requires ce_insert_location != none"
			assert self.ce_insert_size > 0, self.ce_insert_size

			self.ce_fragment_gate = CEFragmentFiLM(
				hidden_size=frag_hidden_size,
				ce_size=self.ce_insert_size,
				gate_hidden_size=ce_fragment_gate_hidden_size,
				dropout=ce_fragment_gate_dropout,
				gamma_scale=ce_fragment_gate_gamma_scale,
				use_depth=ce_fragment_gate_use_depth,
			)
		else:
			self.ce_fragment_gate = None

		# ===== Our CE-conditioned Local Transition Prior =====
		# Residual zero-init CE-conditioned parent->child transition prior.
		# This changes fragmentation mechanism while preserving old behavior at init.
		self.use_ce_local_transition_prior = use_ce_local_transition_prior

		if self.use_ce_local_transition_prior:
			assert self.ce_insert_location != "none", \
				"use_ce_local_transition_prior=True requires ce_insert_location != none"
			assert self.ce_insert_size > 0, self.ce_insert_size

			self.ce_local_transition_prior = CELocalTransitionPrior(
				hidden_size=frag_hidden_size,
				edge_size=self.frag_edge_feats_size,
				ce_size=self.ce_insert_size,
				mlp_hidden_size=ce_local_transition_hidden_size,
				dropout=ce_local_transition_dropout,
				delta_scale=ce_local_transition_delta_scale,
				zero_init=True,
			)
		else:
			self.ce_local_transition_prior = None
		# ===== Our CE-conditioned H-channel Local Transition Prior =====
        # E1c/E1d:
        # Produces a [num_edges, 2*num_hs+1] residual over H-transfer channels.
        # This fixes the E1b problem where cut-edge chemistry only produced
        # one scalar delta shared by all H channels.
		self.use_ce_hchannel_transition_prior = use_ce_hchannel_transition_prior
		self.ce_hchannel_preserve_node_score = ce_hchannel_preserve_node_score
		if self.use_ce_hchannel_transition_prior:
			assert self.ce_insert_location != "none", \
                "use_ce_hchannel_transition_prior=True requires ce_insert_location != none"
			assert self.ce_insert_size > 0, self.ce_insert_size

			self.ce_hchannel_transition_prior = CEHChannelTransitionPrior(
                hidden_size=frag_hidden_size,
                edge_size=self.frag_edge_feats_size,
                ce_size=self.ce_insert_size,
                num_h_channels=2 * self.num_hs + 1,
                mlp_hidden_size=ce_hchannel_transition_hidden_size,
                dropout=ce_hchannel_transition_dropout,
                delta_scale=ce_hchannel_transition_delta_scale,
                zero_init=True,
            )
		else:
			self.ce_hchannel_transition_prior = None
		# ===== Our CE-conditioned Path Energy Propagation =====
        # Learns CE-dependent depth/path bias for fragment nodes.
        # This makes fragmentation prediction path/depth-aware, not only node-score based.
        # Residual zero-init, so initial behavior equals the LocalTrans model.
		self.use_ce_path_energy = use_ce_path_energy
		self.ce_path_energy_delta_scale = float(ce_path_energy_delta_scale)
		self.ce_path_energy_max_depth = int(ce_path_energy_max_depth)

		if self.use_ce_path_energy:
			assert self.ce_insert_location != "none", \
                "use_ce_path_energy=True requires ce_insert_location != none"
			assert self.ce_insert_size > 0, self.ce_insert_size

			self.ce_path_energy_module = nn.Sequential(
                nn.Linear(self.ce_insert_size + 1, ce_path_energy_hidden_size),
                nn.SiLU(),
                nn.Dropout(ce_path_energy_dropout),
                nn.Linear(ce_path_energy_hidden_size, 1),
            )

			last = self.ce_path_energy_module[-1]
			nn.init.zeros_(last.weight)
			nn.init.zeros_(last.bias)
		else:
			self.ce_path_energy_module = None
		# ===== Our CE-aware Learnable Peak Channel Allocation =====
		# Learns CE-dependent redistribution among base and neutral-loss channels.
		# Residual zero-init, so initial behavior equals fixed peak probabilities.
		self.use_ce_peak_channel_allocator = use_ce_peak_channel_allocator
		self.ce_peak_channel_delta_scale = float(ce_peak_channel_delta_scale)
		self.ce_peak_channel_max_channels = int(ce_peak_channel_max_channels)
		self.ce_peak_channel_allocator_mode = ce_peak_channel_allocator_mode
		# ===== R54: m/z-offset peak renderer expansion =====
		# This does not regress m/z. It expands each cached peak-entry into
		# nearby m/z render channels, then lets the peak allocator redistribute
		# intensity among these channels.
		self.use_mz_offset_peak_expansion = bool(use_mz_offset_peak_expansion)

		if mz_offset_peak_steps is None:
			mz_offset_peak_steps = [0.0]

		self.mz_offset_peak_steps = [float(x) for x in mz_offset_peak_steps]
		assert len(self.mz_offset_peak_steps) >= 1, self.mz_offset_peak_steps
		assert any(abs(x) < 1e-12 for x in self.mz_offset_peak_steps), self.mz_offset_peak_steps

		self.mz_offset_peak_prior_sigma = float(mz_offset_peak_prior_sigma)

		if self.use_mz_offset_peak_expansion:
			print(
				"[R54 MZOffsetRenderer] enabled=True, "
				f"steps={self.mz_offset_peak_steps}, "
				f"sigma={self.mz_offset_peak_prior_sigma}"
			)
		if self.use_ce_peak_channel_allocator:
			assert self.ce_insert_location != "none", \
				"use_ce_peak_channel_allocator=True requires ce_insert_location != none"
			assert self.ce_insert_size > 0, self.ce_insert_size

			if self.ce_peak_channel_allocator_mode == "ce_only":
				# v1: global CE -> channel bias. Kept for ablation.
				self.ce_peak_channel_allocator = nn.Sequential(
					nn.Linear(self.ce_insert_size, ce_peak_channel_hidden_size),
					nn.SiLU(),
					nn.Dropout(ce_peak_channel_dropout),
					nn.Linear(ce_peak_channel_hidden_size, self.ce_peak_channel_max_channels),
				)

			elif self.ce_peak_channel_allocator_mode == "entry":
				# v2: per-peak-entry allocator.
				# input = CE embedding + channel one-hot + [mz_norm, base_logprob_norm, formula_logprob_norm]
				entry_input_size = (
					self.ce_insert_size
					+ self.ce_peak_channel_max_channels
					+ 3
				)

				self.ce_peak_channel_allocator = nn.Sequential(
					nn.Linear(entry_input_size, ce_peak_channel_hidden_size),
					nn.SiLU(),
					nn.Dropout(ce_peak_channel_dropout),
					nn.Linear(ce_peak_channel_hidden_size, ce_peak_channel_hidden_size),
					nn.SiLU(),
					nn.Dropout(ce_peak_channel_dropout),
					nn.Linear(ce_peak_channel_hidden_size, 1),
				)

			else:
				raise ValueError(
					f"Unknown ce_peak_channel_allocator_mode="
					f"{self.ce_peak_channel_allocator_mode}"
				)

			last = self.ce_peak_channel_allocator[-1]
			nn.init.zeros_(last.weight)
			nn.init.zeros_(last.bias)
		else:
			self.ce_peak_channel_allocator = None

		# ===== R134: pre-R54 peak-entry scorer =====
		# This acts before m/z-offset expansion, while original peak-entry source
		# channel is still available. It is zero-init, so initial behavior is no-op.
		self.use_pre_r54_peak_entry_gate = bool(use_pre_r54_peak_entry_gate)
		self.pre_r54_peak_entry_delta_scale = float(pre_r54_peak_entry_delta_scale)
		self.pre_r54_peak_entry_max_channels = int(pre_r54_peak_entry_max_channels)

		if self.use_pre_r54_peak_entry_gate:
			assert self.ce_insert_location != "none", (
				"use_pre_r54_peak_entry_gate=True requires CE embedding"
			)
			assert self.ce_insert_size > 0, self.ce_insert_size

			# input = CE embed + original source channel one-hot
			#       + [mz_norm, base_logp_norm, formula_logp_norm,
			#          combined_logp_norm, base_prob_sqrt, formula_prob_sqrt]
			pre_r54_input_size = self.ce_insert_size + self.pre_r54_peak_entry_max_channels + 6

			self.pre_r54_peak_entry_gate = nn.Sequential(
				nn.Linear(pre_r54_input_size, pre_r54_peak_entry_hidden_size),
				nn.SiLU(),
				nn.Dropout(pre_r54_peak_entry_dropout),
				nn.Linear(pre_r54_peak_entry_hidden_size, pre_r54_peak_entry_hidden_size),
				nn.SiLU(),
				nn.Dropout(pre_r54_peak_entry_dropout),
				nn.Linear(pre_r54_peak_entry_hidden_size, 1),
			)

			last = self.pre_r54_peak_entry_gate[-1]
			nn.init.zeros_(last.weight)
			nn.init.zeros_(last.bias)
		else:
			self.pre_r54_peak_entry_gate = None

		print(
			"[R134 PreR54PeakEntryGate] "
			f"enabled={self.use_pre_r54_peak_entry_gate}, "
			f"delta_scale={self.pre_r54_peak_entry_delta_scale}, "
			f"max_channels={self.pre_r54_peak_entry_max_channels}"
		)

		# ===== R71: rendered peak-entry drop gate =====
		# This gate acts after formula/peak rendering, directly on final spectrum entries.
		# It is different from ce_peak_channel_allocator:
		# - peak allocator redistributes mass inside an offset group.
		# - rendered peak gate suppresses false rendered peak entries,
		#   then renormalizes the remaining spectrum mass.
		self.use_rendered_peak_drop_gate = bool(use_rendered_peak_drop_gate)
		self.rendered_peak_gate_delta_scale = float(rendered_peak_gate_delta_scale)
		self.rendered_peak_gate_max_channels = int(rendered_peak_gate_max_channels)
		self.rendered_peak_gate_use_extra_features = bool(rendered_peak_gate_use_extra_features)

		if self.use_rendered_peak_drop_gate:
			assert self.ce_insert_location != "none", (
				"use_rendered_peak_drop_gate=True requires CE embedding"
			)
			assert self.ce_insert_size > 0, self.ce_insert_size

			# input = CE embed + peak-channel one-hot + numeric features.
			# Default keeps old 4 numeric features for compatibility.
			# R111 extra mode adds m/z-shape, channel, and probability-scale features.
			r71_num_numeric = 14 if self.rendered_peak_gate_use_extra_features else 4
			gate_input_size = (
				self.ce_insert_size
				+ self.rendered_peak_gate_max_channels
				+ r71_num_numeric
			)

			self.rendered_peak_drop_gate = nn.Sequential(
				nn.Linear(gate_input_size, rendered_peak_gate_hidden_size),
				nn.SiLU(),
				nn.Dropout(rendered_peak_gate_dropout),
				nn.Linear(rendered_peak_gate_hidden_size, rendered_peak_gate_hidden_size),
				nn.SiLU(),
				nn.Dropout(rendered_peak_gate_dropout),
				nn.Linear(rendered_peak_gate_hidden_size, 1),
			)

			# IMPORTANT:
			# Gate logit means KEEP probability.
			# delta = scale * log(sigmoid(logit)) <= 0.
			# init_bias=+8 makes sigmoid≈1, so delta≈0 and initial model is near no-op.
			last = self.rendered_peak_drop_gate[-1]
			nn.init.zeros_(last.weight)
			nn.init.constant_(last.bias, float(rendered_peak_gate_init_bias))
		else:
			self.rendered_peak_drop_gate = None

		print(
			"[R71 RenderedPeakDropGate] "
			f"enabled={self.use_rendered_peak_drop_gate}, "
			f"delta_scale={self.rendered_peak_gate_delta_scale}, "
			f"max_channels={self.rendered_peak_gate_max_channels}, "
			f"extra_features={self.rendered_peak_gate_use_extra_features}"
		)

		# ===== Our CE-depth Mixture Head =====
		# CE predicts a coarse mixture over fragmentation-depth channels:
		#   0: precursor / shallow fragments
		#   1: primary cleavage
		#   2: deeper / secondary fragments
		#
		# Zero-init makes the initial behavior identical to the old model.
		self.use_ce_depth_mixture_head = use_ce_depth_mixture_head
		self.ce_depth_mixture_delta_scale = float(ce_depth_mixture_delta_scale)
		self.ce_depth_mixture_num_channels = int(ce_depth_mixture_num_channels)

		if self.use_ce_depth_mixture_head:
			assert self.ce_depth_mixture_num_channels == 3, \
				self.ce_depth_mixture_num_channels
			assert self.ce_insert_location != "none", \
				"use_ce_depth_mixture_head=True requires ce_insert_location != none"
			assert self.ce_insert_size > 0, self.ce_insert_size

			self.ce_depth_mixture_head = nn.Sequential(
				nn.Linear(self.ce_insert_size, ce_depth_mixture_hidden_size),
				nn.SiLU(),
				nn.Dropout(ce_depth_mixture_dropout),
				nn.Linear(
					ce_depth_mixture_hidden_size,
					self.ce_depth_mixture_num_channels,
				),
			)

			last = self.ce_depth_mixture_head[-1]
			nn.init.zeros_(last.weight)
			nn.init.zeros_(last.bias)
		else:
			self.ce_depth_mixture_head = None

		# determine mlp input dims
		if self.frag_embed_combine == "cat":
			mlp_input_dim = 2*self.frag_embedder.hidden_size
		else:
			assert self.frag_embed_combine == "avg", self.frag_embed_combine
			mlp_input_dim = self.frag_embedder.hidden_size
		mlp_input_dim += self.ce_mlp_input_dim + self.prec_mlp_input_dim + self.inst_mlp_input_dim

		# ===== L1: CE-aware Formula/Node Allocation Residual =====
		# This module directly adjusts joint logits p(fragment node, H-formula)
		# before scatter_logsoftmax. It is different from peak-channel allocator:
		# peak allocator only redistributes among rendered peaks inside formula,
		# while this module can move probability mass across fragment nodes/formulae.
		self.use_ce_formula_node_allocator = use_ce_formula_node_allocator
		self.ce_formula_node_delta_scale = float(ce_formula_node_delta_scale)
		self.ce_formula_node_center_per_spectrum = bool(ce_formula_node_center_per_spectrum)
		self.ce_formula_node_use_depth = bool(ce_formula_node_use_depth)
		self.ce_formula_node_mode = str(ce_formula_node_mode)
		assert self.ce_formula_node_mode in ["node", "joint"], self.ce_formula_node_mode
		if self.use_ce_formula_node_allocator:
			assert self.mlp_output_format == "formula", \
				"L1 formula/node allocator currently supports mlp_output_format='formula'"
			assert self.ce_insert_location != "none", \
				"use_ce_formula_node_allocator=True requires ce_insert_location != none"
			assert self.ce_insert_size > 0, self.ce_insert_size

			l1_input_dim = mlp_input_dim
			if self.ce_formula_node_use_depth:
				l1_input_dim += 1

			self.ce_formula_node_allocator = nn.Sequential(
				nn.Linear(l1_input_dim, ce_formula_node_hidden_size),
				nn.SiLU(),
				nn.Dropout(ce_formula_node_dropout),
				nn.Linear(ce_formula_node_hidden_size, ce_formula_node_hidden_size),
				nn.SiLU(),
				nn.Dropout(ce_formula_node_dropout),
				nn.Linear(
					ce_formula_node_hidden_size,
					1 if self.ce_formula_node_mode == "node" else 2 * self.num_hs + 1,
				),
			)

			last = self.ce_formula_node_allocator[-1]
			nn.init.zeros_(last.weight)
			nn.init.zeros_(last.bias)
		else:
			self.ce_formula_node_allocator = None

		# ===== K3: true-hit formula-vocabulary residual prior =====
		# Predicts a molecule/CE-conditioned prior over train-only formula vocab.
		# The prior is mapped back to candidate formula rows and added before scatter_logsoftmax.
		self.use_formula_vocab_residual = bool(use_formula_vocab_residual)
		self.formula_vocab_size = int(formula_vocab_size)
		self.formula_vocab_delta_scale = float(formula_vocab_delta_scale)
		self.formula_vocab_center_per_spectrum = bool(formula_vocab_center_per_spectrum)
		self.formula_vocab_oov_id = int(formula_vocab_oov_id)

		if self.use_formula_vocab_residual:
			assert self.mlp_output_format == "formula", (
				"K3 formula-vocab residual currently supports mlp_output_format='formula'"
			)
			assert self.ce_insert_location != "none", (
				"use_formula_vocab_residual=True requires ce_insert_location != none"
			)
			assert self.ce_insert_size > 0, self.ce_insert_size
			assert self.formula_vocab_size > 0, self.formula_vocab_size
			assert 0 <= self.formula_vocab_oov_id <= self.formula_vocab_size, (
				self.formula_vocab_oov_id,
				self.formula_vocab_size,
			)

			k3_input_dim = self.mol_embedder.hidden_size + self.ce_insert_size
			self.formula_vocab_prior_head = nn.Sequential(
				nn.Linear(k3_input_dim, formula_vocab_hidden_size),
				nn.SiLU(),
				nn.Dropout(formula_vocab_dropout),
				nn.Linear(formula_vocab_hidden_size, formula_vocab_hidden_size),
				nn.SiLU(),
				nn.Dropout(formula_vocab_dropout),
				nn.Linear(formula_vocab_hidden_size, self.formula_vocab_size + 1),
			)

			# Zero init: initial behavior equals the R3 backbone.
			last = self.formula_vocab_prior_head[-1]
			nn.init.zeros_(last.weight)
			nn.init.zeros_(last.bias)
		else:
			self.formula_vocab_prior_head = None

		# ===== K3b: formula composition residual scorer =====
		self.use_formula_comp_residual = bool(use_formula_comp_residual)
		self.formula_comp_feat_size = int(formula_comp_feat_size)
		self.formula_comp_delta_scale = float(formula_comp_delta_scale)
		self.formula_comp_center_per_spectrum = bool(formula_comp_center_per_spectrum)

		if self.use_formula_comp_residual:
			assert self.mlp_output_format == "formula", (
				"K3b formula composition residual currently supports mlp_output_format='formula'"
			)

			k3b_input_dim = self.formula_comp_feat_size + mlp_input_dim

			self.formula_comp_residual_head = nn.Sequential(
				nn.Linear(k3b_input_dim, formula_comp_hidden_size),
				nn.SiLU(),
				nn.Dropout(formula_comp_dropout),
				nn.Linear(formula_comp_hidden_size, formula_comp_hidden_size),
				nn.SiLU(),
				nn.Dropout(formula_comp_dropout),
				nn.Linear(formula_comp_hidden_size, 1),
			)

			# zero-init: initial behavior equals the R3 backbone
			last = self.formula_comp_residual_head[-1]
			nn.init.zeros_(last.weight)
			nn.init.zeros_(last.bias)
		else:
			self.formula_comp_residual_head = None

		print(
			"[K3b] formula composition residual: "
			f"enabled={self.use_formula_comp_residual}, "
			f"feat_size={self.formula_comp_feat_size}, "
			f"hidden={formula_comp_hidden_size}, "
			f"dropout={formula_comp_dropout}, "
			f"delta_scale={self.formula_comp_delta_scale}"
		)

		# ===== CE-response candidate scorer =====
		# Direct CE-conditioned candidate intensity response:
		# delta(node, formula/H, CE) is added to frag_joint_logits.
		self.use_ce_response_scorer = bool(use_ce_response_scorer)
		self.ce_response_delta_scale = float(ce_response_delta_scale)
		self.ce_response_center_per_spectrum = bool(ce_response_center_per_spectrum)
		self.ce_response_use_formula_comp = bool(ce_response_use_formula_comp)
		self.ce_response_use_depth = bool(ce_response_use_depth)
		self.ce_response_use_h = bool(ce_response_use_h)

		if self.use_ce_response_scorer:
			assert self.mlp_output_format == "formula", (
				"CE-response scorer currently supports mlp_output_format='formula'"
			)
			assert self.ce_insert_location != "none", (
				"use_ce_response_scorer=True requires CE embedding"
			)
			assert self.ce_insert_size > 0, self.ce_insert_size

			# Inputs:
			# frag_embed                  : mlp_input_dim
			# CE embedding                : ce_insert_size
			# raw CE basis                : 8
			# formula composition feature : formula_comp_feat_size
			# depth                       : 1
			# H-channel features          : 3
			ce_response_input_dim = mlp_input_dim + self.ce_insert_size + 8

			if self.ce_response_use_formula_comp:
				ce_response_input_dim += self.formula_comp_feat_size

			if self.ce_response_use_depth:
				ce_response_input_dim += 1

			if self.ce_response_use_h:
				ce_response_input_dim += 3

			self.ce_response_scorer = nn.Sequential(
				nn.Linear(ce_response_input_dim, ce_response_hidden_size),
				nn.SiLU(),
				nn.Dropout(ce_response_dropout),
				nn.Linear(ce_response_hidden_size, ce_response_hidden_size),
				nn.SiLU(),
				nn.Dropout(ce_response_dropout),
				nn.Linear(ce_response_hidden_size, 1),
			)

			# zero-init: initial behavior equals the current R3/G2 backbone
			last = self.ce_response_scorer[-1]
			nn.init.zeros_(last.weight)
			nn.init.zeros_(last.bias)
		else:
			self.ce_response_scorer = None

		print(
			"[CEResponse] "
			f"enabled={self.use_ce_response_scorer}, "
			f"delta_scale={self.ce_response_delta_scale}, "
			f"formula_comp={self.ce_response_use_formula_comp}, "
			f"depth={self.ce_response_use_depth}, "
			f"h={self.ce_response_use_h}"
		)

		# ===== CutChem-NodeSummary residual =====
		# Keep the original NodeMLP fragment backbone.
		# Aggregate cut-edge chemistry to fragment nodes and add a small
		# zero-init candidate-logit residual.
		self.use_cutchem_node_residual = bool(use_cutchem_node_residual)
		self.cutchem_node_delta_scale = float(cutchem_node_delta_scale)
		self.cutchem_node_center_per_spectrum = bool(cutchem_node_center_per_spectrum)
		self.cutchem_node_use_ce = bool(cutchem_node_use_ce)
		self.cutchem_node_use_depth = bool(cutchem_node_use_depth)
		self.cutchem_node_use_h = bool(cutchem_node_use_h)

		if self.use_cutchem_node_residual:
			assert self.mlp_output_format == "formula", (
				"CutChem-NodeSummary residual currently supports mlp_output_format='formula'"
			)

			cutchem_node_input_dim = mlp_input_dim + 20

			if self.cutchem_node_use_ce:
				assert self.ce_insert_location != "none", (
					"use_cutchem_node_residual=True with cutchem_node_use_ce=True requires CE embedding"
				)
				assert self.ce_insert_size > 0, self.ce_insert_size
				cutchem_node_input_dim += self.ce_insert_size

			if self.cutchem_node_use_depth:
				cutchem_node_input_dim += 1

			if self.cutchem_node_use_h:
				cutchem_node_input_dim += 3

			self.cutchem_node_residual = nn.Sequential(
				nn.Linear(cutchem_node_input_dim, cutchem_node_hidden_size),
				nn.SiLU(),
				nn.Dropout(cutchem_node_dropout),
				nn.Linear(cutchem_node_hidden_size, cutchem_node_hidden_size),
				nn.SiLU(),
				nn.Dropout(cutchem_node_dropout),
				nn.Linear(cutchem_node_hidden_size, 1),
			)

			# zero-init: initial behavior equals the R3/G2 backbone
			last = self.cutchem_node_residual[-1]
			nn.init.zeros_(last.weight)
			nn.init.zeros_(last.bias)
		else:
			self.cutchem_node_residual = None

		print(
			"[CutChemNodeRes] "
			f"enabled={self.use_cutchem_node_residual}, "
			f"delta_scale={self.cutchem_node_delta_scale}, "
			f"center={self.cutchem_node_center_per_spectrum}, "
			f"ce={self.cutchem_node_use_ce}, "
			f"depth={self.cutchem_node_use_depth}, "
			f"h={self.cutchem_node_use_h}"
		)

		# ===== CE-FlowFrag v2: multi-path node-flow prior =====
		self.use_ce_flowfrag = bool(use_ce_flowfrag)
		self.ce_flowfrag_max_depth = int(ce_flowfrag_max_depth)
		self.ce_flowfrag_lambda_max = float(ce_flowfrag_lambda_max)
		self.ce_flowfrag_delta_clip = float(ce_flowfrag_delta_clip)
		self.ce_flowfrag_use_direct_node = bool(ce_flowfrag_use_direct_node)
		self.ce_flowfrag_direct_mix = float(ce_flowfrag_direct_mix)

		if self.use_ce_flowfrag:
			assert self.mlp_output_format == "formula", (
				"CE-FlowFrag v2 currently supports mlp_output_format='formula'"
			)
			assert self.ce_insert_location == "mlp", (
				"CE-FlowFrag v2 expects CE inside frag_embed via ce_insert_location='mlp'"
			)

			# parent, child, child-parent, parent_depth, child_depth, depth_diff
			# frag_embed already contains CE when ce_insert_location='mlp'
			flow_edge_input_dim = 3 * mlp_input_dim + 3

			self.ce_flowfrag_edge_head = nn.Sequential(
				nn.Linear(flow_edge_input_dim, ce_flowfrag_hidden_size),
				nn.SiLU(),
				nn.Dropout(ce_flowfrag_dropout),
				nn.Linear(ce_flowfrag_hidden_size, ce_flowfrag_hidden_size),
				nn.SiLU(),
				nn.Dropout(ce_flowfrag_dropout),
				nn.Linear(ce_flowfrag_hidden_size, 1),
			)

			last = self.ce_flowfrag_edge_head[-1]
			nn.init.zeros_(last.weight)
			nn.init.zeros_(last.bias)

			if self.ce_flowfrag_use_direct_node:
				direct_input_dim = mlp_input_dim + 1

				self.ce_flowfrag_direct_head = nn.Sequential(
					nn.Linear(direct_input_dim, ce_flowfrag_hidden_size),
					nn.SiLU(),
					nn.Dropout(ce_flowfrag_dropout),
					nn.Linear(ce_flowfrag_hidden_size, ce_flowfrag_hidden_size),
					nn.SiLU(),
					nn.Dropout(ce_flowfrag_dropout),
					nn.Linear(ce_flowfrag_hidden_size, 1),
				)

				last = self.ce_flowfrag_direct_head[-1]
				nn.init.zeros_(last.weight)
				nn.init.zeros_(last.bias)
			else:
				self.ce_flowfrag_direct_head = None

			mix_input_dim = self.mol_embedder.hidden_size + self.ce_insert_size

			self.ce_flowfrag_mixture_head = nn.Sequential(
				nn.Linear(mix_input_dim, ce_flowfrag_mixture_hidden_size),
				nn.SiLU(),
				nn.Dropout(ce_flowfrag_mixture_dropout),
				nn.Linear(ce_flowfrag_mixture_hidden_size, 1),
			)

			last = self.ce_flowfrag_mixture_head[-1]
			nn.init.zeros_(last.weight)
			nn.init.constant_(last.bias, ce_flowfrag_mixture_init_bias)
		else:
			self.ce_flowfrag_edge_head = None
			self.ce_flowfrag_direct_head = None
			self.ce_flowfrag_mixture_head = None

		print(
			"[CEFlowFragV2] "
			f"enabled={self.use_ce_flowfrag}, "
			f"lambda_max={self.ce_flowfrag_lambda_max}, "
			f"max_depth={self.ce_flowfrag_max_depth}, "
			f"direct={self.ce_flowfrag_use_direct_node}, "
			f"direct_mix={self.ce_flowfrag_direct_mix}, "
			f"delta_clip={self.ce_flowfrag_delta_clip}"
		)

		if self.mlp_output_format in ["formula","node_formula"]:
			formula_mlp_kwargs = {
				"input_size": mlp_input_dim,
				"output_size": 2*self.num_hs+1,
				"hidden_size": mlp_hidden_size,
				"num_layers": mlp_num_layers,
				"dropout": mlp_dropout,
				"use_residuals": mlp_use_residuals,
				"normalization": mlp_normalization
			}
			self.formula_module = MLPBlocks(**formula_mlp_kwargs)

		# ===== R12: spectrum candidate interaction refiner =====
		self.use_spectrum_candidate_refiner = bool(use_spectrum_candidate_refiner)
		self.spectrum_refiner_topk = int(spectrum_refiner_topk)
		self.spectrum_refiner_delta_scale = float(spectrum_refiner_delta_scale)
		self.spectrum_refiner_center_per_spectrum = bool(spectrum_refiner_center_per_spectrum)
		self.spectrum_refiner_use_logit_feature = bool(spectrum_refiner_use_logit_feature)
		self.spectrum_refiner_use_mz_features = bool(spectrum_refiner_use_mz_features)
		self.spectrum_refiner_use_peak_prior = bool(spectrum_refiner_use_peak_prior)

		if self.use_spectrum_candidate_refiner:
			assert self.mlp_output_format == "formula", (
				"R12 currently supports mlp_output_format='formula' only"
			)
			assert self.ce_insert_location != "none", (
				"R12 needs CE embedding. Use ce_insert_location='mlp'."
			)
			assert self.ce_insert_size > 0, self.ce_insert_size

			# candidate feature:
			# frag_embed + CE embedding + depth scalar + H features + current logit feature
			refiner_input_size = (
				mlp_input_dim
				+ self.ce_insert_size
				+ 1      # depth
				+ 3      # h_norm, h_abs, h_zero
			)
			if self.spectrum_refiner_use_logit_feature:
				refiner_input_size += 1
			if self.spectrum_refiner_use_mz_features:
				refiner_input_size += 3  # mz_norm, mz_log, mz_rank_like
			if self.spectrum_refiner_use_peak_prior:
				refiner_input_size += 1  # log formula_peak_prior

			self.spectrum_candidate_refiner = SpectrumCandidateRefiner(
				input_size=refiner_input_size,
				hidden_size=spectrum_refiner_hidden_size,
				num_layers=spectrum_refiner_num_layers,
				num_heads=spectrum_refiner_num_heads,
				dropout=spectrum_refiner_dropout,
				delta_scale=spectrum_refiner_delta_scale,
				topk=spectrum_refiner_topk,
				center_per_spectrum=spectrum_refiner_center_per_spectrum,
				use_logit_feature=spectrum_refiner_use_logit_feature,
			)
		else:
			self.spectrum_candidate_refiner = None

		print(
			"[R12 SpectrumCandidateRefiner] "
			f"enabled={self.use_spectrum_candidate_refiner}, "
			f"topk={self.spectrum_refiner_topk}, "
			f"hidden={spectrum_refiner_hidden_size}, "
			f"layers={spectrum_refiner_num_layers}, "
			f"heads={spectrum_refiner_num_heads}, "
			f"delta_scale={self.spectrum_refiner_delta_scale}"
		)

		if self.mlp_output_format in ["node_formula"]:
			node_mlp_kwargs = {
				"input_size": 2*self.frag_embedder.hidden_size,
				"output_size": 1,
				"hidden_size": mlp_hidden_size,
				"num_layers": mlp_num_layers,
				"dropout": mlp_dropout,
				"use_residuals": mlp_use_residuals,
				"normalization": mlp_normalization
			}
			self.node_module = MLPBlocks(**node_mlp_kwargs)
		else:
			self.node_module = None
		
		self.predict_oos = predict_oos
		self.use_ce_oos_head = use_ce_oos_head

		if self.use_ce_oos_head:
			assert self.ce_insert_location != "none", \
				"use_ce_oos_head=True requires ce_insert_location != none"
			assert self.ce_insert_size > 0, self.ce_insert_size

		if self.predict_oos:
			# 原始 OOS head 保持不变：只看 molecule pool + fragment pool
			base_oos_input_size = self.mol_embedder.hidden_size + self.frag_embedder.hidden_size

			oos_mlp_kwargs = {
				"input_size": base_oos_input_size,
				"output_size": 1,
				"hidden_size": mlp_hidden_size,
				"num_layers": mlp_num_layers,
				"dropout": mlp_dropout,
				"use_residuals": mlp_use_residuals,
				"normalization": "none"
			}
			self.oos_module = MLPBlocks(**oos_mlp_kwargs)

			# ===== Our CE-aware OOS residual correction =====
			# 不替换原 OOS，只加一个 zero-init residual delta。
			if self.use_ce_oos_head:
				ce_oos_input_size = base_oos_input_size + self.ce_insert_size
				self.ce_oos_delta_module = MLPBlocks(
					input_size=ce_oos_input_size,
					output_size=1,
					hidden_size=mlp_hidden_size,
					num_layers=mlp_num_layers,
					dropout=mlp_dropout,
					use_residuals=mlp_use_residuals,
					normalization="none",
				)

				# 最后一层 zero init，让初始 delta=0，等价于 CE-Gate v1
				last_linear = None
				for m in self.ce_oos_delta_module.modules():
					if isinstance(m, nn.Linear):
						last_linear = m
				assert last_linear is not None
				nn.init.zeros_(last_linear.weight)
				nn.init.zeros_(last_linear.bias)

				self.ce_oos_delta_scale = 0.5
			else:
				self.ce_oos_delta_module = None
				self.ce_oos_delta_scale = 0.0
		else:
			self.oos_module = None
			self.ce_oos_delta_module = None
			self.ce_oos_delta_scale = 0.0

		self.skip_edge_loss = skip_edge_loss
		self.mask_null_formula = mask_null_formula
		self.nb_iso = nb_iso
		self.bin_output = bin_output
		self.mz_bin_res = mz_bin_res
		self.mz_max = mz_max
		self.output_formula_str = output_formula_str

		if self.bin_output:
			self.mz_bins = th.arange(mz_bin_res,mz_max+mz_bin_res,mz_bin_res)

		# this is required
		assert "h_formulae_idx" in self.frag_node_feats

	def _ce_location_check(self):

		assert not self.ce_insert_location == "frag", "ce_insert_location=frag not supported"

	def _prec_location_check(self):

		assert not self.prec_insert_location == "frag", "prec_insert_location=frag not supported"

	def _inst_location_check(self):

		assert not self.inst_insert_location == "frag", "inst_insert_location=frag not supported"

	def _compute_mol_feats_sizes(self):
		""" method compute mol feature size
			these features don't rely on any model parameters
		"""
		self.mol_node_feats_size, self.mol_edge_feats_size = get_mol_feats_sizes(
			self.mol_node_feats, 
			self.mol_edge_feats, 
			self.mol_pe_embed_k
		)

	def _compute_frag_feats_sizes(self):
		""" method compute frag-graph feature size
			these features do depend on model parameters
		"""
		# nodes
		self.frag_node_feats_size = 0
		if "cc" in self.frag_node_feats:
			self.frag_node_feats_size += self.mol_embedder.hidden_size
		if "base_formula" in self.frag_node_feats:
			self.frag_node_feats_size += self.num_elements*self.formula_embedder.num_dim
		if "depth" in self.frag_node_feats:
			self.frag_node_feats_size += self.num_depth*self.depth_embedder.num_dim
		# edges
		self.frag_edge_feats_size = 0
		if "cc" in self.frag_edge_feats:
			self.frag_edge_feats_size += self.mol_embedder.hidden_size
		if "base_formula" in self.frag_edge_feats:
			self.frag_edge_feats_size += self.num_elements*self.formula_embedder.num_dim
		if "cut_chem" in self.frag_edge_feats:
			self.frag_edge_feats_size += CUT_CHEM_EDGE_FEAT_SIZE
		if "complement" in self.frag_edge_feats:
			self.frag_edge_feats_size += self.complement_embedder.num_dim

	def get_compile(self, **kwargs):

		if check_pyg_full_compile():
			return th.compile(self,**kwargs)
		else:
			self.compile_submodules(**kwargs)
			return self

	def compile_submodules(self,**kwargs):
		""" pyg does not support dynamic shape compiling """
		self.formula_embedder = th.compile(self.formula_embedder,**kwargs)
		self.depth_embedder = th.compile(self.depth_embedder,**kwargs)
		self.complement_embedder = th.compile(self.complement_embedder,**kwargs)
		if hasattr(self,"ce_embedder"):
			self.ce_embedder = th.compile(self.ce_embedder,**kwargs)
		if hasattr(self,"m_ce_embedder"):
			self.m_ce_embedder = th.compile(self.m_ce_embedder,**kwargs)
		if check_pyg_compile():
			self.mol_embedder = pyg.compile(self.mol_embedder,**kwargs)
			self.frag_embedder = pyg.compile(self.frag_embedder,**kwargs)

	def _compute_ce_flowfrag_node_logprobs(
		self,
		frag_embed: th.Tensor,
		frag_edge_index: th.Tensor,
		frag_node_batch_idxs: th.Tensor,
		frag_depth_value: th.Tensor,
		batch_size: int,
		device: th.device,
	):
		assert frag_depth_value is not None, "CE-FlowFrag requires depth feature"
		assert frag_edge_index is not None
		assert frag_edge_index.shape[0] == 2, frag_edge_index.shape

		num_nodes = frag_embed.shape[0]
		float_dtype = frag_embed.dtype

		depth_norm = frag_depth_value.reshape(-1).float()
		depth_level = depth_norm * max(float(self.num_depth - 1), 1.0)

		u0 = frag_edge_index[0].long()
		v0 = frag_edge_index[1].long()

		if frag_edge_index.numel() == 0:
			path_node_logprobs = scatter_logsoftmax(
				th.zeros([num_nodes], dtype=float_dtype, device=device),
				frag_node_batch_idxs,
			)
		else:
			depth_u = depth_level[u0]
			depth_v = depth_level[v0]

			u_is_parent = depth_u < depth_v
			v_is_parent = depth_v < depth_u

			parent = th.cat([u0[u_is_parent], v0[v_is_parent]], dim=0)
			child = th.cat([v0[u_is_parent], u0[v_is_parent]], dim=0)

			if not hasattr(self, "_ce_flowfrag_edge_debug_printed"):
				print(
					"[CEFlowFragV2EdgeDebug] "
					f"num_nodes={num_nodes}, "
					f"num_raw_edges={frag_edge_index.shape[1]}, "
					f"num_forward_edges={parent.numel()}, "
					f"depth_level_min={depth_level.min().item():.3f}, "
					f"depth_level_max={depth_level.max().item():.3f}"
				)
				self._ce_flowfrag_edge_debug_printed = True

			if parent.numel() == 0:
				path_node_logprobs = scatter_logsoftmax(
					th.zeros([num_nodes], dtype=float_dtype, device=device),
					frag_node_batch_idxs,
				)
			else:
				parent_depth_feat = depth_norm[parent].unsqueeze(1)
				child_depth_feat = depth_norm[child].unsqueeze(1)
				depth_diff_feat = child_depth_feat - parent_depth_feat

				parent_embed = frag_embed[parent]
				child_embed = frag_embed[child]

				edge_input = th.cat(
					[
						parent_embed,
						child_embed,
						child_embed - parent_embed,
						parent_depth_feat,
						child_depth_feat,
						depth_diff_feat,
					],
					dim=1,
				)

				edge_logits = self.ce_flowfrag_edge_head(edge_input).squeeze(1)

				edge_logprobs = scatter_logsoftmax(edge_logits, parent)

				log_zero = LOG_ZERO(float_dtype)
				node_logflow = th.full(
					[num_nodes],
					log_zero,
					dtype=float_dtype,
					device=device,
				)

				min_depth = scatter_reduce(
					depth_level,
					frag_node_batch_idxs,
					reduce="amin",
					dim_size=batch_size,
				)

				is_root = depth_level <= (
					min_depth[frag_node_batch_idxs] + 1e-6
				)

				root_count = scatter_reduce(
					is_root.to(float_dtype),
					frag_node_batch_idxs,
					reduce="sum",
					dim_size=batch_size,
				).clamp_min(1.0)

				node_logflow[is_root] = -th.log(
					root_count[frag_node_batch_idxs[is_root]]
				)

				parent_depth_int = th.round(depth_level[parent]).long()

				for d in range(self.ce_flowfrag_max_depth):
					edge_mask = parent_depth_int == d
					if not th.any(edge_mask):
						continue

					p_d = parent[edge_mask]
					c_d = child[edge_mask]
					e_lp = edge_logprobs[edge_mask]

					cand = node_logflow[p_d] + e_lp

					child_update = scatter_logsumexp(
						cand,
						c_d,
						dim_size=num_nodes,
					)

					node_logflow = th.logaddexp(
						node_logflow,
						child_update,
					)

				path_node_logprobs = scatter_logsoftmax(
					node_logflow,
					frag_node_batch_idxs,
				)

		if self.ce_flowfrag_use_direct_node:
			direct_input = th.cat(
				[
					frag_embed,
					depth_norm.unsqueeze(1),
				],
				dim=1,
			)

			direct_logits = self.ce_flowfrag_direct_head(
				direct_input
			).squeeze(1)

			direct_node_logprobs = scatter_logsoftmax(
				direct_logits,
				frag_node_batch_idxs,
			)

			direct_mix = float(self.ce_flowfrag_direct_mix)
			direct_mix = max(0.0, min(1.0, direct_mix))

			path_w = th.log(
				th.tensor(
					max(1.0 - direct_mix, 1e-8),
					dtype=float_dtype,
					device=device,
				)
			)
			direct_w = th.log(
				th.tensor(
					max(direct_mix, 1e-8),
					dtype=float_dtype,
					device=device,
				)
			)

			flow_node_logprobs = th.logaddexp(
				path_w + path_node_logprobs,
				direct_w + direct_node_logprobs,
			)
		else:
			flow_node_logprobs = path_node_logprobs

		return flow_node_logprobs

	def forward(
		self, 
		mol_pyg: pyg.data.Data,
		frag_pyg: pyg.data.Data,
		mol_num_nodes: th.Tensor,
		frag_num_nodes: th.Tensor,
		frag_formula_peak_idxs: th.Tensor,
		frag_formula_peak_mzs: th.Tensor,
		frag_formula_peak_probs: th.Tensor,
		frag_formula_vocab_idxs: th.Tensor = None,
		frag_formula_comp_feats: th.Tensor = None,
		frag_formula_peak_channels: th.Tensor = None,
		frag_formula_sizes: th.Tensor = None,
		frag_formula_cumsizes: th.Tensor = None,
		frag_formula_peak_sizes: th.Tensor = None,
		frag_formula_str: np.ndarray = None,
		spec_ce: th.Tensor = None,
		spec_ce_batch_idxs: th.Tensor = None,
		spec_prec_type: th.Tensor = None,
		spec_inst_type: th.Tensor = None,
		spec_prec_type_str: np.ndarray = None,
		**kwargs
	):
		"""forward methods for joint predictor

		Args:
			mol_pyg (pyg.data.Data): molecule pyg data object
			frag_pyg (pyg.data.Data): fragmentation graph pyg data object
			mol_num_nodes (th.Tensor): number of nodes in molecule graph
			frag_num_nodes (th.Tensor): number of nodes in fragmentation graph
			frag_formula_peak_idxs (th.Tensor): _description_
			frag_formula_peak_mzs (th.Tensor): _description_
			frag_formula_peak_probs (th.Tensor): _description_
			frag_formula_sizes (th.Tensor): _description_
			frag_formula_cumsizes (th.Tensor): _description_
			frag_formula_peak_sizes (th.Tensor): _description_

		Returns:
			_type_: _description_
		"""

		# mol_x: mol level node feature matrix
		# mol_edge_index: mol graph connectivity in COO format with shape [2, num_edges]
		# edge_attr: mol graph edge feature matrix with shape [num_edges, num_edge_features]
		# batch: sample idx repsect to current batch
		mol_x, mol_edge_index, mol_edge_attr, mol_batch = mol_pyg.x, mol_pyg.edge_index, mol_pyg.edge_attr, mol_pyg.batch
		# frag_x: frag-graph level node feature matrix
		# frag_edge_index: frag graph connectivity in COO format with shape [2, num_edges]
		# frag_edge_attr: frag graph edge feature matrix with shape [num_edges, num_edge_features]
		# batch: sample idx repsect to current batch
		frag_x, frag_edge_index, frag_edge_attr, frag_batch = frag_pyg.x, frag_pyg.edge_index, frag_pyg.edge_attr, frag_pyg.batch

		device = mol_num_nodes.device
		# int_dtype = mol_edge_index.dtype
		float_dtype = mol_edge_attr.dtype
		frag_node_feat_idxs = frag_pyg.node_feat_idxs[0]
		frag_edge_feat_idxs = frag_pyg.edge_feat_idxs[0]
		batch_frag_num_nodes = frag_x.shape[0]
		batch_frag_num_edges = frag_edge_index.shape[1]
		batch_frag_num_formulae = frag_formula_cumsizes[-1]
		batch_size = frag_batch[-1]+1
		
		# get ce value
		ce = spec_ce
		ce_batch_idxs = spec_ce_batch_idxs
		ce_embed = self.embed_ce(ce, ce_batch_idxs, batch_size)
		# get prec value
		prec_embed = self.embed_prec(spec_prec_type)
		# get inst value
		inst_embed = self.embed_inst(spec_inst_type)

		if self.ce_insert_location == "mol":
			mol_ce_embed = th.repeat_interleave(ce_embed,th.unique(mol_batch,return_counts=True)[1],dim=0)
			mol_x = th.cat([mol_x,mol_ce_embed],dim=1)
		if self.prec_insert_location == "mol":
			mol_prec_embed = th.repeat_interleave(prec_embed,th.unique(mol_batch,return_counts=True)[1],dim=0)
			mol_x = th.cat([mol_x,mol_prec_embed],dim=1)
		if self.inst_insert_location == "mol":
			mol_inst_embed = th.repeat_interleave(inst_embed,th.unique(mol_batch,return_counts=True)[1],dim=0)
			mol_x = th.cat([mol_x,mol_inst_embed],dim=1)

		# get per-atom embeddings
		mol_embed_gnn = self.mol_embedder(
			mol_x,
			mol_batch,
			mol_edge_index,
			mol_edge_attr
		)
		mol_embed_gnn_pool = self.mol_pool(mol_embed_gnn,mol_batch)

		# process dag
		# create interstage
		frag_ndata, frag_edata = [], []
		cutchem_node_summary = None
		# node atom embeddings
		if "cc" in self.frag_node_feats:
			frag_node_mask = th_long_to_mask(get_node_feats(frag_x,frag_node_feat_idxs,"cc").to(device))
			frag_node_mask_idxs = th.nonzero(frag_node_mask).long()
			frag_node_offsets = mol_num_nodes[th.bucketize(frag_node_mask_idxs[:,0],frag_num_nodes,right=True)-1]
			frag_node_mask_idxs[:,1] = frag_node_mask_idxs[:,1] + frag_node_offsets
			frag_node_mask_embed = scatter_reduce(
				src=mol_embed_gnn[frag_node_mask_idxs[:,1]],
				index=frag_node_mask_idxs[:,0:1].expand(-1,mol_embed_gnn.shape[1]),
				reduce="sum",
				dim_size=batch_frag_num_nodes
			)
			# frag_node_mask_embed can be one of following
			# 1. frag_node_mask_embed plus mol_embed_gnn_pool
			# 2. frag_node_mask_embed only, with out pooled embed for entire mol
			# 3. a linear layer between masked embed and pooled embed
			if self.cc_interstage_type == "add":
				frag_node_mask_embed = frag_node_mask_embed + mol_embed_gnn_pool[frag_batch]
			elif self.cc_interstage_type == "sub":
				frag_node_mask_embed = frag_node_mask_embed - mol_embed_gnn_pool[frag_batch]
			elif self.cc_interstage_type == "linear":
				frag_node_mask_embed = self.cc_interstage(th.cat([frag_node_mask_embed,mol_embed_gnn_pool[frag_batch]], dim = 1))
			else:
				assert self.cc_interstage_type == "direct", self.cc_interstage_type
				frag_node_mask_embed = frag_node_mask_embed
			frag_ndata.append(frag_node_mask_embed)
		# edge atom embeddings
		if "cc" in self.frag_edge_feats:
			# connected competents
			frag_edge_mask = th_long_to_mask(get_edge_feats(frag_edge_attr,frag_edge_feat_idxs,"cc").to(device))
			frag_edge_mask_idxs = th.nonzero(frag_edge_mask).long()
			frag_edge_node_idxs = frag_edge_index[0][frag_edge_mask_idxs[:,0]]
			frag_edge_offsets = mol_num_nodes[th.bucketize(frag_edge_node_idxs,frag_num_nodes,right=True)-1]
			frag_edge_mask_idxs[:,1] = frag_edge_mask_idxs[:,1] + frag_edge_offsets
			frag_edge_mask_embed = scatter_reduce(
				src=mol_embed_gnn[frag_edge_mask_idxs[:,1]],
				index=frag_edge_mask_idxs[:,0:1].expand(-1,mol_embed_gnn.shape[1]),
				reduce="sum",
				dim_size=batch_frag_num_edges
			)
			frag_edata.append(frag_edge_mask_embed)
		# node formulae
		if "base_formula" in self.frag_node_feats:
			frag_node_formula = self.formula_embedder(
				get_node_feats(frag_x,frag_node_feat_idxs,"base_formula").reshape(batch_frag_num_nodes,-1)
			)
			frag_ndata.append(frag_node_formula)
		# edge formulae
		if "base_formula" in self.frag_edge_feats:
			frag_edge_formula = self.formula_embedder(
				get_edge_feats(frag_edge_attr,frag_edge_feat_idxs,"base_formula").reshape(batch_frag_num_edges,-1)
			)
			frag_edata.append(frag_edge_formula)
		# E1: chemistry-aware cut-edge features
		# For NodeMLP backbone, cut_chem is not used by the fragment encoder.
		# CutChemNodeRes uses it by aggregating edge chemistry to fragment nodes.
		if ("cut_chem" in self.frag_edge_feats) or self.use_cutchem_node_residual:
			if self.use_cutchem_node_residual and "cut_chem" not in self.frag_edge_feats:
				raise RuntimeError(
					"use_cutchem_node_residual=True requires cut_chem edge features. "
					"Please set frag_params.pyg_edge_feats: ['cut_chem'] in the config."
				)

			frag_edge_cut_chem = get_edge_feats(
				frag_edge_attr,
				frag_edge_feat_idxs,
				"cut_chem",
			).float()

			# Normalize integer-coded cut chemistry features to roughly [0, 1].
			# Feature order:
			# [bond_count, max_order, sum_order, aromatic, ring, conjugated,
			#  hetero, atom_pair_type, child_frac_bin, loss_frac_bin]
			cut_chem_den = th.tensor(
				[8.0, 3.0, 12.0, 1.0, 1.0, 1.0, 1.0, 7.0, 10.0, 10.0],
				dtype=frag_edge_cut_chem.dtype,
				device=device,
			).reshape(1, -1)

			frag_edge_cut_chem = frag_edge_cut_chem / cut_chem_den.clamp_min(1.0)

			if "cut_chem" in self.frag_edge_feats:
				frag_edata.append(frag_edge_cut_chem)

			if self.use_cutchem_node_residual:
				cut_dim = frag_edge_cut_chem.shape[1]
				edge_src = frag_edge_index[0]
				edge_dst = frag_edge_index[1]

				in_index = edge_dst.unsqueeze(1).expand(-1, cut_dim)
				out_index = edge_src.unsqueeze(1).expand(-1, cut_dim)

				in_sum = scatter_reduce(
					src=frag_edge_cut_chem,
					index=in_index,
					reduce="sum",
					dim_size=batch_frag_num_nodes,
				)
				out_sum = scatter_reduce(
					src=frag_edge_cut_chem,
					index=out_index,
					reduce="sum",
					dim_size=batch_frag_num_nodes,
				)

				in_count = scatter_reduce(
					src=th.ones(
						(batch_frag_num_edges, 1),
						dtype=frag_edge_cut_chem.dtype,
						device=device,
					),
					index=edge_dst.unsqueeze(1),
					reduce="sum",
					dim_size=batch_frag_num_nodes,
				).clamp_min(1.0)

				out_count = scatter_reduce(
					src=th.ones(
						(batch_frag_num_edges, 1),
						dtype=frag_edge_cut_chem.dtype,
						device=device,
					),
					index=edge_src.unsqueeze(1),
					reduce="sum",
					dim_size=batch_frag_num_nodes,
				).clamp_min(1.0)

				in_mean = in_sum / in_count
				out_mean = out_sum / out_count

				in_mean = th.nan_to_num(in_mean, nan=0.0, posinf=0.0, neginf=0.0)
				out_mean = th.nan_to_num(out_mean, nan=0.0, posinf=0.0, neginf=0.0)

				cutchem_node_summary = th.cat([in_mean, out_mean], dim=1)
		# node depth
		frag_depth_value = None
		if "depth" in self.frag_node_feats:
			raw_frag_depth = get_node_feats(
				frag_x,
				frag_node_feat_idxs,
				"depth"
			).reshape(batch_frag_num_nodes, -1)

			# 原始 FraGNNet 的 depth 特征在 D3 下不是 1 维标量，
			# 而是 num_depth 维 one-hot / multi-column 表示。
			# CE-Gate 只需要一个归一化 depth scalar，所以这里转成 [N_frag, 1]。
			if raw_frag_depth.shape[1] == 1:
				frag_depth_scalar = raw_frag_depth.float()
				depth_denom = max(float(self.num_depth - 1), 1.0)
			else:
				depth_bins = th.arange(
					raw_frag_depth.shape[1],
					device=device,
					dtype=raw_frag_depth.dtype,
				).reshape(1, -1)

				raw_depth_float = raw_frag_depth.float()
				depth_bins_float = depth_bins.float()

				depth_mass = raw_depth_float.sum(dim=1, keepdim=True).clamp_min(1.0)
				frag_depth_scalar = (
					raw_depth_float * depth_bins_float
				).sum(dim=1, keepdim=True) / depth_mass

				depth_denom = max(float(raw_frag_depth.shape[1] - 1), 1.0)

			frag_depth_value = frag_depth_scalar / depth_denom

			# 原来的 depth embedding 逻辑保持不变，给 Fragment GNN 用
			frag_depth = self.depth_embedder(raw_frag_depth)
			frag_ndata.append(frag_depth)
		# edge complement
		if "complement" in self.frag_edge_feats:
			frag_edge_complement = self.complement_embedder(
				get_edge_feats(frag_edge_attr,frag_edge_feat_idxs,"complement").reshape(batch_frag_num_edges,-1)
			)
			frag_edata.append(frag_edge_complement)
		# empty feats check
		if len(frag_ndata) == 0:
			assert self.frag_node_feats_size == 0, self.frag_node_feats_size
			frag_ndata.append(th.zeros([batch_frag_num_nodes,0],dtype=float_dtype,device=device))
		if len(frag_edata) == 0:
			assert self.frag_edge_feats_size == 0, self.frag_edge_feats_size
			frag_edata.append(th.zeros([batch_frag_num_edges,0],dtype=float_dtype,device=device))

		# get output formula aggregation 
		frag_node_batch_idxs = frag_batch
		frag_formula_batch_idxs = th.repeat_interleave(th.arange(batch_size,device=device),frag_formula_cumsizes[1:]-frag_formula_cumsizes[:-1])
		frag_node_offsets = frag_formula_cumsizes[frag_node_batch_idxs]
		frag_joint_formula_idxs = (get_node_feats(frag_x,frag_node_feat_idxs,"h_formulae_idx")+frag_node_offsets.unsqueeze(-1)).flatten()
		frag_formula_idxs = th.unique(frag_joint_formula_idxs)
		# remove formulae if necessary
		frag_joint_formula_idxs = frag_joint_formula_idxs.reshape(batch_frag_num_nodes,-1)
		num_hs_diff = (frag_joint_formula_idxs.shape[1]-1)//2 - self.num_hs
		assert num_hs_diff >= 0 and num_hs_diff <= (frag_joint_formula_idxs.shape[1]-1)//2, num_hs_diff
		if num_hs_diff > 0:
			# ===== Multi-peak-aware formula pruning =====
			# 原始 FraGNNet 这里默认每个 formula 只有 1 个 peak。
			# 加入 neutral-loss pseudo peaks 后，每个 formula 可能有多个 peak，
			# 所以必须按 peak-entry 级别重新映射，而不能按 formula 级别简单 mask。

			old_frag_formula_cumsizes = frag_formula_cumsizes
			old_frag_formula_peak_idxs = frag_formula_peak_idxs
			old_frag_formula_peak_channels = frag_formula_peak_channels
			old_frag_formula_peak_mzs = frag_formula_peak_mzs
			old_frag_formula_peak_probs = frag_formula_peak_probs
			old_frag_formula_peak_sizes = frag_formula_peak_sizes

			# remove extra H-transfer formula slots
			frag_joint_formula_idxs = frag_joint_formula_idxs[:, :-2*num_hs_diff].flatten()

			# keep NULL formula for every graph by concatenating old cumsizes
			frag_joint_formula_idxs_un, frag_joint_formula_idxs_inv = th.unique(
				th.cat([frag_joint_formula_idxs, old_frag_formula_cumsizes[:-1]], dim=0),
				return_inverse=True
			)

			frag_joint_formula_idxs_inv = frag_joint_formula_idxs_inv[:frag_joint_formula_idxs.shape[0]]
			batch_frag_num_formulae = frag_joint_formula_idxs_un.shape[0]

			# map old global formula idx -> new global formula idx
			new_formula_batch_idxs = frag_formula_batch_idxs[frag_joint_formula_idxs_un]
			frag_joint_formula_idxs = th.arange(
				batch_frag_num_formulae,
				device=device
			)[frag_joint_formula_idxs_inv]

			frag_formula_idxs = th.arange(batch_frag_num_formulae, device=device)
			frag_formula_batch_idxs = new_formula_batch_idxs

			if frag_formula_vocab_idxs is not None:
				frag_formula_vocab_idxs = frag_formula_vocab_idxs[frag_joint_formula_idxs_un]
			if frag_formula_comp_feats is not None:
				frag_formula_comp_feats = frag_formula_comp_feats[frag_joint_formula_idxs_un]

			frag_formula_sizes = scatter_reduce(
				th.ones_like(frag_joint_formula_idxs_un),
				frag_formula_batch_idxs,
				reduce="sum",
				dim_size=batch_size
			)
			assert not th.any(frag_formula_sizes <= 1), frag_formula_sizes

			frag_formula_cumsizes = th.cat(
				[
					th.zeros([1], device=device, dtype=frag_formula_sizes.dtype),
					frag_formula_sizes
				],
				dim=0
			)
			frag_formula_cumsizes = th.cumsum(frag_formula_cumsizes, dim=0)
			frag_node_offsets = frag_formula_cumsizes[frag_node_batch_idxs]

			# ===== peak stuff: multi-peak aware =====
			# old_frag_formula_peak_idxs 是每个 peak-entry 对应的 local formula idx。
			# 先转成 old global formula idx。
			old_peak_batch_idxs = th.repeat_interleave(
				th.arange(batch_size, device=device),
				old_frag_formula_peak_sizes
			)

			old_peak_global_formula_idxs = (
				old_frag_formula_peak_idxs
				+ old_frag_formula_cumsizes[:-1][old_peak_batch_idxs]
			)

			# 只保留仍然存在于 new formula set 里的 peak entries。
			peak_keep_mask = th.isin(
				old_peak_global_formula_idxs,
				frag_joint_formula_idxs_un
			)

			kept_old_global_formula_idxs = old_peak_global_formula_idxs[peak_keep_mask]

			# frag_joint_formula_idxs_un 是 unique 后的旧 global formula idx，
			# searchsorted 得到它们在新 formula table 中的位置。
			kept_new_global_formula_idxs = th.searchsorted(
				frag_joint_formula_idxs_un,
				kept_old_global_formula_idxs
			)

			assert th.all(
				frag_joint_formula_idxs_un[kept_new_global_formula_idxs]
				== kept_old_global_formula_idxs
			)

			kept_batch_idxs = frag_formula_batch_idxs[kept_new_global_formula_idxs]

			# 转回每个 graph 内的 local formula idx，给后面 rendering 用。
			frag_formula_peak_idxs = (
				kept_new_global_formula_idxs
				- frag_formula_cumsizes[:-1][kept_batch_idxs]
			)

			frag_formula_peak_mzs = old_frag_formula_peak_mzs[peak_keep_mask]
			frag_formula_peak_probs = old_frag_formula_peak_probs[peak_keep_mask]
			if old_frag_formula_peak_channels is not None:
				frag_formula_peak_channels = old_frag_formula_peak_channels[peak_keep_mask]
			frag_formula_peak_sizes = scatter_reduce(
				th.ones_like(kept_batch_idxs),
				kept_batch_idxs,
				reduce="sum",
				dim_size=batch_size
			)
		else:
			# no removal required
			frag_joint_formula_idxs = frag_joint_formula_idxs.flatten()

		# get isomorphism aggregation
		if self.nb_iso:
			frag_nb_idxs = get_node_feats(frag_x,frag_node_feat_idxs,"nb_iso_idx").flatten()
			frag_nb_offsets = scatter_reduce(
				frag_nb_idxs,
				frag_node_batch_idxs,
				reduce="amax",
				dim_size=batch_size
			)
			frag_nb_offsets = th.cat(
				[
					th.zeros([1],dtype=frag_nb_offsets.dtype,device=device),
					frag_nb_offsets+1
				], dim=0
			)
			frag_nb_offsets = th.cumsum(frag_nb_offsets,dim=0)
			batch_frag_nb_num_nodes = frag_nb_offsets[-1].item()
			frag_nb_offsets = th.gather(
				input=frag_nb_offsets[:-1],
				index=frag_node_batch_idxs,
				dim=0
			)
			frag_nb_idxs = frag_nb_idxs + frag_nb_offsets
			assert th.max(frag_nb_idxs) < batch_frag_nb_num_nodes, (th.max(frag_nb_idxs),batch_frag_nb_num_nodes)
			frag_nb_un_idxs, frag_nb_inv_idxs = th.unique(frag_nb_idxs,return_inverse=True)

		# assemble all features for dag
		# concatenate everything 
		frag_x_embed = th.cat(frag_ndata,dim=-1)
		# concatenate everything 
		frag_edge_attr_embed = th.cat(frag_edata,dim=-1)

		# define frag network
		frag_embed_gnn = self.frag_embedder(
			frag_x_embed,
			frag_node_batch_idxs,
			frag_edge_index,
			frag_edge_attr_embed
		)
		frag_embed_node = self.frag_embedder.input_project(frag_x_embed)
		frag_embed_gnn_pool = self.frag_pool(frag_embed_gnn,frag_batch)
		frag_embed_node_pool = self.frag_pool(frag_embed_node,frag_batch)
		if self.frag_pool_combine == "subtract":
			frag_embed_gnn = frag_embed_gnn - frag_embed_gnn_pool[frag_batch]
			frag_embed_node = frag_embed_node - frag_embed_node_pool[frag_batch]
		elif self.frag_pool_combine == "add":
			frag_embed_gnn = frag_embed_gnn + frag_embed_gnn_pool[frag_batch]
			frag_embed_node = frag_embed_node + frag_embed_node_pool[frag_batch]
		else:
			assert self.frag_pool_combine == "none", self.frag_pool_combine

		# ===== Our CE-conditioned Local Transition Prior =====
		# Score DAG transition edge parent -> child under CE.
		# Important: some molecules/batches may have no fragment DAG edges.
		# In that case, use zero residual delta so the model falls back to the old path.
		frag_node_transition_delta = None

		if self.use_ce_local_transition_prior:
			assert ce_embed is not None, "CE local transition prior requires ce_embed"

			edge_src = frag_edge_index[0]
			edge_dst = frag_edge_index[1]

			if edge_dst.numel() == 0:
				frag_node_transition_delta = th.zeros(
					[batch_frag_num_nodes],
					dtype=frag_embed_gnn.dtype,
					device=device,
				)
			else:
				edge_batch_idxs = frag_node_batch_idxs[edge_src]
				ce_edge_embed = ce_embed[edge_batch_idxs]

				edge_delta = self.ce_local_transition_prior(
					parent_h=frag_embed_gnn[edge_src],
					child_h=frag_embed_gnn[edge_dst],
					edge_h=frag_edge_attr_embed,
					ce_edge_h=ce_edge_embed,
				)

				frag_node_transition_delta = scatter_reduce(
					edge_delta,
					edge_dst,
					reduce="mean",
					dim_size=batch_frag_num_nodes,
					include_self=False,
				)

				frag_node_transition_delta = th.nan_to_num(
					frag_node_transition_delta,
					nan=0.0,
					posinf=0.0,
					neginf=0.0,
				)
		# ===== E1c/E1d: CE-conditioned H-channel transition prior =====
        # Unlike frag_node_transition_delta, this produces a separate residual
        # for each H-transfer channel:
        #     [num_nodes, 2*num_hs+1]
		frag_node_hchannel_transition_delta = None

		if self.use_ce_hchannel_transition_prior:
			assert ce_embed is not None, "CE H-channel transition prior requires ce_embed"

			edge_src = frag_edge_index[0]
			edge_dst = frag_edge_index[1]
			num_h_channels = 2 * self.num_hs + 1

			if edge_dst.numel() == 0:
				frag_node_hchannel_transition_delta = th.zeros(
                    [batch_frag_num_nodes, num_h_channels],
                    dtype=frag_embed_gnn.dtype,
                    device=device,
                )
			else:
				edge_batch_idxs = frag_node_batch_idxs[edge_src]
				ce_edge_embed = ce_embed[edge_batch_idxs]

				edge_hchannel_delta = self.ce_hchannel_transition_prior(
                    parent_h=frag_embed_gnn[edge_src],
                    child_h=frag_embed_gnn[edge_dst],
                    edge_h=frag_edge_attr_embed,
                    ce_edge_h=ce_edge_embed,
                )

				h_index = edge_dst.unsqueeze(1).expand(
                    -1,
                    edge_hchannel_delta.shape[1],
                )

				frag_node_hchannel_transition_delta = scatter_reduce(
                    edge_hchannel_delta,
                    h_index,
                    reduce="mean",
                    dim_size=batch_frag_num_nodes,
                    include_self=False,
                )

				frag_node_hchannel_transition_delta = th.nan_to_num(
                    frag_node_hchannel_transition_delta,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
		# ===== Our CE-conditioned Path Energy Propagation =====
        # Approximate path/depth by propagating from DAG edges.
        # This avoids relying on cached nodes_min_depth, so old caches still work.
		frag_node_path_energy_delta = None

		if self.use_ce_path_energy:
			assert ce_embed is not None, "CE path energy requires ce_embed"

			edge_src = frag_edge_index[0]
			edge_dst = frag_edge_index[1]

			node_depth = th.zeros(
                [batch_frag_num_nodes],
                dtype=frag_embed_gnn.dtype,
                device=device,
            )

			if edge_dst.numel() > 0:
                # Iterative max-depth propagation.
                # parent -> child increases depth by 1.
                # max_depth=4 is enough for D3/D4 experiments.
				for _ in range(self.ce_path_energy_max_depth):
					cand_depth = node_depth[edge_src] + 1.0
					updated_depth = scatter_reduce(
                        cand_depth,
                        edge_dst,
                        reduce="max",
                        dim_size=batch_frag_num_nodes,
                        include_self=True,
                    )
					node_depth = th.maximum(node_depth, updated_depth)
			node_depth_norm = (
                node_depth / max(float(self.ce_path_energy_max_depth), 1.0)
            ).clamp(0.0, 2.0).unsqueeze(1)
			ce_node_embed = ce_embed[frag_node_batch_idxs]
			path_energy_input = th.cat(
                [
                    ce_node_embed,
                    node_depth_norm,
                ],
                dim=1,
            )
			frag_node_path_energy_delta = self.ce_path_energy_module(
                path_energy_input
            ).squeeze(1)
			frag_node_path_energy_delta = (
                self.ce_path_energy_delta_scale
                * th.tanh(frag_node_path_energy_delta)
            )
		# ===== Our CE-depth Mixture Head =====
		# A coarse, interpretable CE-controlled depth prior.
		# Different from ce_path_energy:
		#   ce_path_energy is a continuous CE-depth scalar MLP;
		#   ce_depth_mixture is a 3-channel CE-controlled probability reallocation.
		frag_node_depth_mixture_delta = None

		if self.use_ce_depth_mixture_head:
			assert ce_embed is not None, "CE-depth mixture requires ce_embed"

			edge_src = frag_edge_index[0]
			edge_dst = frag_edge_index[1]

			# Compute discrete DAG depth from edges. This does not rely on old cache fields.
			node_depth = th.zeros(
				[batch_frag_num_nodes],
				dtype=frag_embed_gnn.dtype,
				device=device,
			)

			if edge_dst.numel() > 0:
				for _ in range(max(int(self.num_depth), 1)):
					cand_depth = node_depth[edge_src] + 1.0
					updated_depth = scatter_reduce(
						cand_depth,
						edge_dst,
						reduce="max",
						dim_size=batch_frag_num_nodes,
						include_self=True,
					)
					node_depth = th.maximum(node_depth, updated_depth)

			# 0: root/shallow, 1: primary cleavage, 2: deeper fragments.
			depth_channel = th.zeros_like(node_depth, dtype=th.long)
			depth_channel = depth_channel + (node_depth >= 1.0).long()
			depth_channel = depth_channel + (node_depth >= 2.0).long()
			depth_channel = depth_channel.clamp(
				min=0,
				max=self.ce_depth_mixture_num_channels - 1,
			)

			ce_depth_logits = self.ce_depth_mixture_head(ce_embed)
			node_depth_delta = ce_depth_logits[
				frag_node_batch_idxs,
				depth_channel,
			]

			frag_node_depth_mixture_delta = (
				self.ce_depth_mixture_delta_scale
				* th.tanh(node_depth_delta)
			)

		# get frag dag embedding
		if self.frag_embed_combine == "cat":
			frag_embed_base_parts = [frag_embed_gnn, frag_embed_node]
		else:
			assert self.frag_embed_combine == "avg", self.frag_embed_combine
			frag_embed_base_parts = [0.5 * frag_embed_gnn + 0.5 * frag_embed_node]

		# ===== Our CE-Gated Fragment Activation =====
		# 对 fragment node embedding 做 CE-conditioned FiLM 调制。
		# 注意：这里仍保留后面的 ce_insert_location == "mlp" 拼接，
		# 这样初期不会破坏原模型的信息路径。
		if self.use_ce_fragment_gate:
			assert ce_embed is not None, "CE gate requires ce_embed"
			ce_node_embed = ce_embed[frag_node_batch_idxs]

			if frag_depth_value is not None:
				gate_depth = frag_depth_value
			else:
				gate_depth = None

			frag_embed_base_parts = [
				self.ce_fragment_gate(part, ce_node_embed, gate_depth)
				for part in frag_embed_base_parts
			]

		frag_embed_parts = frag_embed_base_parts
		
		if self.ce_insert_location == "mlp":
			mlp_ce_embed = th.repeat_interleave(ce_embed,th.unique(frag_node_batch_idxs,return_counts=True)[1],dim=0)
			frag_embed_parts.append(mlp_ce_embed)
		if self.prec_insert_location == "mlp":
			mlp_prec_embed = th.repeat_interleave(prec_embed,th.unique(frag_node_batch_idxs,return_counts=True)[1],dim=0)
			frag_embed_parts.append(mlp_prec_embed)
		if self.inst_insert_location == "mlp":
			mlp_inst_embed = th.repeat_interleave(inst_embed,th.unique(frag_node_batch_idxs,return_counts=True)[1],dim=0)
			frag_embed_parts.append(mlp_inst_embed)

		frag_embed = th.cat(frag_embed_parts, dim=1)

		frag_joint_batch_idxs = th.repeat_interleave(frag_node_batch_idxs,2*self.num_hs+1)
		frag_joint_mask = (~th.isin(frag_joint_formula_idxs,frag_formula_cumsizes[:-1])).float()

		# ===== L1: CE-aware Formula/Node Allocation Residual =====
		frag_formula_node_allocator_delta = None

		if self.use_ce_formula_node_allocator:
			assert self.ce_formula_node_allocator is not None
			assert ce_embed is not None, "L1 formula/node allocator requires ce_embed"

			if self.ce_formula_node_use_depth:
				if frag_depth_value is not None:
					l1_depth = frag_depth_value
				else:
					l1_depth = th.zeros(
						[batch_frag_num_nodes, 1],
						dtype=frag_embed.dtype,
						device=device,
					)
				l1_input = th.cat([frag_embed, l1_depth], dim=1)
			else:
				l1_input = frag_embed

			frag_formula_node_allocator_delta = self.ce_formula_node_allocator(l1_input)

			if self.ce_formula_node_mode == "node":
				assert frag_formula_node_allocator_delta.shape == (
					batch_frag_num_nodes,
					1,
				), frag_formula_node_allocator_delta.shape

				frag_formula_node_allocator_delta = frag_formula_node_allocator_delta.expand(
					-1,
					2 * self.num_hs + 1,
				)

			else:
				assert self.ce_formula_node_mode == "joint", self.ce_formula_node_mode
				assert frag_formula_node_allocator_delta.shape == (
					batch_frag_num_nodes,
					2 * self.num_hs + 1,
				), frag_formula_node_allocator_delta.shape

			frag_formula_node_allocator_delta = (
				self.ce_formula_node_delta_scale
				* th.tanh(frag_formula_node_allocator_delta)
			)

			# Remove per-spectrum mean delta for stability.
			# This keeps the module focused on relative allocation rather than
			# learning a useless global shift, since scatter_logsoftmax is shift-invariant.
			if self.ce_formula_node_center_per_spectrum:
				l1_delta_flat = frag_formula_node_allocator_delta.flatten()
				l1_center = scatter_reduce(
					l1_delta_flat,
					frag_joint_batch_idxs,
					reduce="mean",
					dim_size=batch_size,
					include_self=False,
				)
				l1_center = th.nan_to_num(
					l1_center,
					nan=0.0,
					posinf=0.0,
					neginf=0.0,
				)
				frag_formula_node_allocator_delta = (
					l1_delta_flat - l1_center[frag_joint_batch_idxs]
				).reshape(batch_frag_num_nodes, 2 * self.num_hs + 1)
		# ===== K3: true-hit formula-vocabulary residual prior =====
		formula_vocab_joint_delta = None

		if self.use_formula_vocab_residual:
			assert self.formula_vocab_prior_head is not None
			assert ce_embed is not None, "K3 formula-vocab residual requires ce_embed"
			assert frag_formula_vocab_idxs is not None, (
				"K3 requires batch field frag_formula_vocab_idxs. "
				"Set frag_params.formula_vocab_idxs=True."
			)

			frag_formula_vocab_idxs = frag_formula_vocab_idxs.to(device).long()
			assert frag_formula_vocab_idxs.numel() == batch_frag_num_formulae, (
				frag_formula_vocab_idxs.numel(),
				batch_frag_num_formulae,
			)

			frag_formula_vocab_idxs = frag_formula_vocab_idxs.clamp(
				min=0,
				max=self.formula_vocab_size,
			)

			k3_input = th.cat([mol_embed_gnn_pool, ce_embed], dim=1)
			k3_vocab_logits = self.formula_vocab_prior_head(k3_input)
			assert k3_vocab_logits.shape == (
				batch_size,
				self.formula_vocab_size + 1,
			), k3_vocab_logits.shape

			k3_delta_by_formula = k3_vocab_logits[
				frag_formula_batch_idxs.long(),
				frag_formula_vocab_idxs,
			]
			k3_delta_by_formula = (
				self.formula_vocab_delta_scale
				* th.tanh(k3_delta_by_formula)
			)

			# Do not add arbitrary prior to OOV / NULL rows.
			k3_active = (
				frag_formula_vocab_idxs != self.formula_vocab_oov_id
			).to(dtype=k3_delta_by_formula.dtype)

			k3_delta_by_formula = k3_delta_by_formula * k3_active

			if self.formula_vocab_center_per_spectrum:
				k3_sum = scatter_reduce(
					k3_delta_by_formula,
					frag_formula_batch_idxs,
					reduce="sum",
					dim_size=batch_size,
				)
				k3_count = scatter_reduce(
					k3_active,
					frag_formula_batch_idxs,
					reduce="sum",
					dim_size=batch_size,
				).clamp_min(1.0)

				k3_center = k3_sum / k3_count
				k3_center = th.nan_to_num(
					k3_center,
					nan=0.0,
					posinf=0.0,
					neginf=0.0,
				)

				k3_delta_by_formula = (
					k3_delta_by_formula
					- k3_center[frag_formula_batch_idxs]
				) * k3_active

			formula_vocab_joint_delta = k3_delta_by_formula[
				frag_joint_formula_idxs
			].reshape(batch_frag_num_nodes, 2 * self.num_hs + 1)

		# ===== K3b: formula composition residual scorer =====
		formula_comp_joint_delta = None

		if self.use_formula_comp_residual:
			assert self.formula_comp_residual_head is not None
			assert frag_formula_comp_feats is not None, (
				"K3b requires batch field frag_formula_comp_feats. "
				"Set frag_params.formula_comp_feats=True."
			)

			frag_formula_comp_feats = frag_formula_comp_feats.to(device).float()
			assert frag_formula_comp_feats.shape[0] == batch_frag_num_formulae, (
				frag_formula_comp_feats.shape,
				batch_frag_num_formulae,
			)
			assert frag_formula_comp_feats.shape[1] == self.formula_comp_feat_size, (
				frag_formula_comp_feats.shape,
				self.formula_comp_feat_size,
			)

			num_h_channels = 2 * self.num_hs + 1

			joint_comp = frag_formula_comp_feats[frag_joint_formula_idxs]
			joint_node = frag_embed.unsqueeze(1).expand(
				-1,
				num_h_channels,
				-1,
			).reshape(batch_frag_num_nodes * num_h_channels, -1)

			k3b_input = th.cat([joint_comp, joint_node], dim=1)

			k3b_delta_flat = self.formula_comp_residual_head(k3b_input).squeeze(1)
			k3b_delta_flat = (
				self.formula_comp_delta_scale
				* th.tanh(k3b_delta_flat)
			)

			# Do not apply residual to NULL formula rows.
			k3b_delta_flat = k3b_delta_flat * frag_joint_mask

			if self.formula_comp_center_per_spectrum:
				k3b_sum = scatter_reduce(
					k3b_delta_flat,
					frag_joint_batch_idxs,
					reduce="sum",
					dim_size=batch_size,
				)
				k3b_count = scatter_reduce(
					frag_joint_mask,
					frag_joint_batch_idxs,
					reduce="sum",
					dim_size=batch_size,
				).clamp_min(1.0)

				k3b_center = k3b_sum / k3b_count
				k3b_center = th.nan_to_num(
					k3b_center,
					nan=0.0,
					posinf=0.0,
					neginf=0.0,
				)

				k3b_delta_flat = (
					k3b_delta_flat
					- k3b_center[frag_joint_batch_idxs]
				) * frag_joint_mask

			formula_comp_joint_delta = k3b_delta_flat.reshape(
				batch_frag_num_nodes,
				num_h_channels,
			)


		h_counts = th.zeros([2*self.num_hs+1],device=device,dtype=frag_joint_batch_idxs.dtype)
		h_counts[1+2*th.arange(self.num_hs,device=device)] = -th.arange(1,self.num_hs+1,device=device)
		h_counts[2+2*th.arange(self.num_hs,device=device)] = th.arange(1,self.num_hs+1,device=device)
		frag_joint_h_counts = h_counts.repeat(frag_node_batch_idxs.shape[0])

		# ===== CE-response candidate scorer =====
		cutchem_joint_delta = None
		ce_response_joint_delta = None

		if self.use_ce_response_scorer:
			assert self.ce_response_scorer is not None
			assert ce_embed is not None, "CE-response scorer requires ce_embed"

			num_h_channels = 2 * self.num_hs + 1

			# 1) node embedding for every node/H candidate
			joint_node_embed = frag_embed.unsqueeze(1).expand(
				-1,
				num_h_channels,
				-1,
			).reshape(batch_frag_num_nodes * num_h_channels, -1)

			# 2) CE embedding for every node/H candidate
			joint_ce_embed = ce_embed[frag_node_batch_idxs].unsqueeze(1).expand(
				-1,
				num_h_channels,
				-1,
			).reshape(batch_frag_num_nodes * num_h_channels, -1)

			# 3) raw continuous CE basis
			ce_raw = ce.reshape(-1).to(device).float()

			if ce_raw.numel() == batch_size:
				ce_raw_batch = ce_raw
			else:
				assert ce_batch_idxs is not None, (
					"CE-response scorer needs spec_ce_batch_idxs when CE rows "
					"are not one-per-spectrum"
				)
				ce_raw_batch = scatter_reduce(
					ce_raw,
					ce_batch_idxs.long(),
					reduce="mean",
					dim_size=batch_size,
				)
				ce_raw_batch = th.nan_to_num(
					ce_raw_batch,
					nan=float(self.ce_mean),
					posinf=float(self.ce_mean),
					neginf=float(self.ce_mean),
				)

			ce_z = (
				(ce_raw_batch - float(self.ce_mean))
				/ max(float(self.ce_std), 1e-6)
			)

			ce_scaled = (
				ce_raw_batch / max(float(self.ce_max), 1.0)
			).clamp(0.0, 2.0)

			ce_basis_batch = th.stack(
				[
					ce_scaled,
					ce_z,
					ce_z * ce_z,
					th.log1p(ce_raw_batch.clamp_min(0.0))
					/ th.log1p(
						th.tensor(
							max(float(self.ce_max), 1.0),
							device=device,
							dtype=ce_raw_batch.dtype,
						)
					),
					th.sigmoid((ce_raw_batch - 10.0) / 5.0),
					th.sigmoid((ce_raw_batch - 20.0) / 5.0),
					th.sigmoid((ce_raw_batch - 40.0) / 10.0),
					th.sigmoid((ce_raw_batch - 60.0) / 15.0),
				],
				dim=1,
			).to(dtype=frag_embed.dtype)

			joint_ce_basis = ce_basis_batch[
				frag_node_batch_idxs
			].unsqueeze(1).expand(
				-1,
				num_h_channels,
				-1,
			).reshape(batch_frag_num_nodes * num_h_channels, -1)

			ce_response_parts = [
				joint_node_embed,
				joint_ce_embed,
				joint_ce_basis,
			]

			# 4) formula composition features
			if self.ce_response_use_formula_comp:
				assert frag_formula_comp_feats is not None, (
					"use_ce_response_scorer=True with formula_comp requires "
					"frag_params.formula_comp_feats=True"
				)
				frag_formula_comp_feats = frag_formula_comp_feats.to(device).float()
				assert frag_formula_comp_feats.shape[0] == batch_frag_num_formulae, (
					frag_formula_comp_feats.shape,
					batch_frag_num_formulae,
				)
				assert frag_formula_comp_feats.shape[1] == self.formula_comp_feat_size, (
					frag_formula_comp_feats.shape,
					self.formula_comp_feat_size,
				)

				joint_formula_comp = frag_formula_comp_feats[
					frag_joint_formula_idxs
				]
				ce_response_parts.append(joint_formula_comp)

			# 5) depth feature
			if self.ce_response_use_depth:
				if frag_depth_value is not None:
					depth_feat = frag_depth_value.reshape(-1, 1).to(
						device=device,
						dtype=frag_embed.dtype,
					)
				else:
					depth_feat = th.zeros(
						[batch_frag_num_nodes, 1],
						device=device,
						dtype=frag_embed.dtype,
					)

				joint_depth = depth_feat.unsqueeze(1).expand(
					-1,
					num_h_channels,
					-1,
				).reshape(batch_frag_num_nodes * num_h_channels, -1)
				ce_response_parts.append(joint_depth)

			# 6) H-transfer channel features
			if self.ce_response_use_h:
				h_scalar = frag_joint_h_counts.float().to(device)
				h_norm = h_scalar / max(float(self.num_hs), 1.0)
				h_abs = h_norm.abs()
				h_sign = th.sign(h_norm)
				h_feat = th.stack([h_norm, h_abs, h_sign], dim=1).to(
					dtype=frag_embed.dtype
				)
				ce_response_parts.append(h_feat)

			ce_response_input = th.cat(ce_response_parts, dim=1)

			ce_delta_flat = self.ce_response_scorer(
				ce_response_input
			).squeeze(1)

			ce_delta_flat = (
				self.ce_response_delta_scale
				* th.tanh(ce_delta_flat)
			)

			# Do not score NULL formula rows.
			ce_delta_flat = ce_delta_flat * frag_joint_mask

			if self.ce_response_center_per_spectrum:
				ce_sum = scatter_reduce(
					ce_delta_flat,
					frag_joint_batch_idxs,
					reduce="sum",
					dim_size=batch_size,
				)

				ce_count = scatter_reduce(
					frag_joint_mask,
					frag_joint_batch_idxs,
					reduce="sum",
					dim_size=batch_size,
				).clamp_min(1.0)

				ce_center = ce_sum / ce_count
				ce_center = th.nan_to_num(
					ce_center,
					nan=0.0,
					posinf=0.0,
					neginf=0.0,
				)

				ce_delta_flat = (
					ce_delta_flat
					- ce_center[frag_joint_batch_idxs]
				) * frag_joint_mask

			ce_response_joint_delta = ce_delta_flat.reshape(
				batch_frag_num_nodes,
				num_h_channels,
			)

		# ===== CutChem-NodeSummary residual delta =====
		if self.use_cutchem_node_residual:
			if cutchem_node_summary is None:
				raise RuntimeError(
					"use_cutchem_node_residual=True requires cut_chem edge features. "
					"Please set frag_params.pyg_edge_feats: ['cut_chem'] in the config."
				)

			num_h_channels = 2 * self.num_hs + 1
			# In this forward path, frag_joint_mask is a flat vector:
			# [batch_frag_num_nodes * num_h_channels].
			# CutChemNodeRes works with [batch_frag_num_nodes, num_h_channels],
			# so make an explicit 2D mask.
			if frag_joint_mask.dim() == 1:
				assert frag_joint_mask.numel() == batch_frag_num_nodes * num_h_channels, (
					frag_joint_mask.shape,
					batch_frag_num_nodes,
					num_h_channels,
				)
				cutchem_mask_2d = frag_joint_mask.reshape(
					batch_frag_num_nodes,
					num_h_channels,
				).bool()
			else:
				assert frag_joint_mask.shape == (batch_frag_num_nodes, num_h_channels), (
					frag_joint_mask.shape,
					batch_frag_num_nodes,
					num_h_channels,
				)
				cutchem_mask_2d = frag_joint_mask.bool()
			joint_node_embed = (
				frag_embed
				.unsqueeze(1)
				.expand(-1, num_h_channels, -1)
				.reshape(batch_frag_num_nodes * num_h_channels, -1)
			)

			joint_cutchem_summary = (
				cutchem_node_summary
				.unsqueeze(1)
				.expand(-1, num_h_channels, -1)
				.reshape(batch_frag_num_nodes * num_h_channels, -1)
			)

			cutchem_inputs = [joint_node_embed, joint_cutchem_summary]

			if self.cutchem_node_use_ce:
				joint_ce_embed = (
					ce_embed[frag_node_batch_idxs]
					.unsqueeze(1)
					.expand(-1, num_h_channels, -1)
					.reshape(batch_frag_num_nodes * num_h_channels, -1)
				)
				cutchem_inputs.append(joint_ce_embed)

			if self.cutchem_node_use_depth:
				if frag_depth_value is None:
					joint_depth = th.zeros(
						(batch_frag_num_nodes, 1),
						dtype=frag_embed.dtype,
						device=device,
					)
				else:
					joint_depth = frag_depth_value.to(dtype=frag_embed.dtype)

				joint_depth = (
					joint_depth
					.unsqueeze(1)
					.expand(-1, num_h_channels, -1)
					.reshape(batch_frag_num_nodes * num_h_channels, -1)
				)
				cutchem_inputs.append(joint_depth)

			if self.cutchem_node_use_h:
				h_vals = frag_joint_h_counts.to(dtype=frag_embed.dtype)
				h_abs = h_vals.abs() / max(float(self.num_hs), 1.0)
				h_sign = th.sign(h_vals)
				h_zero = (h_vals == 0).to(dtype=frag_embed.dtype)
				joint_h_feats = th.stack(
					[h_abs, h_sign, h_zero],
					dim=-1,
				).reshape(batch_frag_num_nodes * num_h_channels, 3)
				cutchem_inputs.append(joint_h_feats)

			cutchem_input = th.cat(cutchem_inputs, dim=1)

			cutchem_joint_delta = self.cutchem_node_residual(cutchem_input).reshape(
				batch_frag_num_nodes,
				num_h_channels,
			)

			cutchem_joint_delta = self.cutchem_node_delta_scale * th.tanh(
				cutchem_joint_delta
			)

			cutchem_joint_delta = th.where(
				cutchem_mask_2d,
				cutchem_joint_delta,
				th.zeros_like(cutchem_joint_delta),
			)

			if self.cutchem_node_center_per_spectrum:
				cutchem_flat_delta = cutchem_joint_delta.flatten()
				cutchem_flat_mask = cutchem_mask_2d.flatten()
				cutchem_flat_batch = (
					frag_node_batch_idxs
					.unsqueeze(1)
					.expand(-1, num_h_channels)
					.flatten()
				)

				cutchem_valid_delta = th.where(
					cutchem_flat_mask,
					cutchem_flat_delta,
					th.zeros_like(cutchem_flat_delta),
				)

				cutchem_count = scatter_reduce(
					src=cutchem_flat_mask.to(dtype=cutchem_flat_delta.dtype).unsqueeze(1),
					index=cutchem_flat_batch.unsqueeze(1),
					reduce="sum",
					dim_size=batch_size,
				).squeeze(1).clamp_min(1.0)

				cutchem_sum = scatter_reduce(
					src=cutchem_valid_delta.unsqueeze(1),
					index=cutchem_flat_batch.unsqueeze(1),
					reduce="sum",
					dim_size=batch_size,
				).squeeze(1)

				cutchem_mean = cutchem_sum / cutchem_count

				cutchem_flat_delta = th.where(
					cutchem_flat_mask,
					cutchem_flat_delta - cutchem_mean[cutchem_flat_batch],
					cutchem_flat_delta,
				)

				cutchem_joint_delta = cutchem_flat_delta.reshape(
					batch_frag_num_nodes,
					num_h_channels,
				)

		if self.mlp_output_format == "formula":

			# log p(f,n)
			frag_joint_logits = self.formula_module(frag_embed)
			assert frag_joint_logits.shape[1] == 2*self.num_hs+1, (frag_joint_logits.shape[1],2*self.num_hs+1)

			if frag_node_transition_delta is not None:
				frag_joint_logits = frag_joint_logits + frag_node_transition_delta.unsqueeze(-1)
			if frag_node_hchannel_transition_delta is not None:
				assert frag_node_hchannel_transition_delta.shape == frag_joint_logits.shape, (
					frag_node_hchannel_transition_delta.shape,
					frag_joint_logits.shape,
				)

				if self.ce_hchannel_preserve_node_score:
					# Preserve per-node total logit mass.
					# The H-channel transition is only allowed to redistribute probability
					# among H-transfer channels, not change the node's total score.
					old_node_lse = th.logsumexp(
						frag_joint_logits,
						dim=1,
						keepdim=True,
					)

					new_joint_logits = frag_joint_logits + frag_node_hchannel_transition_delta

					new_node_lse = th.logsumexp(
						new_joint_logits,
						dim=1,
						keepdim=True,
					)

					frag_joint_logits = new_joint_logits - (new_node_lse - old_node_lse)

				else:
					frag_joint_logits = frag_joint_logits + frag_node_hchannel_transition_delta
			if frag_node_path_energy_delta is not None:
				frag_joint_logits = frag_joint_logits + frag_node_path_energy_delta.unsqueeze(-1)

			if frag_node_depth_mixture_delta is not None:
				frag_joint_logits = frag_joint_logits + frag_node_depth_mixture_delta.unsqueeze(-1)

			if frag_formula_node_allocator_delta is not None:
				assert frag_formula_node_allocator_delta.shape == frag_joint_logits.shape, (
					frag_formula_node_allocator_delta.shape,
					frag_joint_logits.shape,
				)
				frag_joint_logits = frag_joint_logits + frag_formula_node_allocator_delta

			if formula_vocab_joint_delta is not None:
				assert formula_vocab_joint_delta.shape == frag_joint_logits.shape, (
					formula_vocab_joint_delta.shape,
					frag_joint_logits.shape,
				)
				frag_joint_logits = frag_joint_logits + formula_vocab_joint_delta

			if formula_comp_joint_delta is not None:
				assert formula_comp_joint_delta.shape == frag_joint_logits.shape, (
					formula_comp_joint_delta.shape,
					frag_joint_logits.shape,
				)
				frag_joint_logits = frag_joint_logits + formula_comp_joint_delta

			if cutchem_joint_delta is not None:
				assert cutchem_joint_delta.shape == frag_joint_logits.shape, (
					cutchem_joint_delta.shape,
					frag_joint_logits.shape,
				)
				frag_joint_logits = frag_joint_logits + cutchem_joint_delta

			if ce_response_joint_delta is not None:
				assert ce_response_joint_delta.shape == frag_joint_logits.shape, (
					ce_response_joint_delta.shape,
					frag_joint_logits.shape,
				)
				frag_joint_logits = frag_joint_logits + ce_response_joint_delta

			# ===== R56: expose R12 candidate features for offline snap/delta gate =====
			r56_real_joint_r12_feats = None

			# ===== R12: spectrum-level candidate interaction refinement =====
			# This is applied after all local residuals but before scatter_logsoftmax.
			if self.use_spectrum_candidate_refiner:
				assert self.spectrum_candidate_refiner is not None
				assert ce_embed is not None, "R12 requires CE embedding"

				num_h_channels = 2 * self.num_hs + 1

				# [num_nodes * H, frag_embed_dim]
				joint_node_embed = (
					frag_embed
					.unsqueeze(1)
					.expand(-1, num_h_channels, -1)
					.reshape(batch_frag_num_nodes * num_h_channels, -1)
				)

				# [num_nodes * H, ce_dim]
				joint_ce_embed = (
					ce_embed[frag_node_batch_idxs]
					.unsqueeze(1)
					.expand(-1, num_h_channels, -1)
					.reshape(batch_frag_num_nodes * num_h_channels, -1)
				)

				# depth scalar
				if frag_depth_value is not None:
					depth_feat = frag_depth_value.to(dtype=frag_embed.dtype)
				else:
					depth_feat = th.zeros(
						[batch_frag_num_nodes, 1],
						dtype=frag_embed.dtype,
						device=device,
					)

				joint_depth = (
					depth_feat
					.unsqueeze(1)
					.expand(-1, num_h_channels, -1)
					.reshape(batch_frag_num_nodes * num_h_channels, 1)
				)

				# H-channel features
				h_vals = frag_joint_h_counts.to(dtype=frag_embed.dtype)
				h_norm = h_vals / max(float(self.num_hs), 1.0)
				h_abs = h_norm.abs()
				h_zero = (h_vals == 0).to(dtype=frag_embed.dtype)
				h_feat = th.stack(
					[h_norm, h_abs, h_zero],
					dim=1,
				)

				r12_feats = [
					joint_node_embed,
					joint_ce_embed,
					joint_depth,
					h_feat,
				]

				# ===== R13: explicit candidate m/z features =====
				# R12 only saw latent node embedding + CE + depth + H + old logit.
				# R13 gives the refiner explicit spectrum-coordinate information.
				if self.spectrum_refiner_use_mz_features:
					assert frag_joint_formula_idxs.dim() == 1, frag_joint_formula_idxs.shape
					joint_mz = frag_formula_peak_mzs[
						frag_joint_formula_idxs
					].to(device=device, dtype=frag_embed.dtype)

					mz_max_t = th.tensor(
						float(self.mz_max),
						device=device,
						dtype=frag_embed.dtype,
					).clamp_min(1.0)

					mz_norm = (joint_mz / mz_max_t).clamp(min=0.0, max=2.0)
					mz_log = th.log1p(joint_mz.clamp_min(0.0)) / th.log1p(mz_max_t)
					mz_rank_like = mz_norm

					r12_feats.append(
						th.stack(
							[
								mz_norm,
								mz_log,
								mz_rank_like,
							],
							dim=1,
						)
					)

				# ===== R13: cached formula peak prior =====
				# formula_peak_probs is already part of the fragment cache.
				# This tells the refiner which formula candidates were a priori plausible.
				if self.spectrum_refiner_use_peak_prior:
					assert frag_joint_formula_idxs.dim() == 1, frag_joint_formula_idxs.shape
					joint_peak_prior = frag_formula_peak_probs[
						frag_joint_formula_idxs
					].to(device=device, dtype=frag_embed.dtype)
					joint_peak_prior = (
						joint_peak_prior
						.clamp_min(1.0e-12)
						.log()
						.unsqueeze(1)
					)
					r12_feats.append(joint_peak_prior)

				frag_joint_logits_flat_for_r12 = frag_joint_logits.reshape(-1)

				if self.spectrum_refiner_use_logit_feature:
					r12_feats.append(
						frag_joint_logits_flat_for_r12.detach().unsqueeze(1)
					)

				r12_feats = th.cat(r12_feats, dim=1)
				r56_real_joint_r12_feats = r12_feats[frag_joint_mask.reshape(-1).bool()].detach()

				r12_delta_flat = self.spectrum_candidate_refiner(
					cand_feats=r12_feats,
					base_logits_flat=frag_joint_logits_flat_for_r12,
					cand_batch_idxs=frag_joint_batch_idxs,
					valid_mask=frag_joint_mask.bool(),
					batch_size=int(batch_size),
				)

				r12_delta = r12_delta_flat.reshape(
					batch_frag_num_nodes,
					num_h_channels,
				)

				assert r12_delta.shape == frag_joint_logits.shape, (
					r12_delta.shape,
					frag_joint_logits.shape,
				)

				frag_joint_logits = frag_joint_logits + r12_delta

			frag_joint_logits = frag_joint_logits.flatten()

			# compute total NULL probability (before renormalization)
			frag_joint_logprobs = scatter_logsoftmax(
				frag_joint_logits,
				frag_joint_batch_idxs
			)
			frag_null_formula_logprob = scatter_logsumexp(
				(1.-frag_joint_mask) * frag_joint_logprobs + frag_joint_mask * LOG_ZERO(frag_joint_logprobs.dtype),
				frag_joint_batch_idxs
			)
			frag_null_formula_logprob = th.clamp(frag_null_formula_logprob, max=0.)

			if self.mask_null_formula:
				# compute non-NULL renormalized intensity
				frag_joint_logprobs = scatter_masked_softmax(
					frag_joint_logits,
					frag_joint_mask,
					frag_joint_batch_idxs,
					log=True
				)
			
			# reshape
			frag_joint_logprobs = frag_joint_logprobs.reshape(-1,2*self.num_hs+1)

			# Base node/emission decomposition.
			base_joint_logprobs = frag_joint_logprobs

			base_node_logprobs = th.logsumexp(
				base_joint_logprobs,
				dim=1,
			)
			base_node_logprobs = scatter_logsoftmax(
				base_node_logprobs,
				frag_node_batch_idxs,
			)

			base_emit_logprobs = (
				base_joint_logprobs
				- base_node_logprobs.unsqueeze(1)
			)
			base_emit_logprobs = th.log_softmax(
				base_emit_logprobs,
				dim=1,
			)

			if self.use_ce_flowfrag:
				flow_node_logprobs = self._compute_ce_flowfrag_node_logprobs(
					frag_embed=frag_embed,
					frag_edge_index=frag_edge_index,
					frag_node_batch_idxs=frag_node_batch_idxs,
					frag_depth_value=frag_depth_value,
					batch_size=batch_size,
					device=device,
				)

				mix_input = th.cat([mol_embed_gnn_pool, ce_embed], dim=1)

				lambda_flow = (
					self.ce_flowfrag_lambda_max
					* th.sigmoid(self.ce_flowfrag_mixture_head(mix_input))
				).flatten()

				if not hasattr(self, "_ce_flowfrag_debug_printed"):
					print(
						"[CEFlowFragV2Debug] "
						f"lambda_mean={lambda_flow.detach().mean().item():.6f}, "
						f"lambda_min={lambda_flow.detach().min().item():.6f}, "
						f"lambda_max={lambda_flow.detach().max().item():.6f}"
					)
					self._ce_flowfrag_debug_printed = True

				lambda_node = lambda_flow[
					frag_node_batch_idxs
				].clamp(
					min=0.0,
					max=1.0,
				)

				node_delta = flow_node_logprobs - base_node_logprobs
				node_delta = node_delta.clamp(
					min=-self.ce_flowfrag_delta_clip,
					max=self.ce_flowfrag_delta_clip,
				)

				final_node_logits = (
					base_node_logprobs
					+ lambda_node * node_delta
				)

				frag_node_logprobs = scatter_logsoftmax(
					final_node_logits,
					frag_node_batch_idxs,
				)

				frag_joint_logprobs = (
					frag_node_logprobs.unsqueeze(1)
					+ base_emit_logprobs
				)
			else:
				frag_node_logprobs = base_node_logprobs
				frag_joint_logprobs = base_joint_logprobs

			frag_node_formula_logprobs = base_emit_logprobs

		else:

			assert self.mlp_output_format == "node_formula", self.mlp_output_format

			# log p(f|n)
			frag_node_formula_logits = self.formula_module(frag_embed)
			assert frag_node_formula_logits.shape[1] == 2*self.num_hs+1, (frag_node_formula_logits.shape[1],2*self.num_hs+1)
			if frag_node_hchannel_transition_delta is not None:
				assert frag_node_hchannel_transition_delta.shape == frag_node_formula_logits.shape, (
                    frag_node_hchannel_transition_delta.shape,
                    frag_node_formula_logits.shape,
                )
				frag_node_formula_logits = frag_node_formula_logits + frag_node_hchannel_transition_delta
			frag_node_formula_logprobs = th.log_softmax(frag_node_formula_logits,dim=1)

			# log p(n)
			frag_node_logits = self.node_module(frag_embed).squeeze(1)

			if frag_node_transition_delta is not None:
				frag_node_logits = frag_node_logits + frag_node_transition_delta

			if frag_node_path_energy_delta is not None:
				frag_node_logits = frag_node_logits + frag_node_path_energy_delta

			if frag_node_depth_mixture_delta is not None:
				frag_node_logits = frag_node_logits + frag_node_depth_mixture_delta

			frag_node_logprobs = scatter_logsoftmax(
				frag_node_logits,
				frag_node_batch_idxs
			)

			# log p(f,n) = log p(f|n) + log p(n)
			frag_joint_logprobs = frag_node_formula_logprobs + frag_node_logprobs.unsqueeze(-1)
			frag_joint_logprobs = frag_joint_logprobs.flatten()

			# compute total NULL probability (before renormalization)
			frag_null_formula_logprob = scatter_logsumexp(
				(1.-frag_joint_mask) * frag_joint_logprobs + frag_joint_mask * LOG_ZERO(frag_joint_logprobs.dtype),
				frag_joint_batch_idxs
			)
			frag_null_formula_logprob = th.clamp(frag_null_formula_logprob, max=0.)

			if self.mask_null_formula:
				# compute non-NULL renormalized intensity
				frag_joint_logprobs = scatter_masked_softmax(
					frag_joint_logprobs,
					frag_joint_mask,
					frag_joint_batch_idxs,
					log=True
				)

		# aggregate by formula
		# log p(f) = logsumexp_n log p(f,n)
		frag_formula_mask = th.ones_like(frag_formula_batch_idxs,dtype=float_dtype)
		frag_formula_mask[frag_formula_cumsizes[:-1]] = 0.
		# aggregate by formula
		frag_formula_logprobs = scatter_logsumexp(
			frag_joint_logprobs.flatten(),
			frag_joint_formula_idxs
		)

		# softmax over formulae
		if self.mask_null_formula:
			frag_formula_logprobs = scatter_masked_softmax(
				frag_formula_logprobs,
				frag_formula_mask,
				frag_formula_batch_idxs,
				log=True
			)
		else:
			frag_formula_logprobs = scatter_masked_softmax(
				frag_formula_logprobs,
				th.ones_like(frag_formula_logprobs),
				frag_formula_batch_idxs,
				log=True
			)

		if self.predict_oos:
            # 原始 OOS logits
			base_oos_input = th.cat(
                [mol_embed_gnn_pool, frag_embed_gnn_pool],
                dim=1,
            )
			oos_logits = self.oos_module(base_oos_input).flatten()

            # ===== Our CE-aware OOS residual correction =====
            # 注意：这里不是替换 OOS，而是 residual 修正。
			if self.use_ce_oos_head:
				assert ce_embed is not None, "CE-aware OOS requires ce_embed"
				assert ce_embed.shape[0] == batch_size, (ce_embed.shape, batch_size)

				ce_oos_input = th.cat(
                    [mol_embed_gnn_pool, frag_embed_gnn_pool, ce_embed],
                    dim=1,
                )
				ce_oos_delta = self.ce_oos_delta_module(ce_oos_input).flatten()

                # tanh + scale，防止 CE delta 过大，破坏 OOS 分支
				oos_logits = oos_logits + self.ce_oos_delta_scale * th.tanh(ce_oos_delta)

			oos_logprobs = F.logsigmoid(oos_logits)
			not_oos_logprobs = F.logsigmoid(-oos_logits)
		else:
			# set them to 0
			oos_logprobs = LOG_ZERO(frag_formula_logprobs.dtype)*th.ones([batch_size],device=device)
			not_oos_logprobs = th.zeros([batch_size],device=device)
		
		# adjust frag_formula_logprobs
		frag_formula_oos_logprobs = frag_formula_logprobs + \
			th.repeat_interleave(not_oos_logprobs, frag_formula_sizes, dim=0)
		# ===== R58B: local offset-channel allocator bookkeeping =====
		# These tensors are only used when R54 m/z-offset expansion is enabled.
		# They let the allocator redistribute probability within each original
		# cached peak-entry's offset copies, instead of across all peaks of a formula.
		r58_offset_group_idxs = None
		r58_old_peak_logprobs = None
		r58_offset_prior_logprobs = None

		# ===== V8B: preserve original pre-R54 peak-entry identity =====
		# These tensors contain no trainable parameters. They only expose the
		# source channel, original m/z and original conditional peak prior that
		# existed before R54 replaced the channel id with the offset id.
		v8b_original_peak_channels = None
		v8b_original_peak_mzs = None
		v8b_original_peak_probs = None

		# ===== R134: pre-R54 peak-entry scorer =====
		pre_r54_peak_entry_gate_logits = None
		pre_r54_peak_entry_gate_delta = None

		if self.use_pre_r54_peak_entry_gate and frag_formula_peak_mzs.numel() > 0:
			assert frag_formula_peak_channels is not None, (
				"use_pre_r54_peak_entry_gate=True requires original frag_formula_peak_channels"
			)

			pre_spec_batch_idxs = th.repeat_interleave(
				th.arange(frag_formula_peak_sizes.shape[0], device=device),
				frag_formula_peak_sizes,
			)

			pre_formula_offsets = frag_formula_cumsizes[:-1][pre_spec_batch_idxs]
			pre_formula_global_idxs = frag_formula_peak_idxs + pre_formula_offsets

			pre_base_logprobs = th.log(frag_formula_peak_probs.clamp_min(1e-12))
			pre_formula_logprobs = frag_formula_oos_logprobs[pre_formula_global_idxs]
			pre_combined_logprobs = pre_base_logprobs + pre_formula_logprobs

			pre_channels = frag_formula_peak_channels.long().clamp(
				min=0,
				max=self.pre_r54_peak_entry_max_channels - 1,
			)

			pre_channel_oh = th.nn.functional.one_hot(
				pre_channels,
				num_classes=self.pre_r54_peak_entry_max_channels,
			).to(dtype=pre_base_logprobs.dtype)

			pre_ce_embed = ce_embed[pre_spec_batch_idxs]

			pre_mz_norm = (
				frag_formula_peak_mzs.float() / max(float(self.mz_max), 1.0)
			).clamp(0.0, 1.5)

			pre_base_logprob_norm = pre_base_logprobs.clamp(min=-20.0, max=0.0) / 20.0
			pre_formula_logprob_norm = pre_formula_logprobs.clamp(min=-20.0, max=0.0) / 20.0
			pre_combined_logprob_norm = pre_combined_logprobs.clamp(min=-20.0, max=0.0) / 20.0

			pre_base_prob_sqrt = pre_base_logprobs.clamp(min=-20.0, max=0.0).exp().sqrt().clamp(0.0, 1.0)
			pre_formula_prob_sqrt = pre_formula_logprobs.clamp(min=-20.0, max=0.0).exp().sqrt().clamp(0.0, 1.0)

			pre_numeric = th.stack(
				[
					pre_mz_norm,
					pre_base_logprob_norm,
					pre_formula_logprob_norm,
					pre_combined_logprob_norm,
					pre_base_prob_sqrt,
					pre_formula_prob_sqrt,
				],
				dim=1,
			).to(dtype=pre_base_logprobs.dtype)

			pre_input = th.cat(
				[
					pre_ce_embed,
					pre_channel_oh,
					pre_numeric,
				],
				dim=1,
			)

			pre_r54_peak_entry_gate_logits = self.pre_r54_peak_entry_gate(pre_input).squeeze(1)
			pre_r54_peak_entry_gate_delta = (
				self.pre_r54_peak_entry_delta_scale
				* th.tanh(pre_r54_peak_entry_gate_logits)
			)

			# Add delta to original peak-entry prior, then renormalize within each
			# global formula id. This preserves total formula probability mass.
			pre_new_logprobs = pre_base_logprobs + pre_r54_peak_entry_gate_delta

			old_lse = scatter_logsumexp(
				pre_base_logprobs,
				pre_formula_global_idxs,
				dim_size=batch_frag_num_formulae,
			)
			new_lse = scatter_logsumexp(
				pre_new_logprobs,
				pre_formula_global_idxs,
				dim_size=batch_frag_num_formulae,
			)

			pre_new_logprobs = (
				pre_new_logprobs
				- new_lse[pre_formula_global_idxs]
				+ old_lse[pre_formula_global_idxs]
			)

			frag_formula_peak_probs = pre_new_logprobs.exp()

		# ===== R54: m/z-offset peak-entry expansion =====
		# Expand each cached formula peak-entry into several nearby m/z copies.
		# The copies share the same formula idx, but have different rendered m/z.
		# Their initial probabilities are normalized Gaussian priors, so the total
		# peak-entry probability mass is preserved before the allocator.
		if self.use_mz_offset_peak_expansion:
			offset_steps = th.tensor(
				self.mz_offset_peak_steps,
				device=device,
				dtype=frag_formula_peak_mzs.dtype,
			)

			num_offset_channels = int(offset_steps.numel())
			assert num_offset_channels >= 1

			prior_sigma = max(float(self.mz_offset_peak_prior_sigma), 1e-8)
			offset_priors = th.exp(
				-0.5 * (offset_steps / prior_sigma) ** 2
			)
			offset_priors = offset_priors / offset_priors.sum().clamp_min(1e-12)

			num_old_peak_entries = int(frag_formula_peak_mzs.shape[0])

			if num_old_peak_entries > 0:
				old_peak_idxs = frag_formula_peak_idxs
				old_peak_mzs = frag_formula_peak_mzs
				old_peak_probs = frag_formula_peak_probs

				if frag_formula_peak_channels is None:
					old_peak_channels = th.zeros(
						num_old_peak_entries,
						device=device,
						dtype=th.long,
					)
				else:
					old_peak_channels = frag_formula_peak_channels.long()

				assert old_peak_channels.shape[0] == num_old_peak_entries, (
					old_peak_channels.shape,
					num_old_peak_entries,
				)

				# Preserve the pre-R54 identity on every expanded offset copy.
				v8b_original_peak_channels = th.repeat_interleave(
					old_peak_channels,
					num_offset_channels,
					dim=0,
				)

				v8b_original_peak_mzs = th.repeat_interleave(
					old_peak_mzs,
					num_offset_channels,
					dim=0,
				)

				v8b_original_peak_probs = th.repeat_interleave(
					old_peak_probs,
					num_offset_channels,
					dim=0,
				)

				# R58B local offset groups:
				# group id = original cached peak-entry id.
				# All 5 offset copies of the same original peak-entry share one group.
				r58_offset_group_idxs = th.repeat_interleave(
					th.arange(num_old_peak_entries, device=device, dtype=th.long),
					num_offset_channels,
					dim=0,
				)

				# old peak-entry probability must be preserved.
				r58_old_peak_logprobs = th.log(
					th.repeat_interleave(
						old_peak_probs.clamp_min(1e-12),
						num_offset_channels,
						dim=0,
					)
				)

				# offset prior only controls allocation inside the 5 offset copies.
				r58_offset_prior_logprobs = th.log(
					offset_priors.repeat(num_old_peak_entries).clamp_min(1e-12)
				)

				frag_formula_peak_idxs = th.repeat_interleave(
					old_peak_idxs,
					num_offset_channels,
					dim=0,
				)

				frag_formula_peak_mzs = (
					th.repeat_interleave(
						old_peak_mzs,
						num_offset_channels,
						dim=0,
					)
					+ offset_steps.repeat(num_old_peak_entries)
				)

				frag_formula_peak_probs = (
					th.repeat_interleave(
						old_peak_probs,
						num_offset_channels,
						dim=0,
					)
					* offset_priors.repeat(num_old_peak_entries)
				)

				# For R54, channel id means offset channel.
				# Use ce_peak_channel_max_channels >= num_offset_channels.
				frag_formula_peak_channels = th.arange(
					num_offset_channels,
					device=device,
					dtype=th.long,
				).repeat(num_old_peak_entries)

				frag_formula_peak_sizes = frag_formula_peak_sizes * num_offset_channels

				if not hasattr(self, "_r54_debug_printed"):
					print(
						"[R54 DEBUG] "
						f"enabled={self.use_mz_offset_peak_expansion}, "
						f"num_offset_channels={num_offset_channels}, "
						f"old_peak_entries={num_old_peak_entries}, "
						f"new_peak_entries={frag_formula_peak_mzs.shape[0]}, "
						f"bin_output={self.bin_output}, "
						f"mz_bin_res={self.mz_bin_res}"
					)
					self._r54_debug_printed = True
				if self.use_ce_peak_channel_allocator:
					assert self.ce_peak_channel_max_channels >= num_offset_channels, (
						self.ce_peak_channel_max_channels,
						num_offset_channels,
					)
		# convert to spectrum
		# ===== Multi-peak-aware formula rendering =====
		# 原始 FraGNNet 在 num_isotopes=1 时，每个 non-null formula 只有 1 个 peak，
		# 所以可以用 frag_formula_sizes-1 来 repeat formula offset。
		# 加入 neutral-loss pseudo peaks 后，每个 formula 可能有多个 peak-entry，
		# 因此 offset 必须按 peak-entry 的 batch idx 来取。
		spec_mzs = frag_formula_peak_mzs

		spec_batch_idxs = th.repeat_interleave(
			th.arange(frag_formula_peak_sizes.shape[0], device=device),
			frag_formula_peak_sizes
		)

		frag_formula_offsets = frag_formula_cumsizes[:-1][spec_batch_idxs]

		assert frag_formula_peak_idxs.shape[0] == frag_formula_peak_probs.shape[0], (
			frag_formula_peak_idxs.shape,
			frag_formula_peak_probs.shape,
		)
		assert frag_formula_peak_idxs.shape[0] == spec_batch_idxs.shape[0], (
			frag_formula_peak_idxs.shape,
			spec_batch_idxs.shape,
		)
		assert frag_formula_peak_mzs.shape[0] == frag_formula_peak_probs.shape[0], (
			frag_formula_peak_mzs.shape,
			frag_formula_peak_probs.shape,
		)

		spec_formula_global_idxs = frag_formula_peak_idxs + frag_formula_offsets

		assert spec_formula_global_idxs.shape[0] == frag_formula_peak_probs.shape[0], (
			spec_formula_global_idxs.shape,
			frag_formula_peak_probs.shape,
		)

		base_peak_logprobs = th.log(frag_formula_peak_probs.clamp_min(1e-12))

		if self.use_ce_peak_channel_allocator:
			assert frag_formula_peak_channels is not None, (
				"use_ce_peak_channel_allocator=True requires frag_formula_peak_channels "
				"from batch_mols_frags"
			)

			peak_channels = frag_formula_peak_channels.long().clamp(
				min=0,
				max=self.ce_peak_channel_max_channels - 1,
			)

			if self.ce_peak_channel_allocator_mode == "ce_only":
				ce_channel_logits = self.ce_peak_channel_allocator(ce_embed)
				ce_channel_logits = ce_channel_logits[spec_batch_idxs]

				peak_channel_delta = ce_channel_logits.gather(
					1,
					peak_channels.unsqueeze(1),
				).squeeze(1)

			elif self.ce_peak_channel_allocator_mode == "entry":
				ce_peak_embed = ce_embed[spec_batch_idxs]

				peak_channel_oh = F.one_hot(
					peak_channels,
					num_classes=self.ce_peak_channel_max_channels,
				).to(dtype=base_peak_logprobs.dtype)

				mz_norm = (
					spec_mzs.float()
					/ max(float(self.mz_max), 1.0)
				).clamp(0.0, 1.5)

				base_logprob_norm = (
					base_peak_logprobs.clamp(min=-20.0, max=0.0)
					/ 20.0
				)

				formula_logprob_norm = (
					frag_formula_oos_logprobs[spec_formula_global_idxs]
					.clamp(min=-20.0, max=0.0)
					/ 20.0
				)

				peak_numeric = th.stack(
					[
						mz_norm,
						base_logprob_norm,
						formula_logprob_norm,
					],
					dim=1,
				)

				peak_alloc_input = th.cat(
					[
						ce_peak_embed,
						peak_channel_oh,
						peak_numeric,
					],
					dim=1,
				)

				peak_channel_delta = self.ce_peak_channel_allocator(
					peak_alloc_input
				).squeeze(1)

			else:
				raise ValueError(
					f"Unknown ce_peak_channel_allocator_mode="
					f"{self.ce_peak_channel_allocator_mode}"
				)

			peak_channel_delta = self.ce_peak_channel_delta_scale * th.tanh(
				peak_channel_delta
			)

			# R58B:
			# For R54 offset expansion, normalize only within each original peak-entry's
			# offset copies. This preserves the old cached peak-entry probability and
			# only reallocates it among [-0.002,-0.001,0,+0.001,+0.002].
			if (
				self.use_mz_offset_peak_expansion
				and r58_offset_group_idxs is not None
				and r58_old_peak_logprobs is not None
				and r58_offset_prior_logprobs is not None
			):
				peak_offset_logits = r58_offset_prior_logprobs + peak_channel_delta
				peak_offset_logprobs = scatter_logsoftmax(
					peak_offset_logits,
					r58_offset_group_idxs,
				)
				peak_logprobs = r58_old_peak_logprobs + peak_offset_logprobs
			else:
				peak_logits = base_peak_logprobs + peak_channel_delta

				# Fallback old behavior for non-R54 cases.
				peak_logprobs = scatter_logsoftmax(
					peak_logits,
					spec_formula_global_idxs,
				)
		else:
			peak_logprobs = base_peak_logprobs

		spec_logprobs = (
			frag_formula_oos_logprobs[spec_formula_global_idxs]
			+ peak_logprobs
		)

		# ===== R71: rendered peak-entry drop gate =====
		# Directly suppress false rendered peaks.
		# After applying negative gate delta, renormalize inside each spectrum
		# so the original spectrum probability mass is preserved.
		rendered_peak_gate_logits = None
		rendered_peak_gate_delta = None

		if self.use_rendered_peak_drop_gate:
			assert self.rendered_peak_drop_gate is not None
			assert ce_embed is not None

			ce_peak_embed = ce_embed[spec_batch_idxs]

			if frag_formula_peak_channels is not None:
				r71_peak_channels = frag_formula_peak_channels.long().clamp(
					min=0,
					max=self.rendered_peak_gate_max_channels - 1,
				)
			else:
				r71_peak_channels = th.zeros_like(spec_batch_idxs).long()

			peak_channel_oh = F.one_hot(
				r71_peak_channels,
				num_classes=self.rendered_peak_gate_max_channels,
			).to(dtype=spec_logprobs.dtype)

			mz_norm = (
				spec_mzs.float()
				/ max(float(self.mz_max), 1.0)
			).clamp(0.0, 1.5)

			base_peak_logprob_norm = (
				base_peak_logprobs.clamp(min=-20.0, max=0.0) / 20.0
			)

			formula_logprob_norm = (
				frag_formula_oos_logprobs[spec_formula_global_idxs]
				.clamp(min=-20.0, max=0.0)
				/ 20.0
			)

			peak_logprob_norm = (
				peak_logprobs.clamp(min=-20.0, max=0.0) / 20.0
			)

			if self.rendered_peak_gate_use_extra_features:
				mz2_norm = (mz_norm * mz_norm).clamp(0.0, 2.25)
				is_low_mz = (mz_norm < 0.15).to(dtype=spec_logprobs.dtype)
				is_mid_mz = ((mz_norm >= 0.15) & (mz_norm < 0.50)).to(dtype=spec_logprobs.dtype)
				is_high_mz = (mz_norm >= 0.50).to(dtype=spec_logprobs.dtype)

				channel_norm = (
					r71_peak_channels.to(dtype=spec_logprobs.dtype)
					/ max(float(self.rendered_peak_gate_max_channels - 1), 1.0)
				).clamp(0.0, 1.0)

				combined_logprob_norm = (
					spec_logprobs.clamp(min=-20.0, max=0.0) / 20.0
				)

				base_peak_prob_sqrt = (
					base_peak_logprobs.clamp(min=-20.0, max=0.0).exp().sqrt()
				).clamp(0.0, 1.0)
				formula_prob_sqrt = (
					frag_formula_oos_logprobs[spec_formula_global_idxs]
					.clamp(min=-20.0, max=0.0)
					.exp()
					.sqrt()
				).clamp(0.0, 1.0)
				peak_prob_sqrt = (
					peak_logprobs.clamp(min=-20.0, max=0.0).exp().sqrt()
				).clamp(0.0, 1.0)
				combined_prob_sqrt = (
					spec_logprobs.clamp(min=-20.0, max=0.0).exp().sqrt()
				).clamp(0.0, 1.0)

				r71_numeric = th.stack(
					[
						mz_norm,
						mz2_norm,
						is_low_mz,
						is_mid_mz,
						is_high_mz,
						channel_norm,
						base_peak_logprob_norm,
						formula_logprob_norm,
						peak_logprob_norm,
						combined_logprob_norm,
						base_peak_prob_sqrt,
						formula_prob_sqrt,
						peak_prob_sqrt,
						combined_prob_sqrt,
					],
					dim=1,
				).to(dtype=spec_logprobs.dtype)
			else:
				r71_numeric = th.stack(
					[
						mz_norm,
						base_peak_logprob_norm,
						formula_logprob_norm,
						peak_logprob_norm,
					],
					dim=1,
				).to(dtype=spec_logprobs.dtype)

			r71_input = th.cat(
				[
					ce_peak_embed,
					peak_channel_oh,
					r71_numeric,
				],
				dim=1,
			)

			rendered_peak_gate_logits = self.rendered_peak_drop_gate(
				r71_input
			).squeeze(1)

			# log keep-probability, <= 0.
			# Positive/hit peaks should learn high logits => delta≈0.
			# False peaks should learn low logits => negative delta.
			rendered_peak_gate_delta = (
				self.rendered_peak_gate_delta_scale
				* F.logsigmoid(rendered_peak_gate_logits)
			)

			old_lse = scatter_logsumexp(
				spec_logprobs,
				spec_batch_idxs,
			)

			gated_logits = spec_logprobs + rendered_peak_gate_delta

			new_lse = scatter_logsumexp(
				gated_logits,
				spec_batch_idxs,
			)

			spec_logprobs = (
				gated_logits
				- new_lse[spec_batch_idxs]
				+ old_lse[spec_batch_idxs]
			)

		if not self.skip_edge_loss:

			frag_edge_batch_idxs = frag_batch[frag_edge_index[0]]
			frag_edge_logits = frag_node_logprobs[frag_edge_index[0]] + frag_node_logprobs[frag_edge_index[1]]
			frag_edge_logprobs = scatter_logsoftmax(
				frag_edge_logits,
				frag_edge_batch_idxs
			)
			# print(scatter_logsumexp(frag_edge_logprobs,frag_edge_batch_idxs))
			frag_node_h_counts = get_node_feats(frag_x,frag_node_feat_idxs,"h_counts")
			frag_edge_h_ranges = get_edge_feats(frag_edge_attr,frag_edge_feat_idxs,"h_range")
			frag_edge_h_diffs = frag_node_h_counts[frag_edge_index[0]].unsqueeze(1) - frag_node_h_counts[frag_edge_index[1]].unsqueeze(2)
			frag_edge_h_diffs = frag_edge_h_diffs.reshape(frag_edge_h_diffs.shape[0],-1)
			frag_edge_h_range_masks = th.logical_or(
				frag_edge_h_diffs<frag_edge_h_ranges[:,0].unsqueeze(-1), 
				frag_edge_h_diffs>frag_edge_h_ranges[:,1].unsqueeze(-1)
			)
			frag_edge_h_logprobs = (frag_node_formula_logprobs[frag_edge_index[0]]).unsqueeze(1)  \
				+ (frag_node_formula_logprobs[frag_edge_index[1]]).unsqueeze(2)
			frag_edge_h_logprobs = frag_edge_h_logprobs.reshape(frag_edge_h_logprobs.shape[0],-1) 

		else:

			frag_edge_logprobs = None
			frag_edge_h_diffs = None
			frag_edge_h_range_masks = None
			frag_edge_h_logprobs = None
			frag_edge_batch_idxs = None

		frag_joint_node_idxs = th.repeat_interleave(th.arange(frag_joint_mask.shape[0]//(2*self.num_hs+1),device=device),2*self.num_hs+1)
		# select (will remove all NULL formula idxs, potentially some node idxs too if they contain only NULLs)
		frag_real_joint_logits = frag_joint_logits[frag_joint_mask.bool()]
		frag_real_joint_h_counts = frag_joint_h_counts[frag_joint_mask.bool()]
		frag_real_joint_node_idxs = frag_joint_node_idxs[frag_joint_mask.bool()]
		frag_real_joint_formula_idxs = frag_joint_formula_idxs[frag_joint_mask.bool()]
		frag_real_joint_batch_idxs = frag_joint_batch_idxs[frag_joint_mask.bool()]
		# P(f,n)
		frag_real_joint_logprobs = scatter_logsoftmax(
			frag_real_joint_logits,
			frag_real_joint_batch_idxs
		)
		# P(n) - sum, renormalize, but keep all-NULL nodes (as zeros)
		frag_real_node_node_idxs = th.arange(batch_frag_num_nodes,device=device)
		frag_real_node_logprobs = scatter_logsumexp(
			frag_real_joint_logprobs,
			frag_real_joint_node_idxs,
			dim_size=batch_frag_num_nodes
		)
		frag_real_node_logprobs = scatter_logsoftmax(
			frag_real_node_logprobs,
			frag_node_batch_idxs
		)
		# P(f) - sum, renormalize, but keep NULL formulae (as zeros)
		frag_real_formula_logprobs = scatter_logsumexp(
			frag_real_joint_logprobs,
			frag_real_joint_formula_idxs,
			dim_size=batch_frag_num_formulae
		)
		frag_real_formula_logprobs = scatter_logsoftmax(
			frag_real_formula_logprobs,
			frag_formula_batch_idxs,
		)
		frag_real_formula_formula_idxs = th.arange(batch_frag_num_formulae,device=device)
		# P(f|n) - remove all NULLs from conditionals
		frag_real_node_formula_logprobs = frag_real_joint_logprobs - frag_real_node_logprobs[frag_real_joint_node_idxs]
		frag_real_node_formula_logprobs = scatter_logsoftmax(
			frag_real_node_formula_logprobs,
			frag_real_joint_node_idxs
		)
		# P(n|f) - remove all NULLs from conditionals
		frag_real_formula_node_logprobs = frag_real_joint_logprobs - frag_real_formula_logprobs[frag_real_joint_formula_idxs]
		frag_real_formula_node_logprobs = scatter_logsoftmax(
			frag_real_formula_node_logprobs,
			frag_real_joint_formula_idxs
		)
		# hydrogens
		frag_real_joint_h_idxs = (frag_real_joint_h_counts + self.num_hs) + frag_real_joint_batch_idxs * (2*self.num_hs+1)
		frag_real_h_logprobs = scatter_logsumexp(
			frag_real_joint_logprobs,
			frag_real_joint_h_idxs,
			dim_size=batch_size*(2*self.num_hs+1)
		)
		frag_real_h_logprobs = th.clamp(frag_real_h_logprobs, max=0.)
		frag_real_h_counts = th.arange(-self.num_hs,self.num_hs+1,device=device).repeat(batch_size)
		frag_real_h_batch_idxs = th.repeat_interleave(th.arange(batch_size,device=device),2*self.num_hs+1)

		# calculate isomorphic distributions
		if self.nb_iso:
			# P(n')
			frag_nb_node_logprobs = scatter_logsumexp(
				frag_real_node_logprobs,
				frag_nb_idxs,
				dim_size=batch_frag_nb_num_nodes
			)
			frag_nb_node_batch_idxs = scatter_reduce(
				frag_node_batch_idxs,
				frag_nb_idxs,
				reduce="amax",
				dim_size=batch_frag_nb_num_nodes
			)
			frag_nb_node_node_idxs = frag_nb_un_idxs
			# P(n'|f)
			frag_nb_joint_idxs = frag_nb_idxs[frag_real_joint_node_idxs]
			frag_nb_joint_both_idxs = th.stack(
				[
					frag_nb_joint_idxs,
					frag_real_joint_formula_idxs
				], dim=1
			)
			frag_nb_joint_both_un_idxs, frag_nb_joint_both_inv_idxs = th.unique(
				frag_nb_joint_both_idxs,
				return_inverse=True,
				dim=0
			)
			frag_nb_joint_node_idxs = frag_nb_joint_both_un_idxs[:,0]
			frag_nb_joint_formula_idxs = frag_nb_joint_both_un_idxs[:,1]
			frag_nb_joint_batch_idxs = frag_nb_node_batch_idxs[frag_nb_joint_node_idxs]
			frag_nb_formula_node_logprobs = scatter_logsumexp(
				frag_real_formula_node_logprobs,
				frag_nb_joint_both_inv_idxs,
				dim_size=frag_nb_joint_both_un_idxs.shape[0]
			)
			frag_nb_formula_node_logprobs = th.clamp(frag_nb_formula_node_logprobs, max=0.)
			# P(n|n')
			frag_nb_node_node_logprobs = scatter_logsoftmax(
				frag_real_node_logprobs,
				frag_nb_idxs
			)
			frag_nb_node_node_node_idxs = frag_nb_idxs # frag_nb_node_node_idxs[frag_nb_inv_idxs]
			frag_nb_node_node_batch_idxs = frag_nb_node_batch_idxs[frag_nb_inv_idxs]
			assert th.all(frag_nb_node_node_logprobs <= 0.)
			# P(f|n')
			frag_nb_node_formula_logprobs = frag_real_node_formula_logprobs + frag_nb_node_node_logprobs[frag_real_joint_node_idxs]
			frag_nb_node_formula_logprobs = scatter_logsumexp(
				frag_nb_node_formula_logprobs,
				frag_nb_joint_both_inv_idxs,
				dim_size=frag_nb_joint_both_un_idxs.shape[0]
			)
			frag_nb_node_formula_logprobs = th.clamp(frag_nb_node_formula_logprobs, max=0.)
			# P(f,n')
			frag_nb_joint_logprobs = scatter_logsumexp(
				frag_real_joint_logprobs,
				frag_nb_joint_both_inv_idxs,
				dim_size=frag_nb_joint_both_un_idxs.shape[0]
			)
		else:
			frag_nb_node_logprobs = None
			frag_nb_node_formula_logprobs = None
			frag_nb_formula_node_logprobs = None
			frag_nb_node_node_logprobs = None
			frag_nb_node_node_idxs = None
			frag_nb_node_batch_idxs = None
			frag_nb_joint_node_idxs = None
			frag_nb_joint_formula_idxs = None
			frag_nb_joint_batch_idxs = None
			frag_nb_joint_logprobs = None
			frag_nb_node_node_node_idxs = None
			frag_nb_node_node_batch_idxs = None

		assert th.unique(frag_real_node_node_idxs).shape[0] == frag_real_node_node_idxs.max()+1 == batch_frag_num_nodes
		assert th.unique(frag_real_formula_formula_idxs).shape[0] == frag_real_formula_formula_idxs.max()+1 == batch_frag_num_formulae
		if self.nb_iso:
			assert th.unique(frag_nb_node_node_idxs).shape[0] == frag_nb_node_node_idxs.max()+1 == batch_frag_nb_num_nodes

		if self.bin_output:
			# import pdb; pdb.set_trace()
			spec_bin_mzs, spec_bin_logprobs, spec_bin_batch_idxs = batched_bin_func(
				mzs=spec_mzs,
				ints=spec_logprobs,
				batch_idxs=spec_batch_idxs,
				mz_max=self.mz_max,
				mz_bin_res=self.mz_bin_res,
				agg="lse",
				sparse=True,
				return_mzs=True
			)
			spec_mzs = spec_bin_mzs
			spec_logprobs = spec_bin_logprobs
			spec_batch_idxs = spec_bin_batch_idxs

		# UNIFIED_CEFLOW_CONDITIONAL_MASS_RENORM
		# CEFlow changes the node distribution. Renormalize the
		# rendered support and null-formula branches together so
		# that their conditional probability given not-OOS is 1.
		if (
			getattr(self, "use_ce_flowfrag", False)
			and float(
				getattr(
					self,
					"ce_flowfrag_lambda_max",
					0.0,
				)
			) > 0.0
		):
			conditional_spec_logprobs = (
				spec_logprobs
				- not_oos_logprobs[spec_batch_idxs]
			)
			support_cond_logmass = scatter_logsumexp(
				conditional_spec_logprobs,
				spec_batch_idxs,
				dim_size=int(not_oos_logprobs.numel()),
			)
			inside_cond_logz = th.logaddexp(
				support_cond_logmass,
				frag_null_formula_logprob,
			)
			if not th.all(th.isfinite(inside_cond_logz)):
				raise RuntimeError(
					"CEFlow conditional mass normalization is non-finite"
				)
			spec_logprobs = (
				spec_logprobs
				- inside_cond_logz[spec_batch_idxs]
			)
			frag_null_formula_logprob = (
				frag_null_formula_logprob
				- inside_cond_logz
			)
			if not hasattr(
				self,
				"_ceflow_mass_renorm_debug_printed",
			):
				print(
					"[CEFlowMassRenorm] enabled=True, "
					f"max_abs_logz="
					f"{inside_cond_logz.detach().abs().max().item():.6e}"
				)
				self._ceflow_mass_renorm_debug_printed = True

		prob_sum_check = (
			scatter_logsumexp(spec_logprobs, spec_batch_idxs).exp()
			+ oos_logprobs.exp()
			+ (not_oos_logprobs + frag_null_formula_logprob).exp()
		)
		prob_sum_atol = 3e-2 if getattr(self, "use_ce_flowfrag", False) else 1e-2
		assert th.all(
			th.isclose(
				prob_sum_check,
				th.ones_like(oos_logprobs),
				rtol=0.,
				atol=prob_sum_atol,
			)
		), (
			prob_sum_check,
			scatter_logsumexp(spec_logprobs, spec_batch_idxs).exp(),
			oos_logprobs.exp(),
			(not_oos_logprobs + frag_null_formula_logprob).exp(),
			prob_sum_atol,
		)

		# ===== G1a support: node depth values for CE-pair trajectory loss =====
		# Normalized depth in [0, 1]. It is used only by pl_model.py regularizer.
		# This does not change prediction probabilities.
		frag_rank_node_depth = th.zeros(
			[batch_frag_num_nodes],
			dtype=frag_real_node_logprobs.dtype,
			device=device,
		)

		rank_edge_src = frag_edge_index[0]
		rank_edge_dst = frag_edge_index[1]
		if rank_edge_dst.numel() > 0:
			for _ in range(max(int(self.num_depth), 1)):
				cand_depth = frag_rank_node_depth[rank_edge_src] + 1.0
				updated_depth = scatter_reduce(
					cand_depth,
					rank_edge_dst,
					reduce="max",
					dim_size=batch_frag_num_nodes,
					include_self=True,
				)
				frag_rank_node_depth = th.maximum(
					frag_rank_node_depth,
					updated_depth,
				)

		frag_rank_node_depth = (
			frag_rank_node_depth / max(float(self.num_depth), 1.0)
		).clamp(0.0, 1.0)

		out_d = {
			"pred_mzs": spec_mzs,
			"pred_logprobs": spec_logprobs,
			"pred_batch_idxs": spec_batch_idxs,
			# ===== R55: expose spec-level candidate identity for offline true-hit gate =====
			"pred_spec_formula_global_idxs": spec_formula_global_idxs if spec_formula_global_idxs.shape[0] == spec_mzs.shape[0] else None,
			"pred_spec_formula_logprobs": frag_formula_oos_logprobs[spec_formula_global_idxs] if spec_formula_global_idxs.shape[0] == spec_mzs.shape[0] else None,
                  "pred_spec_formula_comp_feats": frag_formula_comp_feats[spec_formula_global_idxs] if (frag_formula_comp_feats is not None and spec_formula_global_idxs.shape[0] == spec_mzs.shape[0]) else None,
			"pred_spec_base_peak_logprobs": base_peak_logprobs if base_peak_logprobs.shape[0] == spec_mzs.shape[0] else None,
			"pred_spec_peak_logprobs": peak_logprobs if peak_logprobs.shape[0] == spec_mzs.shape[0] else None,
			"pred_spec_peak_channels": frag_formula_peak_channels.long() if (frag_formula_peak_channels is not None and frag_formula_peak_channels.shape[0] == spec_mzs.shape[0]) else th.zeros_like(spec_batch_idxs),
			# ===== R58C: expose local offset group id for supervised offset-channel loss =====
			"pred_spec_offset_group_idxs": r58_offset_group_idxs if (r58_offset_group_idxs is not None and r58_offset_group_idxs.shape[0] == spec_mzs.shape[0]) else None,

			# ===== V8B: original pre-R54 peak-entry metadata =====
			"pred_spec_original_peak_channels": v8b_original_peak_channels if (v8b_original_peak_channels is not None and v8b_original_peak_channels.shape[0] == spec_mzs.shape[0]) else None,
			"pred_spec_original_peak_mzs": v8b_original_peak_mzs if (v8b_original_peak_mzs is not None and v8b_original_peak_mzs.shape[0] == spec_mzs.shape[0]) else None,
			"pred_spec_original_peak_probs": v8b_original_peak_probs if (v8b_original_peak_probs is not None and v8b_original_peak_probs.shape[0] == spec_mzs.shape[0]) else None,

			# ===== R71: expose rendered peak gate tensors for train-time supervision =====
			"pred_rendered_peak_gate_logits": rendered_peak_gate_logits,
			"pred_rendered_peak_gate_delta": rendered_peak_gate_delta,
			"pred_rendered_peak_gate_batch_idxs": spec_batch_idxs if rendered_peak_gate_logits is not None else None,

			"pred_formula_logprobs": frag_real_formula_logprobs,
			"pred_formula_formula_idxs": frag_real_formula_formula_idxs,
			"pred_formula_batch_idxs": frag_formula_batch_idxs,
			"pred_node_logprobs": frag_real_node_logprobs,
			"pred_node_node_idxs": frag_real_node_node_idxs,
			"pred_node_batch_idxs": frag_node_batch_idxs,
			"pred_node_depths": frag_rank_node_depth,
			"pred_node_formula_logprobs": frag_real_node_formula_logprobs,
			"pred_formula_node_logprobs": frag_real_formula_node_logprobs,
			"pred_joint_logprobs": frag_real_joint_logprobs,
			"pred_joint_node_idxs": frag_real_joint_node_idxs,
			"pred_joint_formula_idxs": frag_real_joint_formula_idxs,
			"pred_joint_batch_idxs": frag_real_joint_batch_idxs,
			"pred_joint_h_counts": frag_real_joint_h_counts,
			"pred_joint_h_idxs": frag_real_joint_h_idxs,
			"pred_joint_r12_feats": r56_real_joint_r12_feats,
			"pred_null_formula_logprob": frag_null_formula_logprob,
			"pred_edge_logprobs": frag_edge_logprobs,
			"pred_edge_h_diffs": frag_edge_h_diffs,
			"pred_edge_h_range_masks": frag_edge_h_range_masks,
			"pred_edge_h_logprobs": frag_edge_h_logprobs,
			"pred_edge_batch_idxs": frag_edge_batch_idxs,
			"pred_oos_logprobs": oos_logprobs,
			"pred_h_counts": frag_real_h_counts,
			"pred_h_batch_idxs": frag_real_h_batch_idxs,
			"pred_h_logprobs": frag_real_h_logprobs,
			"pred_nb_node_logprobs": frag_nb_node_logprobs,
			"pred_nb_node_formula_logprobs": frag_nb_node_formula_logprobs,
			"pred_nb_formula_node_logprobs": frag_nb_formula_node_logprobs,
			"pred_nb_node_node_logprobs": frag_nb_node_node_logprobs,
			"pred_nb_node_node_idxs": frag_nb_node_node_idxs,
			"pred_nb_node_batch_idxs": frag_nb_node_batch_idxs,
			"pred_nb_joint_logprobs": frag_nb_joint_logprobs,
			"pred_nb_joint_node_idxs": frag_nb_joint_node_idxs,
			"pred_nb_joint_formula_idxs": frag_nb_joint_formula_idxs,
			"pred_nb_joint_batch_idxs": frag_nb_joint_batch_idxs,
			"pred_nb_node_node_node_idxs": frag_nb_node_node_node_idxs,
			"pred_nb_node_node_batch_idxs": frag_nb_node_node_batch_idxs,
		}

		# ===== R40D: expose internal refiner residual for train-time no-harm loss =====
		# r12_delta_flat is the internal candidate-logit residual added before scatter_logsoftmax.
		if "r12_delta_flat" in locals():
			out_d["pred_refiner_delta"] = r12_delta_flat
			out_d["pred_refiner_delta_batch_idxs"] = frag_joint_batch_idxs
			out_d["pred_refiner_delta_valid_mask"] = frag_joint_mask.bool()

		if self.output_formula_str:

			assert frag_formula_str is not None, "frag_formula_strs must be provided if output_formula_str=True"
			assert spec_prec_type_str is not None, "spec_prec_type_strs must be provided if output_formula_str=True"			
			assert frag_formula_str.shape[0] == frag_real_formula_logprobs.shape[0]
			prec_type_str = spec_prec_type_str
			prec_type_delta_comp = np.array([PREC_TYPE_TO_FORMULA_DIFF[prec_type_str[i]] for i in range(len(prec_type_str))])
			prec_type_delta_comp = prec_type_delta_comp[frag_formula_batch_idxs.cpu().numpy()]
			pred_formula_str = [
				combine_formulae(frag_formula_str[i], prec_type_delta_comp[i]) if frag_formula_str[i] != "" else "" for i in range(len(frag_formula_str))
			]
			pred_formula_str = np.array(pred_formula_str)
			out_d["pred_formula_str"] = pred_formula_str

		for k, v in out_d.items():
			if "logprob" in k and v is not None and v.max() > 0.:
				print(f"> Warning: {k} has value {v.max()} > 0")
			if "batch_idx" in k and v is not None:
				if v.numel() == 0:
					raise ValueError(f"Empty batch index: {k}")
				elif th.unique(v).shape[0] != batch_size:
					raise ValueError(f"Missing items in batch: {k}")

		return out_d

class NeimsModel(nn.Module, CEModel, PrecModel, InstModel):

	def __init__(
		self,
		mol_fingerprint_morgan: bool,
		mol_fingerprint_rdkit: bool,
		mol_fingerprint_maccs: bool,
		mlp_hidden_size: int,
		mlp_dropout: float,
		mlp_num_layers: int,
		mlp_use_residuals: bool,
		mz_max: int,
		mz_bin_res: float,
		ff_prec_mz_offset: int,
		ff_bidirectional: bool,
		ff_output_map_size: int,
		ff_output_activation: str,
		int_embedder: str,
		ce_insert_type: str,
		ce_insert_location: str,
		ce_insert_merge: bool,
		ce_insert_size: int,
  		ce_max: float,
		ce_mean: float,
		ce_std: float,
		prec_insert_location: str,
		prec_insert_size: int,
		prec_types: list[str],
		inst_insert_location: str,
		inst_insert_size: int,
		inst_types: list[str],
		log_min: float):

		# nn.Module init
		super().__init__()
		
		self.mol_fingerprint_morgan = mol_fingerprint_morgan
		self.mol_fingerprint_rdkit = mol_fingerprint_rdkit
		self.mol_fingerprint_maccs = mol_fingerprint_maccs
		
		# input size
		self.mol_fp_dim = get_mol_fp_size(self.mol_fingerprint_morgan, self.mol_fingerprint_rdkit, self.mol_fingerprint_maccs)
		self.mlp_input_dim = self.mol_fp_dim

		# ce stuff
		self._ce_init(
			int_embedder=int_embedder,
			ce_insert_type=ce_insert_type,
			ce_insert_location=ce_insert_location,
			ce_insert_merge=ce_insert_merge,
			ce_insert_size=ce_insert_size,
   			ce_max=ce_max,
			ce_mean=ce_mean,
			ce_std=ce_std)
		self.mlp_input_dim += self.ce_mlp_input_dim

		# prec stuff
		self._prec_init(
			prec_insert_location=prec_insert_location,
			prec_insert_size=prec_insert_size,
			prec_num_types=len(prec_types))
		self.mlp_input_dim += self.prec_mlp_input_dim

		# inst stuff
		self._inst_init(
			inst_insert_location=inst_insert_location,
			inst_insert_size=inst_insert_size,
			inst_num_types=len(inst_types))
		self.mlp_input_dim += self.inst_mlp_input_dim

		self.ffn = SpecFFN(
			input_size=self.mlp_input_dim,
			hidden_size=mlp_hidden_size,
			mz_max=mz_max,
			mz_bin_res=mz_bin_res,
			num_layers=mlp_num_layers,
			dropout=mlp_dropout,
			use_residuals=mlp_use_residuals,
			bidirectional=ff_bidirectional,
			prec_mz_offset=ff_prec_mz_offset,
			output_map_size=ff_output_map_size,
			output_activation=ff_output_activation,
			log_min=log_min
		)

	def _ce_location_check(self):

		assert self.ce_insert_location in ["mlp","none"], f"ce_insert_location={self.ce_insert_location} not supported"

	def _prec_location_check(self):
		
		assert self.prec_insert_location in ["mlp","none"], f"prec_insert_location={self.prec_insert_location} not supported"

	def _inst_location_check(self):

		assert self.inst_insert_location in ["mlp","none"], f"prec_insert_location={self.inst_insert_location} not supported"

	def forward(
		self,
		mol_fingerprint: th.Tensor, 
		spec_prec_mz: th.Tensor,
		spec_ce: th.Tensor = None,
		spec_ce_batch_idxs: th.Tensor = None,
		spec_prec_type: th.Tensor = None,
		spec_inst_type: th.Tensor = None,
		**kwargs
	):

		fh = mol_fingerprint.reshape(-1,self.mol_fp_dim)
		batch_size = fh.shape[0]
		# get ce
		ce = spec_ce
		ce_batch_idxs = spec_ce_batch_idxs
		ce_embed = self.embed_ce(ce, ce_batch_idxs, batch_size)
		prec_embed = self.embed_prec(spec_prec_type)
		inst_embed = self.embed_inst(spec_inst_type)
		if self.ce_insert_location == "mlp":
			fh = th.cat([fh,ce_embed],dim=1)
		if self.prec_insert_location == "mlp":
			fh = th.cat([fh,prec_embed],dim=1)
		if self.inst_insert_location == "mlp":
			fh = th.cat([fh,inst_embed],dim=1)

		# apply ffn
		pred_mzs, pred_logprobs, pred_batch_idxs, pred_specs = self.ffn(fh,spec_prec_mz)
		# ===== R172_SAFE_RICH_FEATURES_V2: robust candidate-level feature export =====
		# This block does not assume specific internal variable names.
		# It only collects tensors whose first dimension matches pred_logprobs.
		r172_peak_rich_feats = None
		try:
			_r172_torch = __import__("torch")
			_r172_n = int(pred_logprobs.shape[0])
			_r172_cols = []
			_r172_names = []
			_r172_max_total_dim = 64
			_r172_max_per_tensor = 8
			_r172_total_dim = 0
			_r172_keywords = (
				"base", "peak", "spec", "formula", "joint", "node",
				"gate", "delta", "logit", "score", "depth", "flow",
				"response", "comp", "frag", "h_count", "channel",
				"prior", "prob", "render", "ce"
			)
			_r172_skip = {
				"pred_mzs", "pred_logprobs", "pred_batch_idxs",
				"true_mzs", "true_logprobs", "true_batch_idxs",
				"unique_id", "mean_loss", "loss"
			}
			for _r172_name, _r172_val in list(locals().items()):
				if _r172_name.startswith("_r172") or _r172_name.startswith("r172_"):
					continue
				if _r172_name in _r172_skip:
					continue
				_r172_lname = str(_r172_name).lower()
				if not any(_kw in _r172_lname for _kw in _r172_keywords):
					continue
				if not isinstance(_r172_val, _r172_torch.Tensor):
					continue
				if _r172_val.numel() == 0 or _r172_val.dim() == 0:
					continue
				if int(_r172_val.shape[0]) != _r172_n:
					continue
				_x = _r172_val.float()
				if _x.dim() == 1:
					_x = _x.reshape(-1, 1)
				else:
					_x = _x.reshape(_r172_n, -1)
				_take = min(int(_x.shape[1]), _r172_max_per_tensor, _r172_max_total_dim - _r172_total_dim)
				if _take <= 0:
					break
				_x = _x[:, :_take]
				_x = _r172_torch.nan_to_num(_x, nan=0.0, posinf=20.0, neginf=-20.0)
				_x = _x.clamp(-20.0, 20.0)
				_r172_cols.append(_x)
				_r172_names.append(str(_r172_name) + ":" + str(_take))
				_r172_total_dim += _take
				if _r172_total_dim >= _r172_max_total_dim:
					break
			if len(_r172_cols) > 0:
				r172_peak_rich_feats = _r172_torch.cat(_r172_cols, dim=1)
			else:
				r172_peak_rich_feats = _r172_torch.zeros((_r172_n, 1), dtype=pred_logprobs.dtype, device=pred_logprobs.device)
				_r172_names = ["fallback_zero"]
			r172_peak_rich_feats = _r172_torch.nan_to_num(
				r172_peak_rich_feats, nan=0.0, posinf=20.0, neginf=-20.0
			).clamp(-20.0, 20.0)
			if not hasattr(self, "_r172_safe_printed"):
				print("[R172_SAFE] exported r172_peak_rich_feats", tuple(r172_peak_rich_feats.shape), "from", _r172_names[:40])
				self._r172_safe_printed = True
		except Exception as _r172_e:
			if not hasattr(self, "_r172_safe_failed_printed"):
				print("[R172_SAFE] failed; using zero feature:", repr(_r172_e))
				self._r172_safe_failed_printed = True
			_r172_torch = __import__("torch")
			r172_peak_rich_feats = _r172_torch.zeros(
				(int(pred_logprobs.shape[0]), 1),
				dtype=pred_logprobs.dtype,
				device=pred_logprobs.device,
			)
		# ===== end R172_SAFE_RICH_FEATURES_V2 =====

		out_d = {
			"pred_mzs": pred_mzs,
			"pred_logprobs": pred_logprobs,
			"pred_batch_idxs": pred_batch_idxs,
			"r172_peak_rich_feats": r172_peak_rich_feats,
			"pred_specs": pred_specs
		}
		return out_d
	
class PrecursorModel(nn.Module):

	def __init__(self):

		super().__init__()
		self.dummy_params = nn.Parameter(th.zeros((1,), dtype=th.float32))

	def forward(
		self, 
		spec_prec_mz: th.Tensor,
		**kwargs):

		pred_mzs = spec_prec_mz
		pred_logprobs = 0.*self.dummy_params + th.zeros_like(pred_mzs)
		pred_batch_idxs = th.arange(pred_mzs.shape[0],device=pred_mzs.device)

		out_d = {
			"pred_mzs": pred_mzs,
			"pred_logprobs": pred_logprobs,
			"pred_batch_idxs": pred_batch_idxs
		}
		return out_d

class GNNModel(nn.Module, CEModel, PrecModel, InstModel):

	def __init__(
		self,
		mol_node_feats: list[str],
		mol_edge_feats: list[str],
		mol_pe_embed_k: int,
		mol_hidden_size: int,
		mol_num_layers: int,
		mol_gnn_type: str,
		mol_normalization: str,
		mol_dropout: float,
		mol_pool_type: str,
		mlp_hidden_size: int,
		mlp_dropout: float,
		mlp_num_layers: int,
		mlp_use_residuals: bool,
		mz_max: int,
		mz_bin_res: float,
		ff_prec_mz_offset: int,
		ff_bidirectional: bool,
		ff_output_map_size: int,
		ff_output_activation: str,
		int_embedder: str,
		ce_insert_type: str,
		ce_insert_location: str,
		ce_insert_merge: bool,
		ce_insert_size: int,
		ce_max: float,
		ce_mean: float,
		ce_std: float,
		prec_insert_location: str,
		prec_insert_size: int,
		prec_types: list[str],
		inst_insert_location: str,
		inst_insert_size: int,
		inst_types: list[str],
		log_min: float
	):
		# nn.Module init
		super().__init__()
		# collision energy
		self._ce_init(
			int_embedder=int_embedder,
			ce_insert_location=ce_insert_location,
			ce_insert_type=ce_insert_type,
			ce_insert_merge=ce_insert_merge,
			ce_insert_size=ce_insert_size,
			ce_max=ce_max,
			ce_mean=ce_mean,
			ce_std=ce_std
		)
		# precursor
		self._prec_init(
			prec_insert_location=prec_insert_location,
			prec_insert_size=prec_insert_size,
			prec_num_types=len(prec_types)
		)
		# instrument
		self._inst_init(
			inst_insert_location=inst_insert_location,
			inst_insert_size=inst_insert_size,
			inst_num_types=len(inst_types))

		# calculate node/edge feats sizes
		self.mol_node_feats = mol_node_feats
		self.mol_edge_feats = mol_edge_feats
		self.mol_pe_embed_k = mol_pe_embed_k
		self._compute_mol_feats_sizes()

		# setup mol gnn
		self.mol_node_feats_size += self.ce_mol_input_dim + self.prec_mol_input_dim + self.inst_mol_input_dim
		mol_kwargs = {
			"node_feats_size": self.mol_node_feats_size,
			"edge_feats_size": self.mol_edge_feats_size,
			"hidden_size": mol_hidden_size,
			"num_layers": mol_num_layers,
			"gnn_type": mol_gnn_type,
			"dropout": mol_dropout,
			"normalization": mol_normalization,
		}
		# Mol GNN
		self.mol_embedder = GNN(**mol_kwargs)
		self.mol_pool_type = mol_pool_type
		self.mol_pool = build_pool_module(mol_pool_type,mol_hidden_size)

		# MLP input = GNN output
		self.mlp_input_dim = mol_hidden_size
		# metadata
		self.mlp_input_dim += self.ce_mlp_input_dim + self.prec_mlp_input_dim + self.inst_mlp_input_dim

		self.ffn = SpecFFN(
			input_size=self.mlp_input_dim,
			hidden_size=mlp_hidden_size,
			mz_max=mz_max,
			mz_bin_res=mz_bin_res,
			num_layers=mlp_num_layers,
			dropout=mlp_dropout,
			use_residuals=mlp_use_residuals,
			bidirectional=ff_bidirectional,
			prec_mz_offset=ff_prec_mz_offset,
			output_map_size=ff_output_map_size,
			output_activation=ff_output_activation,
			log_min=log_min
		)

	def forward(
		self, 
		mol_pyg: pyg.data.Data,
		spec_prec_mz: th.Tensor,
		spec_nce: th.Tensor = None,
		spec_nce_batch_idxs: th.Tensor = None,
		spec_prec_type: th.Tensor = None,
		spec_inst_type: th.Tensor = None,
		**kwargs
	):
		# mol features
		# mol_x: mol level node feature matrix
		# mol_edge_index: mol graph connectivity in COO format with shape [2, num_edges]
		# edge_attr: mol graph edge feature matrix with shape [num_edges, num_edge_features]
		# batch: sample idx repsect to current batch
		mol_x, mol_edge_index, mol_edge_attr, mol_batch = mol_pyg.x, mol_pyg.edge_index, mol_pyg.edge_attr, mol_pyg.batch

		# int_dtype = mol_edge_index.dtype
		batch_size = mol_batch[-1]+1

		# metadata embedders
		# get ce value
		ce = spec_nce
		ce_batch_idxs = spec_nce_batch_idxs
		ce_embed = self.embed_ce(ce, ce_batch_idxs, batch_size)
		# get prec value
		prec_embed = self.embed_prec(spec_prec_type)
		# get inst value
		inst_embed = self.embed_inst(spec_inst_type)		

		# metadata embeddings at the node feature level
		if self.ce_insert_location == "mol":
			mol_ce_embed = th.repeat_interleave(ce_embed,th.unique(mol_batch,return_counts=True)[1],dim=0)
			mol_x = th.cat([mol_x,mol_ce_embed],dim=1)
		if self.prec_insert_location == "mol":
			mol_prec_embed = th.repeat_interleave(prec_embed,th.unique(mol_batch,return_counts=True)[1],dim=0)
			mol_x = th.cat([mol_x,mol_prec_embed],dim=1)
		if self.inst_insert_location == "mol":
			mol_inst_embed = th.repeat_interleave(inst_embed,th.unique(mol_batch,return_counts=True)[1],dim=0)
			mol_x = th.cat([mol_x,mol_inst_embed],dim=1)
		
		# get per-atom embeddings
		mol_embed_gnn = self.mol_embedder(
			mol_x,
			mol_batch,
			mol_edge_index,
			mol_edge_attr
		)
		mol_embed_gnn_pool = self.mol_pool(mol_embed_gnn,mol_batch)
		ffn_input = mol_embed_gnn_pool

		if self.ce_insert_location == "mlp":
			ffn_input = th.cat([ffn_input,ce_embed],dim=1)
		if self.prec_insert_location == "mlp":
			ffn_input = th.cat([ffn_input,prec_embed],dim=1)
		if self.inst_insert_location == "mlp":
			ffn_input = th.cat([ffn_input,inst_embed],dim=1)

		# apply ffn
		pred_mzs, pred_logprobs, pred_batch_idxs, pred_specs = self.ffn(ffn_input,spec_prec_mz)
		out_d = {
			"pred_mzs": pred_mzs,
			"pred_logprobs": pred_logprobs,
			"pred_batch_idxs": pred_batch_idxs,
			"pred_specs": pred_specs
		}
		return out_d
		
	def _compute_mol_feats_sizes(self):
		""" method compute mol feature size
			these features don't rely on any model parameters
		"""
		self.mol_node_feats_size, self.mol_edge_feats_size = get_mol_feats_sizes(
			self.mol_node_feats, 
			self.mol_edge_feats, 
			self.mol_pe_embed_k
		)

	def _ce_location_check(self):

		assert self.ce_insert_location in ["mlp","mol","none"], f"ce_insert_location={self.ce_insert_location} not supported"

	def _prec_location_check(self):
		
		assert self.prec_insert_location in ["mlp","mol","none"], f"prec_insert_location={self.prec_insert_location} not supported"

	def _inst_location_check(self):

		assert self.inst_insert_location in ["mlp","mol","none"], f"prec_insert_location={self.inst_insert_location} not supported"
