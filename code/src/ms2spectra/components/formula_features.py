import torch as th
import torch.nn as nn
import numpy as np

DEFAULT_MAX_COUNT_INT = 255
DEFAULT_NUM_EXTRA_EMBEDDINGS = 1

class IntFeaturizer(nn.Module):
	"""
	Base class for mapping integers to a vector representation (primarily to be used as a "richer" embedding for NNs
	processing integers).

	Subclasses should define `self.int_to_feat_matrix`, a matrix where each row is the vector representation for that
	integer, i.e. to get a vector representation for `5`, one could call `self.int_to_feat_matrix[5]`.

	Note that this class takes care of creating a fixed number (`self.NUM_EXTRA_EMBEDDINGS` to be precise) of extra
	"learned" embeddings these will be concatenated after the integer embeddings in the forward pass,
	be learned, and be used for extra  non-integer tokens such as the "to be confirmed token" (i.e., pad) token.
	They are indexed starting from `self.max_count_int`.
	"""

	# MAX_COUNT_INT = 255  # the maximum number of integers that we are going to see as a "count", i.e. 0 to MAX_COUNT_INT-1
	# NUM_EXTRA_EMBEDDINGS = 1  # Number of extra embeddings to learn -- one for the "to be confirmed" embedding.

	def __init__(self, embedding_dim, max_count_int=DEFAULT_MAX_COUNT_INT, num_extra_embeddings=DEFAULT_NUM_EXTRA_EMBEDDINGS):
		super().__init__()
		self.max_count_int = max_count_int
		self.num_extra_embeddings = num_extra_embeddings
		weights = th.zeros(self.num_extra_embeddings, embedding_dim)
		self._extra_embeddings = nn.Parameter(weights, requires_grad=True)
		nn.init.normal_(self._extra_embeddings, 0.0, 1.0)
		self.embedding_dim = embedding_dim

	def forward(self, tensor):
		"""
		Convert the integer `tensor` into its new representation -- note that it gets stacked along final dimension.
		"""
		orig_shape = tensor.shape
		# out_tensor = th.empty(
		# 	(*orig_shape, self.embedding_dim), device=tensor.device
		# )
		extra_embed_mask = (tensor >= self.max_count_int).float()

		tensor = tensor.long()
		norm_embeds = self.int_to_feat_matrix[tensor]
		extra_embeds = self._extra_embeddings[th.maximum(tensor,self.max_count_int*th.ones_like(tensor))-self.max_count_int]
		out_tensor = (1.-extra_embed_mask).unsqueeze(-1)*norm_embeds + extra_embed_mask.unsqueeze(-1)*extra_embeds

		temp_out = out_tensor.reshape(*orig_shape[:-1], -1)
		return temp_out

	@property
	def num_dim(self):
		return self.int_to_feat_matrix.shape[1]

	# @property
	# def full_dim(self):
	#	 return self.num_dim * common.NORM_VEC.shape[0]


class FourierFeaturizer(IntFeaturizer):
	"""
	Inspired by:
	Tancik, M., Srinivasan, P.P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R.,
	Barron, J.T. and Ng, R. (2020) ‘Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional
	 Domains’, arXiv [cs.CV]. Available at: http://arxiv.org/abs/2006.10739.

	Some notes:
	* we'll put the frequencies at powers of 1/2 rather than random Gaussian samples; this means it will match the
		Binarizer quite closely but be a bit smoother.
	"""

	def __init__(self, max_count_int=DEFAULT_MAX_COUNT_INT, num_extra_embeddings=DEFAULT_NUM_EXTRA_EMBEDDINGS):

		num_freqs = int(np.ceil(np.log2(max_count_int))) + 2
		# ^ need at least this many to ensure that the whole input range can be represented on the half circle.

		freqs = 0.5 ** th.arange(num_freqs, dtype=th.float32)
		freqs_time_2pi = 2 * np.pi * freqs

		super().__init__(
			embedding_dim=2 * freqs_time_2pi.shape[0],
			max_count_int=max_count_int,
			num_extra_embeddings=num_extra_embeddings,
		)  # 2 for cosine and sine

		# we will define the features at this frequency up front (as we only will ever see a fixed number of counts):
		combo_of_sinusoid_args = (
			th.arange(self.max_count_int, dtype=th.float32)[:, None]
			* freqs_time_2pi[None, :]
		)
		all_features = th.cat(
			[th.cos(combo_of_sinusoid_args), th.sin(combo_of_sinusoid_args)],
			dim=1,
		)

		# ^ shape:  MAX_COUNT_INT x 2 * num_freqs
		self.int_to_feat_matrix = nn.Parameter(all_features.float())
		self.int_to_feat_matrix.requires_grad = False


class FourierFeaturizerSines(IntFeaturizer):
	"""
	Like other fourier feats but sines only

	Inspired by:
	Tancik, M., Srinivasan, P.P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R.,
	Barron, J.T. and Ng, R. (2020) ‘Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional
	 Domains’, arXiv [cs.CV]. Available at: http://arxiv.org/abs/2006.10739.

	Some notes:
	* we'll put the frequencies at powers of 1/2 rather than random Gaussian samples; this means it will match the
		Binarizer quite closely but be a bit smoother.
	"""

	def __init__(self, max_count_int=DEFAULT_MAX_COUNT_INT, num_extra_embeddings=DEFAULT_NUM_EXTRA_EMBEDDINGS):

		num_freqs = int(np.ceil(np.log2(max_count_int))) + 2
		# ^ need at least this many to ensure that the whole input range can be represented on the half circle.

		freqs = (0.5 ** th.arange(num_freqs, dtype=th.float32))[2:]
		freqs_time_2pi = 2 * np.pi * freqs

		super().__init__(
			embedding_dim=freqs_time_2pi.shape[0],
			max_count_int=max_count_int,
			num_extra_embeddings=num_extra_embeddings	
		)

		# we will define the features at this frequency up front (as we only will ever see a fixed number of counts):
		combo_of_sinusoid_args = (
			th.arange(self.max_count_int, dtype=th.float32)[:, None]
			* freqs_time_2pi[None, :]
		)
		# ^ shape:  MAX_COUNT_INT x 2 * num_freqs
		self.int_to_feat_matrix = nn.Parameter(
			th.sin(combo_of_sinusoid_args).float()
		)
		self.int_to_feat_matrix.requires_grad = False


class FourierFeaturizerAbsoluteSines(IntFeaturizer):
	"""
	Like other fourier feats but sines only and absoluted.

	Inspired by:
	Tancik, M., Srinivasan, P.P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R.,
	Barron, J.T. and Ng, R. (2020) ‘Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional
	 Domains’, arXiv [cs.CV]. Available at: http://arxiv.org/abs/2006.10739.

	Some notes:
	* we'll put the frequencies at powers of 1/2 rather than random Gaussian samples; this means it will match the
		Binarizer quite closely but be a bit smoother.
	"""

	def __init__(self, max_count_int=DEFAULT_MAX_COUNT_INT, num_extra_embeddings=DEFAULT_NUM_EXTRA_EMBEDDINGS):

		num_freqs = int(np.ceil(np.log2(max_count_int))) + 2

		freqs = (0.5 ** th.arange(num_freqs, dtype=th.float32))[2:]
		freqs_time_2pi = 2 * np.pi * freqs

		super().__init__(
			embedding_dim=freqs_time_2pi.shape[0],
			max_count_int=max_count_int,
			num_extra_embeddings=num_extra_embeddings
		)

		# we will define the features at this frequency up front (as we only will ever see a fixed number of counts):
		combo_of_sinusoid_args = (
			th.arange(self.max_count_int, dtype=th.float32)[:, None]
			* freqs_time_2pi[None, :]
		)
		# ^ shape:  MAX_COUNT_INT x 2 * num_freqs
		self.int_to_feat_matrix = nn.Parameter(
			th.abs(th.sin(combo_of_sinusoid_args)).float()
		)
		self.int_to_feat_matrix.requires_grad = False


class RBFFeaturizer(IntFeaturizer):
	"""
	A featurizer that puts radial basis functions evenly between 0 and max_count-1. These will have a width of
	(max_count-1) / (num_funcs) to decay to about 0.6 of its original height at reaching the next func.

	"""

	def __init__(self, num_funcs=32, max_count_int=DEFAULT_MAX_COUNT_INT, num_extra_embeddings=DEFAULT_NUM_EXTRA_EMBEDDINGS):
		"""
		:param num_funcs: number of radial basis functions to use: their width will automatically be chosen -- see class
							docstring.
		"""
		super().__init__(
			embedding_dim=num_funcs,
			max_count_int=max_count_int,
			num_extra_embeddings=num_extra_embeddings
		)
		width = (self.max_count_int - 1) / num_funcs
		centers = th.linspace(0, self.max_count_int- 1, num_funcs)

		pre_exponential_terms = (
			-0.5
			* ((th.arange(self.max_count_int)[:, None] - centers[None, :]) / width)
			** 2
		)
		# ^ shape: MAX_COUNT_INT x num_funcs
		feats = th.exp(pre_exponential_terms)

		self.int_to_feat_matrix = nn.Parameter(feats.float())
		self.int_to_feat_matrix.requires_grad = False


class OneHotFeaturizer(IntFeaturizer):
	"""
	A featurizer that turns integers into their one hot encoding.

	Represents:
	 - 0 as 1000000000...
	 - 1 as 0100000000...
	 - 2 as 0010000000...
	 and so on.
	"""

	def __init__(self, max_count_int=DEFAULT_MAX_COUNT_INT, num_extra_embeddings=DEFAULT_NUM_EXTRA_EMBEDDINGS):
		super().__init__(
			embedding_dim=max_count_int,
			max_count_int=max_count_int,
			num_extra_embeddings=num_extra_embeddings
		)
		feats = th.eye(self.max_count_int)
		self.int_to_feat_matrix = nn.Parameter(feats.float())
		self.int_to_feat_matrix.requires_grad = False


class LearnedFeaturizer(IntFeaturizer):
	"""
	Learns the features for the different integers.

	Pretty much `nn.Embedding` but we get to use the forward of the superclass which behaves a bit differently.
	"""

	def __init__(self, feature_dim=32, max_count_int=DEFAULT_MAX_COUNT_INT, num_extra_embeddings=DEFAULT_NUM_EXTRA_EMBEDDINGS):
		super().__init__(
			embedding_dim=feature_dim,
			max_count_int=max_count_int,
			num_extra_embeddings=num_extra_embeddings
		)
		weights = th.zeros(self.max_count_int, feature_dim)
		self.int_to_feat_matrix = nn.Parameter(weights, requires_grad=True)
		nn.init.normal_(self.int_to_feat_matrix, 0.0, 1.0)


# class FloatFeaturizer(IntFeaturizer):
#	 """
#	 Norms the features
#	 """

#	 def __init__(self):
#		 # Norm vec
#		 # Placeholder..
#		 super().__init__(embedding_dim=1)
#		 self.norm_vec = th.from_numpy(common.NORM_VEC).float()
#		 self.norm_vec = nn.Parameter(self.norm_vec)
#		 self.norm_vec.requires_grad = False

#	 def forward(self, tensor):
#		 """
#		 Convert the integer `tensor` into its new representation -- note that it gets stacked along final dimension.
#		 """
#		 tens_shape = tensor.shape
#		 out_shape = [1] * (len(tens_shape) - 1) + [-1]
#		 return tensor / self.norm_vec.reshape(*out_shape)

#	 @property
#	 def num_dim(self):
#		 return 1


def build_formula_embedder(embedder,**kwargs):
	if embedder == "fourier":
		embedder = FourierFeaturizer(**kwargs)
	elif embedder == "rbf":
		embedder = RBFFeaturizer(**kwargs)
	elif embedder == "one-hot":
		embedder = OneHotFeaturizer(**kwargs)
	elif embedder == "learnt":
		embedder = LearnedFeaturizer(**kwargs)
	elif embedder == "float":
		raise NotImplementedError
		# embedder = FloatFeaturizer()
	elif embedder == "fourier-sines":
		embedder = FourierFeaturizerSines(**kwargs)
	elif embedder == "abs-sines":
		embedder = FourierFeaturizerAbsoluteSines(**kwargs)
	else:
		raise NotImplementedError
	return embedder


"""
这个文件是**“把整数特征变成神经网络友好的连续向量”的统一编码系统**。
在 FragnNet / Iceberg / MassFormer 这种模型里，它通常用来编码：

> 碎片深度、计数、同位素索引、节点层级、CE bucket、峰排序 index、路径长度、图 hop 数……

我给你**从设计哲学 → 数学直觉 → 每个类干嘛 → forward 怎么拼 → 工程坑点**全拆开讲。

---

# 一、核心设计思想

## 问题

神经网络不擅长直接吃“整数”：

```text
3 和 4
→ 网络不知道它们“接近”
```

所以要做：

> **整数 → 连续向量映射（Embedding / Feature Map）**

---

## 这个系统的统一接口

所有 featurizer 都继承：

```python
class IntFeaturizer(nn.Module)
```

你只要给：

```python
tensor = [0, 1, 2, 5, 200]
```

它给你：

```text
[N, embedding_dim] 的连续向量
```

---

# 二、IntFeaturizer：系统“骨架”

## 两类 embedding

### 1️⃣ 固定 embedding（0 ~ max_count_int-1）

来自：

```python
self.int_to_feat_matrix
```

### 2️⃣ 额外 embedding（>= max_count_int）

用于：

* PAD
* UNK
* MASK
* “未确认 token”

来自：

```python
self._extra_embeddings
```

---

## forward 关键逻辑

### 输入

```python
tensor.shape = [..., K]
```

---

### mask 判断

```python
extra_embed_mask = (tensor >= max_count_int)
```

---

### 查表

```python
norm_embeds = self.int_to_feat_matrix[tensor]
extra_embeds = self._extra_embeddings[tensor - max_count_int]
```

---

### 混合

```python
out = (1-mask)*norm + mask*extra
```

---

### reshape

```python
[..., K, embedding_dim] → [..., K*embedding_dim]
```

这一步很关键：

> 它不是给你多一维，而是**把 embedding “摊平拼到特征维”**

---

## 为什么这么做？

方便直接 concat 到 MLP / Transformer 输入：

```text
原特征: [..., F]
整数特征: [..., K] → [..., K*D]
拼起来: [..., F + K*D]
```

---

# 三、FourierFeaturizer（论文级编码）

## 数学直觉

来自：

> Tancik et al. 2020 — Fourier Features

### 核心思想

把整数 x 映射为：

```math
[cos(2πf₁x), sin(2πf₁x), ..., cos(2πfₙx), sin(2πfₙx)]
```

这样：

* 小频率 → 捕捉“平滑变化”
* 大频率 → 捕捉“离散跳变”

---

## 这里的频率设计

```python
freqs = 0.5 ** arange(num_freqs)
```

是：

```text
1, 1/2, 1/4, 1/8, ...
```

这和**二进制位结构对齐**
→ 非常适合 encoding “计数 / 层级 / 深度”

---

## 特点

| 优点     | 缺点   |
| ------ | ---- |
| 连续、可区分 | 维度较大 |
| 可外推    | 不可学习 |

---

# 四、FourierFeaturizerSines

只用：

```math
sin(2πfx)
```

### 为什么？

* 降维
* 更平滑
* 减少模型容量

适合：

> 深度、层级、hop 数

---

# 五、AbsoluteSines

```math
|sin(2πfx)|
```

### 意义

消掉符号
只保留：

> 距离周期中心的“幅度”

适合：

* 层级强度
* 相对深度

---

# 六、RBFFeaturizer（工程派最爱）

## 思想

不是周期函数，而是：

> “x 离哪个中心最近？”

用一堆高斯核：

```math
exp(-(x-cᵢ)² / σ²)
```

---

## 直觉

```text
x=7
→ 在 center=6 上高
→ 在 center=8 上高
→ 其他低
```

相当于：

> 连续 soft one-hot

---

## 特点

| 优点    | 缺点   |
| ----- | ---- |
| 平滑    | 不周期  |
| 物理直觉强 | 不能外推 |

---

# 七、OneHotFeaturizer

## 最暴力

```text
3 → 0001000
```

## 适合

* max_count 很小（<64）
* 离散类别

## 不适合

* max_count = 255
  → 255 维直接爆参数

---

# 八、LearnedFeaturizer

等价于：

```python
nn.Embedding(max_count, feature_dim)
```

但走统一 forward

## 特点

| 优点    | 缺点     |
| ----- | ------ |
| 最强表达力 | 需要大量数据 |
| 自适应   | 不可外推   |

---

# 九、extra embedding 机制（非常工程化）

## 设计目的

处理：

* PAD = 255
* UNK = 256
* MASK = 257

## 用法

如果：

```python
num_extra_embeddings = 2
```

你就可以用：

```text
255 → extra[0]
256 → extra[1]
```

模型会学：

> 什么是 padding / missing / masked

---

# 十、维度变化示意

## 输入

```python
tensor = [3, 5, 7]
shape = [3]
```

## FourierFeaturizer

假设：

```text
num_freqs = 8
embedding_dim = 16
```

## 输出

```text
shape = [3*16] = [48]
```

---

# 十一、在 FragnNet 里通常用在哪？

常见字段：

| 字段          | 编码方式          |
| ----------- | ------------- |
| frag_depth  | Fourier / RBF |
| isotope_idx | OneHot        |
| path_len    | Fourier       |
| node_hop    | RBF           |
| peak_rank   | Fourier       |
| count       | Learned       |

---

# 十二、选型指南（工程实战）

## 小数据

> 用 Fourier 或 RBF
> 不容易过拟合

## 大数据

> Learned
> 表达力强

## 层级 / 深度

> Fourier-sines

## 分类 token

> One-hot

---

# 十三、forward 里隐藏的大坑 ⚠️

### 1️⃣ tensor 必须小于：

```python
max_count_int + num_extra_embeddings
```

否则：

```text
索引越界，直接炸
```

---

### 2️⃣ reshape 会“摊平特征”

如果你以为输出是：

```text
[..., K, D]
```

其实是：

```text
[..., K*D]
```

这在接 Transformer 时非常关键

---

# 十四、论文级总结语

你可以这样写方法部分：

> *We encode discrete integer-valued features using a unified featurization framework, including fixed Fourier feature mappings, radial basis functions, and learned embeddings. Each integer is mapped to a continuous vector and concatenated into the model input, with dedicated learned embeddings for special tokens such as padding and unknown values.*

---

# 十五、如果你告诉我

> **这个 embedder 在你模型里具体喂给了哪一层**

我可以帮你：

* 画出**完整张量维度流**
* 帮你算**显存 / 参数量**
* 推荐**最优 embedding 维度**

这一步对调 FragnNet / Iceberg 的性能影响非常大 🔥

"""