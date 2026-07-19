# Data Preprocessing

## Step 0: Exporting the NIST 2020 MS/MS library

All of the data used in the paper is based on the NIST 2020 MS/MS library. Since this library is not publicly available, we are unfortunately unable to provide access to it.

The raw NIST data can be exported using the instructions [here](https://github.com/Roestlab/massformer). It can then be processed using the scripts in the [preroc_scripts directory](preproc_scripts/) (see [this README](preproc_scripts/README.md) for more details).

## Step 1: Processing raw data into CSV

This assumes that the NIST20 data is in a folder `data/raw/nist_20`, containing two files: `hr_nist_msms.MSP` (the MSP file with the spectrum information) and `hr_nist_msms.MOL` (the MOL file with the molecule structures).

This will create a new directory `data/df`

```bash
python preproc_scripts/01_prepare_df.py --msp_file nist_20/hr_nist_msms.MSP --mol_dir nist_20/hr_nist_msms.MOL --input_format msp+mol --output_format csv --output_dp data/df --output_name nist20_hr
```

## Step 2: Processing CSV into Pickle Files

This step converts the data into binary format (pickle) for training and evaluation.

`spec_df.pkl` contains spectrum information.
`mol_df.pkl` contains molecule information.
`ann_df.pkl` contains annotation information.

```bash
python preproc_scripts/02_prepare_proc.py --df_dp data/df --dsets nist20_hr --proc_dp data/proc/nist20
```

## Step 3: Generating Fragmentation DAGs

This step generates fragmentation DAGs that are used for the FraGNNet models. Since fragmentation can be slow, we recommend using a compute with many cores (by default, the script will use all available cores). 

Frag configuration: depth 3 (used for FraGNNet-D3), isomorphism check with no bond information (nb, 3 WL iterations)

```bash
python preproc_scripts/03_prepare_dag_feats.py --max_depth 3 --frag_dp data/frag/nist20_d3 --proc_dp data/proc/nist20 --prec_types "[M+H]+" --inst_types FT --frag_modes HCD --ion_modes P --wandb_mode disabled
```

Frag configuration: depth 4 (used for FraGNNet-D4), isomorphism check with no bond information (nb, 3 WL iterations)

```bash
python preproc_scripts/03_prepare_dag_feats.py --max_depth 4 --frag_dp data/frag/nist20_d4 --proc_dp data/proc/nist20 --prec_types "[M+H]+" --inst_types FT --frag_modes HCD --ion_modes P --wandb_mode disabled
```

## Step 4: Preparing Splits

This step splits the data into training, validation, and test sets.

Inchikey split:

```bash
python preproc_scripts/04_prepare_split.py --split_key inchikey_s --primary_dsets nist20_hr --prec_types "[M+H]+" --inst_types FT --frag_modes HCD --ion_modes P --proc_dp data/proc/nist20 --frag_dp data/frag/nist20_d4 --split_dp data/split/nist20_inchikey
```

Scaffold split:

```bash
python preproc_scripts/04_prepare_split.py --split_key scaffold --primary_dsets nist20_hr --prec_types "[M+H]+" --inst_types FT --frag_modes HCD --ion_modes P --proc_dp data/proc/nist20 --frag_dp data/frag/nist20_d4 --split_dp data/split/nist20_scaffold
```

## Step 5: Preparing Optimal MAGMa DAGs (ICEBERG)

This step generates optimal MAGMa DAGs for training/evaluating the ICEBERG generator model optimal ICEBERG intensity model.

```bash
python preproc_scripts/05_prepare_magma_feats.py --magma_dp data/magma/gen/nist20mona23v3 --proc_dp data/proc/nist20mona23v3 --dsets nist20_hr mona23
```

## Step 6: Preparing Approximate MAGMa DAGs (ICEBERG)

This step generates approximate MAGMa DAGs using a trained ICEBERG generator model, for training and evaluating the ICEBERG intensity model.

For each seed `${SEED}` (see [config README](../config/README.md)) repeat the following:

```bash
python preproc_scripts/06_predict_magma_dags.py --magma_dp data/magma/inten/nist20_inchikey_s$SEED --proc_dp data/proc/nist20 --gen_ckpt_fp data/magma/ckpt/nist20_inchikey_s$SEED.ckpt --dsets nist20_hr
```

Don't forget to symlink the formula directory:

```bash
ln -s $HOME/frag-gnn/data/magma/gen/nist20/magma_formula $HOME/frag-gnn/data/magma/inten/nist20_inchikey_s$SEED/magma_formula
```



# 数据预处理
## 步骤 0：导出 NIST 2020 串联质谱库
本文所用的全部数据均基于 NIST 2020 串联质谱库。由于该数据库未公开，我们遗憾地无法提供访问权限。

原始 NIST 数据可按照[此处](https://github.com/Roestlab/massformer)的说明进行导出。导出后的数据可通过`preproc_scripts`目录下的脚本进行处理（详细信息请参见该目录下的[自述文件](preproc_scripts/README.md)）。

## 步骤 1：将原始数据处理为 CSV 格式
此步骤假设 NIST20 数据存放于`data/raw/nist_20`文件夹中，该文件夹包含两个文件：`hr_nist_msms.MSP`（存储质谱信息的 MSP 文件）和`hr_nist_msms.MOL`（存储分子结构的 MOL 文件）。

执行以下命令会新建一个`data/df`目录：
```bash
python preproc_scripts/01_prepare_df.py --msp_file nist_20/hr_nist_msms.MSP --mol_dir nist_20/hr_nist_msms.MOL --input_format msp+mol --output_format csv --output_dp data/df --output_name nist20_hr
```

## 步骤 2：将 CSV 格式数据转换为 Pickle 文件
此步骤将数据转换为二进制格式（Pickle 格式），用于后续的模型训练与验证。

生成的文件说明如下：
- `spec_df.pkl`：存储质谱信息
- `mol_df.pkl`：存储分子信息
- `ann_df.pkl`：存储注释信息

执行命令：
```bash
python preproc_scripts/02_prepare_proc.py --df_dp data/df --dsets nist20_hr --proc_dp data/proc/nist20
```

## 步骤 3：生成碎裂有向无环图
此步骤生成用于 FraGNNet 模型的**碎裂有向无环图（Fragmentation DAGs）**。由于碎裂过程耗时较长，建议使用多核计算设备（脚本默认会调用所有可用核心）。

### 碎裂配置 1
深度 3（用于 FraGNNet-D3 模型），采用不含键信息的同构校验（参数`nb`，3 次 WL 迭代）
```bash
python preproc_scripts/03_prepare_dag_feats.py --max_depth 3 --frag_dp data/frag/nist20_d3 --proc_dp data/proc/nist20 --prec_types "[M+H]+" --inst_types FT --frag_modes HCD --ion_modes P --wandb_mode disabled
```

### 碎裂配置 2
深度 4（用于 FraGNNet-D4 模型），采用不含键信息的同构校验（参数`nb`，3 次 WL 迭代）
```bash
python preproc_scripts/03_prepare_dag_feats.py --max_depth 4 --frag_dp data/frag/nist20_d4 --proc_dp data/proc/nist20 --prec_types "[M+H]+" --inst_types FT --frag_modes HCD --ion_modes P --wandb_mode disabled
```

## 步骤 4：划分数据集
此步骤将数据集划分为训练集、验证集和测试集。

### 基于 Inchikey 的划分
```bash
python preproc_scripts/04_prepare_split.py --split_key inchikey_s --primary_dsets nist20_hr --prec_types "[M+H]+" --inst_types FT --frag_modes HCD --ion_modes P --proc_dp data/proc/nist20 --frag_dp data/frag/nist20_d4 --split_dp data/split/nist20_inchikey
```

### 基于分子骨架的划分
```bash
python preproc_scripts/04_prepare_split.py --split_key scaffold --primary_dsets nist20_hr --prec_types "[M+H]+" --inst_types FT --frag_modes HCD --ion_modes P --proc_dp data/proc/nist20 --frag_dp data/frag/nist20_d4 --split_dp data/split/nist20_scaffold
```

## 步骤 5：生成优化的 MAGMa 有向无环图（用于 ICEBERG 模型）
此步骤生成优化的 MAGMa 有向无环图，用于 ICEBERG 生成器模型和 ICEBERG 强度模型的训练与验证。
```bash
python preproc_scripts/05_prepare_magma_feats.py --magma_dp data/magma/gen/nist20mona23v3 --proc_dp data/proc/nist20mona23v3 --dsets nist20_hr mona23
```

## 步骤 6：生成近似 MAGMa 有向无环图（用于 ICEBERG 模型）
此步骤利用训练好的 ICEBERG 生成器模型，生成近似的 MAGMa 有向无环图，用于 ICEBERG 强度模型的训练与验证。

对于每个随机种子`${SEED}`（参见[配置自述文件](../config/README.md)），重复执行以下命令：
```bash
python preproc_scripts/06_predict_magma_dags.py --magma_dp data/magma/inten/nist20_inchikey_s$SEED --proc_dp data/proc/nist20 --gen_ckpt_fp data/magma/ckpt/nist20_inchikey_s$SEED.ckpt --dsets nist20_hr
```

**注意**：不要忘记创建分子式目录的符号链接：
```bash
ln -s $HOME/frag-gnn/data/magma/gen/nist20/magma_formula $HOME/frag-gnn/data/magma/inten/nist20_inchikey_s$SEED/magma_formula
```

---

### 关键专业术语说明
1.  **MS/MS library**：串联质谱库，是存储化合物串联质谱数据的数据库，用于化合物定性分析。
2.  **Fragmentation DAG**：碎裂有向无环图，用于表征分子碎裂过程的拓扑结构，是 FraGNNet 模型的核心输入特征。
3.  **WL iterations**：Weisfeiler-Lehman 迭代，一种图同构判定的迭代算法，用于图结构的特征提取。
4.  **Pickle**：Python 特有的对象序列化格式，能高效存储和读取复杂数据结构。

我可以帮你整理这份预处理流程中的**关键参数释义表**，方便你理解每个命令的作用，需要吗？




这两个截图展示了 **FraGNNet** 项目的核心源代码结构。这个结构非常规范，分为了**核心算法**、**基线模型**、**工具库**和**模型主程序**四个部分。

我来为你详细解读每一个文件和文件夹的作用：

### 1. 核心算法与基线模型 (Image 1: `src/fragnnet`)

这个文件夹包含了本项目最硬核的底层算法以及所有用来对比的基线模型实现。

* **`frag/` (碎片生成核心)**
* **`compute_frags.pyx`**: **这是全项目的核心引擎**。它是一个 Cython 文件，实现了论文中的**递归断键算法 (Recursive Fragmentation Algorithm)**。因为它需要进行大量的图遍历和组合计算，普通的 Python 跑不动，所以必须用 Cython 写成 C 扩展来加速。


* **`graff/`, `iceberg/`, `massformer/` (基线模型)**
* 这三个文件夹分别是论文对比过的三个基线模型：**GrAFF-MS**, **ICEBERG**, 和 **MassFormer**。作者为了公平对比，在同一个框架下重写了这些模型。
* 每个文件夹内部的结构都很类似：
* **`dataset.py`**: 专门针对该模型的数据加载器。
* **`model.py`**: 该模型的网络架构定义（如 MassFormer 的 Graph Transformer）。
* **`pl_model.py`**: 该模型的 PyTorch Lightning 训练模块（定义了 training_step, validation_step 等）。
* **`algos.pyx` (在 massformer 下)**: MassFormer 特有的高性能算法实现。
* **`fragmentation.py` (在 iceberg 下)**: ICEBERG 特有的碎片生成逻辑（基于 MAGMa 算法）。





### 2. 工具库 (Image 2: `utils`)

这个文件夹是项目的“工具箱”，存放了各种通用的辅助函数。

* **数据与特征处理**:
* **`data_utils.py` / `dataset.py**`: 处理原始 NIST 数据，读写文件等。
* **`feat_utils.py`**: **特征化工具**。负责把 RDKit 读取的分子对象转换成 GNN 能吃的图特征（原子类型、键类型等）。
* **`formula_utils.py`**: **分子式工具**。计算分子量、解析分子式字符串（如 "C6H6" -> `{'C':6, 'H':6}`）、处理同位素等。这是 FraGNNet 基于 Formula 预测的基础。


* **任务相关**:
* **`frag_utils.py`**: 处理碎片 DAG 结构的辅助函数。
* **`ms2c_utils.py`**: **检索任务工具**。负责计算检索准确率（Top-k Accuracy），模拟从候选库中查找分子的过程。
* **`spec_utils.py`**: **谱图处理工具**。负责谱图的合并、分箱（Binning）、计算余弦相似度（Cosine Similarity）、Hungarian Matching 等。


* **通用工具**:
* **`plot_utils.py`**: **画图工具**。论文里的那些漂亮的谱图对比图（如 Fig 3）就是用这个画出来的。
* **`nn_utils.py`**: 通用的神经网络组件（如 MLP 层、激活函数）。
* **`pl_utils.py`**: PyTorch Lightning 的辅助工具（如 Checkpoint 回调）。
* **`profile_utils.py`**: 性能分析工具，用来测试代码跑得快不快。



### 3. FraGNNet 模型主程序 (Image 2 底部)

在 `utils` 文件夹之外（位于 `src/fragnnet/` 根目录下）的这些文件，是 **FraGNNet 模型本身** 的实现代码：

* **`form_embedder.py`**: **分子式嵌入器**。实现了论文提到的 **Fourier Embeddings**，把离散的分子式（如 C、H 的数量）转换成连续的向量。
* **`loss.py`**: **损失函数**。实现了论文公式 (8) 和 (9)，包含了 **OS (Out-of-Support) 损失** 和 **熵正则化（Entropy Regularization）**。
* **`model.py`**: **模型架构**。定义了 `Molecule GNN` 和 `Fragment GNN` 的具体网络结构。
* **`pl_model.py`**: **训练流程**。定义了 FraGNNet 如何进行一次训练迭代、如何计算 Loss、如何记录日志。
* **`runner.py`**: **启动脚本**。这是整个程序的入口，负责解析命令行参数，加载配置，然后启动训练或测试。

### 总结

这套代码结构非常清晰：

1. **`frag/*.pyx`** 负责**算**（高性能计算碎片）。
2. **根目录下的 `model.py` 等** 负责**学**（FraGNNet 模型定义）。
3. **`utils/*`** 负责**帮**（处理数据、画图、算指标）。
4. **`graff/`, `iceberg/` 等** 负责**比**（基线模型对照）。