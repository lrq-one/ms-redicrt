import argparse
import os
import pandas as pd
import numpy as np
import json

from ms2spectra.utils import frag_utils
from ms2spectra.utils.misc_utils import booltype, np_temp_seed
from ms2spectra.utils.proc_utils import filter_spec_mol

def make_splits(args):

	mol_df = pd.read_pickle(os.path.join(args.proc_dp,"mol_df.pkl"))
	spec_df = pd.read_pickle(os.path.join(args.proc_dp,"spec_df.pkl"))

	print("> Spectrum filters")

	# filter spectra 
	prec_types = args.prec_types if 'any' not in args.prec_types else None
	frag_modes = args.frag_modes if 'any' not in args.frag_modes else None
	ion_modes = args.ion_modes if 'any' not in args.ion_modes else None
	inst_types = args.inst_types if 'any' not in args.inst_types else None

	dsets = args.primary_dsets + args.secondary_dsets
	f_spec_df, f_mol_df = filter_spec_mol(
		spec_df,
		mol_df,
		dsets=dsets,
		prec_types=prec_types,
		max_peak_mz=args.max_peak_mz,
		max_prec_mz=args.max_prec_mz,
		min_prec_mz=args.min_prec_mz,
		elements=args.elements,
		frag_modes=frag_modes,
		ion_modes=ion_modes,
		inst_types=inst_types,
		ces = args.ces
	)
	print(f"> Selected {f_spec_df.shape[0]}/{spec_df.shape[0]} spectra ({(f_spec_df.shape[0]/spec_df.shape[0]) * 100} % total spectrum)")
	spec_df = f_spec_df
	mol_df = f_mol_df
	primary_spec_df = spec_df[spec_df["dset"].isin(args.primary_dsets)]
	secondary_spec_df = spec_df[spec_df["dset"].isin(args.secondary_dsets)]

	if args.dag_filtering:
		print("> Read DAG stats")
		dag_stats_df = pd.read_pickle(
			os.path.join(args.frag_dp,f"{args.dag_filter_grouping}_stats_df.pkl")
		)
		print("> DAG filters")
		# filter dags
		node_key = "dag_num_nodes"
		edge_key = "dag_num_edges"
		wrecall_key = "wrecall_10ppm_h4"
		dag_masks = []
		if args.max_num_dag_nodes != -1:
			print(f"> filtering on max dag nodes {args.max_num_dag_nodes}")
			dag_masks.append(dag_stats_df[node_key] <= args.max_num_dag_nodes)
		if args.max_num_dag_edges != -1:
			print(f"> filtering: max dag edges {args.max_num_dag_edges}")
			dag_masks.append(dag_stats_df[edge_key] <= args.max_num_dag_edges)
		print(f"> filtering: min dag wrecall {args.min_dag_wrecall}")
		dag_masks.append(dag_stats_df[wrecall_key] >= args.min_dag_wrecall)
		dag_masks = np.stack(dag_masks,axis=1)
		dag_mask = np.all(dag_masks,axis=1)
		print(f"> Selected {np.sum(dag_mask)}/{dag_mask.shape[0]} Dags ({np.mean(dag_mask) * 100 } % total dags)")
		dag_stats_df = dag_stats_df[dag_mask]
		# dag_mol_id = dag_stats_df["mol_id"]
		if args.dag_filter_grouping in ["m_spec","m_mol"]:
			dag_spec_id = None
			dag_group_id = dag_stats_df["group_id"]
		else:
			dag_spec_id = dag_stats_df["spec_id"]
			dag_group_id = None

		print("> Intersection")
		# get intersection
		if args.dag_filter_grouping in ["m_spec","m_mol"]:
			primary_both_group_id = np.intersect1d(primary_spec_df["group_id"],dag_group_id)
			primary_split_df = primary_spec_df[primary_spec_df["group_id"].isin(primary_both_group_id)]
			secondary_both_group_id = np.intersect1d(secondary_spec_df["group_id"],dag_group_id)
			secondary_split_df = secondary_spec_df[secondary_spec_df["group_id"].isin(secondary_both_group_id)]
		else:
			primary_both_spec_id = np.intersect1d(primary_spec_df["spec_id"],dag_spec_id)
			primary_split_df = primary_spec_df[primary_spec_df["spec_id"].isin(primary_both_spec_id)]
			secondary_both_spec_id = np.intersect1d(secondary_spec_df["spec_id"],dag_spec_id)
			secondary_split_df = secondary_spec_df[secondary_spec_df["spec_id"].isin(secondary_both_spec_id)]
		primary_split_df = primary_split_df.merge(mol_df[["mol_id",args.split_key]],on="mol_id",how="inner")
		secondary_split_df = secondary_split_df.merge(mol_df[["mol_id",args.split_key]],on="mol_id",how="inner")
	else:
		primary_split_df = primary_spec_df.merge(mol_df[["mol_id",args.split_key]],on="mol_id",how="inner")
		secondary_split_df = secondary_spec_df.merge(mol_df[["mol_id",args.split_key]],on="mol_id",how="inner")

	print("> create split(s)")
	split_data_list = []
	if args.split_type in ["random", "random_folds"]:
		# split based on molecule
		# primary split
		primary_split_mol_id = np.unique(primary_split_df["mol_id"])
		split_keys = primary_split_df[primary_split_df["mol_id"].isin(primary_split_mol_id)][args.split_key]
		split_keys = np.unique(split_keys)
		if args.total_frac == 1.0:
			total_num = split_keys.shape[0]
		else:
			total_num = int(np.ceil(split_keys.shape[0]*args.total_frac))
		
		if args.split_type == "random":
			test_num = int(np.ceil(total_num*args.test_frac))
			val_num = int(np.ceil(total_num*args.val_frac))
			with np_temp_seed(args.meta_rseed):
				if args.total_frac < 1.0:
					split_keys = np.random.choice(split_keys,size=total_num,replace=False)
				test_keys = np.random.choice(split_keys,size=test_num,replace=False)
				train_val_keys = np.setdiff1d(split_keys,test_keys)
				val_keys = np.random.choice(train_val_keys,size=val_num,replace=False)
				train_keys = np.setdiff1d(train_val_keys,val_keys)
			
			train_df = primary_split_df[primary_split_df[args.split_key].isin(train_keys)][["spec_id","mol_id","group_id"]]
			val_df = primary_split_df[primary_split_df[args.split_key].isin(val_keys)][["spec_id","mol_id","group_id"]]
			test_df = primary_split_df[primary_split_df[args.split_key].isin(test_keys)][["spec_id","mol_id","group_id"]]
   
			# secondary split
			secondary_df = secondary_split_df[~secondary_split_df[args.split_key].isin(train_val_keys)][["spec_id","mol_id","group_id"]]
			split_data = { 
				"split_dp": args.split_dp,
				"train_df": train_df,
				"val_df": val_df,
				"test_df": test_df,
				"secondary_df": secondary_df
			}
			split_data_list.append(split_data)
		elif args.split_type == "random_folds":
			with np_temp_seed(args.meta_rseed):
				if args.total_frac < 1.0:
					split_keys = np.random.choice(split_keys,size=total_num,replace=False)
		
				num_cv = int(1 / args.test_frac)
				test_keys_cvs = np.array_split(split_keys, num_cv)
				print(f"> creating split for {num_cv} cv folds for {len(split_keys)} grouped cases")
				for i in range(num_cv):
					test_keys = test_keys_cvs[i]
					print(f"> creating split for {i} th cv, {len(test_keys)} grouped test cases")
					# test and val
					train_val_keys = np.setdiff1d(split_keys,test_keys)
					val_num = int(np.ceil(total_num*args.val_frac))
					val_keys = np.random.choice(train_val_keys,size=val_num,replace=False)
					train_keys = np.setdiff1d(train_val_keys,val_keys)
	
					train_df = primary_split_df[primary_split_df[args.split_key].isin(train_keys)][["spec_id","mol_id","group_id"]]
					val_df = primary_split_df[primary_split_df[args.split_key].isin(val_keys)][["spec_id","mol_id","group_id"]]
					test_df = primary_split_df[primary_split_df[args.split_key].isin(test_keys)][["spec_id","mol_id","group_id"]]
					# secondary split
					secondary_df = secondary_split_df[~secondary_split_df[args.split_key].isin(train_val_keys)][["spec_id","mol_id","group_id"]]
					split_data = { 
						"split_dp": os.path.join(args.split_dp, f"cv_{i}"),
						"train_df": train_df,
						"val_df": val_df,
						"test_df": test_df,
						"secondary_df": secondary_df
					}
					split_data_list.append(split_data)
	elif args.split_type == "predefined":
		#assert args.split_type == "predefined"
		assert len(args.secondary_dsets) == 0, len(args.secondary_dsets)
		assert len(args.train_ids) > 1 and len(args.val_ids) > 1 and len(args.test_ids) > 1
		train_ids = np.array(args.train_ids)
		val_ids = np.array(args.val_ids)
		test_ids = np.array(args.test_ids)

		train_df = primary_split_df[primary_split_df[args.id_type].isin(train_ids)][["spec_id","mol_id","group_id"]]
		val_df = primary_split_df[primary_split_df[args.id_type].isin(val_ids)][["spec_id","mol_id","group_id"]]
		test_df = primary_split_df[primary_split_df[args.id_type].isin(test_ids)][["spec_id","mol_id","group_id"]]
		secondary_df = primary_split_df[np.zeros(primary_split_df.shape[0],dtype=bool)][["spec_id","mol_id","group_id"]]

		split_data = { 
				"split_dp": args.split_dp,
				"train_df": train_df,
				"val_df": val_df,
				"test_df": test_df,
				"secondary_df": secondary_df
		}
		split_data_list.append(split_data)
	elif args.split_type == "predefined_dsetid_csv":
		# support more then one predefined set, useful when combine more then one set
		predefined_dfs = []
		for fp in args.predefined_dsetid_csv:
			case_predefined_df = pd.read_csv(fp)
			predefined_dfs.append(case_predefined_df)
		predefined_df = pd.concat(predefined_dfs)
  
		train_dest_ids = predefined_df[predefined_df['fold'] == 'train']['dset_spec_id'].to_list()
		train_df = primary_split_df[primary_split_df['dset_spec_id'].isin(train_dest_ids)][["spec_id","mol_id","group_id"]]
		val_dest_ids = predefined_df[predefined_df['fold'] == 'val']['dset_spec_id'].to_list()
		val_df = primary_split_df[primary_split_df['dset_spec_id'].isin(val_dest_ids)][["spec_id","mol_id","group_id"]]
		test_dest_ids = predefined_df[predefined_df['fold'] == 'test']['dset_spec_id'].to_list()
		test_df = primary_split_df[primary_split_df['dset_spec_id'].isin(test_dest_ids)][["spec_id","mol_id","group_id"]]
		secondary_df = primary_split_df[np.zeros(primary_split_df.shape[0],dtype=bool)][["spec_id","mol_id","group_id"]]

		split_data = { 
				"split_dp": args.split_dp,
				"train_df": train_df,
				"val_df": val_df,
				"test_df": test_df,
				"secondary_df": secondary_df
		}
		split_data_list.append(split_data)
	else:
		raise ValueError(f"{args.split_type} is not supported")
  
	# create split directory
	for split_data in split_data_list:
		os.makedirs(split_data['split_dp'],exist_ok=True)
		print(f"> Save split to {split_data['split_dp']}")
  
		for split_type in ["train_df","val_df","test_df","secondary_df"]:
			print(">", split_type)
			print("> number of unique mols", split_data[split_type]['mol_id'].nunique())
			print("> number of unique spec", split_data[split_type]['spec_id'].nunique())
			print("> number of unique groups", split_data[split_type]['group_id'].nunique())
			print("-" * 16)
		#print(f"> Split sizes (spec_id): train = {split_data['train_df'].shape[0]},"
		#	f"val = {split_data['val_df'].shape[0]}, test = {split_data['test_df'].shape[0]} "
		#	f"secondary test df = {split_data['secondary_df'].shape[0]}")
		
		# save ids
		train_fp = os.path.join(split_data['split_dp'],"train_ids.csv")
		val_fp = os.path.join(split_data['split_dp'],"val_ids.csv")
		test_fp = os.path.join(split_data['split_dp'],"test_ids.csv")
		secondary_fp = os.path.join(split_data['split_dp'],"secondary_ids.csv")
		split_data['train_df'].to_csv(train_fp,index=False)
		split_data['val_df'].to_csv(val_fp,index=False)
		split_data['test_df'].to_csv(test_fp,index=False)
		split_data['secondary_df'].to_csv(secondary_fp,index=False)
		
		# save split metadata
		meta_d = vars(args)
		meta_fp = os.path.join(split_data['split_dp'],"meta.json")
		with open(meta_fp,"w") as f:
			json.dump(meta_d,f,indent=4)
	
	return

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--split_type",type=str,choices=["random","predefined","predefined_dsetid_csv","random_folds"],default="random")
	parser.add_argument("--split_key",type=str,choices=["inchikey_s","scaffold"],default="inchikey_s")
	parser.add_argument("--id_type",type=str,choices=["spec_id","mol_id","group_id"],default="group_id")
	parser.add_argument("--train_ids",type=int,nargs="+",default=[62611])
	parser.add_argument("--val_ids",type=int,nargs="+",default=[62611])
	parser.add_argument("--test_ids",type=int,nargs="+",default=[62611])
	parser.add_argument("--predefined_dsetid_csv", nargs="+", type=str, default=[])
	parser.add_argument("--num_folds",type=int,default=0)
	# spec filtering criteria
	parser.add_argument("--primary_dsets",type=str,nargs="+",default=["nist20_hr","mona23"])
	parser.add_argument("--secondary_dsets",type=str,nargs="+",default=[])
	parser.add_argument("--max_peak_mz",type=float,default=1500.)
	parser.add_argument("--max_prec_mz",type=float,default=1500.)
	parser.add_argument("--min_prec_mz",type=float,default=0.)
	# dag filtering criteria
	parser.add_argument("--dag_filtering",type=booltype,default=True)
	parser.add_argument("--dag_filter_grouping",type=str,choices=["mol","spec","m_mol","m_spec"],default="m_spec")
	parser.add_argument("--max_num_dag_nodes",type=int,default=100000)
	parser.add_argument("--max_num_dag_edges",type=int,default=250000)
	parser.add_argument("--min_dag_wrecall",type=float,default=0.00)
	parser.add_argument("--elements",type=str,nargs="+",default=frag_utils.ELEMENTS)
	parser.add_argument("--prec_types", type=str,nargs="+", required=False, default='any')
	parser.add_argument("--inst_types", type=str, nargs="+", required=False, default='any')
	parser.add_argument("--frag_modes", type=str, nargs="+", required=False, default='any')
	parser.add_argument("--ion_modes", type=str, nargs="+", required=False, default='any')
	parser.add_argument('--ces',required=False, choices=['nce', 'ace', 'any'], default='any')

	# non-filtering args
	parser.add_argument("--meta_rseed",type=int,default=420420)
	parser.add_argument("--total_frac",type=float,default=1.0)
	parser.add_argument("--test_frac",type=float,default=0.2)
	parser.add_argument("--val_frac",type=float,default=0.2)
	parser.add_argument("--proc_dp",type=str,required=True)
	parser.add_argument("--frag_dp",type=str,required=True)
	parser.add_argument("--split_dp",type=str,required=True)
	args = parser.parse_args()

	make_splits(args)

"""
你想理解的这段代码是**质谱（MS/MS）数据集的标准化划分脚本**，核心作用是基于前序预处理的分子/谱图数据，结合「谱图过滤+DAG过滤」筛选高质量数据，并支持多种灵活的划分策略（随机、交叉验证、预定义ID等），将数据集拆分为训练/验证/测试集（含secondary测试集），最终保存划分后的ID文件和元数据，为后续的机器学习模型训练（如碎片预测）提供标准化、可复现的数据集划分。

### 代码核心结构与功能拆解
#### 1. 前置准备：模块导入
- 导入参数解析（`argparse`）、文件操作（`os`）、数据处理（`pandas`/`numpy`）、JSON处理（`json`）；
- 导入`ms2spectra`自定义工具：碎片工具（`frag_utils`）、随机种子控制（`np_temp_seed`）、数据过滤（`filter_spec_mol`）。

#### 2. make_splits函数：核心划分流程（分5大步骤）
##### 步骤1：加载数据与谱图过滤
- 加载前序预处理的`mol_df.pkl`（分子特征）和`spec_df.pkl`（质谱谱图）；
- 按指定条件过滤谱图数据（`filter_spec_mol`），过滤维度包括：
  - 数据集来源（`primary_dsets`/`secondary_dsets`）；
  - 实验条件（前体类型、碎裂模式、离子模式、仪器类型、碰撞能量类型）；
  - 数值范围（最大/最小前体质荷比、最大峰m/z）；
  - 元素限制（仅保留含指定元素的分子）；
- 将过滤后的数据拆分为`primary_spec_df`（主要数据集，用于训练/验证/测试）和`secondary_spec_df`（次要数据集，仅用于额外测试）。

##### 步骤2：DAG过滤（可选，核心质量控制）
若开启`dag_filtering`，进一步过滤碎片生成质量不达标的数据：
- 读取DAG统计数据（`dag_stats_df`），根据以下条件过滤：
  - DAG节点数 ≤ `max_num_dag_nodes`；
  - DAG边数 ≤ `max_num_dag_edges`；
  - 加权召回率（`wrecall_10ppm_h4`）≥ `min_dag_wrecall`；
- 按`dag_filter_grouping`（m_spec/m_mol/spec/mol）取过滤后DAG数据与谱图数据的交集，确保仅保留碎片生成质量达标的条目。

##### 步骤3：数据集划分（支持4种策略，核心核心）
根据`split_type`选择划分策略，所有策略均基于`split_key`（`inchikey_s`/`scaffold`，保证同分子/同骨架不跨集）：

| 划分策略               | 适用场景                                                                 | 核心逻辑                                                                 |
|------------------------|--------------------------------------------------------------------------|--------------------------------------------------------------------------|
| `random`               | 基础随机划分                                                             | 按`test_frac`/`val_frac`随机拆分primary数据为train/val/test，secondary数据排除train/val集 |
| `random_folds`         | 交叉验证（CV）                                                           | 将primary数据拆分为`1/test_frac`个fold，每个fold轮流作为test集，剩余拆分为train/val |
| `predefined`           | 基于指定ID列表划分                                                       | 按`train_ids`/`val_ids`/`test_ids`直接划分，无secondary集                 |
| `predefined_dsetid_csv`| 基于CSV文件中的预定义划分（支持多文件合并）                              | 按CSV中`dset_spec_id`和`fold`列（train/val/test）划分，无secondary集       |

##### 步骤4：结果统计与保存
- 为每个划分（或每个CV fold）创建独立目录，统计并输出各集的关键信息：
  - 唯一分子数、唯一谱图数、唯一分组数（`group_id`）；
- 保存划分后的ID文件：
  - `train_ids.csv`/`val_ids.csv`/`test_ids.csv`：训练/验证/测试集的spec_id/mol_id/group_id；
  - `secondary_ids.csv`：次要测试集的ID；
- 保存划分元数据（`meta.json`）：记录所有命令行参数，保证划分可复现。

##### 步骤5：返回结果
完成所有划分和保存，流程结束。

#### 3. 主函数：参数解析与执行
- 解析命令行参数（覆盖过滤条件、划分策略、随机种子、路径等）；
- 调用`make_splits`函数执行核心划分流程。

### 核心输入输出说明
| 输入                          | 输出                                                                 |
|-------------------------------|----------------------------------------------------------------------|
| 预处理后的`mol_df.pkl`/`spec_df.pkl` | 1. 划分目录（`split_dp`/`cv_i`）：<br>   - train/val/test/secondary_ids.csv（ID列表）；<br>   - meta.json（划分参数） |
| DAG统计数据（可选）           | 控制台输出：各集的唯一分子数/谱图数/分组数，划分比例                  |
| 命令行参数（过滤/划分策略）   |                                                                      |

### 总结
1. **核心目标**：为质谱碎片预测模型提供标准化、可复现的数据集划分，兼顾数据质量（谱图+DAG过滤）和划分灵活性（多种策略）；
2. **关键特色**：
   - 多层过滤：谱图实验条件过滤 + DAG碎片质量过滤，保证数据高质量；
   - 防信息泄漏：基于`inchikey_s`/`scaffold`划分，避免同分子/同骨架跨训练/测试集；
   - 多策略支持：覆盖基础随机划分、交叉验证、预定义划分，满足不同训练需求；
3. **应用场景**：输出的ID文件可直接用于后续模型训练，指定训练/验证/测试集的范围，元数据保证实验可复现。
"""