import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os
from pprint import pformat
import json

import ms2spectra.utils.frag_utils as frag_utils
import ms2spectra.utils.data_utils as data_utils
from ms2spectra.utils.misc_utils import booltype
from ms2spectra.utils.proc_utils import filter_spec_mol, merge_spec_df

"""
面向质谱碎片预测模型（如 FragGNN）的核心特征工程脚本，它基于前序标准化的分子（mol_df）和质谱（spec_df）数据，完成「分子碎裂路径 DAG 生成」「碎片匹配统计计算」「多粒度指标分析」三大核心任务，最终输出可直接用于模型训练 / 评估的碎片特征和统计指标，是连接原始质谱数据与机器学习模型的关键桥梁。
"""
def print_and_log(name,series,wandb_flag,stats_d):

	stats = series.describe()
	# print(f"> {name}")
	# print(stats)
	# print()
	for stat in ["mean","std","min","25%","50%","75%","max"]:
		stats_d[f"{name}/{stat}"] = stats[stat]

	if wandb_flag:
		import wandb
		log_d = {"step":0}
		for stat in ["mean","std","min","25%","50%","75%","max"]:
			log_d[f"{name}/{stat}"] = stats[stat]
		wandb.log(log_d)
		
def main(args):

	assert args.max_time < data_utils.JOBLIB_TIMEOUT, (args.max_time,data_utils.JOBLIB_TIMEOUT)

	elements = args.elements
	dsets = args.dsets
	# init wandb
	wandb_flag = args.wandb_mode != "off"
	if wandb_flag:
		import wandb
		wandb_config = vars(args)
		wandb.init(
			project=args.wandb_project,
			entity=args.wandb_entity,
			name=args.wandb_run_name,
			mode=args.wandb_mode,
			config=wandb_config,
			dir=args.project_dp,
			group="prepare_dag_feats"
		)

	# read in the molecule data
	print("read in mol")
	mol_df = pd.read_pickle(os.path.join(args.project_dp,args.proc_dp,"mol_df.pkl"))
	print("read in spec")
	spec_df = pd.read_pickle(os.path.join(args.project_dp,args.proc_dp,"spec_df.pkl"))
	print()

	# perform filter selection
	prec_types = args.prec_types if 'any' not in args.prec_types else None
	frag_modes = args.frag_modes if 'any' not in args.frag_modes else None
	ion_modes = args.ion_modes if 'any' not in args.ion_modes else None
	inst_types = args.inst_types if 'any' not in args.inst_types else None
	
	spec_df, mol_df = filter_spec_mol(
		spec_df,
		mol_df,
		elements=elements,
		dsets=dsets,
		prec_types=prec_types,
		num_entries=args.num_entries,
  		frag_modes=frag_modes,
    	ion_modes=ion_modes,
		inst_types=inst_types)
 
	m_spec_df = merge_spec_df(spec_df)

	print(f">> {len(spec_df)} spectra, {len(mol_df)} molecules") #, {len(m_spec_df)} merged spectra")
	print()

	num_atoms = mol_df["num_atoms"]
	print_and_log("num_atoms",num_atoms,wandb_flag,{})

	num_bonds = mol_df["num_bonds"]
	print_and_log("num_bonds",num_bonds,wandb_flag,{})

	os.makedirs(args.frag_dp,exist_ok=True)
	dag_dp = os.path.join(args.frag_dp, "dags")
	os.makedirs(dag_dp,exist_ok=True)
	group_stats_fp = os.path.join(args.frag_dp, "group_stats_df.pkl")
	global_stats_fp = os.path.join(args.frag_dp, "global_stats.json")

	def compute_spectra_stats(
			peaks,
			formula_peak_mzs,
			formula_peak_probs,
			idx_by_h_delta,
			prec_mz,
			prec_type,
			tolerances:list, 
			max_h_transfer:int):
		"""_summary_

		Args:
			tolerances (list): _description_
			h_transfer (int): _description_

		Returns:
			_type_: _description_
		"""
		# TODO: get this out of the main() scope
		results = []
		cols = []
		for tolerance in tolerances:
			for h_transfer in range(1, max_h_transfer + 1):
				keys = [
					f"recall_{tolerance}_h{h_transfer}",
					f"wrecall_{tolerance}_h{h_transfer}",
					f"prec_{tolerance}_h{h_transfer}",
					f"ppt_peak_{tolerance}_h{h_transfer}",
					f"ppt_formula_{tolerance}_h{h_transfer}",
					f"prec_recall_{tolerance}_h{h_transfer}",
					f"prec_spec_recall_{tolerance}_h{h_transfer}",
				]
				if 'ppm' in tolerance:
					result = frag_utils.compute_frag_peak_stats(peaks,formula_peak_mzs,formula_peak_probs,idx_by_h_delta,\
						 prec_mz,h_transfer,tolerance=float(tolerance[:-3]),prec_type=prec_type,is_ppm=True)
				else:
					result = frag_utils.compute_frag_peak_stats(peaks,formula_peak_mzs,formula_peak_probs,idx_by_h_delta,\
						 prec_mz,h_transfer,tolerance=float(tolerance),prec_type=prec_type)
				cols += keys
				results.append(result)
		result_series = pd.concat(results, axis=0)
		result_df = result_series.to_frame().T
		result_df.columns = cols
		return result_df
	
	print("> Compute Fragments")
	mol_input_rows = []
	for _, row in tqdm(mol_df.iterrows(), total=mol_df.shape[0], desc= "prepare inputs"):
		mol_input_rows.append(
			[
				row['mol'],
				row['mol_id'], 
				args.max_depth,
				True, # h_prior
				args.max_h_transfer,
				args.max_time,
				args.isotopes, 
				args.nb_isomorphic,
				args.max_iterations,
				dag_dp,
				args.use_cached_dag, # use_cached, default to false
				args.compress_dags
			]
		)
  
	print("> Running frag gen")
	frag_results_gen = data_utils.par_apply(iter(mol_input_rows),frag_utils.timed_get_dags,True, return_as_generator=True)
	frag_results = []
	for fr in tqdm(frag_results_gen, total = len(mol_input_rows), desc = "Compute Frags"):
		frag_results.append(fr)

	print("> Fragments Computed")
	frag_stats_df = pd.DataFrame({"mol_id":mol_df["mol_id"]})
	frag_stats_df["depth"] = list(map(lambda x: x.pop("max_depth",np.nan),frag_results))
	frag_stats_df["formula_peak_mzs"] = list(map(lambda x: x.pop("formula_peak_mzs",np.nan),frag_results))
	frag_stats_df["formula_peak_probs"] = list(map(lambda x: x.pop("formula_peak_probs",np.nan),frag_results))
	frag_stats_df["idx_to_formula"] = list(map(lambda x: x.pop("idx_to_formula",np.nan),frag_results))
	frag_stats_df["dag_num_edges"] = list(map(lambda x: x.pop("dag_num_edges",np.nan),frag_results))
	frag_stats_df["dag_num_nodes"] = list(map(lambda x: x.pop("dag_num_nodes",np.nan),frag_results))
	frag_stats_df["dag_sparsity"] = list(map(lambda x: x.pop("dag_sparsity",np.nan),frag_results))
	frag_stats_df["dag_num_nodes_nb"] = list(map(lambda x: x.pop("dag_num_nodes_nb",np.nan),frag_results))
	frag_stats_df["formula_redundancy"] = list(map(lambda x: x.pop("formula_redundancy",np.nan),frag_results))	
	frag_stats_df["idx_by_h_delta"] = list(map(lambda x: x.pop("idx_by_h_delta",np.nan),frag_results))	

	del frag_results

	frag_stats_d = {}
	# count failures, then remove them
	num_failures = frag_stats_df.isna().any(axis=1).sum()
	frag_stats_d["total_num_failures"] = num_failures
	print(f"> total num failures: {num_failures}")
	print()
	if wandb_flag:
		wandb.log({"total_num_failures": num_failures,"step":0})
	frag_stats_df = frag_stats_df.dropna(axis=0)

	### global properties

	# compute total number of formulae
	unique_formulae = set()
	for idx_to_formula in frag_stats_df["idx_to_formula"].values:
		unique_formulae.update(list(idx_to_formula.values()))
	frag_stats_d["total_num_formulae"] = len(unique_formulae)
	print(f"> total num formulae: {len(unique_formulae)}")
	print()

	# compute total number of depths
	print("> depth:")
	print(frag_stats_df["depth"].value_counts())
	print()
	depth_d = {f"depth/{k}":v for k,v in frag_stats_df["depth"].value_counts().to_dict().items()}
	frag_stats_d.update(depth_d)
	if wandb_flag:
		depth_d["step"] = 0
		wandb.log(depth_d)

	### molecule properties
	frag_stats_d = {}
	print_and_log("dag_num_edges",frag_stats_df["dag_num_edges"],wandb_flag,frag_stats_d)
	print_and_log("dag_num_nodes",frag_stats_df["dag_num_nodes"],wandb_flag,frag_stats_d)
	print_and_log("dag_sparsity",frag_stats_df["dag_sparsity"],wandb_flag,frag_stats_d)
	print_and_log("dag_num_nodes_nb",frag_stats_df["dag_num_nodes_nb"],wandb_flag,frag_stats_d)
	print_and_log("formula_redundancy",frag_stats_df["formula_redundancy"],wandb_flag,frag_stats_d)

	# count number of formula per molecule
	frag_stats_df.loc[:,"num_formulae"] = frag_stats_df["idx_to_formula"].apply(lambda x: len(x)-1)
	print_and_log("num_formulae",frag_stats_df["num_formulae"],wandb_flag,frag_stats_d)

	assert (frag_stats_df["num_formulae"]>0).all()

	# drop idx_to_formula
	frag_stats_df = frag_stats_df.drop(columns=["idx_to_formula"])

	print("> Compute Spectra Stats")

	stats_cols = ["num_formulae","depth","formula_redundancy","dag_num_edges","dag_num_nodes","dag_sparsity","dag_num_nodes_nb"]
	id_cols = ["mol_id"]
	data_cols = list(set(frag_stats_df.columns)-set(stats_cols)-set(id_cols))

	for merged in [False,True]:

		if not merged:
			spec_key = "spec_id"
			_spec_df = spec_df
			spec_prefix = "spec/"
			mol_prefix = "mol/"
			spec_stats_fp = os.path.join(args.frag_dp, "spec_stats_df.pkl")
			mol_stats_fp = os.path.join(args.frag_dp, "mol_stats_df.pkl")
		else:
			spec_key = "group_id"
			_spec_df = m_spec_df
			spec_prefix = "m_spec/"
			mol_prefix = "m_mol/"
			spec_stats_fp = os.path.join(args.frag_dp, "m_spec_stats_df.pkl")
			mol_stats_fp = os.path.join(args.frag_dp, "m_mol_stats_df.pkl")

		# compute spectra stats
		peak_spec_df = _spec_df[[spec_key,"mol_id","peaks","prec_mz","prec_type"]].merge(frag_stats_df[id_cols+data_cols],on="mol_id",how="inner")
		assert peak_spec_df.shape[0] == peak_spec_df.drop_duplicates(subset=[spec_key,"mol_id"]).shape[0]

		spectra_input_rows = []
		tolerances = args.tolerances
		for _, row in tqdm(peak_spec_df.iterrows(), total=peak_spec_df.shape[0], desc= "prepare spectra stats inputs"):
			spectra_input_rows.append(
				[
					row["peaks"], 
					row["formula_peak_mzs"], 
					row["formula_peak_probs"], 
					row["idx_by_h_delta"], 
					row["prec_mz"],
					row["prec_type"],
					tolerances, 
					args.max_h_transfer
				]
			)
		tqdm_iter = tqdm(spectra_input_rows,desc=pformat(compute_spectra_stats),total=len(spectra_input_rows))
		# run stats
		stats_results = data_utils.par_apply(tqdm_iter,compute_spectra_stats,True)
		# Loky and joblib should keep ordering
		stats_results_df = pd.concat(stats_results, axis=0, ignore_index=True)
		metric_keys = list(stats_results_df.columns)

		# add peak stats
		peak_spec_df = pd.concat((peak_spec_df[[spec_key,"mol_id"]],stats_results_df), axis=1)
		# add dag stats
		peak_spec_df = peak_spec_df.merge(frag_stats_df[id_cols+stats_cols],on="mol_id",how="inner")

		# save spectrum-level stats
		peak_spec_df.to_pickle(spec_stats_fp)

		# update the frag stats d
		for key in metric_keys+stats_cols:
			print_and_log(spec_prefix+key,peak_spec_df[key],wandb_flag,frag_stats_d)

		print("> Compute Molecule Stats")	

		# collect across molecules and report summary statistics
		peak_mol_df = peak_spec_df.drop(columns=[spec_key]).groupby("mol_id").agg(np.nanmean).reset_index()
		for key in metric_keys+stats_cols:
			print_and_log(mol_prefix+key,peak_mol_df[key],wandb_flag,frag_stats_d)

		# save molecule-level stats
		peak_mol_df.to_pickle(mol_stats_fp)


	# save the frag stats dict
	with open(global_stats_fp,"w",encoding="utf-8") as f:
		frag_stats_d = {k:str(v) for k,v in frag_stats_d.items()}
		json.dump(frag_stats_d,f,ensure_ascii=False,indent=4)

	if wandb_flag:
		wandb.finish()


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--num_entries",type=int,default=-1)
	parser.add_argument("--max_depth",type=int,default=4)
	parser.add_argument("--max_time",type=int,default=150)
	parser.add_argument("--project_dp",type=str,default=os.getcwd())
	parser.add_argument("--frag_dp",type=str,required=True)
	parser.add_argument("--proc_dp",type=str,required=True)
	parser.add_argument("--max_h_transfer",type=int,default=4)
	parser.add_argument("--nb_isomorphic",type=booltype,default=True)
	parser.add_argument("--max_iterations",type=int,default=3)
	parser.add_argument("--dsets",type=str,nargs="+",default=["nist20_hr","mona23"])
	parser.add_argument("--tolerances",type=str,nargs="+",default=["0.01","0.005","0.001","0.0001","10ppm","5ppm"])
	parser.add_argument("--isotopes",type=booltype,default=True)
	parser.add_argument("--wandb_mode",type=str,default="disabled",choices=["online","offline","disabled"])
	parser.add_argument("--wandb_project",type=str,default="frag-gnn")
	parser.add_argument("--wandb_entity",type=str,default="frag-gnn")
	parser.add_argument("--wandb_dp",type=str,default="wandb")
	parser.add_argument("--compress_dags",type=booltype,default=True)
	parser.add_argument("--wandb_run_name",type=str)
	parser.add_argument("--save_dag",type=booltype,default=True)
	parser.add_argument("--elements",type=str,nargs="+",default=frag_utils.ELEMENTS)
	parser.add_argument("--prec_types",nargs="+",default='any')
	parser.add_argument("--inst_types",nargs="+",default='any')
	parser.add_argument("--frag_modes",nargs="+",default='any')
	parser.add_argument("--ion_modes",nargs="+",default='any')
	parser.add_argument("--use_cached_dag",type=booltype,default=False)
	args = parser.parse_args()
 
	# Check if --flag is True and --conditional-arg is missing
	if args.wandb_mode != 'disabled' and args.wandb_run_name:
		parser.error("--wandb_run_name is required when --wandb_mode is not disabled")

	tqdm.pandas()

	main(args)


"""
步骤 1：初始化与参数校验
解析命令行参数（如最大碎裂深度、计算超时时间、容忍度列表、数据集筛选条件等），校验max_time不超过 JOBLIB 并行计算的超时阈值；
初始化 WandB（若wandb_mode非disabled）：配置项目 / 实体 / 运行名，将所有参数记录到 WandB，方便实验复现；
加载前序预处理好的mol_df.pkl（分子特征）和spec_df.pkl（质谱谱图）。
步骤 2：数据过滤与合并
按指定条件（元素、数据集、前体类型、碎裂模式、离子模式、仪器类型等）过滤无效数据（filter_spec_mol）；
合并相同实验条件的谱图（merge_spec_df）生成m_spec_df（合并谱图版），减少重复计算；
输出过滤后的数据量（谱图数、分子数），并统计分子的原子数、键数等基础特征。
步骤 3：分子碎裂 DAG 生成（最核心）
DAG（有向无环图）是描述分子碎裂路径的核心结构，这一步为每个分子生成碎裂 DAG 并提取特征：
准备输入：遍历mol_df，组装每个分子的计算参数（mol 对象、mol_id、最大碎裂深度、最大氢转移数、超时时间、是否考虑同位素等）；
并行生成 DAG：调用frag_utils.timed_get_dags（带超时控制）并行生成碎片 DAG，避免单个分子计算卡死，返回 DAG 相关特征；
整理碎片统计：
生成frag_stats_df，包含 DAG 核心特征（节点数、边数、稀疏性、公式冗余度、氢转移索引等）；
过滤 DAG 生成失败的条目，统计失败数并记录到 WandB；
统计全局唯一分子式数量、DAG 深度分布等关键指标。
步骤 4：定义质谱碎片统计函数
compute_spectra_stats是核心统计函数，按不同容忍度 / 氢转移数计算碎片匹配指标：
遍历指定的容忍度（如0.01绝对误差、5ppm相对误差）和氢转移数（1~max_h_transfer）；
计算 7 类核心指标（召回率 / 加权召回率 / 精确率 / 峰占比 / 公式占比等），指标命名规则为指标名_容忍度_h氢转移数（如recall_5ppm_h2）；
区分绝对误差（数值）和相对误差（ppm）的计算逻辑，返回结构化的统计结果。
步骤 5：多粒度质谱碎片统计计算
分别处理「原始谱图（spec_id）」和「合并谱图（group_id）」两种粒度，流程完全一致：
合并谱图数据与碎片统计数据（peak_spec_df），确保无重复条目；
准备统计输入行（峰数据、公式峰 m/z、容忍度列表等）；
并行调用compute_spectra_stats计算谱图级统计指标，保存为spec_stats_df/m_spec_stats_df；
按mol_id聚合谱图级指标，得到分子级统计数据，保存为mol_stats_df/m_mol_stats_df；
调用print_and_log记录所有指标的统计描述，更新全局统计字典。
步骤 6：结果保存
将全局统计字典（包含所有指标的均值 / 标准差 / 分位数等）保存为global_stats.json；
DAG 文件保存到frag_dp/dags目录（可选压缩）；
所有统计 DataFrame 保存为 pkl 格式（保留复杂数据类型，如列表 / 字典）。
步骤 7：收尾
关闭 WandB（若开启），完成整个流程。
3. 主入口：参数解析与合法性校验
解析命令行参数（覆盖碎片生成、统计、WandB 配置、数据过滤等所有维度）；
校验 WandB 参数合法性（非 disabled 模式时必须指定wandb_run_name）；
启用 tqdm 的 pandas 适配，调用main函数执行核心流程。
核心输入输出说明
输入	输出
预处理后的mol_df.pkl/spec_df.pkl	1. DAG 文件：frag_dp/dags/（分子碎裂路径图，可选压缩）；
2. 谱图级统计：spec_stats_df.pkl/m_spec_stats_df.pkl；
3. 分子级统计：mol_stats_df.pkl/m_mol_stats_df.pkl；
4. 全局统计：global_stats.json（所有指标的描述性统计）
命令行参数（过滤 / 计算参数）	控制台输出：数据量、失败数、DAG 深度分布、各指标统计；
WandB（可选）：指标可视化图表
总结
核心目标：为质谱碎片预测模型生成分子碎裂 DAG 特征，并计算不同容忍度 / 氢转移数下的碎片匹配指标（召回率 / 精确率等）；
关键特色：
并行计算 + 超时控制：避免单个分子计算卡死，提升效率；
多粒度分析：覆盖「原始谱图 / 合并谱图 / 分子」三个维度；
多条件统计：支持不同误差容忍度、氢转移数的碎片匹配计算；
应用场景：输出的特征和统计指标可直接用于 FragGNN 等模型的训练、评估，或质谱碎片算法的性能对比。
"""