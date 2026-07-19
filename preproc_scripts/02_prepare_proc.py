import numpy as np
import os
import pandas as pd
import argparse
from rdkit import RDLogger

import ms2spectra.utils.data_utils as data_utils
from ms2spectra.utils.data_utils import par_apply_series, par_apply_df_rows, seq_apply_df_rows
from ms2spectra.utils.misc_utils import booltype

"""
质谱（MS/MS）数据的深度清洗与标准化预处理脚本，它承接上一步生成的统一格式 CSV 文件，
进一步完成分子结构标准化、字段类型转换、数据一致性校验、特征提取等核心操作，
最终输出三个结构化的核心 DataFrame（spec_df、mol_df、ann_df）和调试用数据，
为后续的质谱数据分析、机器学习建模（如碎片预测、化合物识别）提供高质量的结构化数据集。
"""
"""
加载并合并多源数据集
功能：读取上一步生成的多个数据集 CSV 文件，合并为统一的 DataFrame，支持采样限制条目数。
遍历指定数据集（如nist20_hr、mona23），读取每个数据集的{dset}_df.csv文件，添加dset字段标记数据来源；
若指定num_entries>0，对每个数据集固定随机种子（保证可复现）采样指定数量条目；
合并所有数据集，重置索引，返回统一的all_df。
"""
def load_df(df_dp,dsets,num_entries):
	
	dfs = []
	for dset in dsets:
		dset_df = pd.read_csv(os.path.join(df_dp,f"{dset}_df.csv"),dtype=str)
		dset_df.loc[:,"dset"] = dset
		dfs.append(dset_df)
	if num_entries > 0:
		dfs = [df.sample(n=num_entries,replace=False,random_state=420) for df in dfs]
	if len(dfs) > 1:
		all_df = pd.concat(dfs,ignore_index=True)
	else:
		all_df = dfs[0]
	all_df = all_df.reset_index(drop=True)
	return all_df

def preprocess_spec(spec_df):

	# drop entries with the same dset_spec_id (this happens sometimes in MoNA)
	spec_df = spec_df.drop_duplicates(subset=["dset","dset_spec_id"],keep="first")
	# convert smiles to mol and back (for standardization/stereochemistry)
	spec_df.loc[:,"mol"] = par_apply_series(spec_df["smiles"],data_utils.mol_from_smiles)
	spec_df.loc[:,"smiles"] = par_apply_series(spec_df["mol"],data_utils.mol_to_smiles)
	no_mol_df = spec_df[spec_df["mol"].isna()|spec_df["smiles"].isna()][["dset_spec_id"]]
	spec_df = spec_df.dropna(subset=["mol","smiles"])
	assert not (spec_df["smiles"] == "").any()
	# enumerate smiles to create molecule ids
	smiles_set = set(spec_df["smiles"])
	assert not "" in smiles_set
	print("> num_smiles", len(smiles_set))
	print("> sorting by smiles")
	smiles_to_mid = {smiles:i for i,smiles in enumerate(sorted(smiles_set))}
	print("> updating mol_id")
	spec_df.loc[:,"mol_id"] = spec_df["smiles"].map(smiles_to_mid)#.replace(smiles_to_mid)

	# copy for ann_df
	ann_df = spec_df[["dset_spec_id","mol_id","notes","peaks","dset"]].copy()

	# extract peak info (still represented as str)
	spec_df.loc[:,"peaks"] = par_apply_series(spec_df["peaks"],data_utils.parse_peaks_str)
	# get mz resolution
	spec_df.loc[:,"res"] = par_apply_series(spec_df["peaks"],data_utils.get_res)
	# standardize the instrument type and frag_mode
	inst_type, frag_mode = seq_apply_df_rows(spec_df,data_utils.parse_inst_info)
	spec_df.loc[:,"inst_type"] = inst_type
	spec_df.loc[:,"frag_mode"] = frag_mode
	# standardize ce
	spec_df.loc[:,"ace"] = par_apply_series(spec_df["col_energy"],data_utils.parse_ace_str)
	spec_df.loc[:,"nce"] = par_apply_series(spec_df["col_energy"],data_utils.parse_nce_str)
	spec_df = spec_df.drop(columns=["col_energy"])
	# standardise prec_type
	spec_df.loc[:,"prec_type"] = par_apply_series(spec_df["prec_type"],data_utils.parse_prec_type_str)
	# convert prec_mz
	spec_df.loc[:,"prec_mz"] = pd.to_numeric(spec_df["prec_mz"],errors="coerce")
	# convert ion_mode
	spec_df.loc[:,"ion_mode"] = par_apply_series(spec_df["ion_mode"],data_utils.parse_ion_mode_str)
	# convert peaks to float
	spec_df.loc[:,"peaks"] = par_apply_series(spec_df["peaks"],data_utils.convert_peaks_to_float)
	# get retention index
	spec_df.loc[:,"ri"] = par_apply_series(spec_df["ri"],data_utils.parse_ri_str)
	# convert exact_mass
	spec_df.loc[:,"exact_mass"] = pd.to_numeric(spec_df["exact_mass"],errors="coerce")

	# remove columns from spec_df
	spec_df = spec_df[["spec_id","mol_id","prec_type","inst_type","frag_mode","spec_type","ion_mode","dset","dset_spec_id","col_gas","res","ace","nce","prec_mz","peaks","ri","formula","inchikey","exact_mass"]]
	# relabel spec_id (this is to make it unique across datasets)
	spec_df.loc[:,"spec_id"] = np.arange(spec_df.shape[0])
	# set group_id (same compound and precursor)
	group_df = spec_df.drop(columns=["spec_id","dset_spec_id","peaks","nce","ace","res","prec_mz","ri","col_gas","formula","inchikey","exact_mass"]).drop_duplicates()
	group_df.loc[:,"group_id"] = np.arange(group_df.shape[0])
	spec_df = spec_df.merge(group_df,how="inner")

	# get mol df
	mol_df = pd.DataFrame(zip(sorted(smiles_set),list(range(len(smiles_set)))),columns=["smiles","mol_id"])
	mol_df.loc[:,"mol"] = par_apply_series(mol_df["smiles"],data_utils.mol_from_smiles)
	mol_df.loc[:,"inchikey_s"] = par_apply_series(mol_df["mol"],data_utils.mol_to_inchikey_s)
	mol_df.loc[:,"scaffold"] = par_apply_series(mol_df["mol"],data_utils.get_murcko_scaffold)
	mol_df.loc[:,"formula"] = par_apply_series(mol_df["mol"],data_utils.mol_to_formula)
	mol_df.loc[:,"inchi"] = par_apply_series(mol_df["mol"],data_utils.mol_to_inchi)
	mol_df.loc[:,"mw"] = par_apply_series(mol_df["mol"],lambda mol: data_utils.mol_to_mol_weight(mol,exact=False))
	mol_df.loc[:,"exact_mw"] = par_apply_series(mol_df["mol"],lambda mol: data_utils.mol_to_mol_weight(mol,exact=True))
	mol_df.loc[:,"num_atoms"] = par_apply_series(mol_df["mol"],data_utils.mol_to_num_atoms)
	mol_df.loc[:,"num_bonds"] = par_apply_series(mol_df["mol"],data_utils.mol_to_num_bonds)
	mol_df.loc[:,"charge"] = par_apply_series(mol_df["mol"],data_utils.mol_to_charge)
	mol_df.loc[:,"single_mol"] = par_apply_series(mol_df["mol"],data_utils.check_single_mol)
	mol_df.loc[:,"num_radicals"] = par_apply_series(mol_df["mol"],data_utils.mol_to_num_radicals)
	assert not (mol_df["smiles"] == "").any()
	assert not (mol_df["formula"] == "").any()
	assert not (mol_df["exact_mw"] == 0).any()

	# remove invalid mols and corresponding spectra
	all_mol_id = set(mol_df["mol_id"])
	mol_df = mol_df.dropna(subset=["mol"],axis=0)
	bad_mol_id = all_mol_id - set(mol_df["mol_id"])
	print("> bad_mol_id",len(bad_mol_id))
	spec_df = spec_df[~spec_df["mol_id"].isin(bad_mol_id)]

	# check how many formulae/inchikeys are different
	formula_df = spec_df[["dset_spec_id","spec_id","mol_id","formula"]].copy()
	formula_df = formula_df[formula_df["formula"] != ""]
	formula_df = formula_df.merge(mol_df[["mol_id","formula"]],on="mol_id",how="inner")
	diff_formula_df = formula_df[formula_df["formula_x"] != formula_df["formula_y"]]
	print("> formula inconsistencies")
	print(diff_formula_df["mol_id"].nunique(), diff_formula_df["spec_id"].nunique())
	inchikey_df = spec_df[["dset_spec_id","spec_id","mol_id","inchikey"]].copy().dropna()
	inchikey_df["inchikey_s"] = inchikey_df["inchikey"].str[:14]
	inchikey_df = inchikey_df.drop(columns=["inchikey"])
	inchikey_df = inchikey_df.merge(mol_df[["mol_id","inchikey_s"]],on="mol_id",how="inner")
	diff_inchikey_df = inchikey_df[inchikey_df["inchikey_s_x"] != inchikey_df["inchikey_s_y"]]
	print("> inchikey_s inconsistencies")
	print(diff_inchikey_df["mol_id"].nunique(), diff_inchikey_df["spec_id"].nunique())
	mass_df = spec_df[["dset_spec_id","spec_id","mol_id","exact_mass"]].copy().rename(columns={"exact_mass":"exact_mw"})
	# keep only non-trivial exact_mw annotations from the spec_df
	mass_df = mass_df[~mass_df["exact_mw"].isna() & mass_df["exact_mw"] > 0.0]
	# compare with reported exact_mw from the mol_df
	mass_df = mass_df.merge(mol_df[["mol_id","exact_mw"]],on="mol_id",how="inner")
	diff_mass_df = mass_df[(mass_df["exact_mw_x"] - mass_df["exact_mw_y"]).abs() > 0.1]
	print("> mass inconsistencies")
	print(diff_mass_df["mol_id"].nunique(), diff_mass_df["spec_id"].nunique())
	spec_df = spec_df.drop(columns=["formula","inchikey","exact_mass"])

	# fill in missing prec_mz by inferring them
	spec_df = spec_df[~spec_df["spec_id"].isin(diff_formula_df["spec_id"])]
	spec_df = spec_df[~spec_df["spec_id"].isin(diff_mass_df["spec_id"])]
	spec_df = spec_df.merge(mol_df[["mol_id","exact_mw"]],on="mol_id",how="inner")
	spec_df.loc[:,"prec_mz"] = par_apply_df_rows(spec_df,data_utils.infer_prec_mz)
	spec_df = spec_df.drop(columns=["exact_mw"])
 
	# extract annotation info
	ann_df = ann_df.merge(spec_df[["dset_spec_id","spec_id","prec_type"]],on="dset_spec_id",how="inner")
	ann_df = ann_df.merge(mol_df[["mol_id","formula"]],on="mol_id",how="inner")
	ann_results = par_apply_df_rows(ann_df,data_utils.parse_annotations)
	ann_df.loc[:,"ann_peak_mzs"] = ann_results[0]
	ann_df.loc[:,"ann_products"] = ann_results[1]
	ann_df.loc[:,"ann_losses"] = ann_results[2]
	ann_df.loc[:,"ann_isotopes"] = ann_results[3]
	ann_df.loc[:,"ann_exact_mzs"] = ann_results[4]
	ann_df = ann_df.dropna(axis=0,how="any")
	ann_df = ann_df[ann_df["ann_peak_mzs"].apply(len) > 0]
	ann_df = ann_df.drop(columns=["notes","peaks"])
	ann_df = ann_df.reset_index(drop=True)
   
	# reset indices
	spec_df = spec_df.reset_index(drop=True)
	mol_df = mol_df.reset_index(drop=True)

	debug_dfs = {
		"no_mol_df": no_mol_df,
		"diff_formula_df": diff_formula_df,
		"diff_inchikey_df": diff_inchikey_df,
		"diff_mass_df": diff_mass_df
	}

	return spec_df, mol_df, ann_df, debug_dfs


if __name__ == "__main__":

	RDLogger.DisableLog("rdApp.*")

	parser = argparse.ArgumentParser()
	parser.add_argument("--df_dp", type=str, default="data/df")
	parser.add_argument("--proc_dp", type=str, default="data/proc/nist")
	parser.add_argument("--num_entries", type=int, default=-1)
	parser.add_argument("--ow_spec_mol", type=booltype, default=True, help="overwrite existing spec mol")
	parser.add_argument("--dsets", type=str, nargs="+", default=["nist20_hr","mona23"])
	flags = parser.parse_args()

	print(f"> df_dp: {flags.df_dp}")
	print(f"> proc_dp: {flags.proc_dp}")
	print(f"> num_entries: {flags.num_entries}")
	print(f"> force update spec and mol df: {flags.ow_spec_mol}")
	print("> dsets", flags.dsets)
 
	spec_df_fp = os.path.join(flags.proc_dp,"spec_df.pkl")
	mol_df_fp = os.path.join(flags.proc_dp,"mol_df.pkl")
	ann_df_fp = os.path.join(flags.proc_dp,"ann_df.pkl")

	if not flags.ow_spec_mol and os.path.isfile(spec_df_fp) and os.path.isfile(mol_df_fp):

		print("> loading previous spec_df, mol_df, ann_df")
		assert os.path.isdir(flags.proc_dp), flags.proc_dp
		assert os.path.isfile(spec_df_fp), spec_df_fp
		assert os.path.isfile(mol_df_fp), mol_df_fp
		spec_df = pd.read_pickle(spec_df_fp)
		mol_df = pd.read_pickle(mol_df_fp)
		ann_df = pd.read_pickle(ann_df_fp)

	else:

		print("> creating new spec_df, mol_df, ann_df")
		assert os.path.isdir(flags.df_dp), flags.df_dp
		os.makedirs(flags.proc_dp,exist_ok=True)
		all_df = load_df(flags.df_dp,flags.dsets,flags.num_entries)

		spec_df, mol_df, ann_df, debug_dfs = preprocess_spec(all_df)

		# save everything to file
		spec_df.to_pickle(spec_df_fp)
		mol_df.to_pickle(mol_df_fp)
		ann_df.to_pickle(ann_df_fp)
		print(f"> saved spec_df to {spec_df_fp}")
		print(f"> saved mol_df to {mol_df_fp}")
  
		for k, v in debug_dfs.items():
			v.to_pickle(os.path.join(flags.proc_dp,f"{k}.pkl"))
		
	print(f"> spec_df {spec_df.shape}")
	print(f"> spec_df num nan:")
	print(spec_df.isna().sum())
	print(f"> mol_df {mol_df.shape}")
	print(f"> mol_df num nan:")
	print(mol_df.isna().sum())
	print(f"> ann_df {ann_df.shape}")
	print(f"> ann_df num nan:")
	print(ann_df.isna().sum())
 
 
"""
步骤 1：去重与分子结构标准化
按dset + dset_spec_id去重（解决 MoNA 等数据集重复条目问题）；
通过 RDKit 将 SMILES→mol 对象→再转回 SMILES（标准化分子结构，统一立体化学、格式等）；
过滤无法生成 mol/SMILES 的无效条目（记录到no_mol_df）；
为每个唯一 SMILES 分配唯一mol_id（方便后续关联分子与质谱数据）。
步骤 2：拆分注释数据（ann_df）
复制注释相关核心字段（dset_spec_id/mol_id/notes/peaks/dset）到ann_df，后续专门处理注释信息。
步骤 3：质谱字段标准化与类型转换
解析peaks字符串为结构化数据，计算质谱分辨率res；
标准化仪器类型（inst_type）、碎裂模式（frag_mode）；
解析碰撞能量（col_energy）为绝对碰撞能量（ace） 和归一化碰撞能量（nce），删除原字段；
标准化前体离子类型（prec_type）、离子模式（ion_mode）；
转换prec_mz（前体质荷比）、exact_mass（精确质量）为数值类型；
将peaks中的质荷比 / 强度转为浮点数，解析保留指数（ri）为标准化格式；
筛选 spec_df 核心字段，重置spec_id（保证跨数据集唯一）。
步骤 4：生成分组 ID（group_id）
按「分子 + 前体类型 + 仪器类型 + 碎裂模式 + 谱类型 + 离子模式 + 数据集」分组，为每组分配唯一group_id（标记 “相同化合物 + 相同实验条件” 的条目）。
步骤 5：生成分子特征表（mol_df）
基于唯一 SMILES 生成mol_df，提取分子级核心特征：
标识类：简化版 InChIKey、Murcko 骨架、分子式、InChI；
质量类：常规分子量、精确分子量；
结构类：原子数、键数、电荷、是否单分子、自由基数量；
过滤无效 mol 条目，同步删除 spec_df 中对应无效mol_id的条目。
步骤 6：数据一致性校验（核心质量控制）
校验 spec_df 与 mol_df 的字段一致性，找出并输出不一致条目（保证数据无矛盾）：
分子式一致性：对比 spec_df 的formula和 mol_df 生成的formula；
InChIKey 一致性：对比 spec_df 的inchikey（前 14 位）和 mol_df 的inchikey_s；
精确质量一致性：对比 spec_df 的exact_mass和 mol_df 的exact_mw（误差 > 0.1 视为不一致）；
删除所有不一致条目，保证数据集质量。
步骤 7：补全缺失的前体质荷比（prec_mz）
基于 mol_df 的精确分子量和前体离子类型（prec_type），推断并补全 spec_df 中缺失的prec_mz值。
步骤 8：处理注释数据（ann_df）
关联 spec_df/mol_df 的信息到 ann_df；
解析注释信息，提取注释峰的 m/z、产物、损失、同位素、精确 m/z 等特征；
过滤空注释、无峰的条目，清理冗余字段后重置索引。
步骤 9：返回结果
返回标准化后的spec_df（质谱谱图信息）、mol_df（分子特征）、ann_df（注释信息），以及调试用的不一致数据（no_mol_df/diff_formula_df等）。
4. 主函数：流程控制与结果保存
解析命令行参数（数据路径、输出路径、条目数、是否覆盖已有文件、目标数据集列表）；
复用逻辑：若指定不覆盖且已有spec_df.pkl/mol_df.pkl，直接加载已有文件；否则执行load_df + preprocess_spec；
保存结果：将核心 DataFrame 保存为 pkl 文件（二进制格式，保留类型信息），调试数据也保存为 pkl；
输出统计：打印各 DataFrame 的形状、缺失值数量，方便检查数据质量。
输入输出说明
输入	输出
上一步生成的{dset}_df.csv	核心文件：spec_df.pkl（质谱谱图）、mol_df.pkl（分子特征）、ann_df.pkl（注释）
（如 nist20_hr_df.csv）	调试文件：no_mol_df.pkl、diff_formula_df.pkl 等（记录异常数据）
控制台输出：各 DataFrame 形状、缺失值统计、一致性校验结果
总结
核心目标：将初步标准化的质谱数据进一步清洗、校验、特征提取，生成spec_df（谱图）、mol_df（分子）、ann_df（注释）三个结构化核心表；
关键价值：通过分子结构标准化、数据一致性校验（分子式 / 质量 / InChIKey）、缺失值补全，保证数据集高质量、无矛盾；
应用场景：输出的 pkl 文件可直接用于质谱碎片预测、化合物识别、质谱数据挖掘等机器学习 / 数据分析任务。
 """