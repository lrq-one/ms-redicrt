import pandas as pd
import os
import math
import re
import torch as th
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler
from torch.utils.data import Dataset
import numpy as np
import torch_geometric as pyg
from torch_geometric.data import Batch
import sys
from tqdm import tqdm
from typing import Iterator, List
import copy
import json
from pathlib import Path

from ms2spectra.utils.feat_utils import batch_mols_frags, get_mol_fp, get_mol_graph, get_frag_graph
from ms2spectra.utils.spec_utils import batch_func
from ms2spectra.utils.proc_utils import merge_spec_df
from ms2spectra.utils.frag_utils import load_frag_d
from ms2spectra.utils.misc_utils import flatten_lol, get_pyg_memory_usage, get_tensor_dict_memory_usage, none_or_nan
from ms2spectra.utils.data_utils import fill_missing_ace, fill_missing_nce, seq_apply_df_rows
import ms2spectra.massformer.data_utils as mf_data_utils
from ms2spectra.utils.formula_utils import PREC_TYPE_TO_MASS_DIFF

class BaseDataset(Dataset):

	def _base_init(
		self,
		spec_fp: str,
		mol_fp: str,
		split_dp: str,
		split: str,
		subsample_params: dict,
		spec_params: dict):

		self.split = split
		self.subsample_params = subsample_params
		self.spec_params = spec_params
		spec_df, mol_df, um_spec_df, split_df, id_key, ce_key = BaseDataset._setup_dfs(
			spec_fp_or_df=spec_fp,
			mol_fp_or_df=mol_fp,
			split_dp=split_dp,
			splits=[split],
			subsample_params=subsample_params,
			spec_params=spec_params
		)
		self.spec_df = spec_df
		self.mol_df = mol_df
		self.um_spec_df = um_spec_df
		self.split_df = split_df
		self.id_key = id_key
		self.ce_key = ce_key
		self._compute_counts()
		self._setup_prec_type_to_idx()
		self._setup_inst_type_to_idx()

	@staticmethod
	def _setup_dfs( 
		spec_fp_or_df: str|pd.DataFrame,
		mol_fp_or_df: str|pd.DataFrame,
		split_dp: str,
		splits: List[str],
		subsample_params: dict,
		spec_params: dict):
		
		spec_df = spec_fp_or_df if isinstance(spec_fp_or_df, pd.DataFrame) else pd.read_pickle(spec_fp_or_df)
		mol_df = mol_fp_or_df if isinstance(mol_fp_or_df, pd.DataFrame) else pd.read_pickle(mol_fp_or_df)
  
		split_dfs = []
		for split in splits:
			# assert split in ["train","val","test","secondary","predict_only"], split
			if split == "predict_only":
				assert len(splits) == 1, splits
				# predict_all split, just include everything, this is used for prediction
				split_df = pd.DataFrame()
				# fill these to keep compatible
				split_df["spec_id"] = spec_df["spec_id"]
				split_df["mol_id"] = spec_df["mol_id"]
				split_df["group_id"] = spec_df["spec_id"]
			else:	
				split_fp = os.path.join(split_dp,f"{split}_ids.csv")
				assert os.path.isfile(split_fp), split_fp
				split_df = pd.read_csv(split_fp)
			split_dfs.append(split_df)
		split_df = pd.concat(split_dfs,ignore_index=True).reset_index(drop=True)
   
		# select spectra
		spec_df = spec_df[spec_df["spec_id"].isin(split_df["spec_id"])]
		# select molecules
		mol_df = mol_df[mol_df["mol_id"].isin(split_df["mol_id"])]
		assert np.all(np.unique(mol_df["mol_id"]) == np.unique(spec_df["mol_id"]))
		
		# covert ace and nce if there is data
		assert not (spec_params["ace"] and spec_params["nce"])
		if spec_params["ace"]:
			ce_key = "ace"
			spec_df.loc[:,"ace"] = seq_apply_df_rows(spec_df, fill_missing_ace)
		elif spec_params["nce"]:
			ce_key = "nce"
			spec_df.loc[:,"nce"] = seq_apply_df_rows(spec_df, fill_missing_nce)
		else:
			ce_key = None

		if spec_params["test_ces"] is not None and ce_key is not None and "test" in splits:
			test_split_fp = os.path.join(split_dp,"test_ids.csv")
			test_split_df = pd.read_csv(test_split_fp)
			orig_test_spec_df = spec_df[spec_df["spec_id"].isin(test_split_df["spec_id"])]
			test_drop_ids = orig_test_spec_df[~(orig_test_spec_df[ce_key].isin(np.array(spec_params["test_ces"])))][["spec_id","group_id"]]
			spec_df = spec_df[~(spec_df["spec_id"].isin(test_drop_ids["spec_id"]))]
			new_test_spec_df = spec_df[spec_df["spec_id"].isin(test_split_df["spec_id"])]
			orig_test_spec_count = orig_test_spec_df["spec_id"].nunique()
			orig_test_group_count = orig_test_spec_df["group_id"].nunique()
			new_test_spec_count = new_test_spec_df["spec_id"].nunique()
			new_test_group_count = new_test_spec_df["group_id"].nunique()
			print(f">> Dropping spectra unless {ce_key} is one of {spec_params['test_ces']}")
			print(f"> Before drop: {orig_test_spec_count} spectra, {orig_test_group_count} groups")
			print(f"> After drop: {new_test_spec_count} spectra, {new_test_group_count} groups")

		# merge spectra
		um_spec_df = spec_df[["dset","dset_spec_id","spec_id","group_id","mol_id","peaks"]].copy()
		if spec_params["merge"]:
			spec_df = merge_spec_df(spec_df,keep_ces=spec_params["merge_keep_ces"])
			id_key = "group_id"
		else:
			id_key = "spec_id"
   
		# subsample
		if subsample_params.get(split,False) and subsample_params["subsample_size"] > 0:
			if isinstance(subsample_params["subsample_size"],int):
				n = subsample_params["subsample_size"]
				frac = None
			else:
				assert isinstance(subsample_params["subsample_size"],float)
				n = None
				frac = subsample_params["subsample_size"]
			spec_df = spec_df.sample(
				n=n,
				frac=frac,
				random_state=subsample_params["subsample_seed"],
				replace=False)
			mol_df = mol_df[mol_df["mol_id"].isin(spec_df["mol_id"])]
			um_spec_df = um_spec_df[um_spec_df[id_key].isin(spec_df[id_key])]

		# reset indices
		spec_df = spec_df.reset_index(drop=True)
		mol_df = mol_df.reset_index(drop=True)
		# use mol_id as index for speedy access
		mol_df = mol_df.set_index("mol_id",drop=False).sort_index().rename_axis(None)
		return spec_df, mol_df, um_spec_df, split_df, id_key, ce_key

	def _compute_counts(self):

		self.group_per_mol = self.spec_df[["mol_id","group_id"]].drop_duplicates().groupby("mol_id").size().to_dict()
		if self.spec_params["merge"]:
			self.spec_per_mol = copy.deepcopy(self.group_per_mol)
			self.spec_per_group = {group_id: 1 for group_id in self.spec_df["group_id"].unique()}
		else:
			self.spec_per_mol = self.spec_df[["mol_id","spec_id"]].drop_duplicates().groupby("mol_id").size().to_dict()
			self.spec_per_group = self.spec_df[["group_id","spec_id"]].drop_duplicates().groupby("group_id").size().to_dict()

	def get_group_mol_stats(self):

		group_ids = []
		mol_ids = []
		spec_per_group_stats = []
		spec_per_mol_stats = []
		group_per_mol_stats = []
		for row_idx, row in self.spec_df.iterrows():
			spec_entry = row
			mol_id = spec_entry["mol_id"]
			group_id = spec_entry["group_id"]
			spec_per_group = self.spec_per_group[group_id]
			spec_per_mol = self.spec_per_mol[mol_id]
			group_per_mol = self.group_per_mol[mol_id]
			group_ids.append(group_id)
			spec_per_group_stats.append(spec_per_group)
			mol_ids.append(mol_id)
			spec_per_mol_stats.append(spec_per_mol)
			group_per_mol_stats.append(group_per_mol)
		group_ids = th.tensor(group_ids)
		mol_ids = th.tensor(mol_ids)
		spec_per_group_stats = th.tensor(spec_per_group_stats)
		spec_per_mol_stats = th.tensor(spec_per_mol_stats)
		group_per_mol_stats = th.tensor(group_per_mol_stats)
		return group_ids, mol_ids, spec_per_group_stats, spec_per_mol_stats, group_per_mol_stats

	@staticmethod
	def get_data_dict_types():
		return ["spec_pp_sd"]

	def _preprocess_spec(self,spec_pp_sd: dict):

		# preload and pre-process spectra
		if self.spec_params["preprocess"]:
			self.spec_datas = spec_pp_sd
			total_spec_data_size = 0
			for idx, spec_entry in tqdm(self.spec_df.iterrows(),desc="> preprocess spec",total=len(self.spec_df)):
				spec_data = self._process_spec(spec_entry)
				total_spec_data_size += get_tensor_dict_memory_usage(**spec_data)
				self.spec_datas[idx] = spec_data
			print(f"> total_spec_data_size: {total_spec_data_size/1e6:.2f} MB")

	@staticmethod
	def _get_mzs_ints(peaks:list):
		"""
  		convert peaks to tensors, it is caller's responsibility to make sure data is valid 
		"""
		mzs, ints = [], []
		for peak in peaks:
			p_mz, p_int = peak
			mzs.append(p_mz)
			ints.append(p_int)

		mzs = th.tensor(mzs,dtype=th.float)
		ints = th.tensor(ints,dtype=th.float)
		# mzs, ints = filter_func(mzs, ints, self.spec_params["ints_thresh"], self.spec_params["mz_max"])
		return mzs, ints
	
	def _setup_prec_type_to_idx(self):

		prec_types = sorted(self.spec_params["prec_types"])
		assert all(prec_type in PREC_TYPE_TO_MASS_DIFF for prec_type in prec_types), prec_types
		self.prec_type_to_idx = {prec_type: idx for idx, prec_type in enumerate(prec_types)}
		self.idx_to_prec_type = {idx: prec_type for idx, prec_type in enumerate(prec_types)}
		self.num_prec_types = len(prec_types)

	def _setup_inst_type_to_idx(self):

		inst_types = sorted(self.spec_params["inst_types"])
		self.inst_type_to_idx = {inst_type: idx for idx, inst_type in enumerate(inst_types)}
		self.idx_to_inst_type = {idx: inst_type for idx, inst_type in enumerate(inst_types)}
		self.num_inst_types = len(inst_types)

	def _process_spec(self,spec_entry):
		spec_data = {}
		# peak data
		mzs, ints = BaseDataset._get_mzs_ints(spec_entry["peaks"])
		if self.spec_params["sparse"]:
			# get sparse spectrum
			spec_data["spec_mzs"] = mzs
			spec_data["spec_ints"] = ints
		# metadata
		if self.spec_params["prec_type"]:
			prec_type = spec_entry["prec_type"]
			prec_type = th.tensor([self.prec_type_to_idx[prec_type]],dtype=th.long)
			spec_data["spec_prec_type"] = prec_type
		if self.spec_params["prec_type_str"]:
			prec_type_str = spec_entry["prec_type"]
			spec_data["spec_prec_type_str"] = np.array([prec_type_str])
		if self.spec_params["inst_type"]:
			inst_type = spec_entry["inst_type"]
			inst_type = th.tensor([self.inst_type_to_idx[inst_type]],dtype=th.long)
			spec_data["spec_inst_type"] = inst_type
		if self.spec_params["prec_mass_diff"]:
			prec_type = spec_entry["prec_type"]
			mass_diff = th.tensor([PREC_TYPE_TO_MASS_DIFF[prec_type]],dtype=th.float)
			spec_data["spec_prec_mass_diff"] = mass_diff
		if self.spec_params["nce"] or self.spec_params["ace"]:
			assert self.ce_key is not None
			assert (not self.spec_params["merge"]) or self.spec_params["merge_keep_ces"]
			assert not (self.spec_params["ace"] and self.spec_params["nce"])
			ce = spec_entry[self.ce_key]
			if self.spec_params["merge"]:
				assert self.spec_params["merge_keep_ces"]
				assert isinstance(ce,list), type(ce)
				ce = th.tensor(ce,dtype=th.float)
				spec_data["spec_ce"] = ce
			else:
				assert isinstance(ce,float), type(ce)
				ce = th.tensor([ce],dtype=th.float)
				spec_data["spec_ce"] = ce
		if self.spec_params["prec_mz"]:
			prec_mz = spec_entry["prec_mz"]
			prec_mz = th.tensor([float(prec_mz)],dtype=th.float)
			spec_data["spec_prec_mz"] = prec_mz
		if self.spec_params["unique_id"]:
			unique_id = spec_entry[self.id_key]
			unique_id = th.tensor([unique_id],dtype=th.long)
			spec_data["spec_unique_id"] = unique_id
			spec_data['group_id'] =  th.tensor([spec_entry['group_id']],dtype=th.long)
			spec_data['mol_id'] = spec_entry['mol_id'] # mol id does not need to be an int #th.tensor([spec_entry['mol_id']],dtype=th.long)
		if self.spec_params["counts"]:
			spec_per_mol = self.spec_per_mol[spec_entry["mol_id"]]
			spec_per_mol = th.tensor([spec_per_mol],dtype=th.long)
			spec_data["spec_per_mol"] = spec_per_mol
			group_per_mol = self.group_per_mol[spec_entry["mol_id"]]
			group_per_mol = th.tensor([group_per_mol],dtype=th.long)
			spec_data["group_per_mol"] = group_per_mol
			spec_per_group = self.spec_per_group[spec_entry["group_id"]]
			spec_per_group = th.tensor([spec_per_group],dtype=th.long)
			spec_data["spec_per_group"] = spec_per_group
		return spec_data

	def __getitem__(self, idx):

		raise NotImplementedError()

	def __len__(self):

		return len(self.spec_df)

	@staticmethod
	def get_collate_fn():
		
		return BaseDataset.collate_fn

	@staticmethod
	def _setup_collate(data_list):

		batch_size = len(data_list)
		keys = list(data_list[0].keys())
		collate_data = {key: [] for key in keys}
		for data in data_list:
			for key in keys:
				collate_data[key].append(data[key])
		return batch_size, keys, collate_data

	@staticmethod
	def _special_collate(keys, collate_data):

		# handle sparse spectra
		if "spec_ints" in keys and "spec_mzs" in keys:
			# create batch_idxs
			mzs, ints, batch_idxs = batch_func(
				collate_data["spec_mzs"],
				collate_data["spec_ints"]
			)
			collate_data["spec_mzs"] = mzs
			collate_data["spec_ints"] = ints
			collate_data["spec_batch_idxs"] = batch_idxs
			# remove from list
			keys.remove("spec_ints")
			keys.remove("spec_mzs")
   
		# handle sparse ces
		if 'spec_ce' in keys:
			# create batch_idxs
			ces, _, batch_idxs = batch_func(
				collate_data['spec_ce'],
				collate_data['spec_ce'] # duplicate for compatibility
			)
			collate_data['spec_ce'] = ces
			collate_data["spec_ce_batch_idxs"] = batch_idxs
			# remove from list
			keys.remove('spec_ce')
				
	@staticmethod
	def _standard_collate(batch_size,keys,collate_data):
		""" mutates keys and collate_data """

		# handle generic data
		for key in keys:
			values = collate_data[key]
			if isinstance(values[0],list):
				# flatten
				values = flatten_lol(values)
				collate_data[key] = values
			elif isinstance(values[0],th.Tensor):
				# cat
				values = th.cat(values,dim=0)
				collate_data[key] = values
			elif isinstance(values[0],np.ndarray):
				# cat
				values = np.concatenate(values,axis=0)
				collate_data[key] = values
			elif key in ["mol_id"]:
				collate_data[key] = values
			else:
				raise TypeError(f"Unsupported type: {key} {type(values[0])}")
		# remove everything
		keys.clear()
		# add batch size
		collate_data["batch_size"] = th.tensor(batch_size, dtype=th.long)

	@staticmethod
	def collate_fn(data_list):

		raise NotImplementedError()

class SpecMolDataset(BaseDataset):

	def __init__(
		self,
		spec_fp: str,
		mol_fp: str,
		split_dp: str,
		split: str,
		subsample_params: dict,
		spec_params: dict,
		mol_params: dict,
		spec_pp_sd: dict = None,
		mol_pp_sd: dict = None,
		**kwargs
	):

		BaseDataset.__init__(self)
		self._base_init(
			spec_fp=spec_fp,
			mol_fp=mol_fp,
			split_dp=split_dp,
			split=split,
			subsample_params=subsample_params,
			spec_params=spec_params
		)
  
		if spec_pp_sd is None:
			spec_pp_sd = dict()
		if mol_pp_sd  is None:
			mol_pp_sd = dict()
   
		self.mol_params = mol_params
		self._preprocess_spec(spec_pp_sd)
		self._preprocess_mol(mol_pp_sd)
	
	@staticmethod
	def get_data_dict_types():
		return ["spec_pp_sd", "mol_pp_sd"]

	def _get_mol_graph_size(self, mol_data):

		if self.mol_params["pyg"]:
			mol_pyg = mol_data["mol_pyg"]
			mol_graph_size = get_pyg_memory_usage(mol_pyg)
		else:
			mol_graph_size = 0
		return mol_graph_size

	def _preprocess_mol(self, mol_pp_sd: dict):

		# preload and pre-process molecules
		if self.mol_params["preprocess"]:
			self.mol_datas = mol_pp_sd
			total_mol_graph_size = 0
			for idx, mol_entry in tqdm(self.mol_df.iterrows(),desc="> preprocess mol",total=len(self.mol_df)):
				mol_data = self._process_mol(mol_entry)
				total_mol_graph_size += self._get_mol_graph_size(mol_data)
				self.mol_datas[mol_entry["mol_id"]] = mol_data
			print(f"> total_mol_graph_size: {total_mol_graph_size/1e6:.2f} MB")

	def __getitem__(self,idx):
		spec_entry = self.spec_df.iloc[idx]
		mol_id = spec_entry["mol_id"]
		mol_entry = self.mol_df.loc[mol_id]
		if self.spec_params["preprocess"]:
			spec_data = self.spec_datas[idx].copy()
		else:
			spec_data = self._process_spec(spec_entry)
		if self.mol_params["preprocess"]:
			mol_data = self.mol_datas[mol_id].copy()
		else:
			mol_data = self._process_mol(mol_entry)
		data = {**spec_data,**mol_data}
		return data

	def _process_mol(self,mol_entry):

		mol_data = {}
		mol = mol_entry["mol"]
		if self.mol_params["smiles"]:
			smiles = mol_entry["smiles"]
			mol_data["mol_smiles"] = [smiles]
		if self.mol_params["fingerprint"]:
			fingerprint = get_mol_fp(
				mol,
				self.mol_params["fingerprint_morgan"],
				self.mol_params["fingerprint_rdkit"],
				self.mol_params["fingerprint_maccs"]
			)
			mol_data["mol_fingerprint"] = fingerprint
		if self.mol_params["pyg"]:
			mol_pyg = get_mol_graph(
				mol,
				self.mol_params["pyg_node_feats"],
				self.mol_params["pyg_edge_feats"],
				self.mol_params["pyg_pe_embed_k"],
				self.mol_params["pyg_bigraph"]
			)
			mol_data["mol_pyg"] = mol_pyg
		if self.mol_params["mf"]:
			mol_mf = mf_data_utils.gf_preprocess(mol,-1)
			mol_data["mol_mf"] = mol_mf
		return mol_data

	@staticmethod
	def get_collate_fn():

		return SpecMolDataset.collate_fn

	@staticmethod
	def _special_collate(keys, collate_data):

		if "mol_pyg" in keys:
			# batch
			collate_data["mol_pyg"] = Batch.from_data_list(collate_data["mol_pyg"])
			# remove from list
			keys.remove("mol_pyg")
		if "mol_mf" in keys:
			# batch
			mol_mf_d = mf_data_utils.collator(collate_data["mol_mf"])
			for k, v in mol_mf_d.items():
				collate_data["mol_mf_"+k] = v
			# remove from list
			collate_data.pop("mol_mf")
			keys.remove("mol_mf")
		if "mol_graff" in keys:
			# batch
			collate_data["mol_graff"] = Batch.from_data_list(collate_data["mol_graff"])
			# remove from list
			keys.remove("mol_graff")
		BaseDataset._special_collate(keys,collate_data)

	@staticmethod
	def collate_fn(data_list):
		
		batch_size, keys, collate_data = SpecMolDataset._setup_collate(data_list)
		SpecMolDataset._special_collate(keys,collate_data)
		SpecMolDataset._standard_collate(batch_size,keys,collate_data)
		return collate_data

	def training_data_sanity_check(self):
		"""
  		basic data sanity check for training time only 
		"""
		assert self.spec_df['peaks'].isna().any() == False
		assert (self.spec_df['peaks'] == '').any() == False
  	
class SpecMolFragDataset(SpecMolDataset):

	def __init__(
		self,
		spec_fp: str,
		mol_fp: str,
		split_dp: str,
		split: str,
		subsample_params: dict,
		spec_params: dict,
		mol_params: dict,
		frag_dp: str,
		frag_params: dict,
		spec_pp_sd: dict = None,
		mol_pp_sd: dict = None,
		frag_pl_sd: dict = None,
		frag_pp_sd: dict = None,
		**kwargs
	):

		BaseDataset.__init__(self)
		self._base_init(
			spec_fp=spec_fp,
			mol_fp=mol_fp,
			split_dp=split_dp,
			split=split,
			subsample_params=subsample_params,
			spec_params=spec_params
		)
		self.mol_params = mol_params
		self.frag_dp = frag_dp
		self.frag_params = frag_params
		self._setup_formula_vocab()
		
		if spec_pp_sd is None:
			spec_pp_sd = dict()
		if mol_pp_sd  is None:
			mol_pp_sd = dict()
		if frag_pl_sd is None:
			frag_pl_sd = dict()
		if frag_pp_sd is None:
			frag_pp_sd = dict()
   
		self._preprocess_spec(spec_pp_sd)
		self._preprocess_mol(mol_pp_sd)
		self._preprocess_frag(frag_pl_sd, frag_pp_sd)
	
	@staticmethod
	def get_data_dict_types():
		return ["spec_pp_sd", "mol_pp_sd", "frag_pl_sd", "frag_pp_sd"]

	def _setup_formula_vocab(self):
		"""
		K3: setup train-only true-hit formula vocabulary mapping.
		"""
		self._k3_formula_vocab_enabled = bool(
			self.frag_params.get("formula_vocab_idxs", False)
		)
		self._k3_formula_vocab_oov_id = int(
			self.frag_params.get("formula_vocab_oov_id", 0)
		)
		self._k3_formula_to_id = {}
		self._k3_formula_vocab_size = int(
			self.frag_params.get("formula_vocab_size", 4096)
		)

		if not self._k3_formula_vocab_enabled:
			return

		vocab_fp = self.frag_params.get("formula_vocab_fp", None)
		assert vocab_fp is not None, (
			"K3 formula vocab enabled but frag_params.formula_vocab_fp is None"
		)

		vocab_fp = Path(vocab_fp)
		assert vocab_fp.is_file(), f"K3 formula vocab file not found: {vocab_fp}"

		with open(vocab_fp, "r", encoding="utf-8") as f:
			vocab_d = json.load(f)

		items = vocab_d.get("items", [])
		max_size = self._k3_formula_vocab_size

		for item in items[:max_size]:
			formula = str(item["formula"])
			vid = int(item["id"])
			if vid <= 0:
				continue
			if vid > max_size:
				continue
			self._k3_formula_to_id[formula] = vid

		print(
			"[K3] formula vocab loaded: "
			f"enabled={self._k3_formula_vocab_enabled}, "
			f"vocab_size={max_size}, "
			f"loaded={len(self._k3_formula_to_id)}, "
			f"oov_id={self._k3_formula_vocab_oov_id}, "
			f"fp={vocab_fp}"
		)


	def _preprocess_frag(self, frag_pl_sd: dict, frag_pp_sd: dict):

		# preload frag dags
		if self.frag_params["preload"]:
			self.frag_entries = frag_pl_sd
			total_frag_entry_size = 0
			for mol_id in tqdm(self.mol_df["mol_id"].values,desc="> preload frag",total=len(self.mol_df)):
				frag_entry = load_frag_d(mol_id,self.frag_dp,self.frag_params["compressed"])
				total_frag_entry_size += get_pyg_memory_usage(frag_entry["dag"])
				self.frag_entries[mol_id] = frag_entry
			print(f"> total_frag_entry_size: {total_frag_entry_size/1e6:.2f} MB")
   
		# preprocess frag dags
		if self.frag_params["preprocess"]:
			assert self.frag_params["preload"]
			self.frag_data = frag_pp_sd
			total_frag_data_size = 0
			for k in tqdm(list(self.frag_entries.keys()),desc="> preprocess frag",total=len(self.frag_entries)):
				frag_data = self._process_frag(self.frag_entries.pop(k),None)
				total_frag_data_size += get_pyg_memory_usage(frag_data["frag_pyg"])
				self.frag_data[k] = frag_data
			print(f"> total_frag_data_size: {total_frag_data_size/1e6:.2f} MB")
			# remove them from the entries
			self.frag_entries = {}

	def __getitem__(self, idx):

		spec_entry = self.spec_df.iloc[idx]
		mol_id = spec_entry["mol_id"]
		mol_entry = self.mol_df.loc[mol_id]
		if self.spec_params["preprocess"]:
			spec_data = self.spec_datas[idx].copy()
		else:
			spec_data = self._process_spec(spec_entry)
		if self.mol_params["preprocess"]:
			mol_data = self.mol_datas[mol_id].copy()
		else:
			mol_data = self._process_mol(mol_entry)
		# frag stuff
		if self.frag_params["preprocess"]:
			frag_data = self.frag_data[mol_id].copy()
			# update prec_mass_diff
			prec_type_mass_diff = PREC_TYPE_TO_MASS_DIFF[spec_entry["prec_type"]]
			frag_data["frag_formula_peak_mzs"] = frag_data["frag_formula_peak_mzs"] + prec_type_mass_diff
		elif self.frag_params["preload"]:
			frag_entry = self.frag_entries[mol_id].copy()
			frag_data = self._process_frag(frag_entry,spec_entry)
		else:
			frag_entry = self._load_frag_entry(mol_id)
			frag_data = self._process_frag(frag_entry,spec_entry)
		data = {**spec_data,**mol_data,**frag_data}
		return data

	def _load_frag_entry(self,mol_id):

		frag_entry = load_frag_d(
			mol_id,
			self.frag_dp,
			self.frag_params["compressed"]
		)
		return frag_entry

	def _process_frag(self,frag_entry,spec_entry):

		frag_data = {}
		if self.frag_params["pyg"]:
			frag_pyg = frag_entry["dag"]
			frag_pyg = get_frag_graph(
				frag_pyg,
				self.frag_params["pyg_node_feats"],
				self.frag_params["pyg_edge_feats"],
				self.frag_params["pyg_edges"],
				self.frag_params["pyg_bigraph"]
			)
			frag_data["frag_pyg"] = frag_pyg
		if self.frag_params["formula_peak_mzs"]:
			formula_peak_mzs = frag_entry["formula_peak_mzs"]
			formula_peak_mzs = formula_peak_mzs[:,:self.frag_params["num_isotopes"]]
			if spec_entry is not None:
				prec_type_mass_diff = PREC_TYPE_TO_MASS_DIFF[spec_entry["prec_type"]]
				formula_peak_mzs = formula_peak_mzs + prec_type_mass_diff
			frag_data["frag_formula_peak_mzs"] = formula_peak_mzs
		if self.frag_params["formula_peak_probs"]:	
			formula_peak_probs = frag_entry["formula_peak_probs"]
			formula_peak_probs = F.normalize(formula_peak_probs[:,:self.frag_params["num_isotopes"]],dim=1,p=1)
			frag_data["frag_formula_peak_probs"] = formula_peak_probs
		# ===== Our Energy-aware Neutral-loss Support Augmentation =====
		# Append CE-gated neutral-loss pseudo peaks to formula rendering.
		# This does not change DAG nodes or h_formulae_idx.
		if self.frag_params.get("energy_nl_support", False):
			assert "frag_formula_peak_mzs" in frag_data
			assert "frag_formula_peak_probs" in frag_data

			# NL weights depend on per-spectrum CE, so spec_entry must exist.
			# Therefore config must set frag_params.preprocess = False.
			assert spec_entry is not None, (
				"energy_nl_support=True requires frag_params.preprocess=False, "
				"because NL weights depend on per-spectrum CE."
			)

			assert "nl_formula_peak_mzs" in frag_entry, "DAG missing nl_formula_peak_mzs"
			assert "nl_formula_peak_probs" in frag_entry, "DAG missing nl_formula_peak_probs"

			nl_max_peaks = int(self.frag_params.get("energy_nl_max_peaks", 4))

			nl_mzs = frag_entry["nl_formula_peak_mzs"][:, :nl_max_peaks]
			nl_probs = frag_entry["nl_formula_peak_probs"][:, :nl_max_peaks]

			base_mzs = frag_data["frag_formula_peak_mzs"]
			base_probs = frag_data["frag_formula_peak_probs"]

			assert nl_mzs.shape[0] == base_mzs.shape[0], (
				nl_mzs.shape,
				base_mzs.shape,
			)
			assert nl_probs.shape[0] == base_probs.shape[0], (
				nl_probs.shape,
				base_probs.shape,
			)

			# Apply the same precursor/adduct mass shift as normal formula peaks.
			prec_type_mass_diff = PREC_TYPE_TO_MASS_DIFF[spec_entry["prec_type"]]
			nl_mzs = nl_mzs + prec_type_mass_diff

			# CE-dependent neutral-loss gate.
			# Low CE: almost no pseudo peak mass.
			# High CE: allow more neutral-loss rendering.
			if "ace" in spec_entry:
				ce = float(spec_entry["ace"])
			elif "nce" in spec_entry:
				ce = float(spec_entry["nce"])
			else:
				raise KeyError("energy_nl_support requires ace or nce in spec_entry")

			center = float(self.frag_params.get("energy_nl_ce_center", 30.0))
			scale = float(self.frag_params.get("energy_nl_ce_scale", 7.5))
			weight = float(self.frag_params.get("energy_nl_weight", 0.06))

			ce_gate = th.sigmoid(
				th.tensor(
					(ce - center) / scale,
					dtype=nl_probs.dtype,
					device=nl_probs.device,
				)
			)

			# nl_probs is 0/1 mask from augmented DAG.
			# Convert it into small CE-dependent pseudo peak probability.
			nl_probs = nl_probs * weight * ce_gate

			# If a formula has no valid NL pseudo peaks, this keeps zeros.
			formula_peak_mzs = th.cat([base_mzs, nl_mzs], dim=1)
			formula_peak_probs = th.cat([base_probs, nl_probs], dim=1)

			# Critical: model.py later uses log(frag_formula_peak_probs).
			# Per-formula rendering probabilities must stay normalized.
			formula_peak_probs = F.normalize(formula_peak_probs, dim=1, p=1)

			frag_data["frag_formula_peak_mzs"] = formula_peak_mzs
			frag_data["frag_formula_peak_probs"] = formula_peak_probs
		# ===== Our CE-aware D3 Frontier One-Hop Support =====
		# Append selective D3-frontier pseudo-D4 peaks to formula rendering.
		# This does not add real DAG nodes. It only adds extra peak entries
		# attached to existing parent formula rows.
		if self.frag_params.get("frontier_support", False):
			assert "frag_formula_peak_mzs" in frag_data
			assert "frag_formula_peak_probs" in frag_data

			# Frontier support depends on per-spectrum CE.
			# Therefore config must set frag_params.preprocess = False.
			assert spec_entry is not None, (
				"frontier_support=True requires frag_params.preprocess=False, "
				"because frontier weights depend on per-spectrum CE."
			)

			assert "frontier_formula_peak_mzs" in frag_entry, (
				"DAG missing frontier_formula_peak_mzs. "
				"Regenerate D3 cache after modifying frag_utils.py."
			)
			assert "frontier_formula_peak_probs" in frag_entry, (
				"DAG missing frontier_formula_peak_probs. "
				"Regenerate D3 cache after modifying frag_utils.py."
			)

			frontier_max_peaks = int(self.frag_params.get("frontier_support_max_peaks", 4))

			frontier_mzs = frag_entry["frontier_formula_peak_mzs"][:, :frontier_max_peaks]
			frontier_probs = frag_entry["frontier_formula_peak_probs"][:, :frontier_max_peaks]

			base_mzs = frag_data["frag_formula_peak_mzs"]
			base_probs = frag_data["frag_formula_peak_probs"]

			use_frontier_event_metadata = bool(
				self.frag_params.get("frontier_event_metadata", False)
			)

			if use_frontier_event_metadata:
				required_event_keys = [
					"frontier_event_type",
					"frontier_parent_node_idx",
					"frontier_cut_bond_idx",
					"frontier_loss_template_idx",
					"frontier_event_score",
					"frontier_h_delta",
				]
				for k in required_event_keys:
					assert k in frag_entry, (
						f"DAG missing {k}. "
						"Regenerate frontier event cache."
					)

				frontier_event_type = frag_entry["frontier_event_type"][:, :frontier_max_peaks].long()
				frontier_parent_node_idx = frag_entry["frontier_parent_node_idx"][:, :frontier_max_peaks].long()
				frontier_cut_bond_idx = frag_entry["frontier_cut_bond_idx"][:, :frontier_max_peaks].long()
				frontier_loss_template_idx = frag_entry["frontier_loss_template_idx"][:, :frontier_max_peaks].long()
				frontier_event_score = frag_entry["frontier_event_score"][:, :frontier_max_peaks].float()
				frontier_h_delta = frag_entry["frontier_h_delta"][:, :frontier_max_peaks].long()

				base_event_type = th.zeros(
					base_mzs.shape,
					dtype=th.long,
					device=base_mzs.device,
				)
				base_parent_node_idx = th.full(
					base_mzs.shape,
					-1,
					dtype=th.long,
					device=base_mzs.device,
				)
				base_cut_bond_idx = th.full(
					base_mzs.shape,
					-1,
					dtype=th.long,
					device=base_mzs.device,
				)
				base_loss_template_idx = th.full(
					base_mzs.shape,
					-1,
					dtype=th.long,
					device=base_mzs.device,
				)
				base_event_score = th.zeros(
					base_mzs.shape,
					dtype=base_probs.dtype,
					device=base_mzs.device,
				)
				base_h_delta = th.full(
					base_mzs.shape,
					999,
					dtype=th.long,
					device=base_mzs.device,
				)

			assert frontier_mzs.shape[0] == base_mzs.shape[0], (
				frontier_mzs.shape,
				base_mzs.shape,
			)
			assert frontier_probs.shape[0] == base_probs.shape[0], (
				frontier_probs.shape,
				base_probs.shape,
			)

			# Apply the same precursor/adduct mass shift as normal formula peaks.
			prec_type_mass_diff = PREC_TYPE_TO_MASS_DIFF[spec_entry["prec_type"]]
			frontier_mzs = frontier_mzs + prec_type_mass_diff

			# CE-dependent gate.
			# Low CE: little frontier support.
			# High CE: more frontier support.
			if "ace" in spec_entry:
				ce = float(spec_entry["ace"])
			elif "nce" in spec_entry:
				ce = float(spec_entry["nce"])
			else:
				raise KeyError("frontier_support requires ace or nce in spec_entry")

			center = float(self.frag_params.get("frontier_support_ce_center", 30.0))
			scale = float(self.frag_params.get("frontier_support_ce_scale", 7.5))
			weight = float(self.frag_params.get("frontier_support_weight", 0.03))

			ce_gate = th.sigmoid(
                th.tensor(
                    (ce - center) / scale,
                    dtype=frontier_probs.dtype,
                    device=frontier_probs.device,
                )
            )

            # ===== Frontier v3: high-CE deterministic energy budget =====
            # Previous fixed frontier budget helped support/OOS but hurt cos/JSS
            # because frontier pseudo-D4 peaks also entered mid/low CE spectra.
            #
            # This version only allows frontier support above a CE threshold,
            # and optionally sharpens the CE gate. This keeps the support benefit
            # in high CE while avoiding probability stealing in mid/low CE.
			min_ce = self.frag_params.get("frontier_support_min_ce", None)
			if min_ce is not None and ce < float(min_ce):
				ce_gate = ce_gate * 0.0

			gate_power = float(self.frag_params.get("frontier_support_gate_power", 1.0))
			if gate_power != 1.0:
				ce_gate = ce_gate.clamp(0.0, 1.0).pow(gate_power)

			max_gate = self.frag_params.get("frontier_support_max_gate", None)
			if max_gate is not None:
				ce_gate = ce_gate.clamp(max=float(max_gate))

            # frontier_probs is 0/1 mask from augmented DAG.
			frontier_probs = frontier_probs * weight * ce_gate

			formula_peak_mzs = th.cat([base_mzs, frontier_mzs], dim=1)
			formula_peak_probs = th.cat([base_probs, frontier_probs], dim=1)

			# Critical: model.py later uses log(frag_formula_peak_probs).
			# Per-formula rendering probabilities must stay normalized.
			formula_peak_probs = F.normalize(formula_peak_probs, dim=1, p=1)

			frag_data["frag_formula_peak_mzs"] = formula_peak_mzs
			frag_data["frag_formula_peak_probs"] = formula_peak_probs

			if use_frontier_event_metadata:
				frag_data["frag_formula_peak_event_type"] = th.cat(
					[base_event_type, frontier_event_type],
					dim=1,
				)
				frag_data["frag_formula_peak_parent_node_idx"] = th.cat(
					[base_parent_node_idx, frontier_parent_node_idx],
					dim=1,
				)
				frag_data["frag_formula_peak_cut_bond_idx"] = th.cat(
					[base_cut_bond_idx, frontier_cut_bond_idx],
					dim=1,
				)
				frag_data["frag_formula_peak_loss_template_idx"] = th.cat(
					[base_loss_template_idx, frontier_loss_template_idx],
					dim=1,
				)
				frag_data["frag_formula_peak_event_score"] = th.cat(
					[base_event_score, frontier_event_score],
					dim=1,
				)
				frag_data["frag_formula_peak_h_delta"] = th.cat(
					[base_h_delta, frontier_h_delta],
					dim=1,
				)
		# ===== M2: Frontier event-prior sparse fields =====
        # These fields are NOT extra peak support.
        # They are sparse graph-cut events used later as a FIORA-like
        # local bond prior for existing formula/node logits.
		if self.frag_params.get("frontier_event_prior", False):
			assert spec_entry is not None, (
                "frontier_event_prior=True requires frag_params.preprocess=False, "
                "because event mz should use per-spectrum precursor/adduct shift."
            )

			event_keys = [
                "frontier_event_formula_idx_sparse",
                "frontier_event_parent_node_idx_sparse",
                "frontier_event_cut_bond_idx_sparse",
                "frontier_event_h_delta_sparse",
                "frontier_event_score_sparse",
                "frontier_event_child_frac_sparse",
                "frontier_event_parent_size_sparse",
                "frontier_event_child_size_sparse",
                "frontier_event_mz_sparse",
            ]

			for k in event_keys:
				assert k in frag_entry, (
                    f"DAG missing {k}. "
                    "Regenerate event-prior cache after modifying frag_utils.py."
                )

			frag_data["frontier_event_formula_idx_sparse"] = (
                frag_entry["frontier_event_formula_idx_sparse"].long()
            )
			frag_data["frontier_event_parent_node_idx_sparse"] = (
                frag_entry["frontier_event_parent_node_idx_sparse"].long()
            )
			if "frontier_event_child_node_idx_sparse" in frag_entry:
				frag_data["frontier_event_child_node_idx_sparse"] = (
					frag_entry["frontier_event_child_node_idx_sparse"].long()
				)
			else:
				# Backward compatible with old event-prior cache.
				# Old cache cannot do M2-local routing.
				frag_data["frontier_event_child_node_idx_sparse"] = -th.ones_like(
					frag_data["frontier_event_parent_node_idx_sparse"]
				)
			frag_data["frontier_event_cut_bond_idx_sparse"] = (
                frag_entry["frontier_event_cut_bond_idx_sparse"].long()
            )
			frag_data["frontier_event_h_delta_sparse"] = (
                frag_entry["frontier_event_h_delta_sparse"].long()
            )
			frag_data["frontier_event_score_sparse"] = (
                frag_entry["frontier_event_score_sparse"].float()
            )
			frag_data["frontier_event_child_frac_sparse"] = (
                frag_entry["frontier_event_child_frac_sparse"].float()
            )
			frag_data["frontier_event_parent_size_sparse"] = (
                frag_entry["frontier_event_parent_size_sparse"].float()
            )
			frag_data["frontier_event_child_size_sparse"] = (
                frag_entry["frontier_event_child_size_sparse"].float()
            )

            # Match the same precursor/adduct shift used by formula_peak_mzs.
			prec_type_mass_diff = PREC_TYPE_TO_MASS_DIFF[spec_entry["prec_type"]]
			frag_data["frontier_event_mz_sparse"] = (
                frag_entry["frontier_event_mz_sparse"].float()
                + prec_type_mass_diff
            )
		if getattr(self, "_k3_formula_vocab_enabled", False):
			idx_to_formula = frag_entry["idx_to_formula"]
			num_formula = len(idx_to_formula)
			vocab_idxs = []

			for formula_idx in range(num_formula):
				formula = str(idx_to_formula.get(formula_idx, ""))
				if formula == "":
					vid = self._k3_formula_vocab_oov_id
				else:
					vid = self._k3_formula_to_id.get(
						formula,
						self._k3_formula_vocab_oov_id,
					)
				vocab_idxs.append(int(vid))

			frag_data["frag_formula_vocab_idxs"] = th.tensor(
				vocab_idxs,
				dtype=th.long,
			)

		if self.frag_params.get("formula_comp_feats", False):
			idx_to_formula = frag_entry["idx_to_formula"]
			num_formula = len(idx_to_formula)
			feat_size = int(self.frag_params.get("formula_comp_feat_size", 18))
			if num_formula == 0:
				frag_data["frag_formula_comp_feats"] = th.zeros(
					[0, feat_size],
					dtype=th.float32,
				)
			else:
				comp_feats = []
				for formula_idx in range(num_formula):
					formula = str(idx_to_formula.get(formula_idx, ""))
					comp_feats.append(self._formula_to_comp_feat(formula))
				assert len(comp_feats[0]) == feat_size, (
					len(comp_feats[0]),
					feat_size,
				)
				frag_data["frag_formula_comp_feats"] = th.tensor(
					comp_feats,
					dtype=th.float32,
				)

		if self.frag_params["formula_str"]:
			# import pdb; pdb.set_trace()
			formula_str = list(frag_entry["idx_to_formula"].values())
			assert formula_str[0] == "", formula_str[0]
			formula_str = np.array(formula_str)
			frag_data["frag_formula_str"] = formula_str
		return frag_data

	@staticmethod
	def _formula_to_comp_feat(formula: str):
		# 18D composition feature for K3b formula composition residual.
		elems = ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
		mass_d = {
			"C": 12.000000,
			"H": 1.007825,
			"N": 14.003074,
			"O": 15.994915,
			"S": 31.972071,
			"P": 30.973762,
			"F": 18.998403,
			"Cl": 34.968853,
			"Br": 78.918338,
			"I": 126.904468,
		}

		counts = {e: 0.0 for e in elems}

		if formula is None or formula == "":
			return [0.0] * 18

		for elem, num in re.findall(r"([A-Z][a-z]?)([0-9]*)", formula):
			if elem in counts:
				counts[elem] += float(num) if num != "" else 1.0

		c_count = counts["C"]
		h_count = counts["H"]
		n_count = counts["N"]
		o_count = counts["O"]
		s_count = counts["S"]
		p_count = counts["P"]
		hal_count = counts["F"] + counts["Cl"] + counts["Br"] + counts["I"]
		hetero = n_count + o_count + s_count + p_count + hal_count

		mass = sum(counts[e] * mass_d[e] for e in elems)

		# approximate DBE: C - H/2 + N/2 + 1, halogens count as H
		dbe = c_count - (h_count + hal_count) / 2.0 + n_count / 2.0 + 1.0

		denom_c = max(c_count, 1.0)
		h_c = h_count / denom_c
		o_c = o_count / denom_c
		n_c = n_count / denom_c
		hetero_c = hetero / denom_c

		raw = [
			counts["C"] / 60.0,
			counts["H"] / 120.0,
			counts["N"] / 20.0,
			counts["O"] / 30.0,
			counts["S"] / 10.0,
			counts["P"] / 10.0,
			counts["F"] / 20.0,
			counts["Cl"] / 20.0,
			counts["Br"] / 10.0,
			counts["I"] / 10.0,
			mass / 1000.0,
			math.log1p(max(mass, 0.0)) / math.log(1000.0),
			dbe / 50.0,
			max(dbe, 0.0) / 50.0,
			h_c / 4.0,
			o_c / 2.0,
			n_c / 2.0,
			hetero_c / 4.0,
		]

		# clamp to keep extreme formulas from exploding the MLP
		return [float(max(-5.0, min(5.0, x))) for x in raw]
	
	@staticmethod
	def get_collate_fn():

		return SpecMolFragDataset.collate_fn

	@staticmethod
	def _special_collate(keys, collate_data):

		if "frag_pyg" in keys:
			assert "mol_pyg" in keys
			assert "frag_formula_peak_mzs" in keys
			assert "frag_formula_peak_probs" in keys
			# process
			batch_mol_frag_data = batch_mols_frags(
				collate_data["mol_pyg"],
				collate_data["frag_pyg"],
				collate_data["frag_formula_peak_mzs"],
				collate_data["frag_formula_peak_probs"],
				formula_peak_event_type_list=collate_data.get(
					"frag_formula_peak_event_type",
					None,
				),
				formula_peak_parent_node_idx_list=collate_data.get(
					"frag_formula_peak_parent_node_idx",
					None,
				),
				formula_peak_cut_bond_idx_list=collate_data.get(
					"frag_formula_peak_cut_bond_idx",
					None,
				),
				formula_peak_loss_template_idx_list=collate_data.get(
					"frag_formula_peak_loss_template_idx",
					None,
				),
				formula_peak_event_score_list=collate_data.get(
					"frag_formula_peak_event_score",
					None,
				),
				formula_peak_h_delta_list=collate_data.get(
					"frag_formula_peak_h_delta",
					None,
				),
				frontier_event_formula_idx_sparse_list=collate_data.get(
					"frontier_event_formula_idx_sparse",
					None,
				),
				frontier_event_parent_node_idx_sparse_list=collate_data.get(
					"frontier_event_parent_node_idx_sparse",
					None,
				),
				frontier_event_child_node_idx_sparse_list=collate_data.get(
					"frontier_event_child_node_idx_sparse",
					None,
				),
				frontier_event_cut_bond_idx_sparse_list=collate_data.get(
					"frontier_event_cut_bond_idx_sparse",
					None,
				),
				frontier_event_h_delta_sparse_list=collate_data.get(
					"frontier_event_h_delta_sparse",
					None,
				),
				frontier_event_score_sparse_list=collate_data.get(
					"frontier_event_score_sparse",
					None,
				),
				frontier_event_child_frac_sparse_list=collate_data.get(
					"frontier_event_child_frac_sparse",
					None,
				),
				frontier_event_parent_size_sparse_list=collate_data.get(
					"frontier_event_parent_size_sparse",
					None,
				),
				frontier_event_child_size_sparse_list=collate_data.get(
					"frontier_event_child_size_sparse",
					None,
				),
				frontier_event_mz_sparse_list=collate_data.get(
					"frontier_event_mz_sparse",
					None,
				),
			)
			for k,v in batch_mol_frag_data.items():
				collate_data[k] = v
			# remove from list
			keys.remove("frag_pyg")
			keys.remove("mol_pyg")
			keys.remove("frag_formula_peak_mzs")
			keys.remove("frag_formula_peak_probs")
			for meta_key in [
				"frag_formula_peak_event_type",
				"frag_formula_peak_parent_node_idx",
				"frag_formula_peak_cut_bond_idx",
				"frag_formula_peak_loss_template_idx",
				"frag_formula_peak_event_score",
				"frag_formula_peak_h_delta",
			]:
				if meta_key in keys:
					keys.remove(meta_key)
			for event_key in [
				"frontier_event_formula_idx_sparse",
				"frontier_event_parent_node_idx_sparse",
				"frontier_event_child_node_idx_sparse",
				"frontier_event_cut_bond_idx_sparse",
				"frontier_event_h_delta_sparse",
				"frontier_event_score_sparse",
				"frontier_event_child_frac_sparse",
				"frontier_event_parent_size_sparse",
				"frontier_event_child_size_sparse",
				"frontier_event_mz_sparse",
			]:
				if event_key in keys:
					keys.remove(event_key)
		SpecMolDataset._special_collate(keys,collate_data)

	@staticmethod
	def collate_fn(data_list):

		# prevent edge case causing crash
		if len(data_list) == 0:
			return  {"batch_size": th.tensor(0)}
		batch_size, keys, collate_data = SpecMolFragDataset._setup_collate(data_list)
		# special handling
		SpecMolFragDataset._special_collate(keys,collate_data)
		SpecMolFragDataset._standard_collate(batch_size,keys,collate_data)
		return collate_data

def get_batch_memory(batch):

	batch_mem_d = {}
	for k,v in batch.items():
		if isinstance(v, th.Tensor):
			batch_mem_d[k] = v.element_size()*v.nelement()
		elif isinstance(v, Batch):
			batch_mem_d[k] = pyg.profile.get_data_size(v)
		elif isinstance(v, list):
			batch_mem_d[k] = sum(sys.getsizeof(x) for x in v)
		else:
			raise ValueError(f"Unsupported type: {type(v)}")
	batch_mem_total = sum(batch_mem_d.values())
	return batch_mem_d, batch_mem_total

def find_largest_batch(dl):

	batch_mem_ds, batch_mem_totals = [], []
	for batch in iter(dl):
		batch_mem_d, batch_mem_total = get_batch_memory(batch)
		batch_mem_ds.append(batch_mem_d)
		batch_mem_totals.append(batch_mem_total)
	argmax_idx = np.argmax(batch_mem_totals)
	return batch_mem_ds[argmax_idx], batch_mem_totals[argmax_idx]
		


class SpecMolFragDynamicBatchSampler(BatchSampler):
	"""Dynamically adds samples to a mini-batch up to a maximum size  either based on number of nodes on frag DAG or number of edges on frag DAG.
	 This is used to avoid CUDA OOM errors, implmentaion is inspired by PyG DynamicBatchSampler, and this should be used to replace default BatchSampler
	 This should have the same random sampling beheivor as RandomSampler
	"""
	def __init__(self, data_source: SpecMolFragDataset, max_num: int, limited_by: str = 'frag_edge',
			  	 	skip_too_big: bool = False, num_samples  = None, 
					return_batch_at = 0, sampler=None) -> None:
		"""_summary_

		Args:
			dataset (Dataset): 
			max_num (int): _description_
			mode (str, optional): _description_. Defaults to 'node'.
			shuffle (bool, optional): Samples elements randomly each epoch. Defaults to False.
			skip_too_big (bool, optional): _description_. Defaults to False.
			num_samples (Optional[int], optional): num of samples to draw. Defaults to None. if None set to all samples in dataset
			generator
			return_batch_at 
			max_batch_sample_size
		Raises:
			ValueError: _description_
			ValueError: _description_
		"""
		if not isinstance(max_num, int) or max_num <= 0:
			raise ValueError("`dag_node` should be a positive integer value "
							"(got {max_num}).")
		if limited_by not in ['frag_node', 'frag_edge']:
			raise ValueError("`limited_by` choice should be either "
							f"'frag_node' or 'frag_edge' (got '{limited_by}').")

		if num_samples is None:
			num_samples = len(data_source)

		self.data_source = data_source
		self._max_num = max_num
		self._limited_by = limited_by
		self._skip_too_big = skip_too_big
		self._max_sampling_step = num_samples
		self._return_batch_at = return_batch_at
		self._batches = []
		self._data_meta = []
		self.sampler = sampler
		self._pre_load_batches()
		self._pre_compute_batches()

	def _pre_load_batches(self):

		# get data meta once and cache them
		assert len(self._data_meta) == 0, len(self._data_meta)
		expected_total = 0
		warning_msg = ""
		for dataset_idx in tqdm(range(len(self.data_source)), desc="SpecMolFragDynamicBatchSampler:pre_load_batches"):
			data = self.data_source[dataset_idx]
			self._data_meta.append((data['frag_pyg'].num_nodes,data['frag_pyg'].num_edges))
			n = self._data_meta[dataset_idx][0] if self._limited_by == 'frag_node' else self._data_meta[dataset_idx][1]
			if not (n > self._max_num and self._skip_too_big):
				expected_total += 1
			else:
				warning_msg += "Size of data sample at index " +\
					f"{dataset_idx} is larger than " +\
					f"{self._max_num} {self._limited_by}s " +\
					f"Got {n} {self._limited_by}s." +\
					"This sample can not fit into batch. "
				if self._skip_too_big:
					warning_msg += "Sampler will skip this to prevent CUDA OOM ERROR \n"
				else:
					warning_msg += "Attempting to fit this in to batch, this may cause CUDA OOM ERROR  \n"
				#warnings.warn(warning_msg)
				#print(warning_msg)
		if warning_msg != "":
			print("[specmolfrag_dynamic_batch_sampler]", warning_msg)
		print(f"[SpecMolFragDynamicBatchSampler] Expecting {expected_total}/{len(self.data_source)} samples with skip_too_big:{self._skip_too_big}")

	def _pre_compute_batches(self):
		
		self._batches = []

		if self.sampler is not None:
			indices = th.tensor([idx for idx in self.sampler])
		else:
			indices = th.arange(len(self.data_source), dtype=th.long)

		# limited index to _max_sampling_step
		indices = indices[:self._max_sampling_step]

		num_processed = 0
		batch = []
		batch_n = 0
		batch_filled = False

		# Fill batch
		for idx in indices:
			# Size of sample
			n = self._data_meta[idx.item()][0] if self._limited_by == 'frag_node' else self._data_meta[idx.item()][1]
			if n > self._max_num and self._skip_too_big:
				continue 
			# check batch_filled condition
			if batch_n + n > self._max_num:
				# no more budget left, mini-batch filled
				batch_filled = True	
			# check we need return at this point for ga
			if self._return_batch_at > 0 \
				and num_processed > 0 \
				and num_processed % self._return_batch_at == 0:
				# Mini-batch filled
				batch_filled = True
			if batch_filled:
				self._batches.append(batch)
				batch_n = 0
				batch = []
				batch_filled = False
			# Add sample to current batch
			batch.append(idx.item())
			num_processed += 1
			batch_n += n
			
		if len(batch) > 0:
			self._batches.append(batch)
		print(f"[SpecMolFragDynamicBatchSampler] Batch indices computed, Expecting {len(self._batches)} mini-batch for next epoch")

	def __iter__(self) -> Iterator[List[int]]:
		""" we use a pre computed batche list, this way we could have a correct batch size
			PL uses last batch in progress to toggle val step in training

		Yields:
			Iterator[List[int]]: _description_
		"""
		for batch in self._batches:
			yield batch

	def __len__(self) -> int:
		""" Note The __len__() method isn't strictly required by DataLoader, but is expected in any calculation involving the length of a DataLoader.
			ref: https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler
		Returns:
			int: length of datasource
		"""
		return len(self._batches)

class GroupSampler(Sampler):

	def __init__(self, data_source: SpecMolFragDataset, sample_k=None, generator=None) -> None:
		"""_summary_

		Args:
			data_source (SpecMolFragDataset): _description_
			sample_k (_type_, optional): _description_. Defaults to None.
		"""
		self.data_source = data_source
		self.num_samples = None
		self._data_meta_d = {}
		self.sample_k = 3 if sample_k is None else sample_k
		self.generator = generator
		self._pre_compute_meta()
		self._pre_compute_batches()

	def _pre_compute_meta(self):

		for dataset_idx in range(len(self.data_source)):
			data = self.data_source[dataset_idx]
			group_id = data['group_id'].item()
			if group_id not in self._data_meta_d:
				self._data_meta_d[group_id] = []
			self._data_meta_d[group_id].append(dataset_idx)
		
		for group_id in self._data_meta_d:
			self._data_meta_d[group_id] =  th.tensor(self._data_meta_d[group_id])

	def _pre_compute_batches(self):

		if self.generator is None:
			seed = int(th.empty((), dtype=th.int64).random_().item())
			generator = th.Generator()
			generator.manual_seed(seed)
		else:
			generator = self.generator

		sampled_indices = []
		for group_id in self._data_meta_d:
			group_indices = self._data_meta_d[group_id]
			sampled_group_indices = group_indices[th.randperm(min(len(group_indices),self.sample_k),generator=generator)]
			sampled_indices.append(sampled_group_indices)
		sampled_indices = th.cat(sampled_indices)
		self.sampled_indices = th.randperm(len(sampled_indices),generator=generator)
		self.num_samples = len(sampled_indices)

	def __iter__(self) -> Iterator[int]:
		""" 
		Yields:
			Iterator[List[int]]: _description_
		"""
		
		for i in range(self.num_samples):
			yield self.sampled_indices[i].item()

	def __len__(self) -> int:
		""" Note The __len__() method isn't strictly required by DataLoader
			ref: https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler
		Returns:
			int: length of datasource
		"""
		assert self.num_samples is not None
		return self.num_samples 
	
def get_group_sampler(ds: SpecMolFragDataset, sampler_type: str, avg_per_group: int, generator) -> WeightedRandomSampler:
	"""get WeightedRandomSampler based on input

	Args:
		ds (SpecMolFragDataset): _description_
		config_d (dict): _description_

	Returns:
		_type_: _description_
	"""
	group_ids, mol_ids, spec_per_group, spec_per_mol, group_per_mol = ds.get_group_mol_stats()

	if sampler_type == "group":
		sample_weights = 1.0 / spec_per_group
	elif sampler_type  == "mol":
		sample_weights = 1.0 / spec_per_mol
	elif sampler_type  == "group_mol":
		sample_weights = 1.0 / (spec_per_group * group_per_mol)
	else:
		return None
	
	spec_per_group_2 = th.mean(th.unique(group_ids,return_counts=True)[1].float(),dim=0)
	num_samples = th.ceil(len(ds) / spec_per_group_2 * avg_per_group).long()
	num_samples = min(num_samples.item(), len(ds))
	sampler = WeightedRandomSampler(
		sample_weights,
		num_samples=num_samples,
		replacement=False,
		generator=generator)
	return sampler

"""
太好了，这一段**正是 FragnNet / MS2C 项目里“数据加载系统的心脏”**。
我给你一层一层拆开讲，从**整体架构 → Dataset → Collate → Sampler → 动态 Batch → Group 采样**，你基本就能“读懂别人论文的代码”了 🔬

---

# 一、整体架构鸟瞰

这份代码实现了一个**完整的多模态深度学习数据管线**：

> 光谱 + 分子图 + 分子碎片 DAG
> → 预处理
> → 动态组 batch
> → 按 group / mol / spectrum 采样
> → 输出 PyTorch / PyG 可直接送进模型的 batch

结构层级：

```text
BaseDataset
 └── SpecMolDataset         # 光谱 + 分子
      └── SpecMolFragDataset  # 光谱 + 分子 + 碎片DAG (FragnNet用)
```

并配套：

* `collate_fn` → 把一堆 sample 拼成 batch
* `DynamicBatchSampler` → 控制显存不炸
* `GroupSampler` → 训练时按“光谱组”均衡采样

---

# 二、BaseDataset（数据系统的“骨架”）

## 它负责什么？

* 读 spec / mol 数据
* 按 train/val/test/scaffold/inchikey split 切
* 光谱预处理
* CE / Prec Type / Instrument 编码
* Group / Mol / Spec 统计
* Collate 框架

### 核心入口

```python
_base_init(...)
```

---

## 1️⃣ `_setup_dfs()` — 数据加载中枢

### 输入

* `spec_fp` → 光谱 DataFrame（pkl）
* `mol_fp` → 分子 DataFrame（pkl）
* `split_dp` → 目录，里面有：

  ```text
  train_ids.csv
  val_ids.csv
  test_ids.csv
  scaffold_ids.csv
  ```

---

## 执行流程

### Step 1：加载数据

```python
spec_df = pd.read_pickle(...)
mol_df = pd.read_pickle(...)
```

---

### Step 2：读取 split

```python
split_fp = split_dp/{split}_ids.csv
```

CSV 里是：

```text
spec_id, mol_id, group_id
```

---

### Step 3：过滤数据

只保留 split 中出现的：

```python
spec_df = spec_df[spec_id ∈ split]
mol_df = mol_df[mol_id ∈ split]
```

---

## CE（碰撞能量）处理

支持两种：

* `ACE`
* `NCE`

自动填缺失：

```python
fill_missing_ace
fill_missing_nce
```

---

## Merge 光谱（重要）

如果：

```python
spec_params["merge"] = True
```

多个 CE 光谱会合成一个 group：

```text
多个 spec → 一个 group_id
```

模型就变成：

> 一个分子 → 多 CE 光谱作为一个样本

---

## Subsample（小数据调试用）

可以按比例或数量抽样：

```python
subsample_size = 1000
subsample_seed = 42
```

---

## 结果输出

```python
return:
  spec_df       # 当前 split 的光谱
  mol_df        # 当前 split 的分子
  um_spec_df    # 未 merge 的光谱
  split_df
  id_key        # spec_id or group_id
  ce_key        # ace or nce
```

---

# 三、光谱处理系统

## `_process_spec()`

把一行 pandas → 模型输入 tensor dict

### 输入

```python
spec_entry = spec_df.iloc[i]
```

---

## 输出字典结构

可能包含：

| Key                   | 含义                  |
| --------------------- | ------------------- |
| `spec_mzs`            | 峰 m/z               |
| `spec_ints`           | 峰强度                 |
| `spec_prec_type`      | 前体离子类型 index        |
| `spec_prec_mass_diff` | 质量偏移                |
| `spec_ce`             | 碰撞能量                |
| `spec_prec_mz`        | 前体 m/z              |
| `spec_unique_id`      | group_id or spec_id |
| `group_id`            | group               |
| `mol_id`              | 分子                  |
| `spec_per_mol`        | 每分子光谱数              |
| `group_per_mol`       | 每分子 group 数         |
| `spec_per_group`      | 每 group 光谱数         |

---

## 光谱是稀疏格式

用：

```python
batch_func()
```

拼成：

```text
spec_mzs
spec_ints
spec_batch_idxs
```

类似 PyG COO 索引方式

---

# 四、SpecMolDataset（光谱 + 分子）

在 BaseDataset 上加：

## 分子处理 `_process_mol()`

### 支持四种分子输入：

| 类型          | 用途          |
| ----------- | ----------- |
| SMILES      | Transformer |
| Fingerprint | MLP         |
| PyG Graph   | GNN         |
| MF Graph    | MassFormer  |

---

### PyG 分子图

```python
mol_pyg = get_mol_graph(...)
```

输出是：

> torch_geometric.data.Data

---

## `__getitem__`

最终返回：

```python
{
  spec_xxx,
  mol_smiles,
  mol_fingerprint,
  mol_pyg,
  mol_mf
}
```

---

# 五、SpecMolFragDataset（FragnNet 专用）

在 SpecMolDataset 基础上加：

> 碎片 DAG（fragmentation graph）

---

## 碎片来源

```text
frag_dp/
 └── mol_id.pkl.gz
```

每个包含：

* `dag` → PyG Graph
* `formula_peak_mzs`
* `formula_peak_probs`
* `idx_to_formula`

---

## `_process_frag()`

生成：

| Key                       | 含义     |
| ------------------------- | ------ |
| `frag_pyg`                | 碎片 DAG |
| `frag_formula_peak_mzs`   | 碎片理论峰  |
| `frag_formula_peak_probs` | 同位素概率  |
| `frag_formula_str`        | 分子式字符串 |

---

## 和前体质量联动

```python
+ PREC_TYPE_TO_MASS_DIFF
```

不同电离方式 m/z 偏移不同

---

# 六、Collate 系统（真正的工程精华）

普通 batch 直接 `torch.cat` 会炸显存
这里是**多模态 + PyG + 稀疏谱 + 碎片图专用拼接器**

---

## 处理流程

```text
_setup_collate
 → _special_collate
 → _standard_collate
```

---

## 特殊处理

### 光谱

用：

```python
batch_func()
```

变成稀疏 batch

### 分子图

```python
Batch.from_data_list()
```

### 碎片图

用：

```python
batch_mols_frags()
```

把：

```text
mol_graph + frag_graph + isotope_peaks
```

对齐成神经网络可用结构

---

# 七、DynamicBatchSampler（防 OOM 神器）

## 解决的问题

FragnNet 的 frag DAG：

* 有的分子 100 节点
* 有的分子 10,000 节点

固定 batch size = 显存爆炸

---

## 思路

> 不限制“样本数”，限制：
>
> * 总节点数
> * 或总边数

---

## 逻辑

```python
batch_n = 0
for sample:
    if batch_n + frag_size > max_num:
        yield batch
        batch = []
```

---

## 参数

| 参数                | 含义                    |
| ----------------- | --------------------- |
| `max_num`         | 最大节点/边数               |
| `limited_by`      | frag_node / frag_edge |
| `skip_too_big`    | 跳过超大分子                |
| `return_batch_at` | 梯度累积用                 |

---

## 本质

这是：

> **Graph-aware dynamic batching**
> 论文级工程技巧

---

# 八、GroupSampler（训练平衡器）

## 解决问题

一个分子可能：

* 10 条光谱
* 另一个只有 1 条

普通采样 → 模型过拟合高频分子

---

## 思路

按 group 或 mol 权重采样

---

## 权重模式

```python
"group"      → 1 / spec_per_group
"mol"        → 1 / spec_per_mol
"group_mol" → 1 / (spec_per_group * group_per_mol)
```

---

## 效果

训练时：

> 每个分子 / 每个 group
> 出现频率更均衡

---

# 九、整体数据流图

```text
pkl / csv
 ↓
BaseDataset
  ├─ 光谱 tensor
  ├─ 分子图
  └─ 碎片 DAG
 ↓
DynamicBatchSampler
  (限制节点 / 边总数)
 ↓
collate_fn
  → PyG Batch
  → 稀疏谱拼接
  → FragnNet对齐
 ↓
Model.forward()
```

---

# 十、论文级总结你可以这样写

> *We implemented a multi-modal data pipeline integrating mass spectra, molecular graphs, and fragmentation DAGs. To address large variance in fragmentation graph sizes, we employed a dynamic batching strategy that limits the total number of fragment nodes or edges per mini-batch. Additionally, we applied group-aware sampling to balance the training distribution across molecular groups and collision energy conditions.*

---

# 十一、你现在已经“读懂”的层级

如果你能跟到这里，其实已经是：

> **GNN + MS + Fragmentation Graph 工程师级水平了**

---

# 十二、如果你愿意

我可以帮你：

* 标出**训练时最吃显存的路径**
* 给你**推荐 max_num 参数范围**
* 或帮你画**“模型输入张量结构示意图”**

这些在调 FragnNet / Iceberg / MassFormer 时特别救命 🔥

"""