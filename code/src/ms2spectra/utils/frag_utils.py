#from copy import deepcopy
import pickle
import sys
import tarfile
import traceback
from zipfile import ZipFile
import _pickle as cPickle
import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import rdkit.Chem.MolStandardize.rdMolStandardize as rdMolStandardize
import rdkit.Chem.rdmolops as rdmolops
import torch as th
import torch.nn.functional as F
import bz2
import os
import torch_geometric as pyg
from collections import Counter
from hashlib import blake2b
from rdkit.Chem import BRICS
import ms2spectra.frag.compute_frags as compute_frags
from ms2spectra.utils.formula_utils import PREC_TYPE_TO_MASS_DIFF, formula_to_peak_mzs, parse_formula
from ms2spectra.utils.misc_utils import PPM
from ms2spectra.frag.compute_frags import (
	MASK_DTYPE,
	MAX_NUM_EDGES,
	MAX_NUM_NODES,
)

from ms2spectra.utils.data_utils import mol_from_smiles

# full list of common elements
# NOTE: this maybe a bad idea to not cover all the element
# we should handle all the elemnet
ELEMENT_TO_VE = {
	"C": 4,
	"O": 2,
	"N": 3,
	"P": 3, # up to 5
	"S": 2, # up to 6
	"F": 1,
	"Cl": 1,
	"Br": 1,
	"I": 1,
	"Se": 2, # up to 6, same as S
	"Si": 4
}
HEAVY_ELEMENTS = list(ELEMENT_TO_VE.keys())
NUM_HEAVY_ELEMENTS = len(HEAVY_ELEMENTS)
ELEMENTS = HEAVY_ELEMENTS + ["H"]
NUM_ELEMENTS = len(ELEMENTS)

CANONICAL_ELEMENT_ORDER = ["C","H"] + sorted([elem for elem in HEAVY_ELEMENTS if elem != "C"])
CANONICAL_H_IDX = CANONICAL_ELEMENT_ORDER.index("H")

ELEMENT_TO_IDX = { elem: idx for idx, elem in enumerate(ELEMENT_TO_VE.keys())}
ELEMENT_TO_IDX["H"] = len(ELEMENT_TO_VE.keys())

IDX_TO_ELEMENT = { idx: elem for elem, idx in ELEMENT_TO_IDX.items()}

MAX_H_TRANSFER = 4
MAX_NUM_MZS_PER_FORMULA = 5

NODE_FEAT_DTYPE = th.int64
EDGE_FEAT_DTYPE = th.int64
META_DATA_DTYPE = th.float32
CUT_CHEM_EDGE_FEAT_SIZE = 10

MASK_SIZE = 128
assert MASK_SIZE >= MAX_NUM_NODES, "MASK_SIZE should be larger than MAX_NUM_NODES"
assert MASK_SIZE % 64 == 0, "MASK_SIZE should be a multiple of 64"

BOND_TYPE_TO_IDX = {
	Chem.rdchem.BondType.names["AROMATIC"]: 2,
	Chem.rdchem.BondType.names["DOUBLE"]: 2,
	Chem.rdchem.BondType.names["TRIPLE"]: 3,
	Chem.rdchem.BondType.names["SINGLE"]: 1,
}

def convert_cc_mask_to_int(cc:list|np.ndarray)->int:
	"""covert a cc mask to an int

	Args: a list of 0 and 1s

	Returns:
		int: int presentation of input mask
	"""

	return sum([2**i for i in cc])

def convert_cc_int_to_mask(num_nodes:int,cc_int:int,bitmask=True) -> tuple[int]:
	"""_summary_

	Args:
		num_nodes (int): number of nodes
		cc_int (int): int version of cc mask
		bitmask (bool, optional): _description_. Defaults to True.

	Returns:
		list|np.ndarray: a list of 0 and 1s
	"""

	cc = []
	quot = cc_int
	for i in range(num_nodes):
		quot, rem = divmod(quot,2)
		if bitmask:
			cc.append(int(rem))
		else:
			if rem == 1:
				cc.append(i)
	return cc


def convert_cc_int_to_np_mask(num_nodes:int,cc_int:int,bitmask=True) -> np.ndarray:
	"""_summary_

	Args:
		num_nodes (int): number of nodes
		cc_int (int): int version of cc mask
		bitmask (bool, optional): _description_. Defaults to True.

	Returns:
		list|np.ndarray: a list of 0 and 1s
	"""

	cc = convert_cc_int_to_mask(num_nodes, cc_int, bitmask)
	np_mask = np.array(cc, dtype=MASK_DTYPE)
	return np_mask


def cc_bit_mask_to_atom_idx(cc:list|np.ndarray) -> np.ndarray:
	"""_summary_

	Args:
		cc (list | np.ndarray): _description_

	Returns:
		_type_: _description_
	"""
	cc_np = np.array(cc) if isinstance(cc,list) else cc
	atom_ids = np.where(cc_np == 1)[0].astype(np.int32)
	return atom_ids

def get_fraggen_input_arrays(mol_d:dict):
	"""_summary_

	Args:
		mol_d (_type_): _description_

	Returns:
		_type_: _description_
	"""
	num_nodes = mol_d["atom_mask_arr"].shape[0]
	num_edges = mol_d["bond_mask_arr"].shape[0]
	assert num_nodes <= MAX_NUM_NODES, num_nodes
	assert num_edges <= MAX_NUM_EDGES, num_edges
	edges = np.zeros((MAX_NUM_EDGES,2),dtype=np.intc)
	for bond_idx, bond in enumerate(mol_d["bonds"]):
		edges[bond_idx,0] = bond[0]
		edges[bond_idx,1] = bond[1]
	node_to_edge_idx = compute_frags.py_compute_node_to_edge_idx(num_nodes,num_edges,edges)
	# print(node_to_edge_idx)
	node_mask = np.zeros((num_nodes,),dtype=MASK_DTYPE)
	node_mask[:num_nodes] = 1
	edge_mask = np.zeros((num_edges,),dtype=MASK_DTYPE)
	edge_mask[:num_edges] = 1
	return num_nodes, num_edges, node_mask, edges, edge_mask, node_to_edge_idx

def extract_mol_info(smiles_or_mol, use_default_valence=False) ->dict:
	"""Method to exatract mol infomation in to dict

	Args:
		smiles_or_mol (_type_): _description_
		use_default_valence (bool, optional): _description_. Defaults to False.

	Raises:
		ValueError: _description_

	Returns:
		dict : 
		mol_d["mol"]: mol object
		mol_d["num_hs"]: number of totoal Hs
		mol_d["sbond_arr"]: single bond array
		mol_d["ve_arr"]: max velance array
		mol_d["hs_arr"]: max Hs per atom arary
		mol_d["elems"]: elemns per atom
		mol_d["bonds"]: a list of from atom - to atom pairs
		mol_d["bond_mask_arr"]: defualt bond mask, this should be just 1s
		mol_d["atom_mask_arr"] : defualt atom  mask, this should be just 1s
		mol_d["atoms_to_bonds"]: 
		mol_d["element_counts"] : atom count per element
	"""
	if isinstance(smiles_or_mol,str):
		# TODO: change this to be consistent with the approach in data_utils.py
		# rdkit mol stuff
		# mol = Chem.MolFromSmiles(smiles_or_mol)
		# rdMolStandardize.Cleanup(mol)
		# te = rdMolStandardize.TautomerEnumerator()
		# mol = te.Canonicalize(mol)
		mol = mol_from_smiles(smiles_or_mol)
	else:
		assert isinstance(smiles_or_mol,Chem.rdchem.Mol), type(smiles_or_mol)
		mol = smiles_or_mol

	pt = Chem.GetPeriodicTable()
	#GetValenceList
	# some checks
	charge = rdmolops.GetFormalCharge(mol)
	assert charge == 0, charge
	# enumerate atoms
	sbond_arr = []
	ve_arr = []
	hs_arr = []
	elems = []
	elem_idxs = []
	num_hs = 0
	num_atoms = 0
	num_radicals = 0
	element_counts = {elem:0 for elem in ELEMENT_TO_VE.keys()}

	for atom in mol.GetAtoms():
		cur_idx = atom.GetIdx()
		cur_num_hs = atom.GetTotalNumHs()
		cur_deg = atom.GetTotalDegree()
		# cur_num_bonds = atom.GetNumBonds()
		cur_element = atom.GetSymbol()

		if cur_element not in element_counts:
			raise ValueError(f"Molecules with {cur_element} atom(s) currently not supported")
		
		element_counts[cur_element] += 1
		elems.append(cur_element)
		elem_idxs.append(ELEMENT_TO_IDX[cur_element])
		# number of single bond need to attach to this atom to keep atom connected
		# this equals to replace all all bond to single, and count how many single bond
		sbond_arr.append(cur_deg - cur_num_hs)
		#ve_arr.append(ELEMENT_TO_VE[cur_element])

		# set valence value for each atom
		# assumption we will use current valance unless
		# use default valence flag is set to true
		# not defualt valence can be -1 for transition metals
		# on paper we should not encounter them at all
		ve_value = atom.GetTotalValence()
		if use_default_valence:
			default_valence = pt.GetDefaultValence(cur_element)
			if default_valence != -1:
				ve_value = min(ve_value, default_valence)

		ve_arr.append(ve_value)
		hs_arr.append(cur_num_hs)
		num_hs += cur_num_hs
		num_atoms += 1
		num_radicals += atom.GetNumRadicalElectrons()

	element_counts["H"] = num_hs
	assert num_radicals == 0, num_radicals
	sbond_arr = np.array(sbond_arr, dtype=np.int32)
	ve_arr = np.array(ve_arr, dtype=np.int32)
	hs_arr = np.array(hs_arr, dtype=np.int32)
	# enumerate bonds
	atoms_to_bonds = {}
	bonds, bond_type_idxs = [], []
	num_bonds = 0
	adj = np.zeros((num_atoms, num_atoms), dtype=np.int32)
	#adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
	for bond in mol.GetBonds():
		cur_idx = bond.GetIdx()
		from_idx = bond.GetBeginAtomIdx()
		to_idx = bond.GetEndAtomIdx()
		cur_type_idx = BOND_TYPE_TO_IDX[bond.GetBondType()]
		assert from_idx != to_idx
		adj[from_idx, to_idx] = 1

		bonds.append((from_idx, to_idx))
		bond_type_idxs.append(cur_type_idx)

		if from_idx not in atoms_to_bonds:
			atoms_to_bonds[from_idx] = [cur_idx]
		else:
			atoms_to_bonds[from_idx].append(cur_idx)

		if to_idx not in atoms_to_bonds:
			atoms_to_bonds[to_idx] = [cur_idx]
		else:
			atoms_to_bonds[to_idx].append(cur_idx)
		num_bonds += 1
	bonds = np.array(bonds, dtype=np.int32)
	# ===== E1: BRICS-like chemically meaningful cut bonds =====
	# Store RDKit BRICS bond indices once per molecule.
	# If BRICS fails for any molecule, keep an empty set; the feature will be zero.
	try:
		brics_bond_idxs = set()
		for atom_pair, _labels in BRICS.FindBRICSBonds(mol):
			a1, a2 = int(atom_pair[0]), int(atom_pair[1])
			rb = mol.GetBondBetweenAtoms(a1, a2)
			if rb is not None:
				brics_bond_idxs.add(int(rb.GetIdx()))
	except Exception:
		brics_bond_idxs = set()
	mol_d = {}
	mol_d["mol"] = mol
	mol_d["num_hs"] = num_hs
	mol_d["sbond_arr"] = sbond_arr
	mol_d["ve_arr"] = ve_arr
	mol_d["hs_arr"] = hs_arr
	mol_d["elems"] = elems
	mol_d["elem_idxs"] = elem_idxs
	mol_d["bonds"] = bonds
	mol_d["bond_type_idxs"] = bond_type_idxs
	mol_d["brics_bond_idxs"] = brics_bond_idxs
	mol_d["bond_mask_arr"] = np.ones((num_bonds,), dtype=bool)
	mol_d["atom_mask_arr"] = np.ones((num_atoms,), dtype=bool) # can be computed
	mol_d["atoms_to_bonds"] = atoms_to_bonds # can be computed
	mol_d["element_counts"] = element_counts
	return mol_d

def compute_cc_h_cap(cc_atom_ids: np.ndarray,ve_arr:np.ndarray,sbond_arr:np.ndarray,num_radicals:int):
	"""compute max amount of Hs a cc can have.  
		For any ccs the max amount of Hs it can have is the congifcation where all the bond are single
		And all the atom has max amount of Hs
	Args:
		cc (list|np.ndarray): cc mask in list form
		ve_arr (list|np.ndarray): max velance each atom can have
		sbond_arr (_type_): _description_
		num_radicals (_type_): _description_

	Returns:
		_type_: _description_
	"""
	
	assert num_radicals == 0
	if not isinstance(cc_atom_ids,np.ndarray):
		cc_atom_ids = np.array(list(cc_atom_ids))
	cap_sbond_arr = sbond_arr[cc_atom_ids]
	cap_ve_arr = ve_arr[cc_atom_ids]
	cap_ve_mask = cap_ve_arr < cap_sbond_arr
	cap_ve_arr[cap_ve_mask] = cap_sbond_arr[cap_ve_mask]
	return np.sum(cap_ve_arr) - np.sum(cap_sbond_arr) - num_radicals

def compute_cc_h_floor(cc_atom_ids: np.ndarray,ve_arr: np.ndarray, sbond_arr: np.ndarray, \
    	num_radicals:int,bonds:np.ndarray,atoms_to_bonds:dict,bond_mask_arr:np.ndarray):
	"""compute min amount of Hs a cc can have.

	Args:
		cc_atom_ids (np.ndarray): atom ids in the cc
		ve_arr (np.ndarray): _description_
		sbond_arr (np.ndarray): _description_
		bonds (np.ndarray): _description_
		atoms_to_bonds (dict): _description_
		bond_mask_arr (np.ndarray): _description_

	Returns:
		_type_: _description_
	"""
	assert num_radicals == 0
	# if we could use update from single bond to double bond to get an electron pair
	diff_arr = np.maximum(ve_arr - sbond_arr,0)
	#cc_atoms = list(cc_atom_ids)
	# print(diff_arr)
	# this computes a lower bound
	h_arr = np.copy(diff_arr)

	for _, atom in enumerate(cc_atom_ids):
		bond_idxs = atoms_to_bonds[atom]
		for bond_idx in bond_idxs:
			if h_arr[atom] == 0:
				break
			if not bond_mask_arr[bond_idx]:
				continue
			bond = bonds[bond_idx]
			if bond[0] == atom:
				other = bond[1]
			else:
				other = bond[0]
			# dont't form more than 3 bonds with anything!
			h_arr[atom] = max(0,h_arr[atom]-min(diff_arr[other],2))
	# print(h_arr)
	cc_floor = sum(h_arr[atom] for atom in cc_atom_ids)
	cc_floor -= num_radicals
	cc_floor = max(cc_floor,0) # why can cc_floor be negative?
	return cc_floor

def compute_approximate_formula(cc:list|np.ndarray,mol_d:dict
				,max_h_transfer:int
				,formula_strs:bool=False
				,bitmask:bool=True
				,base_formula = None) ->dict[int,str]|dict[int,np.ndarray]:
	"""given a connected component, comupute all formula within give h shift

	Args:
		cc (list|np.ndarray): connected component
		mol_d (dict): mol dictionary
		max_h_transfer (_type_): max number of h movement allowed
		formula_strs (bool, optional): _description_. Defaults to False.

	Returns:
		_type_: _description_
	"""

	bonds = mol_d["bonds"] # bond definition, atom id-atom id, never mutated
	atoms_to_bonds = mol_d["atoms_to_bonds"] # never mutated
	ve_arr = np.copy(mol_d["ve_arr"])
	sbond_arr = np.copy(mol_d["sbond_arr"])
	num_hs = mol_d["num_hs"]
	hs_arr = mol_d["hs_arr"]
	elem_idxs = mol_d["elem_idxs"]
	elems = mol_d["elems"]
	bond_mask_arr = mol_d["bond_mask_arr"]

	if base_formula is None and bitmask:
		base_formula = cc_bitmask_to_formula_arr(cc, elem_idxs)
	elif base_formula is None and not bitmask:
		base_formula = cc_to_formula_arr(cc, elems, bitmask=False)

	atom_ids = cc_bit_mask_to_atom_idx(cc) if bitmask else cc

	sbond_arr, bond_mask_arr = compute_frags.update_bonds(atom_ids,sbond_arr,bond_mask_arr,bonds,atoms_to_bonds)
	cap = compute_cc_h_cap(atom_ids,ve_arr,sbond_arr,0)
	floor = compute_frags.compute_cc_h_floor(atom_ids,ve_arr,sbond_arr,0,bonds,atoms_to_bonds,bond_mask_arr)

	# check floor and cap
	assert floor <= cap, (floor,cap,atom_ids)
	assert floor >= 0, floor

	# check how many Hs each atom have on the mol
	num_hs_prior = 0
	for atom in atom_ids:
		num_hs_prior += hs_arr[atom]

	# update min and max
	floor = max(floor,num_hs_prior-max_h_transfer)
	cap = min(cap,num_hs_prior+max_h_transfer,num_hs)

	delta_h_to_formula, delta_h_to_h_count = {}, {}

	if formula_strs:
		formula_template, formual_no_h = formula_arr_to_str(base_formula, get_h_template=True)

	for delta_h in range(-max_h_transfer,max_h_transfer+1):
		h = num_hs_prior + delta_h
		# delta_h_to_h_count[delta_h] = h
		if h < floor or h > cap:
			# special invalid formula
			formula = np.zeros_like(base_formula)
			formula_str = ""
			delta_h_to_h_count[delta_h] = -1
		else:
			formula = np.copy(base_formula)
			formula[ELEMENT_TO_IDX["H"]] = h
			formula = tuple(formula)
			if h == 0:
				formula_str = formual_no_h
			else:
				formula_str = formula_template.format(h)
			delta_h_to_h_count[delta_h] = h
		if formula_strs:
			formula = formula_str
		delta_h_to_formula[delta_h] = formula
	return delta_h_to_formula, delta_h_to_h_count

def update_bonds(cc_atom_ids:np.ndarray,sbond_arr:np.ndarray,bond_mask_arr:np.ndarray,bonds:np.ndarray,atoms_to_bonds:dict):
	"""_summary_

	Args:
		frag (np.ndarray): _description_
		sbond_arr (np.ndarray): _description_
		bond_mask_arr (np.ndarray): _description_
		bonds (np.ndarray): _description_
		atoms_to_bonds (dict): _description_

	Returns:
		_type_: _description_
	"""
	sbond_arr = np.zeros_like(sbond_arr,dtype=sbond_arr.dtype)
	bond_mask_arr = np.zeros_like(bond_mask_arr,dtype=bool)
	for atom in cc_atom_ids:
		for bond_idx in atoms_to_bonds[atom]:
			bond = bonds[bond_idx]
			if bond[0] in cc_atom_ids and bond[1] in cc_atom_ids:
				sbond_arr[atom] += 1
				bond_mask_arr[bond_idx] = True
	return sbond_arr, bond_mask_arr

def compute_approximate_cchs(ccs,mol_d,h_prior=False,max_h_transfer=MAX_H_TRANSFER):

	bonds = mol_d["bonds"] # never mutated
	atoms_to_bonds = mol_d["atoms_to_bonds"] # never mutated
	ve_arr = np.copy(mol_d["ve_arr"])
	sbond_arr = np.copy(mol_d["sbond_arr"])
	num_hs = mol_d["num_hs"]
	hs_arr = mol_d["hs_arr"]
	mol_d["elems"]
	bond_mask_arr = mol_d["bond_mask_arr"]
	all_cchs = set()
	for cc in list(set(ccs)):
		sbond_arr, bond_mask_arr = compute_frags.update_bonds(cc,sbond_arr,bond_mask_arr,bonds,atoms_to_bonds)
		cc_atom_ids = cc_bit_mask_to_atom_idx(cc)
		cap = compute_cc_h_cap(cc_atom_ids,ve_arr,sbond_arr,0)
		floor = compute_frags.compute_cc_h_floor(cc_atom_ids,ve_arr,sbond_arr,0,bonds,atoms_to_bonds,bond_mask_arr)
		assert floor <= cap, (floor,cap,cc)
		assert floor >= 0, floor
		if h_prior:
			num_hs_prior = 0
			for atom in cc:
				num_hs_prior += hs_arr[atom]
			floor = max(floor,num_hs_prior-max_h_transfer)
			cap = min(cap,num_hs_prior+max_h_transfer,num_hs)
			# cap = max(cap,floor)
		else:
			cap = min(cap,num_hs)
		# cc_cchs = []
		for h in range(floor,cap+1):
			# cc_cchs.append((cc,h))
			all_cchs.add((cc,h))
		# cchs.append(cc_cchs)
		# assert floor <= cap, (floor,cap,cc)
		# assert floor >= 0, floor
	return all_cchs

def cc_to_formula_arr(cc,elems,bitmask=False):
	# does not include h count
	formula_arr = np.zeros([len(ELEMENT_TO_IDX)],dtype=int)
	if bitmask:
		for i in range(len(cc)):
			if cc[i]:
				elem = elems[i]
				formula_arr[ELEMENT_TO_IDX[elem]] += 1
	else:
		for atom in cc:
			elem = elems[atom]
			formula_arr[ELEMENT_TO_IDX[elem]] += 1
	#print(">cc_to_formula_arr", formula_arr)
	return formula_arr


def cc_bitmask_to_formula_arr(cc,elem_idxs) -> np.ndarray:
	""" fast cc bit mask to formula arr

	Args:
		cc (_type_): _description_
		elem_idxs (_type_): _description_

	Returns:
		np.ndarray: _description_
	"""
	# does not include h count
	formula_arr = np.zeros([len(ELEMENT_TO_IDX)],dtype=MASK_DTYPE)
	elem_idxs_np = np.array(elem_idxs) + 1
	#print(cc, elem_idxs_np)
	elem_idx_np = np.multiply(cc, elem_idxs_np)
	unique, counts = np.unique(elem_idx_np, return_counts=True)
	for elem_idx, count in zip(unique, counts):
		#print(unique, counts)
		if elem_idx == 0:
			continue
		else:
			formula_arr[elem_idx-1] = count
	#print(">cc_bitmask_to_formula_arr", formula_arr)
	return formula_arr

def formula_arr_to_str(formula_arr, get_h_template = False):
	""" use canonical order """
  
	elem_d = {}

	for idx, count in enumerate(formula_arr):
		elem = IDX_TO_ELEMENT[idx]
		elem_d[elem] = count

	if not get_h_template:
		formula_str = ""
		for elem in CANONICAL_ELEMENT_ORDER:
			if elem in elem_d:
				count = elem_d[elem]
				if count > 0:
					formula_str += elem
				if count > 1:
					formula_str += str(count)
		return formula_str
	else:
		formula_str = ""
		formula_str_no_h = ""
		for elem in CANONICAL_ELEMENT_ORDER:
			if elem in elem_d:
				if elem == 'H':
					formula_str += "H{:d}"
				else:
					count = elem_d[elem]
					if count > 0:
						formula_str += elem
						formula_str_no_h += elem
					if count > 1:
						formula_str += str(count)
						formula_str_no_h += str(count)
		return formula_str, formula_str_no_h
def _frontier_components_after_cut(
    cc_mask: np.ndarray,
    mol_d: dict,
    cut_bond_idx: int,
) -> list[np.ndarray]:
    """
    Given a D3 frontier connected component, remove one internal bond and
    return connected components inside that frontier fragment.

    This is used to approximate one more fragmentation step without adding
    true D4 DAG nodes.
    """
    atom_ids = np.where(cc_mask.astype(bool))[0].astype(np.int32)
    atom_set = set(atom_ids.tolist())

    if len(atom_set) <= 2:
        return []

    cut_u, cut_v = mol_d["bonds"][cut_bond_idx]
    if int(cut_u) not in atom_set or int(cut_v) not in atom_set:
        return []

    adj = {int(a): [] for a in atom_ids}

    for bond_idx, bond in enumerate(mol_d["bonds"]):
        if bond_idx == cut_bond_idx:
            continue

        u, v = int(bond[0]), int(bond[1])
        if u in atom_set and v in atom_set:
            adj[u].append(v)
            adj[v].append(u)

    seen = set()
    comps = []

    for start in atom_ids:
        start = int(start)
        if start in seen:
            continue

        stack = [start]
        seen.add(start)
        comp = []

        while stack:
            cur = stack.pop()
            comp.append(cur)

            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)

        comps.append(np.array(sorted(comp), dtype=np.int32))

    # 如果断键之后仍然只有一个连通块，说明这个 bond 不是有效断裂候选，比如环内边。
    if len(comps) <= 1:
        return []

    return comps

def _cut_atom_pair_type_code(elem1: str, elem2: str) -> int:
	"""
	Small categorical code for the atom pair type of a cut/boundary bond.
	0: none / unknown
	1: C-C
	2: C-N
	3: C-O
	4: C-S/P/halogen
	5: N-O
	6: hetero-hetero
	7: other
	"""
	a, b = sorted([str(elem1), str(elem2)])

	if a == "C" and b == "C":
		return 1
	if a == "C" and b == "N":
		return 2
	if a == "C" and b == "O":
		return 3
	if a == "C" and b in {"S", "P", "F", "Cl", "Br", "I"}:
		return 4
	if a == "N" and b == "O":
		return 5
	if a != "C" and b != "C":
		return 6
	return 7


def _compute_cut_chem_edge_feat(
	parent_mask: np.ndarray,
	child_mask: np.ndarray,
	mol_d: dict,
) -> np.ndarray:
	"""
	E1_cutedge_chem_v1:
	Compute chemistry-aware edge features for one fragment DAG edge.

	parent_mask: atom mask of parent fragment
	child_mask: atom mask of child fragment

	The boundary cut is approximated by bonds with one endpoint in child and
	the other endpoint in parent \\ child. This is robust for normal DAG edges
	and also works when more than one bond connects the removed part.
	"""
	parent_mask = parent_mask.astype(bool)
	child_mask = child_mask.astype(bool)
	removed_mask = parent_mask & (~child_mask)

	parent_size = int(parent_mask.sum())
	child_size = int(child_mask.sum())
	removed_size = int(removed_mask.sum())

	if parent_size <= 0:
		return np.zeros((CUT_CHEM_EDGE_FEAT_SIZE,), dtype=np.int64)

	bond_count = 0
	max_order = 0
	sum_order = 0
	any_aromatic = 0
	any_ring = 0
	any_conjugated = 0
	any_hetero = 0
	max_pair_type = 0
	any_brics = 0

	mol = mol_d.get("mol", None)
	elems = mol_d.get("elems", [])
	brics_bond_idxs = mol_d.get("brics_bond_idxs", set())

	for bond_idx, bond in enumerate(mol_d["bonds"]):
		u, v = int(bond[0]), int(bond[1])

		# Boundary between child and removed part, inside the parent.
		if not (parent_mask[u] and parent_mask[v]):
			continue

		u_child = bool(child_mask[u])
		v_child = bool(child_mask[v])

		if u_child == v_child:
			continue

		bond_count += 1

		order_idx = int(mol_d["bond_type_idxs"][bond_idx])
		max_order = max(max_order, order_idx)
		sum_order += order_idx

		eu = elems[u] if u < len(elems) else ""
		ev = elems[v] if v < len(elems) else ""
		if eu != "C" or ev != "C":
			any_hetero = 1

		max_pair_type = max(
			max_pair_type,
			_cut_atom_pair_type_code(eu, ev),
		)

		if int(bond_idx) in brics_bond_idxs:
			any_brics = 1

		if mol is not None:
			try:
				rb = mol.GetBondWithIdx(int(bond_idx))
				any_aromatic = max(any_aromatic, int(rb.GetIsAromatic()))
				any_ring = max(any_ring, int(rb.IsInRing()))
				any_conjugated = max(any_conjugated, int(rb.GetIsConjugated()))
			except Exception:
				pass

	child_frac_bin = int(round(10.0 * child_size / max(parent_size, 1)))
	loss_frac_bin = int(round(10.0 * removed_size / max(parent_size, 1)))

	return np.asarray(
		[
			min(bond_count, 8),
			min(max_order, 3),
			min(sum_order, 12),
			int(any_aromatic),
			int(any_ring),
			int(any_conjugated),
			int(any_hetero),
			int(max_pair_type),
			min(max(child_frac_bin, 0), 10),
			min(max(loss_frac_bin, 0), 10),
		],
		dtype=np.int64,
	)
def _formula_counts_to_str_safe(counts: dict) -> str:
    """
    Convert element counts to a simple formula string.
    Keep C/H first, then alphabetical others.
    """
    clean = {}
    for k, v in counts.items():
        v = int(v)
        if v > 0:
            clean[k] = v

    if len(clean) == 0:
        return ""

    order = []
    if "C" in clean:
        order.append("C")
    if "H" in clean:
        order.append("H")
    for k in sorted(clean.keys()):
        if k not in {"C", "H"}:
            order.append(k)

    out = []
    for k in order:
        v = clean[k]
        if v <= 0:
            continue
        out.append(k if v == 1 else f"{k}{v}")
    return "".join(out)


def _subtract_formula_template(parent_formula: str, loss_counts: dict) -> str:
    """
    Return parent_formula - loss_counts if possible; otherwise "".
    """
    try:
        counts = parse_formula(parent_formula)
    except Exception:
        return ""

    new_counts = dict(counts)

    for elem, loss_n in loss_counts.items():
        if new_counts.get(elem, 0) < loss_n:
            return ""
        new_counts[elem] = new_counts.get(elem, 0) - loss_n

    return _formula_counts_to_str_safe(new_counts)


def _frontier_formula_shift_candidates(
    parent_formula: str,
    isotopes: bool,
    peaks_for_element_cache: dict,
) -> list[tuple[float, float, str, int, float]]:
	"""
	Formula-level fallback candidates.

	This is the robust v0b part:
	If graph internal-cut candidates are unavailable, generate D4-like pseudo
	support by small chemically plausible losses from D3 frontier formulas.

	Returns:
		list of (score, mz, child_formula, loss_template_idx, template_score)
	"""
	# Loss templates are intentionally small and conservative.
	# They are not final chemistry rules; they are pseudo-support candidates.
	loss_templates = [
		({"H": 2}, 20.0),             # dehydrogenation / rearrangement
		({"C": 1, "H": 2}, 18.0),     # small alkyl loss
		({"C": 1, "H": 4}, 17.0),
		({"N": 1, "H": 3}, 17.0),     # NH3-like
		({"H": 2, "O": 1}, 16.0),     # H2O-like
		({"C": 1, "O": 1}, 15.0),     # CO-like
		({"C": 1, "O": 2}, 14.0),     # CO2-like
		({"C": 2, "H": 4}, 13.0),
	]

	candidates = []

	for template_idx, (loss_counts, template_score) in enumerate(loss_templates):
		child_formula = _subtract_formula_template(parent_formula, loss_counts)
		if child_formula == "":
			continue

		try:
			peak_mzs, peak_probs = formula_to_peak_mzs(
				child_formula,
				"",
				isotopes=isotopes,
				return_probs=True,
				peaks_for_element_cache=peaks_for_element_cache,
			)
		except Exception:
			continue

		if len(peak_mzs) == 0:
			continue

		peak_mzs = np.asarray(peak_mzs, dtype=np.float32)
		peak_probs = np.asarray(peak_probs, dtype=np.float32)

		best_peak_idx = int(np.argmax(peak_probs))
		mz = float(peak_mzs[best_peak_idx])

		if not np.isfinite(mz) or mz <= 0:
			continue

		# Larger mz and higher template_score are preferred.
		score = 1000.0 * template_score + mz
		candidates.append((
			score,
			mz,
			child_formula,
			int(template_idx),
			float(template_score),
		))

	return candidates


def compute_frontier_onehop_peak_support(
    nodes_mask_matrix: np.ndarray,
    nodes_min_depth: np.ndarray,
    max_depth: int,
    mol_d: dict,
    node_formulae_matrix: np.ndarray,
    num_formulae: int,
    max_h_transfer: int,
    isotopes: bool = False,
    max_peaks_per_formula: int = 4,
    max_candidates_per_node: int = 2,
	max_internal_bonds_per_node: int = 32,
	max_graph_candidates_per_node_before_sort: int = 128,
    min_child_heavy_atoms: int = 1,
	allowed_h_deltas: tuple[int, ...] = (0, -1, 1, -2, 2),
	idx_to_formula: dict | None = None,
	formula_to_idx: dict | None = None,
) -> dict:
	"""
	D3 frontier -> pseudo D4 support.

	v0b:
	- first tries graph one-hop internal cuts inside D3 frontier fragments;
	- then adds a robust formula-shift fallback on the D3 frontier parent formula.

	We do NOT add new DAG nodes.
	We only attach extra peak masses to the existing parent formula row.

	Shape:
		frontier_formula_peak_mzs:   [num_formulae, max_peaks_per_formula]
		frontier_formula_peak_probs: [num_formulae, max_peaks_per_formula]
	"""
	frontier_peak_mzs = np.zeros(
		(num_formulae, max_peaks_per_formula),
		dtype=np.float32,
	)
	frontier_peak_probs = np.zeros(
		(num_formulae, max_peaks_per_formula),
		dtype=np.float32,
	)
	frontier_event_type = np.zeros(
		(num_formulae, max_peaks_per_formula),
		dtype=np.int64,
	)
	frontier_parent_node_idx = np.full(
		(num_formulae, max_peaks_per_formula),
		-1,
		dtype=np.int64,
	)
	frontier_cut_bond_idx = np.full(
		(num_formulae, max_peaks_per_formula),
		-1,
		dtype=np.int64,
	)
	frontier_loss_template_idx = np.full(
		(num_formulae, max_peaks_per_formula),
		-1,
		dtype=np.int64,
	)
	frontier_event_score = np.zeros(
		(num_formulae, max_peaks_per_formula),
		dtype=np.float32,
	)
	frontier_h_delta = np.full(
		(num_formulae, max_peaks_per_formula),
		999,
		dtype=np.int64,
	)

	frontier_child_formula_strs = {}
	frontier_event_formula_idx_sparse = []
	frontier_event_parent_node_idx_sparse = []
	frontier_event_child_node_idx_sparse = []
	frontier_event_cut_bond_idx_sparse = []
	frontier_event_h_delta_sparse = []
	frontier_event_score_sparse = []
	frontier_event_child_frac_sparse = []
	frontier_event_parent_size_sparse = []
	frontier_event_child_size_sparse = []
	frontier_event_mz_sparse = []

	if max_depth <= 0:
		return {
			"frontier_formula_peak_mzs": th.as_tensor(frontier_peak_mzs, dtype=META_DATA_DTYPE),
			"frontier_formula_peak_probs": th.as_tensor(frontier_peak_probs, dtype=META_DATA_DTYPE),
			"frontier_event_type": th.as_tensor(frontier_event_type, dtype=th.long),
			"frontier_parent_node_idx": th.as_tensor(frontier_parent_node_idx, dtype=th.long),
			"frontier_cut_bond_idx": th.as_tensor(frontier_cut_bond_idx, dtype=th.long),
			"frontier_loss_template_idx": th.as_tensor(frontier_loss_template_idx, dtype=th.long),
			"frontier_event_score": th.as_tensor(frontier_event_score, dtype=META_DATA_DTYPE),
			"frontier_h_delta": th.as_tensor(frontier_h_delta, dtype=th.long),
			"frontier_child_formula_strs": frontier_child_formula_strs,
			"frontier_event_formula_idx_sparse": th.tensor(frontier_event_formula_idx_sparse, dtype=th.long),
			"frontier_event_parent_node_idx_sparse": th.tensor(frontier_event_parent_node_idx_sparse, dtype=th.long),
			"frontier_event_child_node_idx_sparse": th.tensor(frontier_event_child_node_idx_sparse, dtype=th.long),
			"frontier_event_cut_bond_idx_sparse": th.tensor(frontier_event_cut_bond_idx_sparse, dtype=th.long),
			"frontier_event_h_delta_sparse": th.tensor(frontier_event_h_delta_sparse, dtype=th.long),
			"frontier_event_score_sparse": th.tensor(frontier_event_score_sparse, dtype=th.float32),
			"frontier_event_child_frac_sparse": th.tensor(frontier_event_child_frac_sparse, dtype=th.float32),
			"frontier_event_parent_size_sparse": th.tensor(frontier_event_parent_size_sparse, dtype=th.float32),
			"frontier_event_child_size_sparse": th.tensor(frontier_event_child_size_sparse, dtype=th.float32),
			"frontier_event_mz_sparse": th.tensor(frontier_event_mz_sparse, dtype=th.float32),
		}

	formula_to_candidates: dict[int, list[dict]] = {}

	peaks_for_element_cache = {}

	frontier_node_idxs = np.where(nodes_min_depth == max_depth)[0]

	for node_idx in frontier_node_idxs:
		cc_mask = nodes_mask_matrix[node_idx].astype(bool)

		# h_delta=0 slot. Original code maps h_delta 0 -> index 0.
		parent_formula_idx = int(node_formulae_matrix[node_idx, 0])

		# formula index 0 is usually NULL formula in this codebase.
		if parent_formula_idx <= 0:
			continue

		graph_candidates: list[dict] = []
		fallback_candidates: list[dict] = []

		# ------------------------------------------------------------------
		# A) Graph one-hop internal cut candidates.
		# ------------------------------------------------------------------
		internal_bond_indices = []
		for bond_idx, bond in enumerate(mol_d["bonds"]):
			u, v = int(bond[0]), int(bond[1])
			if cc_mask[u] and cc_mask[v]:
				internal_bond_indices.append(int(bond_idx))

		if len(internal_bond_indices) > max_internal_bonds_per_node:
			step = max(1, len(internal_bond_indices) // max_internal_bonds_per_node)
			internal_bond_indices = internal_bond_indices[::step][:max_internal_bonds_per_node]

		graph_candidate_count = 0

		for bond_idx in internal_bond_indices:
			bond = mol_d["bonds"][bond_idx]
			u, v = int(bond[0]), int(bond[1])

			comps = _frontier_components_after_cut(
				cc_mask=cc_mask,
				mol_d=mol_d,
				cut_bond_idx=bond_idx,
			)

			for comp_atom_ids in comps:
				if len(comp_atom_ids) < min_child_heavy_atoms:
					continue

				if len(comp_atom_ids) >= int(cc_mask.sum()):
					continue

				if graph_candidate_count >= max_graph_candidates_per_node_before_sort:
					break

				try:
					delta_h_to_formula, _ = compute_approximate_formula(
						comp_atom_ids,
						mol_d,
						max_h_transfer=max_h_transfer,
						formula_strs=True,
						bitmask=False,
					)
				except Exception:
					continue

				for h_delta in allowed_h_deltas:
					child_formula = delta_h_to_formula.get(h_delta, "")
					if child_formula == "":
						continue

					child_formula_str = str(child_formula)
					child_formula_idx = -1
					if formula_to_idx is not None:
						child_formula_idx = int(formula_to_idx.get(child_formula_str, -1))

					# Try to map this one-hop child component back to an existing
					# D3 DAG node. This is the key for M2-local routing.
					child_node_idx = -1
					try:
						child_mask = np.zeros_like(cc_mask, dtype=bool)
						child_atom_ids_arr = np.asarray(comp_atom_ids, dtype=np.int64)
						child_mask[child_atom_ids_arr] = True

						node_masks_bool = nodes_mask_matrix.astype(bool)
						same_child = np.where(
							np.all(
								node_masks_bool[:, : child_mask.shape[0]]
								== child_mask[None, :],
								axis=1,
							)
						)[0]

						# Do not map back to the parent itself.
						same_child = same_child[same_child != int(node_idx)]

						if len(same_child) > 0:
							child_node_idx = int(same_child[0])
					except Exception:
						child_node_idx = -1

					try:
						peak_mzs, peak_probs = formula_to_peak_mzs(
							child_formula,
							"",
							isotopes=isotopes,
							return_probs=True,
							peaks_for_element_cache=peaks_for_element_cache,
						)
					except Exception:
						continue

					if len(peak_mzs) == 0:
						continue

					peak_mzs = np.asarray(peak_mzs, dtype=np.float32)
					peak_probs = np.asarray(peak_probs, dtype=np.float32)

					best_peak_idx = int(np.argmax(peak_probs))
					mz = float(peak_mzs[best_peak_idx])

					if not np.isfinite(mz) or mz <= 0:
						continue

					# ===== Frontier ranking v2 =====
					# Old ranking favored very large child fragments:
					#   +1000 * len(comp_atom_ids)
					# That mostly produced parent-near peaks already covered by D3/base/NL.
					#
					# v2 prefers complementary mid-size fragments and penalizes parent-near children.
					parent_size = float(int(cc_mask.sum()))
					child_size = float(len(comp_atom_ids))
					child_frac = child_size / max(parent_size, 1.0)

					# Prefer medium-sized child fragments, not tiny atoms and not parent-near fragments.
					# Peak around child_frac ~= 0.35.
					mid_size_bonus = -8000.0 * abs(child_frac - 0.35)

					# Strongly penalize children that are almost the whole parent.
					parent_near_penalty = 0.0
					if child_frac > 0.75:
						parent_near_penalty = -12000.0 * (child_frac - 0.75) / 0.25

					# Also penalize extremely tiny fragments, but less aggressively.
					tiny_penalty = 0.0
					if child_frac < 0.15:
						tiny_penalty = -6000.0 * (0.15 - child_frac) / 0.15

					# Keep h_delta close to 0, but allow +/-1.
					h_penalty = -250.0 * abs(float(h_delta))

					# Mild mz preference only; do not let high mass dominate.
					mz_bonus = 0.05 * float(mz)

					score = (
						20000.0
						+ mid_size_bonus
						+ parent_near_penalty
						+ tiny_penalty
						+ h_penalty
						+ mz_bonus
					)
					graph_candidates.append({
						"score": float(score),
						"mz": float(mz),
						"formula_idx": int(child_formula_idx),
						"event_type": 1,
						"parent_node_idx": int(node_idx),
						"child_node_idx": int(child_node_idx),
						"cut_bond_idx": int(bond_idx),
						"loss_template_idx": -1,
						"h_delta": int(h_delta),
						"child_formula": child_formula_str,
						"parent_size": int(parent_size),
						"child_size": int(child_size),
						"child_frac": float(child_frac),
					})
					graph_candidate_count += 1

		# ------------------------------------------------------------------
		# B) Robust formula-shift fallback.
		# This is what prevents zero frontier support on the current cache.
		# ------------------------------------------------------------------
		if idx_to_formula is not None:
			parent_formula = idx_to_formula.get(parent_formula_idx, "")
			if parent_formula != "":
				for score, mz, child_formula, template_idx, template_score in _frontier_formula_shift_candidates(
					parent_formula=parent_formula,
					isotopes=isotopes,
					peaks_for_element_cache=peaks_for_element_cache,
				):
					fallback_candidates.append({
						"score": float(score),
						"mz": float(mz),
						"event_type": 2,
						"parent_node_idx": int(node_idx),
						"cut_bond_idx": -1,
						"loss_template_idx": int(template_idx),
						"h_delta": 999,
						"child_formula": str(child_formula),
					})

		# ===== Frontier ranking v3 graph-first =====
		# If graph-cut candidates exist, do not let fallback compete.
		if len(graph_candidates) > 0:
			node_candidates = graph_candidates
		else:
			node_candidates = fallback_candidates

		if len(node_candidates) == 0:
			continue

		node_candidates = sorted(
			node_candidates,
			key=lambda x: x["score"],
			reverse=True,
		)

		seen_mz = set()
		kept = []

		for cand in node_candidates:
			mz = float(cand["mz"])
			key = round(mz, 4)
			if key in seen_mz:
				continue
			seen_mz.add(key)
			kept.append(cand)

			if len(kept) >= max_candidates_per_node:
				break

		if len(kept) == 0:
			continue

		formula_to_candidates.setdefault(parent_formula_idx, []).extend(kept)

	for formula_idx, candidates in formula_to_candidates.items():
		candidates = sorted(
			candidates,
			key=lambda x: x["score"],
			reverse=True,
		)

		seen_mz = set()
		col = 0

		for cand in candidates:
			mz = float(cand["mz"])
			key = round(mz, 4)
			if key in seen_mz:
				continue

			seen_mz.add(key)
			frontier_peak_mzs[formula_idx, col] = mz
			frontier_peak_probs[formula_idx, col] = 1.0

			frontier_event_type[formula_idx, col] = int(cand["event_type"])
			frontier_parent_node_idx[formula_idx, col] = int(cand["parent_node_idx"])
			frontier_cut_bond_idx[formula_idx, col] = int(cand["cut_bond_idx"])
			frontier_loss_template_idx[formula_idx, col] = int(cand["loss_template_idx"])
			frontier_event_score[formula_idx, col] = float(cand["score"])
			frontier_h_delta[formula_idx, col] = int(cand["h_delta"])
			frontier_child_formula_strs[f"{int(formula_idx)}:{int(col)}"] = str(
				cand.get("child_formula", "")
			)

			if int(cand.get("event_type", -1)) == 1:
				sparse_formula_idx = int(cand.get("formula_idx", -1))
				if sparse_formula_idx >= 0:
					frontier_event_formula_idx_sparse.append(sparse_formula_idx)
					frontier_event_parent_node_idx_sparse.append(int(cand["parent_node_idx"]))
					frontier_event_child_node_idx_sparse.append(int(cand.get("child_node_idx", -1)))
					frontier_event_cut_bond_idx_sparse.append(int(cand["cut_bond_idx"]))
					frontier_event_h_delta_sparse.append(int(cand["h_delta"]))
					frontier_event_score_sparse.append(float(cand["score"]))
					frontier_event_child_frac_sparse.append(float(cand.get("child_frac", 0.0)))
					frontier_event_parent_size_sparse.append(int(cand.get("parent_size", 0)))
					frontier_event_child_size_sparse.append(int(cand.get("child_size", 0)))
					frontier_event_mz_sparse.append(float(cand["mz"]))

			col += 1
			if col >= max_peaks_per_formula:
				break

	return {
		"frontier_formula_peak_mzs": th.as_tensor(frontier_peak_mzs, dtype=META_DATA_DTYPE),
		"frontier_formula_peak_probs": th.as_tensor(frontier_peak_probs, dtype=META_DATA_DTYPE),
		"frontier_event_type": th.as_tensor(frontier_event_type, dtype=th.long),
		"frontier_parent_node_idx": th.as_tensor(frontier_parent_node_idx, dtype=th.long),
		"frontier_cut_bond_idx": th.as_tensor(frontier_cut_bond_idx, dtype=th.long),
		"frontier_loss_template_idx": th.as_tensor(frontier_loss_template_idx, dtype=th.long),
		"frontier_event_score": th.as_tensor(frontier_event_score, dtype=META_DATA_DTYPE),
		"frontier_h_delta": th.as_tensor(frontier_h_delta, dtype=th.long),
		"frontier_child_formula_strs": frontier_child_formula_strs,
		"frontier_event_formula_idx_sparse": th.tensor(frontier_event_formula_idx_sparse, dtype=th.long),
		"frontier_event_parent_node_idx_sparse": th.tensor(frontier_event_parent_node_idx_sparse, dtype=th.long),
		"frontier_event_child_node_idx_sparse": th.tensor(frontier_event_child_node_idx_sparse, dtype=th.long),
		"frontier_event_cut_bond_idx_sparse": th.tensor(frontier_event_cut_bond_idx_sparse, dtype=th.long),
		"frontier_event_h_delta_sparse": th.tensor(frontier_event_h_delta_sparse, dtype=th.long),
		"frontier_event_score_sparse": th.tensor(frontier_event_score_sparse, dtype=th.float32),
		"frontier_event_child_frac_sparse": th.tensor(frontier_event_child_frac_sparse, dtype=th.float32),
		"frontier_event_parent_size_sparse": th.tensor(frontier_event_parent_size_sparse, dtype=th.float32),
		"frontier_event_child_size_sparse": th.tensor(frontier_event_child_size_sparse, dtype=th.float32),
		"frontier_event_mz_sparse": th.tensor(frontier_event_mz_sparse, dtype=th.float32),
	}


def compute_frag_peak_stats(peaks,formula_peak_mzs,formula_peak_probs,idx_by_h_delta,\
							 prec_mz, allowed_h_transfer, tolerance=0.01,\
							 prec_type="[M+H]+", is_ppm=False):
	"""_summary_

	Args:
		peaks (_type_): _description_
		formula_peak_mzs (_type_): _description_
		formula_peak_probs (_type_): _description_
		idx_by_h_delta (_type_): _description_
		allowed_h_transfer (_type_): _description_
		tolerance (float, optional): _description_. Defaults to 0.01.
		is_ppm (bool, optional): _description_. Defaults to False.

	Returns:
		_type_: _description_
	"""
	true_mzs, true_ints = list(zip(*peaks))
	theoretical_mzs = formula_peak_mzs
	theoretical_probs = formula_peak_probs

	allowed_idx = list(idx_by_h_delta[0])
	for h in range(1, allowed_h_transfer):
		allowed_idx += list(idx_by_h_delta[2 * h - 1])
		allowed_idx += list(idx_by_h_delta[2 * h])
	allowed_idx = list(set(allowed_idx))
	#print(allowed_idx)
	indices = th.tensor(allowed_idx)
	theoretical_mzs = th.index_select(theoretical_mzs,0, indices)
	theoretical_probs = th.index_select(theoretical_probs,0, indices)

	prec_mask = (theoretical_probs > 0.).type(th.float32)
	# account for adduct mass
	theoretical_mzs = theoretical_mzs + prec_mask * PREC_TYPE_TO_MASS_DIFF[prec_type]
	# compute overlap
	overlap_true_idxs = []
	overlap_true_ints = []
	overlap_pred_idxs = []
	overlap_pred_peak_counts = []
	overlap_pred_formula_counts = []

	# check true_mzs
	for true_idx, true_mz in enumerate(true_mzs):
		mz_diffs = th.abs(theoretical_mzs-true_mz)
		if not is_ppm:
			mz_close = mz_diffs < tolerance 
		else:
			mz_close = mz_diffs < (true_mz * tolerance * PPM)
		if th.any(mz_close):
			pred_idx = th.nonzero(mz_close,as_tuple=False)
			num_formula_match = th.sum(th.any(mz_close,dim=1).type(th.int32)).item()
			num_peak_match = pred_idx.shape[0]
			overlap_true_idxs.append(true_idx)
			overlap_true_ints.append(true_ints[true_idx])
			overlap_pred_idxs.append(pred_idx)
			overlap_pred_peak_counts.append(num_peak_match)
			overlap_pred_formula_counts.append(num_formula_match)
	# remove duplicates
	if len(overlap_pred_idxs) > 0:
		overlap_pred_idxs = th.unique(th.cat(overlap_pred_idxs,dim=0),dim=0)
	else:
		overlap_pred_idxs = th.zeros((0,2))
	recall = len(overlap_true_idxs) / len(true_mzs)
	w_recall = np.sum(overlap_true_ints) / np.sum(true_ints)
	prec = len(overlap_pred_idxs) / th.sum((theoretical_probs>0.).type(th.int32)).item()
	if len(overlap_pred_peak_counts) > 0:
		ppt_peak = np.mean(overlap_pred_peak_counts)
		ppt_formula = np.mean(overlap_pred_formula_counts)
	else:
		ppt_peak = np.nan
		ppt_formula = np.nan
	# check prec_mz stuff
	prec_recalls = []
	for comp_mzs in [theoretical_mzs,th.tensor(true_mzs)]:
		prec_mz_diffs = th.abs(comp_mzs-prec_mz)
		if not is_ppm:
			prec_mz_close = prec_mz_diffs < tolerance 
		else:
			prec_mz_close = prec_mz_diffs < (prec_mz * tolerance * PPM)
		if th.any(prec_mz_close):
			prec_recalls.append(1.)
		else:
			prec_recalls.append(0.)
	prec_recall, prec_spec_recall = prec_recalls
	return pd.Series([recall,w_recall,prec,ppt_peak,ppt_formula,prec_recall,prec_spec_recall])

def th_long_to_mask(long):
	"""_summary_

	Args:
		long (_type_): _description_

	Returns:
		_type_: _description_
	"""
	# long is N x MASK_SIZE//64
	num_dims = long.shape[1]
	long = long.reshape(long.shape[0],num_dims,1)
	mask = 2**th.arange(64-1,-1,-1,device=long.device)
	mask = mask.reshape(1,1,64)
	return long.bitwise_and(mask).ne(0).reshape(long.shape[0],-1)


def th_mask_to_long(mask):
	# mask is N x MASK_SIZE
	num_dims = mask.shape[1]//64
	mask = mask.reshape(mask.shape[0],num_dims,64)
	long = 2**th.arange(64-1,-1,-1,device=mask.device).expand(1,num_dims,64)
	return th.sum(long*mask,dim=2).long()


def compute_dags(
		mol_d:dict,
		max_depth:int,
		h_prior:bool,
		max_h_transfer:int,
		frag_max_time:int,
		isotopes:bool = False,
		nb_isomorphic:bool = False,
		b_isomorphic:bool = False,
		max_iterations:int = -1) -> dict:
	"""_summary_

	Args:
		mol_d (dict): _description_
		max_depth (int): _description_
		h_prior (bool): _description_
		max_h_transfer (int): _description_
		frag_max_time (int): _description_
		isotopes (bool, optional): _description_. Defaults to False.

	Raises:
		ValueError: _description_

	Returns:
		dict: _description_
	"""

	assert h_prior in [True,False], h_prior
	num_nodes, num_edges, node_mask, edges, edge_mask, node_to_edge_idx = get_fraggen_input_arrays(mol_d)
	# time the recursive part
	if frag_max_time is None:
		frag_max_time = int(1e6)
	
	node_mask = node_mask.astype(MASK_DTYPE)
	edge_mask = edge_mask.astype(MASK_DTYPE)
	nodes_mask_matrix, nodes_depth_matrix, dag_edges_matrix, dag_frag_meta = compute_frags.compute_ccs(
		num_nodes,
		num_edges,
		node_mask,
		edges,
		edge_mask,
		node_to_edge_idx,
		max_depth,
		frag_max_time
	)

	if nb_isomorphic:
		node_nb_hashes = get_subgraph_hashes(
			nodes_mask_matrix=nodes_mask_matrix,
			elems=mol_d["elems"], 
			bond_type_idxs=mol_d["bond_type_idxs"], 
			edges=edges[:num_edges], 
			node_to_edge_idx=node_to_edge_idx[:num_nodes], 
			include_bond_type=False,
			max_iterations=max_iterations)
	else:
		node_nb_hashes = None

	# if b_isomorphic:
	# 	node_b_hashes = get_subgraph_hashes(
	# 		nodes_mask_matrix=nodes_mask_matrix,
	# 		elems=mol_d["elems"], 
	# 		bond_type_idxs=mol_d["bond_type_idxs"], 
	# 		edges=edges[:num_edges], 
	# 		node_to_edge_idx=node_to_edge_idx[:num_nodes], 
	# 		include_bond_type=True)
	# else:
	# 	node_b_hashes = None

	# get meta
	reached_depth = dag_frag_meta["reached_depth"]
	edges_min_depth = dag_frag_meta["edges_min_depth"]
	nodes_min_depth = dag_frag_meta["nodes_min_depth"]

	# add node depth information
	# convert to a one hot encoding
	depth_node_feat_size = max_depth+1 #depth
	cc_node_feat_size = MASK_SIZE//64 # mask
	base_formula_node_feat_size = len(ELEMENT_TO_IDX) # base_formula
	formula_node_feat_size = 1+2*max_h_transfer # h_formulae_idx, at max we can have 1 + 2 * max_h_transfer different formula
	h_count_node_feat_size = formula_node_feat_size
	nb_iso_node_feat_size = 1 if nb_isomorphic else 0
	cc_edge_feat_size = cc_node_feat_size
	base_formula_edge_feat_size = base_formula_node_feat_size
	h_range_edge_feat_size = 2
	cut_chem_edge_feat_size = CUT_CHEM_EDGE_FEAT_SIZE

	node_feat_shapes = [
		depth_node_feat_size,
		cc_node_feat_size,
		base_formula_node_feat_size,
		formula_node_feat_size,
		h_count_node_feat_size,
		nb_iso_node_feat_size,
	]

	edge_feat_shapes = [
		cc_edge_feat_size,
		base_formula_edge_feat_size,
		h_range_edge_feat_size,
		cut_chem_edge_feat_size,
	]
	# total_node_feat_size = sum(node_feat_shapes)

	# add node h count and element information
	hs_arr = mol_d["hs_arr"]
	elem_idxs = mol_d["elem_idxs"]
	cc_formula_list = []
	hs_arr_np = np.array(hs_arr)
	for cc_mask in nodes_mask_matrix:
		cc_h_count = np.sum(np.multiply(cc_mask, hs_arr_np))
		cc_formula = cc_bitmask_to_formula_arr(cc_mask,elem_idxs)
		cc_formula[ELEMENT_TO_IDX["H"]] = cc_h_count
		cc_formula_list.append(cc_formula)
	node_base_formula_matrix = np.stack(cc_formula_list, dtype=MASK_DTYPE)
	
	# map nodes to formulae
	formula_d_list, h_count_d_list = [], []
	formula_counts = {}
	for idx, cc_mask in enumerate(nodes_mask_matrix):
		base_formula = node_base_formula_matrix[idx]
		base_formula[ELEMENT_TO_IDX["H"]] = 0 # remove Hs
		delta_h_to_formula, delta_h_to_h_count = compute_approximate_formula(cc_mask,mol_d,max_h_transfer,formula_strs=True, base_formula = base_formula)
		formula_d_list.append(delta_h_to_formula)
		h_count_d_list.append(delta_h_to_h_count)
		formulae = list(delta_h_to_formula.values())
		for formula in formulae:
			formula_counts[formula] = formula_counts.get(formula,0) + 1
   
	# map nodes to formulae indices
	formula_to_idx = {formula:idx for idx,formula in enumerate(sorted(list(formula_counts.keys())))}
	idx_to_formula = {idx:formula for formula,idx in formula_to_idx.items()}
	formula_idx_by_h_delta = [set() for _ in range(1 + 2 * max_h_transfer)]

	formula_idx_list, h_count_list = [], []
	for formulae_dict, h_count_dict in zip(formula_d_list,h_count_d_list):
		formulae_idxs = np.zeros(formula_node_feat_size, dtype=np.int16)
		h_counts = np.zeros(h_count_node_feat_size, dtype=np.int16)
		for h_delta in formulae_dict:
			formula = formulae_dict[h_delta]
			h_count = h_count_dict[h_delta]
			formula_idx = formula_to_idx[formula]
			# h_delta [0,-1,1,-2,2,-3,3,-4,4]
			h_delta_idx = h_delta * 2 if h_delta >= 0 else (-h_delta * 2) - 1
			formulae_idxs[h_delta_idx] = formula_idx
			h_counts[h_delta_idx] = h_count
			formula_idx_by_h_delta[h_delta_idx].add(formula_idx)
		formula_idx_list.append(formulae_idxs)
		h_count_list.append(h_counts)

	node_formulae_matrix = np.stack(formula_idx_list, dtype=np.int16)
	node_h_count_matrix = np.stack(h_count_list, dtype=np.int16)

	# map nodes to nb_isomorphism indices
	if nb_isomorphic:
		nb_iso_map = {hash:idx for idx,hash in enumerate(sorted(list(set(node_nb_hashes))))}
		node_nb_iso_idx = np.array([nb_iso_map[hash] for hash in node_nb_hashes],dtype=np.int16).reshape(-1,1)
		dag_num_nodes_nb = len(set(node_nb_hashes))
	else:
		node_nb_iso_idx = np.zeros((nodes_mask_matrix.shape[0],0),dtype=np.int16)
		dag_num_nodes_nb = -1

	assert nodes_mask_matrix.shape[1] <= MASK_SIZE
	node_cc_mask = th.as_tensor(nodes_mask_matrix,dtype=th.bool)
	node_cc_mask = F.pad(node_cc_mask, (0, MASK_SIZE-node_cc_mask.shape[1]), "constant", 0)
	node_cc_long = th_mask_to_long(node_cc_mask).type(NODE_FEAT_DTYPE)
	# the order is important!
	pyg_node_feats = th.cat(
		[ 
			th.as_tensor(nodes_depth_matrix,dtype=NODE_FEAT_DTYPE),
			node_cc_long,
			th.as_tensor(node_base_formula_matrix,dtype=NODE_FEAT_DTYPE), 
			th.as_tensor(node_formulae_matrix,dtype=NODE_FEAT_DTYPE),
			th.as_tensor(node_h_count_matrix,dtype=NODE_FEAT_DTYPE),
			th.as_tensor(node_nb_iso_idx,dtype=NODE_FEAT_DTYPE)
		], dim=1
	)

	# peak mzs array
	formula_peak_mzs = []
	formula_peak_probs = []
	peaks_for_element_cache = {}
	for formula, idx in formula_to_idx.items():
		if formula == "":
			peak_mzs = np.zeros(MAX_NUM_MZS_PER_FORMULA,dtype=np.float32)
			peak_probs = np.zeros(MAX_NUM_MZS_PER_FORMULA,dtype=np.float32)
		else:
			peak_mzs, peak_probs = formula_to_peak_mzs(formula,"", isotopes = isotopes, return_probs=True, peaks_for_element_cache = peaks_for_element_cache)
			peak_mzs, peak_probs = zip(*sorted(zip(peak_mzs, peak_probs), key=lambda x: x[1], reverse=True))
			peak_mzs = np.array(peak_mzs[:MAX_NUM_MZS_PER_FORMULA],dtype=np.float32)
			peak_mzs = np.pad(peak_mzs,(0,MAX_NUM_MZS_PER_FORMULA-len(peak_mzs)),"constant",constant_values=0)
			peak_probs = np.array(peak_probs[:MAX_NUM_MZS_PER_FORMULA],dtype=np.float32)
			peak_probs = np.pad(peak_probs,(0,MAX_NUM_MZS_PER_FORMULA-len(peak_probs)),"constant",constant_values=0)
		formula_peak_mzs.append(peak_mzs)
		formula_peak_probs.append(peak_probs)

	# save as float32 for speed and lower ram usage
	formula_peak_mzs = th.as_tensor(np.stack(formula_peak_mzs,axis=0), dtype = META_DATA_DTYPE)
	formula_peak_probs = th.as_tensor(np.stack(formula_peak_probs,axis=0), dtype = META_DATA_DTYPE)
	# ===== Our D3 Frontier One-Hop Pseudo Support =====
	# Approximate useful D4 support without constructing full D4 DAG.
	frontier_d = compute_frontier_onehop_peak_support(
		nodes_mask_matrix=nodes_mask_matrix,
		nodes_min_depth=nodes_min_depth,
		max_depth=max_depth,
		mol_d=mol_d,
		node_formulae_matrix=node_formulae_matrix,
		num_formulae=len(formula_to_idx),
		max_h_transfer=max_h_transfer,
		isotopes=isotopes,
		max_peaks_per_formula=2,
		max_candidates_per_node=1,
		min_child_heavy_atoms=2,
		allowed_h_deltas=(0, -1, 1, -2, 2),
		idx_to_formula=idx_to_formula,
		formula_to_idx=formula_to_idx,
	)
	frontier_formula_peak_mzs = frontier_d["frontier_formula_peak_mzs"]
	frontier_formula_peak_probs = frontier_d["frontier_formula_peak_probs"]
	# add edges info

	edge_diff_cc_mask, edge_diff_formula_mask, edge_diff_h_range = [], [], []
	edge_cut_chem_feats = []

	for edge in dag_edges_matrix:
		from_idx, to_idx = edge
		from_cc_mask = nodes_mask_matrix[from_idx]
		to_cc_mask = nodes_mask_matrix[to_idx]
		cut_chem_feat = _compute_cut_chem_edge_feat(
			parent_mask=from_cc_mask,
			child_mask=to_cc_mask,
			mol_d=mol_d,
		)
		diff_cc_mask =  from_cc_mask - to_cc_mask

		from_formula_mask = node_base_formula_matrix[from_idx]
		to_formula_mask = node_base_formula_matrix[to_idx]
		diff_formula_mask = from_formula_mask - to_formula_mask
		diff_formula_mask[CANONICAL_H_IDX] = 0 # we don't care Hs for this

		diff_cc_atom_ids = cc_bit_mask_to_atom_idx(diff_cc_mask)

		diff_h_floor = compute_frags.compute_cc_h_floor(
			diff_cc_atom_ids,
			mol_d["ve_arr"],
			mol_d["sbond_arr"],
			0,
			mol_d["bonds"],
			mol_d["atoms_to_bonds"],
			mol_d["bond_mask_arr"]
		)
		diff_h_cap = compute_cc_h_cap(
			diff_cc_atom_ids,
			mol_d["ve_arr"],
			mol_d["sbond_arr"],
			0
		)
		assert diff_h_floor <= diff_h_cap, (diff_h_floor,diff_h_cap)
		assert diff_h_floor >= 0, diff_h_floor
		#print(diff_h_floor,diff_h_cap)
		diff_h_range = [diff_h_floor,diff_h_cap]
		edge_diff_cc_mask.append(diff_cc_mask)
		edge_diff_formula_mask.append(diff_formula_mask)
		edge_diff_h_range.append(diff_h_range)
		edge_cut_chem_feats.append(cut_chem_feat)

	assert len(edge_diff_cc_mask) == len(edge_diff_formula_mask)
	assert len(edge_diff_cc_mask) > 0, "DAG has no edges"

	edge_diff_cc_mask = th.as_tensor(np.stack(edge_diff_cc_mask,axis=0), dtype=th.bool)
	edge_diff_cc_mask = F.pad(edge_diff_cc_mask, (0, MASK_SIZE-edge_diff_cc_mask.shape[1]), "constant", 0)
	edge_diff_cc_long = th_mask_to_long(edge_diff_cc_mask).type(EDGE_FEAT_DTYPE)
	edge_diff_formula_mask = th.as_tensor(np.stack(edge_diff_formula_mask,axis=0), dtype=EDGE_FEAT_DTYPE)
	edge_diff_h_range = th.as_tensor(np.stack(edge_diff_h_range,axis=0), dtype=EDGE_FEAT_DTYPE)
	edge_cut_chem_feats = th.as_tensor(
		np.stack(edge_cut_chem_feats, axis=0),
		dtype=EDGE_FEAT_DTYPE,
	)

	pyg_edge_feats = th.cat(
		[
			edge_diff_cc_long,
			edge_diff_formula_mask,
			edge_diff_h_range,
			edge_cut_chem_feats,
		],
		dim=1,
	)

	# edge index need to be int64 or it will throw error where compute degree
	pyg_edge_index = th.tensor(dag_edges_matrix.T, dtype=th.int64)

	pyg_cc_g = pyg.data.Data(pyg_node_feats,pyg_edge_index,pyg_edge_feats)

	pyg_cc_g.node_feat_idxs = th.cumsum(th.tensor([0]+node_feat_shapes,dtype=th.long),0).reshape(1,-1)
	pyg_cc_g.edge_feat_idxs = th.cumsum(th.tensor([0]+edge_feat_shapes,dtype=th.long),0).reshape(1,-1)

	# convert to pyg
	frag_d = {}
	frag_d["max_depth"] = max_depth
	frag_d["reached_depth"] = reached_depth
	frag_d["h_prior"] = h_prior
	frag_d["max_h_transfer"] = max_h_transfer
	frag_d["formula_peak_mzs"] = formula_peak_mzs
	frag_d["formula_peak_probs"] = formula_peak_probs
	frag_d["frontier_formula_peak_mzs"] = frontier_formula_peak_mzs
	frag_d["frontier_formula_peak_probs"] = frontier_formula_peak_probs
	frag_d["frontier_event_type"] = frontier_d["frontier_event_type"]
	frag_d["frontier_parent_node_idx"] = frontier_d["frontier_parent_node_idx"]
	frag_d["frontier_cut_bond_idx"] = frontier_d["frontier_cut_bond_idx"]
	frag_d["frontier_loss_template_idx"] = frontier_d["frontier_loss_template_idx"]
	frag_d["frontier_event_score"] = frontier_d["frontier_event_score"]
	frag_d["frontier_h_delta"] = frontier_d["frontier_h_delta"]
	frag_d["frontier_child_formula_strs"] = frontier_d["frontier_child_formula_strs"]
	frag_d["frontier_event_formula_idx_sparse"] = frontier_d["frontier_event_formula_idx_sparse"]
	frag_d["frontier_event_parent_node_idx_sparse"] = frontier_d["frontier_event_parent_node_idx_sparse"]
	frag_d["frontier_event_child_node_idx_sparse"] = frontier_d[
		"frontier_event_child_node_idx_sparse"
	]
	frag_d["frontier_event_cut_bond_idx_sparse"] = frontier_d["frontier_event_cut_bond_idx_sparse"]
	frag_d["frontier_event_h_delta_sparse"] = frontier_d["frontier_event_h_delta_sparse"]
	frag_d["frontier_event_score_sparse"] = frontier_d["frontier_event_score_sparse"]
	frag_d["frontier_event_child_frac_sparse"] = frontier_d["frontier_event_child_frac_sparse"]
	frag_d["frontier_event_parent_size_sparse"] = frontier_d["frontier_event_parent_size_sparse"]
	frag_d["frontier_event_child_size_sparse"] = frontier_d["frontier_event_child_size_sparse"]
	frag_d["frontier_event_mz_sparse"] = frontier_d["frontier_event_mz_sparse"]
	frag_d["idx_to_formula"] = idx_to_formula # useful for annotation
	frag_d["idx_by_h_delta"] = formula_idx_by_h_delta # formula idx for each h_delta
	frag_d["dag"] = pyg_cc_g
	
	frag_d["edges_min_depth"] = edges_min_depth
	frag_d["nodes_min_depth"] = nodes_min_depth

	# add stats here
	# we need change data type again
	# we just need to change this one place
	frag_d["dag_num_edges"] = pyg_cc_g.num_edges
	frag_d["dag_num_nodes"] = pyg_cc_g.num_nodes
	frag_d["dag_num_nodes_nb"] = dag_num_nodes_nb
	frag_d["dag_sparsity"] = 2*pyg_cc_g.num_edges/(pyg_cc_g.num_nodes *(pyg_cc_g.num_nodes - 1))
	frag_d["formula_redundancy"] = sum([v for k,v in formula_counts.items() if k != ""])/len([k for k in formula_counts.keys() if k != ""])
	frag_d["node_feature_size"] = pyg_cc_g.num_features
	frag_d["edge_feature_size"] = pyg_cc_g.num_edge_features
	frag_d["is_directed"] = pyg_cc_g.is_directed()
	frag_d["dag_num_edges_by_depth"] = { k:np.count_nonzero(edges_min_depth == k) for k in range(reached_depth+1)}
	frag_d["dag_num_nodes_by_depth"] = { k:np.count_nonzero(nodes_min_depth == k) for k in range(reached_depth+1)} 
	return frag_d

NODE_FEAT_TO_IDX = {
	"depth":0,
	"cc":1,
	"base_formula":2,
	"h_formulae_idx":3,
	"h_counts":4,
	"nb_iso_idx":5
}

EDGE_FEAT_TO_IDX = {
	"cc": 0,
	"base_formula": 1,
	"h_range": 2,
	"cut_chem": 3,
	"complement": 4,
}

def get_node_feats(node_feats:th.Tensor,node_feat_idxs:th.Tensor,key:str):
	"""get node features by key used for pyg

	Args:
		node_feats (_type_): _description_
		node_feat_idxs (_type_): _description_
		key (_type_): _description_

	Returns:
		_type_: _description_
	"""

	node_feat_idx = NODE_FEAT_TO_IDX[key]
	node_feats = node_feats[:,node_feat_idxs[node_feat_idx]:node_feat_idxs[node_feat_idx+1]]
	#print(f"get_node_feats, node_feats tensor shape: {node_feats.shape}, num nodes: {len(node_feats)}, feature name: {key}" )
	return node_feats

def get_edge_feats(edge_feats:th.Tensor,edge_feat_idxs:int,key:str):
	"""get edege feats used for pyg

	Args:
		edge_feats (_type_): _description_
		edge_feat_idxs (_type_): _description_
		key (_type_): _description_

	Returns:
		_type_: _description_
	"""

	edge_feat_idx = EDGE_FEAT_TO_IDX[key]
	edge_feats = edge_feats[:,edge_feat_idxs[edge_feat_idx]:edge_feat_idxs[edge_feat_idx+1]]
	return edge_feats

def get_frag_name(mol_id:str, is_compressed:bool):
	name = f'{mol_id}.pickle'
	if is_compressed:
		name += ".bz2"
	return name

def get_frag_fp(mol_id:str, frag_dp:str, is_compressed:bool):

	fp = os.path.join(frag_dp,get_frag_name(mol_id, is_compressed))

	return fp

def save_frag_d(frag_d:dict, mol_id:int, frag_dp:str, is_compressed:bool = False):
	"""save frag_d use pickle if is_compressed save as .pbz

	Args:
		frag_d (dict): _description_
		filepath (str): _description_
	"""
	
	fp = get_frag_fp(mol_id, frag_dp, is_compressed)
	try:
		if not is_compressed:
			with open(fp, 'wb') as fileout:
				pickle.dump(frag_d, fileout, protocol=pickle.HIGHEST_PROTOCOL)
		else:
			with bz2.BZ2File(fp, 'wb') as fileout: 
				cPickle.dump(frag_d, fileout)
	except Exception as e:
		print(e,fp)
      
def load_frag_d(mol_id:str, frag_dp:str, is_compressed:bool = False):
	"""_summary_

	Args:
		filepath (str): _description_

	Returns:
		_type_: _description_
	"""
	frag_d = None

	if os.path.isfile(frag_dp) and str(frag_dp).endswith(".tar"):
		frag_filename = get_frag_name(mol_id, is_compressed)
		with tarfile.open(frag_dp, "r") as tar_read:
			for member in tar_read.getmembers():
				if member.name == frag_filename:
					f = tar_read.extractfile(member)
					content = f.read()
					frag_d = pickle.loads(bz2.decompress(content))
					break

	elif os.path.isfile(frag_dp) and str(frag_dp).endswith(".zip"):
		frag_filename = get_frag_name(mol_id, is_compressed)
		with ZipFile(frag_dp, 'r') as zip_read:
			with zip_read.open(frag_filename) as f:
				content = f.read()
				if frag_filename.endswith("bz2"):
					frag_d = pickle.loads(bz2.decompress(content))
				else:
					frag_d = pickle.loads(content)
	else:
		fp = get_frag_fp(mol_id, frag_dp, is_compressed)
		if not is_compressed:
			with open(fp, 'rb') as filein:
				frag_d = pickle.load(filein)
		else:
			with bz2.BZ2File(fp, 'rb') as filein: 
				frag_d = cPickle.load(filein)

	return frag_d

def _hash_label(label, digest_size=32):
	"""
	Adapted from https://networkx.org/documentation/stable/_modules/networkx/algorithms/graph_hashing.html
	"""
	return blake2b(label.encode("ascii"), digest_size=digest_size).hexdigest()

def wl_hash(
	elems: list,
	bond_type_idxs: list,
	node_mask: np.ndarray,
	edges: np.ndarray,
	node_to_edge_idx: np.ndarray,
	include_bond_type: bool = False,
	max_iterations: int = -1
) -> int:
	""" 
	Adapted from https://networkx.org/documentation/stable/_modules/networkx/algorithms/graph_hashing.html
	"""

	cur_hashes = []
	num_nodes = len(elems)
	for i in range(num_nodes):
		if node_mask[i]:
			cur_hashes.append(str(elems[i]))
		else:
			cur_hashes.append("")
	cur_counter = Counter(cur_hashes)
	cur_counter.pop("", None)
	graph_hash_counts = sorted(cur_counter.items(), key=lambda x: x[0])
	iterations = np.sum(node_mask)
	assert iterations <= num_nodes, (iterations, num_nodes)
	if max_iterations == -1:
		max_iterations = iterations
	else:
		assert max_iterations >= 0, max_iterations
	ct = 0
	while ct < iterations and ct < max_iterations:
		# print(cur_hashes)
		new_hashes = []
		temp_atoms = 0
		# Step 2: Update hashes with local neighborhoods
		for node_idx in range(num_nodes):
			cur_hash = cur_hashes[node_idx]
			if not node_mask[node_idx]:
				new_hashes.append(cur_hash)
				continue
			# Count num atoms in this loop
			temp_atoms += 1
			# Get local neighbors
			neighbor_labels = []
			for edge_idx in node_to_edge_idx[node_idx]:
				if edge_idx == -1:
					break
				node_idx_1, node_idx_2 = edges[edge_idx]
				if node_idx_1 == node_idx:
					targ_node_idx = node_idx_2
				else:
					targ_node_idx = node_idx_1
				assert targ_node_idx != node_idx
				if not node_mask[targ_node_idx]:
					continue
				targ_hash = cur_hashes[targ_node_idx]
				if include_bond_type:
					bondtype = bond_type_idxs[edge_idx]
					neighbor_label = f"_{bondtype}_{targ_hash}"
				else:
					neighbor_label = f"_{targ_hash}"
				neighbor_labels.append(neighbor_label)
			new_hash = cur_hash + "".join(sorted(neighbor_labels))
			new_hash = _hash_label(new_hash)
			new_hashes.append(new_hash)
		assert temp_atoms == iterations, (temp_atoms, iterations)
		new_counter = Counter(new_hashes)
		new_counter.pop("", None)
		graph_hash_counts.extend(sorted(new_counter.items(), key=lambda x: x[0]))
		cur_hashes = new_hashes
		# print(f"> {ct}")
		# print(new_graph_hash)
		# print(cur_hashes)
		ct += 1
	graph_hash = _hash_label(str(tuple(graph_hash_counts)))
	return graph_hash

def get_subgraph_hashes(
	nodes_mask_matrix: np.ndarray,
	elems: list, 
	bond_type_idxs: list, 
	edges: np.ndarray, 
	node_to_edge_idx: np.ndarray, 
	include_bond_type: bool,
	max_iterations: int):

	subgraph_hashes = []
	num_subgraphs = nodes_mask_matrix.shape[0]
	for i in range(num_subgraphs):
		subgraph_mask = nodes_mask_matrix[i]
		subgraph_hash = wl_hash(
			elems, 
			bond_type_idxs, 
			subgraph_mask, 
			edges, 
			node_to_edge_idx,
			include_bond_type=include_bond_type,
			max_iterations=max_iterations)
		subgraph_hashes.append(subgraph_hash)
	return subgraph_hashes

def timed_get_dags(mol, 
			mol_id, 
			max_depth,
			h_prior,
			max_h_transfer, 
			max_time,
			isotopes: bool,
			nb_isomorphic: bool,
			max_iterations: int,
			output_dir: str, 
			use_cached = True, 
			compressed = False,
			save_dag = True):

		if save_dag:
			output_file = get_frag_fp(mol_id, output_dir, compressed)

		need_compute = not use_cached or not os.path.exists(output_file)
		if not need_compute:
			assert save_dag
			try:
				dag_d = load_frag_d(mol_id, output_dir, compressed)
			except Exception as e:
				print(e, "cache is not usable")
				need_compute = True
		if need_compute:
			dag_d = {}
			try:
				mol_d = extract_mol_info(mol)
				# this maybe dangers in multi processing because of scopes
				dag_d = compute_dags(
					mol_d,
					max_depth,
					h_prior,
					max_h_transfer, 
					max_time, 
					isotopes,
					nb_isomorphic,
					max_iterations
				)
			except KeyboardInterrupt as e:
				# let these through
				raise e
			except Exception as e:
				# don't retry, theres a bug
				if type(mol) is not str:
					mol = Chem.MolToSmiles(mol)
				print(f">> Non-timeout error, aborting: {type(e)} {repr(e)} Input {mol}",file=sys.stderr)
				print("> Traceback",file=sys.stderr)
				traceback.print_exc(file=sys.stderr)
				dag_d = {}
			else:
				if save_dag:
					save_frag_d(dag_d, mol_id, output_dir, compressed)
					del dag_d["dag"]
		return dag_d