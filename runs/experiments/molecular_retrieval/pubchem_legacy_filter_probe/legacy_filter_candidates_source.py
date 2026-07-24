def filter_candidates(db_results, target_smiles, target_mol_id, top_k = None, morgen_radius = 3):

	chem, rd_fpgen, rd_ds, rd_moldesc, rd_logger = data_utils.rdkit_import("rdkit.Chem",
																			   "rdkit.Chem.rdFingerprintGenerator", 
																			   "rdkit.DataStructs", 
																			   "rdkit.Chem.rdMolDescriptors",
																			   "rdkit.RDLogger")
	
	rd_logger.DisableLog('rdApp.*') 

	mfpgen = rd_fpgen.GetMorganGenerator(radius=morgen_radius)
	t_mol = chem.MolFromSmiles(target_smiles)
	t_fp = mfpgen.GetFingerprint(t_mol)

	# make sure target in there
	db_results.append((-1, data_utils.mol_to_inchikey(t_mol),
					target_smiles, 
					data_utils.mol_to_formula(t_mol),
					data_utils.mol_to_mol_weight(t_mol,exact=True)))
	 
	allowed_elements = set(frag_utils.ELEMENTS)
 
	case_cdf = pd.DataFrame(db_results,columns=['mol_id','inchikey','smiles','formula','mw'])
	case_cdf = case_cdf[['mol_id','smiles','formula','mw']]
 
	# drop not single mol rows
	case_cdf = case_cdf[~(case_cdf['smiles'].str.contains('.', regex=False))]
  
	case_cdf['mol'] = case_cdf['smiles'].apply(lambda x: data_utils.mol_from_smiles(x))
	# drop duplicate no mol rows
	case_cdf = case_cdf.dropna()
	# drop duplicate by canonical smiles and always keep self
	case_cdf['smiles'] = case_cdf['mol'].apply(lambda x:data_utils.mol_to_smiles(x))
	case_cdf = case_cdf.drop_duplicates(subset="smiles", keep="last")
 
	# drop duplicate by inchikey_s smiles and always keep self
	case_cdf['inchikey_s'] = case_cdf['mol'].apply(lambda x:data_utils.mol_to_inchikey_s(x))
	# drop duplicate by inchikey and always keep self
	case_cdf = case_cdf.drop_duplicates(subset="inchikey_s", keep="last")
 
	# filter by heavy atom account
	case_cdf['num_heavy_atoms'] = case_cdf['mol'].apply(lambda x: rd_moldesc.CalcNumHeavyAtoms(x))
	case_cdf = case_cdf[case_cdf['num_heavy_atoms'] <= MAX_NUM_NODES]
	# filter by radicals
	case_cdf['num_radicals'] = case_cdf['mol'].apply(lambda x: sum([atom.GetNumRadicalElectrons() for atom in x.GetAtoms()]))
	case_cdf = case_cdf[case_cdf['num_radicals'] == 0]
 
	# filter by out_set_elements
	case_cdf['num_out_set_elements'] = case_cdf['formula'].apply(lambda x: len(set(list(formula_utils.parse_formula(x).keys())) - allowed_elements))
	case_cdf = case_cdf[case_cdf['num_out_set_elements'] == 0]
 
	# filter by charges
	case_cdf['charge'] = case_cdf["mol"].apply(lambda x: data_utils.mol_to_charge(x))
	case_cdf = case_cdf[case_cdf['charge'] == 0]

	# filter by tanimoto and sort
	case_cdf["tanimoto"] = case_cdf['mol'].apply(lambda x: rd_ds.TanimotoSimilarity(t_fp, mfpgen.GetFingerprint(x)))
	case_cdf = case_cdf[case_cdf['tanimoto'] >= 0.0]
	case_cdf = case_cdf.sort_values('tanimoto', ascending = False)
	
	if top_k is not None:
		case_cdf = case_cdf[:top_k]
 
	return target_mol_id, target_smiles, case_cdf
