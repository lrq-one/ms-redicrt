import torch as th
import torch.nn.functional as F
import numpy as np
import logging
import inspect

# make cc happy
try:
	import lightning.pytorch as pl
	from lightning.fabric.utilities.seed import _collect_rng_states, _set_rng_states
except ModuleNotFoundError:
	import pytorch_lightning as pl
	from pytorch_lightning.utilities.seed import _collect_rng_states, _set_rng_states
	
import torch._dynamo as th_dynamo

from ms2spectra.utils.nn_utils import build_lr_scheduler
from ms2spectra.model import FragGNNModel, NeimsModel, PrecursorModel, GNNModel
from ms2spectra.losses.base import get_sparse_cross_entropy_fn, sparse_entropy_fn, sparse_conditional_entropy_fn, get_edge_loss_fn, sparse_cosine_distance, sparse_cosine_distance_hungarian, sparse_cosine_distance_binned
from ms2spectra.utils.spec_utils import *
from ms2spectra.utils.misc_utils import safelog, flatten_lol, th_temp_seed, scatter_logsumexp, scatter_reduce, to_cpu, scatter_l1normalize, LOG_ZERO
from ms2spectra.utils.plot_utils import plot_spectra_sparse
from ms2spectra.losses.candidate import candidate_presence_rank_loss
from ms2spectra.losses.oracle import setup_oracle_teacher_bins, oracle_teacher_bin_loss
from ms2spectra.losses.window import true_window_distribution_loss
from ms2spectra.components.ce_bin_residual import CEBinResidualHead, apply_ce_bin_residual
from ms2spectra.losses.rendered import r180_ce_weighted_spectrum_loss
class SpectrumPL(pl.LightningModule):

	def __init__(self,**kwargs):

		super().__init__()
		self.save_hyperparameters()
		# setup functions
		self.metric_names = set()
		self._setup_model()
		self._setup_tolerance()
		self._setup_loss_fn()
		self._setup_spec_fns()
		self._setup_metric_fns()
		self._setup_batch_metric_reduce_fns()
		self._setup_result_trackers()
		self._setup_sampler()
		setup_oracle_teacher_bins(self)
		self.ce_bin_residual_head = None
		if bool(getattr(self.hparams, "use_ce_bin_residual_head", False)):
			self.ce_bin_residual_head = CEBinResidualHead(
				input_size=14,
				hidden_size=int(getattr(self.hparams, "ce_bin_residual_hidden_size", 128)),
				dropout=float(getattr(self.hparams, "ce_bin_residual_dropout", 0.05)),
			)

			if bool(getattr(self.hparams, "r107_freeze_base_model", True)):
				for p in self.model.parameters():
					p.requires_grad = False

			head_params = sum(p.numel() for p in self.ce_bin_residual_head.parameters())
			trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
			total = sum(p.numel() for p in self.parameters())

			print(
				"[R107 CEBinResidual] "
				f"enabled=True, head_params={head_params}, "
				f"trainable={trainable}, total={total}, "
				f"freeze_base={bool(getattr(self.hparams, 'r107_freeze_base_model', True))}"
			)
	def _setup_model(self):
		raise NotImplementedError

	def _apply_spectrum_refiner_train_scope(self):
		"""R13: restrict trainable parameters for spectrum refiner experiments.

		Scopes:
		- all: default old behavior, all parameters trainable.
		- refiner_only: only model.spectrum_candidate_refiner trainable.
		- refiner_formula: refiner + formula_module trainable.
		"""
		generic_scope = getattr(self.hparams, "train_scope", "all")
		if generic_scope != "all":
			allowed_map = {
				"ce_delta_only": [
					"ce_formula_node_allocator",
					"ce_response_scorer",
				],
				"ce_delta_peak": [
					"ce_formula_node_allocator",
					"ce_response_scorer",
					"ce_peak_channel_allocator",
				],
				"ce_delta_gate": [
					"ce_fragment_gate",
					"ce_formula_node_allocator",
					"ce_response_scorer",
				],
				"ce_gate_only": [
					"ce_fragment_gate",
				],
				"ce_gate_refiner": [
					"spectrum_candidate_refiner",
					"ce_fragment_gate",
				],
				"ce_local_refiner": [
					"spectrum_candidate_refiner",
					"ce_local_transition_prior",
				],
				"ce_path_refiner": [
					"spectrum_candidate_refiner",
					"ce_path_energy_module",
				],
				"ce_depth_refiner": [
					"spectrum_candidate_refiner",
					"ce_depth_mixture_head",
				],
				"ce_struct_refiner": [
					"spectrum_candidate_refiner",
					"ce_fragment_gate",
					"ce_local_transition_prior",
					"ce_hchannel_transition_prior",
					"ce_path_energy_module",
					"ce_peak_channel_allocator",
					"ce_depth_mixture_head",
					"ce_formula_node_allocator",
					"ce_response_scorer",
				],
			}

			if generic_scope not in allowed_map:
				raise ValueError(f"Unknown train_scope={generic_scope}")

			for p in self.model.parameters():
				p.requires_grad = False

			allowed_prefixes = allowed_map[generic_scope]
			for name, p in self.model.named_parameters():
				if any(name.startswith(prefix) for prefix in allowed_prefixes):
					p.requires_grad = True

			trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
			total = sum(p.numel() for p in self.model.parameters())

			print(
				f"[MYMODEL train_scope] scope={generic_scope}, "
				f"trainable={trainable}, total={total}, "
				f"trainable_frac={trainable / max(total, 1):.4f}"
			)

			shown = 0
			for name, p in self.model.named_parameters():
				if p.requires_grad:
					print(f"[MYMODEL trainable] {name} {tuple(p.shape)}")
					shown += 1
					if shown >= 20:
						print("[MYMODEL trainable] ...")
						break

			return

		scope = getattr(self.hparams, "spectrum_refiner_train_scope", "all")

		if not getattr(self.hparams, "use_spectrum_candidate_refiner", False):
			print("[R13 train scope] spectrum refiner disabled; all parameters keep default")
			return

		if scope == "all":
			trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
			total = sum(p.numel() for p in self.model.parameters())
			print(
				f"[R13 train scope] scope=all, trainable={trainable}, "
				f"total={total}, trainable_frac={trainable / max(total, 1):.4f}"
			)
			return

		# freeze everything first
		for p in self.model.parameters():
			p.requires_grad = False

		if scope == "refiner_only":
			allowed_prefixes = [
				"spectrum_candidate_refiner",
			]
		elif scope == "refiner_formula":
			allowed_prefixes = [
				"spectrum_candidate_refiner",
				"formula_module",
			]
		elif scope == "refiner_peakalloc":
			allowed_prefixes = [
				"spectrum_candidate_refiner",
				"ce_peak_channel_allocator",
			]
		elif scope == "refiner_formula_peakalloc":
			allowed_prefixes = [
				"spectrum_candidate_refiner",
				"formula_module",
				"ce_peak_channel_allocator",
			]
		elif scope == "rendered_peak_gate":
			allowed_prefixes = [
				"rendered_peak_drop_gate",
			]
		elif scope == "pre_r54_peak_entry_gate":
			allowed_prefixes = [
				"pre_r54_peak_entry_gate",
			]
		elif scope == "refiner_rendered_peak_gate":
			allowed_prefixes = [
				"spectrum_candidate_refiner",
				"rendered_peak_drop_gate",
			]
		else:
			raise ValueError(f"Unknown spectrum_refiner_train_scope={scope}")

		for name, p in self.model.named_parameters():
			if any(name.startswith(prefix) for prefix in allowed_prefixes):
				p.requires_grad = True

		trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
		total = sum(p.numel() for p in self.model.parameters())

		print(
			f"[R13 train scope] scope={scope}, trainable={trainable}, "
			f"total={total}, trainable_frac={trainable / max(total, 1):.4f}"
		)

		# Print trainable parameter names once for sanity checking.
		shown = 0
		for name, p in self.model.named_parameters():
			if p.requires_grad:
				print(f"[R13 trainable] {name} {tuple(p.shape)}")
				shown += 1
				if shown >= 20:
					print("[R13 trainable] ...")
					break

	def _setup_tolerance(self):

		# set tolerance
		if self.hparams.loss_tolerance_rel is not None:
			self.tolerance = self.hparams.loss_tolerance_rel
			self.relative = True
			self.tolerance_min_mz = self.hparams.loss_tolerance_min_mz
		else:
			assert self.hparams.loss_tolerance_abs is not None
			self.tolerance = self.hparams.loss_tolerance_abs
			self.relative = False
			self.tolerance_min_mz = None

	def _setup_loss_fn(self):
		raise NotImplementedError

	def _setup_spec_fns(self):

		def _filter_func(mzs, ints, batch_idxs):
			return batched_filter_func(mzs,ints,batch_idxs,self.hparams.ints_thresh,self.hparams.mz_max)
		self.filter_func = _filter_func
		def _bin_func(mzs, ints, batch_idxs):
			agg = "sum" if self.hparams.sum_ints else "amax"
			bin_idxs, bin_ints, bin_batch_idxs = batched_bin_func(
				mzs,
				ints,
				batch_idxs,
				self.hparams.mz_max,
				self.hparams.mz_bin_res,
				agg,
				sparse=True)
			return bin_idxs, bin_ints, bin_batch_idxs
		self.bin_func = _bin_func
		self.ints_transform_func = get_ints_transform_func(self.hparams.ints_transform)
		self.ints_untransform_func = get_ints_untransform_func(self.hparams.ints_transform)
		self.ints_normalize_func = batched_l1_normalize

	def _setup_metric_fns(self):

		# setup metrics
		self.metric_min_max = {"loss": "min"}
		self.metric_bests = ["loss"]
		for k in self.metric_names:
			if k not in self.metric_min_max:
				self.metric_min_max[k] = None
		self.auxiliary_metric_names = set()

		bin_sqrt_flags = [False]
		if self.hparams.eval_bin_sqrt:
			bin_sqrt_flags.append(True)
		bin_remove_prec_peak_flags = [False]
		if self.hparams.eval_bin_remove_prec_peaks:
			bin_remove_prec_peak_flags.append(True)
		hun_sqrt_flags = [False]
		if self.hparams.eval_hun_sqrt:
			hun_sqrt_flags.append(True)
		hun_remove_prec_peak_flags = [False]
		if self.hparams.eval_hun_remove_prec_peaks:
			hun_remove_prec_peak_flags.append(True)
		nb_flags = [False]
		if self.hparams.nb_iso:
			nb_flags.append(True)
		self.extra_metric_args = []

		eval_mz_bin_reses = self.hparams.eval_mz_bin_res

		if len(self.hparams.auxiliary_scores) > 0:
			assert self.hparams.sparse_cosine_similarity
		compute_match_mzs = False
		compute_rounded_match_mzs = False
		compute_bin_specs = False
		compute_entropies = False

		for metric_name in self.hparams.auxiliary_scores:
			if metric_name == "cos_sim":
				for remove_prec_peak in bin_remove_prec_peak_flags:
					for sqrt in bin_sqrt_flags:
						for mz_bin_res in eval_mz_bin_reses:
							fname_base = "cos_sim"
							if sqrt:
								fname_base += "_sqrt"
							if remove_prec_peak:
								fname_base += "_np"
							fname = f"{fname_base}_{mz_bin_res}"
							self.metric_min_max[fname] = "max"
							self.metric_bests.append(fname)
							self.auxiliary_metric_names.add(fname)
					compute_bin_specs = True
			elif metric_name == "jss":
				for remove_prec_peak in bin_remove_prec_peak_flags:
					for sqrt in bin_sqrt_flags:
						for mz_bin_res in eval_mz_bin_reses:
							fname_base = "jss"
							if sqrt:
								fname_base += "_sqrt"
							if remove_prec_peak:
								fname_base += "_np"
							fname = f"{fname_base}_{mz_bin_res}"
							self.metric_min_max[fname] = "max"
							self.metric_bests.append(fname)
							self.auxiliary_metric_names.add(fname)
			elif metric_name == "recall":
				self.metric_min_max["recall"] = "max"
				self.auxiliary_metric_names.add("recall")
				compute_match_mzs = True
			elif metric_name == "wrecall":
				self.metric_min_max["wrecall"] = "max"
				self.auxiliary_metric_names.add("wrecall")
				compute_match_mzs = True
			elif metric_name == "opt_cos_sim":
				for remove_prec_peak in bin_remove_prec_peak_flags:
					for sqrt in bin_sqrt_flags:
						for mz_bin_res in eval_mz_bin_reses:
							fname_base = "opt_cos_sim"
							if sqrt:
								fname_base += "_sqrt"
							if remove_prec_peak:
								fname_base += "_np"
							fname = f"{fname_base}_{mz_bin_res}"
							self.metric_min_max[fname] = None
							self.auxiliary_metric_names.add(fname)
					compute_bin_specs = True
			elif metric_name == "true_spec_e":
				self.metric_min_max["true_spec_e"] = None
				self.auxiliary_metric_names.add("true_spec_e")
				compute_entropies = True
			elif metric_name == "true_spec_ne":
				self.metric_min_max["true_spec_ne"] = None
				self.auxiliary_metric_names.add("true_spec_ne")
				compute_entropies = True
			elif metric_name == "cos_hun":
				for remove_prec_peak in hun_remove_prec_peak_flags:
					for sqrt in hun_sqrt_flags:
						fname_base = "cos_hun"
						if sqrt:
							fname_base += "_sqrt"
						if remove_prec_peak:
							fname_base += "_np"
						fname = f"{fname_base}"
						self.metric_min_max[fname] = "max"
						self.metric_bests.append(fname)
						self.auxiliary_metric_names.add(fname)
				compute_rounded_match_mzs = True
			elif metric_name == "precision":
				self.metric_min_max["precision"] = "max"
				self.metric_bests.append("precision")
				self.auxiliary_metric_names.add("precision")
				compute_match_mzs = True
			elif metric_name == "wprecision":
				self.metric_min_max["wprecision"] = "max"
				self.metric_bests.append("wprecision")
				self.auxiliary_metric_names.add("wprecision")
				compute_match_mzs = True
			# elif metric_name == "dice":
			# 	self.metric_min_max["dice"] = "max"
			# 	self.metric_bests.append("dice")
			# 	compute_match_mzs = True
			elif metric_name == "ndcg":
				for union in [True,False]:
					if union:
						fname_base = "ndcg_un"
						tie_break_flags = [True,False]
					else:
						fname_base = "ndcg_int"
						tie_break_flags = [True]
					for optimistic in tie_break_flags:
						if union:
							suffix = "_opt" if optimistic else "_pess"
						else:
							suffix = ""
						fname = f"{fname_base}{suffix}"
						self.metric_min_max[fname] = "max"
						self.metric_bests.append(fname)
						self.auxiliary_metric_names.add(fname)
				compute_rounded_match_mzs = True
			elif metric_name == "jss_hun":
				for remove_prec_peak in hun_remove_prec_peak_flags:
					for sqrt in hun_sqrt_flags:
						fname_base = "jss_hun"
						if sqrt:
							fname_base += "_sqrt"
						if remove_prec_peak:
							fname_base += "_np"
						fname = f"{fname_base}"
						self.metric_min_max[fname] = "max"
						self.metric_bests.append(fname)
						self.auxiliary_metric_names.add(fname)
				compute_rounded_match_mzs = True
			elif metric_name == "true_oos_prob":
				self.metric_min_max["true_oos_prob"] = None
				self.auxiliary_metric_names.add("true_oos_prob")
				compute_match_mzs = True
			elif metric_name == "true_oos_e":
				self.metric_min_max["true_oos_e"] = None
				self.auxiliary_metric_names.add("true_oos_e")
				compute_match_mzs = True
			elif metric_name == "pred_node_count":
				for nb_flag in nb_flags:
					if nb_flag:
						fname = "pred_nb_node_count"
						self.extra_metric_args.extend(["pred_nb_node_batch_idxs"])
					else:
						fname = "pred_node_count"
						self.extra_metric_args.extend(["pred_node_batch_idxs"])
					self.metric_min_max[fname] = None
					self.auxiliary_metric_names.add(fname)
			elif metric_name == "pred_formula_count":
				fname = "pred_formula_count"
				self.extra_metric_args.extend(["pred_formula_batch_idxs"])
				self.metric_min_max[fname] = None
				self.auxiliary_metric_names.add(fname)
			elif metric_name == "pred_edge_count":
				fname = "pred_edge_count"
				self.extra_metric_args.extend(["pred_edge_batch_idxs"])
				self.metric_min_max[fname] = None
				self.auxiliary_metric_names.add(fname)
			else:
				raise ValueError(f"metric_name {metric_name} not recognized")

		self.metric_names.update(self.auxiliary_metric_names)

		def calculate_all_auxiliary_metrics(
			true_mzs,
			true_ints,
			true_batch_idxs,
			pred_mzs,
			pred_ints,
			pred_batch_idxs,
			true_prec_mzs,
			**kwargs):

			# assumes both true/pred spectra are already L1-normalized

			batch_size = th.max(true_batch_idxs)+1
			assert batch_size == th.max(pred_batch_idxs)+1, (batch_size, th.max(pred_batch_idxs)+1)

			metric_d = {k: -th.ones([batch_size], dtype=true_ints.dtype, device=true_ints.device) for k in self.auxiliary_metric_names}

			# define aggregation
			agg = "sum" if self.hparams.sum_ints else "amax"

			# global calculations
			if compute_rounded_match_mzs:
				true_mzs_r, true_ints_r, true_batch_idxs_r = round_aggregate_peaks(
					true_mzs,
					true_ints,
					true_batch_idxs,
					agg=agg
				)
				pred_mzs_r, pred_ints_r, pred_batch_idxs_r = round_aggregate_peaks(
					pred_mzs,
					pred_ints,
					pred_batch_idxs,
					agg=agg
				)

			if compute_bin_specs:
				for remove_prec_peak in bin_remove_prec_peak_flags:
					for sqrt in bin_sqrt_flags:
						if sqrt:
							ints_transform = get_ints_transform_func("sqrt")
						else:
							ints_transform = get_ints_transform_func("none")
						for mz_bin_res in eval_mz_bin_reses:
							# bin, transform, normalize
							true_bin_idxs, true_bin_ints, true_bin_batch_idxs = batched_bin_func(
								true_mzs,
								true_ints,
								true_batch_idxs,
								mz_max=self.hparams.mz_max,
								mz_bin_res=mz_bin_res,
								agg=agg,
								sparse=True,
								remove_prec_peaks=remove_prec_peak,
								prec_mzs=true_prec_mzs
							)
							true_ints = scatter_l1normalize(ints_transform(true_ints), true_batch_idxs)
							pred_bin_idxs, pred_bin_ints, pred_bin_batch_idxs = batched_bin_func(
								pred_mzs,
								pred_ints,
								pred_batch_idxs,
								mz_max=self.hparams.mz_max,
								mz_bin_res=mz_bin_res,
								agg=agg,
								sparse=True,
								remove_prec_peaks=remove_prec_peak,
								prec_mzs=true_prec_mzs
							)
							pred_ints = scatter_l1normalize(ints_transform(pred_ints), pred_batch_idxs)
							if "cos_sim" in self.hparams.auxiliary_scores:
								fname_base = f"cos_sim"
								if sqrt:
									fname_base += "_sqrt"
								if remove_prec_peak:
									fname_base += "_np"
								fname = f"{fname_base}_{mz_bin_res}"
								cos_sim = cos_sim_helper(
									true_bin_idxs,
									true_bin_ints,
									true_bin_batch_idxs,
									pred_bin_idxs,
									pred_bin_ints,
									pred_bin_batch_idxs
								)
								# print(cos_sim)
								metric_d[fname] = cos_sim
							if "opt_cos_sim" in self.hparams.auxiliary_scores:
								fname_base = f"opt_cos_sim"
								if sqrt:
									fname_base += "_sqrt"
								if remove_prec_peak:
									fname_base += "_np"
								fname = f"{fname_base}_{mz_bin_res}"
								opt_cos_sim = opt_cos_sim_helper(
									true_bin_idxs,
									true_bin_ints,
									true_bin_batch_idxs,
									pred_bin_idxs,
									pred_bin_ints,
									pred_bin_batch_idxs,
								)
								metric_d[fname] = opt_cos_sim
							if "jss" in self.hparams.auxiliary_scores:
								fname_base = f"jss"
								if sqrt:
									fname_base += "_sqrt"
								if remove_prec_peak:
									fname_base += "_np"
								fname = f"{fname_base}_{mz_bin_res}"
								jss = jss_helper(
									true_bin_idxs,
									true_bin_ints,
									true_bin_batch_idxs,
									pred_bin_idxs,
									pred_bin_ints,
									pred_bin_batch_idxs,
									log_min=self.hparams.log_min
								)
								metric_d[fname] = jss

			if compute_entropies:
				true_log_probs = safelog(scatter_l1normalize(true_ints,true_batch_idxs), eps=self.hparams.log_min)
				true_spec_e, true_spec_ne = sparse_entropy_fn(true_log_probs,true_batch_idxs)
				if "true_spec_e" in self.hparams.auxiliary_scores:
					metric_d["true_spec_e"] = true_spec_e
				if "true_spec_ne" in self.hparams.auxiliary_scores:
					metric_d["true_spec_ne"] = true_spec_ne

			# local calculations
			for b_idx in range(batch_size):

				b_true_mask = (true_batch_idxs==b_idx)
				b_pred_mask = (pred_batch_idxs==b_idx)
				b_true_mzs = true_mzs[b_true_mask]
				b_pred_mzs = pred_mzs[b_pred_mask]
				b_true_ints = true_ints[b_true_mask]
				b_pred_ints = pred_ints[b_pred_mask]
				b_true_prec_mz = true_prec_mzs[b_idx:b_idx+1]

				if compute_match_mzs:
					b_match_mask = calculate_match_mzs(
						b_true_mzs,
						b_pred_mzs,
						tolerance=self.tolerance,
						relative=self.relative,
						tolerance_min_mz=self.tolerance_min_mz
					)
					b_true_match_mask = th.any(b_match_mask,dim=1)
					b_pred_match_mask = th.any(b_match_mask,dim=0)

				if compute_rounded_match_mzs:
					b_true_mask_r = (true_batch_idxs_r==b_idx)
					b_pred_mask_r = (pred_batch_idxs_r==b_idx)
					b_true_mzs_r = true_mzs_r[b_true_mask_r]
					b_pred_mzs_r = pred_mzs_r[b_pred_mask_r]
					b_true_ints_r = true_ints_r[b_true_mask_r]
					b_pred_ints_r = pred_ints_r[b_pred_mask_r]
					# precursor
					b_true_prec_mask_r = calculate_match_mzs(
						b_true_mzs_r,
						b_true_prec_mz,
						tolerance=self.tolerance,
						relative=self.relative,
						tolerance_min_mz=self.tolerance_min_mz
					).squeeze(1)
					b_pred_prec_mask_r = calculate_match_mzs(
						b_pred_mzs_r,
						b_true_prec_mz,
						tolerance=self.tolerance,
						relative=self.relative,
						tolerance_min_mz=self.tolerance_min_mz
					).squeeze(1)
					# match
					b_match_mask_r = calculate_match_mzs(
						b_true_mzs_r,
						b_pred_mzs_r,
						tolerance=self.tolerance,
						relative=self.relative,
						tolerance_min_mz=self.tolerance_min_mz
					)
					b_true_match_mask_r = th.any(b_match_mask_r,dim=1)
					b_pred_match_mask_r = th.any(b_match_mask_r,dim=0)
					

				if "recall" in self.hparams.auxiliary_scores:
					b_recall = th.sum(b_true_match_mask.float()) / b_true_match_mask.shape[0]
					metric_d["recall"][b_idx] = b_recall
								
				if "wrecall" in self.hparams.auxiliary_scores:
					b_wrecall = th.sum(b_true_ints[b_true_match_mask]) #/ th.sum(b_true_ints)
					metric_d["wrecall"][b_idx] = b_wrecall

				if "precision" in self.hparams.auxiliary_scores:
					b_precision = th.sum(b_pred_match_mask.float()) / b_pred_match_mask.shape[0]
					metric_d["precision"][b_idx] = b_precision

				if "wprecision" in self.hparams.auxiliary_scores:
					b_wprecision = th.sum(b_pred_ints[b_pred_match_mask])
					metric_d["wprecision"][b_idx] = b_wprecision

				if "cos_hun" in self.hparams.auxiliary_scores:
					for remove_prec_peak in hun_remove_prec_peak_flags:
						for sqrt in hun_sqrt_flags:
							if sqrt:
								ints_transform = get_ints_transform_func("sqrt")
							else:
								ints_transform = get_ints_transform_func("none")
							b_cos_hun = cos_hun_helper(
								ints_transform(b_true_ints_r),
								ints_transform(b_pred_ints_r),
								b_match_mask_r,
								b_true_match_mask_r,
								b_pred_match_mask_r,
								remove_prec_peak,
								b_true_prec_mask_r,
								b_pred_prec_mask_r
							)
							fname_base = f"cos_hun"
							if sqrt:
								fname_base += "_sqrt"
							if remove_prec_peak:
								fname_base += "_np"
							fname = f"{fname_base}"
							metric_d[fname][b_idx] = b_cos_hun

				if "ndcg" in self.hparams.auxiliary_scores:
					for union in [True,False]:
						if union:
							fname_base = "ndcg_un"
							tie_break_flags = [True,False]
						else:
							fname_base = "ndcg_int"
							tie_break_flags = [True]
						for optimistic in tie_break_flags:
							if union:
								suffix = "_opt" if optimistic else "_pess"
							else:
								suffix = ""
							b_ndcg = ndcg_helper(
								b_true_ints,
								b_pred_ints,
								b_match_mask,
								b_true_match_mask,
								b_pred_match_mask,
								optimistic,
								union
							)
							fname = f"{fname_base}{suffix}"
							metric_d[fname][b_idx] = b_ndcg

				if "jss_hun" in self.hparams.auxiliary_scores:
					for remove_prec_peak in hun_remove_prec_peak_flags:
						for sqrt in hun_sqrt_flags:
							if sqrt:
								ints_transform = get_ints_transform_func("sqrt")
							else:
								ints_transform = get_ints_transform_func("none")
							b_jss_hun = jss_hun_helper(
								ints_transform(b_true_ints_r),
								ints_transform(b_pred_ints_r),
								b_match_mask_r,
								b_true_match_mask_r,
								b_pred_match_mask_r,
								remove_prec_peak,
								b_true_prec_mask_r,
								b_pred_prec_mask_r,
								log_min=self.hparams.log_min
							)
							fname_base = f"jss_hun"
							if sqrt:
								fname_base += "_sqrt"
							if remove_prec_peak:
								fname_base += "_np"
							fname = f"{fname_base}"
							metric_d[fname][b_idx] = b_jss_hun

				if "true_oos_prob" in self.hparams.auxiliary_scores:
					
					b_true_oos_prob = th.sum(b_true_ints[~b_true_match_mask]) / th.sum(b_true_ints)
					metric_d["true_oos_prob"][b_idx] = b_true_oos_prob

				if "true_oos_e" in self.hparams.auxiliary_scores:

					b_true_oos_probs = b_true_ints[~b_true_match_mask] / th.sum(b_true_ints[~b_true_match_mask])
					b_true_oos_logprobs = safelog(b_true_oos_probs, eps=self.hparams.log_min)
					b_true_oos_e = -th.sum(b_true_oos_probs * b_true_oos_logprobs)
					metric_d["true_oos_e"][b_idx] = b_true_oos_e

			if "pred_node_count" in self.hparams.auxiliary_scores:

				for nb_flag in nb_flags:
					if nb_flag:
						key = "nb_node"
					else:
						key = "node"
					b_pred_node_count = scatter_reduce(
						th.ones_like(kwargs[f"pred_{key}_batch_idxs"]),
						kwargs[f"pred_{key}_batch_idxs"],
						reduce="sum",
						dim=0
					)
					assert th.min(b_pred_node_count) > 0, b_pred_node_count
					metric_d[f"pred_{key}_count"] = b_pred_node_count

			if "pred_formula_count" in self.hparams.auxiliary_scores:

				b_pred_formula_count = scatter_reduce(
					th.ones_like(kwargs["pred_formula_batch_idxs"]),
					kwargs["pred_formula_batch_idxs"],
					reduce="sum",
					dim=0
				)
				b_pred_formula_count = b_pred_formula_count - 1  # -1 for OOS!
				assert th.min(b_pred_formula_count) > 0, b_pred_formula_count
				metric_d["pred_formula_count"] = b_pred_formula_count

			if "pred_edge_count" in self.hparams.auxiliary_scores:

				b_pred_edge_count = scatter_reduce(
					th.ones_like(kwargs["pred_edge_batch_idxs"]),
					kwargs["pred_edge_batch_idxs"],
					reduce="sum",
					dim=0
				)
				assert th.min(b_pred_edge_count) > 0, b_pred_edge_count
				metric_d["pred_edge_count"] = b_pred_edge_count

			return metric_d
	
		self.metric_fn = calculate_all_auxiliary_metrics

	def _get_batch_metric_reduce_fn(self, sample_weight):

		if sample_weight == "none":
			calc_sample_weights = lambda spec_per_group, spec_per_mol, group_per_mol: th.ones_like(spec_per_group, dtype=th.float32)
		elif sample_weight == "group":
			calc_sample_weights = lambda spec_per_group, spec_per_mol, group_per_mol: 1. / spec_per_group
		elif sample_weight == "mol":
			calc_sample_weights = lambda spec_per_group, spec_per_mol, group_per_mol: 1. / spec_per_mol
		elif sample_weight == "group_mol":
			calc_sample_weights = lambda spec_per_group, spec_per_mol, group_per_mol: 1. / (spec_per_group*group_per_mol)
		def _batch_metric_reduce(b_metric, b_spec_per_group, b_spec_per_mol, b_group_per_mol, reduce, return_weights=False):
			b_sample_weight = calc_sample_weights(b_spec_per_group, b_spec_per_mol, b_group_per_mol)
			b_total_weight = th.sum(b_sample_weight, dim=0)
			if reduce == "w_mean":
				b_reduce_metric = th.sum(b_sample_weight * b_metric, dim=0) / b_total_weight
			elif reduce == "w_std":
				b_reduce_metric = th.sqrt(
					th.sum(b_sample_weight * (b_metric - th.sum(b_sample_weight * b_metric, dim=0) / b_total_weight)**2, dim=0) / b_total_weight
				)
			else:
				assert reduce == "w_sum", reduce
				b_reduce_metric = th.sum(b_sample_weight * b_metric, dim=0)
			if return_weights:
				return b_reduce_metric, b_total_weight
			else:
				return b_reduce_metric
		return _batch_metric_reduce

	def _setup_batch_metric_reduce_fns(self):
		
		self.train_batch_metric_reduce_fn = self._get_batch_metric_reduce_fn(self.hparams.train_sample_weight)
		self.eval_batch_metric_reduce_fn = self._get_batch_metric_reduce_fn(self.hparams.eval_sample_weight)
		def _batch_metric_reduce_fn(split,**kwargs):
			if split == "train":
				return self.train_batch_metric_reduce_fn(**kwargs)
			else:
				return self.eval_batch_metric_reduce_fn(**kwargs)
		self.batch_metric_reduce_fn = _batch_metric_reduce_fn

	def _setup_result_trackers(self):

		if self.hparams.spec_params["merge"]:
			max_num_datapoints = 22000
		else:
			max_num_datapoints = 270000
		for split in ["train","val","test"]:
			setattr(self,f"{split}_results",None)
			setattr(self,f"{split}_counter",0)
			setattr(self,f"{split}_mean_metrics",{})
			setattr(self,f"{split}_std_metrics",{})
			if self.hparams.track_datapoint_metrics:
				setattr(self,f"{split}_datapoint_metrics",{})
				setattr(self,f"{split}_num_datapoints",-th.ones([1],dtype=th.int64))
			for name in self.metric_names:
				_name = name.replace(".","-")
				mean_metrics = getattr(self,f"{split}_mean_metrics")
				std_metrics = getattr(self,f"{split}_std_metrics")
				if split != "test":
					mean_metrics[_name] = -th.ones([self.hparams.max_epochs],dtype=th.float32)
					std_metrics[_name] = -th.ones([self.hparams.max_epochs],dtype=th.float32)
				else:
					mean_metrics[_name] = -th.ones([1],dtype=th.float32)
					std_metrics[_name] = -th.ones([1],dtype=th.float32)
				if self.hparams.track_datapoint_metrics:
					datapoint_metrics = getattr(self,f"{split}_datapoint_metrics")
					datapoint_metrics[_name] = -th.ones([max_num_datapoints],dtype=th.float32)

	def _setup_sampler(self):

		self._cur_batch_size = 0
		self._cur_batch_weight = 0.
		self._max_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches
		self.automatic_optimization = self.hparams.automatic_optimization
		train_dl_generator = th.Generator()
		train_dl_generator.manual_seed(self.hparams.seed)
		self.train_dl_seeds = th.randint(
			low=0,
			high=2**32-1,
			size=[self.hparams.max_epochs+1],
			generator=train_dl_generator)

	def _preproc_spec(self,spec_mzs,spec_ints,spec_batch_idxs,filter_spec=False,bin_spec=False,transform_spec=False,normalize_spec=False):
		# assumes spec_ints are not logged, or transformed in any way
		# does not assume any particular kind of normalization

		if filter_spec:
			# filter
			spec_mzs, spec_ints, spec_batch_idxs = self.filter_func(
				spec_mzs,
				spec_ints,
				spec_batch_idxs
			)
		if bin_spec:
			# bin
			spec_mzs, spec_ints, spec_batch_idxs = self.bin_func(
				spec_mzs,
				spec_ints,
				spec_batch_idxs
			)
		if transform_spec:
			# normalize to mf1000
			spec_ints = batched_mf1000_normalize(spec_ints, spec_batch_idxs)
			# transform
			spec_ints = self.ints_transform_func(spec_ints)
		if normalize_spec:
			# renormalize
			spec_ints = self.ints_normalize_func(spec_ints, spec_batch_idxs)
		return spec_mzs, spec_ints, spec_batch_idxs

	def preproc_spec(
		self,
		spec_mzs,
		spec_ints,
		spec_batch_idxs,
		train: bool,
		pred: bool,
		log_in: bool = False,
		log_out: bool = False
	):
		if log_in:
			spec_ints = spec_ints.exp()
		if train and pred:
			# if you normalize here, you would mess up some losses (i.e. OOS)
			spec_mzs, spec_ints, spec_batch_idxs = self._preproc_spec(
				spec_mzs,
				spec_ints,
				spec_batch_idxs,
				filter_spec=False,
				bin_spec=self.binned_loss,
				transform_spec=False,
				normalize_spec=False
			)
		elif train and (not pred):
			spec_mzs, spec_ints, spec_batch_idxs = self._preproc_spec(
				spec_mzs,
				spec_ints,
				spec_batch_idxs,
				filter_spec=True,
				bin_spec=self.binned_loss,
				transform_spec=True,
				normalize_spec=True
			)
		elif (not train) and pred:
			# untransform
			spec_ints = self.ints_untransform_func(spec_ints, spec_batch_idxs)
			# normalize (note that this messes up OOS stuff, which is fine for eval...)
			spec_ints = self.ints_normalize_func(spec_ints, spec_batch_idxs)
		elif (not train) and (not pred):
			spec_mzs, spec_ints, spec_batch_idxs =  self._preproc_spec(
				spec_mzs,
				spec_ints,
				spec_batch_idxs,
				filter_spec=True,
				bin_spec=False,
				transform_spec=False,
				normalize_spec=True
			)
		if log_out:
			spec_ints = safelog(spec_ints, eps=self.hparams.log_min)
		return spec_mzs, spec_ints, spec_batch_idxs

	def predict_step(self,**batch_kwargs):
		
		return self.forward(**batch_kwargs)

	def forward(self,**batch_kwargs):

		# get predictions
		if self.hparams.activation_checkpointing:
			forward_keys = list(inspect.signature(self.model.forward).parameters.keys())
			forward_keys.remove("kwargs")
			forward_args = [batch_kwargs.get(k,None) for k in forward_keys]
			pred = th.utils.checkpoint.checkpoint(
				self.model.forward,
				*forward_args,
				use_reentrant=False)
		else:
			pred = self.model.forward(**batch_kwargs)
		return pred

	def _ce_depth_rank_loss(self, batch, pred_d, batch_size):
		"""CE trajectory regularizer.

		For the same molecule within a batch:
		if CE_high > CE_low, predicted expected fragment depth should not decrease.

		Returns:
			loss: scalar tensor connected to graph
			num_pairs: scalar tensor for logging
		"""

		# Keep graph connection even when no valid pair exists.
		base_zero = pred_d["pred_node_logprobs"].sum() * 0.0
		device = pred_d["pred_node_logprobs"].device

		if "pred_node_depths" not in pred_d:
			return base_zero, th.zeros([], dtype=th.float32, device=device)

		mol_ids = batch.get("mol_id", None)
		ce = batch.get("spec_ce", None)
		if mol_ids is None or ce is None:
			return base_zero, th.zeros([], dtype=th.float32, device=device)

		batch_n = int(batch_size.item()) if hasattr(batch_size, "item") else int(batch_size)
		if batch_n <= 1:
			return base_zero, th.zeros([], dtype=th.float32, device=device)

		# Convert CE to one scalar per spectrum.
		ce = ce.to(device=device, dtype=pred_d["pred_node_logprobs"].dtype).flatten()
		if ce.numel() == batch_n:
			ce_vals = ce
		else:
			ce_batch_idxs = batch.get("spec_ce_batch_idxs", None)
			if ce_batch_idxs is None:
				return base_zero, th.zeros([], dtype=th.float32, device=device)
			ce_batch_idxs = ce_batch_idxs.to(device=device)
			ce_sum = scatter_reduce(
				ce,
				ce_batch_idxs,
				reduce="sum",
				dim_size=batch_n,
			)
			ce_count = scatter_reduce(
				th.ones_like(ce),
				ce_batch_idxs,
				reduce="sum",
				dim_size=batch_n,
			).clamp_min(1.0)
			ce_vals = ce_sum / ce_count

		# Expected predicted depth per spectrum.
		node_probs = pred_d["pred_node_logprobs"].exp()
		node_depths = pred_d["pred_node_depths"].to(
			device=device,
			dtype=node_probs.dtype,
		)
		node_batch_idxs = pred_d["pred_node_batch_idxs"].to(device=device)

		depth_num = scatter_reduce(
			node_probs * node_depths,
			node_batch_idxs,
			reduce="sum",
			dim_size=batch_n,
		)
		depth_den = scatter_reduce(
			node_probs,
			node_batch_idxs,
			reduce="sum",
			dim_size=batch_n,
		).clamp_min(1e-8)

		exp_depth = depth_num / depth_den

		min_ce_diff = float(self.hparams.ce_depth_rank_min_ce_diff)
		margin = float(self.hparams.ce_depth_rank_margin)

		low_idxs = []
		high_idxs = []

		# Batch size is small; Python pair construction is fine and safer.
		for i in range(batch_n):
			for j in range(batch_n):
				if i == j:
					continue
				if mol_ids[i] != mol_ids[j]:
					continue
				if float(ce_vals[j].detach().cpu()) >= float(ce_vals[i].detach().cpu()) + min_ce_diff:
					low_idxs.append(i)
					high_idxs.append(j)

		if len(low_idxs) == 0:
			return base_zero, th.zeros([], dtype=th.float32, device=device)

		low_idxs = th.tensor(low_idxs, dtype=th.long, device=device)
		high_idxs = th.tensor(high_idxs, dtype=th.long, device=device)

		depth_diff = exp_depth[high_idxs] - exp_depth[low_idxs]
		rank_loss = th.relu(margin - depth_diff).mean()

		num_pairs = th.tensor(
			float(low_idxs.numel()),
			dtype=th.float32,
			device=device,
		)
		return rank_loss, num_pairs

	def _major_peak_reweight_logprobs(self, true_logprobs, true_batch_idxs, split):
		"""
		H1a: major-peak weighted target distribution.

		This only changes the training target distribution used by the spectrum loss.
		It does not change model forward, fragment support, sampler, OOS head, or evaluation targets.

		Original target:
		    p_i = true intensity probability

		H1a target:
		    p'_i = normalize(p_i * (1 + alpha * relative_intensity_i ** gamma))

		Then mix:
		    final_p = (1 - mix) * p + mix * p'
		"""
		if split != "train":
			return true_logprobs

		if not getattr(self.hparams, "use_major_peak_weighted_loss", False):
			return true_logprobs

		if getattr(self.hparams, "loss_type", "cross_entropy") != "cross_entropy":
			if not hasattr(self, "_h1a_loss_type_warned"):
				self._h1a_loss_type_warned = True
				print(f"[H1a] skipped because loss_type={self.hparams.loss_type}, expected cross_entropy")
			return true_logprobs

		if true_logprobs.numel() == 0:
			return true_logprobs

		eps = float(getattr(self.hparams, "log_min", 1e-12))
		eps = max(eps, 1e-12)

		alpha = float(getattr(self.hparams, "major_peak_weight_alpha", 0.5))
		gamma = float(getattr(self.hparams, "major_peak_weight_gamma", 0.5))
		clip = float(getattr(self.hparams, "major_peak_weight_clip", 3.0))
		mix = float(getattr(self.hparams, "major_peak_weight_mix", 0.5))
		mix = max(0.0, min(1.0, mix))

		batch_idxs = true_batch_idxs.long()
		p = true_logprobs.exp().clamp_min(eps)

		final_p = p.clone()

		num_specs = int(batch_idxs.max().item()) + 1

		weight_means = []
		weight_maxs = []

		for b in range(num_specs):
			mask = batch_idxs == b
			if not mask.any().item():
				continue

			pb = p[mask]
			pb = pb / pb.sum().clamp_min(eps)

			max_pb = pb.max().clamp_min(eps)
			rel = (pb / max_pb).clamp(min=0.0, max=1.0)

			peak_w = 1.0 + alpha * rel.pow(gamma)
			peak_w = peak_w.clamp(max=clip)

			weighted_pb = pb * peak_w
			weighted_pb = weighted_pb / weighted_pb.sum().clamp_min(eps)

			mixed_pb = (1.0 - mix) * pb + mix * weighted_pb
			mixed_pb = mixed_pb / mixed_pb.sum().clamp_min(eps)

			final_p[mask] = mixed_pb

			weight_means.append(peak_w.mean().detach())
			weight_maxs.append(peak_w.max().detach())

		if not hasattr(self, "_h1a_major_peak_logged"):
			self._h1a_major_peak_logged = True
			if len(weight_means) > 0:
				mean_w = th.stack(weight_means).mean().item()
				max_w = th.stack(weight_maxs).max().item()
			else:
				mean_w = 1.0
				max_w = 1.0

			print(
				"[H1a] major peak weighted target enabled: "
				f"alpha={alpha}, gamma={gamma}, clip={clip}, mix={mix}, "
				f"mean_peak_weight={mean_w:.4f}, max_peak_weight={max_w:.4f}"
			)

		return final_p.clamp_min(eps).log()

	def _binned_intensity_aux_loss(
		self,
		true_mzs,
		true_logprobs,
		true_batch_idxs,
		pred_mzs,
		pred_logprobs,
		pred_batch_idxs,
	):
		"""
		R4a: binned intensity auxiliary KL loss.

		This loss is applied only during training.
		It compares true/pred spectrum distributions after 0.01 Da binning.
		The goal is to distill the useful signal found by D2 intensity-aware calibration
		back into the main model training objective.

		Returns:
		    Tensor of shape [batch_size], same style as other per-spectrum losses.
		"""
		eps = float(getattr(self.hparams, "binned_intensity_aux_eps", 1e-9))
		eps = max(eps, 1e-12)

		mz_bin_res = float(
			getattr(
				self.hparams,
				"binned_intensity_aux_mz_bin_res",
				self.hparams.mz_bin_res,
			)
		)

		mz_max = float(self.hparams.mz_max)
		agg = "sum" if self.hparams.sum_ints else "amax"

		if true_mzs.numel() == 0 or pred_mzs.numel() == 0:
			batch_size = int(true_batch_idxs.max().item()) + 1
			return th.zeros(
				[batch_size],
				dtype=true_logprobs.dtype,
				device=true_logprobs.device,
			)

		true_probs = true_logprobs.exp()
		pred_probs = pred_logprobs.exp()

		true_bin_idxs, true_bin_probs, true_bin_batch_idxs = batched_bin_func(
			true_mzs,
			true_probs,
			true_batch_idxs,
			mz_max,
			mz_bin_res,
			agg,
			sparse=True,
		)

		pred_bin_idxs, pred_bin_probs, pred_bin_batch_idxs = batched_bin_func(
			pred_mzs,
			pred_probs,
			pred_batch_idxs,
			mz_max,
			mz_bin_res,
			agg,
			sparse=True,
		)

		true_bin_probs = scatter_l1normalize(
			true_bin_probs,
			true_bin_batch_idxs,
		).clamp_min(eps)

		pred_bin_probs = scatter_l1normalize(
			pred_bin_probs,
			pred_bin_batch_idxs,
		).clamp_min(eps)

		batch_size = int(true_batch_idxs.max().item()) + 1

		loss_vec = th.zeros(
			[batch_size],
			dtype=true_logprobs.dtype,
			device=true_logprobs.device,
		)

		for b in range(batch_size):
			t_mask = true_bin_batch_idxs == b
			p_mask = pred_bin_batch_idxs == b

			if not t_mask.any():
				continue

			t_bins = true_bin_idxs[t_mask].long()
			t_probs = true_bin_probs[t_mask]
			t_probs = t_probs / t_probs.sum().clamp_min(eps)

			if not p_mask.any():
				loss_vec[b] = -th.sum(t_probs * th.log(th.full_like(t_probs, eps)))
				continue

			p_bins = pred_bin_idxs[p_mask].long()
			p_probs = pred_bin_probs[p_mask]
			p_probs = p_probs / p_probs.sum().clamp_min(eps)

			# B1: soft-triangular bin matching.
			# Old behavior was exact-bin only:
			#     true bin k only sees predicted probability at bin k.
			# SnapGate diagnostics showed many useful peaks fall on neighboring
			# hard-0.01 bins, so optionally give k±r partial credit during training.
			soft_radius = int(getattr(self.hparams, "binned_intensity_aux_soft_radius", 0))
			soft_neighbor_weight = float(
				getattr(self.hparams, "binned_intensity_aux_soft_neighbor_weight", 0.0)
			)

			if soft_radius > 0 and soft_neighbor_weight > 0.0:
				dist = (t_bins.reshape(-1, 1) - p_bins.reshape(1, -1)).abs()

				# exact bin weight = 1.0
				match_w = dist.eq(0).to(dtype=p_probs.dtype)

				# neighbor bins get triangular-decayed partial weight.
				near = (dist > 0) & (dist <= soft_radius)
				if soft_radius <= 1:
					decay = th.ones_like(dist, dtype=p_probs.dtype)
				else:
					decay = 1.0 - (dist.to(dtype=p_probs.dtype) - 1.0) / float(soft_radius)
					decay = decay.clamp_min(0.0)

				near_w = soft_neighbor_weight * decay
				match_w = th.where(near, near_w, match_w)

				p_at_t = (match_w * p_probs.reshape(1, -1)).sum(dim=1)
			else:
				match = t_bins.reshape(-1, 1).eq(p_bins.reshape(1, -1))
				p_at_t = (match.to(dtype=p_probs.dtype) * p_probs.reshape(1, -1)).sum(dim=1)

			p_at_t = p_at_t.clamp(min=eps, max=1.0)

			ce = -th.sum(t_probs * p_at_t.log())
			ent = -th.sum(t_probs * t_probs.clamp_min(eps).log())

			# KL(true || pred) on binned spectrum support.
			loss_vec[b] = th.clamp(ce - ent, min=0.0)

		return loss_vec
	def _binned_false_mass_loss(
		self,
		true_mzs,
		true_logprobs,
		true_batch_idxs,
		pred_mzs,
		pred_logprobs,
		pred_batch_idxs,
	):
		"""
		Penalize predicted probability mass assigned to m/z bins that are absent
		from the true spectrum.

		This targets the failure mode:
		wrecall is high but cosine is near zero, meaning support exists but
		probability mass is allocated to wrong / false bins.
		"""
		eps = float(self.hparams.log_min)
		mz_bin_res = float(
			getattr(
				self.hparams,
				"binned_false_mass_mz_bin_res",
				self.hparams.mz_bin_res,
			)
		)

		mz_max = float(self.hparams.mz_max)
		agg = "sum" if self.hparams.sum_ints else "amax"

		if true_mzs.numel() == 0 or pred_mzs.numel() == 0:
			batch_size = int(true_batch_idxs.max().item()) + 1
			return th.zeros(
				[batch_size],
				dtype=true_logprobs.dtype,
				device=true_logprobs.device,
			)

		true_probs = true_logprobs.exp()
		pred_probs = pred_logprobs.exp()

		true_bin_idxs, true_bin_probs, true_bin_batch_idxs = batched_bin_func(
			true_mzs,
			true_probs,
			true_batch_idxs,
			mz_max,
			mz_bin_res,
			agg,
			sparse=True,
		)

		pred_bin_idxs, pred_bin_probs, pred_bin_batch_idxs = batched_bin_func(
			pred_mzs,
			pred_probs,
			pred_batch_idxs,
			mz_max,
			mz_bin_res,
			agg,
			sparse=True,
		)

		pred_bin_probs = scatter_l1normalize(
			pred_bin_probs,
			pred_bin_batch_idxs,
		).clamp_min(eps)

		batch_size = int(true_batch_idxs.max().item()) + 1

		loss_vec = th.zeros(
			[batch_size],
			dtype=true_logprobs.dtype,
			device=true_logprobs.device,
		)

		for b in range(batch_size):
			t_mask = true_bin_batch_idxs == b
			p_mask = pred_bin_batch_idxs == b

			if not p_mask.any():
				continue

			p_bins = pred_bin_idxs[p_mask].long()
			p_probs = pred_bin_probs[p_mask]
			p_probs = p_probs / p_probs.sum().clamp_min(eps)

			if not t_mask.any():
				loss_vec[b] = p_probs.sum()
				continue

			t_bins = true_bin_idxs[t_mask].long()

			match = p_bins.reshape(-1, 1).eq(t_bins.reshape(1, -1))
			is_true_bin = match.any(dim=1)

			false_mass = p_probs[~is_true_bin].sum()
			loss_vec[b] = false_mass

		return loss_vec









	def _r117_support_oracle_reweight_loss(
		self,
		true_mzs,
		true_logprobs,
		true_batch_idxs,
		pred_mzs,
		pred_logprobs,
		pred_batch_idxs,
		batch_size,
	):
		"""
		R117 train-only support-conditional oracle reweight loss.

		For each spectrum, keep the current predicted m/z support fixed.
		Use the true spectrum mass on the covered predicted bins as an oracle target.
		This directly attacks the large gap:
		current raw cosine << fixed-support oracle cosine.

		Legal: train-only, uses only the current batch labels.
		"""
		device = pred_logprobs.device
		dtype = pred_logprobs.dtype

		bs = int(batch_size.detach().cpu().item()) if hasattr(batch_size, "detach") else int(batch_size)

		every_n = int(getattr(self.hparams, "r117_support_oracle_every_n_steps", 1))
		every_n = max(1, every_n)
		if int(self.global_step) % every_n != 0:
			z = th.zeros([bs], dtype=dtype, device=device)
			return z, z.mean(), z.mean(), z.mean(), z.mean()

		bin_res = float(getattr(self.hparams, "r117_oracle_bin_res", 0.01))
		false_w = float(getattr(self.hparams, "r117_false_mass_weight", 0.25))
		min_covered = float(getattr(self.hparams, "r117_min_covered_true_mass", 1.0e-6))
		eps = float(getattr(self.hparams, "r117_eps", 1.0e-12))

		loss_vec = th.zeros([bs], dtype=dtype, device=device)
		valid_vec = th.zeros([bs], dtype=dtype, device=device)
		covered_vec = th.zeros([bs], dtype=dtype, device=device)
		false_vec = th.zeros([bs], dtype=dtype, device=device)
		target_bins_vec = th.zeros([bs], dtype=dtype, device=device)

		true_probs = true_logprobs.exp()
		pred_probs = pred_logprobs.exp()

		for bi in range(bs):
			tmask = true_batch_idxs == bi
			pmask = pred_batch_idxs == bi

			if not tmask.any() or not pmask.any():
				continue

			t_mz = true_mzs[tmask].to(device=device)
			t_prob = true_probs[tmask].to(device=device, dtype=dtype)
			p_mz = pred_mzs[pmask].to(device=device)
			p_logp = pred_logprobs[pmask].to(device=device, dtype=dtype)
			p_prob = pred_probs[pmask].to(device=device, dtype=dtype)

			if t_mz.numel() == 0 or p_mz.numel() == 0:
				continue

			t_bins = th.round(t_mz / bin_res).to(dtype=th.long)
			p_bins = th.round(p_mz / bin_res).to(dtype=th.long)

			p_unique, p_inv = th.unique(p_bins, sorted=False, return_inverse=True)

			# Aggregate predicted probability mass by predicted bin.
			p_bin_prob = th.zeros([p_unique.numel()], dtype=dtype, device=device)
			p_bin_prob.scatter_add_(0, p_inv, p_prob)
			p_bin_logp = p_bin_prob.clamp_min(eps).log()

			# True mass on each predicted bin.
			eq = p_unique.reshape(-1, 1).eq(t_bins.reshape(1, -1))
			target_mass = (eq.to(dtype=dtype) * t_prob.reshape(1, -1)).sum(dim=1)

			covered_mass = target_mass.sum()
			if covered_mass <= min_covered:
				continue

			target_prob = target_mass / covered_mass.clamp_min(eps)
			target_keep = target_prob > 0

			if not target_keep.any():
				continue

			# KL(target || pred_on_current_support). This pushes mass onto true-covered bins.
			tp = target_prob[target_keep]
			lp = p_bin_logp[target_keep]
			kl = (tp * (tp.clamp_min(eps).log() - lp)).sum()

			# Explicit penalty for predicted mass on bins without true support.
			false_mass = p_bin_prob[~target_keep].sum()

			loss_vec[bi] = kl + false_w * false_mass
			valid_vec[bi] = 1.0
			covered_vec[bi] = covered_mass.detach()
			false_vec[bi] = false_mass.detach()
			target_bins_vec[bi] = target_keep.to(dtype=dtype).sum().detach()

		valid_frac = valid_vec.mean()
		covered_mean = covered_vec[valid_vec > 0].mean() if valid_vec.any() else covered_vec.mean()
		false_mean = false_vec[valid_vec > 0].mean() if valid_vec.any() else false_vec.mean()
		target_bins_mean = target_bins_vec[valid_vec > 0].mean() if valid_vec.any() else target_bins_vec.mean()

		return loss_vec, valid_frac, covered_mean, false_mean, target_bins_mean

	def _rendered_peak_gate_bce_loss(
		self,
		true_mzs,
		true_logprobs,
		true_batch_idxs,
		pred_mzs,
		pred_logprobs,
		pred_batch_idxs,
		pred_d,
		batch_size,
	):
		"""R71 rendered peak hit/false gate loss."""
		gate_logits = pred_d.get("pred_rendered_peak_gate_logits", None)
		base_zero = pred_logprobs.sum() * 0.0

		if gate_logits is None:
			device = pred_logprobs.device
			z = th.zeros([], dtype=pred_logprobs.dtype, device=device)
			return base_zero, z, z, z

		device = pred_logprobs.device
		dtype = pred_logprobs.dtype
		bs = int(batch_size.detach().cpu().item()) if hasattr(batch_size, "detach") else int(batch_size)
		gate_logits = gate_logits.reshape(-1).to(device=device, dtype=dtype)

		if gate_logits.numel() != pred_mzs.numel():
			z = th.zeros([], dtype=dtype, device=device)
			return base_zero, z, z, z

		match_tol = float(getattr(self.hparams, "rendered_peak_gate_match_tol", 0.006))
		global_shift = float(getattr(self.hparams, "rendered_peak_gate_global_shift", 0.0005))
		min_true_prob = float(getattr(self.hparams, "rendered_peak_gate_min_true_prob", 0.0005))
		max_pos = int(getattr(self.hparams, "rendered_peak_gate_max_pos_per_spec", 128))
		max_neg = int(getattr(self.hparams, "rendered_peak_gate_max_neg_per_spec", 512))
		focal_gamma = float(getattr(self.hparams, "rendered_peak_gate_focal_gamma", 2.0))
		pos_weight = float(getattr(self.hparams, "rendered_peak_gate_pos_weight", 8.0))

		loss_vec = th.zeros([bs], dtype=dtype, device=device)
		valid_vec = th.zeros([bs], dtype=dtype, device=device)
		pos_rate_vec = th.zeros([bs], dtype=dtype, device=device)
		used_vec = th.zeros([bs], dtype=dtype, device=device)
		true_probs = true_logprobs.exp()

		for b in range(bs):
			tmask = true_batch_idxs == b
			pmask = pred_batch_idxs == b

			if not tmask.any() or not pmask.any():
				continue

			t_mz = true_mzs[tmask].to(device=device, dtype=dtype)
			t_prob = true_probs[tmask].to(device=device, dtype=dtype)

			if min_true_prob > 0.0:
				t_keep = t_prob >= min_true_prob
				t_mz = t_mz[t_keep]
				t_prob = t_prob[t_keep]

			if t_mz.numel() == 0:
				continue

			p_idx = th.nonzero(pmask, as_tuple=False).reshape(-1)
			p_mz = pred_mzs[p_idx].to(device=device, dtype=dtype) + global_shift
			p_score = pred_logprobs[p_idx].to(device=device, dtype=dtype)
			p_gate = gate_logits[p_idx]

			dist = (p_mz.reshape(-1, 1) - t_mz.reshape(1, -1)).abs()
			hit = dist.le(match_tol).any(dim=1)
			pos_idx = th.nonzero(hit, as_tuple=False).reshape(-1)
			neg_idx = th.nonzero(~hit, as_tuple=False).reshape(-1)

			if pos_idx.numel() == 0 or neg_idx.numel() == 0:
				continue

			if pos_idx.numel() > max_pos:
				_, nearest = dist[pos_idx].min(dim=1)
				pos_w = t_prob[nearest]
				top = th.topk(pos_w, k=max_pos, largest=True).indices
				pos_idx = pos_idx[top]

			if neg_idx.numel() > max_neg:
				top = th.topk(p_score[neg_idx], k=max_neg, largest=True).indices
				neg_idx = neg_idx[top]

			keep = th.cat([pos_idx, neg_idx], dim=0)
			y = hit[keep].to(dtype=dtype)
			logits = p_gate[keep]

			bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
			prob = th.sigmoid(logits)
			pt = th.where(y > 0.5, prob, 1.0 - prob)
			focal = (1.0 - pt).clamp_min(0.0).pow(focal_gamma)
			weight = th.where(y > 0.5, th.ones_like(y) * pos_weight, th.ones_like(y))

			loss_vec[b] = (bce * focal * weight).mean()
			valid_vec[b] = 1.0
			pos_rate_vec[b] = y.mean()
			used_vec[b] = float(int(keep.numel()))

		return (
			loss_vec + base_zero,
			valid_vec.mean(),
			pos_rate_vec.mean(),
			used_vec.mean(),
		)

	def _r58c_offset_channel_aux_loss(
		self,
		true_mzs,
		true_logprobs,
		true_batch_idxs,
		pred_mzs,
		pred_logprobs,
		pred_batch_idxs,
		pred_d,
		batch_size,
	):
		"""R58C train-only supervised local offset-channel loss.
		
		Each original peak-entry is expanded into 5 offset channels.
		This loss teaches the allocator which channel is closest to a true peak.
		"""
		group_idxs = pred_d.get("pred_spec_offset_group_idxs", None)
		peak_channels = pred_d.get("pred_spec_peak_channels", None)
		base_zero = pred_logprobs.sum() * 0.0
		device = pred_logprobs.device
		dtype = pred_logprobs.dtype
		bs = int(batch_size.detach().cpu().item()) if hasattr(batch_size, "detach") else int(batch_size)
		loss_vec = th.zeros([bs], dtype=dtype, device=device)
		valid_vec = th.zeros([bs], dtype=dtype, device=device)
		target_sum_vec = th.zeros([bs], dtype=dtype, device=device)
		target_cnt_vec = th.zeros([bs], dtype=dtype, device=device)
		if group_idxs is None or peak_channels is None:
			return loss_vec + base_zero, valid_vec.mean(), target_sum_vec.mean()
		group_idxs = group_idxs.to(device=device).long().reshape(-1)
		peak_channels = peak_channels.to(device=device).long().reshape(-1)
		if group_idxs.numel() != pred_mzs.numel():
			return loss_vec + base_zero, valid_vec.mean(), target_sum_vec.mean()
		num_channels = int(getattr(self.hparams, "r58c_offset_num_channels", 5))
		if num_channels <= 1:
			return loss_vec + base_zero, valid_vec.mean(), target_sum_vec.mean()
		total_n = int(group_idxs.numel())
		if total_n < num_channels or total_n % num_channels != 0:
			return loss_vec + base_zero, valid_vec.mean(), target_sum_vec.mean()
		num_groups = total_n // num_channels
		try:
			g_view = group_idxs.view(num_groups, num_channels)
			mz_view = pred_mzs.view(num_groups, num_channels)
			logit_view = pred_logprobs.view(num_groups, num_channels)
			b_view = pred_batch_idxs.view(num_groups, num_channels).long()
		except Exception:
			return loss_vec + base_zero, valid_vec.mean(), target_sum_vec.mean()
		same_group = (g_view == g_view[:, :1]).all(dim=1)
		same_batch = (b_view == b_view[:, :1]).all(dim=1)
		valid_shape = same_group & same_batch
		if not valid_shape.any():
			return loss_vec + base_zero, valid_vec.mean(), target_sum_vec.mean()
		group_batch = b_view[:, 0].clamp(0, bs - 1)
		match_tol = float(getattr(self.hparams, "r58c_offset_aux_match_tol", 0.006))
		global_shift = float(getattr(self.hparams, "r58c_offset_aux_global_shift", 0.0005))
		min_true_prob = float(getattr(self.hparams, "r58c_offset_aux_min_true_prob", 0.0005))
		weight_gamma = float(getattr(self.hparams, "r58c_offset_aux_weight_gamma", 0.5))
		max_groups_per_spec = int(getattr(self.hparams, "r58c_offset_aux_max_groups_per_spec", 2048))
		mz_for_label = mz_view + global_shift
		for b in range(bs):
			gmask = valid_shape & (group_batch == b)
			if not gmask.any():
				continue
			tmask = true_batch_idxs == b
			if not tmask.any():
				continue
			t_mz = true_mzs[tmask].to(device=device, dtype=mz_for_label.dtype)
			t_logp = true_logprobs[tmask].to(device=device, dtype=dtype)
			t_prob = t_logp.exp()
			t_keep = t_prob >= min_true_prob
			if not t_keep.any():
				continue
			t_mz = t_mz[t_keep]
			t_prob = t_prob[t_keep]
			g_idx = th.nonzero(gmask, as_tuple=False).reshape(-1)
			mz_b = mz_for_label[g_idx]
			logits_b = logit_view[g_idx]
			dist = (mz_b.unsqueeze(2) - t_mz.view(1, 1, -1)).abs()
			min_dist_chan, nearest_true_idx = dist.min(dim=2)
			group_min_dist, target_ch = min_dist_chan.min(dim=1)
			valid = group_min_dist <= match_tol
			if not valid.any():
				continue
			row_idx = th.arange(g_idx.numel(), device=device)
			nearest_idx_for_target = nearest_true_idx[row_idx, target_ch]
			group_weight = t_prob[nearest_idx_for_target].clamp_min(1e-8).pow(weight_gamma)
			if max_groups_per_spec > 0 and int(valid.sum().item()) > max_groups_per_spec:
				valid_idxs = th.nonzero(valid, as_tuple=False).reshape(-1)
				w_valid = group_weight[valid_idxs]
				topk = th.topk(w_valid, k=max_groups_per_spec, largest=True).indices
				new_valid = th.zeros_like(valid)
				new_valid[valid_idxs[topk]] = True
				valid = new_valid
			logp = th.log_softmax(logits_b[valid], dim=1)
			target = target_ch[valid].long()
			w = group_weight[valid].to(dtype=dtype)
			ce = -logp.gather(1, target.view(-1, 1)).squeeze(1)
			spec_loss = (ce * w).sum() / w.sum().clamp_min(1e-8)
			loss_vec[b] = spec_loss
			valid_vec[b] = valid.float().mean()
			target_sum_vec[b] = target.float().mean()
			target_cnt_vec[b] = 1.0
		target_mean = target_sum_vec.sum() / target_cnt_vec.sum().clamp_min(1.0)
		return loss_vec + base_zero, valid_vec.mean(), target_mean
	def _r98_apply_binned_spectrum_renderer(
		self,
		pred_mzs,
		pred_logprobs,
		pred_batch_idxs,
		batch_size,
		split,
	):
		"""R98: merge final rendered spectrum entries into 0.01-Da bins."""
		if not getattr(self.hparams, "use_binned_spectrum_renderer", False):
			return pred_mzs, pred_logprobs, pred_batch_idxs, None
		if split == "train" and (not getattr(self.hparams, "binned_spectrum_renderer_apply_train", False)):
			return pred_mzs, pred_logprobs, pred_batch_idxs, None

		if pred_mzs.numel() == 0:
			return pred_mzs, pred_logprobs, pred_batch_idxs, None

		bin_res = float(getattr(self.hparams, "binned_spectrum_renderer_bin_res", 0.01))
		bin_res = max(bin_res, 1.0e-12)
		max_bins = int(getattr(self.hparams, "binned_spectrum_renderer_max_bins", 0))
		preserve_mass = bool(getattr(self.hparams, "binned_spectrum_renderer_preserve_mass", True))
		bs = int(batch_size.detach().cpu().item()) if hasattr(batch_size, 'detach') else int(batch_size)
		device = pred_mzs.device
		dtype = pred_logprobs.dtype

		valid = (
			th.isfinite(pred_mzs)
			& th.isfinite(pred_logprobs)
			& (pred_mzs >= 0.0)
			& (pred_mzs < float(self.hparams.mz_max))
		)
		if not valid.all():
			pred_mzs = pred_mzs[valid]
			pred_logprobs = pred_logprobs[valid]
			pred_batch_idxs = pred_batch_idxs[valid]

		if pred_mzs.numel() == 0:
			return pred_mzs, pred_logprobs, pred_batch_idxs, None

		old_lse = scatter_logsumexp(
			pred_logprobs,
			pred_batch_idxs,
			dim_size=bs,
		)

		bin_ids = th.round(pred_mzs.float() / bin_res).long()
		pair = th.stack([pred_batch_idxs.long(), bin_ids], dim=1)
		uniq_pair, inv = th.unique(pair, dim=0, return_inverse=True)
		merged_logprobs = scatter_logsumexp(
			pred_logprobs,
			inv,
			dim_size=uniq_pair.shape[0],
		)
		merged_batch_idxs = uniq_pair[:, 0].long()
		merged_mzs = uniq_pair[:, 1].to(dtype=pred_mzs.dtype) * bin_res

		if max_bins > 0:
			keep_parts = []
			for b in range(bs):
				idx = th.nonzero(merged_batch_idxs == b, as_tuple=False).reshape(-1)
				if idx.numel() == 0:
					continue
				if idx.numel() > max_bins:
					rel = th.topk(merged_logprobs[idx], k=max_bins, largest=True).indices
					idx = idx[rel]
				keep_parts.append(idx)
			if len(keep_parts) > 0:
				keep = th.cat(keep_parts, dim=0)
				merged_mzs = merged_mzs[keep]
				merged_logprobs = merged_logprobs[keep]
				merged_batch_idxs = merged_batch_idxs[keep]

		if preserve_mass:
			new_lse = scatter_logsumexp(
				merged_logprobs,
				merged_batch_idxs,
				dim_size=bs,
			)
			merged_logprobs = merged_logprobs - new_lse[merged_batch_idxs] + old_lse[merged_batch_idxs]

		kept_frac = th.tensor(
			float(merged_logprobs.numel()) / max(float(pred_logprobs.numel()), 1.0),
			device=device,
			dtype=dtype,
		)
		return merged_mzs, merged_logprobs, merged_batch_idxs, kept_frac


	def _common_step(self, batch, split="train", log=True):

		# preprocess spec
		batch_size = batch["batch_size"]
		unique_id = batch["spec_unique_id"]
		smiles = batch["mol_smiles"]
		true_mzs = batch["spec_mzs"]
		true_ints = batch["spec_ints"]
		true_batch_idxs = batch["spec_batch_idxs"]
		true_prec_mzs = batch["spec_prec_mz"]
		true_ces = batch.get("spec_ce", None)
		spec_per_group = batch["spec_per_group"]
		spec_per_mol = batch["spec_per_mol"]
		group_per_mol = batch["group_per_mol"]
		pred_d = self.forward(**batch)
		pred_mzs = pred_d.pop("pred_mzs")
		pred_logprobs = pred_d.pop("pred_logprobs")
		pred_batch_idxs = pred_d.pop("pred_batch_idxs")
		pred_oos_logprobs = pred_d.pop("pred_oos_logprobs",None)
		# R110B: keep pre-R98 rendered peak entries for rendered-peak gate BCE.
		r110_rendered_pred_mzs = pred_mzs
		r110_rendered_pred_logprobs = pred_logprobs
		r110_rendered_pred_batch_idxs = pred_batch_idxs

		# R113A: optional pre-R98 rendered-entry hard prune.
		# Eval-only diagnostic: remove low-confidence rendered entries before R98 bin merge.
		if bool(getattr(self.hparams, "use_r113_rendered_peak_hard_prune", False)):
			r113_delta = pred_d.get("pred_rendered_peak_gate_delta", None)
			r113_logits = pred_d.get("pred_rendered_peak_gate_logits", None)

			if r113_delta is not None and r113_delta.numel() == pred_mzs.numel():
				r113_mode = str(getattr(self.hparams, "r113_rendered_peak_hard_prune_mode", "delta"))
				r113_min_delta = float(getattr(self.hparams, "r113_rendered_peak_hard_prune_min_delta", -0.20))
				r113_min_prob = float(getattr(self.hparams, "r113_rendered_peak_hard_prune_min_prob", 0.50))
				r113_min_keep = int(getattr(self.hparams, "r113_rendered_peak_hard_prune_min_keep_per_spec", 64))

				if r113_mode == "prob" and r113_logits is not None and r113_logits.numel() == pred_mzs.numel():
					r113_score = th.sigmoid(r113_logits.reshape(-1).to(device=pred_logprobs.device, dtype=pred_logprobs.dtype))
					r113_keep = r113_score >= r113_min_prob
				else:
					r113_score = r113_delta.reshape(-1).to(device=pred_logprobs.device, dtype=pred_logprobs.dtype)
					r113_keep = r113_score >= r113_min_delta

				# Guard: keep at least K entries per spectrum by best gate score.
				if r113_min_keep > 0 and pred_batch_idxs.numel() > 0:
					for _b in pred_batch_idxs.unique(sorted=False):
						_bmask = pred_batch_idxs == _b
						_bidx = th.nonzero(_bmask, as_tuple=False).reshape(-1)
						if _bidx.numel() == 0:
							continue
						_need = min(r113_min_keep, int(_bidx.numel()))
						_have = int(r113_keep[_bidx].sum().detach().cpu().item())
						if _have < _need:
							_top = th.topk(r113_score[_bidx], k=_need, largest=True).indices
							r113_keep[_bidx[_top]] = True

				# Apply prune only if it removes something and does not empty the batch.
				if r113_keep.any() and int(r113_keep.sum().detach().cpu().item()) < int(r113_keep.numel()):
					pred_mzs = pred_mzs[r113_keep]
					pred_logprobs = pred_logprobs[r113_keep]
					pred_batch_idxs = pred_batch_idxs[r113_keep]

					# Renormalize per spectrum after pruning.
					_r113_norm = scatter_logsumexp(pred_logprobs, pred_batch_idxs)
					pred_logprobs = pred_logprobs - _r113_norm[pred_batch_idxs]
		# ===== R98: binned spectrum renderer inside model common_step =====
		r98_kept_frac = None
		pred_mzs, pred_logprobs, pred_batch_idxs, r98_kept_frac = self._r98_apply_binned_spectrum_renderer(
			pred_mzs=pred_mzs,
			pred_logprobs=pred_logprobs,
			pred_batch_idxs=pred_batch_idxs,
			batch_size=batch_size,
			split=split,
		)
		if r98_kept_frac is not None:
			pred_d["pred_binned_spectrum_renderer_kept_frac"] = r98_kept_frac
			if log:
				self.log(f"{split}_binned_spectrum_renderer_kept_frac", r98_kept_frac.detach(), batch_size=batch_size, on_step=False, on_epoch=True)
		if self.ce_bin_residual_head is not None:
			pred_mzs, pred_logprobs, pred_batch_idxs, ce_bin_residual_stats = apply_ce_bin_residual(
				head=self.ce_bin_residual_head,
				hparams=self.hparams,
				batch=batch,
				pred_mzs=pred_mzs,
				pred_logprobs=pred_logprobs,
				pred_batch_idxs=pred_batch_idxs,
				true_prec_mzs=true_prec_mzs,
				batch_size=batch_size,
			)
			pred_d["pred_r107_ce_bin_delta_abs_mean"] = ce_bin_residual_stats["delta_abs_mean"]
			if log:
				self.log(
					f"{split}_r107_ce_bin_delta_abs_mean",
					ce_bin_residual_stats["delta_abs_mean"],
					batch_size=batch_size,
					on_step=False,
					on_epoch=True,
				)
				self.log(
					f"{split}_r107_ce_bin_delta_mean",
					ce_bin_residual_stats["delta_mean"],
					batch_size=batch_size,
					on_step=False,
					on_epoch=True,
				)
  		# prepare training dict (TODO: do preprocessing in log space)
		train_true_mzs, train_true_logprobs, train_true_batch_idxs = self.preproc_spec(
			true_mzs,
			true_ints,
			true_batch_idxs,
			train=True,
			pred=False,
			log_in=False,
			log_out=True
		)
		train_pred_mzs, train_pred_logprobs, train_pred_batch_idxs = self.preproc_spec(
			pred_mzs,
			pred_logprobs,
			pred_batch_idxs,
			train=True,
			pred=True,
			log_in=True,
			log_out=True
		)

		# H1a: reweight only the training target distribution.
		# Evaluation below still uses original true_ints / true_mzs.
		train_true_logprobs = self._major_peak_reweight_logprobs(
			train_true_logprobs,
			train_true_batch_idxs,
			split=split
		)

		train_d = {
			"true_mzs": train_true_mzs,
			"true_logprobs": train_true_logprobs,
			"true_batch_idxs": train_true_batch_idxs,
			"pred_mzs": train_pred_mzs,
			"pred_logprobs": train_pred_logprobs,
			"pred_batch_idxs": train_pred_batch_idxs,
			"pred_oos_logprobs": pred_oos_logprobs,
			**pred_d
		}
		loss_d = self.loss_fn(**train_d)
		loss = loss_d["loss"]
		# ===== R71: rendered peak-entry drop gate BCE =====
		if split == "train" and getattr(self.hparams, "use_rendered_peak_gate_loss", False):
			r71_loss_vec, r71_valid_frac, r71_pos_rate, r71_used = self._rendered_peak_gate_bce_loss(
				true_mzs=train_true_mzs,
				true_logprobs=train_true_logprobs,
				true_batch_idxs=train_true_batch_idxs,
				pred_mzs=r110_rendered_pred_mzs,
				pred_logprobs=r110_rendered_pred_logprobs,
				pred_batch_idxs=r110_rendered_pred_batch_idxs,
				pred_d=pred_d,
				batch_size=batch_size,
			)
			r71_loss_vec = th.nan_to_num(r71_loss_vec, nan=0.0, posinf=10.0, neginf=0.0)
			r71_w = float(getattr(self.hparams, "rendered_peak_gate_loss_weight", 0.002))
			loss = loss + r71_w * r71_loss_vec
			if log:
				self.log("train_rendered_peak_gate_bce_loss", r71_loss_vec.detach().mean(), batch_size=batch_size, on_step=False, on_epoch=True)
				self.log("train_rendered_peak_gate_valid_frac", r71_valid_frac.detach(), batch_size=batch_size, on_step=False, on_epoch=True)
				self.log("train_rendered_peak_gate_pos_rate", r71_pos_rate.detach(), batch_size=batch_size, on_step=False, on_epoch=True)
				self.log("train_rendered_peak_gate_used", r71_used.detach(), batch_size=batch_size, on_step=False, on_epoch=True)
		# ===== R64: true-window distribution loss =====
		if split == "train" and getattr(self.hparams, "use_r64_true_window_dist_loss", False):
			r64_loss_vec, r64_outside_vec, r64_valid_frac, r64_support_frac, r64_pred_used = (
				true_window_distribution_loss(
					true_mzs=train_true_mzs,
					true_logprobs=train_true_logprobs,
					true_batch_idxs=train_true_batch_idxs,
					pred_mzs=train_pred_mzs,
					pred_logprobs=train_pred_logprobs,
					pred_batch_idxs=train_pred_batch_idxs,
					batch_size=batch_size,
					match_tol=float(getattr(self.hparams, "r64_match_tol", 0.006)),
					sigma=float(getattr(self.hparams, "r64_sigma", 0.002)),
					max_pred_per_spec=int(getattr(self.hparams, "r64_max_pred_per_spec", 1024)),
					max_target_pred_per_spec=int(getattr(self.hparams, "r64_max_target_pred_per_spec", 512)),
					max_true_per_spec=int(getattr(self.hparams, "r64_max_true_per_spec", 256)),
					min_true_prob=float(getattr(self.hparams, "r64_min_true_prob", 0.0)),
					min_target_mass=float(getattr(self.hparams, "r64_min_target_mass", 1.0e-12)),
					pred_temperature=float(getattr(self.hparams, "r64_pred_temperature", 1.0)),
					chunk_size=int(getattr(self.hparams, "r64_chunk_size", 2048)),
				)
			)

			r64_loss_vec = th.nan_to_num(
				r64_loss_vec,
				nan=0.0,
				posinf=20.0,
				neginf=0.0,
			)
			r64_outside_vec = th.nan_to_num(
				r64_outside_vec,
				nan=0.0,
				posinf=1.0,
				neginf=0.0,
			)

			r64_w = float(getattr(self.hparams, "r64_true_window_loss_weight", 0.02))
			r64_out_w = float(getattr(self.hparams, "r64_outside_mass_loss_weight", 0.005))

			loss = loss + r64_w * r64_loss_vec + r64_out_w * r64_outside_vec

			if log:
				self.log(
					"train_r64_true_window_dist_loss",
					r64_loss_vec.detach().mean(),
					batch_size=batch_size,
					on_step=False,
					on_epoch=True,
				)
				self.log(
					"train_r64_outside_mass_loss",
					r64_outside_vec.detach().mean(),
					batch_size=batch_size,
					on_step=False,
					on_epoch=True,
				)
				self.log(
					"train_r64_valid_frac",
					r64_valid_frac.detach(),
					batch_size=batch_size,
					on_step=False,
					on_epoch=True,
				)
				self.log(
					"train_r64_support_frac",
					r64_support_frac.detach(),
					batch_size=batch_size,
					on_step=False,
					on_epoch=True,
				)
				self.log(
					"train_r64_pred_used",
					r64_pred_used.detach(),
					batch_size=batch_size,
					on_step=False,
					on_epoch=True,
				)
		# ===== R58C: train-only supervised local offset-channel loss =====
		if split == "train" and getattr(self.hparams, "use_r58c_offset_channel_aux_loss", False):
			offset_loss_vec, offset_valid_frac, offset_target_mean = self._r58c_offset_channel_aux_loss(
				true_mzs=train_true_mzs,
				true_logprobs=train_true_logprobs,
				true_batch_idxs=train_true_batch_idxs,
				pred_mzs=pred_mzs,
				pred_logprobs=pred_logprobs,
				pred_batch_idxs=pred_batch_idxs,
				pred_d=pred_d,
				batch_size=batch_size,
			)
			offset_loss_vec = th.nan_to_num(offset_loss_vec, nan=0.0, posinf=10.0, neginf=0.0)
			offset_w = float(getattr(self.hparams, "r58c_offset_aux_weight", 0.02))
			loss = loss + offset_w * offset_loss_vec
			if log:
				self.log("train_r58c_offset_channel_loss", offset_loss_vec.detach().mean(), batch_size=batch_size, on_step=False, on_epoch=True)
				self.log("train_r58c_offset_channel_valid_frac", offset_valid_frac.detach(), batch_size=batch_size, on_step=False, on_epoch=True)
				self.log("train_r58c_offset_channel_target_mean", offset_target_mean.detach(), batch_size=batch_size, on_step=False, on_epoch=True)

		# ===== R28: oracle/reference teacher bin distillation =====
		if split == "train" and getattr(self.hparams, "use_oracle_teacher_bin_loss", False):
			teacher_loss_vec, teacher_valid = oracle_teacher_bin_loss(
				self,
				unique_id=unique_id,
				pred_mzs=train_pred_mzs,
				pred_logprobs=train_pred_logprobs,
				pred_batch_idxs=train_pred_batch_idxs,
			)

			teacher_loss_vec = th.nan_to_num(
				teacher_loss_vec,
				nan=0.0,
				posinf=10.0,
				neginf=0.0,
			)

			teacher_w = float(getattr(self.hparams, "oracle_teacher_bin_loss_weight", 0.0008))
			loss = loss + teacher_w * teacher_loss_vec

			if log:
				if teacher_valid.any():
					teacher_log_value = teacher_loss_vec[teacher_valid].detach().mean()
				else:
					teacher_log_value = teacher_loss_vec.detach().mean()

				self.log(
					"train_oracle_teacher_bin_loss",
					teacher_log_value,
					batch_size=batch_size,
					on_step=False,
					on_epoch=True,
				)
				self.log(
					"train_oracle_teacher_valid_frac",
					teacher_valid.detach().float().mean(),
					batch_size=batch_size,
					on_step=False,
					on_epoch=True,
				)
		# ===== R7: CE-weighted binned intensity auxiliary loss =====
		# Diagnostic result:
		# CE 20~40 and 40~60 spectra have much lower binned cosine.
		# This term directly aligns training with binned spectrum similarity.
		if split == "train" and getattr(self.hparams, "use_ce_weighted_binned_aux_loss", False):
			if true_ces is None:
				raise RuntimeError("use_ce_weighted_binned_aux_loss=True requires spec_ce in batch")

			binned_aux_loss_vec = self._binned_intensity_aux_loss(
				true_mzs=train_true_mzs,
				true_logprobs=train_true_logprobs,
				true_batch_idxs=train_true_batch_idxs,
				pred_mzs=train_pred_mzs,
				pred_logprobs=train_pred_logprobs,
				pred_batch_idxs=train_pred_batch_idxs,
			)

			binned_aux_loss_vec = th.nan_to_num(
				binned_aux_loss_vec,
				nan=0.0,
				posinf=10.0,
				neginf=0.0,
			)

			ce_vec = true_ces.reshape(-1).to(device=loss.device, dtype=loss.dtype)
			assert ce_vec.shape[0] == loss.shape[0], (ce_vec.shape, loss.shape)

			valid_ce = th.isfinite(ce_vec)
			ce_w = th.ones_like(loss)

			mid_thr = float(getattr(self.hparams, "ce_binned_aux_mid_threshold", 20.0))
			high_thr = float(getattr(self.hparams, "ce_binned_aux_high_threshold", 40.0))

			low_w = float(getattr(self.hparams, "ce_binned_aux_low_weight", 0.5))
			mid_w = float(getattr(self.hparams, "ce_binned_aux_mid_weight", 2.0))
			high_w = float(getattr(self.hparams, "ce_binned_aux_high_weight", 3.0))

			# low CE already performs well; do not over-optimize it.
			ce_w = th.where(
				valid_ce & (ce_vec < mid_thr),
				ce_w * low_w,
				ce_w,
			)

			# CE >= 20 is the main weak region.
			ce_w = th.where(
				valid_ce & (ce_vec >= mid_thr),
				th.ones_like(ce_w) * mid_w,
				ce_w,
			)

			# CE >= 40 is the worst region, but small; give a little extra.
			ce_w = th.where(
				valid_ce & (ce_vec >= high_thr),
				th.ones_like(ce_w) * high_w,
				ce_w,
			)

			aux_w = float(getattr(self.hparams, "ce_binned_aux_loss_weight", 0.003))
			loss = loss + aux_w * ce_w * binned_aux_loss_vec

			if log:
				self.log(
					"train_ce_binned_aux_loss",
					binned_aux_loss_vec.detach().mean(),
					batch_size=batch_size,
					on_step=False,
					on_epoch=True,
				)
				self.log(
					"train_ce_binned_aux_weight",
					ce_w.detach().mean(),
					batch_size=batch_size,
					on_step=False,
					on_epoch=True,
				)
		# ===== R8: binned false-mass penalty =====
		# Directly penalizes probability mass on predicted bins not present in the true spectrum.
		# This targets spectra with high wrecall but near-zero cosine.
		if split == "train" and getattr(self.hparams, "use_binned_false_mass_loss", False):
			false_mass_loss_vec = self._binned_false_mass_loss(
				true_mzs=train_true_mzs,
				true_logprobs=train_true_logprobs,
				true_batch_idxs=train_true_batch_idxs,
				pred_mzs=train_pred_mzs,
				pred_logprobs=train_pred_logprobs,
				pred_batch_idxs=train_pred_batch_idxs,
			)

			false_mass_loss_vec = th.nan_to_num(
				false_mass_loss_vec,
				nan=0.0,
				posinf=1.0,
				neginf=0.0,
			)

			false_w = th.ones_like(loss)

			if getattr(self.hparams, "binned_false_mass_use_ce_weight", True):
				if true_ces is None:
					raise RuntimeError("binned_false_mass_use_ce_weight=True requires spec_ce in batch")

				ce_vec = true_ces.reshape(-1).to(device=loss.device, dtype=loss.dtype)
				assert ce_vec.shape[0] == loss.shape[0], (ce_vec.shape, loss.shape)

				valid_ce = th.isfinite(ce_vec)

				mid_thr = float(getattr(self.hparams, "binned_false_mass_mid_threshold", 20.0))
				high_thr = float(getattr(self.hparams, "binned_false_mass_high_threshold", 40.0))

				low_w = float(getattr(self.hparams, "binned_false_mass_low_weight", 0.5))
				mid_w = float(getattr(self.hparams, "binned_false_mass_mid_weight", 1.5))
				high_w = float(getattr(self.hparams, "binned_false_mass_high_weight", 2.0))

				false_w = th.where(
					valid_ce & (ce_vec < mid_thr),
					false_w * low_w,
					false_w,
				)

				false_w = th.where(
					valid_ce & (ce_vec >= mid_thr),
					th.ones_like(false_w) * mid_w,
					false_w,
				)

				false_w = th.where(
					valid_ce & (ce_vec >= high_thr),
					th.ones_like(false_w) * high_w,
					false_w,
				)

			false_mass_w = float(getattr(self.hparams, "binned_false_mass_loss_weight", 0.05))
			loss = loss + false_mass_w * false_w * false_mass_loss_vec

			if log:
				self.log(
					"train_binned_false_mass_loss",
					false_mass_loss_vec.detach().mean(),
					batch_size=batch_size,
					on_step=False,
					on_epoch=True,
				)
				self.log(
					"train_binned_false_mass_weight",
					false_w.detach().mean(),
					batch_size=batch_size,
					on_step=False,
					on_epoch=True,
				)

		# ===== R180: CE-weighted direct spectrum cosine/JSS loss =====
		# This is the first non-reranker step after R172E.
		# It directly optimizes final binned spectrum similarity and upweights mid/high CE.
		if split == "train" and getattr(self.hparams, "use_r180_ce_weighted_spectrum_loss", False):
			r180_d = r180_ce_weighted_spectrum_loss(
				batch=batch,
				batch_size=batch_size,
				hparams=self.hparams,
				true_mzs=train_true_mzs,
				true_logprobs=train_true_logprobs,
				true_batch_idxs=train_true_batch_idxs,
				pred_mzs=train_pred_mzs,
				pred_logprobs=train_pred_logprobs,
				pred_batch_idxs=train_pred_batch_idxs,
			)

			r180_loss_vec = r180_d["loss_vec"]
			r180_w = float(getattr(self.hparams, "r180_spectrum_loss_weight", 0.03))
			loss = loss + r180_w * r180_loss_vec

			if log:
				self.log(
					"train_r180_spectrum_loss",
					r180_loss_vec.detach().mean(),
					batch_size=batch_size,
					on_step=False,
					on_epoch=True,
				)
				self.log(
					"train_r180_cos_dist",
					r180_d["cos_dist"].detach().mean(),
					batch_size=batch_size,
					on_step=False,
					on_epoch=True,
				)
				self.log(
					"train_r180_jss_dist",
					r180_d["jss_dist"].detach().mean(),
					batch_size=batch_size,
					on_step=False,
					on_epoch=True,
				)
				self.log(
					"train_r180_ce_weight",
					r180_d["ce_weight"].detach().mean(),
					batch_size=batch_size,
					on_step=False,
					on_epoch=True,
				)

		# ===== R117: train-only support-conditional oracle reweight loss =====
		# R116 ceiling audit showed fixed-support oracle cosine is much higher than current cosine.
		# This loss keeps current predicted support fixed and teaches intensity allocation on that support.
		if split == "train" and getattr(self.hparams, "use_r117_support_oracle_reweight_loss", False):
			r117_loss_vec, r117_valid_frac, r117_covered_mass, r117_false_mass, r117_target_bins = (
				self._r117_support_oracle_reweight_loss(
					true_mzs=train_true_mzs,
					true_logprobs=train_true_logprobs,
					true_batch_idxs=train_true_batch_idxs,
					pred_mzs=train_pred_mzs,
					pred_logprobs=train_pred_logprobs,
					pred_batch_idxs=train_pred_batch_idxs,
					batch_size=batch_size,
				)
			)

			r117_loss_vec = th.nan_to_num(
				r117_loss_vec,
				nan=0.0,
				posinf=20.0,
				neginf=0.0,
			)

			r117_w = float(getattr(self.hparams, "r117_support_oracle_weight", 0.02))
			loss = loss + r117_w * r117_loss_vec

			if log:
				self.log(
					"train_r117_support_oracle_loss",
					r117_loss_vec.detach().mean(),
					batch_size=batch_size,
					on_step=False,
					on_epoch=True,
				)
				self.log(
					"train_r117_support_oracle_valid_frac",
					r117_valid_frac.detach(),
					batch_size=batch_size,
					on_step=False,
					on_epoch=True,
				)
				self.log(
					"train_r117_support_oracle_covered_mass",
					r117_covered_mass.detach(),
					batch_size=batch_size,
					on_step=False,
					on_epoch=True,
				)
				self.log(
					"train_r117_support_oracle_false_mass",
					r117_false_mass.detach(),
					batch_size=batch_size,
					on_step=False,
					on_epoch=True,
				)
				self.log(
					"train_r117_support_oracle_target_bins",
					r117_target_bins.detach(),
					batch_size=batch_size,
					on_step=False,
					on_epoch=True,
				)

		# ===== R5: precursor-mz region weighted loss =====
		# Region diagnosis shows low precursor-mz spectra dominate the catastrophic tail.
		# This reweights per-spectrum training loss only; val/test metrics stay unchanged.
		if split == "train" and getattr(self.hparams, "use_prec_mz_region_weighted_loss", False):
			prec_mz_vec = true_prec_mzs.reshape(-1).to(device=loss.device, dtype=loss.dtype)
			assert prec_mz_vec.shape[0] == loss.shape[0], (prec_mz_vec.shape, loss.shape)

			region_w = th.ones_like(loss)

			low_thr = float(getattr(self.hparams, "prec_mz_low_weight_threshold", 200.0))
			low_w = float(getattr(self.hparams, "prec_mz_low_loss_weight", 1.5))

			high_thr = float(getattr(self.hparams, "prec_mz_high_weight_threshold", 500.0))
			high_w = float(getattr(self.hparams, "prec_mz_high_loss_weight", 1.0))

			region_w = th.where(
				prec_mz_vec <= low_thr,
				region_w * low_w,
				region_w,
			)

			region_w = th.where(
				prec_mz_vec >= high_thr,
				region_w * high_w,
				region_w,
			)

			loss = loss * region_w

			if log:
				self.log(
					"train_prec_mz_region_loss_weight",
					region_w.detach().mean(),
					batch_size=batch_size,
					on_step=False,
					on_epoch=True,
				)

		# ===== R4c: train-only candidate presence ranking loss =====
		# Only supervises ranking among the model's own predicted candidates.
		# It is NOT added to val/test loss, so checkpoint selection is not polluted.
		if split == "train" and getattr(self.hparams, "use_candidate_presence_rank_loss", False):
			candidate_rank_loss_vec = candidate_presence_rank_loss(
				true_mzs=train_true_mzs,
				true_logprobs=train_true_logprobs,
				true_batch_idxs=train_true_batch_idxs,
				pred_mzs=train_pred_mzs,
				pred_logprobs=train_pred_logprobs,
				pred_batch_idxs=train_pred_batch_idxs,
				mz_max=float(self.hparams.mz_max),
				bin_res=float(getattr(self.hparams, "candidate_rank_mz_bin_res", 0.01)),
				topk=int(getattr(self.hparams, "candidate_rank_topk", 200)),
				max_pos_per_spec=int(getattr(self.hparams, "candidate_rank_max_pos_per_spec", 64)),
				max_neg_per_spec=int(getattr(self.hparams, "candidate_rank_max_neg_per_spec", 128)),
				margin=float(getattr(self.hparams, "candidate_rank_margin", 0.0)),
				match_tolerance_abs=float(getattr(self.hparams, "candidate_rank_match_tolerance_abs", 0.01)),
				min_true_prob=float(getattr(self.hparams, "candidate_rank_min_true_prob", 0.001)),
				pos_weight_gamma=float(getattr(self.hparams, "candidate_rank_pos_weight_gamma", 0.5)),
			)

			candidate_rank_loss_vec = th.nan_to_num(
				candidate_rank_loss_vec,
				nan=0.0,
				posinf=10.0,
				neginf=0.0,
			)

			candidate_rank_w = float(getattr(self.hparams, "candidate_rank_loss_weight", 0.02))
			loss = loss + candidate_rank_w * candidate_rank_loss_vec

			if log:
				self.log(
					"train_candidate_rank_loss",
					candidate_rank_loss_vec.detach().mean(),
					batch_size=batch_size,
					on_step=False,
					on_epoch=True,
				)

		# ===== R40D: train-only no-harm regularization for spectrum refiner residual =====
		# R40B2/R40C showed selected spectra improve but non-selected/easy spectra can be harmed.
		# This regularizer constrains excessive internal refiner movement without changing support.
		if split == "train" and getattr(self.hparams, "use_refiner_delta_noharm_loss", False):
			ref_delta = pred_d.get("pred_refiner_delta", None)
			ref_batch_idxs = pred_d.get("pred_refiner_delta_batch_idxs", None)
			ref_valid_mask = pred_d.get("pred_refiner_delta_valid_mask", None)

			if ref_delta is not None and ref_batch_idxs is not None:
				ref_delta = th.nan_to_num(
					ref_delta.reshape(-1),
					nan=0.0,
					posinf=0.0,
					neginf=0.0,
				)
				ref_batch_idxs = ref_batch_idxs.reshape(-1).long()

				if ref_valid_mask is not None:
					ref_valid_mask = ref_valid_mask.reshape(-1).bool()
					if ref_valid_mask.numel() == ref_delta.numel():
						ref_delta = ref_delta[ref_valid_mask]
						ref_batch_idxs = ref_batch_idxs[ref_valid_mask]

				if ref_delta.numel() > 0:
					delta_abs = ref_delta.abs()
					target = float(getattr(self.hparams, "refiner_delta_noharm_target", 0.01))
					power = float(getattr(self.hparams, "refiner_delta_noharm_power", 1.0))

					excess = (delta_abs - target).clamp_min(0.0)
					if abs(power - 1.0) < 1.0e-8:
						entry_loss = excess
					else:
						entry_loss = excess.pow(power)

					noharm_vec = th.zeros(
						int(batch_size),
						device=ref_delta.device,
						dtype=ref_delta.dtype,
					)
					noharm_cnt = th.zeros(
						int(batch_size),
						device=ref_delta.device,
						dtype=ref_delta.dtype,
					)

					clamped_batch = ref_batch_idxs.clamp(0, int(batch_size) - 1)
					noharm_vec.index_add_(0, clamped_batch, entry_loss)
					noharm_cnt.index_add_(0, clamped_batch, th.ones_like(entry_loss))
					noharm_vec = noharm_vec / noharm_cnt.clamp_min(1.0)
					noharm_vec = th.nan_to_num(noharm_vec, nan=0.0, posinf=0.0, neginf=0.0)

					noharm_w = float(getattr(self.hparams, "refiner_delta_noharm_weight", 0.0))

					if tuple(loss.shape) == tuple(noharm_vec.shape):
						loss = loss + noharm_w * noharm_vec
					else:
						loss = loss + noharm_w * noharm_vec.mean()

					if log:
						self.log(
							"train_refiner_delta_noharm_loss",
							noharm_vec.detach().mean(),
							batch_size=batch_size,
							on_step=False,
							on_epoch=True,
						)
						self.log(
							"train_refiner_delta_abs_mean",
							delta_abs.detach().mean(),
							batch_size=batch_size,
							on_step=False,
							on_epoch=True,
						)


		# ===== R39: auxiliary CE-pair spectrum-delta regularizer =====


		# ===== R43: auxiliary CE-pair signed top-bin ranking regularizer =====
		# Unlike R39 dense delta-MSE, this only supervises bins with the largest
		# true CE-induced changes and enforces the direction of high-vs-low CE.

		# ===== G1a: CE-pair predicted-depth ranking regularizer =====
		if split == "train" and getattr(self.hparams, "use_ce_depth_rank_loss", False):
			ce_depth_rank_loss, ce_depth_rank_pairs = self._ce_depth_rank_loss(
				batch=batch,
				pred_d=pred_d,
				batch_size=batch_size,
			)
			loss = loss + float(self.hparams.ce_depth_rank_weight) * ce_depth_rank_loss

			if log:
				self.log(
					"train_ce_depth_rank_loss",
					ce_depth_rank_loss.detach(),
					batch_size=batch_size,
					on_step=False,
					on_epoch=True,
				)
				self.log(
					"train_ce_depth_rank_pairs",
					ce_depth_rank_pairs.detach(),
					batch_size=batch_size,
					on_step=False,
					on_epoch=True,
				)


		# ===== G2: auxiliary CE-pair predicted-depth ranking regularizer =====
		# Unlike G1, this does not alter the main random batch distribution.

		mean_loss = self.batch_metric_reduce_fn(
			b_metric=loss,
			b_spec_per_group=spec_per_group,
			b_spec_per_mol=spec_per_mol,
			b_group_per_mol=group_per_mol,
			reduce="w_mean",
			split=split
		)
		total_loss, total_weight = self.batch_metric_reduce_fn(
			b_metric=loss,
			b_spec_per_group=spec_per_group,
			b_spec_per_mol=spec_per_mol,
			b_group_per_mol=group_per_mol,
			reduce="w_sum",
			return_weights=True,
			split=split
		)
		# prepare eval dict
		eval_true_mzs, eval_true_probs, eval_true_batch_idxs = self.preproc_spec(
			true_mzs,
			true_ints,
			true_batch_idxs,
			train=False,
			pred=False,
			log_in=False,
			log_out=False
		)
		eval_true_logprobs = safelog(eval_true_probs, eps=self.hparams.log_min)
		eval_pred_mzs, eval_pred_probs, eval_pred_batch_idxs = self.preproc_spec(
			pred_mzs,
			pred_logprobs,
			pred_batch_idxs,
			train=False,
			pred=True,
			log_in=True,
			log_out=False
		)
		eval_pred_logprobs = safelog(eval_pred_probs, eps=self.hparams.log_min)
		both_d = {
			"true_mzs": eval_true_mzs,
			"true_logprobs": eval_true_logprobs,
			"true_batch_idxs": eval_true_batch_idxs,
			"true_prec_mzs": true_prec_mzs,
			"pred_mzs": eval_pred_mzs,
			"pred_logprobs": eval_pred_logprobs,
			"pred_batch_idxs": eval_pred_batch_idxs,
			**pred_d
		}
		# just log loss
		if log:
			self.log(
				f"{split}_batch_loss",
				mean_loss,
				batch_size=batch_size,
				on_epoch=True
			)
		results = {
			"unique_id": unique_id,
			"smiles": smiles,
			"true_prec_mzs": true_prec_mzs,
   			"input_ce": true_ces,
			"mean_loss": mean_loss,
			"total_loss": total_loss,
			"total_weight": total_weight,
			"spec_per_group": spec_per_group,
			"spec_per_mol": spec_per_mol,
			"group_per_mol": group_per_mol,
			**both_d,
			**loss_d
		}
		assert true_mzs.shape[0] > 0, true_mzs.shape[0]
		with th.inference_mode():
			metric_input_d = {
				"true_mzs": eval_true_mzs,
				"true_ints": eval_true_probs,
				"true_batch_idxs": eval_true_batch_idxs,
				"pred_mzs": eval_pred_mzs,
				"pred_ints": eval_pred_probs,
				"pred_batch_idxs": eval_pred_batch_idxs,
				"true_prec_mzs": true_prec_mzs
			}
			for arg in self.extra_metric_args:
				metric_input_d[arg] = both_d[arg]
			metric_output_d = self.metric_fn(**metric_input_d)
			for k,v in metric_output_d.items():
				assert k not in results, k
				results[k] = v
		for metric_name in self.metric_names:
			assert metric_name in results, metric_name
		return results

	def training_step(self, batch, batch_idx):
		""" training loop

		Args:
			batch (_type_): _description_
			batch_idx (_type_): _description_

		Raises:
			NotImplementedError: _description_

		Returns:
			_type_: _description_
		"""

		if self.hparams.automatic_optimization:
			batch_results = self._common_step(batch, split="train")
			mean_loss = batch_results["mean_loss"]
			self._update_results(batch_results,"train")
			return mean_loss
		else:
			raise NotImplementedError("Manual optimization not implemented")

	def validation_step(self, batch, batch_idx):

		assert not self.training
		batch_results = self._common_step(batch, split="val")
		mean_loss = batch_results["mean_loss"]
		self._update_results(batch_results,"val")
		return mean_loss

	def test_step(self, batch, batch_idx):

		assert not self.training
		batch_results = self._common_step(batch, split="test")
		mean_loss = batch_results["mean_loss"]
		self._update_results(batch_results,"test")
		return mean_loss
	
	def inference_step(self, batch, split, untransform_spec=False):

		if self.training:
			print("Warning: model is in training mode")
		batch_results = self._common_step(batch, split=split, log=False)
		if untransform_spec:
			raise NotImplementedError("untransform_spec not implemented")
			# pred_logprobs = batch_results["pred_logprobs"]
			# pred_batch_idxs = batch_results["pred_batch_idxs"]
			# true_logprobs = batch_results["true_logprobs"]
			# true_batch_idxs = batch_results["true_batch_idxs"]
			# pred_ints = self.ints_untransform_func(
			# 	pred_logprobs.exp(),
			# 	pred_batch_idxs
			# )
			# pred_ints = self.ints_normalize_func(
			# 	pred_ints, 
			# 	pred_batch_idxs
			# )
			# if "pred_oos_logprobs" in batch_results:
			# 	pred_oos_logprobs = batch_results["pred_oos_logprobs"][pred_batch_idxs]
			# 	pred_ints = pred_ints * (1.-pred_oos_logprobs.exp())
			# batch_results["pred_logprobs"] = safelog(pred_ints, eps=self.hparams.log_min)
			# true_ints = self.ints_untransform_func(
			# 	true_logprobs.exp(),
			# 	true_batch_idxs
			# )
			# true_ints = self.ints_normalize_func(
			# 	true_ints, 
			# 	true_batch_idxs
			# )
			# batch_results["true_logprobs"] = safelog(true_ints, eps=self.hparams.log_min)
		return batch_results

	def configure_optimizers(self):

		if self.hparams.optimizer == "adam":
			optimizer_cls = th.optim.Adam
		elif self.hparams.optimizer == "adamw":
			optimizer_cls = th.optim.AdamW
		elif self.hparams.optimizer == "sgd":
			optimizer_cls = th.optim.SGD
		else:
			raise ValueError(f"Unknown optimizer {self.optimizer}")
		optimizer = optimizer_cls(
			self.parameters(), 
			lr=self.hparams.lr, 
			weight_decay=self.hparams.weight_decay
		)
		ret = {
			"optimizer": optimizer,
		}
		if self.hparams.lr_schedule:
			scheduler = build_lr_scheduler(
				optimizer=optimizer, 
				decay_rate=self.hparams.lr_decay_rate, 
				warmup_steps=self.hparams.lr_warmup_steps,
				decay_steps=self.hparams.lr_decay_steps,
			)
			ret["lr_scheduler"] = {
				"scheduler": scheduler,
				"frequency": 1,
				"interval": "step",
			}
		return ret

	def _get_wandb_logger(self):

		wandb_logger = None
		for logger in self.loggers:
			if isinstance(logger, pl.loggers.WandbLogger):
				wandb_logger = logger		
		return wandb_logger

	def _update_results(self, batch_results, split):

		results_attr = f"{split}_results"
		counter_attr = f"{split}_counter"
		# filter keys (filtering first to save time/memory)
		keys = [
			"unique_ids",
			"smiles",
			"true_prec_mzs",
			"true_mzs",
			"true_logprobs",
			"true_unique_ids",
			"pred_mzs",
			"pred_logprobs",
			"pred_unique_ids",
			"spec_per_group",
			"spec_per_mol",
			"group_per_mol",
			"input_ce",
		]
		keys.extend(list(self.metric_names))
		unique_ids = batch_results.pop("unique_id")
		true_batch_idxs = batch_results.pop("true_batch_idxs")
		pred_batch_idxs = batch_results.pop("pred_batch_idxs")
		batch_results = {k:v for k,v in batch_results.items() if k in keys}
		# update unique_ids
		true_unique_ids = unique_ids[true_batch_idxs]
		pred_unique_ids = unique_ids[pred_batch_idxs]
		batch_results["unique_ids"] = unique_ids
		batch_results["true_unique_ids"] = true_unique_ids
		batch_results["pred_unique_ids"] = pred_unique_ids
		# check all metrics
		assert all([metric_name in batch_results for metric_name in self.metric_names])
		# transfer to cpu
		batch_results = to_cpu(batch_results,detach=True)
		# setup results dict
		if getattr(self,results_attr) is None:
			setattr(self,results_attr,{k:list() for k in keys})
		else:
			assert set(keys) == set(getattr(self,results_attr).keys())
		results_dict = getattr(self,results_attr)
		# add to results dict
		for k,v in batch_results.items():
			results_dict[k].append(v)
		# increment counter
		setattr(self,counter_attr,getattr(self,counter_attr)+unique_ids.shape[0])

	def _reduce_metrics(self, results, split):

		mean_metrics, std_metrics = {}, {}
		for k, v in results.items():
			if k in self.metric_names:
				mean_metrics[k] = self.batch_metric_reduce_fn(
					b_metric=v,
					b_spec_per_group=results["spec_per_group"],
					b_spec_per_mol=results["spec_per_mol"],
					b_group_per_mol=results["group_per_mol"],
					reduce="w_mean",
					split=split
				)
				std_metrics[k] = self.batch_metric_reduce_fn(
					b_metric=v,
					b_spec_per_group=results["spec_per_group"],
					b_spec_per_mol=results["spec_per_mol"],
					b_group_per_mol=results["group_per_mol"],
					reduce="w_std",
					split=split
				)
		return mean_metrics, std_metrics

	def _consolidate_results(self, split):
		
		results = getattr(self,f"{split}_results")
		if results is None:
			return
		keys = results.keys()
		for k in keys:
			if isinstance(results[k][0],th.Tensor):
				results[k] = th.cat(results[k],dim=0)
			else:
				assert isinstance(results[k][0],list)
				results[k] = flatten_lol(results[k])
		# log all the metrics
		mean_metrics, std_metrics = self._reduce_metrics(results, split)
		for k in mean_metrics.keys():
			self.log(
				f"{split}_{k}_epoch/mean",
				mean_metrics[k],
			)
			self.log(
				f"{split}_{k}_epoch/std",
				std_metrics[k],
			)
		if split != "test":
			# update epoch stats
			mean_metrics_epochs = getattr(self,f"{split}_mean_metrics")
			std_metrics_epochs = getattr(self,f"{split}_std_metrics")
			for k in mean_metrics.keys():
				mean_metrics_epochs[k.replace(".","-")][self.current_epoch] = mean_metrics[k]
				std_metrics_epochs[k.replace(".","-")][self.current_epoch] = std_metrics[k]
			# log histograms
			if self.hparams.log_hist_metrics:
				wandb_logger = self._get_wandb_logger()
				import wandb
				for k,v in results.items():
					if k in self.metric_names:
						log_d = {
							f"{split}_{k}_hist": wandb.Histogram(v.cpu()),
							"epoch": self.current_epoch
						}
						if wandb_logger is not None:
							wandb_logger.experiment.log(log_d)
			# log best metric
			update_datapoint_metrics = False
			checkpoint_metric = self.hparams.checkpoint_metric.removeprefix("train_").removeprefix("val_").removesuffix("/mean").removesuffix("/std").removesuffix("_epoch")
			for k in self.metric_bests:
				assert k in self.metric_names, (k, self.metric_names)
				mean_metric_epochs = mean_metrics_epochs[k.replace(".","-")][:self.current_epoch+1]
				std_metric_epochs = std_metrics_epochs[k.replace(".","-")][:self.current_epoch+1]
				if self.metric_min_max[k] == "min":
					argbest_metric = th.argmin(mean_metric_epochs)
				elif self.metric_min_max[k] == "max":
					argbest_metric = th.argmax(mean_metric_epochs)
				else:
					assert self.metric_min_max[k] is None, self.metric_min_max[k]
					continue
				mean_metric_best = mean_metric_epochs[argbest_metric]
				std_metric_best = std_metric_epochs[argbest_metric]
				self.log(
					f"{split}_{k}_best/mean",
					mean_metric_best
				)
				self.log(
					f"{split}_{k}_best/std",
					std_metric_best
				)
				# if it's the best, update the datapoint metrics
				if k == checkpoint_metric and argbest_metric == self.current_epoch:
					update_datapoint_metrics = True
			if checkpoint_metric == "epoch":
				assert not update_datapoint_metrics
				update_datapoint_metrics = True
		else:
			update_datapoint_metrics = True

		if self.hparams.track_datapoint_metrics and update_datapoint_metrics:
			datapoint_metrics = getattr(self,f"{split}_datapoint_metrics")
			num_datapoints_p = getattr(self,f"{split}_num_datapoints")
			example_key = list(mean_metrics.keys())[0]
			num_datapoints = num_datapoints_p.item()
			if num_datapoints == -1:
				num_datapoints_p[0] = len(results[example_key])
				num_datapoints = num_datapoints_p.item()
			assert num_datapoints <= datapoint_metrics[example_key.replace(".","-")].shape[0], (num_datapoints, datapoint_metrics[example_key.replace(".","-")].shape[0])
			for k in mean_metrics.keys():
				assert len(results[k]) == num_datapoints, (k, len(results[k]), num_datapoints)
				datapoint_metrics[k.replace(".","-")][:num_datapoints] = results[k]

	def _log_images(self, split):

		results = getattr(self,f"{split}_results")
		num_log_images = getattr(self.hparams,f"num_log_{split}_images")
		counter = getattr(self,f"{split}_counter")
		num_log_images = min(num_log_images,counter)
		wandb_logger = self._get_wandb_logger()
		if wandb_logger is None:
			return
		# randomly sample unique_ids
		unique_ids = th.unique(results["unique_ids"],sorted=True)
		if num_log_images == unique_ids.shape[0]:
			sample_idxs = th.arange(num_log_images)
			sample_unique_ids = unique_ids
		else:
			with th_temp_seed(420):
				sample_idxs = th.randperm(unique_ids.shape[0])[:num_log_images]
				sample_unique_ids = unique_ids[sample_idxs]
		# plot images
		for i in range(num_log_images):
			unique_id = sample_unique_ids[i].item()
			unique_idx = th.nonzero(results["unique_ids"] == unique_id,as_tuple=False).item()
			true_mask = results["true_unique_ids"] == unique_id
			pred_mask = results["pred_unique_ids"] == unique_id
			smiles = results["smiles"][unique_idx]
			loss = results["loss"][unique_idx].item()
			if "cos_sim" in results:
				cos_sim = results["cos_sim"][unique_idx].item()
			else:
				cos_sim = np.nan
			if "wrecall" in results:
				wrecall = results["wrecall"][unique_idx].item()
			else:
				wrecall = np.nan
			# note: these are already untransformed and normalized (with possibility of oos)
			true_mzs = results["true_mzs"][true_mask]
			true_ints = th.exp(results["true_logprobs"][true_mask])
			pred_mzs = results["pred_mzs"][pred_mask]
			pred_ints = th.exp(results["pred_logprobs"][pred_mask])
			# cast to numpy
			true_mzs = true_mzs.numpy()
			true_ints = true_ints.numpy()
			pred_mzs = pred_mzs.numpy()
			pred_ints = pred_ints.numpy()
			assert np.isclose(np.sum(true_ints),1.0), np.sum(true_ints)
			assert np.isclose(np.sum(pred_ints),1.0), np.sum(pred_ints)
			# plot
			data = plot_spectra_sparse(
				true_mzs,
				true_ints,
				pred_mzs,
				pred_ints,
				smiles,
				return_data=True
			)
			wandb_logger.log_image(
				key=f"{split}_example_{i}",
				caption=[f"unique_id = {unique_id}, epoch = {self.current_epoch:03d}, loss = {loss:.3f}, cos_sim = {cos_sim:.3f}, wrecall = {wrecall:.3f}"],
				images=[data]
			)

	def on_train_epoch_start(self):

		self._seed_dataloader(self.current_epoch)

	def on_train_epoch_end(self):

		# consolidate results
		self._consolidate_results("train")
		# log images
		self._log_images("train")
		# reset
		self.train_results = None
		self.train_counter = 0
		# seed dataloader
		self._seed_dataloader(self.current_epoch+1)

	def on_validation_epoch_end(self):
		
		if not self.trainer.sanity_checking:
			# consolidate results
			self._consolidate_results("val")
			# log images
			self._log_images("val")
		# reset
		self.val_results = None
		self.val_counter = 0

	def _dump_datapoint_metrics_csv(self, split):
		"""Dump per-spectrum metrics for error region analysis."""
		import os
		import pandas as pd
		import torch as th

		results = getattr(self, f"{split}_results")
		if results is None:
			return

		out_dir = getattr(
			self.hparams,
			"datapoint_metrics_out_dir",
			"diagnostics/datapoint_metrics",
		)
		os.makedirs(out_dir, exist_ok=True)

		unique_ids = results["unique_ids"]
		n = len(unique_ids)

		def to_list(x):
			if isinstance(x, th.Tensor):
				x = x.detach().cpu()
				if x.ndim == 0:
					return [float(x.item())] * n
				return x.numpy().tolist()
			return list(x)

		data = {
			"unique_id": to_list(results["unique_ids"]),
			"smiles": to_list(results["smiles"]),
		}

		if "true_prec_mzs" in results:
			data["prec_mz"] = to_list(results["true_prec_mzs"])

		if "input_ce" in results and results["input_ce"] is not None:
			ce_list = to_list(results["input_ce"])
			if len(ce_list) == n:
				data["input_ce"] = ce_list

		for k in self.metric_names:
			if k in results:
				col = k.replace(".", "_")
				v = to_list(results[k])
				if len(v) == n:
					data[col] = v

		df = pd.DataFrame(data)

		# Useful derived columns
		if "cos_sim_0_01" in df.columns:
			df["bad_cos_lt_0_2"] = df["cos_sim_0_01"] < 0.2
			df["bad_cos_lt_0_4"] = df["cos_sim_0_01"] < 0.4

		if "prec_mz" in df.columns:
			df["prec_mz_bin"] = pd.cut(
				df["prec_mz"],
				bins=[0, 200, 300, 400, 500, 700, 1000, 1500, 99999],
				include_lowest=True,
			).astype(str)

		if "input_ce" in df.columns:
			df["input_ce_bin"] = pd.cut(
				df["input_ce"],
				bins=[-1, 10, 20, 30, 40, 60, 80, 999],
				include_lowest=True,
			).astype(str)
		out_path = os.path.join(
			out_dir,
			f"{split}_datapoint_metrics_epoch{self.current_epoch}.csv",
		)
		df.to_csv(out_path, index=False)
		print(f"[DiagDump] wrote {out_path}, rows={len(df)}")

	def on_test_epoch_end(self):

		# consolidate results
		self._consolidate_results("test")

		if getattr(self.hparams, "dump_test_datapoint_metrics", False):
			self._dump_datapoint_metrics_csv("test")
		# reset
		self.test_results = None
		self.test_counter = 0

	def _print_grad_norm(self,prefix=None,total_num_params=5):

		if prefix is None:
			prefix = "no_prefix"
		print(f">> {prefix}, {self._cur_batch_size}, {self._max_batch_size}")
		opt = self.optimizers()
		num_params = 0
		for pg in opt.param_groups:
			for p in pg['params']:
				if p.grad is not None:
					print(th.norm(p.grad))
					num_params += 1
				if num_params > total_num_params:
					break
			if num_params > total_num_params:
				break
	
	def on_after_backward(self):
		
		if self.hparams.check_gradient_norm:
			self._print_grad_norm(prefix="after_backward")

	def on_before_optimizer_step(self, optimizer):

		if self.hparams.check_gradient_norm:
			self._print_grad_norm(prefix="before_optimizer_step")

	def on_after_optimizer_step(self, optimizer):

		self._cur_batch_size = 0
		self._cur_batch_weight = 0.

	def _check_ce_params(self):

		# check merge params
		if self.hparams.spec_params["merge"]:
			assert (not self.hparams.spec_params["nce"]) or self.hparams.spec_params["merge_keep_ces"]
		elif self.hparams.spec_params["nce"]:
			assert not self.hparams.spec_params["ace"]
			assert self.hparams.spec_params["nce"] and (not self.hparams.spec_params["merge_keep_ces"])
		elif self.hparams.spec_params["ace"]:
			assert not self.hparams.spec_params["nce"]
			assert self.hparams.spec_params["ace"] and (not self.hparams.spec_params["merge_keep_ces"])
   
	def _seed_dataloader(self, seed):

		generator = th.Generator()
		generator.manual_seed(self.train_dl_seeds[seed].item())
		train_dataloader = self.trainer.train_dataloader
		batch_sampler = train_dataloader.batch_sampler
		sampler = train_dataloader.sampler
		if self.hparams.dynamic_batch_sampler:
			if hasattr(batch_sampler.sampler, "generator"):
				# print("p3")
				batch_sampler.sampler.generator = generator
			if hasattr(batch_sampler.sampler, "_pre_compute_batches"):
				# print("p4")
				batch_sampler.sampler._pre_compute_batches()
			if hasattr(batch_sampler, "_pre_compute_batches"):
				# print("p5")
				batch_sampler._pre_compute_batches()
		else:
			assert batch_sampler.sampler is sampler
			if hasattr(sampler, "generator"):
				# print("p1")
				sampler.generator = generator
			if hasattr(sampler, "_pre_compute_batches"):
				# print("p2")
				sampler._pre_compute_batches()

	def on_save_checkpoint(self, checkpoint):

		# rng
		checkpoint["rng_states"] = _collect_rng_states(include_cuda=th.cuda.is_available())
		# metrics
		checkpoint["train_mean_metrics"] = self.train_mean_metrics
		checkpoint["val_mean_metrics"] = self.val_mean_metrics
		checkpoint["train_std_metrics"] = self.train_std_metrics
		checkpoint["val_std_metrics"] = self.val_std_metrics
		if self.hparams.track_datapoint_metrics:
			checkpoint["train_num_datapoints"] = self.train_num_datapoints
			checkpoint["val_num_datapoints"] = self.val_num_datapoints
			checkpoint["train_datapoint_metrics"] = self.train_datapoint_metrics
			checkpoint["val_datapoint_metrics"] = self.val_datapoint_metrics

	def on_load_checkpoint(self, checkpoint):
		
		# rng
		if "torch" in checkpoint["rng_states"]:
			checkpoint["rng_states"]["torch"] = checkpoint["rng_states"]["torch"].cpu()
		if "torch.cuda" in checkpoint["rng_states"]:
			checkpoint["rng_states"]["torch.cuda"][0] = checkpoint["rng_states"]["torch.cuda"][0].cpu()
		_set_rng_states(checkpoint["rng_states"])
		# metrics
		self.train_mean_metrics = checkpoint["train_mean_metrics"]
		self.val_mean_metrics = checkpoint["val_mean_metrics"]
		self.train_std_metrics = checkpoint["train_std_metrics"]
		self.val_std_metrics = checkpoint["val_std_metrics"]
		if self.hparams.track_datapoint_metrics:
			self.train_num_datapoints = checkpoint["train_num_datapoints"]
			self.val_num_datapoints = checkpoint["val_num_datapoints"]
			self.train_datapoint_metrics = checkpoint["train_datapoint_metrics"]
			self.val_datapoint_metrics = checkpoint["val_datapoint_metrics"]

class FragGNNPL(SpectrumPL):

	def _setup_model(self):

		# frag GNN
		self.model = FragGNNModel(
			num_depth=self.hparams.num_depth,
			num_hs=self.hparams.num_hs,
			num_elements=self.hparams.num_elements,
			int_embedder=self.hparams.int_embedder,
			int_embedder_tight=self.hparams.int_embedder_tight,
			mol_node_feats=self.hparams.mol_params["pyg_node_feats"],
			mol_edge_feats=self.hparams.mol_params["pyg_edge_feats"],
			mol_pe_embed_k=self.hparams.mol_params["pyg_pe_embed_k"],
			mol_hidden_size=self.hparams.mol_hidden_size,
			mol_num_layers=self.hparams.mol_num_layers,
			mol_gnn_type=self.hparams.mol_gnn_type,
			mol_dropout=self.hparams.mol_dropout,
			mol_normalization=self.hparams.mol_normalization,
			mol_pool_type=self.hparams.mol_pool_type,
			frag_node_feats=self.hparams.frag_params["pyg_node_feats"],
			frag_edge_feats=self.hparams.frag_params["pyg_edge_feats"],
			frag_hidden_size=self.hparams.frag_hidden_size,
			frag_num_layers=self.hparams.frag_num_layers,
			frag_gnn_type=self.hparams.frag_gnn_type,
			frag_dropout=self.hparams.frag_dropout,
			frag_normalization=self.hparams.frag_normalization,
			frag_pool_type=self.hparams.frag_pool_type,
			frag_embed_combine=self.hparams.frag_embed_combine,
			frag_pool_combine=self.hparams.frag_pool_combine,
			mlp_output_format=self.hparams.mlp_output_format,
			mlp_hidden_size=self.hparams.mlp_hidden_size,
			mlp_normalization=self.hparams.mlp_normalization,
			mlp_dropout=self.hparams.mlp_dropout,
			mlp_num_layers=self.hparams.mlp_num_layers,
			mlp_use_residuals=self.hparams.mlp_use_residuals,
			cc_interstage_type=self.hparams.cc_interstage_type,
			nb_iso=self.hparams.nb_iso,
			skip_edge_loss=self.hparams.skip_edge_loss,
			mask_null_formula=self.hparams.mask_null_formula,
			predict_oos=self.hparams.predict_oos,
			bin_output=self.hparams.bin_output,
			mz_bin_res=self.hparams.mz_bin_res,
			mz_max=self.hparams.mz_max,
			ce_insert_type=self.hparams.ce_insert_type,
			ce_insert_location=self.hparams.ce_insert_location,
			ce_insert_merge=self.hparams.ce_insert_merge,
			ce_insert_size=self.hparams.ce_insert_size,
			ce_mean=self.hparams.ce_mean,
			ce_std=self.hparams.ce_std,
			ce_max=self.hparams.ce_max,
			prec_insert_location=self.hparams.prec_insert_location,
			prec_insert_size=self.hparams.prec_insert_size,
			prec_types=self.hparams.spec_params["prec_types"],
			inst_insert_location=self.hparams.inst_insert_location,
			inst_insert_size=self.hparams.inst_insert_size,
			inst_types=self.hparams.spec_params["inst_types"],
			output_formula_str=self.hparams.output_formula_str,
			use_ce_fragment_gate=self.hparams.use_ce_fragment_gate,
			ce_fragment_gate_hidden_size=self.hparams.ce_fragment_gate_hidden_size,
			ce_fragment_gate_dropout=self.hparams.ce_fragment_gate_dropout,
			ce_fragment_gate_gamma_scale=self.hparams.ce_fragment_gate_gamma_scale,
			ce_fragment_gate_use_depth=self.hparams.ce_fragment_gate_use_depth,
   			use_ce_oos_head=self.hparams.use_ce_oos_head,
			use_ce_local_transition_prior=self.hparams.use_ce_local_transition_prior,
			ce_local_transition_hidden_size=self.hparams.ce_local_transition_hidden_size,
			ce_local_transition_dropout=self.hparams.ce_local_transition_dropout,
			ce_local_transition_delta_scale=self.hparams.ce_local_transition_delta_scale,
            use_ce_path_energy=self.hparams.use_ce_path_energy,
            ce_path_energy_hidden_size=self.hparams.ce_path_energy_hidden_size,
            ce_path_energy_dropout=self.hparams.ce_path_energy_dropout,
            ce_path_energy_delta_scale=self.hparams.ce_path_energy_delta_scale,
            ce_path_energy_max_depth=self.hparams.ce_path_energy_max_depth,
            use_ce_depth_mixture_head=self.hparams.use_ce_depth_mixture_head,
            ce_depth_mixture_hidden_size=self.hparams.ce_depth_mixture_hidden_size,
            ce_depth_mixture_dropout=self.hparams.ce_depth_mixture_dropout,
            ce_depth_mixture_delta_scale=self.hparams.ce_depth_mixture_delta_scale,
            ce_depth_mixture_num_channels=self.hparams.ce_depth_mixture_num_channels,
   			use_ce_peak_channel_allocator=self.hparams.use_ce_peak_channel_allocator,
			ce_peak_channel_hidden_size=self.hparams.ce_peak_channel_hidden_size,
			ce_peak_channel_dropout=self.hparams.ce_peak_channel_dropout,
			ce_peak_channel_delta_scale=self.hparams.ce_peak_channel_delta_scale,
			ce_peak_channel_max_channels=self.hparams.ce_peak_channel_max_channels,
   			ce_peak_channel_allocator_mode=self.hparams.ce_peak_channel_allocator_mode,

   			# R54: forward m/z-offset renderer flags into FragGNNModel.
   			use_rendered_peak_drop_gate=getattr(self.hparams, "use_rendered_peak_drop_gate", False),
   			rendered_peak_gate_hidden_size=getattr(self.hparams, "rendered_peak_gate_hidden_size", 128),
   			rendered_peak_gate_dropout=getattr(self.hparams, "rendered_peak_gate_dropout", 0.1),
   			rendered_peak_gate_delta_scale=getattr(self.hparams, "rendered_peak_gate_delta_scale", 4.0),
   			rendered_peak_gate_init_bias=getattr(self.hparams, "rendered_peak_gate_init_bias", 8.0),
   			rendered_peak_gate_max_channels=getattr(self.hparams, "rendered_peak_gate_max_channels", 8),
   			rendered_peak_gate_use_extra_features=getattr(self.hparams, "rendered_peak_gate_use_extra_features", False),
   			use_mz_offset_peak_expansion=self.hparams.use_mz_offset_peak_expansion,
   			mz_offset_peak_steps=self.hparams.mz_offset_peak_steps,
   			mz_offset_peak_prior_sigma=self.hparams.mz_offset_peak_prior_sigma,
			use_ce_formula_node_allocator=self.hparams.use_ce_formula_node_allocator,
			ce_formula_node_hidden_size=self.hparams.ce_formula_node_hidden_size,
			ce_formula_node_dropout=self.hparams.ce_formula_node_dropout,
			ce_formula_node_delta_scale=self.hparams.ce_formula_node_delta_scale,
			ce_formula_node_center_per_spectrum=self.hparams.ce_formula_node_center_per_spectrum,
			ce_formula_node_use_depth=self.hparams.ce_formula_node_use_depth,
			ce_formula_node_mode=self.hparams.ce_formula_node_mode,
			# K3 dense vocab, keep for completeness
			use_formula_vocab_residual=self.hparams.use_formula_vocab_residual,
			formula_vocab_size=self.hparams.formula_vocab_size,
			formula_vocab_hidden_size=self.hparams.formula_vocab_hidden_size,
			formula_vocab_dropout=self.hparams.formula_vocab_dropout,
			formula_vocab_delta_scale=self.hparams.formula_vocab_delta_scale,
			formula_vocab_center_per_spectrum=self.hparams.formula_vocab_center_per_spectrum,
			formula_vocab_oov_id=self.hparams.formula_vocab_oov_id,

			# K3b composition residual
			use_formula_comp_residual=self.hparams.use_formula_comp_residual,
			formula_comp_feat_size=self.hparams.formula_comp_feat_size,
			formula_comp_hidden_size=self.hparams.formula_comp_hidden_size,
			formula_comp_dropout=self.hparams.formula_comp_dropout,
			formula_comp_delta_scale=self.hparams.formula_comp_delta_scale,
			formula_comp_center_per_spectrum=self.hparams.formula_comp_center_per_spectrum,

			# CE-response candidate scorer
			use_ce_response_scorer=self.hparams.use_ce_response_scorer,
			ce_response_hidden_size=self.hparams.ce_response_hidden_size,
			ce_response_dropout=self.hparams.ce_response_dropout,
			ce_response_delta_scale=self.hparams.ce_response_delta_scale,
			ce_response_center_per_spectrum=self.hparams.ce_response_center_per_spectrum,
			ce_response_use_formula_comp=self.hparams.ce_response_use_formula_comp,
			ce_response_use_depth=self.hparams.ce_response_use_depth,
			ce_response_use_h=self.hparams.ce_response_use_h,

			# CutChem-NodeSummary residual
			use_cutchem_node_residual=self.hparams.use_cutchem_node_residual,
			cutchem_node_hidden_size=self.hparams.cutchem_node_hidden_size,
			cutchem_node_dropout=self.hparams.cutchem_node_dropout,
			cutchem_node_delta_scale=self.hparams.cutchem_node_delta_scale,
			cutchem_node_center_per_spectrum=self.hparams.cutchem_node_center_per_spectrum,
			cutchem_node_use_ce=self.hparams.cutchem_node_use_ce,
			cutchem_node_use_depth=self.hparams.cutchem_node_use_depth,
			cutchem_node_use_h=self.hparams.cutchem_node_use_h,

			# CE-FlowFrag v2
			use_ce_flowfrag=self.hparams.use_ce_flowfrag,
			ce_flowfrag_hidden_size=self.hparams.ce_flowfrag_hidden_size,
			ce_flowfrag_dropout=self.hparams.ce_flowfrag_dropout,
			ce_flowfrag_max_depth=self.hparams.ce_flowfrag_max_depth,
			ce_flowfrag_lambda_max=self.hparams.ce_flowfrag_lambda_max,
			ce_flowfrag_mixture_hidden_size=self.hparams.ce_flowfrag_mixture_hidden_size,
			ce_flowfrag_mixture_dropout=self.hparams.ce_flowfrag_mixture_dropout,
			ce_flowfrag_mixture_init_bias=self.hparams.ce_flowfrag_mixture_init_bias,
			ce_flowfrag_delta_clip=self.hparams.ce_flowfrag_delta_clip,
			ce_flowfrag_use_direct_node=self.hparams.ce_flowfrag_use_direct_node,
			ce_flowfrag_direct_mix=self.hparams.ce_flowfrag_direct_mix,
			use_spectrum_candidate_refiner=self.hparams.use_spectrum_candidate_refiner,
			spectrum_refiner_hidden_size=self.hparams.spectrum_refiner_hidden_size,
			spectrum_refiner_num_layers=self.hparams.spectrum_refiner_num_layers,
			spectrum_refiner_num_heads=self.hparams.spectrum_refiner_num_heads,
			spectrum_refiner_dropout=self.hparams.spectrum_refiner_dropout,
			spectrum_refiner_delta_scale=self.hparams.spectrum_refiner_delta_scale,
			spectrum_refiner_topk=self.hparams.spectrum_refiner_topk,
			spectrum_refiner_center_per_spectrum=self.hparams.spectrum_refiner_center_per_spectrum,
			spectrum_refiner_use_logit_feature=self.hparams.spectrum_refiner_use_logit_feature,
			spectrum_refiner_use_mz_features=self.hparams.spectrum_refiner_use_mz_features,
			spectrum_refiner_use_peak_prior=self.hparams.spectrum_refiner_use_peak_prior,
			# R134: pre-R54 peak-entry scorer
			use_pre_r54_peak_entry_gate=getattr(self.hparams, "use_pre_r54_peak_entry_gate", False),
			pre_r54_peak_entry_hidden_size=getattr(self.hparams, "pre_r54_peak_entry_hidden_size", 128),
			pre_r54_peak_entry_dropout=getattr(self.hparams, "pre_r54_peak_entry_dropout", 0.1),
			pre_r54_peak_entry_delta_scale=getattr(self.hparams, "pre_r54_peak_entry_delta_scale", 0.05),
			pre_r54_peak_entry_max_channels=getattr(self.hparams, "pre_r54_peak_entry_max_channels", 16),
		)
		
		self._apply_spectrum_refiner_train_scope()

		self._check_ce_params()

		# check edge loss params
		if not self.hparams.skip_edge_loss:
			assert "h_counts" in self.hparams.frag_params["pyg_node_feats"]
			assert "h_range" in self.hparams.frag_params["pyg_edge_feats"]
		else:
			if "h_counts" in self.hparams.frag_params["pyg_node_feats"]:
				logging.warning("h_counts in frag pyg_node_feats but edge_loss is disabled!")
			if "h_range" in self.hparams.frag_params["pyg_edge_feats"]:
				logging.warning("h_range in frag pyg_edge_feats but edge_loss is disabled!")

		# compile
		if self.hparams.compile:
			th_dynamo.reset()
			self.dynamo_prof = th_dynamo.utils.CompileProfiler()
			self.model = self.model.get_compile(backend=self.dynamo_prof,dynamic=True)

	def _setup_loss_names(self):

		# flag losses for tracking
		loss_names = [
			"loss",
			"primary_loss",
			"null_formula_prob",
			"oos_prob",
		]
		if getattr(self.hparams, "use_aux_cosine_loss", False):
			loss_names.append("aux_cosine_loss")

		if getattr(self.hparams, "use_binned_intensity_aux_loss", False):
			loss_names.append("binned_intensity_aux_loss")
		if self.hparams.loss_type == "cross_entropy":
			loss_names.extend([
				"spec_ce",
				"oos_ce",
				"ios_ce",
			])
		if not self.hparams.skip_extra_losses:
			loss_names.extend([
				"spec_e",
				"spec_ne",
				"formula_e",
				"formula_ne",
				"node_e",
				"node_ne",
				"node_formula_e",
				"node_formula_ne",
				"formula_node_e",
				"formula_node_ne",
				"joint_e",
				"joint_ne",
				"node_formula_mi",
				"formula_node_mi",
				"h_mean",
				"h_e",
				"h_ne"
			])
			if not self.hparams.skip_edge_loss:
				loss_names.extend([
					"edge_h_range_loss",
					"edge_h_transfer_loss",
					"edge_e",
					"edge_ne"
				])
			if self.hparams.nb_iso:
				loss_names.extend([
					"nb_node_e",
					"nb_node_ne",
					"nb_node_formula_e",
					"nb_node_formula_ne",
					"nb_formula_node_e",
					"nb_formula_node_ne",
					"nb_joint_e",
					"nb_joint_ne",
					"nb_node_node_e",
					"nb_node_node_ne",
					"nb_node_formula_mi",
					"nb_formula_node_mi",
					"nb_node_node_mi"
				])

		else:
			if not (self.hparams.formula_entropy_weight == self.hparams.formula_normalized_entropy_weight == 0.):
				loss_names.extend([
					"formula_e",
					"formula_ne",
				])
			if not (self.hparams.node_entropy_weight == self.hparams.node_normalized_entropy_weight == 0.):
				loss_names.extend([
					"node_e",
					"node_ne",
				])
			if not (self.hparams.node_formula_entropy_weight == self.hparams.node_formula_normalized_entropy_weight == 0.):
				loss_names.extend([
					"node_formula_e",
					"node_formula_ne",
				])
			if not (self.hparams.formula_node_entropy_weight == self.hparams.formula_node_normalized_entropy_weight == 0.):
				loss_names.extend([
					"formula_node_e",
					"formula_node_ne",
				])
			if not (self.hparams.joint_entropy_weight == self.hparams.joint_normalized_entropy_weight == 0.):
				loss_names.extend([
					"joint_e",
					"joint_ne",
				])
			if self.hparams.skip_edge_loss:
				if not (self.hparams.edge_h_range_loss_weight == self.hparams.edge_h_transfer_loss_weight == 0.):
					loss_names.extend([
						"edge_h_range_loss",
						"edge_h_transfer_loss",
					])
				if not (self.hparams.edge_entropy_weight == self.hparams.edge_normalized_entropy_weight == 0.):
					loss_names.extend([
						"edge_e",
						"edge_ne",
					])
			if self.hparams.nb_iso:
				if not (self.hparams.nb_node_entropy_weight == self.hparams.nb_node_normalized_entropy_weight == 0.):
					loss_names.extend([
						"nb_node_e",
						"nb_node_ne",
					])
				if not (self.hparams.nb_node_formula_entropy_weight == self.hparams.nb_node_formula_normalized_entropy_weight == 0.):
					loss_names.extend([
						"nb_node_formula_e",
						"nb_node_formula_ne",
					])
				if not (self.hparams.nb_formula_node_entropy_weight == self.hparams.nb_formula_node_normalized_entropy_weight == 0.):
					loss_names.extend([
						"nb_formula_node_e",
						"nb_formula_node_ne",
					])
				if not (self.hparams.nb_joint_entropy_weight == self.hparams.nb_joint_normalized_entropy_weight == 0.):
					loss_names.extend([
						"nb_joint_e",
						"nb_joint_ne",
					])
				if not (self.hparams.nb_node_node_entropy_weight == self.hparams.nb_node_node_normalized_entropy_weight == 0.):
					loss_names.extend([
						"nb_node_node_e",
						"nb_node_node_ne",
					])
		self.loss_names = loss_names
		self.metric_names.update(loss_names)

	def _setup_loss_fn(self):

		# cross entropy
		sparse_ce_fn = get_sparse_cross_entropy_fn(
			dist=self.hparams.output_distribution,
			vectorized=self.hparams.loss_vectorized,
			tolerance=self.tolerance,
			relative=self.relative,
			tolerance_min_mz=self.tolerance_min_mz,
			oos_tolerance_multiple=self.hparams.oos_tolerance_multiple,
			gaussian_renormalize=self.hparams.gaussian_renormalize,
			pm_tolerance_multiple=self.hparams.pm_tolerance_multiple,
			loss_batch_size=self.hparams.loss_batch_size
		)
		# edge loss
		edge_h_range_loss_fn = get_edge_loss_fn(self.hparams.edge_h_range_loss_fn,4.0)
		edge_h_transfer_loss_fn = get_edge_loss_fn(self.hparams.edge_h_transfer_loss_fn,4.0) #,1.0)

		# binned cosine distance
		def cos_dist_fn(
			true_mzs,
			true_logprobs,
			true_batch_idxs,
			pred_mzs,
			pred_logprobs,
			pred_batch_idxs):
			return sparse_cosine_distance_binned(
				true_mzs,
				true_logprobs,
				true_batch_idxs,
				pred_mzs,
				pred_logprobs,
				pred_batch_idxs,
				log_distance=(self.hparams.loss_type == "log_cosine_distance")
			)

		# auxiliary cosine distance for CE training.
		# This version accepts raw sparse mz/logprob peaks and bins internally.
		def aux_cos_dist_fn(
			true_mzs,
			true_logprobs,
			true_batch_idxs,
			pred_mzs,
			pred_logprobs,
			pred_batch_idxs):
			return sparse_cosine_distance(
				true_mzs=true_mzs,
				true_logprobs=true_logprobs,
				true_batch_idxs=true_batch_idxs,
				pred_mzs=pred_mzs,
				pred_logprobs=pred_logprobs,
				pred_batch_idxs=pred_batch_idxs,
				mz_max=self.hparams.mz_max,
				mz_bin_res=float(getattr(self.hparams, "aux_cosine_mz_bin_res", 0.01)),
				log_distance=False,
			)

		# hungarian cosine distance
		def cos_hun_fn(
				true_mzs,
				true_logprobs,
				true_batch_idxs,
				pred_mzs,
				pred_logprobs,
				pred_batch_idxs):
				return sparse_cosine_distance_hungarian(
					true_mzs,
					true_logprobs,
					true_batch_idxs,
					pred_mzs,
					pred_logprobs,
					pred_batch_idxs,
					tolerance=self.tolerance,
					relative=self.relative,
					log_distance=(self.hparams.loss_type == "log_cosine_distance_hungarian")
				)

		assert self.hparams.sparse_cosine_similarity
		assert self.hparams.sum_ints
		if self.hparams.loss_type == "cross_entropy":
			self.binned_loss = False
		elif self.hparams.loss_type in ["cosine_distance", "log_cosine_distance"]:
			self.binned_loss = True
		elif self.hparams.loss_type in ["cosine_distance_hungarian", "log_cosine_distance_hungarian"]:
			self.binned_loss = False
		else:
			raise ValueError(f"Unknown loss type {self.hparams.loss_type}")
			
		if self.binned_loss:
			if self.hparams.ints_transform != "none" and not self.hparams.bin_output:
				print("> Warning: binned loss with ints transform; output is not binned; inverse transform will be biased for binned metrics")
		else:
			assert not self.hparams.bin_output, "cannot train binned model with unbinned loss"

		def edge_loss(
			pred_edge_logprobs,
			pred_edge_h_diffs,
			pred_edge_h_range_masks,
			pred_edge_h_logprobs,
			pred_edge_batch_idxs):
			edge_h_range_costs = edge_h_range_loss_fn(pred_edge_h_diffs * pred_edge_h_range_masks.float())
			edge_h_range_costs = edge_h_range_costs.reshape(edge_h_range_costs.shape[0],-1)
			edge_h_transfer_costs = edge_h_transfer_loss_fn(pred_edge_h_diffs)
			edge_h_transfer_costs = edge_h_transfer_costs.reshape(edge_h_transfer_costs.shape[0],-1)
			edge_h_logprobs = pred_edge_h_logprobs.reshape(pred_edge_h_logprobs.shape[0],-1)
			edge_h_range_avg_cost = th.logsumexp(edge_h_logprobs + safelog(edge_h_range_costs, eps=self.hparams.log_min),dim=1)
			edge_h_transfer_avg_cost = th.logsumexp(edge_h_logprobs + safelog(edge_h_transfer_costs, eps=self.hparams.log_min),dim=1)
			h_range_loss = scatter_logsumexp(
				pred_edge_logprobs + edge_h_range_avg_cost,
				pred_edge_batch_idxs
			)
			h_transfer_loss = scatter_logsumexp(
				pred_edge_logprobs + edge_h_transfer_avg_cost,
				pred_edge_batch_idxs
			)
			return h_range_loss, h_transfer_loss
		
		self._setup_loss_names()

		def _loss_fn(
			true_mzs,
			true_logprobs,
			true_batch_idxs,
			pred_mzs,
			pred_logprobs,
			pred_batch_idxs,
			pred_formula_logprobs,
			pred_formula_formula_idxs,
			pred_formula_batch_idxs,
			pred_node_logprobs,
			pred_node_node_idxs,
			pred_node_batch_idxs,
			pred_node_formula_logprobs,
			pred_formula_node_logprobs,
			pred_joint_logprobs,
			pred_joint_node_idxs,
			pred_joint_formula_idxs,
			pred_joint_batch_idxs,
			pred_null_formula_logprob,
			pred_edge_logprobs,
			pred_edge_h_diffs,
			pred_edge_h_range_masks,
			pred_edge_h_logprobs,
			pred_edge_batch_idxs,
			pred_oos_logprobs,
			pred_h_logprobs,
			pred_h_counts,
			pred_h_batch_idxs,
			pred_nb_node_logprobs,
			pred_nb_node_formula_logprobs,
			pred_nb_formula_node_logprobs,
			pred_nb_joint_logprobs,
			pred_nb_node_node_logprobs,
			pred_nb_node_node_idxs,
			pred_nb_node_batch_idxs,
			pred_nb_joint_node_idxs,
			pred_nb_joint_formula_idxs,
			pred_nb_joint_batch_idxs,
			pred_nb_node_node_node_idxs,
			pred_nb_node_node_batch_idxs,
			**kwargs):
			
			loss_d = {}
			
			if self.hparams.loss_type == "cross_entropy":
				ios_ce, oos_ce, _, _ = sparse_ce_fn(
					true_mzs,
					true_logprobs,
					true_batch_idxs,
					pred_mzs,
					pred_logprobs,
					pred_batch_idxs,
					pred_oos_logprobs
				)
				spec_ce = ios_ce + oos_ce
				if self.hparams.oos_loss:
					primary_loss = spec_ce
				else:
					primary_loss = ios_ce
				loss_d["ios_ce"] = ios_ce
				loss_d["oos_ce"] = oos_ce
				loss_d["spec_ce"] = spec_ce
			elif self.hparams.loss_type in ["cosine_distance", "log_cosine_distance"]:
				primary_loss = cos_dist_fn(
					true_mzs,
					true_logprobs,
					true_batch_idxs,
					pred_mzs,
					pred_logprobs,
					pred_batch_idxs
				)
			elif self.hparams.loss_type in ["cosine_distance_hungarian", "log_cosine_distance_hungarian"]:
				primary_loss = cos_hun_fn(
					true_mzs,
					true_logprobs,
					true_batch_idxs,
					pred_mzs,
					pred_logprobs,
					pred_batch_idxs
				)

			loss = self.hparams.primary_loss_weight * primary_loss
			loss_d["primary_loss"] = primary_loss			

			# ===== C2: auxiliary cosine loss on top of cross entropy =====
			# Keep CE as the main objective to preserve OOS / probability calibration,
			# and add a small cosine-distance term to align with the final metric.
			if getattr(self.hparams, "use_aux_cosine_loss", False):
				aux_cosine_loss = aux_cos_dist_fn(
					true_mzs=true_mzs,
					true_logprobs=true_logprobs,
					true_batch_idxs=true_batch_idxs,
					pred_mzs=pred_mzs,
					pred_logprobs=pred_logprobs,
					pred_batch_idxs=pred_batch_idxs,
				)

				aux_cosine_loss = th.nan_to_num(
					aux_cosine_loss,
					nan=1.0,
					posinf=1.0,
					neginf=1.0,
				)

				aux_w = float(getattr(self.hparams, "aux_cosine_loss_weight", 0.03))
				loss = loss + aux_w * aux_cosine_loss
				loss_d["aux_cosine_loss"] = aux_cosine_loss

			# ===== R4a: binned intensity auxiliary KL loss =====
			# This is a small training-only auxiliary objective.
			# It directly supervises final predicted spectrum intensity distribution
			# after m/z binning, matching the useful signal found by D2 calibration.
			if getattr(self.hparams, "use_binned_intensity_aux_loss", False):
				binned_intensity_aux_loss = self._binned_intensity_aux_loss(
					true_mzs=true_mzs,
					true_logprobs=true_logprobs,
					true_batch_idxs=true_batch_idxs,
					pred_mzs=pred_mzs,
					pred_logprobs=pred_logprobs,
					pred_batch_idxs=pred_batch_idxs,
				)

				binned_intensity_aux_loss = th.nan_to_num(
					binned_intensity_aux_loss,
					nan=0.0,
					posinf=10.0,
					neginf=0.0,
				)

				aux_w = float(
					getattr(
						self.hparams,
						"binned_intensity_aux_loss_weight",
						0.02,
					)
				)

				loss = loss + aux_w * binned_intensity_aux_loss
				loss_d["binned_intensity_aux_loss"] = binned_intensity_aux_loss
			
			if self.hparams.null_loss:
				assert self.hparams.loss_type != "cross_entropy", self.hparams.loss_type
				if self.hparams.loss_type in ["cosine_distance","cosine_distance_hungarian"]:
					loss = loss + self.hparams.null_loss_weight * th.exp(pred_null_formula_logprob)
				elif self.hparams.loss_type in ["log_cosine_distance","log_cosine_distance_hungarian"]:
					loss = loss + self.hparams.null_loss_weight * pred_null_formula_logprob

			if "spec_e" in self.loss_names or "spec_ne" in self.loss_names:
				spec_e, spec_ne = sparse_entropy_fn(
					pred_logprobs,
					pred_batch_idxs,
					oos_logprobs=pred_oos_logprobs,
					renormalize=True
				)
				loss_d["spec_e"] = spec_e
				loss_d["spec_ne"] = spec_ne

			if "formula_e" in self.loss_names or "formula_ne" in self.loss_names:
				# formula logprobs do not consider NULL, OOS
				formula_e, formula_ne = sparse_entropy_fn(
					pred_formula_logprobs,
					pred_formula_batch_idxs,
					support_size_delta=-1. # for NULL formula
				)
				loss = loss + self.hparams.formula_entropy_weight * formula_e \
					+ self.hparams.formula_normalized_entropy_weight * formula_ne
				loss_d["formula_e"] = formula_e
				loss_d["formula_ne"] = formula_ne

			if "node_e" in self.loss_names or "node_ne" in self.loss_names:
				node_e, node_ne = sparse_entropy_fn(
					pred_node_logprobs,
					pred_node_batch_idxs
				)
				loss = loss + self.hparams.node_entropy_weight * node_e \
					+ self.hparams.node_normalized_entropy_weight * node_ne
				loss_d["node_e"] = node_e
				loss_d["node_ne"] = node_ne
			
			if "node_formula_e" in self.loss_names or "node_formula_ne" in self.loss_names:
				# node_formula logprobs do not consider NULL, OOS
				node_formula_e, node_formula_ne = sparse_conditional_entropy_fn(
					pred_node_logprobs,
					pred_node_batch_idxs,
					pred_node_formula_logprobs,
					pred_joint_node_idxs,
					pred_joint_batch_idxs,
				)
				loss = loss + self.hparams.node_formula_entropy_weight * node_formula_e \
					+ self.hparams.node_formula_normalized_entropy_weight * node_formula_ne
				loss_d["node_formula_e"] = node_formula_e
				loss_d["node_formula_ne"] = node_formula_ne
			
			if "formula_node_e" in self.loss_names or "formula_node_ne" in self.loss_names:
				# formula_node logprobs do not consider NULL, OOS
				formula_node_e, formula_node_ne = sparse_conditional_entropy_fn(
					pred_formula_logprobs,
					pred_formula_batch_idxs,
					pred_formula_node_logprobs,
					pred_joint_formula_idxs,
					pred_joint_batch_idxs
				)
				loss = loss + self.hparams.formula_node_entropy_weight * formula_node_e \
					+ self.hparams.formula_node_normalized_entropy_weight * formula_node_ne
				loss_d["formula_node_e"] = formula_node_e
				loss_d["formula_node_ne"] = formula_node_ne
			
			if "joint_e" in self.loss_names or "joint_ne" in self.loss_names:
				# joint
				joint_e, joint_ne = sparse_entropy_fn(
					pred_joint_logprobs,
					pred_joint_batch_idxs
				)
				# assert th.all(th.isclose(joint_e, node_formula_e+node_e,rtol=0.,atol=0.001))
				# assert th.all(th.isclose(joint_e, formula_node_e+formula_e,rtol=0.,atol=0.001))
				loss = loss + self.hparams.joint_entropy_weight * joint_e \
					+ self.hparams.joint_normalized_entropy_weight * joint_ne
				loss_d["joint_e"] = joint_e
				loss_d["joint_ne"] = joint_ne
			
			if "node_formula_mi" in self.loss_names:
				# mi
				node_formula_mi = formula_e - node_formula_e
				loss_d["node_formula_mi"] = node_formula_mi

			if "formula_node_mi" in self.loss_names:
				formula_node_mi = node_e - formula_node_e
				loss_d["formula_node_mi"] = formula_node_mi

			# special probabilities
			null_formula_prob = th.exp(pred_null_formula_logprob)
			oos_prob = th.exp(pred_oos_logprobs)
			loss_d["null_formula_prob"] = null_formula_prob
			loss_d["oos_prob"] = oos_prob

			if "h_mean" in self.loss_names or "h_e" in self.loss_names or "h_ne" in self.loss_names:
				# hydrogens
				h_mean = scatter_reduce(
					th.exp(pred_h_logprobs) * pred_h_counts,
					pred_h_batch_idxs,
					reduce="sum"
				)
				h_e, h_ne = sparse_entropy_fn(
					pred_h_logprobs,
					pred_h_batch_idxs
				)
				loss_d["h_mean"] = h_mean
				loss_d["h_e"] = h_e
				loss_d["h_ne"] = h_ne
			
			if not self.hparams.skip_edge_loss:
				if "edge_h_range_loss" in self.loss_names or "edge_h_transfer_loss" in self.loss_names:
					r_el, t_el = edge_loss(
						pred_edge_logprobs,
						pred_edge_h_diffs,
						pred_edge_h_range_masks,
						pred_edge_h_logprobs,
						pred_edge_batch_idxs
					)
					loss = loss + self.hparams.edge_h_range_loss_weight * r_el \
						+ self.hparams.edge_h_transfer_loss_weight * t_el
					loss_d["edge_h_range_loss"] = r_el
					loss_d["edge_h_transfer_loss"] = t_el
				if "edge_e" in self.loss_names or "edge_ne" in self.loss_names:
					edge_e, edge_ne = sparse_entropy_fn(
						pred_edge_logprobs,
						pred_edge_batch_idxs
					)
					loss = loss + self.hparams.edge_entropy_weight * edge_e \
						+ self.hparams.edge_normalized_entropy_weight * edge_ne
					loss_d["edge_e"] = edge_e
					loss_d["edge_ne"] = edge_ne
			else:
				assert self.hparams.edge_h_range_loss_weight == 0.0
				assert self.hparams.edge_h_transfer_loss_weight == 0.0
				assert self.hparams.edge_entropy_weight == 0.0
				assert self.hparams.edge_normalized_entropy_weight == 0.0
			
			if self.hparams.nb_iso:
				if "nb_node_e" in self.loss_names or "nb_node_ne" in self.loss_names:
					nb_node_e, nb_node_ne = sparse_entropy_fn(
						pred_nb_node_logprobs,
						pred_nb_node_batch_idxs
					)
					loss = loss + self.hparams.nb_node_entropy_weight * nb_node_e + \
						self.hparams.nb_node_normalized_entropy_weight * nb_node_ne
					loss_d["nb_node_e"] = nb_node_e
					loss_d["nb_node_ne"] = nb_node_ne
				if "nb_node_formula_e" in self.loss_names or "nb_node_formula_ne" in self.loss_names:
					nb_node_formula_e, nb_node_formula_ne = sparse_conditional_entropy_fn(
						pred_nb_node_logprobs,
						pred_nb_node_batch_idxs,
						pred_nb_node_formula_logprobs,
						pred_nb_joint_node_idxs,
						pred_nb_joint_batch_idxs,
					)
					loss = loss + self.hparams.nb_node_formula_entropy_weight * nb_node_formula_e + \
						self.hparams.nb_node_formula_normalized_entropy_weight * nb_node_formula_ne
					loss_d["nb_node_formula_e"] = nb_node_formula_e
					loss_d["nb_node_formula_ne"] = nb_node_formula_ne
				if "nb_formula_node_e" in self.loss_names or "nb_formula_node_ne" in self.loss_names:
					nb_formula_node_e, nb_formula_node_ne = sparse_conditional_entropy_fn(
						pred_formula_logprobs,
						pred_formula_batch_idxs,
						pred_nb_formula_node_logprobs,
						pred_nb_joint_formula_idxs,
						pred_nb_joint_batch_idxs
					)
					assert not th.any(nb_formula_node_ne < 0.), nb_formula_node_ne
					assert not th.any(nb_formula_node_ne > 1.), nb_formula_node_ne
					loss = loss + self.hparams.nb_formula_node_entropy_weight * nb_formula_node_e + \
						self.hparams.nb_formula_node_normalized_entropy_weight * nb_formula_node_ne
					loss_d["nb_formula_node_e"] = nb_formula_node_e
					loss_d["nb_formula_node_ne"] = nb_formula_node_ne
				if "nb_node_node_e" in self.loss_names or "nb_node_node_ne" in self.loss_names:
					nb_node_node_e, nb_node_node_ne = sparse_conditional_entropy_fn(
						pred_nb_node_logprobs,
						pred_nb_node_batch_idxs,
						pred_nb_node_node_logprobs,
						pred_nb_node_node_node_idxs,
						pred_nb_node_node_batch_idxs
					)
					loss = loss + self.hparams.nb_node_node_entropy_weight * nb_node_node_e + \
						self.hparams.nb_node_node_normalized_entropy_weight * nb_node_node_ne
					loss_d["nb_node_node_e"] = nb_node_node_e
					loss_d["nb_node_node_ne"] = nb_node_node_ne
				if "nb_joint_e" in self.loss_names or "nb_joint_ne" in self.loss_names:
					nb_joint_e, nb_joint_ne = sparse_entropy_fn(
						pred_nb_joint_logprobs,
						pred_nb_joint_batch_idxs
					)
					# assert th.all(th.isclose(nb_joint_e, nb_node_formula_e+nb_node_e,rtol=0.,atol=0.001))
					# assert th.all(th.isclose(nb_joint_e, nb_formula_node_e+formula_e,rtol=0.,atol=0.001))
					loss = loss + self.hparams.nb_joint_entropy_weight * nb_joint_e \
						+ self.hparams.nb_joint_normalized_entropy_weight * nb_joint_ne
					loss_d["nb_joint_e"] = nb_joint_e
					loss_d["nb_joint_ne"] = nb_joint_ne
				if "nb_node_formula_mi" in self.loss_names:
					nb_node_formula_mi = formula_e - nb_node_formula_e
					loss_d["nb_node_formula_mi"] = nb_node_formula_mi
				if "nb_formula_node_mi" in self.loss_names:
					nb_formula_node_mi = nb_node_e - nb_formula_node_e
					loss_d["nb_formula_node_mi"] = nb_formula_node_mi
				if "nb_node_node_mi" in self.loss_names:
					nb_node_node_mi = nb_node_e - nb_node_node_e
					loss_d["nb_node_node_mi"] = nb_node_node_mi
			else:
				assert self.hparams.nb_node_entropy_weight == 0.0
				assert self.hparams.nb_node_normalized_entropy_weight == 0.0
				assert self.hparams.nb_node_formula_entropy_weight == 0.0
				assert self.hparams.nb_node_formula_normalized_entropy_weight == 0.0
				assert self.hparams.nb_formula_node_entropy_weight == 0.0
				assert self.hparams.nb_formula_node_normalized_entropy_weight == 0.0
			
			# finally, update the loss
			loss_d["loss"] = loss
			
			return loss_d
		
		self.loss_fn = _loss_fn

	def on_train_epoch_end(self):

		if not self.hparams.automatic_optimization and self.hparams.dynamic_batch_sampler and self._cur_batch_size > 0:
			# this is used for manul opt when lr_schedulers is not available from torch lightning
			self._manual_opt()

		super().on_train_epoch_end()

	def _manual_opt(self):
		""" run manual opt
		"""
		opt = self.optimizers()
		# scale gradients
		if self.hparams.dynamic_batch_sampler:
			for pg in opt.param_groups:
				for p in pg['params']:
					if p.grad is not None:
						p.grad.data.mul_(self._max_batch_size / self._cur_batch_weight)					
		self.on_before_optimizer_step(opt)
		# clip gradients
		self.clip_gradients(opt, gradient_clip_val=self.hparams.gradient_clip_val, gradient_clip_algorithm=self.hparams.gradient_clip_algorithm)
		# call opt
		opt.step()
		self.on_after_optimizer_step(opt)
		# clean gradient
		opt.zero_grad()

	def training_step(self, batch, batch_idx):
		""" training loop for FragGNNPL

		Args:
			batch (_type_): _description_
			batch_idx (_type_): _description_

		Raises:
			NotImplementedError: _description_

		Returns:
			_type_: _description_
		"""

		if self.hparams.automatic_optimization:
			batch_results = self._common_step(batch, split="train")
			mean_loss = batch_results["mean_loss"]
			if self.hparams.debug_zero_loss:
				mean_loss = 0. * mean_loss
				self.print(mean_loss)
			self._cur_batch_size += self.hparams.train_batch_size
			self._update_results(batch_results,"train")
			return mean_loss
		
		elif self.hparams.dynamic_batch_sampler:
			batch_size = batch['batch_size'].item()
			assert batch_size > 0, batch_size
			batch_results = self._common_step(batch, split="train")
			total_loss = batch_results["total_loss"]
			total_weight = batch_results["total_weight"]
			mean_loss = total_loss / self._max_batch_size
			if self.hparams.debug_zero_loss:
				mean_loss = 0. * mean_loss
				self.print(mean_loss)
			self.manual_backward(mean_loss)
			# accumulate gradients of N samples
			self._cur_batch_size += batch_size
			self._cur_batch_weight += total_weight
			if self._cur_batch_size >= self._max_batch_size:
				self._manual_opt()
			self._update_results(batch_results,"train")
			return mean_loss
		
		else:
			batch_results = self._common_step(batch, split="train")
			mean_loss = batch_results["mean_loss"] / self.hparams.accumulate_grad_batches
			total_weight = batch_results["total_weight"]
			if self.hparams.debug_zero_loss:
				mean_loss = 0. * mean_loss
				self.print(mean_loss)
			self.manual_backward(mean_loss)
			# accumulate gradients of N samples
			self._cur_batch_size += self.hparams.train_batch_size
			self._cur_batch_weight += total_weight
			if (batch_idx + 1) % self.hparams.accumulate_grad_batches == 0:
				self._manual_opt()
			self._update_results(batch_results,"train")
			return mean_loss
		
	def optimizer_step(self, *args, **kwargs):
		
		super().optimizer_step(*args, **kwargs)
		opt = self.optimizers()
		self.on_after_optimizer_step(opt)

class BinnedPL(SpectrumPL):

	def _setup_loss_fn(self):
		
		# binned cosine distance
		def cos_dist_fn(
			true_mzs,
			true_logprobs,
			true_batch_idxs,
			pred_mzs,
			pred_logprobs,
			pred_batch_idxs):
			return sparse_cosine_distance_binned(
				true_mzs,
				true_logprobs,
				true_batch_idxs,
				pred_mzs,
				pred_logprobs,
				pred_batch_idxs,
				log_distance=(self.hparams.loss_type == "log_cosine_distance")
			)
		
		assert self.hparams.loss_type == "cosine_distance", self.hparams.loss_type
		assert self.hparams.sparse_cosine_similarity
		self.binned_loss = True

		def _loss_fn(
			true_mzs,
			true_logprobs,
			true_batch_idxs,
			pred_mzs,
			pred_logprobs,
			pred_batch_idxs,
			**kwargs):
			
			spec_cd = cos_dist_fn(
				true_mzs,
				true_logprobs,
				true_batch_idxs,
				pred_mzs,
				pred_logprobs,
				pred_batch_idxs
			)
			loss = spec_cd
			loss_d = {
				"loss": loss,
				"spec_cd": spec_cd
			}
			return loss_d

		self.loss_fn = _loss_fn
		# flag metrics for tracking
		loss_names = [
			"loss",
			"spec_cd"
		]
		self.metric_names.update(loss_names)

class NeimsPL(BinnedPL):

	def _setup_model(self):

		self.model = NeimsModel(
			mol_fingerprint_morgan=self.hparams.mol_params["fingerprint_morgan"],
			mol_fingerprint_rdkit=self.hparams.mol_params["fingerprint_rdkit"],
			mol_fingerprint_maccs=self.hparams.mol_params["fingerprint_maccs"],
			mlp_hidden_size=self.hparams.mlp_hidden_size,
			mlp_dropout=self.hparams.mlp_dropout,
			mlp_num_layers=self.hparams.mlp_num_layers,
			mlp_use_residuals=self.hparams.mlp_use_residuals,
			mz_max=self.hparams.mz_max,
			mz_bin_res=self.hparams.mz_bin_res,
			ff_prec_mz_offset=self.hparams.ff_prec_mz_offset,
			ff_bidirectional=self.hparams.ff_bidirectional,
			ff_output_map_size=self.hparams.ff_output_map_size,
			ff_output_activation=self.hparams.ff_output_activation,
			int_embedder=self.hparams.int_embedder,
			ce_insert_type=self.hparams.ce_insert_type,
			ce_insert_location=self.hparams.ce_insert_location,
			ce_insert_merge=self.hparams.ce_insert_merge,
			ce_insert_size=self.hparams.ce_insert_size,
   			ce_mean=self.hparams.ce_mean,
			ce_std=self.hparams.ce_std,
			ce_max=self.hparams.ce_max,
			prec_insert_location=self.hparams.prec_insert_location,
			prec_insert_size=self.hparams.prec_insert_size,
			prec_types=self.hparams.spec_params["prec_types"],
			inst_insert_location=self.hparams.inst_insert_location,
			inst_insert_size=self.hparams.inst_insert_size,
			inst_types=self.hparams.spec_params["inst_types"],
			log_min=self.hparams.log_min
		)
		
		self._check_ce_params()

		# compile
		if self.hparams.compile:
			th_dynamo.reset()
			self.dynamo_prof = th_dynamo.utils.CompileProfiler()
			self.model = th.compile(self.model,backend=self.dynamo_prof,dynamic=True)

class PrecursorPL(SpectrumPL):

	def _setup_model(self):

		assert not self.hparams.compile
		self.model = PrecursorModel()

	def _setup_loss_names(self):

		# flag losses for tracking
		loss_names = [
			"loss",
			"primary_loss",
		]
		self.loss_names = loss_names
		self.metric_names.update(loss_names)

	def _setup_loss_fn(self):

		assert self.hparams.loss_type == "cross_entropy"

		# cross entropy
		sparse_ce_fn = get_sparse_cross_entropy_fn(
			dist=self.hparams.output_distribution,
			vectorized=self.hparams.loss_vectorized,
			tolerance=self.tolerance,
			relative=self.relative,
			tolerance_min_mz=self.tolerance_min_mz,
			oos_tolerance_multiple=self.hparams.oos_tolerance_multiple,
			gaussian_renormalize=self.hparams.gaussian_renormalize,
			pm_tolerance_multiple=self.hparams.pm_tolerance_multiple,
			loss_batch_size=self.hparams.loss_batch_size
		)

		self._setup_loss_names()

		def loss_fn(
			true_mzs,
			true_logprobs,
			true_batch_idxs,
			pred_mzs,
			pred_logprobs,
			pred_batch_idxs,
			**kwargs):

			batch_size = th.max(true_batch_idxs).item()+1
			assert pred_mzs.shape[0] == pred_logprobs.shape[0] == batch_size
			ios_ce, oos_ce, true_oos_logprob, true_oos_e = sparse_ce_fn(
				true_mzs,
				true_logprobs,
				true_batch_idxs,
				pred_mzs,
				pred_logprobs,
				pred_batch_idxs,
				th.full((batch_size,), LOG_ZERO(true_logprobs.dtype), device=pred_logprobs.device, dtype=pred_logprobs.dtype)
			)
			spec_ce = ios_ce + oos_ce
			if self.hparams.oos_loss:
				raise NotImplementedError
				primary_loss = spec_ce
			else:
				primary_loss = ios_ce
			loss = primary_loss
			loss_d = {
				"loss": loss,
				"primary_loss": primary_loss,
			}
			return loss_d

		self.loss_fn = loss_fn
		self.binned_loss = False

class GNNPL(BinnedPL):
	def _setup_model(self):

		self.model = GNNModel( 	# TODO: initialize model based on params. finalize params
			mol_node_feats=self.hparams.mol_params["pyg_node_feats"],	# mol feats
			mol_edge_feats=self.hparams.mol_params["pyg_edge_feats"],
			mol_pe_embed_k=self.hparams.mol_params["pyg_pe_embed_k"],
			mol_hidden_size=self.hparams.mol_hidden_size,
			mol_num_layers=self.hparams.mol_num_layers,
			mol_gnn_type=self.hparams.mol_gnn_type,
			mol_normalization=self.hparams.mol_normalization,
			mol_dropout=self.hparams.mol_dropout,
			mol_pool_type=self.hparams.mol_pool_type,
			mlp_hidden_size=self.hparams.mlp_hidden_size,				# FFN
			mlp_dropout=self.hparams.mlp_dropout,
			mlp_num_layers=self.hparams.mlp_num_layers,
			mlp_use_residuals=self.hparams.mlp_use_residuals,
			mz_max=self.hparams.mz_max,
			mz_bin_res=self.hparams.mz_bin_res,
			ff_prec_mz_offset=self.hparams.ff_prec_mz_offset,
			ff_bidirectional=self.hparams.ff_bidirectional,
			ff_output_map_size=self.hparams.ff_output_map_size,
			ff_output_activation=self.hparams.ff_output_activation,
			int_embedder=self.hparams.int_embedder,						# cross entropy
			ce_insert_type=self.hparams.ce_insert_type,
			ce_insert_location=self.hparams.ce_insert_location,
			ce_insert_merge=self.hparams.ce_insert_merge,
			ce_insert_size=self.hparams.ce_insert_size,
      		ce_mean=self.hparams.ce_mean,
			ce_std=self.hparams.ce_std,
			ce_max=self.hparams.ce_max,
			prec_insert_location=self.hparams.prec_insert_location,		# precursor
			prec_insert_size=self.hparams.prec_insert_size,
			prec_types=self.hparams.spec_params["prec_types"],
			inst_insert_location=self.hparams.inst_insert_location,		# instrument
			inst_insert_size=self.hparams.inst_insert_size,
			inst_types=self.hparams.spec_params["inst_types"],
		)

		self._check_ce_params()
	
		# compile
		if self.hparams.compile:
			th_dynamo.reset()
			self.dynamo_prof = th_dynamo.utils.CompileProfiler()
			self.model = th.compile(self.model,backend=self.dynamo_prof,dynamic=True)

def test_models():

	from ms2spectra.workflow import load_config, init_dataset
	from ms2spectra.utils.misc_utils import print_shapes
	from ms2spectra.utils.misc_utils import to_device

	template_fp = "config/template.yml"
	custom_fp_to_model_cls = {
		# "config/debug_m/debug_d3_m.yml": FragGNNPL,
		# "config/debug_m/debug_d3_prec_m.yml": FragGNNPL,
		# "config/debug_m/debug_neims_prec_m.yml": NeimsPL,
		# "config/debug_um/debug_d3_prec_um.yml": FragGNNPL,
		'config/debug_m/debug_gnn.yml': GNNPL
	}

	for custom_fp, pl_model_cls in custom_fp_to_model_cls.items():

		print(">>>", custom_fp)

		config_d = load_config(template_fp, custom_fp)
		for k in ["spec_params","mol_params","frag_params"]:
			config_d[k]["preprocess"] = False
			if "preload" in config_d[k]:
				config_d[k]["preload"] = False
		device = "cuda:0" if config_d["accelerator"] == "gpu" else "cpu"

		ds = init_dataset(config_d, splits=("train",))[0]
		dl = th.utils.data.DataLoader(
			ds,
			batch_size=8,
			shuffle=False,
			num_workers=0,
			collate_fn=ds.get_collate_fn(),
		)
		dl_iter = iter(dl)
		batch = next(dl_iter)
		print("> batch")
		print_shapes(batch)
		batch = to_device(batch,device)

		pl_model = pl_model_cls(**config_d)
		pl_model.train()
		pl_model.to(device)

		outputs = pl_model.forward(**batch)
		print("> train outputs")
		print_shapes(outputs)
		print(th.exp(outputs["pred_logprobs"]).sum())

		results = pl_model._common_step(batch, split="train", log=False)
		print_shapes(results)

		mean_loss = results["mean_loss"]
		print("> loss")
		print(mean_loss)
		mean_loss.backward()

		print()


if __name__ == "__main__":

	from ms2spectra.utils.misc_utils import th_temp_seed

	with th_temp_seed(420):
		test_models()


"""
这是一个基于 **PyTorch Lightning** 的代码文件，用于管理模型的**训练、验证、测试循环**，以及**损失函数计算**、**指标评估**和**日志记录**。

如果说前一个文件定义了“大脑”（模型结构），那么这个文件定义了“健身房”（如何训练这个大脑）。

核心类结构如下：
*   `SpectrumPL`: 基类，定义了通用的谱图处理、指标计算、优化器配置和 Logging 逻辑。
*   `FragGNNPL`: 继承自基类，专门用于训练 `FragGNNModel`（核心模型），包含复杂的物理约束损失函数。
*   `BinnedPL` / `NeimsPL` / `GNNPL`: 用于训练对比模型（NEIMS, End-to-End GNN），通常使用分桶（Binned）损失。

下面非常详细地解释各个模块：

---

### 1. 基类: `SpectrumPL` (Spectrum PyTorch Lightning)

这是所有模型的父类，处理共性逻辑。

#### **关键功能**

1.  **数据预处理 (`_setup_spec_fns`, `preproc_spec`)**:
    *   在计算损失或指标前，对谱图数据进行处理。
    *   **Filtering**: 过滤掉强度低于阈值的噪音峰。
    *   **Binning**: 如果需要（如 NEIMS 模型），将连续的 m/z 值映射到固定的 Grid 上。
    *   **Normalization**: L1 归一化（峰强度之和为 1）。
    *   **Transformation**: 支持对强度取 `sqrt` 或 `log`，这在质谱对比中常用，用于平衡高丰度和低丰度峰的影响。

2.  **指标系统 (`_setup_metric_fns`)**:
    *   定义了极其丰富的评价指标，用于评估预测谱图和真实谱图的相似度：
    *   **Cosine Similarity (余弦相似度)**: 最常用的指标。支持 Sparse（基于匹配峰）和 Binned（基于桶）。
    *   **Recall / Precision**: 预测出的峰有多少是真实的，真实的峰有多少被预测到了。
    *   **JSS (Jensen-Shannon Similarity)**: 概率分布的相似度。
    *   **NDCG**: 考虑排名的指标，关注高强度峰是否预测准确。
    *   **Hungarian Matching**: 代码中出现了 `hun` 字样（如 `cos_hun`），指的是使用匈牙利算法进行峰的精确匹配（解决 m/z 漂移问题），而不是简单的分桶。

3.  **通用训练步 (`_common_step`)**:
    *   这是 `training_step`, `validation_step`, `test_step` 调用的核心函数。
    *   流程：前向传播 -> 预处理真实值和预测值 -> 计算损失 -> 计算所有指标 -> 聚合结果。
    *   **Sample Weighting**: 支持根据分子（Mol）或采集组（Group）对 Loss 进行加权。

4.  **可视化 (`_log_images`)**:
    *   从 Batch 中随机抽取样本，绘制“真实谱图 vs 预测谱图”的对比图（蝴蝶图），并记录到 WandB (Weights & Biases) 中。这对于直观判断模型效果非常重要。

---

### 2. 核心实现: `FragGNNPL`

这是专门配合前一个文件中的 `FragGNNModel` 使用的训练模块。

#### **模型初始化 (`_setup_model`)**
*   实例化 `FragGNNModel`，传入了巨量的超参数（层数、隐藏层大小、GNN类型、Dropout等）。
*   支持 `torch.compile` (PyTorch 2.0) 进行图编译加速。

#### **复杂的损失函数 (`_setup_loss_fn`)**
这是该文件最硬核的部分。为了训练一个符合物理规律的碎片图模型，损失函数不仅仅是“预测峰准不准”，还包含大量**正则化项**和**物理约束**：

1.  **Primary Loss (主损失)**:
    *   **Sparse Cross Entropy**: 核心损失。预测每个碎片节点生成的化学式概率 $P(f, n)$ 与真实谱图的差异。
    *   支持 **OOS (Out-of-Scope)**: 如果真实谱图中有峰无法被碎片图解释，模型会尝试通过一个特殊的 OOS 节点来“吸收”这些概率，而不是强行匹配错误碎片。

2.  **Entropy Regularization (熵正则化)**:
    *   代码中包含大量的 `_e` (entropy) 和 `_ne` (normalized entropy) 后缀的损失项，例如 `node_e`, `formula_e`, `joint_e`。
    *   **目的**: 防止模型“坍缩”或过度自信。通过惩罚或鼓励概率分布的熵，使模型在节点概率 $P(n)$、化学式概率 $P(f)$ 以及联合概率 $P(f,n)$ 之间保持一致性。
    *   **互信息 (Mutual Information, MI)**: 计算节点和化学式之间的互信息，鼓励模型学到有意义的依赖关系。

3.  **Edge Loss (边损失 - 物理约束)**:
    *   如果未跳过 (`skip_edge_loss=False`)，会计算 `edge_h_range_loss` 和 `edge_h_transfer_loss`。
    *   **逻辑**: 父碎片变成子碎片时，氢原子的变化量 ($\Delta H$) 必须符合化学规律（例如，不能通过断裂一个键突然增加 10 个氢）。
    *   模型预测父子节点间的氢转移概率，并惩罚违反物理范围 (`h_range`) 的预测。

4.  **Isomorphism Loss (同分异构体损失)**:
    *   如果开启 `nb_iso`，代码会计算关于同分异构子图的损失，确保结构相似的碎片有相似的概率分布。

#### **手动优化 (`_manual_opt`, `training_step`)**
*   `FragGNNPL` 实现了手动反向传播 (`manual_backward`)。
*   **原因**: 图神经网络的数据（Batch）大小不固定（节点数、边数差异巨大）。为了稳定训练，代码实现了 **Dynamic Batch Sampler** 和 **Gradient Accumulation**（梯度累积），确保每次更新参数时，处理的有效数据量大致相同，而不是简单地按图的数量计算。

---

### 3. 对比模型实现

#### `BinnedPL`
*   一个中间类。强制使用 **Binned Cosine Distance** 作为损失函数。
*   适用于输出固定长度向量（例如 10000 维的 intensity 向量）的模型。

#### `NeimsPL`
*   用于训练 `NeimsModel`。
*   输入是指纹，输出是 Binned 谱图。
*   结构简单，主要用于作为 Baseline 对比。

#### `GNNPL`
*   用于训练 `GNNModel`（端到端 GNN）。
*   它不构建碎片图，而是直接从分子图 Embedding 映射到谱图向量。
*   同样使用 Binned Loss。

#### `PrecursorPL`
*   哑模型，只基于前体离子预测，用于测试数据管道是否正常，或者作为最差情况的 Baseline。

---

### 4. 工具函数与执行

*   **`configure_optimizers`**: 支持 Adam, AdamW, SGD，以及 Learning Rate Warmup 和 Decay 策略。
*   **`_check_ce_params`**: 检查碰撞能量 (Collision Energy) 的配置是否冲突（例如，不能同时做 NCE 和 ACE 的特定组合）。
*   **`test_models`**: 文件末尾的一个测试函数。
    *   它加载配置文件。
    *   初始化数据集和 DataLoader。
    *   运行一个 Batch 的前向传播和反向传播。
    *   **用途**: 用于开发阶段快速 Debug，确保代码没有语法错误，Tensor 形状对齐。

### 总结

这个文件是整个深度学习项目的**控制中心**。它体现了非常工程化的深度学习实践：
1.  **高度模块化**: 将数据、模型、训练逻辑解耦。
2.  **鲁棒性**: 包含大量的 Assert 和形状检查，以及针对 NaN/Inf 的处理（`safelog`）。
3.  **科研导向**: 极其丰富的 Metrics 和 Loss 选项，允许研究人员精细地调整模型行为（如调整熵的权重、开启/关闭物理约束），并详细记录实验结果。
"""