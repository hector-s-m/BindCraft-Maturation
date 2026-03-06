#!/usr/bin/env python
"""
Standalone maturation runner for FreeBindCraft.

Takes an existing MPNN-predicted PDB (from a prior BindCraft run) and runs
only the maturation loop — skipping trajectory generation, MPNN sequence
optimization, and initial AF2 prediction.

Usage:
    python -u bindcraft_maturation_only.py \
        --i designs/PDL1-mat/MPNN/PDL1_l69_s833262_mpnn15_model1.pdb \
        --settings settings_target/PDL1.json \
        --filters settings_filters/relaxed_filters.json \
        --advanced settings_advanced/maturation_bindcraft_flow.json \
        --verbose
"""
import argparse
import gc
import json
import logging
import os
import re
import shutil
import sys
import time
import traceback

try:
    import resource
except Exception:
    resource = None

# --- Bootstrap ---
bindcraft_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, bindcraft_folder)

from functions import *
from functions.generic_utils import (
    insert_data, maturation_col_start,
    MAT_ROUNDS, MAT_CONVERGED, MAT_FIXED_RES, MAT_REDESIGN_RES,
    MAT_PRE_IPTM, MAT_POST_IPTM, MAT_PRE_IPSAE, MAT_POST_IPSAE,
    MAT_SCAN_MUTATIONS,
)
from functions.biopython_utils import clear_dssp_cache
from functions.logging_utils import vprint

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Run maturation loop on an existing MPNN PDB.")
parser.add_argument("--i", "-i", dest="input_pdb", required=True,
                    help="Path to the relaxed MPNN model PDB (the maturation candidate).")
parser.add_argument("--settings", "-s", required=True,
                    help="Path to target settings JSON (e.g. settings_target/PDL1.json).")
parser.add_argument("--filters", "-f", default="./settings_filters/default_filters.json",
                    help="Path to filters JSON.")
parser.add_argument("--advanced", "-a", default="./settings_advanced/default_4stage_multimer_maturation.json",
                    help="Path to advanced settings JSON.")
parser.add_argument("--no-pyrosetta", action="store_true",
                    help="Run without PyRosetta.")
parser.add_argument("--verbose", action="store_true",
                    help="Enable verbose logging.")
parser.add_argument("--debug-pdbs", action="store_true",
                    help="Write intermediate debug PDBs during OpenMM relax.")
args = parser.parse_args()

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for noisy in ("jax", "jaxlib", "jax._src", "absl", "flax", "colabdesign", "tensorflow", "xla"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
if args.verbose:
    logging.getLogger("functions").setLevel(logging.DEBUG)
    logging.getLogger("__main__").setLevel(logging.DEBUG)

# ------------------------------------------------------------------
# GPU check
# ------------------------------------------------------------------
check_jax_gpu()

# ------------------------------------------------------------------
# Load settings
# ------------------------------------------------------------------
settings_path, filters_path, advanced_path = perform_input_check(args)
target_settings, advanced_settings, filters = load_json_settings(settings_path, filters_path, advanced_path)

# Provide context to OpenMM relax
try:
    os.environ["BINDCRAFT_STARTING_PDB"] = os.path.abspath(target_settings["starting_pdb"])
    os.environ["BINDCRAFT_TARGET_CHAINS"] = str(target_settings.get("chains", "A"))
    os.environ["BINDCRAFT_DEBUG_PDBS"] = "1" if args.debug_pdbs else "0"
except Exception:
    pass

# AF2 model lists
design_models, prediction_models, multimer_validation = load_af2_models(
    advanced_settings["use_multimer_design"])

# Advanced settings validation
advanced_settings = perform_advanced_settings_check(advanced_settings, bindcraft_folder)

# Ensure binaries are executable
def _ensure_binaries(use_pyrosetta):
    functions_dir = os.path.join(bindcraft_folder, "functions")
    binaries = ["dssp", "sc", "FASPR"]
    if use_pyrosetta:
        binaries.append("DAlphaBall.gcc")
    for b in binaries:
        p = os.path.join(functions_dir, b)
        if os.path.isfile(p) and not os.access(p, os.X_OK):
            os.chmod(p, os.stat(p).st_mode | 0o755)

# ------------------------------------------------------------------
# PyRosetta init
# ------------------------------------------------------------------
use_pyrosetta = False
if args.no_pyrosetta:
    print("Running in PyRosetta-free mode.")
else:
    if "PYROSETTA_AVAILABLE" in dir() and PYROSETTA_AVAILABLE and pr is not None:
        try:
            pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all '
                    f'-holes:dalphaball {advanced_settings["dalphaball_path"]} '
                    f'-corrections::beta_nov16 true -relax:default_repeats 1')
            print("PyRosetta initialized successfully.")
            use_pyrosetta = True
        except Exception as e:
            print(f"PyRosetta init failed: {e}. Using OpenMM fallback.")
    else:
        print("PyRosetta not found. Using OpenMM routines.")

_ensure_binaries(use_pyrosetta)

# ------------------------------------------------------------------
# Directories
# ------------------------------------------------------------------
design_paths = generate_directories(target_settings["design_path"])

# ------------------------------------------------------------------
# Extract info from input PDB
# ------------------------------------------------------------------
input_pdb = os.path.abspath(args.input_pdb)
if not os.path.isfile(input_pdb):
    print(f"Error: input PDB not found: {input_pdb}")
    sys.exit(1)

# Derive design name from filename (e.g. PDL1_l69_s833262_mpnn15_model1.pdb -> PDL1_l69_s833262_mpnn15)
input_basename = os.path.splitext(os.path.basename(input_pdb))[0]
# Strip _modelN suffix if present
design_name = re.sub(r"_model\d+$", "", input_basename)
print(f"Design name: {design_name}")
print(f"Input PDB:   {input_pdb}")

# Extract binder sequence and length from PDB
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

pdb_parser = PDBParser(QUIET=True)
structure = pdb_parser.get_structure("input", input_pdb)
model = structure[0]

binder_chain = "B"  # Standard BindCraft convention
target_chain = target_settings["chains"]

# Get binder sequence
chain_obj = model[binder_chain]
binder_seq = ""
for residue in chain_obj.get_residues():
    if residue.get_id()[0] == " ":  # Skip hetero residues
        binder_seq += seq1(residue.get_resname())

length = len(binder_seq)
print(f"Binder chain: {binder_chain}, length: {length}, sequence: {binder_seq}")

# ------------------------------------------------------------------
# Run initial AF2 prediction to get PAE/pLDDT caches
# ------------------------------------------------------------------
print("\n=== Running initial AF2 prediction to obtain PAE/pLDDT ===")

# The trajectory PDB is the input PDB (serves as structural reference)
trajectory_pdb = input_pdb

# Create prediction model (clear_mem first, same as main pipeline)
clear_mem()
complex_prediction_model = mk_afdesign_model(
    protocol="binder",
    num_recycles=advanced_settings["num_recycles_validation"],
    data_dir=advanced_settings["af_params_dir"],
    use_multimer=multimer_validation,
    use_initial_guess=advanced_settings["predict_initial_guess"],
    use_initial_atom_pos=advanced_settings["predict_bigbang"])

if advanced_settings["predict_initial_guess"] or advanced_settings["predict_bigbang"]:
    complex_prediction_model.prep_inputs(
        pdb_filename=trajectory_pdb, chain="A", binder_chain="B",
        binder_len=length, use_binder_template=True,
        rm_target_seq=advanced_settings["rm_template_seq_predict"],
        rm_target_sc=advanced_settings["rm_template_sc_predict"],
        rm_template_ic=True)
else:
    complex_prediction_model.prep_inputs(
        pdb_filename=target_settings["starting_pdb"],
        chain=target_settings["chains"], binder_len=length,
        rm_target_seq=advanced_settings["rm_template_seq_predict"],
        rm_target_sc=advanced_settings["rm_template_sc_predict"])

# Generate CSVs needed by predict_binder_complex
trajectory_labels, design_labels, final_labels = generate_dataframe_labels()

mpnn_csv = os.path.join(target_settings["design_path"], "mpnn_design_stats.csv")
failure_csv = os.path.join(target_settings["design_path"], "failure_csv.csv")

# Ensure CSVs exist
create_dataframe(mpnn_csv, design_labels)
generate_filter_pass_csv(failure_csv, args.filters)

# Predict
pred_stats, pass_af2, _ = predict_binder_complex(
    complex_prediction_model, binder_seq, design_name,
    target_settings["starting_pdb"], target_settings["chains"],
    length, trajectory_pdb, prediction_models, advanced_settings,
    filters, design_paths, failure_csv, use_pyrosetta=use_pyrosetta)

if not pass_af2:
    print("WARNING: Initial AF2 prediction failed quality filters, but continuing with maturation anyway.")

# Pick best model by pLDDT
model_plddt = {m: pred_stats[m].get("pLDDT", 0)
               for m in pred_stats if not str(m).startswith("_")}
best_model_number = max(model_plddt, key=model_plddt.get) if model_plddt else 0
best_model_pdb_name = os.path.join(design_paths["MPNN"],
                                    f"{design_name}_model{best_model_number}.pdb")

# If prediction wrote a PDB, relax it; otherwise use input_pdb directly
if os.path.exists(best_model_pdb_name):
    best_model_pdb = best_model_pdb_name
else:
    print(f"  Predicted PDB not found at {best_model_pdb_name}, using input PDB")
    best_model_pdb = input_pdb

# Compute averages
complex_averages = calculate_averages(pred_stats, handle_aa=False)

print(f"\nInitial prediction results:")
print(f"  Best model: {best_model_number+1}")
print(f"  i_pTM:  {complex_averages.get('i_pTM', 'N/A')}")
print(f"  ipSAE:  {complex_averages.get('ipSAE', 'N/A')}")
print(f"  pLDDT:  {complex_averages.get('pLDDT', 'N/A')}")

# ------------------------------------------------------------------
# Build candidate and context dicts for _run_maturation
# ------------------------------------------------------------------

# Build a minimal mpnn_data row (just needs enough columns for maturation to write into)
mpnn_data = [""] * len(design_labels)
mpnn_data[0] = design_name
mpnn_data[6] = binder_seq  # Sequence column

cand = {
    "design_name": design_name,
    "mpnn_data": mpnn_data,
    "best_model_pdb": best_model_pdb,
    "best_model_number": best_model_number,
    "sequence": binder_seq,
    "complex_averages": complex_averages,
    "complex_statistics": pred_stats,
    "ipSAE": complex_averages.get("ipSAE", 0),
}

# Dummy seed and helicity
import random
seed = random.randint(0, 2**32 - 1)
helicity_value = 0.0

ctx = {
    "complex_prediction_model": complex_prediction_model,
    "design_models": design_models,
    "prediction_models": prediction_models,
    "target_settings": target_settings,
    "length": length,
    "seed": seed,
    "helicity_value": helicity_value,
    "binder_chain": binder_chain,
    "design_paths": design_paths,
    "failure_csv": failure_csv,
    "mpnn_csv": mpnn_csv,
    "use_pyrosetta": use_pyrosetta,
    "filters": filters,
    "trajectory_pdb": trajectory_pdb,
    "multimer_validation": multimer_validation,
}

# ------------------------------------------------------------------
# Import _run_maturation and _check_maturation_revert from bindcraft.py
# We inline them here to avoid importing all of bindcraft.py
# ------------------------------------------------------------------

# Copy of _run_maturation from bindcraft.py (lines 481-730)
def _run_maturation(cand, ctx, advanced_settings, mat_label="maturation"):
    mpnn_data = list(cand['mpnn_data'])
    mpnn_design_name = cand['design_name']
    best_model_pdb = cand['best_model_pdb']
    best_model_number = cand['best_model_number']
    mpnn_complex_averages = cand['complex_averages']
    mpnn_complex_statistics = cand['complex_statistics']

    complex_prediction_model = ctx['complex_prediction_model']
    design_models = ctx['design_models']
    prediction_models = ctx['prediction_models']
    target_settings = ctx['target_settings']
    length = ctx['length']
    seed = ctx['seed']
    helicity_value = ctx['helicity_value']
    binder_chain = ctx['binder_chain']
    design_paths = ctx['design_paths']
    failure_csv = ctx['failure_csv']
    mpnn_csv = ctx['mpnn_csv']
    use_pyrosetta = ctx['use_pyrosetta']
    filters = ctx['filters']
    trajectory_pdb = ctx['trajectory_pdb']
    multimer_validation = ctx['multimer_validation']

    mat_max_rounds = advanced_settings.get("maturation_max_rounds", 3)
    mat_metric_name = advanced_settings.get("maturation_improvement_metric", "i_pTM")
    mat_improvement_thresh = advanced_settings.get("maturation_improvement_threshold", 0.01)
    mat_ipsae_thresh = advanced_settings.get("maturation_ipsae_improvement_threshold", 0.01)

    best_pred_stats = mpnn_complex_statistics.get(best_model_number, {})
    mat_pae = best_pred_stats.get('_pae_matrix')
    mat_plddt = best_pred_stats.get('_plddt_array')
    mat_target_len = best_pred_stats.get('_target_len')
    mat_binder_len = best_pred_stats.get('_binder_len')

    if mat_pae is None or mat_plddt is None:
        vprint(f"[Maturation] Skipped: no cached PAE/pLDDT data available")
        return False, mpnn_data, best_model_pdb

    mat_current_pdb = best_model_pdb
    mat_current_seq = cand['sequence']
    mat_best_metric = get_maturation_metric(mpnn_complex_averages, mat_metric_name)
    mat_best_ipsae = mpnn_complex_averages.get('ipSAE', None)
    mat_pre_metric_iptm = mpnn_complex_averages.get('i_pTM', None)
    mat_pre_metric_ipsae = mat_best_ipsae
    ipsae_str = f", ipSAE: {mat_pre_metric_ipsae:.4f}" if mat_pre_metric_ipsae is not None else ""
    print(f"\n--- Starting {mat_label} maturation for {mpnn_design_name} (max {mat_max_rounds} rounds) ---")
    print(f"  Baseline: {mat_metric_name}={mat_best_metric:.4f}{ipsae_str}")
    mat_fixed_set = set()
    mat_rounds_completed = 0
    mat_converged = False
    mat_all_scan_mutations = []
    new_redesign = set()
    mat_averages = mpnn_complex_averages
    mat_best_averages = None

    for mat_round in range(1, mat_max_rounds + 1):
        mat_per_residue_reu, mat_pose, mat_scorefxn = compute_per_residue_reu(
            mat_current_pdb, binder_chain=binder_chain,
            use_pyrosetta=use_pyrosetta, return_pose=True)
        if mat_per_residue_reu is not None:
            vprint(f"  [Maturation] Per-residue REU computed for {len(mat_per_residue_reu)} residues")
        else:
            vprint(f"  [Maturation] REU unavailable, using pLDDT/PAE/contacts only")

        residue_quality = assess_interface_residue_quality(
            mat_current_pdb, mat_pae, mat_plddt,
            mat_target_len, mat_binder_len,
            advanced_settings, binder_chain=binder_chain,
            per_residue_reu=mat_per_residue_reu)

        if not residue_quality:
            print(f"  Maturation round {mat_round}: no interface residues found, stopping")
            break

        new_fixed, new_redesign, mat_converged = partition_interface_residues(
            residue_quality, mat_fixed_set, advanced_settings)
        mat_fixed_set = new_fixed

        if mat_converged:
            print(f"  Maturation converged at round {mat_round} — all interface residues are high quality")
            break

        if mat_fixed_set and use_pyrosetta:
            scan_output = os.path.join(design_paths["MPNN/Maturation"],
                                       f"{mpnn_design_name}_mat{mat_round}_scanned.pdb")
            scan_seq, scan_mutations = scan_fixed_residues(
                mat_current_pdb, mat_fixed_set, binder_chain,
                advanced_settings, scan_output,
                preloaded_pose=mat_pose,
                preloaded_scorefxn=mat_scorefxn)
            if scan_mutations:
                mat_current_pdb = scan_output
                mat_current_seq = scan_seq
                mat_all_scan_mutations.extend(scan_mutations)
                mut_str = ', '.join(f"{old}{idx+1}{new}" for idx, old, new, _, _ in scan_mutations)
                print(f"  [Scan] {len(scan_mutations)} mutations accepted: {mut_str}")

        mat_design_name = f"{mpnn_design_name}_mat{mat_round}"
        try:
            mat_af_model, mat_traj_pdb = binder_maturation_hallucination(
                mat_design_name, target_settings["starting_pdb"],
                target_settings["chains"], target_settings.get("target_hotspot_residues", ""),
                length, mat_current_seq, list(mat_fixed_set), seed,
                helicity_value, design_models, advanced_settings,
                design_paths, failure_csv)
        except Exception as e:
            print(f"  Maturation hallucination failed at round {mat_round}: {e}")
            traceback.print_exc()
            break

        mat_traj_contacts = hotspot_residues(mat_traj_pdb, binder_chain)

        mat_fix_str = format_fixed_positions_for_mpnn(mat_fixed_set, binder_chain)
        try:
            mat_mpnn_seqs = mpnn_gen_sequence_maturation(
                mat_traj_pdb, binder_chain, mat_fix_str, advanced_settings)
        except Exception as e:
            print(f"  Maturation MPNN failed at round {mat_round}: {e}")
            traceback.print_exc()
            break

        if mat_mpnn_seqs is None or mat_mpnn_seqs.get('seq') is None or len(mat_mpnn_seqs['seq']) == 0:
            print(f"  No MPNN sequences generated at round {mat_round}, stopping")
            break

        mat_seq_list = [
            {'seq': mat_mpnn_seqs['seq'][n][-length:],
             'score': mat_mpnn_seqs['score'][n]}
            for n in range(len(mat_mpnn_seqs['seq']))
        ]
        mat_best_seq = min(mat_seq_list, key=lambda x: x['score'])
        mat_mpnn_name = f"{mat_design_name}_mpnn1"

        # Create prediction model AFTER MPNN — mpnn_gen_sequence_maturation() calls clear_mem()
        complex_prediction_model = mk_afdesign_model(
            protocol="binder",
            num_recycles=advanced_settings["num_recycles_validation"],
            data_dir=advanced_settings["af_params_dir"],
            use_multimer=multimer_validation,
            use_initial_guess=advanced_settings["predict_initial_guess"],
            use_initial_atom_pos=advanced_settings["predict_bigbang"])
        if advanced_settings["predict_initial_guess"] or advanced_settings["predict_bigbang"]:
            complex_prediction_model.prep_inputs(
                pdb_filename=trajectory_pdb, chain='A', binder_chain='B',
                binder_len=length, use_binder_template=True,
                rm_target_seq=advanced_settings["rm_template_seq_predict"],
                rm_target_sc=advanced_settings["rm_template_sc_predict"],
                rm_template_ic=True)
        else:
            complex_prediction_model.prep_inputs(
                pdb_filename=target_settings["starting_pdb"],
                chain=target_settings["chains"], binder_len=length,
                rm_target_seq=advanced_settings["rm_template_seq_predict"],
                rm_target_sc=advanced_settings["rm_template_sc_predict"])

        mat_pred_stats, mat_pass_af2, _ = predict_binder_complex(
            complex_prediction_model, mat_best_seq['seq'], mat_mpnn_name,
            target_settings["starting_pdb"], target_settings["chains"],
            length, mat_traj_pdb, prediction_models, advanced_settings,
            filters, design_paths, failure_csv, use_pyrosetta=use_pyrosetta)

        if not mat_pass_af2:
            print(f"  Maturation round {mat_round}: AF2 prediction failed, stopping")
            break

        for mat_model_num in prediction_models:
            mat_mpnn_pdb = os.path.join(design_paths["MPNN"],
                                        f"{mat_mpnn_name}_model{mat_model_num+1}.pdb")
            mat_relaxed_pdb = os.path.join(design_paths["MPNN/Maturation"],
                                           f"{mat_mpnn_name}_model{mat_model_num+1}.pdb")
            if os.path.exists(mat_mpnn_pdb):
                try:
                    pr_relax(mat_mpnn_pdb, mat_relaxed_pdb, use_pyrosetta=use_pyrosetta)
                except Exception as e:
                    print(f"  Maturation relax failed for model {mat_model_num+1}: {e}")
                    traceback.print_exc()
                    shutil.copy(mat_mpnn_pdb, mat_relaxed_pdb)

        mat_averages = calculate_averages(mat_pred_stats, handle_aa=False)
        mat_new_metric = get_maturation_metric(mat_averages, mat_metric_name)

        mat_new_ipsae = mat_averages.get('ipSAE', None)
        log_maturation_round(mat_round, mat_fixed_set, new_redesign,
                             residue_quality, mat_metric_name,
                             mat_best_metric, mat_new_metric,
                             ipsae=mat_new_ipsae)

        metric_improved = mat_new_metric > mat_best_metric + mat_improvement_thresh
        if mat_new_ipsae is not None and mat_best_ipsae is not None:
            ipsae_improved = mat_new_ipsae > mat_best_ipsae + mat_ipsae_thresh
        else:
            ipsae_improved = True

        if metric_improved and ipsae_improved:
            mat_best_metric = mat_new_metric
            mat_best_ipsae = mat_new_ipsae
            mat_current_seq = mat_best_seq['seq']
            mat_model_plddt = {m: mat_pred_stats[m].get('pLDDT', 0)
                               for m in mat_pred_stats if not str(m).startswith('_')}
            mat_best_model_num = max(mat_model_plddt, key=mat_model_plddt.get) if mat_model_plddt else best_model_number
            mat_best_model_stats = mat_pred_stats.get(mat_best_model_num, {})
            if mat_best_model_stats.get('_pae_matrix') is not None:
                mat_pae = mat_best_model_stats['_pae_matrix']
                mat_plddt = mat_best_model_stats['_plddt_array']
            mat_relaxed_best = os.path.join(design_paths["MPNN/Maturation"],
                                             f"{mat_mpnn_name}_model{mat_best_model_num}.pdb")
            if os.path.exists(mat_relaxed_best):
                mat_current_pdb = mat_relaxed_best
                best_model_pdb = mat_relaxed_best
            mat_rounds_completed = mat_round
            mat_best_averages = mat_averages
        else:
            reasons = []
            if not metric_improved:
                reasons.append(f"{mat_metric_name} {mat_new_metric - mat_best_metric:+.4f} < {mat_improvement_thresh}")
            if not ipsae_improved:
                delta = (mat_new_ipsae or 0) - (mat_best_ipsae or 0)
                reasons.append(f"ipSAE {delta:+.4f} < {mat_ipsae_thresh}")
            print(f"  No improvement at round {mat_round} ({'; '.join(reasons)}), stopping maturation")
            break

    # Update maturation columns in mpnn_data
    mat_post_iptm = mpnn_complex_averages.get('i_pTM', None) if mat_rounds_completed == 0 else mat_best_averages.get('i_pTM', None)
    mat_post_ipsae = mpnn_complex_averages.get('ipSAE', None) if mat_rounds_completed == 0 else mat_best_averages.get('ipSAE', None)
    fixed_res_str = ','.join(str(i) for i in sorted(mat_fixed_set))
    redesign_res_str = ','.join(str(i) for i in sorted(new_redesign)) if not mat_converged else ''

    mcs = maturation_col_start(mpnn_data)
    mpnn_data[mcs + MAT_ROUNDS] = mat_rounds_completed
    mpnn_data[mcs + MAT_CONVERGED] = mat_converged
    mpnn_data[mcs + MAT_FIXED_RES] = fixed_res_str
    mpnn_data[mcs + MAT_REDESIGN_RES] = redesign_res_str
    mpnn_data[mcs + MAT_PRE_IPTM] = mat_pre_metric_iptm
    mpnn_data[mcs + MAT_POST_IPTM] = mat_post_iptm
    mpnn_data[mcs + MAT_PRE_IPSAE] = mat_pre_metric_ipsae
    mpnn_data[mcs + MAT_POST_IPSAE] = mat_post_ipsae
    scan_mut_str = ' '.join(
        f"{old}{idx+1}{new}({new_reu-old_reu:.1f})"
        for idx, old, new, old_reu, new_reu in mat_all_scan_mutations
    ) if mat_all_scan_mutations else ''
    mpnn_data[mcs + MAT_SCAN_MUTATIONS] = scan_mut_str

    mat_ran = mat_rounds_completed > 0
    if mat_ran:
        mpnn_data[6] = mat_current_seq
        ipsae_summary = f", ipSAE {mat_pre_metric_ipsae:.4f} -> {mat_post_ipsae:.4f}" if mat_pre_metric_ipsae is not None and mat_post_ipsae is not None else ""
        iptm_summary = f"{mat_pre_metric_iptm:.4f} -> {mat_post_iptm:.4f}" if mat_pre_metric_iptm is not None and mat_post_iptm is not None else "N/A"
        print(f"--- {mat_label.capitalize()} maturation complete: {mat_rounds_completed} rounds, "
              f"{mat_metric_name} {iptm_summary}{ipsae_summary} ---\n")

    return mat_ran, mpnn_data, best_model_pdb


# ------------------------------------------------------------------
# Run maturation
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("  MATURATION-ONLY RUN")
print("=" * 60)

mat_start = time.time()
mat_ran, updated_data, final_pdb = _run_maturation(cand, ctx, advanced_settings, mat_label="standalone")

elapsed = time.time() - mat_start
print(f"\n{'=' * 60}")
print(f"  MATURATION RESULT")
print(f"{'=' * 60}")
print(f"  Maturation ran:  {mat_ran}")
print(f"  Final PDB:       {final_pdb}")
print(f"  Elapsed time:    {elapsed:.1f}s")

if mat_ran:
    # Revert check
    mat_revert_on_worse = advanced_settings.get("maturation_revert_on_worse", True)
    mcs = maturation_col_start(updated_data)
    pre_iptm = updated_data[mcs + MAT_PRE_IPTM]
    post_iptm = updated_data[mcs + MAT_POST_IPTM]
    pre_ipsae = updated_data[mcs + MAT_PRE_IPSAE]
    post_ipsae = updated_data[mcs + MAT_POST_IPSAE]

    print(f"  i_pTM:   {pre_iptm} -> {post_iptm}")
    print(f"  ipSAE:   {pre_ipsae} -> {post_ipsae}")
    print(f"  Rounds:  {updated_data[mcs + MAT_ROUNDS]}")
    print(f"  Fixed:   {updated_data[mcs + MAT_FIXED_RES]}")
    print(f"  Redesign:{updated_data[mcs + MAT_REDESIGN_RES]}")
    if updated_data[mcs + MAT_SCAN_MUTATIONS]:
        print(f"  Scan mutations: {updated_data[mcs + MAT_SCAN_MUTATIONS]}")

    if mat_revert_on_worse:
        should_revert = False
        if post_iptm is not None and pre_iptm is not None and post_iptm < pre_iptm:
            print(f"\n  WARNING: Maturation WORSENED i_pTM ({pre_iptm:.4f} -> {post_iptm:.4f})")
            should_revert = True
        if post_ipsae is not None and pre_ipsae is not None and post_ipsae < pre_ipsae:
            print(f"\n  WARNING: Maturation WORSENED ipSAE ({pre_ipsae:.4f} -> {post_ipsae:.4f})")
            should_revert = True
        if should_revert:
            print("  In full pipeline this would trigger revert to pre-maturation state.")
else:
    print("  Maturation did not complete any rounds.")

print(f"\nDone.")
