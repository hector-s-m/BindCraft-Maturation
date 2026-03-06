####################################
###################### BindCraft Run
####################################
### Import dependencies
import gc
from functions import *
from functions.generic_utils import insert_data, maturation_col_start, MAT_ROUNDS, MAT_CONVERGED, MAT_FIXED_RES, MAT_REDESIGN_RES, MAT_PRE_IPTM, MAT_POST_IPTM, MAT_PRE_IPSAE, MAT_POST_IPSAE, MAT_SCAN_MUTATIONS
from functions.biopython_utils import clear_dssp_cache # Explicit import for DSSP cache management
import logging
import os
import sys
import subprocess
import traceback
try:
    import resource  # POSIX-only; used to raise RLIMIT_NOFILE (ulimit -n)
except Exception:
    resource = None

def _bump_open_files_limit(min_soft=65536):
    """Attempt to raise the soft RLIMIT_NOFILE up to min_soft (not above hard)."""
    if resource is None:
        print("Warning: 'resource' module not available; cannot adjust open files limit.")
        return
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        desired_soft = min(max(soft, int(min_soft)), hard if hard != resource.RLIM_INFINITY else max(soft, int(min_soft)))
        if desired_soft > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (desired_soft, hard))
            print(f"Adjusted open files soft limit: {soft} -> {desired_soft} (hard={hard})")
        else:
            print(f"Open files limits OK (soft={soft}, hard={hard})")
    except Exception as e:
        print(f"Warning: Unable to adjust open files limit: {e}")

# Raise file descriptor soft limit early to avoid 'Too many open files'
_bump_open_files_limit(min_soft=65536)

# Defer GPU availability check until after CLI/interactive handling

def ensure_binaries_executable(use_pyrosetta=True):
    """Ensure all required binaries in functions/ are executable."""
    bindcraft_folder = os.path.dirname(os.path.realpath(__file__))
    functions_dir = os.path.join(bindcraft_folder, "functions")
    
    # Always needed binaries
    binaries = ["dssp", "sc", "FASPR"]
    
    # Only add DAlphaBall.gcc if using PyRosetta
    if use_pyrosetta:
        binaries.append("DAlphaBall.gcc")
    
    for binary in binaries:
        binary_path = os.path.join(functions_dir, binary)
        if os.path.isfile(binary_path):
            try:
                # Check if already executable
                if not os.access(binary_path, os.X_OK):
                    # Make executable
                    current_mode = os.stat(binary_path).st_mode
                    os.chmod(binary_path, current_mode | 0o755)
                    print(f"Made {binary} executable")
            except Exception as e:
                print(f"Warning: Failed to make {binary} executable: {e}")

# Ensure binaries are executable at startup (will be called again with proper use_pyrosetta flag later)
ensure_binaries_executable()

######################################
### parse input paths and interactive mode
parser = argparse.ArgumentParser(description='Script to run BindCraft binder design.')

parser.add_argument('--settings', '-s', type=str, required=False,
                    help='Path to the basic settings.json file. If omitted in a TTY, interactive mode is used.')
parser.add_argument('--filters', '-f', type=str, default='./settings_filters/default_filters.json',
                    help='Path to the filters.json file used to filter design. If not provided, default will be used.')
parser.add_argument('--advanced', '-a', type=str, default='./settings_advanced/default_4stage_multimer.json',
                    help='Path to the advanced.json file with additional design settings. If not provided, default will be used.')
parser.add_argument('--no-pyrosetta', action='store_true',
                    help='Run without PyRosetta (skips relaxation and PyRosetta-based scoring)')
parser.add_argument('--verbose', action='store_true',
                    help='Enable detailed timing/progress logs')
parser.add_argument('--debug-pdbs', action='store_true',
                    help='Write intermediate debug PDBs during OpenMM relax (deconcat, PDBFixer, post-initial-relax, post-FASPR)')
parser.add_argument('--no-plots', action='store_true',
                    help='Disable saving design trajectory plots (overrides advanced settings)')
parser.add_argument('--no-animations', action='store_true',
                    help='Disable saving design animations (overrides advanced settings)')
parser.add_argument('--interactive', action='store_true',
                    help='Force interactive mode to collect target settings and options')
parser.add_argument('--rank-by', type=str, default='i_pTM',
                    choices=['i_pTM', 'ipSAE'],
                    help='Metric to rank final designs by (default: i_pTM)')

args = parser.parse_args()

def _isatty_stdin():
    try:
        return sys.stdin.isatty()
    except Exception:
        return False

def _input_with_default(prompt_text, default_value=None):
    if default_value is None:
        return input(prompt_text).strip()
    resp = input(f"{prompt_text} ").strip()
    return resp if resp else default_value

def _yes_no(prompt_text, default_yes=False):
    default_hint = 'Y/n' if default_yes else 'y/N'
    resp = input(f"{prompt_text} ({default_hint}): ").strip().lower()
    if resp == '':
        return default_yes
    return resp in ('y', 'yes')

def _list_json_choices(folder_path):
    try:
        entries = [f for f in os.listdir(folder_path) if f.endswith('.json') and not f.startswith('.')]
    except Exception:
        entries = []
    entries.sort()
    # return list of (display_name_without_ext, abspath)
    return [(os.path.splitext(f)[0], os.path.join(folder_path, f)) for f in entries]

def _prompt_interactive_and_prepare_args(args):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    filters_dir = os.path.join(base_dir, 'settings_filters')
    advanced_dir = os.path.join(base_dir, 'settings_advanced')

    print("\nBindCraft Interactive Setup\n")

    while True:
        # Design type selection
        print("Design type:")
        print("1. Miniprotein (31+ aa)")
        print("2. Peptide (8-30 aa)")
        dtype_choice = _input_with_default("Choose design type (press Enter for Miniprotein):", "")
        is_peptide = (dtype_choice.strip() == '2')

        # Required inputs
        project_name = _input_with_default("Enter project/binder name:", None)
        while not project_name:
            print("Project name is required.")
            project_name = _input_with_default("Enter project/binder name:", None)

        # Require a valid existing PDB file path; re-prompt until valid
        while True:
            pdb_raw = _input_with_default("Enter path to PDB file:", None)
            if not pdb_raw:
                print("PDB path is required.")
                continue
            candidate = os.path.abspath(os.path.expanduser(pdb_raw))
            if os.path.isfile(candidate):
                pdb_path = candidate
                break
            print(f"Error: No PDB file found at '{candidate}'. Please re-enter.")

        output_dir = _input_with_default("Enter output directory:", os.path.join(os.getcwd(), f"{project_name}_bindcraft_out"))
        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        os.makedirs(output_dir, exist_ok=True)

        chains = _input_with_default("Enter target chains (e.g., A or A,B):", "A")
        hotspots = _input_with_default("Enter hotspot residue(s) for BindCraft to target. Use format: chain letter + residue numbers (e.g., 'A1,B20-25'). Leave empty for no preference:", "")

        if is_peptide:
            lengths_prompt_default = "8 25"
            lengths_input = _input_with_default("Enter peptide min and max lengths (8-30) separated by space or comma (default 8 25):", lengths_prompt_default)
        else:
            lengths_prompt_default = "65 150"
            lengths_input = _input_with_default("Enter miniprotein min and max lengths separated by space or comma (min>=31, default 65 150):", lengths_prompt_default)
        try:
            normalized = lengths_input.replace(',', ' ').split()
            min_len_val, max_len_val = int(normalized[0]), int(normalized[1])
        except Exception:
            if is_peptide:
                min_len_val, max_len_val = 8, 25
            else:
                min_len_val, max_len_val = 65, 150
        if is_peptide:
            # Clamp within [8,30]
            min_len_val = max(8, min(min_len_val, 30))
            max_len_val = max(8, min(max_len_val, 30))
            if min_len_val > max_len_val:
                min_len_val, max_len_val = max_len_val, min_len_val
        else:
            # Enforce min >= 31; ensure order
            if min_len_val < 31:
                min_len_val = 31
            if max_len_val < min_len_val:
                max_len_val = min_len_val
        lengths = [min_len_val, max_len_val]

        num_designs_str = _input_with_default("Enter number of final designs (default 100):", "100")
        try:
            num_designs = int(num_designs_str)
        except Exception:
            num_designs = 100

        # List choices
        print("\nAvailable filter settings:")
        filter_choices_all = _list_json_choices(filters_dir)
        name_to_filter = {name: path for name, path in filter_choices_all}
        if is_peptide:
            filter_order = ['peptide_filters', 'peptide_relaxed_filters', 'no_filters']
            default_filter_name = 'peptide_filters'
        else:
            filter_order = ['default_filters', 'relaxed_filters', 'no_filters']
            default_filter_name = 'default_filters'
        ordered_filters = [(name, name_to_filter[name]) for name in filter_order if name in name_to_filter]
        for i, (name, _) in enumerate(ordered_filters, 1):
            print(f"{i}. {name}")
        filter_idx = _input_with_default(f"Choose filter (press Enter for {default_filter_name}):", "")
        if filter_idx:
            try:
                filter_idx_int = int(filter_idx)
                selected_filter = ordered_filters[filter_idx_int - 1][1]
            except Exception:
                selected_filter = name_to_filter.get(default_filter_name, os.path.join(filters_dir, f"{default_filter_name}.json"))
        else:
            selected_filter = name_to_filter.get(default_filter_name, os.path.join(filters_dir, f"{default_filter_name}.json"))

        print("\nAvailable advanced settings:")
        advanced_choices_all = _list_json_choices(advanced_dir)
        name_to_adv = {name: path for name, path in advanced_choices_all}
        if is_peptide:
            adv_order = [
                'peptide_3stage_multimer',
                'peptide_3stage_multimer_mpnn',
                'peptide_3stage_multimer_flexible',
                'peptide_3stage_multimer_mpnn_flexible'
            ]
            default_adv_name = 'peptide_3stage_multimer'
        else:
            adv_order = [
                'default_4stage_multimer',
                'default_4stage_multimer_mpnn',
                'default_4stage_multimer_flexible',
                'default_4stage_multimer_hardtarget',
                'default_4stage_multimer_flexible_hardtarget',
                'default_4stage_multimer_mpnn_flexible',
                'default_4stage_multimer_mpnn_hardtarget',
                'default_4stage_multimer_mpnn_flexible_hardtarget',
                'betasheet_4stage_multimer',
                'betasheet_4stage_multimer_mpnn',
                'betasheet_4stage_multimer_flexible',
                'betasheet_4stage_multimer_hardtarget',
                'betasheet_4stage_multimer_flexible_hardtarget',
                'betasheet_4stage_multimer_mpnn_flexible',
                'betasheet_4stage_multimer_mpnn_hardtarget',
                'betasheet_4stage_multimer_mpnn_flexible_hardtarget'
            ]
            default_adv_name = 'default_4stage_multimer'
        ordered_adv = [(name, name_to_adv[name]) for name in adv_order if name in name_to_adv]
        for i, (name, _) in enumerate(ordered_adv, 1):
            print(f"{i}. {name}")
        advanced_idx = _input_with_default(f"Choose advanced (press Enter for {default_adv_name}):", "")
        if advanced_idx:
            try:
                advanced_idx_int = int(advanced_idx)
                selected_advanced = ordered_adv[advanced_idx_int - 1][1]
            except Exception:
                selected_advanced = name_to_adv.get(default_adv_name, os.path.join(advanced_dir, f"{default_adv_name}.json"))
        else:
            selected_advanced = name_to_adv.get(default_adv_name, os.path.join(advanced_dir, f"{default_adv_name}.json"))

        # Toggles
        verbose = _yes_no("Enable verbose output?", default_yes=False)
        plots_on = _yes_no("Enable saving plots?", default_yes=True)
        animations_on = _yes_no("Enable saving animations?", default_yes=True)
        run_with_pyrosetta = _yes_no("Run with PyRosetta?", default_yes=True)
        
        # Only ask about debug PDbs if not using PyRosetta (since debug PDbs are for OpenMM relax)
        debug_pdbs = False
        if not run_with_pyrosetta:
            debug_pdbs = _yes_no("Write intermediate debug PDBs during OpenMM relax?", default_yes=False)

        # Ranking method selection
        print("\nRanking method for final designs:")
        print("1. i_pTM (interface predicted TM-score)")
        print("2. ipSAE (interface predicted Structural Alignment Error)")
        rank_choice = _input_with_default("Choose ranking method (press Enter for i_pTM):", "")
        if rank_choice.strip() == '2':
            rank_by_metric = 'ipSAE'
        else:
            rank_by_metric = 'i_pTM'

        # Summary for confirmation
        print("\nConfiguration Summary:")
        print(f"Project Name: {project_name}")
        print(f"PDB File: {pdb_path}")
        print(f"Output Directory: {output_dir}")
        print(f"Chains: {chains}")
        print(f"Hotspots: {hotspots if hotspots else 'None'}")
        print(f"Length Range: {lengths}")
        print(f"Design Type: {'Peptide' if is_peptide else 'Miniprotein'}")
        print(f"Number of Final Designs: {num_designs}")
        print(f"Filter Setting: {os.path.splitext(os.path.basename(selected_filter))[0]}")
        print(f"Advanced Setting: {os.path.splitext(os.path.basename(selected_advanced))[0]}")
        print(f"Verbose: {'Yes' if verbose else 'No'}")
        if not run_with_pyrosetta:
            print(f"Debug PDbs: {'Yes' if debug_pdbs else 'No'}")
        print(f"Plots: {'On' if plots_on else 'Off'}")
        print(f"Animations: {'On' if animations_on else 'Off'}")
        print(f"PyRosetta: {'On' if run_with_pyrosetta else 'Off'}")
        print(f"Ranking Method: {rank_by_metric}")

        if _yes_no("Proceed with these settings?", default_yes=True):
            break
        else:
            print("Let's re-enter the details.\n")

    # Prepare target settings JSON
    target_settings = {
        "design_path": output_dir,
        "binder_name": project_name,
        "starting_pdb": pdb_path,
        "chains": chains,
        "target_hotspot_residues": hotspots,
        "lengths": lengths,
        "number_of_final_designs": num_designs
    }

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    settings_filename = f"interactive_{project_name}_{timestamp}.json"
    settings_path_out = os.path.join(output_dir, settings_filename)
    try:
        with open(settings_path_out, 'w') as f:
            json.dump(target_settings, f, indent=2)
    except Exception as e:
        print(f"Error writing settings JSON: {e}")
        sys.exit(1)

    # Map to args
    args.settings = settings_path_out
    args.filters = selected_filter
    args.advanced = selected_advanced
    args.verbose = verbose
    args.debug_pdbs = debug_pdbs
    args.no_plots = (not plots_on)
    args.no_animations = (not animations_on)
    args.no_pyrosetta = (not run_with_pyrosetta)
    args.rank_by = rank_by_metric

    return args

# Enter interactive mode if requested or if no settings were provided in a TTY
if args.interactive or (not args.settings and _isatty_stdin()):
    args = _prompt_interactive_and_prepare_args(args)
elif not args.settings and not _isatty_stdin():
    # No TTY and no settings -> cannot prompt
    print("Error: --settings is required in non-interactive environments.")
    sys.exit(1)

# perform checks of input setting files
settings_path, filters_path, advanced_path = perform_input_check(args)

# Configure standard logging based on --verbose
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s'
)

# Reduce noise from third-party libraries regardless of verbosity
for noisy_logger in (
    "jax", "jaxlib", "jax._src", "absl", "flax", "colabdesign", "tensorflow", "xla"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# Enable detailed logs only for our modules when --verbose is set
if args.verbose:
    logging.getLogger("functions").setLevel(logging.DEBUG)
    logging.getLogger("bindcraft").setLevel(logging.DEBUG)

# Check if JAX-capable GPU is available, otherwise exit (after interactive/container handling)
check_jax_gpu()

### load settings from JSON
target_settings, advanced_settings, filters = load_json_settings(settings_path, filters_path, advanced_path)

settings_file = os.path.basename(settings_path).split('.')[0]
filters_file = os.path.basename(filters_path).split('.')[0]
advanced_file = os.path.basename(advanced_path).split('.')[0]

# Provide context to OpenMM relax for de-concatenation/re-concatenation and debug PDBs
try:
    os.environ['BINDCRAFT_STARTING_PDB'] = os.path.abspath(os.path.expanduser(target_settings["starting_pdb"]))
    os.environ['BINDCRAFT_TARGET_CHAINS'] = str(target_settings.get("chains", "A"))
    os.environ['BINDCRAFT_DEBUG_PDBS'] = '1' if args.debug_pdbs else '0'
except Exception:
    pass

### load AF2 model settings
design_models, prediction_models, multimer_validation = load_af2_models(advanced_settings["use_multimer_design"])

### perform checks on advanced_settings
bindcraft_folder = os.path.dirname(os.path.realpath(__file__))
advanced_settings = perform_advanced_settings_check(advanced_settings, bindcraft_folder)

# CLI overrides for plots/animations
if args.no_plots:
    advanced_settings["save_design_trajectory_plots"] = False
if args.no_animations:
    advanced_settings["save_design_animations"] = False

### generate directories, design path names can be found within the function
design_paths = generate_directories(target_settings["design_path"])

### generate dataframes
trajectory_labels, design_labels, final_labels = generate_dataframe_labels()

trajectory_csv = os.path.join(target_settings["design_path"], 'trajectory_stats.csv')
mpnn_csv = os.path.join(target_settings["design_path"], 'mpnn_design_stats.csv')
final_csv = os.path.join(target_settings["design_path"], 'final_design_stats.csv')
failure_csv = os.path.join(target_settings["design_path"], 'failure_csv.csv')

# Migrate existing CSVs to include new columns (backwards compatibility for resumed jobs)
from functions.generic_utils import migrate_csv_columns
migrate_csv_columns(trajectory_csv, trajectory_labels)
migrate_csv_columns(mpnn_csv, design_labels)
migrate_csv_columns(final_csv, final_labels)

create_dataframe(trajectory_csv, trajectory_labels)
create_dataframe(mpnn_csv, design_labels)
create_dataframe(final_csv, final_labels)
generate_filter_pass_csv(failure_csv, args.filters)

### Define and initialize rejected_mpnn_full_stats.csv
# Ensure failure_csv exists and has headers to read its column structure
if not os.path.exists(failure_csv):
    # This should ideally not happen if generate_filter_pass_csv worked, create an empty one if it's missing.
    temp_failure_df_for_cols = pd.DataFrame()
    print(f"Warning: {failure_csv} was not found after generate_filter_pass_csv. rejected_mpnn_full_stats.csv might have incorrect filter columns.")
else:
    try:
        temp_failure_df_for_cols = pd.read_csv(failure_csv)
    except pd.errors.EmptyDataError:
        # If failure_csv is empty we need to get column names from how generate_filter_pass_csv would create them.
        print(f"Warning: {failure_csv} is empty. rejected_mpnn_full_stats.csv may lack detailed filter columns initially if no filters are defined or an issue occurred.")
        temp_failure_df_for_cols = pd.DataFrame() # Fallback- we don't want BindCraft to crash over this

filter_column_names_for_rejected_log = temp_failure_df_for_cols.columns.tolist()
del temp_failure_df_for_cols # Free memory

rejected_stats_columns = ['Design', 'Sequence'] + filter_column_names_for_rejected_log
rejected_mpnn_full_stats_csv = os.path.join(target_settings["design_path"], 'rejected_mpnn_full_stats.csv')
create_dataframe(rejected_mpnn_full_stats_csv, rejected_stats_columns)
####################################
####################################
####################################
### initialise PyRosetta if not disabled
use_pyrosetta = False

if args.no_pyrosetta:
    # Quiet when user explicitly disables PyRosetta
    print("Running in PyRosetta-free mode as requested by --no-pyrosetta flag.")
else:
    if 'PYROSETTA_AVAILABLE' in globals() and PYROSETTA_AVAILABLE and pr is not None:
        try:
            pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {advanced_settings["dalphaball_path"]} -corrections::beta_nov16 true -relax:default_repeats 1')
            print("PyRosetta initialized successfully.")
            use_pyrosetta = True
        except Exception as e:
            print(f"PyRosetta detected but failed to initialize: {e}")
            print("Falling back to OpenMM and Biopython routines.")
    else:
        print("PyRosetta not found. Using OpenMM and Biopython routines.")

# Ensure binaries are executable with correct PyRosetta mode
ensure_binaries_executable(use_pyrosetta=use_pyrosetta)

print(f"Running binder design for target {settings_file}")
print(f"Design settings used: {advanced_file}")
print(f"Filtering designs based on {filters_file}")

####################################
# initialise counters
script_start_time = time.time()
trajectory_n = 1
accepted_designs = 0

### Helper functions for Phase 3 processing

def _run_maturation(cand, ctx, advanced_settings, mat_label="maturation"):
    """
    Run the maturation loop on a single candidate.
    ctx is a dict with keys: complex_prediction_model, design_models, prediction_models,
        target_settings, length, seed, helicity_value, binder_chain, design_paths,
        failure_csv, mpnn_csv, use_pyrosetta, filters (for AF2 prediction).
    Returns (mat_ran, updated_mpnn_data, updated_best_model_pdb).
    mat_ran is True if at least one round completed with improvement.
    """
    mpnn_data = list(cand['mpnn_data'])  # Copy to avoid mutating caller's data
    mpnn_design_name = cand['design_name']
    best_model_pdb = cand['best_model_pdb']
    best_model_number = cand['best_model_number']
    mpnn_complex_averages = cand['complex_averages']
    mpnn_complex_statistics = cand['complex_statistics']

    # Unpack context
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

    mat_max_rounds = advanced_settings.get("maturation_max_rounds", 3)
    mat_metric_name = advanced_settings.get("maturation_improvement_metric", "i_pTM")
    mat_improvement_thresh = advanced_settings.get("maturation_improvement_threshold", 0.01)
    mat_ipsae_thresh = advanced_settings.get("maturation_ipsae_improvement_threshold", 0.01)

    # Get cached PAE/pLDDT from the best model's prediction
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
    mat_best_averages = None  # Set when a round improves

    for mat_round in range(1, mat_max_rounds + 1):
        # Step 0: Compute per-residue REU (primary quality metric)
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
        mat_pred_stats, mat_pass_af2, _ = predict_binder_complex(
            complex_prediction_model, mat_best_seq['seq'], mat_mpnn_name,
            target_settings["starting_pdb"], target_settings["chains"],
            length, mat_traj_pdb, prediction_models, advanced_settings,
            filters, design_paths, failure_csv, use_pyrosetta=use_pyrosetta)

        if not mat_pass_af2:
            print(f"  Maturation round {mat_round}: AF2 prediction failed, stopping")
            break

        for mat_model_num in prediction_models:
            mat_mpnn_pdb = os.path.join(design_paths["MPNN"], f"{mat_mpnn_name}_model{mat_model_num+1}.pdb")
            mat_relaxed_pdb = os.path.join(design_paths["MPNN/Maturation"], f"{mat_mpnn_name}_model{mat_model_num+1}.pdb")
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
            mat_model_plddt = {m: mat_pred_stats[m].get('pLDDT', 0) for m in mat_pred_stats if not str(m).startswith('_')}
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


def _check_maturation_revert(mat_ran, mat_revert_on_worse, cand_mpnn_data,
                             cand_pre_mat_data, cand_pre_mat_pdb, cand_best_model_pdb):
    """Check if maturation worsened i_pTM or ipSAE; revert if so.
    Returns (reverted, mpnn_data, best_model_pdb, mat_ran)."""
    if not mat_ran or not mat_revert_on_worse:
        return False, cand_mpnn_data, cand_best_model_pdb, mat_ran

    mcs = maturation_col_start(cand_mpnn_data)
    mat_pre_iptm = cand_mpnn_data[mcs + MAT_PRE_IPTM]
    mat_post_iptm = cand_mpnn_data[mcs + MAT_POST_IPTM]
    mat_pre_ipsae = cand_mpnn_data[mcs + MAT_PRE_IPSAE]
    mat_post_ipsae = cand_mpnn_data[mcs + MAT_POST_IPSAE]

    reason = None
    if mat_post_iptm is not None and mat_pre_iptm is not None and mat_post_iptm < mat_pre_iptm:
        reason = f"i_pTM ({mat_pre_iptm:.4f} -> {mat_post_iptm:.4f})"
    elif mat_post_ipsae is not None and mat_pre_ipsae is not None and mat_post_ipsae < mat_pre_ipsae:
        reason = f"ipSAE ({mat_pre_ipsae:.4f} -> {mat_post_ipsae:.4f})"

    if reason:
        print(f"  Maturation worsened {reason}, reverting")
        return True, cand_pre_mat_data, cand_pre_mat_pdb, False

    return False, cand_mpnn_data, cand_best_model_pdb, mat_ran


def _accept_design(best_model_pdb, mpnn_data, design_name,
                   design_paths, final_csv, advanced_settings):
    """Copy accepted design to Accepted folder, write to final CSV, copy animations/plots."""
    shutil.copy(best_model_pdb, design_paths["Accepted"])

    final_data = [''] + mpnn_data
    insert_data(final_csv, final_data)

    if advanced_settings["save_design_animations"]:
        accepted_animation = os.path.join(design_paths["Accepted/Animation"], f"{design_name}.html")
        if not os.path.exists(accepted_animation):
            src_animation = os.path.join(design_paths["Trajectory/Animation"], f"{design_name}.html")
            if os.path.exists(src_animation):
                shutil.copy(src_animation, accepted_animation)

    plot_files = os.listdir(design_paths["Trajectory/Plots"])
    plots_to_copy = [f for f in plot_files if f.startswith(design_name) and f.endswith('.png')]
    for accepted_plot in plots_to_copy:
        source_plot = os.path.join(design_paths["Trajectory/Plots"], accepted_plot)
        target_plot = os.path.join(design_paths["Accepted/Plots"], accepted_plot)
        if not os.path.exists(target_plot):
            shutil.copy(source_plot, target_plot)


def _reject_design(design_name, best_model_pdb, mpnn_data,
                   filter_conditions, failure_csv, design_paths,
                   filter_column_names_for_rejected_log,
                   rejected_mpnn_full_stats_csv, sequence):
    """Log filter failures, update failure CSV, copy to Rejected folder."""
    print(f"Unmet filter conditions for {design_name}")
    failure_df = pd.read_csv(failure_csv)
    special_prefixes = ('Average_', '1_', '2_', '3_', '4_', '5_')
    incremented_columns = set()

    for column in filter_conditions:
        base_column = column
        for prefix in special_prefixes:
            if column.startswith(prefix):
                base_column = column.split('_', 1)[1]
                break

        if base_column not in incremented_columns:
            if base_column in failure_df.columns:
                failure_df[base_column] = failure_df[base_column] + 1
            else:
                print(f"Warning: Base column '{base_column}' not found in {failure_csv}.")
            incremented_columns.add(base_column)

    failure_df.to_csv(failure_csv, index=False)

    rejected_data_list_for_csv = [design_name, sequence]
    for filter_col_name in filter_column_names_for_rejected_log:
        if filter_col_name in incremented_columns:
            rejected_data_list_for_csv.append(1)
        else:
            rejected_data_list_for_csv.append(0)
    insert_data(rejected_mpnn_full_stats_csv, rejected_data_list_for_csv)

    shutil.copy(best_model_pdb, design_paths["Rejected"])


### start design loop
while True:
    ### check if we have the target number of binders
    # Map CLI metric name to CSV column name (e.g., 'i_pTM' -> 'Average_i_pTM')
    rank_by_column = f"Average_{args.rank_by}"
    final_designs_reached = check_accepted_designs(design_paths, mpnn_csv, final_labels, final_csv, advanced_settings, target_settings, design_labels, rank_by=rank_by_column)

    if final_designs_reached:
        # stop design loop execution
        break

    ### check if we reached maximum allowed trajectories
    max_trajectories_reached = check_n_trajectories(design_paths, advanced_settings)

    if max_trajectories_reached:
        break

    ### Initialise design
    # measure time to generate design
    trajectory_start_time = time.time()

    # generate random seed to vary designs
    seed = int(np.random.randint(0, high=999999, size=1, dtype=int)[0])

    # sample binder design length randomly from defined distribution
    samples = np.arange(min(target_settings["lengths"]), max(target_settings["lengths"]) + 1)
    length = np.random.choice(samples)

    # load desired helicity value to sample different secondary structure contents
    helicity_value = load_helicity(advanced_settings)

    # generate design name and check if same trajectory was already run
    design_name = target_settings["binder_name"] + "_l" + str(length) + "_s"+ str(seed)
    trajectory_dirs = ["Trajectory", "Trajectory/Relaxed", "Trajectory/LowConfidence", "Trajectory/Clashing"]
    trajectory_exists = any(os.path.exists(os.path.join(design_paths[trajectory_dir], design_name + ".pdb")) for trajectory_dir in trajectory_dirs)

    if not trajectory_exists:
        print("Starting trajectory: "+design_name)

        ### Begin binder hallucination
        trajectory = binder_hallucination(design_name, target_settings["starting_pdb"], target_settings["chains"],
                                            target_settings["target_hotspot_residues"], length, seed, helicity_value,
                                            design_models, advanced_settings, design_paths, failure_csv)
        trajectory_metrics = copy_dict(trajectory._tmp["best"]["aux"]["log"]) # contains plddt, ptm, i_ptm, pae, i_pae
        trajectory_pdb = os.path.join(design_paths["Trajectory"], design_name + ".pdb")

        # round the metrics to two decimal places
        trajectory_metrics = {k: round(v, 2) if isinstance(v, float) else v for k, v in trajectory_metrics.items()}

        # time trajectory
        trajectory_time = time.time() - trajectory_start_time
        trajectory_time_text = f"{'%d hours, %d minutes, %d seconds' % (int(trajectory_time // 3600), int((trajectory_time % 3600) // 60), int(trajectory_time % 60))}"
        print("Starting trajectory took: "+trajectory_time_text)
        print("")

        # Skip trajectory if ipSAE is too low (not worth relaxing and scoring)
        min_traj_ipsae = advanced_settings.get("min_trajectory_ipsae", 0.5)
        traj_ipsae_val = trajectory_metrics.get('ipSAE', None)
        if traj_ipsae_val is not None and traj_ipsae_val < min_traj_ipsae:
            print(f"Trajectory ipSAE ({traj_ipsae_val:.4f}) below threshold ({min_traj_ipsae}), skipping")
            continue

        # Proceed if there is no trajectory termination signal
        if trajectory.aux["log"]["terminate"] == "":
            # Relax binder to calculate statistics
            trajectory_relaxed = os.path.join(design_paths["Trajectory/Relaxed"], design_name + ".pdb")
            pr_relax(trajectory_pdb, trajectory_relaxed, use_pyrosetta=use_pyrosetta)

            # define binder chain, placeholder in case multi-chain parsing in ColabDesign gets changed
            binder_chain = "B"

            # Calculate clashes before and after relaxation
            num_clashes_trajectory = calculate_clash_score(trajectory_pdb)
            num_clashes_relaxed = calculate_clash_score(trajectory_relaxed)

            # secondary structure content of starting trajectory binder and interface
            trajectory_alpha, trajectory_beta, trajectory_loops, trajectory_alpha_interface, trajectory_beta_interface, trajectory_loops_interface, trajectory_i_plddt, trajectory_ss_plddt = calc_ss_percentage(trajectory_pdb, advanced_settings, binder_chain)

            # analyze interface scores for relaxed af2 trajectory
            trajectory_interface_scores, trajectory_interface_AA, trajectory_interface_residues = score_interface(trajectory_relaxed, binder_chain, use_pyrosetta=use_pyrosetta)

            # starting binder sequence
            trajectory_sequence = trajectory.get_seq(get_best=True)[0]

            # analyze sequence
            traj_seq_notes = validate_design_sequence(trajectory_sequence, num_clashes_relaxed, advanced_settings)

            # target structure RMSD compared to input PDB
            trajectory_target_rmsd = target_pdb_rmsd(trajectory_pdb, target_settings["starting_pdb"], target_settings["chains"])

            # save trajectory statistics into CSV
            trajectory_data = [design_name, advanced_settings["design_algorithm"], length, seed, helicity_value, target_settings["target_hotspot_residues"], trajectory_sequence, trajectory_interface_residues, 
                                trajectory_metrics['plddt'], trajectory_metrics['ptm'], trajectory_metrics['i_ptm'], trajectory_metrics['pae'], trajectory_metrics['i_pae'],
                                trajectory_metrics.get('ipSAE', None),
                                None, None, None,  # pDockQ, pDockQ2, LIS — not computed for trajectory
                                trajectory_i_plddt, trajectory_ss_plddt, num_clashes_trajectory, num_clashes_relaxed, trajectory_interface_scores['binder_score'],
                                trajectory_interface_scores['surface_hydrophobicity'], trajectory_interface_scores['interface_sc'], trajectory_interface_scores['interface_packstat'],
                                trajectory_interface_scores['interface_dG'], trajectory_interface_scores['interface_dSASA'], trajectory_interface_scores['interface_dG_SASA_ratio'],
                                trajectory_interface_scores['interface_fraction'], trajectory_interface_scores['interface_hydrophobicity'], trajectory_interface_scores['interface_nres'], trajectory_interface_scores['interface_interface_hbonds'],
                                trajectory_interface_scores['interface_hbond_percentage'], trajectory_interface_scores['interface_delta_unsat_hbonds'], trajectory_interface_scores['interface_delta_unsat_hbonds_percentage'],
                                trajectory_alpha_interface, trajectory_beta_interface, trajectory_loops_interface, trajectory_alpha, trajectory_beta, trajectory_loops, trajectory_interface_AA, trajectory_target_rmsd, 
                                trajectory_time_text, traj_seq_notes, settings_file, filters_file, advanced_file]
            insert_data(trajectory_csv, trajectory_data)

            # Skip MPNN optimization if no interface residues (no hotspot contact)
            if not trajectory_interface_residues:
                print(f"No interface residues found for {design_name}, skipping MPNN optimization")
                continue
            
            if advanced_settings["enable_mpnn"]:
                # initialise MPNN counters
                mpnn_n = 1
                accepted_mpnn = 0
                mpnn_dict = {}
                design_start_time = time.time()

                ### MPNN redesign of starting binder
                mpnn_trajectories = mpnn_gen_sequence(trajectory_pdb, binder_chain, trajectory_interface_residues, advanced_settings)
                
                existing_mpnn_sequences = set()
                if os.path.exists(mpnn_csv) and os.path.getsize(mpnn_csv) > 0:
                    try:
                        df_mpnn = pd.read_csv(mpnn_csv, usecols=['Sequence'])
                        if not df_mpnn.empty:
                            existing_mpnn_sequences = set(df_mpnn['Sequence'].dropna().astype(str).values)
                    except pd.errors.EmptyDataError:
                        print(f"Warning: {mpnn_csv} is empty or has no columns. Starting with no existing MPNN sequences.")
                    except KeyError:
                        print(f"Warning: 'Sequence' column not found in {mpnn_csv}. Starting with no existing MPNN sequences.")
                    except Exception as e:
                        print(f"Warning: Could not read existing MPNN sequences from {mpnn_csv} due to: {e}. Starting with no existing MPNN sequences.")
                else:
                    print(f"Info: {mpnn_csv} does not exist or is empty. Starting with no existing MPNN sequences.")

                # create set of MPNN sequences with allowed amino acid composition
                restricted_AAs = set(aa.strip().upper() for aa in advanced_settings["omit_AAs"].split(',')) if advanced_settings["force_reject_AA"] else set()

                mpnn_sequences = sorted({
                    mpnn_trajectories['seq'][n][-length:]: {
                        'seq': mpnn_trajectories['seq'][n][-length:],
                        'score': mpnn_trajectories['score'][n],
                        'seqid': mpnn_trajectories['seqid'][n]
                    } for n in range(advanced_settings["num_seqs"])
                    if (not restricted_AAs or not any(aa in mpnn_trajectories['seq'][n][-length:].upper() for aa in restricted_AAs))
                    and mpnn_trajectories['seq'][n][-length:] not in existing_mpnn_sequences
                }.values(), key=lambda x: x['score'])

                del existing_mpnn_sequences
  
                # check whether any sequences are left after amino acid rejection and duplication check, and if yes proceed with prediction
                if mpnn_sequences:
                    # add optimisation for increasing recycles if trajectory is beta sheeted
                    if advanced_settings["optimise_beta"] and float(trajectory_beta) > 15:
                        advanced_settings["num_recycles_validation"] = advanced_settings["optimise_beta_recycles_valid"]

                    ### Compile prediction models once for faster prediction of MPNN sequences
                    clear_mem()
                    # compile complex prediction model
                    complex_prediction_model = mk_afdesign_model(protocol="binder", num_recycles=advanced_settings["num_recycles_validation"], data_dir=advanced_settings["af_params_dir"], 
                                                                use_multimer=multimer_validation, use_initial_guess=advanced_settings["predict_initial_guess"], use_initial_atom_pos=advanced_settings["predict_bigbang"])
                    if advanced_settings["predict_initial_guess"] or advanced_settings["predict_bigbang"]:
                        complex_prediction_model.prep_inputs(pdb_filename=trajectory_pdb, chain='A', binder_chain='B', binder_len=length, use_binder_template=True, rm_target_seq=advanced_settings["rm_template_seq_predict"],
                                                            rm_target_sc=advanced_settings["rm_template_sc_predict"], rm_template_ic=True)
                    else:
                        complex_prediction_model.prep_inputs(pdb_filename=target_settings["starting_pdb"], chain=target_settings["chains"], binder_len=length, rm_target_seq=advanced_settings["rm_template_seq_predict"],
                                                            rm_target_sc=advanced_settings["rm_template_sc_predict"])

                    # compile binder monomer prediction model
                    binder_prediction_model = mk_afdesign_model(protocol="hallucination", use_templates=False, initial_guess=False, 
                                                                use_initial_atom_pos=False, num_recycles=advanced_settings["num_recycles_validation"], 
                                                                data_dir=advanced_settings["af_params_dir"], use_multimer=multimer_validation)
                    binder_prediction_model.prep_inputs(length=length)

                    # iterate over designed sequences — Phase 1: predict and score all
                    maturation_candidates = []
                    for mpnn_sequence in mpnn_sequences:
                        mpnn_time = time.time()

                        # generate mpnn design name numbering
                        mpnn_design_name = design_name + "_mpnn" + str(mpnn_n)
                        mpnn_score = round(mpnn_sequence['score'],2)
                        mpnn_seqid = round(mpnn_sequence['seqid'],2)

                        # add design to dictionary
                        mpnn_dict[mpnn_design_name] = {'seq': mpnn_sequence['seq'], 'score': mpnn_score, 'seqid': mpnn_seqid}

                        # save fasta sequence
                        if advanced_settings["save_mpnn_fasta"] is True:
                            save_fasta(mpnn_design_name, mpnn_sequence['seq'], design_paths)
                        
                        ### Predict mpnn redesigned binder complex using masked templates
                        mpnn_complex_statistics, pass_af2_filters, early_filter_details = predict_binder_complex(complex_prediction_model,
                                                                                        mpnn_sequence['seq'], mpnn_design_name,
                                                                                        target_settings["starting_pdb"], target_settings["chains"],
                                                                                        length, trajectory_pdb, prediction_models, advanced_settings,
                                                                                        filters, design_paths, failure_csv, use_pyrosetta=use_pyrosetta)

                        # if AF2 filters are not passed then skip the scoring but log the failure
                        if not pass_af2_filters:
                            print(f"Base AF2 filters not passed for {mpnn_design_name}, skipping full interface scoring.")

                            # Log to rejected_mpnn_full_stats.csv for early AF2 failures
                            rejected_data_list_for_csv = [mpnn_design_name, mpnn_sequence['seq']]
                            
                            failed_base_metrics_for_this_design = set()
                            if early_filter_details: # early_filter_details is the filter_failures dict e.g. {"1_pLDDT": 1}
                                for specific_model_failure_key in early_filter_details.keys():
                                    parts = specific_model_failure_key.split('_')
                                    if len(parts) > 1 and parts[0].isdigit(): # e.g. "1_pLDDT" -> "pLDDT"
                                        base_metric_name = ''.join(parts[1:]) # Corrected: was '_'.join, should be '' to match pLDDT from pLDDT
                                        # Special case for i_pTM, i_pAE, i_pLDDT as they already contain an underscore
                                        if parts[1] == "i" and len(parts) > 2: # e.g. "1_i_pTM" -> "i_pTM"
                                            base_metric_name = parts[1] + "_" + ''.join(parts[2:])
                                        failed_base_metrics_for_this_design.add(base_metric_name)
                                    else: # Should not happen with current predict_binder_complex failures, but good for robustness
                                        failed_base_metrics_for_this_design.add(specific_model_failure_key)

                            for base_filter_col_name_in_log_csv in filter_column_names_for_rejected_log:
                                if base_filter_col_name_in_log_csv in failed_base_metrics_for_this_design:
                                    rejected_data_list_for_csv.append(1)
                                else:
                                    rejected_data_list_for_csv.append(0)
                            insert_data(rejected_mpnn_full_stats_csv, rejected_data_list_for_csv)
                            
                            mpnn_n += 1
                            continue

                        # calculate statistics for each model individually
                        for model_num in prediction_models:
                            mpnn_design_pdb = os.path.join(design_paths["MPNN"], f"{mpnn_design_name}_model{model_num+1}.pdb")
                            mpnn_design_relaxed = os.path.join(design_paths["MPNN/Relaxed"], f"{mpnn_design_name}_model{model_num+1}.pdb")

                            if os.path.exists(mpnn_design_pdb):
                                # Calculate clashes before and after relaxation
                                num_clashes_mpnn = calculate_clash_score(mpnn_design_pdb)
                                num_clashes_mpnn_relaxed = calculate_clash_score(mpnn_design_relaxed)

                                # analyze interface scores for relaxed af2 trajectory
                                mpnn_interface_scores, mpnn_interface_AA, mpnn_interface_residues = score_interface(mpnn_design_relaxed, binder_chain, use_pyrosetta=use_pyrosetta)

                                # secondary structure content of starting trajectory binder
                                mpnn_alpha, mpnn_beta, mpnn_loops, mpnn_alpha_interface, mpnn_beta_interface, mpnn_loops_interface, mpnn_i_plddt, mpnn_ss_plddt = calc_ss_percentage(mpnn_design_pdb, advanced_settings, binder_chain)
                                
                                # unaligned RMSD calculate to determine if binder is in the designed binding site
                                rmsd_site = unaligned_rmsd(trajectory_pdb, mpnn_design_pdb, binder_chain, binder_chain, use_pyrosetta=use_pyrosetta)

                                # calculate RMSD of target compared to input PDB
                                target_rmsd = target_pdb_rmsd(mpnn_design_pdb, target_settings["starting_pdb"], target_settings["chains"])

                                # add the additional statistics to the mpnn_complex_statistics dictionary
                                mpnn_complex_statistics[model_num+1].update({
                                    'i_pLDDT': mpnn_i_plddt,
                                    'ss_pLDDT': mpnn_ss_plddt,
                                    'Unrelaxed_Clashes': num_clashes_mpnn,
                                    'Relaxed_Clashes': num_clashes_mpnn_relaxed,
                                    'Binder_Energy_Score': mpnn_interface_scores['binder_score'],
                                    'Surface_Hydrophobicity': mpnn_interface_scores['surface_hydrophobicity'],
                                    'ShapeComplementarity': mpnn_interface_scores['interface_sc'],
                                    'PackStat': mpnn_interface_scores['interface_packstat'],
                                    'dG': mpnn_interface_scores['interface_dG'],
                                    'dSASA': mpnn_interface_scores['interface_dSASA'], 
                                    'dG/dSASA': mpnn_interface_scores['interface_dG_SASA_ratio'],
                                    'Interface_SASA_%': mpnn_interface_scores['interface_fraction'],
                                    'Interface_Hydrophobicity': mpnn_interface_scores['interface_hydrophobicity'],
                                    'n_InterfaceResidues': mpnn_interface_scores['interface_nres'],
                                    'n_InterfaceHbonds': mpnn_interface_scores['interface_interface_hbonds'],
                                    'InterfaceHbondsPercentage': mpnn_interface_scores['interface_hbond_percentage'],
                                    'n_InterfaceUnsatHbonds': mpnn_interface_scores['interface_delta_unsat_hbonds'],
                                    'InterfaceUnsatHbondsPercentage': mpnn_interface_scores['interface_delta_unsat_hbonds_percentage'],
                                    'InterfaceAAs': mpnn_interface_AA,
                                    'Interface_Helix%': mpnn_alpha_interface,
                                    'Interface_BetaSheet%': mpnn_beta_interface,
                                    'Interface_Loop%': mpnn_loops_interface,
                                    'Binder_Helix%': mpnn_alpha,
                                    'Binder_BetaSheet%': mpnn_beta,
                                    'Binder_Loop%': mpnn_loops,
                                    'Hotspot_RMSD': rmsd_site,
                                    'Target_RMSD': target_rmsd
                                })

                                # save space by removing unrelaxed predicted mpnn complex pdb?
                                if advanced_settings["remove_unrelaxed_complex"]:
                                    os.remove(mpnn_design_pdb)

                        # calculate complex averages
                        mpnn_complex_averages = calculate_averages(mpnn_complex_statistics, handle_aa=True)
                        
                        ### Predict binder alone in single sequence mode
                        binder_statistics = predict_binder_alone(binder_prediction_model, mpnn_sequence['seq'], mpnn_design_name, length,
                                                                trajectory_pdb, binder_chain, prediction_models, advanced_settings, design_paths, 
                                                                use_pyrosetta=use_pyrosetta)

                        # extract RMSDs of binder to the original trajectory
                        for model_num in prediction_models:
                            mpnn_binder_pdb = os.path.join(design_paths["MPNN/Binder"], f"{mpnn_design_name}_model{model_num+1}.pdb")

                            if os.path.exists(mpnn_binder_pdb):
                                rmsd_binder = unaligned_rmsd(trajectory_pdb, mpnn_binder_pdb, binder_chain, "A", use_pyrosetta=use_pyrosetta)
                            else:
                                rmsd_binder = None

                            # append to statistics
                            binder_statistics[model_num+1].update({
                                    'Binder_RMSD': rmsd_binder
                                })

                            # save space by removing binder monomer models?
                            if advanced_settings["remove_binder_monomer"]:
                                os.remove(mpnn_binder_pdb)

                        # calculate binder averages
                        binder_averages = calculate_averages(binder_statistics)

                        # analyze sequence to make sure there are no cysteins and it contains residues that absorb UV for detection
                        seq_notes = validate_design_sequence(mpnn_sequence['seq'], mpnn_complex_averages.get('Relaxed_Clashes', None), advanced_settings)

                        # measure time to generate design
                        mpnn_end_time = time.time() - mpnn_time
                        elapsed_mpnn_text = f"{'%d hours, %d minutes, %d seconds' % (int(mpnn_end_time // 3600), int((mpnn_end_time % 3600) // 60), int(mpnn_end_time % 60))}"


                        # Insert statistics about MPNN design into CSV, will return None if corresponding model does note exist
                        model_numbers = range(1, 6)
                        statistics_labels = ['pLDDT', 'pTM', 'i_pTM', 'pAE', 'i_pAE', 'ipSAE', 'pDockQ', 'pDockQ2', 'LIS', 'i_pLDDT', 'ss_pLDDT', 'Unrelaxed_Clashes', 'Relaxed_Clashes', 'Binder_Energy_Score', 'Surface_Hydrophobicity',
                                            'ShapeComplementarity', 'PackStat', 'dG', 'dSASA', 'dG/dSASA', 'Interface_SASA_%', 'Interface_Hydrophobicity', 'n_InterfaceResidues', 'n_InterfaceHbonds', 'InterfaceHbondsPercentage',
                                            'n_InterfaceUnsatHbonds', 'InterfaceUnsatHbondsPercentage', 'Interface_Helix%', 'Interface_BetaSheet%', 'Interface_Loop%', 'Binder_Helix%',
                                            'Binder_BetaSheet%', 'Binder_Loop%', 'InterfaceAAs', 'Hotspot_RMSD', 'Target_RMSD']

                        # Initialize mpnn_data with the non-statistical data
                        mpnn_data = [mpnn_design_name, advanced_settings["design_algorithm"], length, seed, helicity_value, target_settings["target_hotspot_residues"], mpnn_sequence['seq'], mpnn_interface_residues, mpnn_score, mpnn_seqid]

                        # Add the statistical data for mpnn_complex
                        for label in statistics_labels:
                            mpnn_data.append(mpnn_complex_averages.get(label, None))
                            for model in model_numbers:
                                mpnn_data.append(mpnn_complex_statistics.get(model, {}).get(label, None))

                        # Add the statistical data for binder
                        for label in ['pLDDT', 'pTM', 'pAE', 'Binder_RMSD']:  # These are the labels for binder alone
                            mpnn_data.append(binder_averages.get(label, None))
                            for model in model_numbers:
                                mpnn_data.append(binder_statistics.get(model, {}).get(label, None))

                        # Add maturation placeholder columns (will be updated if maturation runs)
                        maturation_data = [None] * 9  # 9 maturation columns
                        mpnn_data.extend(maturation_data)

                        # Add the remaining non-statistical data
                        mpnn_data.extend([elapsed_mpnn_text, seq_notes, settings_file, filters_file, advanced_file])

                        # insert data into csv (maturation columns updated in-place below if maturation runs)
                        insert_data(mpnn_csv, mpnn_data)

                        # find best model number by pLDDT
                        plddt_values = {i: mpnn_data[i] for i in range(11, 16) if mpnn_data[i] is not None}

                        if not plddt_values:
                            print(f"Warning: No valid pLDDT values for {mpnn_design_name}, skipping")
                            continue

                        # Find the key with the highest value
                        highest_plddt_key = int(max(plddt_values, key=plddt_values.get))

                        # Output the number part of the key
                        best_model_number = highest_plddt_key - 10
                        best_model_pdb = os.path.join(design_paths["MPNN/Relaxed"], f"{mpnn_design_name}_model{best_model_number}.pdb")

                        # Collect candidate for Phase 2 selection
                        maturation_candidates.append({
                            'design_name': mpnn_design_name,
                            'sequence': mpnn_sequence['seq'],
                            'mpnn_data': mpnn_data,
                            'complex_averages': mpnn_complex_averages,
                            'complex_statistics': mpnn_complex_statistics,
                            'best_model_number': best_model_number,
                            'best_model_pdb': best_model_pdb,
                            'ipSAE': mpnn_complex_averages.get('ipSAE', float('-inf')),
                            'binder_statistics': binder_statistics,
                            'binder_averages': binder_averages,
                            'mpnn_score': mpnn_score,
                            'mpnn_seqid': mpnn_seqid,
                        })

                        # increase MPNN design number
                        mpnn_n += 1

                        # Force garbage collection after each MPNN design to prevent file descriptor accumulation
                        gc.collect()

                    ############################################
                    ### Phase 2: Select best maturation candidate
                    ############################################
                    mat_enabled = advanced_settings.get("enable_maturation", False)
                    mat_pre_filters = advanced_settings.get("maturation_pre_filters", False)
                    mat_revert_on_worse = advanced_settings.get("maturation_revert_on_worse", True)
                    eligible_candidates = []

                    if mat_enabled and maturation_candidates:
                        # Deduplicate by MPNN sequence: keep best ipSAE per unique sequence
                        seen_seqs = {}
                        for cand in maturation_candidates:
                            seq = cand['sequence']
                            if seq not in seen_seqs or cand['ipSAE'] > seen_seqs[seq]['ipSAE']:
                                seen_seqs[seq] = cand
                        unique_candidates = list(seen_seqs.values())

                        # Sort by ipSAE descending
                        ranked = sorted(unique_candidates, key=lambda c: c['ipSAE'], reverse=True)

                        # Count HQ residues for each candidate (including REU)
                        for candidate in ranked:
                            stats = candidate['complex_statistics'].get(candidate['best_model_number'], {})
                            cand_reu = compute_per_residue_reu(
                                candidate['best_model_pdb'], binder_chain=binder_chain,
                                use_pyrosetta=use_pyrosetta)
                            candidate['n_hq'] = count_high_quality_residues(
                                candidate['best_model_pdb'],
                                stats.get('_pae_matrix'), stats.get('_plddt_array'),
                                stats.get('_target_len'), stats.get('_binder_len'),
                                advanced_settings, binder_chain=binder_chain,
                                per_residue_reu=cand_reu)

                        # Build ranked list: eligible (n_hq > 0), sorted by n_hq desc, top 3 only
                        mat_max_candidates = advanced_settings.get("maturation_max_candidates", 3)
                        eligible_candidates = sorted(
                            [c for c in ranked if c.get('n_hq', 0) > 0],
                            key=lambda c: c['n_hq'], reverse=True)[:mat_max_candidates]

                        if eligible_candidates:
                            print(f"\n[Maturation Selection] {len(eligible_candidates)} candidates to try "
                                  f"(best: {eligible_candidates[0]['design_name']}, "
                                  f"ipSAE={eligible_candidates[0]['ipSAE']:.4f}, {eligible_candidates[0]['n_hq']} HQ residues)")
                        else:
                            print(f"\n[Maturation Selection] No candidates with high-quality interface residues, skipping maturation")

                    ############################################
                    ### Phase 3: Process designs
                    ############################################
                    if mat_enabled and maturation_candidates and eligible_candidates:
                        # Build context dict for maturation (shared across candidates)
                        ctx = {
                            'complex_prediction_model': complex_prediction_model,
                            'design_models': design_models,
                            'prediction_models': prediction_models,
                            'target_settings': target_settings,
                            'length': length,
                            'seed': seed,
                            'helicity_value': helicity_value,
                            'binder_chain': binder_chain,
                            'design_paths': design_paths,
                            'failure_csv': failure_csv,
                            'mpnn_csv': mpnn_csv,
                            'use_pyrosetta': use_pyrosetta,
                            'filters': filters,
                        }

                        # Try maturation on each eligible candidate until one succeeds
                        maturation_succeeded = False
                        for cand_idx, cand in enumerate(eligible_candidates):
                            cand_mpnn_data = list(cand['mpnn_data'])
                            cand_design_name = cand['design_name']
                            cand_best_model_pdb = cand['best_model_pdb']
                            cand_pre_mat_pdb = cand['best_model_pdb']  # Save for revert
                            cand_pre_mat_data = list(cand_mpnn_data)  # Save copy for revert

                            if mat_pre_filters:
                                ############################################
                                ### Pre-filter maturation: mature first, then filter check
                                ############################################
                                mat_ran, cand_mpnn_data, cand_best_model_pdb = _run_maturation(
                                    cand, ctx, advanced_settings, mat_label="pre-filter")

                                if not mat_ran and cand_idx < len(eligible_candidates) - 1:
                                    print(f"  Maturation failed for {cand_design_name}, trying next candidate...")
                                    continue

                                # Revert if maturation worsened metrics
                                _, cand_mpnn_data, cand_best_model_pdb, mat_ran = _check_maturation_revert(
                                    mat_ran, mat_revert_on_worse, cand_mpnn_data,
                                    cand_pre_mat_data, cand_pre_mat_pdb, cand_best_model_pdb)

                                # Filter check on matured (or reverted) design
                                filter_conditions = check_filters(cand_mpnn_data, design_labels, filters)
                                if filter_conditions is True:
                                    final_name = cand_design_name + ("_matured" if mat_ran else "")
                                    if mat_ran:
                                        cand_mpnn_data[0] = final_name
                                    print(final_name + " passed all filters")
                                    accepted_mpnn += 1
                                    accepted_designs += 1

                                    if mat_ran:
                                        update_last_csv_row(mpnn_csv, cand_mpnn_data)

                                    _accept_design(cand_best_model_pdb, cand_mpnn_data, design_name,
                                                   design_paths, final_csv, advanced_settings)
                                    maturation_succeeded = True
                                    break
                                else:
                                    _reject_design(cand_design_name, cand_best_model_pdb, cand_mpnn_data,
                                                   filter_conditions, failure_csv, design_paths,
                                                   filter_column_names_for_rejected_log,
                                                   rejected_mpnn_full_stats_csv, cand['sequence'])

                            else:
                                ############################################
                                ### Post-filter maturation: filter check first, then mature
                                ############################################
                                filter_conditions = check_filters(cand_mpnn_data, design_labels, filters)
                                if filter_conditions is True:
                                    print(cand_design_name + " passed all filters")
                                    accepted_mpnn += 1
                                    accepted_designs += 1

                                    mat_ran, cand_mpnn_data, cand_best_model_pdb = _run_maturation(
                                        cand, ctx, advanced_settings, mat_label="post-filter")

                                    if not mat_ran and cand_idx < len(eligible_candidates) - 1:
                                        print(f"  Maturation failed for {cand_design_name}, trying next candidate...")
                                        accepted_mpnn -= 1
                                        accepted_designs -= 1
                                        continue

                                    # Revert if maturation worsened metrics
                                    _, cand_mpnn_data, cand_best_model_pdb, mat_ran = _check_maturation_revert(
                                        mat_ran, mat_revert_on_worse, cand_mpnn_data,
                                        cand_pre_mat_data, cand_pre_mat_pdb, cand_best_model_pdb)

                                    final_name = cand_design_name + ("_matured" if mat_ran else "")
                                    if mat_ran:
                                        cand_mpnn_data[0] = final_name
                                        update_last_csv_row(mpnn_csv, cand_mpnn_data)

                                    _accept_design(cand_best_model_pdb, cand_mpnn_data, design_name,
                                                   design_paths, final_csv, advanced_settings)
                                    maturation_succeeded = True
                                    break
                                else:
                                    _reject_design(cand_design_name, cand_best_model_pdb, cand_mpnn_data,
                                                   filter_conditions, failure_csv, design_paths,
                                                   filter_column_names_for_rejected_log,
                                                   rejected_mpnn_full_stats_csv, cand['sequence'])

                    else:
                        # No maturation candidate — process ALL candidates through normal filter checking
                        for cand in maturation_candidates:
                            cand_mpnn_data = cand['mpnn_data']
                            cand_design_name = cand['design_name']
                            cand_best_model_pdb = cand['best_model_pdb']

                            filter_conditions = check_filters(cand_mpnn_data, design_labels, filters)
                            if filter_conditions is True:
                                print(cand_design_name + " passed all filters")
                                accepted_mpnn += 1
                                accepted_designs += 1

                                _accept_design(cand_best_model_pdb, cand_mpnn_data, design_name,
                                               design_paths, final_csv, advanced_settings)
                            else:
                                _reject_design(cand_design_name, cand_best_model_pdb, cand_mpnn_data,
                                               filter_conditions, failure_csv, design_paths,
                                               filter_column_names_for_rejected_log,
                                               rejected_mpnn_full_stats_csv, cand['sequence'])

                            if accepted_mpnn >= advanced_settings["max_mpnn_sequences"]:
                                break

                    if accepted_mpnn >= 1:
                        print("Found "+str(accepted_mpnn)+" MPNN designs passing filters")
                        print("")
                    else:
                        print("No accepted MPNN designs found for this trajectory.")
                        print("")

                else:
                    print('Duplicate MPNN designs sampled with different trajectory, skipping current trajectory optimisation')
                    print("")

                # save space by removing unrelaxed design trajectory PDB
                if advanced_settings["remove_unrelaxed_trajectory"]:
                    os.remove(trajectory_pdb)

                # measure time it took to generate designs for one trajectory
                design_time = time.time() - design_start_time
                design_time_text = f"{'%d hours, %d minutes, %d seconds' % (int(design_time // 3600), int((design_time % 3600) // 60), int(design_time % 60))}"
                print("Design and validation of trajectory "+design_name+" took: "+design_time_text)

            # analyse the rejection rate of trajectories to see if we need to readjust the design weights
            if trajectory_n >= advanced_settings["start_monitoring"] and advanced_settings["enable_rejection_check"]:
                acceptance = accepted_designs / trajectory_n
                if not acceptance >= advanced_settings["acceptance_rate"]:
                    print("The ratio of successful designs is lower than defined acceptance rate! Consider changing your design settings!")
                    print("Script execution stopping...")
                    break

        # increase trajectory number
        trajectory_n += 1
        
        # Force garbage collection and clear DSSP cache every 10 trajectories to prevent memory/fd accumulation
        if trajectory_n % 10 == 0:
            clear_dssp_cache()
            print(f"Cleared DSSP cache after {trajectory_n} trajectories")
        
        # Force garbage collection more frequently to prevent file descriptor accumulation
        gc.collect()

### Script finished
elapsed_time = time.time() - script_start_time
elapsed_text = f"{'%d hours, %d minutes, %d seconds' % (int(elapsed_time // 3600), int((elapsed_time % 3600) // 60), int(elapsed_time % 60))}"
print("Finished all designs. Script execution for "+str(trajectory_n)+" trajectories took: "+elapsed_text)