####################################
############## Maturation functions
############## PPIFlow-inspired recursive affinity maturation
####################################
### Import dependencies
import numpy as np
from Bio.PDB import PDBParser, Selection
from scipy.spatial import cKDTree
from .logging_utils import vprint
from .biopython_utils import safe_dssp_calculation, three_to_one_map, one_to_three_map

# Conditionally import PyRosetta for per-residue REU scoring
_PR = None
_PYROSETTA_AVAILABLE = False
try:
    import pyrosetta as _PR
    _PYROSETTA_AVAILABLE = True
except ImportError:
    pass


def compute_per_residue_reu(pdb_file, binder_chain="B", use_pyrosetta=True,
                            return_pose=False):
    """
    Compute per-residue Rosetta Energy Units (REU) for binder residues.

    Scores the complex pose and extracts per-residue total energy for each
    binder residue. Lower (more negative) REU indicates more favorable energy.

    Args:
        pdb_file: Path to the relaxed complex PDB
        binder_chain: Chain ID for binder
        use_pyrosetta: Whether PyRosetta is available/enabled
        return_pose: If True, also return (pose, scorefxn) for reuse by scan_fixed_residues()

    Returns:
        If return_pose=False: dict mapping 0-based binder residue index to REU float, or None
        If return_pose=True: (per_residue_reu, pose, scorefxn) tuple; pose/scorefxn are None on failure
    """
    if not use_pyrosetta or not _PYROSETTA_AVAILABLE:
        return (None, None, None) if return_pose else None

    try:
        pose = _PR.pose_from_pdb(pdb_file)
        scorefxn = _PR.get_fa_scorefxn()
        scorefxn(pose)

        # Find binder chain residue range in pose numbering
        per_residue_reu = {}
        binder_idx = 0
        for resid in range(1, pose.total_residue() + 1):
            pdb_info = pose.pdb_info()
            if pdb_info.chain(resid) == binder_chain and pose.residue(resid).is_protein():
                reu = pose.energies().residue_total_energy(resid)
                per_residue_reu[binder_idx] = float(reu)
                binder_idx += 1

        vprint(f"[Maturation-REU] Computed per-residue REU for {len(per_residue_reu)} binder residues")
        if return_pose:
            return per_residue_reu, pose, scorefxn
        return per_residue_reu
    except Exception as e:
        vprint(f"[Maturation-REU] Failed to compute per-residue REU: {e}")
        return (None, None, None) if return_pose else None


def _target_interaction_reu(pose, pose_resid, target_pose_resids, scorefxn):
    """
    Compute REU for a residue considering only its internal (one-body) energy
    and pairwise interactions with target chain residues.

    Ignores interactions with binder scaffold residues, which will be redesigned
    in subsequent maturation steps and could cause false negatives.

    Args:
        pose: Scored PyRosetta Pose
        pose_resid: Pose residue number to evaluate
        target_pose_resids: set of pose residue numbers belonging to the target chain
        scorefxn: Rosetta ScoreFunction (for weight vector)

    Returns:
        float: one-body + target two-body REU
    """
    weights = scorefxn.weights()

    # One-body: internal energy (Ramachandran, rotamer probability, reference)
    onebody = pose.energies().onebody_energies(pose_resid).dot(weights)

    # Two-body: only interactions with target residues
    energy_graph = pose.energies().energy_graph()
    twobody = 0.0
    for t_resid in target_pose_resids:
        edge = energy_graph.find_energy_edge(pose_resid, t_resid)
        if edge is not None:
            twobody += edge.dot(weights)

    return onebody + twobody


def _repack_shell(pose, pose_resid, repack_shell_dist, scorefxn, minimize=False):
    """Repack residues within a shell around a mutation site, then re-score.

    If minimize=True, also runs constrained backbone minimization on the
    neighborhood shell after repacking. This allows local backbone adjustment
    to accommodate the mutation while keeping the rest of the structure fixed.
    """
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.pack.task.operation import (
        InitializeFromCommandline, RestrictToRepacking,
        OperateOnResidueSubset, PreventRepackingRLT
    )
    from pyrosetta.rosetta.core.select.residue_selector import (
        NeighborhoodResidueSelector, ResidueIndexSelector
    )
    from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover

    idx_sel = ResidueIndexSelector(str(pose_resid))
    nbr_sel = NeighborhoodResidueSelector(idx_sel, repack_shell_dist, True)
    prevent = PreventRepackingRLT()
    tf = TaskFactory()
    tf.push_back(InitializeFromCommandline())
    tf.push_back(RestrictToRepacking())
    tf.push_back(OperateOnResidueSubset(prevent, nbr_sel, True))
    task = tf.create_task_and_apply_taskoperations(pose)

    packer = PackRotamersMover(scorefxn)
    packer.task(task)
    packer.apply(pose)

    if minimize:
        _minimize_shell(pose, pose_resid, repack_shell_dist, scorefxn)
        # Re-score without coordinate constraints (minimize adds/removes them,
        # leaving stale constraint energies in the cache)
        scorefxn(pose)


def _minimize_shell(pose, pose_resid, shell_dist, scorefxn):
    """Constrained backbone minimization on the neighborhood shell.

    Enables chi and bb movement only for residues within the shell,
    constrained to starting coordinates. Uses a small number of iterations
    for fast local adjustment.
    """
    from pyrosetta.rosetta.core.kinematics import MoveMap
    from pyrosetta.rosetta.core.select.residue_selector import (
        NeighborhoodResidueSelector, ResidueIndexSelector
    )
    from pyrosetta.rosetta.core.select import get_residues_from_subset
    from pyrosetta.rosetta.protocols.minimization_packing import MinMover

    idx_sel = ResidueIndexSelector(str(pose_resid))
    nbr_sel = NeighborhoodResidueSelector(idx_sel, shell_dist, True)
    subset = nbr_sel.apply(pose)
    shell_residues = get_residues_from_subset(subset)

    mmf = MoveMap()
    mmf.set_chi(False)
    mmf.set_bb(False)
    mmf.set_jump(False)
    for res in shell_residues:
        mmf.set_chi(res, True)
        mmf.set_bb(res, True)

    # Add coordinate constraints to keep backbone close to starting position
    from pyrosetta.rosetta.protocols.relax import add_coordinate_constraints_to_pose
    add_coordinate_constraints_to_pose(pose, scorefxn)

    minmover = MinMover()
    minmover.movemap(mmf)
    minmover.score_function(scorefxn)
    minmover.min_type("lbfgs_armijo_nonmonotone")
    minmover.tolerance(0.01)
    minmover.max_iter(50)
    minmover.apply(pose)

    # Remove coordinate constraints after minimization
    from pyrosetta.rosetta.protocols.relax import remove_coordinate_constraints_from_pose
    remove_coordinate_constraints_from_pose(pose)


def _count_target_hbonds(pose, pose_resid, target_pose_resids, scorefxn):
    """Count hydrogen bonds between a residue and target chain residues.

    Uses Rosetta's hbond_sr_bb, hbond_lr_bb, hbond_bb_sc, hbond_sc score terms
    from the EnergyGraph edges to detect interface H-bonds.
    """
    from pyrosetta.rosetta.core.scoring import ScoreType

    hbond_types = [
        ScoreType.hbond_sr_bb, ScoreType.hbond_lr_bb,
        ScoreType.hbond_bb_sc, ScoreType.hbond_sc
    ]

    energy_graph = pose.energies().energy_graph()
    n_hbonds = 0
    for t_resid in target_pose_resids:
        edge = energy_graph.find_energy_edge(pose_resid, t_resid)
        if edge is None:
            continue
        for st in hbond_types:
            e = edge[st]
            if e < -0.1:  # meaningful H-bond contribution
                n_hbonds += 1
    return n_hbonds


def _decompose_target_energy(pose, pose_resid, target_pose_resids, scorefxn):
    """Decompose target interaction energy into key score terms for logging.

    Returns a dict of score term name -> energy value for the main categories.
    """
    from pyrosetta.rosetta.core.scoring import ScoreType

    term_groups = {
        'fa_atr': ScoreType.fa_atr,
        'fa_rep': ScoreType.fa_rep,
        'fa_elec': ScoreType.fa_elec,
        'hbond': None,  # aggregated below
        'fa_sol': ScoreType.fa_sol,
    }
    hbond_types = [
        ScoreType.hbond_sr_bb, ScoreType.hbond_lr_bb,
        ScoreType.hbond_bb_sc, ScoreType.hbond_sc
    ]

    weights = scorefxn.weights()
    energy_graph = pose.energies().energy_graph()
    decomposition = {k: 0.0 for k in term_groups}

    for t_resid in target_pose_resids:
        edge = energy_graph.find_energy_edge(pose_resid, t_resid)
        if edge is None:
            continue
        for name, st in term_groups.items():
            if st is not None:
                decomposition[name] += edge[st] * weights[st]
        for st in hbond_types:
            decomposition['hbond'] += edge[st] * weights[st]

    return decomposition


def scan_fixed_residues(pdb_file, fixed_binder_indices, binder_chain="B",
                       advanced_settings=None, output_pdb=None,
                       preloaded_pose=None, preloaded_scorefxn=None):
    """
    Saturation mutagenesis scan on fixed (high-quality) interface residues.

    For each fixed residue, tries all candidate amino acids via PyRosetta
    MutateResidue + neighborhood repacking + constrained backbone minimization.
    Mutations are evaluated using target-only interaction REU.

    Uses a two-pass strategy: first screens all candidates with repack-only,
    then runs the expensive minimize step only on the top 3 candidates per residue.

    Per-residue checks:
    - H-bond preservation: if a mutation loses interface H-bonds, it is rejected
      unless the REU improvement exceeds the hbond_override_threshold (default -2.0).
    - Score term decomposition is logged for accepted mutations.

    After all residues are scanned, a post-scan sanity gate checks that the
    overall interface dG has not worsened. If it has, the scan identifies and
    reverts the mutation with the worst per-residue iREU delta until dG recovers.

    Greedy sequential: each accepted mutation updates the pose context for
    subsequent residue evaluations.

    Args:
        pdb_file: Path to relaxed complex PDB
        fixed_binder_indices: set of 0-based binder residue indices to scan
        binder_chain: Chain ID for binder
        advanced_settings: dict with maturation settings
        output_pdb: Path to save updated PDB. If None, overwrites pdb_file.
        preloaded_pose: Optional pre-scored PyRosetta Pose (from compute_per_residue_reu)
        preloaded_scorefxn: Optional ScoreFunction to reuse

    Returns:
        updated_seq: Updated binder sequence string (1-letter codes)
        mutations: list of (binder_idx, old_aa, new_aa, old_reu, new_reu) tuples
    """
    if not _PYROSETTA_AVAILABLE:
        vprint("[Maturation-Scan] PyRosetta unavailable, skipping residue scan")
        return None, []

    if not fixed_binder_indices:
        return None, []

    if advanced_settings is None:
        advanced_settings = {}

    repack_shell = advanced_settings.get("maturation_scan_repack_shell", 8.0)
    exclude_aas = set(advanced_settings.get("maturation_scan_exclude_aas", ["GLY", "CYS"]))
    hbond_override = advanced_settings.get("maturation_scan_hbond_override_reu", -2.0)
    enable_minimize = advanced_settings.get("maturation_scan_minimize", True)

    try:
        from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

        # Reuse pre-loaded pose/scorefxn if available, otherwise load from PDB
        if preloaded_pose is not None and preloaded_scorefxn is not None:
            pose = preloaded_pose.clone()  # clone so we don't mutate the caller's pose
            scorefxn = preloaded_scorefxn
            scorefxn(pose)
        else:
            pose = _PR.pose_from_pdb(pdb_file)
            scorefxn = _PR.get_fa_scorefxn()
            scorefxn(pose)

        # Baseline interface dG for post-scan sanity gate
        def _get_interface_dG(p):
            """Compute interface dG on a pose clone using InterfaceAnalyzerMover."""
            iam = InterfaceAnalyzerMover()
            iam.set_interface("A_B")
            iam.set_scorefunction(scorefxn)
            iam.set_compute_interface_energy(True)
            iam.set_pack_separated(True)
            clone = p.clone()
            iam.apply(clone)
            return iam.get_interface_dG()

        baseline_dG = _get_interface_dG(pose)
        vprint(f"[Maturation-Scan] Baseline interface dG: {baseline_dG:.2f}")

        # Identify target chain residues (pose numbering)
        target_pose_resids = set()
        for resid in range(1, pose.total_residue() + 1):
            if (pose.pdb_info().chain(resid) != binder_chain
                    and pose.residue(resid).is_protein()):
                target_pose_resids.add(resid)

        # Build map: 0-based binder index -> pose residue number
        binder_pose_resids = []
        for resid in range(1, pose.total_residue() + 1):
            if (pose.pdb_info().chain(resid) == binder_chain
                    and pose.residue(resid).is_protein()):
                binder_pose_resids.append(resid)

        # Candidate AAs — exclude configured residue types from scanning
        aa_3letter = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE',
                      'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG',
                      'SER', 'THR', 'VAL', 'TRP', 'TYR']
        if exclude_aas:
            aa_3letter = [aa for aa in aa_3letter if aa not in exclude_aas]

        mutations = []
        n_top_candidates = 3  # number of repack-only candidates to refine with minimize

        for binder_idx in sorted(fixed_binder_indices):
            if binder_idx >= len(binder_pose_resids):
                continue
            pose_resid = binder_pose_resids[binder_idx]
            current_name3 = pose.residue(pose_resid).name3().strip()

            # Baseline: target-only interaction REU and H-bonds for current AA
            current_reu = _target_interaction_reu(pose, pose_resid,
                                                  target_pose_resids, scorefxn)
            current_hbonds = _count_target_hbonds(pose, pose_resid,
                                                   target_pose_resids, scorefxn)

            best_aa3 = current_name3
            best_reu = current_reu
            best_hbonds = current_hbonds

            # --- Pass 1: Screen all candidates with repack-only (no minimize) ---
            screen_results = []  # list of (trial_aa3, repack_reu)
            for trial_aa3 in aa_3letter:
                if trial_aa3 == current_name3:
                    continue

                trial_pose = pose.clone()
                MutateResidue(pose_resid, trial_aa3).apply(trial_pose)
                _repack_shell(trial_pose, pose_resid, repack_shell, scorefxn,
                              minimize=False)

                trial_reu = _target_interaction_reu(trial_pose, pose_resid,
                                                    target_pose_resids, scorefxn)
                if trial_reu < current_reu:
                    screen_results.append((trial_aa3, trial_reu))

            # --- Pass 2: Minimize only top candidates ---
            screen_results.sort(key=lambda x: x[1])
            top_candidates = screen_results[:n_top_candidates] if enable_minimize else screen_results

            for trial_aa3, _ in top_candidates:
                trial_pose = pose.clone()
                MutateResidue(pose_resid, trial_aa3).apply(trial_pose)
                _repack_shell(trial_pose, pose_resid, repack_shell, scorefxn,
                              minimize=enable_minimize)

                trial_reu = _target_interaction_reu(trial_pose, pose_resid,
                                                    target_pose_resids, scorefxn)

                if trial_reu >= best_reu:
                    continue

                # H-bond preservation check
                trial_hbonds = _count_target_hbonds(trial_pose, pose_resid,
                                                     target_pose_resids, scorefxn)
                delta_reu = trial_reu - current_reu
                if trial_hbonds < current_hbonds and delta_reu > hbond_override:
                    vprint(f"[Maturation-Scan] Pos {binder_idx+1}: "
                           f"{three_to_one_map.get(current_name3,'X')}->{three_to_one_map.get(trial_aa3,'X')} "
                           f"rejected: loses {current_hbonds - trial_hbonds} H-bond(s), "
                           f"delta iREU {delta_reu:.2f} > threshold {hbond_override}")
                    continue

                best_aa3 = trial_aa3
                best_reu = trial_reu
                best_hbonds = trial_hbonds

            # Apply best mutation to the main pose if improved
            if best_aa3 != current_name3:
                MutateResidue(pose_resid, best_aa3).apply(pose)
                _repack_shell(pose, pose_resid, repack_shell, scorefxn,
                              minimize=enable_minimize)

                old_1 = three_to_one_map.get(current_name3, 'X')
                new_1 = three_to_one_map.get(best_aa3, 'X')
                delta_reu = best_reu - current_reu

                # Score term decomposition logging
                decomp = _decompose_target_energy(pose, pose_resid,
                                                   target_pose_resids, scorefxn)
                decomp_str = ' '.join(f"{k}={v:.2f}" for k, v in decomp.items())

                mutations.append((binder_idx, old_1, new_1, current_reu, best_reu))
                vprint(f"[Maturation-Scan] Pos {binder_idx+1}: "
                       f"{old_1}->{new_1} (iREU {current_reu:.2f}->{best_reu:.2f}, "
                       f"delta={delta_reu:.2f}, hbonds {current_hbonds}->{best_hbonds}) "
                       f"[{decomp_str}]")

        # --- Post-scan interface sanity gate ---
        if mutations:
            post_dG = _get_interface_dG(pose)
            vprint(f"[Maturation-Scan] Post-scan interface dG: {post_dG:.2f} "
                   f"(baseline: {baseline_dG:.2f}, delta: {post_dG - baseline_dG:.2f})")

            # If dG worsened, revert mutations starting with the worst delta
            if post_dG > baseline_dG:
                vprint(f"[Maturation-Scan] Interface dG worsened — reverting problematic mutations")
                sorted_muts = sorted(mutations, key=lambda m: m[4] - m[3], reverse=True)

                for mut in sorted_muts:
                    b_idx, old_1, new_1, old_reu, new_reu = mut
                    if b_idx >= len(binder_pose_resids):
                        continue
                    pr_resid = binder_pose_resids[b_idx]

                    old_3 = one_to_three_map.get(old_1)
                    if old_3 is None:
                        continue

                    vprint(f"[Maturation-Scan] Reverting pos {b_idx+1}: {new_1}->{old_1}")
                    MutateResidue(pr_resid, old_3).apply(pose)
                    _repack_shell(pose, pr_resid, repack_shell, scorefxn,
                                  minimize=enable_minimize)
                    mutations.remove(mut)

                    check_dG = _get_interface_dG(pose)
                    if check_dG <= baseline_dG:
                        vprint(f"[Maturation-Scan] Interface dG recovered: {check_dG:.2f}")
                        break

        # Extract updated binder sequence
        updated_seq = ''.join(
            three_to_one_map.get(pose.residue(resid).name3().strip(), 'X')
            for resid in binder_pose_resids
        )

        # Save updated PDB
        save_path = output_pdb if output_pdb else pdb_file
        pose.dump_pdb(save_path)
        vprint(f"[Maturation-Scan] Scanned {len(fixed_binder_indices)} residues, "
               f"{len(mutations)} mutations accepted, saved to {save_path}")

        return updated_seq, mutations

    except Exception as e:
        vprint(f"[Maturation-Scan] Failed: {e}")
        return None, []


def assess_interface_residue_quality(pdb_file, pae_matrix, plddt_array,
                                     target_len, binder_len, advanced_settings,
                                     binder_chain="B", target_chain="A",
                                     per_residue_reu=None):
    """
    Assess each binder interface residue's quality using hard thresholds.

    A residue is "high quality" only if it passes ALL applicable filters:
    - REU <= reu_threshold (when PyRosetta available)
    - pLDDT >= plddt_threshold
    - mean interface PAE <= pae_threshold
    - Defined secondary structure (helix or sheet, not loop/coil)

    Using AND logic ensures every fixed residue is genuinely good on all
    metrics. Loop residues are excluded because their backbone flexibility
    makes them poor candidates for positional fixing.

    Args:
        pdb_file: Path to the complex PDB
        pae_matrix: Full PAE matrix from AF2 prediction (shape: [N, N] where N = target_len + binder_len)
        plddt_array: Per-residue pLDDT array from AF2 prediction (length N)
        target_len: Number of target residues
        binder_len: Number of binder residues
        advanced_settings: Dict with maturation parameters
        binder_chain: Chain ID for binder in PDB
        target_chain: Chain ID for target in PDB
        per_residue_reu: Optional dict {0-based binder index: REU float} from compute_per_residue_reu()

    Returns:
        dict mapping binder residue index (0-based relative to binder) to quality info:
        {
            'residue_id': int (PDB residue number),
            'aa': str (single letter),
            'reu': float or None,
            'plddt': float,
            'mean_interface_pae': float,
            'n_contacts': int,
            'pass_reu': bool,
            'pass_plddt': bool,
            'pass_pae': bool,
            'pass_ss': bool,
            'ss': str (DSSP code or 'X'),
            'is_high_quality': bool (all filters passed)
        }
    """
    contact_distance = advanced_settings.get("maturation_contact_distance", 4.0)
    plddt_thresh = advanced_settings.get("maturation_plddt_threshold", 0.85)
    pae_thresh = advanced_settings.get("maturation_pae_threshold", 5.0)
    reu_thresh = advanced_settings.get("maturation_reu_threshold", -5.0)
    require_ss = advanced_settings.get("maturation_require_defined_ss", True)

    has_reu = per_residue_reu is not None and len(per_residue_reu) > 0

    # Parse PDB and identify interface residues via KD-tree (same approach as hotspot_residues)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_file)
    model = structure[0]

    # Compute DSSP for SS assignments
    dssp_path = advanced_settings.get("dssp_path", "")
    dssp = safe_dssp_calculation(model, pdb_file, dssp_path) if require_ss else None
    # Build per-residue SS map for binder chain: (chain, resid) -> SS code
    residue_ss = {}
    if dssp is not None:
        for key in dssp.keys():
            chain_id, res_id = key
            if chain_id == binder_chain:
                ss_code = dssp[key][2]  # DSSP secondary structure code
                residue_ss[res_id[1]] = ss_code  # res_id is (hetflag, resseq, icode); use resseq int

    binder_atoms = Selection.unfold_entities(model[binder_chain], 'A')
    binder_coords = np.array([atom.coord for atom in binder_atoms])
    target_atoms = Selection.unfold_entities(model[target_chain], 'A')
    target_coords = np.array([atom.coord for atom in target_atoms])

    binder_tree = cKDTree(binder_coords)
    target_tree = cKDTree(target_coords)
    pairs = binder_tree.query_ball_tree(target_tree, contact_distance)

    # Collect per-residue contact info
    residue_contacts = {}  # binder_residue_id -> set of target residue ids contacted
    residue_aa = {}  # binder_residue_id -> single letter AA
    residue_to_binder_idx = {}  # binder PDB residue_id -> 0-based binder index

    # Build binder residue index mapping (PDB residue id -> 0-based binder position)
    binder_residues_seen = []
    for atom in binder_atoms:
        res = atom.get_parent()
        res_id = res.id[1]
        if res_id not in residue_to_binder_idx:
            residue_to_binder_idx[res_id] = len(binder_residues_seen)
            binder_residues_seen.append(res_id)
            resname = res.get_resname()
            if resname in three_to_one_map:
                residue_aa[res_id] = three_to_one_map[resname]

    for binder_atom_idx, close_indices in enumerate(pairs):
        if not close_indices:
            continue
        binder_res = binder_atoms[binder_atom_idx].get_parent()
        binder_res_id = binder_res.id[1]

        if binder_res_id not in residue_contacts:
            residue_contacts[binder_res_id] = set()
        for target_atom_idx in close_indices:
            target_res = target_atoms[target_atom_idx].get_parent()
            residue_contacts[binder_res_id].add(target_res.id[1])

    if not residue_contacts:
        vprint("[Maturation] No interface contacts found")
        return {}

    # Build target residue index mapping for PAE extraction
    target_residues_seen = []
    target_res_to_idx = {}
    for atom in target_atoms:
        res = atom.get_parent()
        res_id = res.id[1]
        if res_id not in target_res_to_idx:
            target_res_to_idx[res_id] = len(target_residues_seen)
            target_residues_seen.append(res_id)

    # Assess each interface residue against hard thresholds
    quality_results = {}

    for binder_res_id, contacted_target_res_ids in residue_contacts.items():
        binder_idx = residue_to_binder_idx.get(binder_res_id)
        if binder_idx is None:
            continue

        # --- REU ---
        res_reu = per_residue_reu.get(binder_idx) if has_reu else None
        pass_reu = res_reu is not None and res_reu <= reu_thresh

        # --- pLDDT ---
        res_plddt = float(plddt_array[target_len + binder_idx])
        pass_plddt = res_plddt >= plddt_thresh

        # --- PAE ---
        target_indices = []
        for t_res_id in contacted_target_res_ids:
            t_idx = target_res_to_idx.get(t_res_id)
            if t_idx is not None:
                target_indices.append(t_idx)

        if target_indices:
            pae_values = pae_matrix[target_len + binder_idx, target_indices]
            mean_pae = float(np.mean(pae_values))
        else:
            mean_pae = 30.0

        pass_pae = mean_pae <= pae_thresh

        # --- Contacts ---
        n_contacts = len(contacted_target_res_ids)

        # --- Secondary structure ---
        ss_code = residue_ss.get(binder_res_id, 'X')
        # H, G, I = helix types; E = beta sheet; everything else = loop/coil
        pass_ss = ss_code in ('H', 'G', 'I', 'E') if require_ss else True

        # High quality = passes ALL applicable filters (AND logic)
        if has_reu:
            is_high_quality = pass_reu and pass_plddt and pass_pae and pass_ss
        else:
            is_high_quality = pass_plddt and pass_pae and pass_ss

        quality_results[binder_idx] = {
            'residue_id': binder_res_id,
            'aa': residue_aa.get(binder_res_id, 'X'),
            'reu': res_reu,
            'plddt': res_plddt,
            'mean_interface_pae': mean_pae,
            'n_contacts': n_contacts,
            'pass_reu': pass_reu,
            'pass_plddt': pass_plddt,
            'pass_pae': pass_pae,
            'pass_ss': pass_ss,
            'ss': ss_code,
            'is_high_quality': is_high_quality
        }

    n_hq = sum(1 for v in quality_results.values() if v['is_high_quality'])
    n_ss_fail = sum(1 for v in quality_results.values() if not v['pass_ss'])
    vprint(f"[Maturation] {n_hq}/{len(quality_results)} interface residues pass all filters "
           f"(REU<={reu_thresh}, pLDDT>={plddt_thresh}, PAE<={pae_thresh}, SS=helix/sheet) "
           f"[{n_ss_fail} rejected for loop/coil SS]")

    return quality_results


def partition_interface_residues(residue_quality, existing_fixed_set, advanced_settings):
    """
    Partition interface residues into fixed (high quality) and redesignable sets.
    The fixed set is monotonically growing — once fixed, always fixed.

    Only residues that pass ALL hard thresholds are added to the fixed set.
    No minimum fix fraction is enforced — if no residues pass filters, none
    are fixed and all are redesigned.

    Args:
        residue_quality: dict from assess_interface_residue_quality()
        existing_fixed_set: set of binder residue indices already fixed from previous rounds
        advanced_settings: dict with maturation parameters

    Returns:
        fixed_indices: set of binder residue indices to fix
        redesign_indices: set of binder residue indices to redesign
        converged: bool (True if all interface residues are high quality)
    """
    max_fix_frac = advanced_settings.get("maturation_max_fix_fraction", 0.8)

    all_interface_indices = set(residue_quality.keys())
    if not all_interface_indices:
        return set(), set(), True

    # Start with previously fixed residues (monotonically growing)
    new_high_quality = {idx for idx, info in residue_quality.items() if info['is_high_quality']}
    fixed_indices = existing_fixed_set | new_high_quality

    # Apply max fraction bound (but always keep ALL existing_fixed_set — monotonic invariant)
    n_interface = len(all_interface_indices)
    max_fixed = max(1, int(n_interface * max_fix_frac))

    if len(fixed_indices) > max_fixed:
        # Sort newly added by REU (best first), keep existing_fixed_set always
        newly_added = fixed_indices - existing_fixed_set
        newly_sorted = sorted(
            newly_added,
            key=lambda i: residue_quality[i].get('reu', 0) or 0
        )  # most negative REU first
        allowed_new = max(0, max_fixed - len(existing_fixed_set))
        if allowed_new > 0:
            fixed_indices = existing_fixed_set | set(newly_sorted[:allowed_new])
        else:
            fixed_indices = existing_fixed_set

    redesign_indices = all_interface_indices - fixed_indices
    converged = len(redesign_indices) == 0

    return fixed_indices, redesign_indices, converged


def format_fixed_positions_for_mpnn(fixed_binder_indices, binder_chain="B"):
    """
    Format fixed positions for MPNN's fix_pos parameter.
    Target chain A is always fully fixed. Specific binder residues are added.

    Args:
        fixed_binder_indices: set of 0-based binder residue indices to fix
        binder_chain: chain ID for binder

    Returns:
        str like 'A,B1,B5,B12' for MPNN fix_pos parameter
    """
    # Target is always fixed
    parts = ['A']
    # Add binder residues (MPNN uses 1-based residue numbering)
    for idx in sorted(fixed_binder_indices):
        parts.append(f"{binder_chain}{idx + 1}")
    return ','.join(parts)


def get_maturation_metric(stats, metric_name):
    """
    Extract the improvement metric value from prediction stats.

    Args:
        stats: dict of prediction statistics (averages)
        metric_name: 'i_pTM', 'ipSAE', or 'composite'

    Returns:
        float metric value
    """
    if metric_name == "i_pTM":
        return stats.get('i_pTM', 0.0) or 0.0
    elif metric_name == "ipSAE":
        return stats.get('ipSAE', 0.0) or 0.0
    elif metric_name == "composite":
        iptm = stats.get('i_pTM', 0.0) or 0.0
        ipsae = stats.get('ipSAE', 0.0) or 0.0
        # Equal weight composite; both are 0-1 scale
        return 0.5 * iptm + 0.5 * ipsae
    else:
        return stats.get(metric_name, 0.0) or 0.0


def log_maturation_round(round_num, fixed_indices, redesign_indices, residue_quality,
                         metric_name, old_metric, new_metric, ipsae=None):
    """Print a summary of one maturation round."""
    fixed_aas = ''.join(residue_quality[i]['aa'] for i in sorted(fixed_indices) if i in residue_quality)
    redesign_aas = ''.join(residue_quality[i]['aa'] for i in sorted(redesign_indices) if i in residue_quality)
    delta = (new_metric or 0) - (old_metric or 0)
    ipsae_str = f", ipSAE: {ipsae:.4f}" if ipsae is not None else ""
    print(f"  Maturation round {round_num}: "
          f"fixed {len(fixed_indices)} residues [{fixed_aas}], "
          f"redesigning {len(redesign_indices)} [{redesign_aas}], "
          f"{metric_name}: {old_metric:.4f} -> {new_metric:.4f} (delta={delta:+.4f}){ipsae_str}")
