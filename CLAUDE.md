# FreeBindCraft — Development Guidelines

## Project Overview
FreeBindCraft is a modified BindCraft v1.52 for de novo protein binder design. It features a PyRosetta bypass (`--no-pyrosetta`) using OpenMM, FreeSASA, and sc-rs, plus a PPIFlow-inspired maturation feature for iterative interface refinement.

## Architecture

### Pipeline Flow
AF2 hallucination → MPNN sequence optimization → AF2 validation → relaxation → interface scoring → (optional) maturation loop

When `maturation_pre_filters: true`: hallucination → MPNN → AF2 validate → relax → **maturation** → filter check
When `maturation_pre_filters: false` (default): hallucination → MPNN → AF2 validate → relax → filter check → **maturation** (on accepted only)

### Key Files
- `bindcraft.py` — Main pipeline (~1050 lines). Maturation loop at ~line 834.
- `functions/maturation_utils.py` — Quality assessment, residue partitioning, MPNN formatting
- `functions/colabdesign_utils.py` — AF2 design, MPNN, prediction; includes `binder_maturation_hallucination()` and `mpnn_gen_sequence_maturation()`
- `functions/ipsae_utils.py` — Interface scoring: ipSAE, pDockQ, pDockQ2, LIS
- `functions/generic_utils.py` — CSV labels, directory setup, filter functions
- `functions/biopython_utils.py` — Structural analysis, DSSP, KD-tree contacts
- `functions/pr_alternative_utils.py` — OpenMM relaxation, SASA, scoring (PyRosetta-free path)
- `functions/__init__.py` — Wildcard imports for all modules
- `functions/ipsae.py` — Standalone reference script (Dunbrack original). Not imported by the pipeline.
- `settings_advanced/default_4stage_multimer_maturation.json` — Maturation-enabled config
- `settings_advanced/maturation_bindcraft_flow.json` — BindCraft-Flow maturation config

### PPIFlow Reference
- Located at `PPIFlow/` — reference implementation only, not integrated into runtime
- Key ideas: flow matching, partial flow refinement, hotspot-centric init, diffuse_mask mechanism
- See `PPIFlow/sample_binder_partial.py` for fixed-position refinement logic

## Implementation Rules

### General
- **No JAX locally** — JAX/ColabDesign require GPU. Validate syntax with `python -c "import ast; ast.parse(open('file.py').read())"`.
- **Backward compatibility is mandatory** — All new features must be opt-in via `advanced_settings.get("feature_name", False)`. Never break existing configs.
- **Wildcard imports** — All new modules in `functions/` must be added to `functions/__init__.py` with `from .new_module import *`.

### CSV and Data Schema
- `design_labels` in `generic_utils.py` defines all CSV columns.
- **Column ordering matters** — Maturation columns (9) come before the trailing `DesignTime`, `Notes`, `TargetSettings`, `Filters`, `AdvancedSettings` columns.
- When adding new metrics, insert columns before the trailing group and update `design_labels` accordingly.
- `statistics_labels` in `bindcraft.py` must mirror `core_labels` in `generic_utils.py` (minus the binder-only labels).

### Prediction Stats and Caching
- Internal/cached values use **underscore-prefixed keys** (e.g., `_pae_matrix`, `_plddt_array`, `_target_len`, `_binder_len`).
- `calculate_averages()` automatically **skips** underscore-prefixed keys — this is intentional.
- Never store large arrays (PAE matrices, pLDDT arrays) without the underscore prefix or they will break averaging.

### Maturation System
- **Monotonic fixed set** — Once a residue is marked high-quality and fixed, it stays fixed in all subsequent rounds. Never unfix.
- **Quality scoring** — Hard AND thresholds: a residue is high quality only if it passes ALL enabled filters: REU ≤ -5.0, pLDDT ≥ 0.85, PAE ≤ 5.0. SS filter (`maturation_require_defined_ss`) is off by default — loop residues can be fixed if they pass REU/pLDDT/PAE. No weighted composite — no compensation between metrics.
- **REU computation** — `compute_per_residue_reu()` scores relaxed PDB via `pose.energies().residue_total_energy()`.
- **REU normalization** — More negative REU = better. `norm_reu = clamp(reu / reu_threshold, 0, 1)` where `reu_threshold` defaults to -5.0.
- **Saturation scan** — `scan_fixed_residues()` uses a two-pass strategy: (1) screens all candidate AAs (GLY/CYS excluded) with repack-only, (2) runs expensive minimize on top 3 candidates per residue. Uses **target-only interaction REU** (`_target_interaction_reu()`: one-body + EnergyGraph pairwise with target chain only). Greedy sequential. Accepts `preloaded_pose`/`preloaded_scorefxn` to reuse the pose from `compute_per_residue_reu(return_pose=True)`.
- **Scan safeguards** — (1) H-bond preservation: mutations that lose interface H-bonds are rejected unless iREU improvement exceeds -2.0 REU (`maturation_scan_hbond_override_reu`). (2) Post-scan sanity gate: if overall interface dG worsens, the mutation with worst delta iREU is reverted first, repeating until dG recovers. (3) Score term decomposition (fa_atr, fa_rep, fa_elec, hbond, fa_sol) is logged for each accepted mutation.
- **Scan CSV tracking** — `Maturation_Scan_Mutations` column records all mutations as "R5Y(-3.2) F12W(-1.8)" format (residue, position, new AA, delta iREU).
- **AF2 integration** — Uses `af_model.opt["fix_pos"]` to freeze positions during hallucination.
- **Convergence** — Stops when BOTH i_pTM and ipSAE fail to improve by their respective thresholds, or all interface residues are fixed. Settings: `maturation_improvement_threshold` (i_pTM, default 0.01), `maturation_ipsae_improvement_threshold` (ipSAE, default 0.01). If ipSAE is unavailable, only i_pTM is checked.
- **Output** — Intermediate PDBs go to `MPNN/Maturation/` directory.

### Testing
- No test suite currently exists. Validate changes with `ast.parse()` for syntax.
- For logic changes, add print-based sanity checks that can be run on a GPU node.

### Dependencies
- Core: ColabDesign, AlphaFold2 (JAX), ProteinMPNN, OpenMM, FreeSASA, BioPython
- The `sc-rs` binary handles shape complementarity scoring (Rust, no PyRosetta needed)
- DSSP handled via `functions/dssp/` bundled binary

## Code Style
- Follow existing patterns: functions use snake_case, no type annotations in existing code
- Keep functions in their thematic module (structural analysis in biopython_utils, scoring in pr_alternative_utils, etc.)
- Log with `print()` — no logging framework is used
- Prefer explicit over clever — this is scientific code that must be auditable

## Context Management

### Compaction Instructions
Preserve: all file paths, error messages, modified file list, open TODOs, architectural decisions.

### Delegation Rules
- Any task reading >3 files: use Agent tool with a subagent
- Any bash output >100 lines: pipe through subagent, return summary only
- Log analysis: always delegate

## Simplification and Removal Policy
- **Output quality must never decrease.** Every simplification or removal must be verified to not alter pipeline outputs (CSV values, PDB files, design quality metrics).
- **Remove only truly dead code** — functions that are never called by the pipeline, variables that are set but never read. Verify with grep before deleting.
- **Consolidate redundant I/O** — If the same file is parsed or the same computation is done multiple times, refactor to compute once and share the result.
- **Do not remove reference implementations** — `functions/ipsae.py` is a standalone reference script; it is not imported by the pipeline and must not be deleted.
- **Do not remove safety checks** — Exception handlers, validation guards, and fallback defaults exist to keep the pipeline running on diverse inputs. Only remove them if you can prove the guarded condition is impossible.
