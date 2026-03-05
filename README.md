# FreeBindCraft

A modified [BindCraft v1.52](https://github.com/martinpacesa/BindCraft) for de novo protein binder design. FreeBindCraft adds an **optional PyRosetta bypass** (`--no-pyrosetta`) using fully open-source tools and a **PPIFlow-inspired affinity maturation** pipeline for iterative interface refinement.

Fork: [https://github.com/cytokineking/FreeBindCraft](https://github.com/cytokineking/FreeBindCraft)

For the original BindCraft pipeline, settings, and filter documentation, see the [original repository](https://github.com/martinpacesa/BindCraft) and [preprint](https://www.biorxiv.org/content/10.1101/2024.09.30.615802).

---

## Features

### PyRosetta Bypass (`--no-pyrosetta`)

Run the full design pipeline without a PyRosetta license:

| Component | PyRosetta Path | Bypass Path |
|---|---|---|
| **Relaxation** | FastRelax | OpenMM + PDBFixer + FASPR (GPU-accelerated, 2-4x faster) |
| **Shape Complementarity** | PyRosetta SC | [`sc-rs`](https://github.com/cytokineking/sc-rs) (nearly identical results) |
| **SASA** | PyRosetta SASA | [FreeSASA](https://github.com/mittinatten/freesasa) / Biopython Shrake-Rupley fallback |
| **Interface Analysis** | InterfaceAnalyzerMover | Biopython KD-tree + DSSP routines |

> Rosetta-specific metrics without open-source equivalents use placeholder values. Evaluate design quality accordingly.

### ipSAE Scoring

Interface predicted Structural Alignment Error ([Dunbrack et al. 2025](https://www.biorxiv.org/content/10.1101/2025.02.10.637595v1)) — a PTM-style scoring function with adaptive per-residue normalization. Higher scores = better predicted interface quality (range 0-1).

```bash
python bindcraft.py --settings ... --rank-by ipSAE
```

### Affinity Maturation (PPIFlow-Inspired)

Opt-in recursive interface refinement that improves binder affinity after the initial design passes filters. Enabled via `"enable_maturation": true` in advanced settings.

**How it works:**

1. **Quality assessment** — Score each interface residue against hard AND thresholds:
   - Per-residue REU <= -5.0
   - Per-residue pLDDT >= 0.85
   - Per-residue PAE <= 5.0
   - Secondary structure = helix or sheet (not loop/coil)
2. **Partition** — Residues passing ALL thresholds are marked "high quality" and fixed; the rest are redesignable
3. **Saturation scan** — Each fixed residue is tested with all candidate amino acids (GLY/CYS excluded) via MutateResidue + neighborhood repack (8A shell) + constrained backbone minimization. Scored by target-only interaction REU. Safeguards include H-bond preservation and post-scan dG sanity gating
4. **Re-hallucination** — AF2 redesigns unfixed positions (via `af_model.opt["fix_pos"]`) while frozen positions retain their identity
5. **MPNN + validation** — ProteinMPNN sequence optimization, AF2 prediction, relaxation, and re-scoring
6. **Convergence check** — Stops when improvement < threshold or all interface residues are fixed

The fixed set is **monotonically growing** — once a residue passes all quality filters, it stays fixed in subsequent rounds.

**Maturation settings** (in advanced settings JSON):

```json
{
    "enable_maturation": true,
    "maturation_max_rounds": 3,
    "maturation_improvement_metric": "i_pTM",
    "maturation_improvement_threshold": 0.01,
    "maturation_require_defined_ss": true,
    "maturation_reu_threshold": -5.0,
    "maturation_plddt_threshold": 0.85,
    "maturation_pae_threshold": 5.0,
    "maturation_contact_distance": 4.0,
    "maturation_scan_repack_shell": 8.0,
    "maturation_scan_exclude_aas": ["GLY", "CYS"],
    "maturation_scan_minimize": true,
    "maturation_scan_hbond_override_reu": -2.0,
    "maturation_max_fix_fraction": 0.8
}
```

A ready-to-use config is provided at `settings_advanced/default_4stage_multimer_maturation.json`.

> **Note:** Maturation requires PyRosetta for per-residue REU scoring and saturation scanning. It is not available with `--no-pyrosetta`.

---

## Installation

### Option 1: Conda (recommended)

```bash
git clone https://github.com/cytokineking/FreeBindCraft [install_folder]
cd [install_folder]
```

**Without PyRosetta:**
```bash
bash install_bindcraft.sh --cuda '12.4' --pkg_manager 'conda' --no-pyrosetta
```

**With PyRosetta** (license required for commercial use):
```bash
bash install_bindcraft.sh --cuda '12.4' --pkg_manager 'conda'
```

Script flags:
- `--cuda CUDAVERSION` — CUDA version (e.g., '12.4')
- `--pkg_manager MANAGER` — 'mamba' or 'conda' (default: 'conda')
- `--no-pyrosetta` — Install without PyRosetta
- `--fix-channels` — Fix conda channel config for dependency conflicts

### Option 2: pip (dependencies only)

```bash
pip install -r requirements.txt
```

> JAX/jaxlib GPU support and PyRosetta require conda. See `install_bindcraft.sh` for the full conda-based setup.

### Option 3: Docker

See [Containerized Usage](#containerized-usage-docker) below.

---

## Usage

```bash
conda activate BindCraft
cd /path/to/FreeBindCraft
```

**PyRosetta bypass:**
```bash
python -u ./bindcraft.py \
  --settings './settings_target/your_target.json' \
  --filters './settings_filters/default_filters.json' \
  --advanced './settings_advanced/default_4stage_multimer.json' \
  --no-pyrosetta
```

**With PyRosetta:**
```bash
python -u ./bindcraft.py \
  --settings './settings_target/your_target.json' \
  --filters './settings_filters/default_filters.json' \
  --advanced './settings_advanced/default_4stage_multimer.json'
```

**With maturation enabled:**
```bash
python -u ./bindcraft.py \
  --settings './settings_target/your_target.json' \
  --filters './settings_filters/default_filters.json' \
  --advanced './settings_advanced/default_4stage_multimer_maturation.json'
```

> Even if PyRosetta is installed, you can bypass it at runtime with `--no-pyrosetta`.

### Interactive CLI

Run without `--settings` (or with `--interactive`) for a guided setup wizard:

```bash
python bindcraft.py
python bindcraft.py --interactive
```

The wizard prompts for: project name, input PDB, output directory, design type (miniprotein/peptide), logging options, PyRosetta mode, and ranking method.

### CLI Flags

| Flag | Description |
|---|---|
| `--no-pyrosetta` | Use OpenMM/FreeSASA/sc-rs instead of PyRosetta |
| `--verbose` | Enable detailed timing/progress logs |
| `--debug-pdbs` | Write intermediate PDBs from OpenMM relax |
| `--no-plots` | Disable trajectory plots |
| `--no-animations` | Disable trajectory animations |
| `--rank-by {i_pTM,ipSAE}` | Metric to rank final designs (default: `i_pTM`) |
| `--interactive` | Force interactive setup mode |

---

## Containerized Usage (Docker)

### Prerequisites

- NVIDIA driver (`nvidia-smi`)
- Docker CE + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Verify GPU access:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 nvidia-smi
```

### Build

From the project root:

**Without PyRosetta:**
```bash
docker build -f docker/Dockerfile -t freebindcraft:gpu .
```

**With PyRosetta:**
```bash
docker build -f docker/Dockerfile --build-arg WITH_PYROSETTA=true -t freebindcraft:pyrosetta .
```

### Run

```bash
mkdir -p /path/on/host/run_outputs
docker run --gpus all --rm -it \
  --ulimit nofile=65536:65536 \
  -v /path/on/host/run_outputs:/root/software/pdl1 \
  freebindcraft:gpu \
  python bindcraft.py \
    --settings settings_target/PDL1.json \
    --filters settings_filters/default_filters.json \
    --advanced settings_advanced/default_4stage_multimer.json \
    --no-pyrosetta
```

Mount custom settings/PDBs as needed. Always set `--ulimit nofile=65536:65536`.

### Interactive Docker Wrapper

```bash
python docker/docker_cli.py
```

Guides you through image selection, GPU index, and all design parameters. Outputs persist on host via mounted directories.

---

## Project Structure

```
FreeBindCraft/
  bindcraft.py                  # Main pipeline (AF2 hallucination -> MPNN -> validation -> maturation)
  install_bindcraft.sh          # Conda-based installation script
  requirements.txt              # pip dependencies
  docker/
    Dockerfile                  # Container build
    docker-entrypoint.sh        # Container entrypoint
    docker_cli.py               # Interactive Docker wrapper
  functions/
    __init__.py                 # Wildcard imports for all modules
    colabdesign_utils.py        # AF2 design, MPNN, prediction, maturation hallucination
    biopython_utils.py          # Structural analysis, DSSP, KD-tree contacts
    pr_alternative_utils.py     # OpenMM relaxation, SASA, scoring (PyRosetta-free)
    pyrosetta_utils.py          # PyRosetta relaxation, scoring
    maturation_utils.py         # Quality assessment, residue partitioning, saturation scan
    generic_utils.py            # CSV labels, directory setup, filter functions
    logging_utils.py            # Verbose timing utilities
    dssp/                       # Bundled DSSP binary
  settings_target/              # Target PDB configuration files
  settings_filters/             # Design filter profiles
  settings_advanced/            # Advanced pipeline settings (incl. maturation configs)
  extras/                       # Analysis and utility scripts (see extras/README.md)
  technical_overview/           # In-depth technical documentation
  PPIFlow/                      # Reference implementation (not integrated into runtime)
```

---

## Citations & External Tools

- **BindCraft**: [Pacesa et al. 2024](https://www.biorxiv.org/content/10.1101/2024.09.30.615802)
- **ColabDesign / AlphaFold2**: [Jumper et al. 2021](https://doi.org/10.1038/s41586-021-03819-2)
- **ProteinMPNN**: [Dauparas et al. 2022](https://doi.org/10.1126/science.add2187)
- **sc-rs** (Shape Complementarity): [https://github.com/cytokineking/sc-rs](https://github.com/cytokineking/sc-rs)
- **FreeSASA**: [https://github.com/mittinatten/freesasa](https://github.com/mittinatten/freesasa)
- **FASPR** (Side-chain Packing): [https://github.com/tommyhuangthu/FASPR](https://github.com/tommyhuangthu/FASPR)
- **Biopython**: [https://biopython.org](https://biopython.org)
- **ipSAE**: [Dunbrack et al. 2025](https://www.biorxiv.org/content/10.1101/2025.02.10.637595v1)

## Known Issues

- **OpenMM GPU backend**: OpenCL is preferred over CUDA for OpenMM relax due to reliability. The relax step runs in a subprocess for isolation. Some JIT log spam (`Failed to read file: /tmp/dep-*.d`) may appear.

  Filter the noise with:
  ```bash
  python -u ./bindcraft.py ... --no-pyrosetta \
    |& stdbuf -oL -eL grep -v -E 'Failed to read file: .*/dep-[0-9a-fA-F]+\.d([[:space:]]*)?$' \
    | stdbuf -oL tee -a ./output/your_target.log
  ```

## Extras

Analysis and utility scripts in `extras/`:

- `analyze_bindcraft_rejections.py` — Quantify which filters cause most rejections
- `compare_interface_metrics_all.py` — Compare PyRosetta/FreeSASA/Biopython metrics across PDBs
- `rescore_accepted_with_rosetta.py` — Rescore `--no-pyrosetta` designs with PyRosetta
- `test_openmm_relax.py` — Test harness for OpenMM relax

See `extras/README.md` for detailed usage.

## Contributing

Pull requests are welcome.
