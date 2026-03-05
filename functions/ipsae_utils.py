"""
Interface scoring utilities for BindCraft

Implements ipSAE, pDockQ, pDockQ2, and LIS from ColabDesign PAE matrices
and predicted structures.

References:
  ipSAE:   Dunbrack. https://www.biorxiv.org/content/10.1101/2025.02.10.637595v1
  pDockQ:  Bryant, Pozotti, Elofsson. https://www.nature.com/articles/s41467-022-28865-w
  pDockQ2: Zhu, Shenoy, Kundrotas, Elofsson. https://academic.oup.com/bioinformatics/article/39/7/btad424/7219714
  LIS:     Kim, Hu, Comjean, Rodiger, Mohr, Perrimon. https://www.biorxiv.org/content/10.1101/2024.02.19.580970v1

Original ipSAE implementation by Roland Dunbrack, Fox Chase Cancer Center.
MIT license: script can be modified and redistributed for non-commercial
and commercial use, as long as this information is reproduced.
"""

import math
import numpy as np


def ptm_func(x, d0):
    """
    Calculate the PTM-style score for PAE values.

    This is the core transformation used in ipSAE/ipTM calculations.

    Args:
        x: PAE value(s) - can be scalar or numpy array
        d0: Normalization constant based on alignment length

    Returns:
        PTM-style score in range [0, 1]
    """
    return 1.0 / (1.0 + (x / d0) ** 2.0)


def calc_d0(L, pair_type='protein'):
    """
    Calculate d0 based on alignment length.

    From Yang and Skolnick, PROTEINS: Structure, Function, and
    Bioinformatics 57:702-710 (2004)

    Args:
        L: Number of residues in the alignment
        pair_type: 'protein' or 'nucleic_acid'

    Returns:
        d0 value (minimum 1.0 for protein, 2.0 for nucleic acid)
    """
    min_value = 2.0 if pair_type == 'nucleic_acid' else 1.0
    if L <= 27:
        return min_value
    d0 = 1.24 * (float(L) - 15) ** (1.0 / 3.0) - 1.8
    return max(min_value, d0)


def calculate_ipsae(pae_matrix, target_len, binder_len, pae_cutoff=10.0):
    """
    Calculate ipSAE score from ColabDesign PAE matrix.

    This implements the ipSAE_d0res calculation from the Dunbrack paper,
    which uses adaptive d0 based on the number of residues with good PAE
    values for each aligned residue.

    Args:
        pae_matrix: numpy array of shape (L, L) containing PAE values
                   where L = target_len + binder_len
                   Target residues are indices [0:target_len]
                   Binder residues are indices [target_len:target_len+binder_len]
        target_len: number of residues in target chain
        binder_len: number of residues in binder chain
        pae_cutoff: PAE cutoff for considering residue pairs (default 10.0 A)

    Returns:
        dict with ipSAE metrics:
            - 'ipSAE': max of binder and target direction scores (primary metric)
            - 'ipSAE_binder': max per-residue score from binder -> target
            - 'ipSAE_target': max per-residue score from target -> binder
            - 'n0dom': number of residues with good PAE values
    """
    total_len = target_len + binder_len

    if pae_matrix.shape[0] != total_len or pae_matrix.shape[1] != total_len:
        raise ValueError(f"PAE matrix shape {pae_matrix.shape} does not match "
                        f"target_len ({target_len}) + binder_len ({binder_len}) = {total_len}")

    # Interface PAE: binder rows -> target columns
    interface_pae = pae_matrix[target_len:, :target_len]
    valid_mask = interface_pae < pae_cutoff

    binder_good_residues = np.any(valid_mask, axis=1).sum()
    target_good_residues = np.any(valid_mask, axis=0).sum()
    n0dom = int(binder_good_residues + target_good_residues)

    # Per-binder-residue ipSAE scores (ipSAE_d0res)
    ipsae_byres = []
    for i in range(binder_len):
        valid = valid_mask[i]
        if valid.any():
            d0res = calc_d0(valid.sum())
            ipsae_byres.append(ptm_func(interface_pae[i][valid], d0res).mean())
        else:
            ipsae_byres.append(0.0)

    ipsae_byres = np.array(ipsae_byres)
    ipsae_binder_max = float(ipsae_byres.max()) if len(ipsae_byres) > 0 else 0.0

    # Reverse direction: target rows -> binder columns
    interface_pae_rev = pae_matrix[:target_len, target_len:]
    valid_mask_rev = interface_pae_rev < pae_cutoff

    ipsae_byres_rev = []
    for i in range(target_len):
        valid = valid_mask_rev[i]
        if valid.any():
            d0res = calc_d0(valid.sum())
            ipsae_byres_rev.append(ptm_func(interface_pae_rev[i][valid], d0res).mean())
        else:
            ipsae_byres_rev.append(0.0)

    ipsae_byres_rev = np.array(ipsae_byres_rev)
    ipsae_target_max = float(ipsae_byres_rev.max()) if len(ipsae_byres_rev) > 0 else 0.0

    ipsae = max(ipsae_binder_max, ipsae_target_max)

    return {
        'ipSAE': round(ipsae, 4),
        'ipSAE_binder': round(ipsae_binder_max, 4),
        'ipSAE_target': round(ipsae_target_max, 4),
        'n0dom': n0dom,
    }


def _parse_cb_coordinates(pdb_path):
    """
    Extract one Cb coordinate per residue (Ca for GLY) from a PDB file.

    Returns numpy array of shape (n_residues, 3).
    """
    residue_coords = {}
    residue_order = []

    with open(pdb_path, 'r') as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            atom_name = line[12:16].strip()
            chain_id = line[21]
            res_seq = int(line[22:26])
            coord = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])

            key = (chain_id, res_seq)
            if key not in residue_coords:
                residue_order.append(key)

            # Prefer CB; use CA only if no CB seen yet (GLY has no CB)
            if atom_name == 'CB':
                residue_coords[key] = coord
            elif atom_name == 'CA' and key not in residue_coords:
                residue_coords[key] = coord

    return np.array([residue_coords[key] for key in residue_order])


def calculate_contact_scores(pdb_path, pae_matrix, plddt_array, target_len, binder_len, contact_dist=8.0):
    """
    Calculate pDockQ, pDockQ2, and LIS in a single pass.

    Parses the PDB once and computes the contact/distance matrix once,
    then derives all three scores from the shared data.

    Args:
        pdb_path: path to predicted complex PDB
        pae_matrix: full PAE matrix (total_len x total_len)
        plddt_array: per-residue pLDDT array (0-1 scale from ColabDesign)
        target_len: number of target residues
        binder_len: number of binder residues
        contact_dist: Cb distance cutoff for interface contacts (default 8.0 A)

    Returns:
        dict with 'pDockQ', 'pDockQ2', 'LIS' scores (all float, 0-1)
    """
    total_len = target_len + binder_len

    if pae_matrix.shape[0] != total_len or pae_matrix.shape[1] != total_len:
        raise ValueError(f"PAE matrix shape {pae_matrix.shape} does not match "
                        f"target_len ({target_len}) + binder_len ({binder_len}) = {total_len}")

    if len(plddt_array) < total_len:
        raise ValueError(f"pLDDT array length {len(plddt_array)} < target_len + binder_len = {total_len}")

    # --- LIS (PAE-only, no PDB needed) ---
    # Compute each direction separately, then average (equal-weighted per reference)
    interface_fwd = pae_matrix[target_len:total_len, :target_len].flatten()
    interface_rev = pae_matrix[:target_len, target_len:total_len].flatten()
    valid_fwd = interface_fwd[interface_fwd < 12.0]
    valid_rev = interface_rev[interface_rev < 12.0]
    lis_fwd = float(np.mean((12.0 - valid_fwd) / 12.0)) if valid_fwd.size > 0 else 0.0
    lis_rev = float(np.mean((12.0 - valid_rev) / 12.0)) if valid_rev.size > 0 else 0.0
    if valid_fwd.size > 0 and valid_rev.size > 0:
        lis = round((lis_fwd + lis_rev) / 2.0, 4)
    elif valid_fwd.size > 0 or valid_rev.size > 0:
        lis = round(lis_fwd + lis_rev, 4)  # one is 0.0
    else:
        lis = 0.0

    # --- Contact-based scores (pDockQ, pDockQ2) ---
    cb_coords = _parse_cb_coordinates(pdb_path)
    if len(cb_coords) < total_len:
        return {'pDockQ': 0.0, 'pDockQ2': 0.0, 'LIS': lis}

    target_coords = cb_coords[:target_len]
    binder_coords = cb_coords[target_len:total_len]

    diff = target_coords[:, None, :] - binder_coords[None, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=2))  # (target_len, binder_len)
    contacts = dist_matrix < contact_dist
    npairs = int(contacts.sum())

    if npairs == 0:
        return {'pDockQ': 0.0, 'pDockQ2': 0.0, 'LIS': lis}

    # Shared: interface residues and scaled pLDDT
    target_contact_idx = np.where(contacts.any(axis=1))[0]
    binder_contact_idx = np.where(contacts.any(axis=0))[0] + target_len
    interface_idx = np.concatenate([target_contact_idx, binder_contact_idx])

    plddt_slice = plddt_array[:total_len]
    plddt_100 = plddt_slice * 100.0 if plddt_slice.max() <= 1.0 else plddt_slice
    mean_plddt = float(plddt_100[interface_idx].mean())

    # pDockQ
    x_pdockq = mean_plddt * math.log10(npairs)
    pdockq = round(0.724 / (1.0 + math.exp(-0.052 * (x_pdockq - 152.611))) + 0.018, 4)

    # pDockQ2 — compute both PAE directions and take max (per Zhu et al.)
    def _pdockq2_one_direction(row_start, col_offset, contact_matrix):
        row_idx, col_idx = np.where(contact_matrix)
        if len(row_idx) == 0:
            return 0.0
        pae_vals = pae_matrix[row_start + row_idx, col_offset + col_idx]
        mean_ptm = float(ptm_func(pae_vals, 10.0).mean())
        x = mean_plddt * mean_ptm
        return 1.31 / (1.0 + math.exp(-0.075 * (x - 84.733))) + 0.005

    # target->binder direction: contact rows=target, cols=binder
    pdockq2_fwd = _pdockq2_one_direction(0, target_len, contacts)
    # binder->target direction: contact rows=binder, cols=target
    pdockq2_rev = _pdockq2_one_direction(target_len, 0, contacts.T)
    pdockq2 = round(max(pdockq2_fwd, pdockq2_rev), 4)

    return {'pDockQ': pdockq, 'pDockQ2': pdockq2, 'LIS': lis}
