"""Global-token-set and torsion-mode names for TorsionAR pretraining.

Kept deliberately free of heavy imports (no OpenMM / torch / mdtraj) so it is the
single source of truth for:

- the CLI ``--help`` listing (``gespo pretrain``),
- the docs token catalog (``docs/reference.md``),
- runtime validation inside ``pretrain_torsion_ar``.

Historically an unrecognized ``--global-token-set`` fell through *silently* to
the base ``rmsd_e2e`` globals — a typo produced the wrong tokens with no error.
``validate_global_token_set`` closes that footgun.
"""
from __future__ import annotations


# Every accepted ``--global-token-set`` spelling (canonical names + legacy/nm
# aliases). Grouped by the CV family they select.
KNOWN_GLOBAL_TOKEN_SETS = frozenset({
    # base: [rmsd_to_A, end-to-end distance]
    "rmsd_e2e",
    # + radius of gyration
    "rmsd_e2e_rg", "rmsd_e2e_rg_nm", "rg", "rg_nm",
    # RMSD to endpoint B (requires --pdb-b)
    "rmsd_ab", "rmsd_a_b", "rmsd_ab_nm", "rmsd_a_b_nm",
    "rmsd_ab_e2e", "rmsd_a_b_e2e", "rmsd_ab_e2e_nm", "rmsd_a_b_e2e_nm",
    "path_rmsd_ab", "path_rmsd_ab_e2e",
    # adenylate-kinase domain CVs
    "adk", "adk_domain", "adk_globals", "adk_thetas",
    "adk_domain4", "adk_domain_only", "adk_domain_no_rmsd", "adk_no_rmsd",
    "adk_domain4_rmsd_ab", "adk_domain4_rmsd_a_b",
    "adk_domain4_rmsd_ab_nm", "adk_domain4_rmsd_a_b_nm",
    # vinculin H5 bundle
    "vinculin_h5", "vinculin_h5_bundle", "h5_bundle",
    # protein-protein interface
    "ppi", "ppi_global", "ppi_globals", "ppi_tokens",
    "chain_com_dist_nm,min_interface_ca_dist_nm,q_interface",
    # ligand-distance tokens (with --ligand-smiles)
    "rmsd_e2e_ligdist", "rmsd_e2e_lig",
    # pocket tokens (holo --pdb, or apo --pdb + --pocket-ref-pdb)
    "rmsd_e2e_pocket", "rmsd_e2e_pocket3",
    "rmsd_e2e_pocket_contacts", "rmsd_e2e_pocket_full",
    # TMEM63 CVs
    "tmem63", # all nine TMEM63 tokens
    "tmem63_dists", # TM RMSD, avg latch distance, avg neck distance, il2-tm6b distance
    "tmem63_dists_only", # avg latch distance, avg neck distance, il2-tm6b distance
    "tmem63_hel_hinges", # TM-domain RMSD, TM6 hinge, TM3 hinge, TM5 hinge
    "tmem63_hel_hinges_only", # TM6 hinge, TM3 hinge, TM5 hinge
    "tmem63_hel_tilts", # TM-domain RMSD, TM6 angle, TM4 angle
    "tmem63_hel_tilts_only", # TM6 angle, TM4 angle

})

# The handful worth surfacing in --help (the rest are aliases/legacy spellings).
COMMON_GLOBAL_TOKEN_SETS = (
    "rmsd_e2e", "rmsd_e2e_rg", "rmsd_ab_e2e",
    "rmsd_e2e_pocket3", "rmsd_e2e_pocket_full", "rmsd_e2e_ligdist",
)


def normalize_global_token_set(name) -> str:
    """Normalize a ``--global-token-set`` value the way the tokenizer does."""
    return str(name or "rmsd_e2e").strip().lower()


def validate_global_token_set(name) -> str:
    """Return the normalized token-set name, or raise listing the valid names.

    Fixes a silent footgun: an unrecognized name used to fall through to the
    base ``rmsd_e2e`` tokens instead of erroring.
    """
    token_set = normalize_global_token_set(name)
    if (token_set in KNOWN_GLOBAL_TOKEN_SETS
            or token_set.replace(" ", "") in KNOWN_GLOBAL_TOKEN_SETS):
        return token_set
    valid = ", ".join(sorted(KNOWN_GLOBAL_TOKEN_SETS))
    raise ValueError(
        f"[TorsionAR Pretrain] Unknown --global-token-set '{name}'. "
        f"Valid token sets: {valid}"
    )


def normalize_torsion_mode(mode) -> str:
    """Canonicalize a torsion-mode value to what the tokenizer accepts.

    The tokenizer accepts only ``backbone`` (phi/psi/omega) and
    ``backbone+sidechain`` (adds chi1-chi4). ``backbone_chi`` is a legacy CLI
    spelling that used to parse but then raise ValueError deep in pretraining;
    map it here so both the CLI and the programmatic API accept it.
    """
    m = str(mode or "backbone").strip().lower()
    if m in ("backbone_chi", "backbone+chi", "backbone_sidechain"):
        return "backbone+sidechain"
    return m
