"""Validation shared by CLI, Python API, and policy optimization."""


def validate_grpo_sampling(config) -> None:
    """Reject proposal constraints whose normalized likelihood is unavailable."""
    fields = ('global_rmsd_range', 'global_e2e_range', 'global_rg_range', 'global_token_ranges')
    constrained = [name for name in fields if getattr(config, name, None)]
    if constrained:
        raise ValueError(
            'Range-constrained global proposals are not supported during GRPO: '
            'their rejection/clamping distribution is not represented by log_prob. '
            'Remove these settings: ' + ', '.join(constrained)
        )
