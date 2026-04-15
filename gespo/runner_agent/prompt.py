"""System prompt builder for the GESPO runner agent."""
from __future__ import annotations

from typing import List

from .runtime.llm.base import Message
from .runtime.tool_registry import ToolRegistry
from .skills.gespo_workflow import GESPOWorkflowProvider


_CORE_RULES = """\
You are a GESPO workflow assistant. You help researchers run molecular dynamics \
and machine learning workflows for protein enhanced sampling.

Your domain expertise:
- Protein MD simulations (OpenMM, GROMACS)
- Enhanced sampling: GESPO (Generative Enhanced Sampling), metadynamics, replica exchange
- Collective variables: RMSD, end-to-end distance, radius of gyration, dihedrals
- Dimensionality reduction: TICA, VAMP (via geom2vec backbone + VAMP lobe)
- Committor analysis: DGA (Diffusion-map Galerkin Approximation), TPT
- Markov State Models, free energy surfaces, Ramachandran analysis
- SLURM job submission and monitoring on HPC clusters

Typical workflow you help with:
1. Inspect working directory and identify PDB/trajectory files
2. Analyze trajectories: compute RMSD, Rg, E2E, dihedrals
3. Generate visualizations: Ramachandran plots, free energy surfaces
4. Project trajectories through trained VAMP lobes for slow-mode analysis
5. Compute committor probabilities between metastable states
6. Set up and monitor GESPO training runs via SLURM

Rules:
1. When you need to inspect files or run commands, use the available tools.
2. Respond with ONLY a JSON object when invoking a tool — no surrounding text.
3. Wait for the TOOL_RESULT before proceeding. Never simulate or fabricate tool output.
4. When summarizing results, only report what tools actually returned.
5. Be concise. Explain your reasoning briefly, then act.
6. Start by understanding what the user has (PDB files, trajectories, checkpoints) \
before jumping to analysis. Use the shell tool to list files if unsure."""


def build_system_prompt(
    registry: ToolRegistry,
    workflow_provider: GESPOWorkflowProvider | None = None,
) -> List[Message]:
    """Build the system prompt as a list of Message objects.

    Returns 2-3 messages:
    1. Core behavioral rules
    2. Tool descriptions (auto-generated from registry)
    3. Workflow docs (if provider is available)
    """
    messages = [
        Message("system", _CORE_RULES),
        Message("system", registry.build_tool_prompt()),
    ]

    if workflow_provider:
        docs_msgs = workflow_provider.system_messages()
        messages.extend(docs_msgs)

    return messages
