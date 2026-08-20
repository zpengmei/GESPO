from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).resolve().parent
README = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""

setup(
    name="gespo",
    version="0.1.0",
    description="Generative Enhanced Sampling by Policy Optimization",
    long_description=README,
    long_description_content_type="text/markdown",
    author="GESPO Contributors",
    packages=find_packages(exclude=("examples", "configs", "tests", "docs")),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "scipy",
        "torch",
        "mdtraj",
        "MDAnalysis", # NOTE: Only used for helix hinge tmem63 global metrics
        "pyyaml",
        "matplotlib",
        "scikit-learn",
        "tqdm",
        "requests",
    ],
    entry_points={
        "console_scripts": [
            "gespo=gespo.cli:main",
            "gespo-agent=gespo.runner_agent.cli:main",
        ],
    },
    extras_require={
        # OpenMM + ligand parameterization are best installed via conda-forge;
        # listed here for completeness (`pip install -e .[md]`).
        "md": ["openmm", "openmmforcefields", "openff-toolkit"],
        # geom2vec is a separate package that also ships the pretrained backbones.
        "geom2vec": ["geom2vec @ git+https://github.com/zpengmei/geom2vec.git"],
        "agent-tui": ["textual>=1.0", "rich>=13.0"],
    },
)
