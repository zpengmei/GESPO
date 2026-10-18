"""Exercise the shipped shell recipe without launching expensive simulations."""
import json
import os
from pathlib import Path
import subprocess


def test_round2_separates_biased_seeds_from_unbiased_history(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    first = tmp_path / 'runs/run_01/run_snaps'
    (first / 'epoch_0001').mkdir(parents=True)
    (first / 'epoch_0002').mkdir()
    (first / 'simulation_topology.pdb').write_text('saved atom ordering\n')
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    executable = bin_dir / 'python'
    executable.write_text('#!/usr/bin/env python3\n'
                          'import json, os, sys\n'
                          'with open(os.environ["CALL_LOG"], "a") as f: '
                          'f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    executable.chmod(0o755)
    log = tmp_path / 'calls.jsonl'
    env = dict(os.environ, PATH=f'{bin_dir}:{os.environ["PATH"]}',
               CALL_LOG=str(log), GESPO_REPO=str(repo))
    subprocess.run(['bash', str(repo / 'examples/protein_ligand_trypsin/04_train_r2.sh')],
                   cwd=tmp_path, env=env, check=True)
    pretrain, train = [json.loads(line) for line in log.read_text().splitlines()]
    def option(args, flag):
        return args[args.index(flag) + 1]
    assert 'metad_seeds' in option(pretrain, '--source')
    assert 'epoch_0001' in option(pretrain, '--source')
    assert 'epoch_' not in option(train, '--replay-only-dcd-dir')
    assert 'metad_seeds' not in option(train, '--history-dcd-dir')
    assert 'epoch_0001' in option(train, '--history-dcd-dir')
    assert 'epoch_0002' in option(train, '--history-dcd-dir')
    assert option(pretrain, '--pdb') == option(train, '--pdb') == 'runs/run_01/run_snaps/simulation_topology.pdb'
    assert (tmp_path / 'runs/run_02/run_snaps/simulation_topology.pdb').read_text() == 'saved atom ordering\n'
