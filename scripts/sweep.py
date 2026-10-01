"""Hyperparameter sweep launcher: runs DQN, DDQN, PPO with grid of configs.
Organizes results by algorithm and config, saves experiment summary.
"""
import os
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


SWEEP_CONFIG = {
    'dqn': {
        'script': 'dqn.py',
        'base_args': '--episodes 100 --save-every 10 --log-interval 5',
        'grid': {
            'lr': [1e-3, 5e-4],
            'eps_decay': [0.995, 0.99],
            'target_update': [500, 2000],
        }
    },
    'ddqn': {
        'script': 'ddqn.py',
        'base_args': '--episodes 100 --save-every 10 --log-interval 5',
        'grid': {
            'lr': [1e-3, 5e-4],
            'eps_decay': [0.995, 0.99],
            'update': ['soft', 'hard'],
        }
    },
    'ppo': {
        'script': 'ppo.py',
        'base_args': '--max-iters 50 --steps-per-epoch 512 --eval-eps 5 --save-every 5',
        'grid': {
            'clip_ratio': [0.1, 0.2],
            'gae_lambda': [0.95, 0.97],
        }
    }
}


def generate_configs(grid):
    """Generate all combinations from a hyperparameter grid."""
    keys, values = zip(*grid.items())
    import itertools
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def run_experiment(algo, script, base_args, config, run_dir):
    """Run a single experiment and return exit code."""
    config_str = '_'.join([f'{k}={v}' for k, v in config.items()])
    exp_dir = run_dir / algo / config_str
    exp_dir.mkdir(parents=True, exist_ok=True)

    config_args = ' '.join([f'--{k.replace("_", "-")} {v}' for k, v in config.items()])
    model_dir = exp_dir / 'models'
    log_dir = exp_dir / 'logs'
    figures_dir = exp_dir / 'figures'

    # Use absolute path to venv, with proper quoting
    scripts_dir = Path(__file__).parent.resolve()
    venv_dir = scripts_dir.parent / '.venv'
    cmd = f'source "{venv_dir}/bin/activate" && cd "{scripts_dir}" && python {script} {base_args} {config_args} --model-dir "{model_dir}" --log-dir "{log_dir}" --figures-dir "{figures_dir}"'

    print(f'\n[{algo.upper()}] {config_str}')
    print(f'  Running: {cmd}')
    start = time.time()
    result = subprocess.run(cmd, shell=True, cwd=str(scripts_dir), capture_output=False, executable='/bin/bash')
    elapsed = time.time() - start

    status = 'PASS' if result.returncode == 0 else 'FAIL'
    print(f'  Status: {status} ({elapsed:.1f}s)')
    return result.returncode, exp_dir


def main():
    run_dir = Path('../experiments') / datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f'Sweep results directory: {run_dir}')

    summary = {'timestamp': datetime.now().isoformat(), 'experiments': []}
    os.chdir(os.path.dirname(__file__))

    for algo, algo_config in SWEEP_CONFIG.items():
        script = algo_config['script']
        base_args = algo_config['base_args']
        grid = algo_config['grid']

        print(f'\n\n=== {algo.upper()} ===')
        for config in generate_configs(grid):
            rc, exp_dir = run_experiment(algo, script, base_args, config, run_dir)
            summary['experiments'].append({
                'algo': algo,
                'config': config,
                'exit_code': rc,
                'exp_dir': str(exp_dir),
            })

    # Save experiment summary
    summary_file = run_dir / 'summary.json'
    with open(summary_file, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f'\n\nExperiment summary saved to: {summary_file}')


if __name__ == '__main__':
    main()
