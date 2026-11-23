"""Evaluate the selected SO-101 checkpoint without restoring an optimizer."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=Path('/root/autodl-tmp/so101'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--batch-size', type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument('--smoke-only', action='store_true',
                        help='Use one validation batch, leaving the test set untouched.')
    args = parser.parse_args()
    root = args.root.resolve()
    checkpoint = (args.checkpoint or root / 'checkpoints/pi05_so101_data/'
                  'formal6000_b16_20261118_134620_14830/7000').resolve()
    sys.path[:0] = [str(root), str(root / 'scripts')]
    for key, path in {'HF_LEROBOT_HOME': root / 'cache/lerobot',
                      'HF_HOME': root / 'cache/huggingface',
                      'OPENPI_DATA_HOME': root / 'cache/openpi'}.items():
        os.environ[key] = str(path)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('OMP_NUM_THREADS', '2')

    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from openpi.models import model as model_lib
    from openpi.training import data_loader
    from so101_openpi import make_config
    from training_config import make_training_config

    assert any(d.platform == 'gpu' for d in jax.devices()), 'GPU required'
    assert (checkpoint / '_CHECKPOINT_METADATA').is_file(), checkpoint
    norm_path = checkpoint / 'assets/so101/norm_stats.json'
    norm = json.loads(norm_path.read_text())
    assert norm == json.loads((root / 'openpi_assets/pi05_so101_data/so101/norm_stats.json').read_text())
    split = 'val' if args.smoke_only else 'test'
    dataset = root / 'datasets/red_then_blue_v1_openpi_splits' / split
    info = json.loads((dataset / 'meta/info.json').read_text())
    assert info['total_episodes'] == 20 and info['total_frames'] == 8000
    alias = root / 'cache/lerobot/local' / f'so101_{split}'
    alias.parent.mkdir(parents=True, exist_ok=True)
    if alias.exists() or alias.is_symlink():
        assert alias.resolve() == dataset.resolve(), f'Alias conflict: {alias}'
    else:
        alias.symlink_to(dataset, target_is_directory=True)
    cfg = make_training_config(root, args.batch_size, 'evaluate7000')
    cfg = dataclasses.replace(cfg, num_workers=0,
        data=make_config(split, root / 'openpi_assets', args.batch_size).data)
    count = 1 if args.smoke_only else 8000 // args.batch_size
    loader = data_loader.create_data_loader(cfg, shuffle=False, num_batches=count)
    assert loader.data_config().norm_stats is not None
    report_dir = root / 'reports' / ('eval7000_' + split + '_' +
                 time.strftime('%Y%m%d_%H%M%S') + '_' + str(os.getpid()))
    report_dir.mkdir(parents=True, exist_ok=False)
    print('REPORT DIRECTORY:', report_dir, flush=True)
    print('Loading model parameters only; optimizer is not loaded.', flush=True)
    # Match the official inference loader's bfloat16 parameter conversion.
    params = model_lib.restore_params(checkpoint / 'params', dtype=jnp.bfloat16)
    model = cfg.model.load(params, remove_extra_params=False)
    del params
    graph, state = nnx.split(model)
    del model

    @jax.jit
    def loss_fn(weights, batch, key):
        losses = nnx.merge(graph, weights).compute_loss(key, *batch, train=False)
        return jnp.sum(losses), jnp.asarray(losses.size, dtype=jnp.int32)

    @jax.jit
    def sample_fn(weights, observation):
        return nnx.merge(graph, weights).sample_actions(
            jax.random.key(123), observation, num_steps=10)

    total, elements, frames = 0.0, 0, 0
    started = time.perf_counter()
    with (report_dir / 'batch_metrics.jsonl').open('w', buffering=1) as log:
        for index, batch in enumerate(loader):
            if index == 0:
                print('First model compilation may take several minutes.', flush=True)
                tick = time.perf_counter()
                actions = np.asarray(sample_fn(state, batch[0]).block_until_ready())
                assert actions.shape == (args.batch_size, 50, 32)
                assert np.isfinite(actions).all()
                sample_seconds = time.perf_counter() - tick
                np.save(report_dir / 'first_normalized_actions.npy', actions)
                print('REAL ACTION SAMPLING PASSED:', actions.shape,
                      '(normalized; no robot commands sent)', flush=True)
                # Release inference executables before loss compilation.
                sample_fn.clear_cache()
            summed, size = jax.device_get(loss_fn(
                state, batch, jax.random.fold_in(jax.random.key(424242), index)))
            assert np.isfinite(summed)
            total += float(summed)
            elements += int(size)
            frames += int(batch[0].state.shape[0])
            log.write(json.dumps({'batch': index + 1, 'loss_sum': float(summed),
                                  'loss_elements': int(size)}) + '\n')
            if index == 0 or (index + 1) % 100 == 0 or index + 1 == count:
                print(f'EVAL {index + 1}/{count}; frames={frames}; '
                      f'loss={total / elements:.6f}; '
                      f'elapsed={(time.perf_counter() - started) / 60:.1f} min', flush=True)
    assert frames == count * args.batch_size and elements > 0
    report = dict(status='passed', checkpoint=str(checkpoint), split=split,
        full_test_evaluated=not args.smoke_only, frames=frames,
        episodes=20 if not args.smoke_only else None, batch_size=args.batch_size,
        batches=count, flow_matching_loss=total / elements, seed=424242,
        parameters_cast_to='bfloat16', action_sampling_passed=True,
        first_sampling_seconds_including_compilation=sample_seconds,
        action_sample_shape=list(actions.shape), robot_commands_sent=False,
        optimizer_loaded=False, norm_stats_sha256=hashlib.sha256(norm_path.read_bytes()).hexdigest(),
        seconds=time.perf_counter() - started,
        note='Offline loss is not task success rate. Batch size and bfloat16 '
             'conversion differ from training validation; losses are not strictly identical protocols.')
    (report_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    print('REPORT:', report_dir / 'report.json', flush=True)


if __name__ == '__main__':
    main()
