"""Dataset-only OpenPI integration. Never creates or runs a VLA model."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent.parent


def bootstrap():
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['OPENPI_DATA_HOME'] = str(ROOT/'work/openpi_cache')
    os.environ['HF_LEROBOT_HOME'] = str(ROOT/'work/openpi_lerobot_home')
    os.environ['HF_DATASETS_CACHE'] = str(ROOT/'work/openpi_hf_datasets')
    os.environ['HF_HOME'] = str(ROOT/'work/openpi_hf_home')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    sys.path[:0] = [str(ROOT/'work/openpi/src'),
                   str(ROOT/'work/openpi/packages/openpi-client/src'),
                   str(ROOT/'work/lerobot-openpi')]


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(2*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dataset_hashes():
    paths = ['red_then_blue_v1', 'red_then_blue_v1_openpi_v21',
             'red_then_blue_v1_openpi_splits']
    return {str(f.relative_to(ROOT)): file_hash(f)
            for name in paths for f in sorted((ROOT/'outputs/datasets'/name).rglob('*')) if f.is_file()}


def setup_local_data():
    for split in ['train', 'val', 'test']:
        target = ROOT/'outputs/datasets/red_then_blue_v1_openpi_splits'/split
        alias = Path(os.environ['HF_LEROBOT_HOME'])/'local'/f'so101_{split}'
        alias.parent.mkdir(parents=True, exist_ok=True)
        if alias.is_symlink():
            if alias.resolve() != target.resolve():
                raise ValueError(f'Existing dataset alias points elsewhere: {alias}')
        elif alias.exists():
            raise ValueError(f'Unexpected non-symlink dataset alias: {alias}')
        else:
            alias.symlink_to(target, target_is_directory=True)
    tokenizer = Path(os.environ['OPENPI_DATA_HOME'])/'big_vision/paligemma_tokenizer.model'
    expected_size = 4264023
    expected_md5 = '1420adc9856720a559e8a87284b195e2'
    def tokenizer_ok(path):
        return path.exists() and path.stat().st_size == expected_size and hashlib.md5(path.read_bytes()).hexdigest() == expected_md5
    if not tokenizer_ok(tokenizer):
        print('Downloading official PaliGemma tokenizer only; no VLA weights.', flush=True)
        tokenizer.parent.mkdir(parents=True, exist_ok=True)
        tmp = tokenizer.with_suffix('.partial')
        if tokenizer.exists():
            tokenizer.replace(tmp)
        for attempt in range(5):
            offset = tmp.stat().st_size if tmp.exists() else 0
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            request = urllib.request.Request('https://storage.googleapis.com/big_vision/paligemma_tokenizer.model', headers=headers)
            with urllib.request.urlopen(request, timeout=90) as response:
                append = offset > 0 and response.status == 206
                with tmp.open('ab' if append else 'wb') as out:
                    while chunk := response.read(1024*1024):
                        out.write(chunk)
            if tokenizer_ok(tmp):
                break
            print(f'Tokenizer download incomplete; retry {attempt+1}/5.', flush=True)
        if not tokenizer_ok(tmp):
            raise RuntimeError('Official tokenizer size/MD5 verification failed')
        tmp.rename(tokenizer)
    return tokenizer


def compute_train_stats(config, report_dir):
    import numpy as np
    import pyarrow.parquet as pq
    from openpi.shared import normalize
    from so101_openpi import ACTION_HORIZON
    train = ROOT/'outputs/datasets/red_then_blue_v1_openpi_splits/train'
    states, actions = normalize.RunningStats(), normalize.RunningStats()
    state_count = 0
    inputs = []
    for path in sorted((train/'data').rglob('*.parquet')):
        table = pq.read_table(path)
        s = np.stack(table['observation.state'].to_pylist()).astype(np.float64)
        a = np.stack(table['action'].to_pylist()).astype(np.float64)
        # Matches LeRobot delta_timestamps action windows, including repeated-last
        # padding at episode tails. No image decoding is needed for numeric stats.
        windows = a[np.minimum(np.arange(len(a))[:, None] + np.arange(ACTION_HORIZON), len(a)-1)]
        states.update(s)
        actions.update(windows)
        state_count += len(s)
        inputs.append({'path': str(path.relative_to(ROOT)), 'sha256': file_hash(path)})
    stats = {'state': states.get_statistics(), 'actions': actions.get_statistics()}
    for key, st in stats.items():
        if not np.isfinite(st.mean).all() or not np.isfinite(st.std).all():
            raise ValueError(f'Invalid stats: {key}')
        if np.any(st.q99-st.q01 < 1e-6):
            raise ValueError(f'Degenerate quantile range: {key}')
    destination = config.assets_dirs/'so101'
    normalize.save(destination, stats)
    provenance = dict(source_split='train', episodes=100, state_vectors=state_count,
        action_vectors=state_count*ACTION_HORIZON, action_horizon=ACTION_HORIZON,
        action_representation='absolute degrees for 5 joints; gripper calibrated 0..100 units',
        quantile_normalization=True, validation_test_used=False,
        tail_handling='repeat final action within the same episode', inputs=inputs,
        norm_stats=str(destination/'norm_stats.json'), norm_stats_sha256=file_hash(destination/'norm_stats.json'))
    (report_dir/'norm_stats_provenance.json').write_text(json.dumps(provenance, indent=2))
    print(f'Train-only normalization: {state_count} state vectors, {state_count*ACTION_HORIZON} action vectors.', flush=True)
    return stats, provenance


def describe(value):
    import numpy as np
    arr = np.asarray(value)
    result = dict(shape=list(arr.shape), dtype=str(arr.dtype))
    if arr.size and np.issubdtype(arr.dtype, np.number):
        result.update(min=float(arr.min()), max=float(arr.max()), finite=bool(np.isfinite(arr).all()))
    elif arr.dtype == bool:
        result['true_count'] = int(arr.sum())
    return result


def check_split(split, config, stats_hash, batches):
    import numpy as np
    from openpi.training import data_loader
    from openpi import transforms
    from so101_openpi import ACTION_HORIZON
    dc = config.data.create(config.assets_dirs, config.model)
    assert file_hash(config.assets_dirs/'so101/norm_stats.json') == stats_hash
    assert dc.use_quantile_norm
    dataset = data_loader.create_torch_dataset(dc, ACTION_HORIZON, config.model)
    adapted = data_loader.transform_dataset(dataset, dc)
    root = ROOT/'outputs/datasets/red_then_blue_v1_openpi_splits'/split
    episodes = [json.loads(line) for line in (root/'meta/episodes.jsonl').read_text().splitlines()]
    # Check both boundaries of every episode through the actual OpenPI transforms.
    offset = 0
    boundaries = 0
    for ep in episodes:
        for i in [offset, offset+ep['length']-1]:
            raw = dataset[i]
            # Check action chunks at episode edges without applying normalization.
            import pyarrow.parquet as pq
            path = root/f'data/chunk-000/episode_{ep["episode_index"]:06d}.parquet'
            all_actions = np.stack(pq.read_table(path, columns=['action'])['action'].to_pylist())
            local = i-offset
            expected = all_actions[np.minimum(np.arange(ACTION_HORIZON)+local, len(all_actions)-1)]
            assert np.allclose(np.asarray(raw['action']), expected, atol=1e-5, rtol=0)
            item = adapted[i]
            assert item['state'].shape == (32,)
            assert item['actions'].shape == (50, 32)
            assert item['tokenized_prompt'].shape == (200,)
            assert item['tokenized_prompt_mask'].shape == (200,)
            for key in ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']:
                assert item['image'][key].shape == (224, 224, 3)
            assert bool(item['image_mask']['base_0_rgb']) and bool(item['image_mask']['left_wrist_0_rgb'])
            assert not bool(item['image_mask']['right_wrist_0_rgb'])
            assert np.all(item['state'][6:] == 0) and np.all(item['actions'][:, 6:] == 0)
            assert np.isfinite(item['state']).all() and np.isfinite(item['actions']).all()
            # Quantile normalize/unnormalize must preserve the six raw values.
            repacked = transforms.compose(dc.repack_transforms.inputs + dc.data_transforms.inputs)(raw)
            normed = transforms.Normalize(dc.norm_stats, use_quantiles=True)(repacked)
            restored = transforms.Unnormalize(dc.norm_stats, use_quantiles=True)(normed)
            assert np.allclose(restored['state'], np.asarray(raw['observation.state']), atol=2e-5, rtol=0)
            assert np.allclose(restored['actions'], np.asarray(raw['action']), atol=2e-5, rtol=0)
            boundaries += 1
            if boundaries % 40 == 0:
                print(f'{split}: {boundaries}/{len(episodes)*2} episode boundaries checked.', flush=True)
        offset += ep['length']
    # This is the actual OpenPI loader factory, without a replacement or mock.
    loader = data_loader.create_data_loader(config, shuffle=(split == 'train'),
                                           num_batches=batches, framework='jax')
    batch_reports = []
    for obs, actions in loader:
        b = config.batch_size
        assert obs.state.shape == (b, 32) and actions.shape == (b, 50, 32)
        assert obs.tokenized_prompt.shape == (b, 200)
        assert np.isfinite(np.asarray(obs.state)).all() and np.isfinite(np.asarray(actions)).all()
        images = {k: describe(v) for k, v in obs.images.items()}
        for key, value in obs.images.items():
            assert value.shape == (b, 224, 224, 3)
            assert float(value.min()) >= -1 and float(value.max()) <= 1
        batch_reports.append(dict(images=images, image_masks={k: describe(v) for k,v in obs.image_masks.items()},
            state=describe(obs.state), actions=describe(actions),
            tokenized_prompt=describe(obs.tokenized_prompt), prompt_mask=describe(obs.tokenized_prompt_mask)))
    print(f'{split}: {boundaries} episode boundaries + {len(batch_reports)} actual OpenPI batches passed.', flush=True)
    return dict(split=split, episodes=len(episodes), frames=offset, boundary_samples_checked=boundaries,
        norm_stats_sha256=stats_hash, action_chunk_tail_checks_passed=True,
        normalization_round_trip_passed=True, actual_openpi_loader='openpi.training.data_loader.create_data_loader',
        batches=batch_reports)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--batches', type=int, default=2)
    args = p.parse_args()
    bootstrap()
    report_dir = ROOT/'outputs/checks/openpi_pipeline_140'
    report_dir.mkdir(parents=True, exist_ok=True)
    before = dataset_hashes()
    (report_dir/'datasets_sha256_before.json').write_text(json.dumps(before, indent=2))
    tokenizer = setup_local_data()
    from so101_openpi import make_config
    assets = ROOT/'outputs/openpi_assets'
    train_config = make_config('train', assets, args.batch_size)
    stats, provenance = compute_train_stats(train_config, report_dir)
    results = []
    for split in ['train', 'val', 'test']:
        cfg = make_config(split, assets, args.batch_size)
        results.append(check_split(split, cfg, provenance['norm_stats_sha256'], args.batches))
    after = dataset_hashes()
    (report_dir/'datasets_sha256_after.json').write_text(json.dumps(after, indent=2))
    assert before == after, 'A dataset file changed during preprocessing checks'
    import jax
    import openpi.training.data_loader as actual_loader
    report = dict(status='passed', scope='dataset processing only',
        actual_loader_source=actual_loader.__file__, jax_devices=[str(d) for d in jax.devices()],
        batch_size=args.batch_size, splits=results, training_normalization=provenance,
        tokenizer=dict(path=str(tokenizer), sha256=file_hash(tokenizer)), datasets_unchanged=True,
        model_weights_downloaded=False, model_instantiated=False, model_forward_performed=False,
        fine_tuning_performed=False, excluded_by_user=['model forward', 'fine-tuning'])
    (report_dir/'report.json').write_text(json.dumps(report, indent=2))
    print('PASS: actual OpenPI data pipeline; datasets unchanged; no model forward or fine-tuning.', flush=True)


if __name__ == '__main__':
    main()
