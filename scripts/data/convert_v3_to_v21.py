"""Export finalized LeRobot v3 episodes to OpenPI's pinned LeRobot v2.1.

Keeps joint values and timestamps unchanged. Videos are decoded and trimmed by
frame index, then re-encoded as H.264; source data is never modified.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, list):
        return [plain(x) for x in value]
    if isinstance(value, np.generic):
        return value.item()
    return value

def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--limit', type=int, help='Smoke test only: export first N episodes')
    args = p.parse_args()
    src, dst = args.source.resolve(), args.destination.resolve()
    info = json.loads((src / 'meta/info.json').read_text())
    if info['codebase_version'] != 'v3.0':
        p.error('Expected a finalized v3.0 source dataset')
    if dst.exists() or src == dst or src in dst.parents:
        p.error('Destination must be new and outside the source')
    eps = pd.concat([pd.read_parquet(x) for x in sorted((src / 'meta/episodes').rglob('*.parquet'))])
    eps = eps.sort_values('episode_index')
    if args.limit:
        eps = eps.iloc[:args.limit]
    if not len(eps) or eps.episode_index.tolist() != list(range(len(eps))):
        p.error('Expected nonempty contiguous episodes starting at zero')
    fps = info['fps']
    video_keys = [k for k, v in info['features'].items() if v['dtype'] == 'video']
    ffmpeg = str(Path(sys.executable).parent / 'ffmpeg')
    ffprobe = str(Path(sys.executable).parent / 'ffprobe')
    meta = dst / 'meta'
    meta.mkdir(parents=True)
    tasks = pd.read_parquet(src / 'meta/tasks.parquet')
    with (meta / 'tasks.jsonl').open('w') as f:
        for task, row in tasks.iterrows():
            f.write(json.dumps(dict(task_index=int(row.task_index), task=str(task))) + '\n')
    total = 0
    cached_path, table = None, None
    with (meta / 'episodes.jsonl').open('w') as ef, (meta / 'episodes_stats.jsonl').open('w') as sf:
        for _, ep in eps.iterrows():
            i, n = int(ep.episode_index), int(ep.length)
            chunk = i // 1000
            path = src / info['data_path'].format(chunk_index=int(ep['data/chunk_index']), file_index=int(ep['data/file_index']))
            if path != cached_path:
                table, cached_path = pq.read_table(path), path
            import pyarrow.compute as pc
            part = table.filter(pc.equal(table['episode_index'], i))
            if len(part) != n:
                raise ValueError(f'Episode {i}: frame count {len(part)} != {n}')
            for key in ['action', 'observation.state']:
                values = np.stack(part[key].to_pylist())
                if values.shape != (n, 6) or not np.isfinite(values).all():
                    raise ValueError(f'Episode {i}: invalid {key}')
            if part['index'].to_pylist() != list(range(total, total+n)):
                raise ValueError('Noncontiguous global frame indices')
            if part['frame_index'].to_pylist() != list(range(n)):
                raise ValueError('Noncontiguous episode frame indices')
            if not np.allclose(part['timestamp'].to_numpy(), np.arange(n)/fps, atol=1e-4):
                raise ValueError('Unexpected episode timestamps')
            target = dst / f'data/chunk-{chunk:03d}/episode_{i:06d}.parquet'
            target.parent.mkdir(parents=True, exist_ok=True)
            # New datasets releases encode list features as "List" in schema
            # metadata. Older readers understand the Arrow list type, but not
            # that newer HF metadata tag; let them infer features from Arrow.
            pq.write_table(part.replace_schema_metadata(None), target)
            for key in video_keys:
                prefix = f'videos/{key}'
                video = src / info['video_path'].format(video_key=key, chunk_index=int(ep[prefix+'/chunk_index']), file_index=int(ep[prefix+'/file_index']))
                start = round(float(ep[prefix+'/from_timestamp']) * fps)
                target = dst / f'videos/chunk-{chunk:03d}/{key}/episode_{i:06d}.mp4'
                target.parent.mkdir(parents=True, exist_ok=True)
                # Input-side accurate seek avoids decoding every preceding episode
                # repeatedly. Source PTS are validated at the dataset FPS by audit.
                subprocess.run([ffmpeg, '-v', 'error', '-ss', f'{start/fps:.9f}', '-i', str(video), '-an', '-vf',
                                f'trim=end_frame={n},setpts=PTS-STARTPTS',
                                '-frames:v', str(n), '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
                                '-r', str(fps), str(target)], check=True)
                count = subprocess.check_output([ffprobe, '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                                                  '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', str(target)], text=True).strip()
                if int(count) != n:
                    raise ValueError(f'Video frame count mismatch: {target}')
            stats = {}
            for key, value in ep.items():
                if key.startswith('stats/'):
                    feature, stat = key[6:].rsplit('/', 1)
                    if stat in ['min', 'max', 'mean', 'std', 'count']:
                        stats.setdefault(feature, {})[stat] = plain(value)
            ef.write(json.dumps(dict(episode_index=i, tasks=plain(ep.tasks), length=n)) + '\n')
            sf.write(json.dumps(dict(episode_index=i, stats=stats)) + '\n')
            total += n
            print(f'Exported episode {i}: {n} frames', flush=True)
    out = copy.deepcopy(info)
    out.update(codebase_version='v2.1', total_episodes=len(eps), total_frames=total,
               total_chunks=(len(eps)+999)//1000, total_videos=len(eps)*len(video_keys),
               chunks_size=1000, splits={'train': f'0:{len(eps)}'},
               data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
               video_path='videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4')
    for key in video_keys:
        out['features'][key]['info']['video.codec'] = 'h264'
    (meta / 'info.json').write_text(json.dumps(out, indent=2))
    (meta / 'conversion.json').write_text(json.dumps(dict(source=str(src), source_version='v3.0',
        exported_episodes=len(eps), action_transform='identity', video_encoding='h264 crf18',
        note='Image statistics inherited from original videos; compute OpenPI norm stats separately.'), indent=2))
    print(f'Completed: {dst}')

if __name__ == '__main__':
    main()
