"""Read-only verification of a complete v3 -> v2.1 export."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc


def manifest(root):
    result={}
    for p in sorted(root.rglob('*')):
        if p.is_file():
            h=hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda:f.read(2*1024*1024),b''):h.update(b)
            result[str(p.relative_to(root))]=h.hexdigest()
    return result


def sampled_frame(path, timestamp):
    with av.open(str(path)) as c:
        stream=c.streams.video[0]
        stream.codec_context.thread_count=2
        c.seek(max(0,int((timestamp-0.08)/stream.time_base)),stream=stream,backward=True)
        best=None;best_error=float('inf')
        for frame in c.decode(video=0):
            t=float(frame.pts*frame.time_base);err=abs(t-timestamp)
            if err<best_error:
                best=frame.to_ndarray(format='rgb24');best_error=err
            if t>=timestamp-1e-8:break
        if best_error>1e-4:
            raise ValueError(f'PTS mismatch at {path}: requested {timestamp}, error {best_error}')
        return best


def verify_video(job):
    src,dst,i,view,start,n,fps=job
    timestamps=[];shapes=set()
    with av.open(str(dst)) as c:
        for f in c.decode(video=0):
            timestamps.append(float(f.pts*f.time_base));shapes.add((f.height,f.width))
    if len(timestamps)!=n or shapes!={(480,640)} or not np.allclose(timestamps,np.arange(n)/fps,atol=1e-4,rtol=0):
        raise ValueError(f'Video count/shape/PTS mismatch: {dst}')
    sample_errors={}
    for idx in [0,n//2,n-1]:
        a=sampled_frame(src,(start+idx)/fps)
        b=sampled_frame(dst,idx/fps)
        mae=float(np.abs(a.astype(np.float32)-b.astype(np.float32)).mean())
        sample_errors[str(idx)]=mae
        if mae>5:
            raise ValueError(f'Possible image misalignment at ep {i}/{view}/{idx}: RGB MAE={mae}')
    return dict(episode_index=i,camera=view,decoded_frames=n,pts_errors=0,sample_rgb_mae_0_to_255=sample_errors)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--source-manifest',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args();src=args.source.resolve();dst=args.destination.resolve();out=args.report.resolve();out.mkdir(parents=True,exist_ok=True)
    info=json.loads((src/'meta/info.json').read_text());target=json.loads((dst/'meta/info.json').read_text())
    assert info['total_episodes']==target['total_episodes']==140
    assert info['total_frames']==target['total_frames']==55999
    assert target['codebase_version']=='v2.1'
    episodes=pd.concat([pd.read_parquet(f) for f in sorted((src/'meta/episodes').rglob('*.parquet'))]).sort_values('episode_index')
    converted_episodes=[json.loads(s) for s in (dst/'meta/episodes.jsonl').read_text().splitlines()]
    assert [e['episode_index'] for e in converted_episodes]==list(range(140))
    assert [e['length'] for e in converted_episodes]==episodes.length.tolist()
    tables={};jobs=[]
    for _,ep in episodes.iterrows():
        i,n=int(ep.episode_index),int(ep.length)
        source_path=src/info['data_path'].format(chunk_index=int(ep['data/chunk_index']),file_index=int(ep['data/file_index']))
        if source_path not in tables:tables[source_path]=pq.read_table(source_path)
        original=tables[source_path].filter(pc.equal(tables[source_path]['episode_index'],i))
        result=pq.read_table(dst/target['data_path'].format(episode_chunk=i//1000,episode_index=i))
        assert original.equals(result,check_metadata=False),f'Parquet value mismatch in episode {i}'
        for view in ['front','wrist']:
            key='observation.images.'+view;pre='videos/'+key
            video=src/info['video_path'].format(video_key=key,chunk_index=int(ep[pre+'/chunk_index']),file_index=int(ep[pre+'/file_index']))
            result_video=dst/target['video_path'].format(episode_chunk=i//1000,episode_index=i,video_key=key)
            start=round(float(ep[pre+'/from_timestamp'])*info['fps'])
            jobs.append((video,result_video,i,view,start,n,info['fps']))
    print('All 140 parquet episodes: exact values equal.',flush=True)
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(verify_video,j) for j in jobs]
        for f in as_completed(futures):
            results.append(f.result())
            if len(results)%40==0:print(f'Video verification: {len(results)}/280',flush=True)
    expected=json.loads(args.source_manifest.read_text());after=manifest(src)
    (out/'raw_sha256_after.json').write_text(json.dumps(after,indent=2))
    assert expected==after,'Original source files changed during conversion'
    all_errors=[v for r in results for v in r['sample_rgb_mae_0_to_255'].values()]
    report=dict(episodes=140,frames=55999,videos=len(results),parquet_values_identical=True,all_video_frame_counts_and_pts_passed=True,
                sampled_source_target_image_pairs=len(all_errors),rgb_mae_mean=float(np.mean(all_errors)),rgb_mae_max=float(max(all_errors)),
                raw_source_files=len(after),raw_unchanged=True,video_encoding='H.264 CRF18, lossy; original AV1 retained',
                videos_checked=sorted(results,key=lambda r:(r['episode_index'],r['camera'])))
    (out/'validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='videos_checked'},indent=2),flush=True)

if __name__=='__main__':main()
