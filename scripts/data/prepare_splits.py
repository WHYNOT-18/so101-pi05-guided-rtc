"""Freeze a 100/20/20 episode split and create independent v2.1 subsets."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--raw-source',type=Path,required=True)
    p.add_argument('--layouts',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();src=a.source.resolve();raw=a.raw_source.resolve();dst=a.destination.resolve()
    if dst.exists() or src in dst.parents or raw in dst.parents:
        p.error('Destination must be new and outside both existing datasets')
    info=json.loads((src/'meta/info.json').read_text());assert info['total_episodes']==140 and info['codebase_version']=='v2.1'
    original={str(f.relative_to(src)):digest(f) for f in sorted(src.rglob('*')) if f.is_file()}
    layouts={int(r['episode_index']):r['layout'] for r in json.loads(a.layouts.read_text())['rows']}
    assert Counter(layouts.values())=={'red_left_blue_right':60,'red_right_blue_left':80}
    raw_eps=pd.concat([pd.read_parquet(f) for f in sorted((raw/'meta/episodes').rglob('*.parquet'))]).sort_values('episode_index')
    groups={}
    for _,r in raw_eps.iterrows():
        groups.setdefault((int(r['data/chunk_index']),int(r['data/file_index'])),[]).append(int(r.episode_index))
    main_groups=[g for _,g in sorted(groups.items()) if len(g)>1]
    replacement=[e for g in groups.values() if len(g)==1 for e in g]
    assert len(main_groups)==7 and replacement==[139]
    val_counts=[3,3,3,3,3,3,2];test_counts=[3,3,3,3,3,2,3]
    rng=np.random.default_rng(a.seed)
    target_left={'train':43,'val':9,'test':8}
    for attempt in range(100000):
        splits={'train':replacement.copy(),'val':[],'test':[]}
        for group,nv,nt in zip(main_groups,val_counts,test_counts,strict=True):
            ids=rng.permutation(group).tolist()
            splits['val']+=ids[:nv];splits['test']+=ids[nv:nv+nt];splits['train']+=ids[nv+nt:]
        left={k:sum(layouts[i]=='red_left_blue_right' for i in ids) for k,ids in splits.items()}
        if left==target_left:break
    else:raise RuntimeError('Could not meet fixed batch/layout quotas')
    for ids in splits.values():ids.sort()
    assert {k:len(v) for k,v in splits.items()}=={'train':100,'val':20,'test':20}
    assert sorted(i for ids in splits.values() for i in ids)==list(range(140))
    assert all(not set(splits[x])&set(splits[y]) for x,y in [('train','val'),('train','test'),('val','test')])
    episodes={r['episode_index']:r for r in map(json.loads,(src/'meta/episodes.jsonl').read_text().splitlines())}
    stats={r['episode_index']:r['stats'] for r in map(json.loads,(src/'meta/episodes_stats.jsonl').read_text().splitlines())}
    video_keys=[k for k,v in info['features'].items() if v['dtype']=='video']
    manifest=dict(seed=a.seed,source=str(src),raw_source=str(raw),source_episode_count=140,
        method='Whole episodes; fixed per-acquisition-file quotas; layout-stratified; seeded random search until predeclared 43/9/8 left-layout quotas match. No outcomes or model metrics used.',
        sampling_attempt=attempt+1,replacement_source_episode=139,replacement_assigned_to='train',
        splits=splits,subsets={},source_sha256_before=original)
    dst.mkdir(parents=True)
    for split,ids in splits.items():
        root=dst/split;meta=root/'meta';meta.mkdir(parents=True)
        shutil.copy2(src/'meta/tasks.jsonl',meta/'tasks.jsonl')
        offset=0;mapping=[]
        with (meta/'episodes.jsonl').open('w') as ef,(meta/'episodes_stats.jsonl').open('w') as sf:
            for new,old in enumerate(ids):
                source_path=src/info['data_path'].format(episode_chunk=old//1000,episode_index=old)
                table=pq.read_table(source_path);n=len(table)
                for col,value in [('episode_index',np.full(n,new,dtype=np.int64)),('index',np.arange(offset,offset+n,dtype=np.int64))]:
                    table=table.set_column(table.schema.get_field_index(col),col,pa.array(value))
                target=root/info['data_path'].format(episode_chunk=new//1000,episode_index=new)
                target.parent.mkdir(parents=True,exist_ok=True);pq.write_table(table.replace_schema_metadata(None),target)
                # Values other than the two reindexed columns must match exactly.
                check=pq.read_table(target);original_table=pq.read_table(source_path)
                cols=[c for c in table.column_names if c not in ['episode_index','index']]
                assert check.select(cols).equals(original_table.select(cols),check_metadata=False)
                for key in video_keys:
                    source_video=src/info['video_path'].format(episode_chunk=old//1000,episode_index=old,video_key=key)
                    target_video=root/info['video_path'].format(episode_chunk=new//1000,episode_index=new,video_key=key)
                    target_video.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source_video,target_video)
                    assert digest(target_video)==original[str(source_video.relative_to(src))]
                ep=copy.deepcopy(episodes[old]);ep['episode_index']=new
                ef.write(json.dumps(ep)+'\n')
                st=copy.deepcopy(stats[old])
                for key,values in [('episode_index',np.full(n,new,dtype=float)),('index',np.arange(offset,offset+n,dtype=float))]:
                    st[key]=dict(min=[float(values.min())],max=[float(values.max())],mean=[float(values.mean())],std=[float(values.std())],count=[n])
                sf.write(json.dumps(dict(episode_index=new,stats=st))+'\n')
                mapping.append(dict(subset_episode_index=new,source_episode_index=old,layout=layouts[old],frames=n))
                offset+=n
        outinfo=copy.deepcopy(info);outinfo.update(total_episodes=len(ids),total_frames=offset,total_videos=len(ids)*len(video_keys),total_chunks=1,splits={'train':f'0:{len(ids)}'})
        (meta/'info.json').write_text(json.dumps(outinfo,indent=2))
        (root/'source_episode_mapping.json').write_text(json.dumps(mapping,indent=2))
        manifest['subsets'][split]=dict(directory=str(root),episodes=len(ids),frames=offset,layout_counts=dict(Counter(layouts[i] for i in ids)),acquisition_groups_covered=len(set((int(raw_eps[raw_eps.episode_index==i].iloc[0]['data/chunk_index']),int(raw_eps[raw_eps.episode_index==i].iloc[0]['data/file_index'])) for i in ids)))
        print(split,manifest['subsets'][split],flush=True)
    after={str(f.relative_to(src)):digest(f) for f in sorted(src.rglob('*')) if f.is_file()}
    assert original==after
    manifest['source_unchanged']=True;manifest['normalization_computed']=False
    (dst/'split_manifest.json').write_text(json.dumps(manifest,indent=2))
    (dst/'source_sha256_after.json').write_text(json.dumps(after,indent=2))
    print('PASS: 100/20/20, no episode overlap, all 140 covered, videos copied byte-for-byte; source unchanged.',flush=True)

if __name__=='__main__':main()
