"""SO-101 pi0.5 service with differentiable Guided RTC and official normalization."""
import argparse
import hashlib
import json
import logging
import os
import socket
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'so101_remote')]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, default=Path('/root/autodl-tmp/so101'))
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--beta', type=float, default=5.)
    p.add_argument('--soft-prefix-steps', type=int, default=6,
                   help='Softly constrain only this many steps beyond the committed delay')
    p.add_argument('--num-steps', type=int, default=10)
    args = p.parse_args()
    assert 0 < args.beta <= 10 and args.num_steps >= 1
    assert 0 <= args.soft_prefix_steps <= 50
    root = args.root.resolve()
    sys.path[:0] = [str(root), str(root/'scripts')]
    os.environ['OPENPI_DATA_HOME'] = str(root/'cache/openpi')
    os.environ['HF_HOME'] = str(root/'cache/huggingface')
    os.environ['HF_LEROBOT_HOME'] = str(root/'cache/lerobot')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    os.environ.setdefault('OMP_NUM_THREADS','2')
    logging.basicConfig(level=logging.INFO)
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from openpi.models.model import Observation
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from training_config import make_training_config
    from remote_common import JOINT_NAMES, load_observation, validate_observation, percentiles
    from rtc_sampling import prefix_weights, sample_guided
    from image_transport import decode_observation, SUPPORTED_IMAGE_CODECS

    assert any(d.platform=='gpu' for d in jax.devices()), 'GPU required'
    checkpoint = args.checkpoint or root/'checkpoints/pi05_so101_data/formal6000_b16_20261118_134620_14830/7000'
    stats = checkpoint/'assets/so101/norm_stats.json'
    assert json.loads(stats.read_text()) == json.loads((root/'openpi_assets/pi05_so101_data/so101/norm_stats.json').read_text())
    fixture = HERE.parent/'so101_remote/latency_observation.npz'
    obs = load_observation(fixture)
    out = root/'reports'/('guided_rtc_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))
    out.mkdir(parents=True,exist_ok=False)
    print('Loading model and compiling VJP Guided RTC; no robot connected here.',flush=True)
    base = policy_config.create_trained_policy(make_training_config(root,1,'rtc7000'),checkpoint,
                                              sample_kwargs={'num_steps':args.num_steps})
    graph, state = nnx.split(base._model)

    @jax.jit
    def sample(weights, observation, previous, mask, key):
        model = nnx.merge(graph, weights)
        return sample_guided(model,key,observation,previous,mask,num_steps=args.num_steps,beta=args.beta)

    class GuidedPolicy:
        key = jax.random.key(0)
        def infer(self, incoming):
            start = time.perf_counter()
            incoming = decode_observation(incoming)
            validate_observation(incoming)
            rtc = incoming.get('rtc', {})
            previous = np.asarray(rtc.get('previous_actions',np.zeros((50,6))),dtype=np.float32)
            valid, delay = int(rtc.get('valid_length',0)), int(rtc.get('inference_delay',0))
            if previous.shape!=(50,6) or not np.isfinite(previous).all() or not 0<=delay<=valid<=50:
                raise ValueError('Invalid RTC prefix or delay')
            raw = {k:incoming[k] for k in ('front','wrist','state','prompt')}
            raw['actions'] = previous
            inputs = base._input_transform(raw)
            batched = jax.tree.map(lambda x:jnp.asarray(x)[None,...], inputs)
            observation = Observation.from_dict(batched)
            attention_end = min(valid,delay+args.soft_prefix_steps)
            time_mask = prefix_weights(jnp.asarray(delay),jnp.asarray(valid),
                                       soft_steps=args.soft_prefix_steps)
            mask = time_mask[None,:,None]*(jnp.arange(32)<6)[None,None,:]
            self.key, key = jax.random.split(self.key)
            predicted = sample(state,observation,batched['actions'],mask,key)
            predicted_np = np.asarray(predicted[0])
            previous_np = np.asarray(batched['actions'][0])
            prefix_error = predicted_np[:min(delay,valid),:6] - previous_np[:min(delay,valid),:6]
            result = base._output_transform({'state':np.asarray(batched['state'][0]),
                                             'actions':predicted_np})
            actions = np.asarray(result['actions'],dtype=np.float32)
            if actions.shape!=(50,6) or not np.isfinite(actions).all():
                raise ValueError('Nonfinite RTC action prediction')
            return dict(actions=actions, policy_wall_ms=(time.perf_counter()-start)*1000,
                rtc=dict(guided=attention_end>0,inference_delay=delay,valid_length=valid,
                         soft_prefix_steps=args.soft_prefix_steps,attention_end=attention_end,
                         beta=args.beta,schedule='exp',method='VJP_soft_inpainting',
                         normalized_committed_prefix_rmse=None if not len(prefix_error)
                             else float(np.sqrt(np.mean(prefix_error**2)))))

    policy = GuidedPolicy()
    first = policy.infer(obs)
    print('BASE SAMPLING PASSED:',first['policy_wall_ms'],'ms',flush=True)
    prefix = np.zeros((50,6),dtype=np.float32)
    prefix[:25] = first['actions'][25:]
    times=[]
    for i in range(4):
        result=policy.infer({**obs,'rtc':dict(previous_actions=prefix,valid_length=25,inference_delay=16)})
        print(f'GUIDED WARMUP {i+1}/4: {result["policy_wall_ms"]:.1f} ms',flush=True)
        if i: times.append(result['policy_wall_ms'])
    metadata=dict(checkpoint=str(checkpoint),action_shape=[50,6],joint_names=JOINT_NAMES,
        action_mode='absolute_joint_targets',control_hz=25,num_steps=args.num_steps,
        norm_stats_sha256=hashlib.sha256(stats.read_bytes()).hexdigest(),
        fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
        guided_rtc_enabled=True,rtc_method='VJP_soft_inpainting',rtc_beta=args.beta,
        rtc_soft_prefix_steps=args.soft_prefix_steps,
        rtc_schedule='exp',rtc_constraints_on_physical_dimensions=6)
    metadata.update(image_transport_version=1,supported_image_codecs=list(SUPPORTED_IMAGE_CODECS),
                    server_hostname=socket.gethostname(),device_kind=jax.devices()[0].device_kind)
    report=dict(status='passed',**metadata,warmed_guided_policy=percentiles(times),
        warmup_only=True,robot_commands_sent=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('GUIDED BENCHMARK:',json.dumps(report['warmed_guided_policy']),flush=True)
    print('REPORT:',out/'report.json',flush=True)
    print(f'GUIDED RTC READY: ws://127.0.0.1:{args.port}',flush=True)
    WebsocketPolicyServer(policy,host='127.0.0.1',port=args.port,metadata=metadata).serve_forever()


if __name__=='__main__':
    main()
