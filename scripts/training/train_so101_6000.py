"""SO-101: 6000 updates, full fixed-noise validation/checkpoints every 1000."""
import argparse
import dataclasses
import fcntl
import functools
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/so101"))
    parser.add_argument("--batch-size", type=int, choices=(8, 16), default=16)
    args = parser.parse_args()
    root = args.root.resolve()
    lock = (root / "train_so101_6000.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another formal training process is already running.")
    sys.path[:0] = [str(root), str(root / "scripts")]
    for key, path in {
        "HF_LEROBOT_HOME": root / "cache/lerobot",
        "HF_HOME": root / "cache/huggingface",
        "OPENPI_DATA_HOME": root / "cache/openpi",
    }.items():
        os.environ[key] = str(path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    os.environ.setdefault("OMP_NUM_THREADS", "2")

    import flax.nnx as nnx
    from flax.traverse_util import flatten_dict
    import jax
    import jax.numpy as jnp
    import numpy as np
    from openpi.models.gemma import get_config as gemma_config
    from openpi.training import checkpoints, data_loader, sharding
    from so101_openpi import make_config
    from training_config import make_training_config

    logging.basicConfig(level=logging.INFO)
    assert jax.device_count() == 1 and jax.devices()[0].platform == "gpu", "Expected one GPU"
    # All six full checkpoints are retained. Refuse to start without useful headroom.
    free = shutil.disk_usage(root).free / 1024**3
    assert free >= 90, f"Only {free:.1f} GiB free; need >=90 GiB before starting"
    base = root / "datasets/red_then_blue_v1_openpi_splits"
    infos = {}
    for split, expected in (("train", 100), ("val", 20)):
        dataset = base / split
        info = json.loads((dataset / "meta/info.json").read_text())
        assert info["total_episodes"] == expected
        infos[split] = info
        alias = root / "cache/lerobot/local" / f"so101_{split}"
        alias.parent.mkdir(parents=True, exist_ok=True)
        if alias.exists() or alias.is_symlink():
            assert alias.resolve() == dataset.resolve(), f"Alias conflict: {alias}"
        else:
            alias.symlink_to(dataset, target_is_directory=True)
    assert infos["val"]["total_frames"] == 8000
    assert 8000 % args.batch_size == 0, "Validation would drop tail samples"

    exp = "formal6000_b%d_%s_%d" % (args.batch_size, time.strftime("%Y%m%d_%H%M%S"), os.getpid())
    cfg = make_training_config(root, args.batch_size, exp)
    cfg = dataclasses.replace(cfg, num_train_steps=6000, save_interval=1000,
        keep_period=1000, log_interval=20,
        lr_schedule=dataclasses.replace(cfg.lr_schedule, decay_steps=6000))
    assert cfg.model.paligemma_variant == "gemma_2b_lora"
    assert cfg.model.action_expert_variant == "gemma_300m"
    for lora in gemma_config("gemma_2b_lora").lora_configs.values():
        assert lora.rank == 16 and lora.alpha == 16
    assert cfg.ema_decay is None
    val_cfg = dataclasses.replace(cfg, data=make_config("val", root / "openpi_assets", args.batch_size).data)
    report_dir = root / "reports" / exp
    report_dir.mkdir(parents=True, exist_ok=False)
    norm_path = root / "openpi_assets/pi05_so101_data/so101/norm_stats.json"
    norm_sha = hashlib.sha256(norm_path.read_bytes()).hexdigest()
    config_report = dict(exp_name=exp, batch_size=args.batch_size, steps=6000,
        checkpoint_interval=1000, validation_interval=1000,
        validation_frames=8000, validation_episodes=20, evaluation_seed=424242,
        validation_metric="frame_weighted_flow_matching_loss_fixed_noise",
        peak_lr=2.5e-5, final_lr=2.5e-6, warmup_steps=250, decay_steps=6000,
        optimizer="AdamW", b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-10,
        gradient_clip=1.0, seed=42, ema=None, lora_rank=16, lora_alpha=16,
        vision_encoder_frozen=True, language_backbone_frozen=True,
        action_expert_full_training=True, horizon=50,
        base_weights=str(root / "models/verified_pi05/params"),
        initialized_from_smoke_checkpoint=False, train_episodes=100,
        norm_stats_sha256=norm_sha, test_set_used=False,
        checkpoint_dir=str(cfg.checkpoint_dir), initial_free_disk_GiB=free)
    atomic_json(report_dir / "config.json", config_report)
    shutil.copy2(__file__, report_dir / "train_so101_6000.py")
    shutil.copy2(root / "train_so101_smoke.py", report_dir / "train_so101_smoke.py")
    shutil.copy2(root / "scripts/so101_openpi.py", report_dir / "so101_openpi.py")
    print("RUN DIRECTORY:", report_dir, flush=True)
    print(json.dumps(config_report, indent=2), flush=True)

    jax.config.update("jax_compilation_cache_dir", str(root / "cache/jax"))
    spec = importlib.util.spec_from_file_location("official_openpi_train", root / "openpi-src/scripts/train.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    mesh = sharding.make_mesh(cfg.fsdp_devices)
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    train_loader = data_loader.create_data_loader(cfg, sharding=data_sharding, shuffle=True)
    val_batches = 8000 // args.batch_size
    val_loader = data_loader.create_data_loader(val_cfg, sharding=data_sharding,
        shuffle=False, num_batches=val_batches)
    assert train_loader.data_config().repo_id == "local/so101_train"
    assert val_loader.data_config().repo_id == "local/so101_val"
    for loader in (train_loader, val_loader):
        assert loader.data_config().norm_stats is not None
    iterator = iter(train_loader)
    batch = next(iterator)
    rng, init_rng = jax.random.split(jax.random.key(42))
    print("Initializing from BASE checkpoint, not smoke checkpoint.", flush=True)
    state, state_sharding = train.init_train_state(cfg, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    trainable = flatten_dict(state.params.filter(cfg.trainable_filter).to_pure_dict(), sep="/")
    paths = sorted(trainable)
    assert any("lora" in p for p in paths)
    assert any("llm" in p and "_1" in p and "lora" not in p for p in paths)
    assert not any("img" in p for p in paths)
    assert not any("llm" in p and "_1" not in p and "lora" not in p for p in paths)
    count = sum(int(np.prod(v.shape)) for v in trainable.values())
    assert count == 457967648, f"Trainable parameter count changed: {count}"
    del trainable
    atomic_json(report_dir / "trainable_paths.json", paths)

    step_fn = jax.jit(functools.partial(train.train_step, cfg),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated), donate_argnums=(1,))
    graph = state.model_def
    def eval_loss(params, current_batch, noise_key):
        model = nnx.merge(graph, params)
        losses = model.compute_loss(noise_key, *current_batch, train=False)
        return jnp.sum(losses), jnp.asarray(losses.size, dtype=jnp.int32)
    eval_fn = jax.jit(eval_loss)

    def evaluate(params, current_step):
        print(f"VALIDATION step={current_step}: all 20 episodes / 8000 frames", flush=True)
        total_loss, total_elements, frames = 0.0, 0, 0
        val_started = time.perf_counter()
        for index, val_batch in enumerate(val_loader):
            # Identical frame order and RNG at every checkpoint.
            key = jax.random.fold_in(jax.random.key(424242), index)
            with sharding.set_mesh(mesh):
                loss_sum, elements = jax.device_get(eval_fn(params, val_batch, key))
            assert np.isfinite(loss_sum), f"Non-finite validation loss, batch {index}"
            total_loss += float(loss_sum)
            total_elements += int(elements)
            frames += int(val_batch[0].state.shape[0])
            if (index + 1) % 100 == 0:
                print(f"VALIDATION {index + 1}/{val_batches} batches", flush=True)
        assert frames == 8000 and total_elements > 0
        assert hashlib.sha256(norm_path.read_bytes()).hexdigest() == norm_sha
        return dict(step=current_step, validation_loss=total_loss / total_elements,
            frames=frames, episodes=20, batches=val_batches,
            seconds=time.perf_counter() - val_started, seed=424242)

    manager, _ = checkpoints.initialize_checkpoint_dir(cfg.checkpoint_dir,
        keep_period=1000, overwrite=False, resume=False)
    best, validations = None, []
    schedule = cfg.lr_schedule.create()
    started = time.perf_counter()
    recent = []
    try:
        with (report_dir / "train_metrics.jsonl").open("w", buffering=1) as metrics:
            for step in range(1, 6001):
                step_start = time.perf_counter()
                with sharding.set_mesh(mesh):
                    state, info = step_fn(rng, state, batch)
                info = {k: float(v) for k, v in jax.device_get(info).items()}
                assert all(np.isfinite(v) for v in info.values()), f"Non-finite training metrics: {info}"
                record = dict(step=step, learning_rate=float(schedule(step - 1)),
                    compute_seconds=time.perf_counter() - step_start, **info)
                metrics.write(json.dumps(record) + "\n")
                recent.append(record)
                if step == 1 or step % 20 == 0:
                    print(f"TRAIN {step}/6000 " + json.dumps(dict(
                        loss_mean=float(np.mean([r["loss"] for r in recent])),
                        grad_norm_mean=float(np.mean([r["grad_norm"] for r in recent])),
                        learning_rate=record["learning_rate"],
                        compute_seconds=record["compute_seconds"],
                        elapsed_minutes=(time.perf_counter() - started) / 60)), flush=True)
                    recent = []
                if step % 1000 == 0:
                    assert int(state.step) == step
                    checkpoints.save_state(manager, state, train_loader, step)
                    manager.wait_until_finished()
                    print(f"CHECKPOINT SAVED: {cfg.checkpoint_dir / str(step)}", flush=True)
                    result = evaluate(state.params, step)
                    result["checkpoint"] = str(cfg.checkpoint_dir / str(step))
                    validations.append(result)
                    atomic_json(report_dir / "validation_history.json", validations)
                    if best is None or result["validation_loss"] < best["validation_loss"]:
                        best = dict(result)
                        atomic_json(report_dir / "best_checkpoint.json", best)
                    print("VALIDATION RESULT: " + json.dumps(result), flush=True)
                    print(f"BEST CHECKPOINT: step={best['step']}, val_loss={best['validation_loss']:.6f}", flush=True)
                if step < 6000:
                    batch = next(iterator)
        assert len(validations) == 6 and int(state.step) == 6000
        atomic_json(report_dir / "report.json", dict(status="passed", optimizer_updates=6000,
            validation_runs=6, validation_frames_per_run=8000, best=best,
            all_logged_metrics_finite=True, test_set_used=False,
            checkpoint_dir=str(cfg.checkpoint_dir), total_loop_seconds=time.perf_counter() - started))
        print("FORMAL TRAINING COMPLETE: 6000 updates, 6 full validation runs.\nREPORT: " + str(report_dir / "report.json"), flush=True)
    finally:
        manager.wait_until_finished()
        manager.close()


if __name__ == "__main__":
    main()
