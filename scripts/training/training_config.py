"""Shared OpenPI model, optimizer and training configuration."""
import dataclasses

def make_training_config(root, batch_size, exp_name):
    import flax.nnx as nnx
    from openpi.models.pi0_config import Pi0Config
    from openpi.shared.nnx_utils import PathRegex
    from openpi.training import optimizer, weight_loaders
    from so101_openpi import make_config

    model = Pi0Config(pi05=True, dtype="bfloat16", action_dim=32,
                      action_horizon=50, max_token_len=200,
                      paligemma_variant="gemma_2b_lora",
                      action_expert_variant="gemma_300m")
    # Keep the existing config name so the already validated assets resolve.
    return dataclasses.replace(
        make_config("train", root / "openpi_assets", batch_size),
        exp_name=exp_name, model=model,
        freeze_filter=nnx.Any(model.get_freeze_filter(), PathRegex(".*img.*")),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            str(root / "models/verified_pi05/params")),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=250, peak_lr=2.5e-5,
            decay_steps=5000, decay_lr=2.5e-6),
        optimizer=optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-8,
                                 weight_decay=1e-10, clip_gradient_norm=1.0),
        ema_decay=None, num_workers=4, num_train_steps=100,
        checkpoint_base_dir=str(root / "checkpoints"),
        log_interval=10, save_interval=1000, keep_period=None,
        seed=42, wandb_enabled=False, overwrite=False, resume=False,
    )
