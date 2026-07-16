import argparse
import json
from pathlib import Path

import torch
from transformers import TrainerCallback

from gliner import GLiNER
from gliner.data_processing.trigger_types import apply_derived_trigger_types
from gliner.training.data_utils import (
    DEFAULT_THRESHOLD_GRID,
    BestModelTracker,
    blind_test_by_language,
    evaluate_and_extract_f1,
    load_multi_dataset,
    print_blind_test,
    sweep_thresholds,
    window_records,
)
from gliner.utils import load_config_as_namespace, namespace_to_dict


def build_model(model_cfg: dict, train_cfg: dict):
    """Build or load GLiNER model."""
    prev_path = train_cfg.get("prev_path")
    if prev_path and str(prev_path).lower() not in ("none", "null", ""):
        print(f"Loading pretrained model from: {prev_path}")
        return GLiNER.from_pretrained(prev_path)
    print("Initializing model from config...")
    return GLiNER.from_config(model_cfg)


class BestF1Callback(TrainerCallback):
    """Evaluates on the validation set at each eval step and keeps the
    highest-F1 checkpoint in ``<output_dir>/best`` via ``BestModelTracker``.

    Bypasses transformers' native ``load_best_model_at_end`` /
    ``metric_for_best_model`` machinery, which requires ``compute_metrics``
    to flow through its ``EvalPrediction`` plumbing -- a poor fit for
    GLiNER's structured span/relation outputs. This callback calls the
    model's own ``evaluate()`` method directly instead.

    Also handles early stopping (when ``patience`` is set): unlike
    ``transformers.EarlyStoppingCallback``, which only ever looks at
    ``args.metric_for_best_model`` inside Trainer's own internal ``metrics``
    dict (never populated with our F1, since we bypass that machinery too),
    this tracks consecutive non-improving evals against the same F1 this
    callback already computes, and stops training the same way
    (``control.should_training_stop = True``) once ``patience`` is exceeded.
    """

    def __init__(self, eval_records, output_dir, evaluate_kwargs, patience=None):
        self.eval_records = eval_records
        self.output_dir = output_dir
        self.evaluate_kwargs = evaluate_kwargs
        self.tracker = BestModelTracker()
        self.patience = patience
        self.evals_without_improvement = 0

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        f1, _ = evaluate_and_extract_f1(model, self.eval_records, **self.evaluate_kwargs)
        improved = self.tracker.maybe_save(f1, model, self.output_dir)
        marker = " (new best)" if improved else ""
        print(f"\n[eval] step={state.global_step} F1={f1:.4f} best={self.tracker.best_f1:.4f}{marker}")

        if self.patience is None:
            return

        if improved:
            self.evals_without_improvement = 0
        else:
            self.evals_without_improvement += 1
            if self.evals_without_improvement >= self.patience:
                print(
                    f"\n[early stopping] no improvement for {self.evals_without_improvement} "
                    f"evals (patience={self.patience}) -- stopping training."
                )
                control.should_training_stop = True


def main(cfg_path: str):
    """Main training function."""
    cfg = load_config_as_namespace(cfg_path)

    model_cfg = namespace_to_dict(cfg.model)
    train_cfg = namespace_to_dict(cfg.training)

    output_dir = Path(cfg.data.root_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(train_cfg.get("seed", 42))

    # Overlap between consecutive windows for documents longer than
    # config.model.max_len (see gliner/data_processing/windowing.py).
    # Defaults to 25% of max_len when unset -- None here, not a computed
    # value, so every windowing call below independently defaults the same
    # way rather than freezing one max_len's derived stride into a var that
    # could go stale if max_len differs between callers.
    window_stride = getattr(cfg.training, "window_stride", None)

    # Decision threshold for entity/relation/adjacency decoding. threshold_sweep
    # (unset/false: keep `threshold` as-is; true: DEFAULT_THRESHOLD_GRID; or an
    # explicit list of candidates) calibrates it against val_data after training,
    # replacing `threshold` for the blind test -- see the sweep step below.
    threshold = float(getattr(cfg.training, "threshold", 0.5))
    threshold_sweep_cfg = getattr(cfg.training, "threshold_sweep", None)

    print(f"Loading training data from: {cfg.data.train_data}")
    train_dataset = load_multi_dataset(cfg.data.train_data, seed=seed)

    # Event models split each record's labels into triggers vs. arguments via
    # config.trigger_types. The shipped event configs leave it empty (derive
    # per-dataset); do that here from the raw (pre-windowing) records so the
    # event head is actually fed triggers instead of silently producing none.
    apply_derived_trigger_types(model_cfg, train_dataset)

    train_dataset = window_records(train_dataset, max_len=cfg.model.max_len, stride=window_stride)
    print(f"Training samples: {len(train_dataset)}")

    eval_dataset = load_multi_dataset(getattr(cfg.data, "val_data", None), seed=seed)
    if eval_dataset is not None:
        print(f"Validation samples: {len(eval_dataset)}")

    # HF Trainer's own internal loss-based eval loop (eval_loss/eval_runtime
    # logging, triggered by eval_strategy="steps" below) goes through the
    # plain collator -> preprocess_example path, which has no windowing --
    # unlike BestF1Callback, which calls model.evaluate() directly and
    # windows/merges internally. Feed Trainer's loop a pre-windowed copy so
    # long validation documents aren't silently truncated there too.
    # BestF1Callback still gets the original, unwindowed eval_dataset: its
    # F1 scoring needs whole, untruncated records to score against full gold.
    trainer_eval_dataset = (
        window_records(eval_dataset, max_len=cfg.model.max_len, stride=window_stride)
        if eval_dataset is not None
        else None
    )

    test_dataset = load_multi_dataset(getattr(cfg.data, "test_data", None), seed=seed)
    if test_dataset is not None:
        print(f"Test samples: {len(test_dataset)}")

    eval_by_language = bool(train_cfg.get("eval_by_language", False))

    # Build model
    model = build_model(model_cfg, train_cfg).to(dtype=torch.float32)
    print(f"Model type: {model.__class__.__name__}")

    freeze_components = train_cfg.get("freeze_components", None)
    if freeze_components:
        print(f"Freezing components: {freeze_components}")

    # Schedule: epoch-based when the config sets num_epochs (new configs),
    # step-based when it sets num_steps (pre-existing configs) -- max_steps
    # must be explicitly -1 for num_train_epochs to actually take effect,
    # since create_training_args defaults max_steps to 10000 otherwise and
    # HF Trainer lets a positive max_steps silently override num_train_epochs.
    num_epochs = getattr(cfg.training, "num_epochs", None)
    if num_epochs is not None:
        schedule_kwargs = {
            "num_train_epochs": num_epochs,
            "max_steps": -1,
            "eval_strategy": "epoch" if eval_dataset is not None else "no",
            "save_strategy": "epoch",
            "logging_strategy": "epoch",
        }
        schedule_desc = f"{num_epochs} epochs"
    else:
        schedule_kwargs = {
            "max_steps": cfg.training.num_steps,
            "save_steps": cfg.training.eval_every,
            "logging_steps": cfg.training.eval_every,
            # Eval runs at the same cadence as save/logging so BestF1Callback fires.
            "eval_strategy": "steps" if eval_dataset is not None else "no",
            "eval_steps": cfg.training.eval_every,
        }
        schedule_desc = f"{cfg.training.num_steps} steps"

    early_stopping_patience = getattr(cfg.training, "early_stopping_patience", None)

    callbacks = []
    if eval_dataset is not None:
        callbacks.append(
            BestF1Callback(
                eval_dataset,
                output_dir,
                evaluate_kwargs={"window_stride": window_stride, "threshold": threshold},
                patience=early_stopping_patience,
            )
        )

    # Train
    print(f"\nStarting training ({schedule_desc})...")
    model.train_model(
        train_dataset=train_dataset,
        eval_dataset=trainer_eval_dataset,
        output_dir="models",
        lr_scheduler_type=cfg.training.scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        # Batch & optimization
        per_device_train_batch_size=cfg.training.train_batch_size,
        per_device_eval_batch_size=cfg.training.train_batch_size,
        gradient_accumulation_steps=int(getattr(cfg.training, "gradient_accumulation_steps", 1)),
        learning_rate=float(cfg.training.lr_encoder),
        others_lr=float(cfg.training.lr_others),
        weight_decay=float(cfg.training.weight_decay_encoder),
        others_weight_decay=float(cfg.training.weight_decay_other),
        max_grad_norm=float(cfg.training.max_grad_norm),
        # Loss
        focal_loss_alpha=float(cfg.training.loss_alpha),
        focal_loss_gamma=float(cfg.training.loss_gamma),
        focal_loss_prob_margin=float(getattr(cfg.training, "loss_prob_margin", 0.0)),
        loss_reduction=cfg.training.loss_reduction,
        negatives=float(cfg.training.negatives),
        masking=cfg.training.masking,
        save_total_limit=cfg.training.save_total_limit,
        **schedule_kwargs,
        # Freezing
        freeze_components=freeze_components,
        callbacks=callbacks,
        # Dtype
        bf16=True,
        seed=seed,
    )

    print(f"\n✓ Training complete! Model saved to {output_dir}")

    best_dir = output_dir / "best"
    if best_dir.is_dir():
        print(f"\nReloading best checkpoint from {best_dir} for the blind test...")
        model = GLiNER.from_pretrained(str(best_dir))
    else:
        print("\nNo 'best' checkpoint found (no val_data configured, or training "
              "ended before completing one eval interval); using the final trained model.")

    if threshold_sweep_cfg and eval_dataset is None:
        print(f"\n[threshold sweep] No val data; skipping, keeping threshold={threshold}.")
    elif threshold_sweep_cfg:
        thresholds = DEFAULT_THRESHOLD_GRID if threshold_sweep_cfg is True else list(threshold_sweep_cfg)
        print(f"\n[threshold sweep] Calibrating against {len(eval_dataset)} val samples "
              f"over {thresholds}...")
        threshold, best_f1, sweep_all = sweep_thresholds(
            model, eval_dataset, thresholds=thresholds, window_stride=window_stride,
        )
        print(f"[threshold sweep] Chose threshold={threshold} (F1={best_f1:.4f}); "
              f"this only recalibrates the decision cutoff, it does not retrain the model.")
        sweep_path = best_dir / "threshold_sweep.json" if best_dir.is_dir() else output_dir / "threshold_sweep.json"
        sweep_path.write_text(json.dumps({"chosen_threshold": threshold, "by_threshold": sweep_all}, indent=2))
        print(f"[threshold sweep] Wrote {sweep_path}")

    if test_dataset is not None:
        if eval_by_language:
            blind_test_by_language(
                model, test_dataset, evaluate_kwargs={"window_stride": window_stride, "threshold": threshold}
            )
        else:
            f1, output = evaluate_and_extract_f1(
                model, test_dataset, window_stride=window_stride, threshold=threshold
            )
            print_blind_test("all", f1, output, event_mode=bool(getattr(model.config, "event_mode", False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GLiNER model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file (YAML or JSON)")
    args = parser.parse_args()
    main(args.config)
