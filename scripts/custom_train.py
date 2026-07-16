import argparse
import os
import re
from tqdm import tqdm

from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_inverse_sqrt_schedule,
)
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from gliner import GLiNER, GLiNERConfig
from gliner.data_processing.tokenizer import WordsSplitter
from gliner.data_processing.trigger_types import apply_derived_trigger_types
from gliner.utils import load_config_as_namespace
from gliner.training.data_utils import (
    BestModelTracker,
    blind_test_by_language,
    evaluate_and_extract_f1,
    flatten_namespace,
    load_multi_dataset,
    print_blind_test,
    warn_if_max_types_truncates,
    window_records,
)


def save_top_k_checkpoints(model: GLiNER, save_path: str, checkpoint: int, top_k: int = 5):
    """
    Save the top-k checkpoints (latest k checkpoints) of a model and tokenizer.

    Parameters:
        model (GLiNER): The model to save.
        save_path (str): The directory path to save the checkpoints.
        top_k (int): The number of top checkpoints to keep. Defaults to 5.
    """
    # Save the current model and tokenizer
    if isinstance(model, DDP):
        model.module.save_pretrained(os.path.join(save_path, str(checkpoint)))
    else:
        model.save_pretrained(os.path.join(save_path, str(checkpoint)))
    
    # List all files in the directory
    files = os.listdir(save_path)

    # Filter files to keep only the model checkpoints
    checkpoint_folders = [file for file in files if re.search(r'model_\d+', file)]

    # Sort checkpoint files by modification time (latest first)
    checkpoint_folders.sort(key=lambda x: os.path.getmtime(os.path.join(save_path, x)), reverse=True)

    # Keep only the top-k checkpoints
    for checkpoint_folder in checkpoint_folders[top_k:]:
        checkpoint_folder = os.path.join(save_path, checkpoint_folder)
        checkpoint_files = [os.path.join(checkpoint_folder, f) for f in os.listdir(checkpoint_folder)]
        for file in checkpoint_files:
            os.remove(file)
        os.rmdir(os.path.join(checkpoint_folder))


class Trainer:
    def __init__(self, config, allow_distributed, compile_model=False, device='cuda'):
        self.config = config
        self.lr_encoder = float(self.config.lr_encoder)
        self.lr_others = float(self.config.lr_others)
        self.weight_decay_encoder = float(self.config.weight_decay_encoder)
        self.weight_decay_other = float(self.config.weight_decay_other)

        self.compile_model = compile_model

        self.device = device

        gliner_class = GLiNER._get_gliner_class(GLiNERConfig(**vars(config)))
        self.model_config = gliner_class.config_class(**vars(config))

        self.allow_distributed = allow_distributed

        self.optimizer = None

    def setup_distributed(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def cleanup_distributed(self):
        dist.destroy_process_group()

    def create_optimizer(self, opt_model, **optimizer_kwargs):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.lr_others is not None:
                encoder_parameters = [name for name, _ in opt_model.named_parameters() if "token_rep_layer" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in encoder_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.weight_decay_other,
                        "lr": self.lr_others,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in encoder_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.lr_others,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in encoder_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.weight_decay_encoder,
                        "lr": self.lr_encoder,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in encoder_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.lr_encoder,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.weight_decay_encoder,
                        "lr": self.lr_encoder,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.lr_encoder,
                    },
                ]

            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer
    
    def setup_model_and_optimizer(self, rank=None, device=None):
        if device is None:
            device = self.device
        if self.config.prev_path is not None:
            model = GLiNER.from_pretrained(self.config.prev_path).to(device)
            model.config = self.model_config
        else:
            model = GLiNER.from_config(vars(self.config)).to(device)
        if rank is not None:
            model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        optimizer = self.create_optimizer(model.model)

        if self.compile_model:
            model.compile_for_training()

        return model, optimizer

    def create_dataloader(self, dataset, data_processor, collator_class, sampler=None, shuffle=True):
        collator = collator_class(self.config, data_processor=data_processor, prepare_labels=True)
        data_loader = DataLoader(dataset, batch_size=self.config.train_batch_size, num_workers=12,
                                                        shuffle=shuffle, collate_fn=collator, sampler=sampler)
        return data_loader
    
    def train_dist(self, rank, world_size, dataset, val_records=None, tracker=None):
        # Init distributed process group
        self.setup_distributed(rank, world_size)

        device = f'cuda:{rank}'

        model, optimizer = self.setup_model_and_optimizer(rank, device=device)

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)

        train_loader = self.create_dataloader(dataset, model.data_processor, model.data_collator_class,
                                               sampler=sampler, shuffle=False)

        num_steps = self.config.num_steps // world_size

        self.train(model=model, optimizer=optimizer, train_loader=train_loader,
                   num_steps=num_steps, device=device, rank=rank,
                   val_records=val_records, tracker=tracker)

        self.cleanup_distributed()

    def init_scheduler(self, scheduler_type, optimizer, num_warmup_steps, num_steps):
        if scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_steps
            )
        elif scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_steps
            )
        elif scheduler_type == "constant":
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
            )
        elif scheduler_type == "polynomial":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_steps
            )
        elif scheduler_type == "inverse_sqrt":
            scheduler = get_inverse_sqrt_schedule(
                optimizer,
                num_warmup_steps=num_warmup_steps,
            )
        else:
            raise ValueError(
                f"Invalid scheduler_type value: '{scheduler_type}' \n Supported scheduler types: 'cosine', 'linear', 'constant', 'polynomial', 'inverse_sqrt'"
            )
        return scheduler

    def train(self, model, optimizer, train_loader, num_steps, device='cuda', rank=None,
              val_records=None, tracker=None):
        model.train()
        pbar = tqdm(range(num_steps))

        warmup_ratio = self.config.warmup_ratio
        eval_every = self.config.eval_every
        save_total_limit = self.config.save_total_limit
        log_dir = self.config.log_dir

        num_warmup_steps = int(num_steps * warmup_ratio) if warmup_ratio < 1 else int(warmup_ratio)

        scheduler = self.init_scheduler(self.config.scheduler_type, optimizer, num_warmup_steps, num_steps)
        iter_train_loader = iter(train_loader)

        use_amp = device.startswith('cuda')
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        for step in pbar:
            optimizer.zero_grad()

            try:
                x = next(iter_train_loader)
            except StopIteration:
                iter_train_loader = iter(train_loader)
                x = next(iter_train_loader)

            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    x[k] = v.to(device)

            try:
                with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                    loss = model(alpha = self.config.loss_alpha,
                                    gamma = self.config.loss_gamma,
                                    label_smoothing = self.config.label_smoothing,
                                    reduction = self.config.loss_reduction,
                                    **x).loss

                if torch.isnan(loss).any():
                    print("Warning: NaN loss detected")
                    continue

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                del x
                if use_amp:
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error: {e}")
                del x
                if use_amp:
                    torch.cuda.empty_cache()
                continue

            description = f"step: {step} | epoch: {step // len(train_loader)} | loss: {loss.item():.2f}"
            pbar.set_description(description)

            if (step + 1) % eval_every == 0:
                if rank is None or rank == 0:
                    checkpoint = f'model_{step + 1}'
                    save_top_k_checkpoints(model, log_dir, checkpoint, save_total_limit)
                    if val_records:
                        eval_model = model.module if isinstance(model, DDP) else model
                        f1, _ = evaluate_and_extract_f1(eval_model, val_records)
                        improved = tracker.maybe_save(f1, eval_model, log_dir)
                        marker = " (new best)" if improved else ""
                        print(f"\n[eval] step={step + 1} F1={f1:.4f} best={tracker.best_f1:.4f}{marker}")
                    model.train()

    def run(self):
        seed = int(getattr(self.config, "seed", 42))
        data = load_multi_dataset(self.config.train_data, seed=seed)

        # Event models need config.trigger_types to split labels into triggers
        # vs. arguments; the shipped event configs leave it empty. Derive from
        # the raw (pre-windowing) records so the event head is fed triggers.
        apply_derived_trigger_types(self.config, data)
        warn_if_max_types_truncates(data, getattr(self.config, "max_types", None))

        data = window_records(data, max_len=self.config.max_len)
        val_records = load_multi_dataset(getattr(self.config, "val_data", None), seed=seed)
        test_records = load_multi_dataset(getattr(self.config, "test_data", None), seed=seed)
        tracker = BestModelTracker()

        trained_model = None
        if torch.cuda.device_count() > 1 and self.allow_distributed:
            world_size = torch.cuda.device_count()
            mp.spawn(self.train_dist, args=(world_size, data, val_records, tracker), nprocs=world_size, join=True)
        else:
            model, optimizer = self.setup_model_and_optimizer()

            train_loader = self.create_dataloader(data, model.data_processor, model.data_collator_class, shuffle=True)

            self.train(model, optimizer, train_loader, num_steps=self.config.num_steps, device=self.device,
                       val_records=val_records, tracker=tracker)
            trained_model = model

        if not test_records:
            return

        best_dir = os.path.join(self.config.log_dir, "best")
        if os.path.isdir(best_dir):
            print(f"\nReloading best checkpoint from {best_dir} for the blind test...")
            model_for_test = GLiNER.from_pretrained(best_dir)
        elif trained_model is not None:
            print("\nNo 'best' checkpoint found; using the final trained model.")
            model_for_test = trained_model.module if isinstance(trained_model, DDP) else trained_model
        else:
            print("\nNo 'best' checkpoint found and no in-process model available "
                  "(distributed run with no val_data); skipping blind test.")
            return

        eval_by_language = bool(getattr(self.config, "eval_by_language", False))
        if eval_by_language:
            blind_test_by_language(model_for_test, test_records, evaluate_kwargs={})
        else:
            f1, output = evaluate_and_extract_f1(model_for_test, test_records)
            print_blind_test("all", f1, output, event_mode=bool(getattr(model_for_test.config, "event_mode", False)))


def create_parser():
    parser = argparse.ArgumentParser(description="Span-based NER")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument('--log_dir', type=str, default='logs', help='Path to the log directory')
    parser.add_argument('--allow_distributed', type=bool, default=False,
                        help='Whether to allow distributed training if there are more than one GPU available')
    parser.add_argument('--compile_model', type=bool, default=False,
                        help='Whether to apply torch.compile to a modell or not')
    return parser


def select_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    config = flatten_namespace(load_config_as_namespace(args.config))
    config.log_dir = args.log_dir

    trainer = Trainer(config, allow_distributed=args.allow_distributed,
                      compile_model = args.compile_model,
                      device=select_device())
    trainer.run()