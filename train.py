# from model import ConvTower, DeconvTower, DualAttentionEncoderBlock, NegativeBinomialLayer, GaussianLayer, MONITOR_VALIDATION, MetadataCrossAttention
from model import CANDI, CANDI_LOSS, CANDI, CANDI_UNET, CANDI_Decoder, CANDI_DNA_Encoder, PeakLayer
from _utils import exponential_linspace_int, negative_binomial_loss, Gaussian, Laplace, NegativeBinomial, compute_perplexity, DataMasker, reverse_complement_dna, reverse_signal, AdaBelief
from muon import Muon
from sklearn.metrics import r2_score, roc_auc_score
from scipy.stats import pearsonr, spearmanr
import numpy as np
from data import CANDIDataHandler, CANDIIterableDataset
try:
    from data_zarr import ZarrCANDIIterableDataset, get_prepared_eic_path
except ImportError:
    ZarrCANDIIterableDataset = None
    get_prepared_eic_path = None
try:
    from data_h5 import H5CANDIIterableDataset, get_prepared_eic_h5_path
except ImportError:
    H5CANDIIterableDataset = None
    get_prepared_eic_h5_path = None

from torch import nn
import torch
from torchinfo import summary
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import tracemalloc, sys, argparse
from datetime import datetime
import math
import os, random
import multiprocessing
import signal
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler

import json
from pathlib import Path
import time
import atexit
from tqdm import tqdm
import warnings
import pandas as pd

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

_WANDB_STATE = {
    "run": None,
    "finish_called": False,
    "log_failures": 0,
    "handlers_installed": False,
}


def _safe_wandb_import():
    try:
        import wandb
        return wandb
    except ImportError:
        return None


def _register_wandb_run(run):
    _WANDB_STATE["run"] = run
    _WANDB_STATE["finish_called"] = False
    _WANDB_STATE["log_failures"] = 0


def _safe_wandb_log(log_data, step=None):
    wandb = _safe_wandb_import()
    if wandb is None or wandb.run is None:
        return False

    try:
        if step is None:
            wandb.log(log_data)
        else:
            wandb.log(log_data, step=step)
        _WANDB_STATE["log_failures"] = 0
        return True
    except Exception as e:
        _WANDB_STATE["log_failures"] += 1
        failure_count = _WANDB_STATE["log_failures"]
        if failure_count == 1 or failure_count % 10 == 0:
            print(
                f"Warning: W&B log failed at step {step}. "
                f"Skipping this upload and continuing training. "
                f"(consecutive failures: {failure_count}, error: {e})"
            )
        return False


def _safe_wandb_finish(exit_code=0):
    if _WANDB_STATE["finish_called"]:
        return

    wandb = _safe_wandb_import()
    if wandb is None:
        _WANDB_STATE["finish_called"] = True
        return

    try:
        run = _WANDB_STATE["run"] if _WANDB_STATE["run"] is not None else wandb.run
        if run is not None:
            run.finish(exit_code=exit_code)
    except Exception as e:
        print(f"Warning: W&B finish failed during shutdown: {e}")
    finally:
        _WANDB_STATE["run"] = None
        _WANDB_STATE["finish_called"] = True


def _handle_shutdown_signal(signum, frame):
    signame = signal.Signals(signum).name
    print(f"\nReceived {signame}; finishing W&B run before exit...")
    _safe_wandb_finish(exit_code=128 + signum)
    raise SystemExit(128 + signum)


def _install_wandb_signal_handlers():
    if _WANDB_STATE["handlers_installed"]:
        return

    signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    atexit.register(_safe_wandb_finish)
    _WANDB_STATE["handlers_installed"] = True


def resolve_dataset_class(dataset_params):
    backend = str(dataset_params.get("data_backend", "npz")).lower()
    if backend == "npz":
        return CANDIIterableDataset
    if backend == "h5":
        if H5CANDIIterableDataset is None:
            raise ImportError(
                "HDF5 backend requested, but `data_h5.py` or its dependencies "
                "could not be imported in the current environment."
            )
        return H5CANDIIterableDataset
    if backend == "zarr":
        if ZarrCANDIIterableDataset is None:
            raise ImportError(
                "Zarr backend requested, but `data_zarr.py` or its dependencies "
                "could not be imported in the current environment."
            )
        return ZarrCANDIIterableDataset
    raise ValueError(f"Unsupported data backend: {backend}")


def resolve_dataloader_workers(is_ddp, world_size, explicit_workers=None):
    """
    Resolve DataLoader worker count with the following priority:
    1. Explicit CLI/config override via `--dataloader-workers`
    2. `SLURM_CPUS_PER_TASK` when available
    3. Local CPU count fallback
    """
    if explicit_workers is not None:
        return max(0, int(explicit_workers)), "explicit", None

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    cpu_budget = None
    source = "system"
    if slurm_cpus is not None:
        try:
            cpu_budget = max(1, int(slurm_cpus))
            source = "slurm"
        except ValueError:
            cpu_budget = None

    if cpu_budget is None:
        cpu_budget = multiprocessing.cpu_count()

    if is_ddp:
        world = max(1, int(world_size))
        per_rank_budget = max(0, cpu_budget // world)
        num_workers = min(per_rank_budget, 4)
    else:
        num_workers = min(cpu_budget, 4)

    return num_workers, source, cpu_budget

##=========================================== Trainer =====================================================##

class CANDI_TRAINER(object):
    def __init__(self, model, dataset_params, training_params, device=None, rank=None, world_size=None):
        """
        Initialize CANDI trainer with model, dataset configuration, and training parameters.
        
        Args:
            model: CANDI model instance
            dataset_params: Dict with dataset configuration (base_path, resolution, etc.)
            training_params: Dict with training configuration (optimizer, lr, epochs, etc.)
            device: Device to use for training, auto-detected if None
            rank: Process rank for DDP (None for single-GPU)
            world_size: Total number of processes for DDP (None for single-GPU)
        """
        super(CANDI_TRAINER, self).__init__()

        # DDP setup
        self.rank = rank
        self.world_size = world_size
        self.is_ddp = (rank is not None and world_size is not None)
        self.is_main_process = (rank == 0) if self.is_ddp else True

        # Device setup
        if device is None:
            if self.is_ddp:
                # In DDP mode, use local rank as device
                local_rank = int(os.environ.get('LOCAL_RANK', 0))
                self.device = torch.device(f"cuda:{local_rank}")
            else:
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if self.is_main_process:
            print(f"Training on device: {self.device}. DDP: {self.is_ddp}")

        # Model setup
        self.model = model.to(self.device)
        
        # Dataset setup - initialize CANDIIterableDataset from data.py
        self.dataset_params = dataset_params
        try:
            dataset_cls = resolve_dataset_class(self.dataset_params)
            self.dataset = dataset_cls(**self.dataset_params)
            if self.is_main_process:
                print(f"Successfully initialized dataset with {len(self.dataset.aliases['experiment_aliases'])} assays")
        except Exception as e:
            if self.is_main_process:
                print(f"Warning: Failed to initialize dataset during __init__: {e}")
                print("Dataset will be initialized during train() call")
            self.dataset = None
        
        # Training configuration with defaults
        self.training_params = {
            'optimizer': 'adamax',
            'enable_validation': True,  # Enabled by default
            'DNA': True,
            "specific_ema_alpha": 0.005,
            'inner_epochs': 1,
            **training_params  # Override defaults with provided params
        }
        
        # Initialize criterion with loss weights
        count_weight = self.training_params.get('count_weight', 1.0)
        pval_weight = self.training_params.get('pval_weight', 1.0)
        peak_weight = self.training_params.get('peak_weight', 1.0)
        obs_weight = self.training_params.get('obs_weight', 1.0)
        imp_weight = self.training_params.get('imp_weight', 1.0)
        dist_type = self.training_params.get('dist_type', 'gaussian')
        self.dist_type = dist_type
        self.criterion = CANDI_LOSS(
            count_weight=count_weight, 
            pval_weight=pval_weight, 
            peak_weight=peak_weight,
            obs_weight=obs_weight,
            imp_weight=imp_weight,
            dist_type=dist_type,
            enable_assay_ema_balance=self.training_params.get('enable_assay_ema_balance', False),
            enable_hier_reduction=self.training_params.get('enable_hier_reduction', False),
            assay_ema_decay=self.training_params.get('assay_ema_decay', 0.99),
            assay_ema_eps=self.training_params.get('assay_ema_eps', 1e-6),
            assay_ema_warmup_steps=self.training_params.get('assay_ema_warmup_steps', 100),
            assay_ema_weight_min=self.training_params.get('assay_ema_weight_min', 0.02),
            assay_ema_weight_max=self.training_params.get('assay_ema_weight_max', 1.0),
            enable_fg_bg_balance=self.training_params.get('enable_fg_bg_balance', False),
            fg_weight=self.training_params.get('fg_weight', 0.5),
            fg_min_fraction=self.training_params.get('fg_min_fraction', 0.02),
            enable_uncertainty_weighting=self.training_params.get('enable_uncertainty_weighting', False),
            uncertainty_warmup_steps=self.training_params.get('uncertainty_warmup_steps', 100),
            uncertainty_init_logvar=self.training_params.get('uncertainty_init_logvar', 0.0),
            enable_count_rstable_objective=self.training_params.get('enable_count_rstable_objective', False),
            count_rstable_eps=self.training_params.get('count_rstable_eps', 1e-6),
            count_rstable_ema_decay=self.training_params.get('count_rstable_ema_decay', 0.99),
            count_rstable_warmup_steps=self.training_params.get('count_rstable_warmup_steps', 100),
            count_rstable_denom_min=self.training_params.get('count_rstable_denom_min', 1e-4),
            count_rstable_r_max=self.training_params.get('count_rstable_r_max', 5.0),
            count_rstable_dispersion_min=self.training_params.get('count_rstable_dispersion_min', 1e-3),
            count_rstable_dispersion_max=self.training_params.get('count_rstable_dispersion_max', 1e4),
        ).to(self.device)
        
        # Initialize optimizer and scheduler (after criterion so optional criterion params can be attached).
        self._setup_optimizer_scheduler()
        if self.criterion.has_uncertainty_params():
            self.optimizer.add_param_group({'params': list(self.criterion.parameters())})
        if self.is_main_process:
            print(f"Loss weights - Count: {count_weight}, P-value: {pval_weight}, Peak: {peak_weight}")
            print(f"Obs/Imp weights - Observed: {obs_weight}, Imputed: {imp_weight}")
            print(f"Signal distribution: {dist_type}")
            print(
                "Loss balancing flags - "
                f"EMA:{self.training_params.get('enable_assay_ema_balance', False)} "
                f"Hier:{self.training_params.get('enable_hier_reduction', False)} "
                f"FGBG:{self.training_params.get('enable_fg_bg_balance', False)} "
                f"Uncertainty:{self.training_params.get('enable_uncertainty_weighting', False)} "
                f"CountRStable:{self.training_params.get('enable_count_rstable_objective', False)}"
            )
        
        # Mixed precision support
        # Mixed precision is opt-in; default to False if not provided.
        self.use_mixed_precision = self.training_params.get('use_mixed_precision', False) and self.device.type == 'cuda'
        if self.use_mixed_precision:
            self.scaler = GradScaler('cuda')
            if self.is_main_process:
                print("Mixed precision training enabled")
        else:
            self.scaler = None
            if self.is_main_process:
                print("Mixed precision training disabled")
        
        # Flags
        self.enable_validation = self.training_params.get('enable_validation', False)
        self.enable_supertrack_train_monitor = self.training_params.get('enable_supertrack_train_monitor', False)
        self.supertrack_train_monitor_every = max(1, int(self.training_params.get('supertrack_train_monitor_every', 100)))
        self.supertrack_train_monitor_max_batch = max(1, int(self.training_params.get('supertrack_train_monitor_max_batch', 8)))
        self.grad_accum_steps = max(1, int(self.training_params.get('grad_accum_steps', 1)))
        self.wandb_log_every = max(1, int(self.training_params.get('wandb_log_every', 50)))
        self.full_metrics_every = self.wandb_log_every
        if self.enable_supertrack_train_monitor:
            self.supertrack_train_monitor_every = math.lcm(
                self.supertrack_train_monitor_every,
                self.full_metrics_every,
            )
        self.last_full_metrics = {}
        
        # Initialize progress tracking
        self.progress_data = []
        self.batch_counter = 0
        self.grad_norm = 0.0
        self.progress_dir = training_params.get('progress_dir', './progress')
        self.progress_file = None  # Will be set when first batch is processed
        self.skipped_k0_samples = 0
        self.skipped_k0_batches = 0
        
        # Create progress directory if it doesn't exist
        if self.is_main_process:
            Path(self.progress_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize checkpoint tracking
        self.checkpoint_dir = None
        self.current_checkpoint_path = None
        self.last_table_lines = 0
        
        # Initialize validation monitoring
        self.val_freq = training_params.get('val_freq', 0.1)
        self.validation_monitor = None  # Will be initialized in _setup if enabled
        self.validation_progress_data = []
        self.validation_progress_file = None
        self.last_validation_batch = 0
        self.last_validation_epoch = -1
        self.last_validation_epoch_batch = 0

    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _compute_latent_kl_term(self, global_step: int):
        """
        Returns (kl_loss, beta) for optional latent KL regularization.
        """
        if not bool(self.training_params.get("enable_latent_kl", False)):
            return None, 0.0

        model_obj = self._unwrap_model()
        if not hasattr(model_obj, "get_last_latent_kl"):
            return None, 0.0

        kl_loss = model_obj.get_last_latent_kl()
        if kl_loss is None:
            return None, 0.0

        kl_weight = float(self.training_params.get("latent_kl_weight", 1e-4))
        if kl_weight <= 0.0:
            return kl_loss, 0.0
        warmup_steps = int(self.training_params.get("latent_kl_warmup_steps", 1000))
        det_warmup_steps = max(0, int(self.training_params.get("latent_deterministic_warmup_steps", 0)))
        phase_a_steps = max(0, int(self.training_params.get("latent_transition_phase_a_steps", 500)))
        phase_b_steps = max(0, int(self.training_params.get("latent_transition_phase_b_steps", 1000)))
        sampling_start_step = det_warmup_steps + phase_a_steps + phase_b_steps
        effective_step = max(0, int(global_step) - sampling_start_step)
        if warmup_steps <= 0:
            beta = kl_weight
        else:
            progress = min(1.0, float(effective_step) / float(warmup_steps))
            beta = kl_weight * progress
        return kl_loss, beta

    def _setup_optimizer_scheduler(self):
        """Setup optimizer and scheduler based on training parameters."""
        lr = self.training_params['learning_rate']
        mom = self.training_params.get('momentum', 0.0)
        b1 = self.training_params.get('beta1', 0.9)
        b2 = self.training_params.get('beta2', 0.999)
        
        # Get weight decay with optimizer-specific defaults
        optimizer_type = self.training_params['optimizer'].lower()
        weight_decay = self.training_params.get('weight_decay')
        
        # Apply optimizer-specific defaults if weight_decay not explicitly set
        if weight_decay is None:
            if optimizer_type == 'adamax' or optimizer_type == 'adam' or optimizer_type == 'radam' or optimizer_type == 'adabelief':
                weight_decay = 0.0  # Conservative default
            elif optimizer_type == 'adamw':
                weight_decay = 0.01  # Conservative default for AdamW (can tune to 0.01-0.1)
            else:  # sgd
                weight_decay = 1e-4 # Conservative default for SGD (can tune to 1e-4 to 1e-5)
        
        # Setup optimizer
        if optimizer_type == 'adamax':
            self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        elif optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay, eps=1e-3)
        elif optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay, eps=1e-3)
        elif optimizer_type == 'radam':
            self.optimizer = torch.optim.RAdam(self.model.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay, eps=1e-3)
        elif optimizer_type == 'adabelief':
            self.optimizer = AdaBelief(self.model.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        elif optimizer_type == 'muon':
            # Muon default momentum is 0.95. If user didn't specify momentum (default 0.9 in args), we might want to respect that or warn.
            # We'll use the passed momentum.
            # We also pass weight_decay for the internal AdamW part of Muon
            adamw_params = {'weight_decay': weight_decay if weight_decay is not None else 0.01}
            self.optimizer = Muon(self.model.parameters(), lr=lr, momentum=mom, adamw_params=adamw_params)
        else:  # sgd
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=mom, weight_decay=weight_decay)
        
        # Scheduler will be created in train() with actual batch counts
        self.scheduler = None

    def _ddp_sync_skip(self, local_skip):
        """Synchronize a local skip decision across all ranks."""
        if isinstance(local_skip, torch.Tensor):
            local_skip = bool(local_skip.detach().item())
        if not self.is_ddp:
            return bool(local_skip)
        flag = torch.tensor(1 if local_skip else 0, device=self.device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def _sync_pending_gradients(self):
        """Synchronize local gradients accumulated under DDP no_sync()."""
        if not self.is_ddp:
            return
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(self.world_size)

    def _apply_optimizer_step(self):
        """Apply one optimizer step (including AMP unscale + clipping) and clear grads."""
        clip_mode = self.training_params.get('clip_mode', 'norm')
        clip_value = self.training_params.get('clip_value', 2.0)

        if self.use_mixed_precision:
            self.scaler.unscale_(self.optimizer)

        if clip_mode == 'norm':
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_value)
        else:
            parameters = [p for p in self.model.parameters() if p.grad is not None]
            if len(parameters) > 0:
                grad_device = parameters[0].grad.device
                grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(grad_device) for p in parameters]), 2.0)
            else:
                grad_norm = torch.tensor(0.0, device=self.device)
            if clip_mode == 'value':
                torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=clip_value)

        if self.use_mixed_precision:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
        self.batch_counter += 1
        self.grad_norm = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
    
    def _setup(self):
        """
        Configure logging, optional validation, and wrap model in DDP if multi-GPU.
        """
        # Wrap model in DDP if multi-GPU
        if self.is_ddp:
            if self.is_main_process:
                print(f"Wrapping model in DDP for {self.world_size} processes")
            
            # Wrap model with DDP
            self.model = DDP(
                self.model,
                device_ids=[self.device.index] if self.device.type == 'cuda' else None,
                output_device=self.device.index if self.device.type == 'cuda' else None,
                # Some decoder/head branches are conditionally unsupervised per batch.
                # DDP must track unused params to avoid reduction desync errors.
                find_unused_parameters=True)

        # Keep W&B lightweight: scalar metrics only, no parameter/gradient distributions.
        if self.is_main_process:
            wandb = _safe_wandb_import()
            if wandb is not None and wandb.run is not None:
                print(f"W&B scalar logging enabled (train metrics every {self.wandb_log_every} batches).")

        # Setup validation if enabled
        if self.enable_validation and self.is_main_process:
            if self.is_main_process:
                print("Validation is enabled")
            try:
                # Initialize EIC_VALIDATION_MONITOR for EIC validation
                from model import EIC_VALIDATION_MONITOR
                # context_length in dataset_params is in bp, but we need it in bins for the monitor
                # The monitor expects bins, so we get it from dataset_params and convert
                context_length_bp = self.dataset_params.get('context_length', 1200 * 25)
                context_length_bins = context_length_bp // 25  # Convert bp to bins
                training_batch_size = self.training_params.get('batch_size', 25)
                dist_type = self.training_params.get('dist_type', 'gaussian')
                self.validation_monitor = EIC_VALIDATION_MONITOR(
                    context_length=context_length_bins,
                    training_batch_size=training_batch_size,
                    device=self.device,
                    dist_type=dist_type
                )
                if self.is_main_process:
                    print("EIC validation monitor setup completed successfully")
                    loci_gen_strategy = str(self.dataset_params.get('loci_gen_strategy', '')).lower()
                    if loci_gen_strategy == 'gw':
                        print(f"Validation frequency: {self.val_freq} (every {100.0 * self.val_freq:.1f}% of each epoch for GW runs)")
                    else:
                        print(f"Validation frequency: {self.val_freq} (every {100.0 * self.val_freq:.1f}% of training)")
            except Exception as e:
                if self.is_main_process:
                    print(f"Warning: Failed to setup EIC_VALIDATION_MONITOR: {e}")
                    import traceback
                    traceback.print_exc()
                self.validation_monitor = None
        elif self.enable_validation:
            self.validation_monitor = None
        else:
            if self.is_main_process:
                print("Validation is disabled")
            self.validation_monitor = None
    
    def train(self):
        """
        Main training entry point. Sets up DataLoader and handles epoch loop.
        """        
        # Setup trainer (logging, DDP, validation)
        self._setup()
        
        # Initialize dataset if not already done
        if self.dataset is None:
            try:
                if self.is_main_process:
                    print("Initializing dataset...")
                dataset_cls = resolve_dataset_class(self.dataset_params)
                self.dataset = dataset_cls(**self.dataset_params)
                if self.is_main_process:
                    print(f"Successfully initialized dataset with {len(self.dataset.aliases['experiment_aliases'])} assays")
            except Exception as e:
                error_msg = f"Failed to initialize dataset: {e}"
                if self.is_main_process:
                    print(f"Error: {error_msg}")
                raise RuntimeError(error_msg) from e
        
        # Calculate estimated batches per epoch for progress tracking
        estimated_batches_per_epoch = self._estimate_batches_per_epoch()
        
        # Setup cosine scheduler with actual batch counts
        if self.scheduler is None:
            self._setup_cosine_scheduler(estimated_batches_per_epoch)
        
        # Create DataLoader with configurable workers and Slurm-aware defaults.
        num_workers, worker_source, worker_budget = resolve_dataloader_workers(
            self.is_ddp,
            self.world_size,
            explicit_workers=self.training_params.get('dataloader_workers'),
        )
        
        dataloader = torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=self.training_params['batch_size'],
            num_workers=num_workers,
            pin_memory=(self.device.type == 'cuda'),
            persistent_workers=(num_workers > 0),
        )
        
        if self.is_main_process:
            budget_msg = f", cpu budget: {worker_budget}" if worker_budget is not None else ""
            print(
                f"Using {num_workers} workers for data loading "
                f"(source: {worker_source}{budget_msg}, DDP: {self.is_ddp})"
            )
            print(f"Gradient accumulation steps: {self.grad_accum_steps}")
            print(f"Full train metrics computed every {self.full_metrics_every} batches")
            if self.enable_supertrack_train_monitor:
                print(f"Train-set supertrack monitor runs every {self.supertrack_train_monitor_every} batches")

        self.optimizer.zero_grad(set_to_none=True)
        
        # Main training loop
        for epoch in range(self.training_params['epochs']):
            # Start epoch
            if self.is_main_process:
                print(f"\nEpoch {epoch+1}/{self.training_params['epochs']}")
            
            # Process batches from CANDIIterableDataset
            batch_count = 0
            micro_batches_since_step = 0
            pending_unsynced_grads = False
            for batch_idx, batch in enumerate(dataloader):
                # Validate batch structure
                local_invalid = not self._validate_batch(batch)
                if self._ddp_sync_skip(local_invalid):
                    if self.is_main_process:
                        print(f"Warning: Skipping invalid batch {batch_idx}")
                    continue
                
                batch_count += 1
                
                # Process the batch
                # try:
                batch_start_time = time.time()
                self.current_epoch = epoch
                self.current_batch_idx = batch_idx
                if hasattr(self, 'estimated_batches_per_epoch') and self.estimated_batches_per_epoch is not None:
                    self.current_global_step = int(epoch * self.estimated_batches_per_epoch + batch_idx)
                else:
                    # Fallback when batch count estimate is unavailable.
                    self.current_global_step = int(getattr(self, 'batch_counter', 0))
                # Latent path control with staged transition:
                # - deterministic warmup: decode from raw z, heads frozen
                # - Phase A: heads unfrozen, still decode from raw z
                # - Phase B: blend decode context z -> mu
                # - then enable sampling
                model_obj = self._unwrap_model()
                if hasattr(model_obj, "set_latent_train_controls"):
                    latent_kl_weight = float(self.training_params.get("latent_kl_weight", 0.0))
                    det_warmup_steps = max(0, int(self.training_params.get("latent_deterministic_warmup_steps", 0)))
                    phase_a_steps = max(0, int(self.training_params.get("latent_transition_phase_a_steps", 500)))
                    phase_b_steps = max(0, int(self.training_params.get("latent_transition_phase_b_steps", 1000)))

                    gs = int(self.current_global_step)
                    phase_a_end = det_warmup_steps + phase_a_steps
                    phase_b_end = phase_a_end + phase_b_steps

                    if latent_kl_weight <= 0.0:
                        force_det = True
                        freeze_heads = True
                        blend_alpha = 0.0
                        enable_sampling = False
                    elif gs < det_warmup_steps:
                        force_det = True
                        freeze_heads = True
                        blend_alpha = 0.0
                        enable_sampling = False
                    elif gs < phase_a_end:
                        force_det = False
                        freeze_heads = False
                        blend_alpha = 0.0
                        enable_sampling = False
                    elif gs < phase_b_end:
                        force_det = False
                        freeze_heads = False
                        phase_b_progress = float(gs - phase_a_end) / float(max(1, phase_b_steps))
                        blend_alpha = max(0.0, min(1.0, phase_b_progress))
                        enable_sampling = False
                    else:
                        force_det = False
                        freeze_heads = False
                        blend_alpha = 1.0
                        enable_sampling = True

                    model_obj.set_latent_train_controls(
                        global_step=int(self.current_global_step),
                        force_deterministic_train=force_det,
                        freeze_posterior_heads=freeze_heads,
                        blend_alpha_train=blend_alpha,
                        enable_sampling_train=enable_sampling,
                    )
                batch = self._move_batch_to_device(batch)
                
                # Process batch with masking, forward pass, and loss computation
                should_step = ((micro_batches_since_step + 1) % self.grad_accum_steps == 0)
                use_no_sync = self.is_ddp and hasattr(self.model, 'no_sync') and (not should_step)
                try:
                    if use_no_sync:
                        with self.model.no_sync():
                            result_dict = self._process_batch(batch, loss_scale=(1.0 / self.grad_accum_steps))
                    else:
                        result_dict = self._process_batch(batch, loss_scale=(1.0 / self.grad_accum_steps))
                except Exception as e:
                    if self.is_main_process:
                        import traceback
                        traceback.print_exc()
                        print(f"Warning: Failed to process batch {batch_idx}: {e}")
                    if self.is_ddp:
                        raise
                    continue

                if self._ddp_sync_skip(result_dict is None):  # Batch was skipped due to errors
                    continue

                micro_batches_since_step += 1
                if use_no_sync:
                    pending_unsynced_grads = True
                did_optimizer_step = False
                if should_step:
                    if pending_unsynced_grads:
                        self._sync_pending_gradients()
                        pending_unsynced_grads = False
                    self._apply_optimizer_step()
                    did_optimizer_step = True

                    # Learning rate scheduling - cosine scheduler steps per optimizer step
                    if self.scheduler is not None:
                        if hasattr(self, 'total_scheduler_steps') and self.current_scheduler_step >= self.total_scheduler_steps:
                            if self.is_main_process and batch_idx % 1000 == 0:
                                print(f"Warning: Scheduler has completed all {self.total_scheduler_steps} steps. Continuing with final learning rate.")
                        else:
                            try:
                                self.scheduler.step()
                                if hasattr(self, 'current_scheduler_step'):
                                    self.current_scheduler_step += 1
                            except (ZeroDivisionError, ValueError) as e:
                                if self.is_main_process:
                                    print(f"Warning: Scheduler step failed: {e}. Continuing with current learning rate.")
                    micro_batches_since_step = 0
                
                # Extract loss and metrics
                loss_keys = ['total_loss', 'obs_count_loss', 'imp_count_loss', 'obs_pval_loss', 'imp_pval_loss', 'obs_peak_loss', 'imp_peak_loss']
                loss_dict = {k: result_dict[k] for k in loss_keys if k in result_dict}
                metrics = {k: v for k, v in result_dict.items() if k not in loss_keys}
                
                # Calculate batch processing time
                batch_processing_time = time.time() - batch_start_time
                    
                # Log batch info for first few batches (only in debug mode)
                if self.is_main_process and batch_idx < 3 and self.training_params.get('debug', False):
                    self._log_batch_info(batch, batch_idx, loss_dict)
                
                # Print batch progress
                if self.is_main_process:
                    try:
                        self._print_batch_log(
                            metrics, loss_dict, batch_idx, epoch, batch_processing_time,
                            optimizer_step=did_optimizer_step
                        )
                    except Exception as e:
                        if self.is_main_process:
                            print(f"Warning: Failed to print batch log: {e}")
                            print(f"Batch info: {batch_idx}, {epoch}, {batch_processing_time}")
                            print(f"Metrics: {metrics}")
                            print(f"Loss dict: {loss_dict}")
                
                # Check if EIC validation should run
                if self.enable_validation and self.validation_monitor is not None and self.is_main_process:
                    # Calculate total batches across all epochs
                    if hasattr(self, 'estimated_batches_per_epoch') and self.estimated_batches_per_epoch is not None:
                        loci_gen_strategy = str(self.dataset_params.get('loci_gen_strategy', '')).lower()
                        is_gw_run = (loci_gen_strategy == 'gw')
                        total_batches = self.estimated_batches_per_epoch * self.training_params['epochs']
                        # Calculate current batch index across all epochs
                        current_batch_idx = epoch * self.estimated_batches_per_epoch + batch_idx
                        if is_gw_run:
                            if self.last_validation_epoch != epoch:
                                self.last_validation_epoch = epoch
                                self.last_validation_epoch_batch = 0

                            epoch_batches = max(1, int(self.estimated_batches_per_epoch))
                            current_epoch_batch = batch_idx + 1
                            progress_pct = current_epoch_batch / epoch_batches
                            interval_batches = max(1, int(np.ceil(epoch_batches * self.val_freq)))
                            should_run_validation = (
                                progress_pct < 1.0 and
                                (current_epoch_batch - self.last_validation_epoch_batch) >= interval_batches
                            )
                        else:
                            progress_pct = current_batch_idx / total_batches if total_batches > 0 else 0.0
                            last_val_pct = self.last_validation_batch / total_batches if total_batches > 0 else 0.0
                            should_run_validation = (progress_pct - last_val_pct >= self.val_freq)

                        if should_run_validation:
                            try:
                                val_results = self.validation_monitor.run_validation(
                                    self.model.module if hasattr(self.model, 'module') else self.model,
                                    current_batch_idx,
                                    total_batches
                                )
                                self._record_validation_progress(val_results)
                                
                                # Log to W&B - keys are already namespaced (val_loss/, val_metrics/, etc.)
                                if self.is_main_process:
                                    wandb_val_data = {k: v for k, v in val_results.items()}
                                    wandb_val_data['epoch'] = epoch + 1
                                    _safe_wandb_log(
                                        wandb_val_data,
                                        step=int(current_batch_idx),
                                    )

                                self._save_validation_progress_csv(epoch)
                                # Save checkpoint after validation
                                if hasattr(self, 'model_name') and self.model_name and not self.training_params.get('no_save', False):
                                    self._save_checkpoint(epoch, self.model_name)
                                self.last_validation_batch = current_batch_idx
                                if is_gw_run:
                                    self.last_validation_epoch_batch = current_epoch_batch
                            except Exception as e:
                                if self.is_main_process:
                                    print(f"Warning: EIC validation failed: {e}")
                                    import traceback
                                    traceback.print_exc()

            # Flush trailing partial accumulation window at epoch end
            if micro_batches_since_step > 0:
                if pending_unsynced_grads:
                    self._sync_pending_gradients()
                    pending_unsynced_grads = False
                if micro_batches_since_step < self.grad_accum_steps:
                    trailing_scale = float(self.grad_accum_steps) / float(micro_batches_since_step)
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad.mul_(trailing_scale)
                self._apply_optimizer_step()
                if self.scheduler is not None:
                    if hasattr(self, 'total_scheduler_steps') and self.current_scheduler_step < self.total_scheduler_steps:
                        try:
                            self.scheduler.step()
                            if hasattr(self, 'current_scheduler_step'):
                                self.current_scheduler_step += 1
                        except (ZeroDivisionError, ValueError) as e:
                            if self.is_main_process:
                                print(f"Warning: Scheduler step failed on trailing accumulation flush: {e}.")
                micro_batches_since_step = 0
                    
            # Run validation at epoch end if enabled (old MONITOR_VALIDATION system)
            validation_summary = None
            if self.enable_validation and hasattr(self, 'validator') and self.validator is not None and self.is_main_process:
                validation_summary, validation_metrics = self._validate()
                
            # End epoch
            if self.is_main_process:
                print(f"Epoch {epoch+1} complete - Processed {batch_count} batches")
            
            # Run validation at end of epoch if enabled
            if self.enable_validation and self.validation_monitor is not None and self.is_main_process:
                try:
                    # Calculate current batch index for validation
                    if hasattr(self, 'estimated_batches_per_epoch') and self.estimated_batches_per_epoch is not None:
                        total_batches = self.estimated_batches_per_epoch * self.training_params['epochs']
                        current_batch_idx = (epoch + 1) * self.estimated_batches_per_epoch  # End of epoch
                        
                        print(f"\n{'='*80}\nRunning EIC Validation at end of epoch {epoch+1}...\n{'='*80}")
                        val_results = self.validation_monitor.run_validation(
                            self.model.module if hasattr(self.model, 'module') else self.model,
                            current_batch_idx,
                            total_batches
                        )
                        self._record_validation_progress(val_results)

                        # Log to W&B - keys are already namespaced (val_loss/, val_metrics/, etc.)
                        if self.is_main_process:
                            wandb_val_data = {k: v for k, v in val_results.items()}
                            wandb_val_data['epoch'] = epoch + 1
                            _safe_wandb_log(
                                wandb_val_data,
                                step=int(current_batch_idx),
                            )

                        self._save_validation_progress_csv(epoch)
                        print(f"{'='*80}\nEIC Validation Complete\n{'='*80}\n")
                    else:
                        # Fallback if estimated_batches_per_epoch not available
                        print(f"\n{'='*80}\nRunning EIC Validation at end of epoch {epoch+1}...\n{'='*80}")
                        val_results = self.validation_monitor.run_validation(
                            self.model.module if hasattr(self.model, 'module') else self.model,
                            epoch * 1000,  # Rough estimate
                            1000 * self.training_params['epochs']
                        )
                        self._record_validation_progress(val_results)

                        # Log to W&B - keys are already namespaced (val_loss/, val_metrics/, etc.)
                        if self.is_main_process:
                            wandb_val_data = {k: v for k, v in val_results.items()}
                            wandb_val_data['epoch'] = epoch + 1
                            _safe_wandb_log(
                                wandb_val_data,
                                step=int(epoch * 1000),
                            )

                        self._save_validation_progress_csv(epoch)
                        print(f"{'='*80}\nEIC Validation Complete\n{'='*80}\n")
                except Exception as e:
                    if self.is_main_process:
                        print(f"Warning: EIC validation at epoch end failed: {e}")
                        import traceback
                        traceback.print_exc()
                
            # Save progress at end of each epoch
            if self.is_main_process and self.progress_data:
                self._save_progress_to_csv(epoch, batch_count)
            
            # Save checkpoint after each epoch (only if saving is enabled)
            if hasattr(self, 'model_name') and self.model_name and not self.training_params.get('no_save', False):
                self._save_checkpoint(epoch, self.model_name)
                
            # Synchronize processes at epoch end if using DDP
            if self.is_ddp:
                dist.barrier()
        
        # Save final progress data
        if self.is_main_process and self.progress_data:
            self._save_progress_to_csv(epoch, batch_count)
            print(f"Final training progress saved with {len(self.progress_data)} total records")
        
        # Clean up final checkpoint since we'll save the final model
        if self.is_main_process and self.current_checkpoint_path and self.current_checkpoint_path.exists():
            self.current_checkpoint_path.unlink()
            print(f"🗑️  Removed final checkpoint: {self.current_checkpoint_path.name}")
        
        return self.model
    
    def _process_batch(self, batch, loss_scale=1.0):
        """
        Process a single batch: apply masking, forward pass, compute loss, and backward pass.
        
        Args:
            batch: Dictionary containing batch data from CANDIIterableDataset
            
        Returns:
            dict: Dictionary containing loss values, or None if batch should be skipped
        """
        # Extract data from batch and convert to correct types
        x_data = batch['x_data'].float()      # [B, L, F] - input signal data (convert to float)
        x_meta = batch['x_meta'].float()      # [B, 4, F] - input metadata (convert to float)
        x_avail = batch['x_avail']            # [B, F] - input availability
        x_dna = batch['x_dna'].float()        # [B, L*25, 4] - input DNA sequence (convert to float)
        
        y_data = batch['y_data'].float()      # [B, L, F] - target signal data (convert to float)
        y_meta = batch['y_meta'].float()      # [B, 4, F] - target metadata (convert to float)
        y_avail = batch['y_avail']            # [B, F] - target availability  
        y_pval = batch['y_pval'].float()      # [B, L, F] - target p-values (convert to float)
        y_peaks = batch['y_peaks'].float()    # [B, L, F] - target peak data (convert to float)
        x_dsf = batch.get('x_dsf', None)      # [B, F] optional per-assay X DSF
        y_dsf = batch.get('y_dsf', None)      # [B, F] optional per-assay Y DSF
        control_x_dsf = batch.get('control_x_dsf', None)  # [B] optional control X DSF

        # if y_avail is not None:
        #     equal = torch.equal(x_avail, y_avail)
        #     print(f"Sum x_avail: {x_avail.sum().item()} | Sum y_avail: {y_avail.sum().item()} | x_avail == y_avail: {equal}")
        # else:
        #     print(f"Sum x_avail: {x_avail.sum().item()} | y_avail is None")

        control_data = batch['control_data'].float()   # [B, L, 1] - control signal data (convert to float)
        control_meta = batch['control_meta'].float()   # [B, 4, 1] - control metadata (convert to float)
        control_avail = batch['control_avail']         # [B, 1] - control availability
        
        # Apply reverse complement augmentation with specified probability
        reverse_complement_prob = self.training_params.get('reverse_complement_prob', 0.5)
        if torch.rand(1).item() < reverse_complement_prob:
            print("Applying reverse complement augmentation")
            # Reverse complement DNA sequences
            x_dna = reverse_complement_dna(x_dna)
            x_data = reverse_signal(x_data)

            y_data = reverse_signal(y_data)
            y_pval = reverse_signal(y_pval)
            y_peaks = reverse_signal(y_peaks)
            control_data = reverse_signal(control_data)            
        
        # Apply masking to create imputation targets
        # Use the masker to create masked inputs
        if not hasattr(self, 'masker'):
            # Initialize masker if not already done
            token_dict = {"missing_mask": -1, "cloze_mask": -2, "pad": -3}
            p_full_loci = self.training_params.get('p_full_loci', 0.0)
            p_full_assay = self.training_params.get('p_full_assay', 1.0)
            p_chunks = self.training_params.get('p_chunks', 0.0)
            mask_fraction = self.training_params.get('mask_fraction', 0.20)
            chunk_size = self.training_params.get('chunk_size', 40)
            self.masker = DataMasker(
                mask_value=token_dict["cloze_mask"],
                chunk_size=chunk_size,
                mask_fraction=mask_fraction,
                p_full_loci=p_full_loci,
                p_full_assay=p_full_assay,
                p_chunks=p_chunks
            )
        
        # Apply masking - this modifies x_data, x_meta, x_avail in place
        # The masker expects inputs in a specific format, so we need to handle batch dimension
        B, L, F = x_data.shape
        
        # Clone inputs for masking
        x_data_masked = x_data.clone()
        x_meta_masked = x_meta.clone() 
        x_avail_masked = x_avail.clone()
        
        # Get signal_dim from dataset_params or calculate from aliases
        if hasattr(self.dataset, 'signal_dim'):
            signal_dim = self.dataset.signal_dim 
        elif hasattr(self.dataset, 'aliases'):
            signal_dim = len(self.dataset.aliases['experiment_aliases'])
        else:
            # Fallback - get from dataset_params
            signal_dim = self.dataset_params.get('signal_dim', 35)  # Default to 35 for EIC
        
        # Apply masking using the new probability-based strategy
        # DataMasker.apply_mask handles: full_assay, full_loci, and/or chunks masking
        x_data_masked, x_meta_masked, x_avail_masked = self.masker.apply_mask(
            x_data_masked, x_meta_masked, x_avail_masked
        )

        # Create masks for loss computation
        token_dict = {"missing_mask": -1, "cloze_mask": -2, "pad": -3}
        masked_map = (x_data_masked == token_dict["cloze_mask"])  # Imputation targets
        observed_map = (x_data_masked != token_dict["missing_mask"]) & (x_data_masked != token_dict["cloze_mask"])  # Upsampling targets

        observed_map = observed_map.clone()
        masked_map = masked_map.clone()

        # Enforce GT-valid supervision scope with y-side availability.
        y_avail_bool_cpu = (y_avail > 0)
        y_avail_expand_cpu = y_avail_bool_cpu.unsqueeze(1).expand(-1, observed_map.shape[1], -1)
        observed_map = observed_map & y_avail_expand_cpu
        masked_map = masked_map & y_avail_expand_cpu

        # Query set definition and safety assertion:
        # k_b includes assays with any observed or masked supervision in the window.
        sup_any = observed_map.any(dim=1) | masked_map.any(dim=1)  # [B, F]
        if not torch.equal(y_avail_bool_cpu, sup_any):
            mismatch = (y_avail_bool_cpu != sup_any).sum().item()
            raise ValueError(
                f"Availability/supervision mismatch detected ({mismatch} assay positions). "
                "Expected (y_avail==1) == (observed_any || masked_any)."
            )

        query_mask = y_avail_bool_cpu & sup_any
        if y_dsf is not None:
            query_mask_signal = query_mask & (y_dsf == 1)
        else:
            query_mask_signal = query_mask

        if self.is_main_process:
            print(
                f"[QREQ1_QUERY_MASK] y_avail_sum={int(y_avail_bool_cpu.sum().item())} "
                f"sup_any_sum={int(sup_any.sum().item())} "
                f"query_sum={int(query_mask.sum().item())} "
                f"query_signal_sum={int(query_mask_signal.sum().item())}"
            )

        # Pathological handling: skip per-sample k_b==0 and track.
        valid_samples = query_mask.any(dim=1)
        if (~valid_samples).any():
            skipped = int((~valid_samples).sum().item())
            self.skipped_k0_samples = getattr(self, 'skipped_k0_samples', 0) + skipped
            if self.is_main_process:
                print(f"Warning: skipping {skipped} sample(s) with k_b=0 (no queryable assays).")
                print(
                    f"[QREQ5_K0] skipped_k0_samples_total={self.skipped_k0_samples} "
                    f"current_batch={getattr(self, 'current_batch_idx', -1)}"
                )
            local_all_invalid = (valid_samples.sum().item() == 0)
            if self._ddp_sync_skip(local_all_invalid):
                self.skipped_k0_batches = getattr(self, 'skipped_k0_batches', 0) + 1
                if self.is_main_process:
                    print(f"[QREQ5_K0] skipped_k0_batches_total={self.skipped_k0_batches}")
                return None

            # Filter all batch-aligned tensors to keep only valid samples.
            x_data_masked = x_data_masked[valid_samples]
            x_meta_masked = x_meta_masked[valid_samples]
            x_avail_masked = x_avail_masked[valid_samples]
            x_dna = x_dna[valid_samples]
            y_data = y_data[valid_samples]
            y_meta = y_meta[valid_samples]
            y_avail = y_avail[valid_samples]
            y_pval = y_pval[valid_samples]
            y_peaks = y_peaks[valid_samples]
            control_data = control_data[valid_samples]
            control_meta = control_meta[valid_samples]
            control_avail = control_avail[valid_samples]
            observed_map = observed_map[valid_samples]
            masked_map = masked_map[valid_samples]
            query_mask = query_mask[valid_samples]
            query_mask_signal = query_mask_signal[valid_samples]
            if y_dsf is not None:
                y_dsf = y_dsf[valid_samples]
            if x_dsf is not None:
                x_dsf = x_dsf[valid_samples]

        # Nq debug stats (pre-device move) to diagnose batch-to-batch memory variability.
        nq_count = int(query_mask.sum().item())
        nq_signal = int(query_mask_signal.sum().item())
        k_per_sample = query_mask.sum(dim=1).float()
        k_mean = float(k_per_sample.mean().item()) if k_per_sample.numel() > 0 else 0.0
        k_max = int(k_per_sample.max().item()) if k_per_sample.numel() > 0 else 0

        # Store masking info for logging
        self.last_masking_probs = self.masker.get_probabilities()

        x_data_masked = torch.cat([x_data_masked, control_data], dim=2)      # (B, L, F+1)
        x_meta_masked = torch.cat([x_meta_masked, control_meta], dim=2)      # (B, 4, F+1)
        x_avail_masked = torch.cat([x_avail_masked, control_avail], dim=1)   # (B, F+1)
        
        # Validate that we have observed regions (we need at least some data to train on)
        local_skip_no_observed = not observed_map.any()
        if self._ddp_sync_skip(local_skip_no_observed):
            if self.is_main_process:
                print("Warning: No observed regions found! Skipping batch...")
            return None
        
        # If no regions were masked, we can still do training (just no imputation loss)
        # This is fine - the model can learn from reconstruction loss on observed data
        has_masked_regions = masked_map.any()
        
        # Move masks to device
        masked_map = masked_map.to(self.device)
        observed_map = observed_map.to(self.device)
        query_mask = query_mask.to(self.device)
        query_mask_signal = query_mask_signal.to(self.device)

        # Per-assay gating for signal/peak heads (Solution 3):
        # only supervise pval/peaks where y_dsf == 1 and assay is available.
        signal_observed_map = observed_map
        signal_masked_map = masked_map
        if y_dsf is not None:
            y_dsf = y_dsf.to(self.device)
            y_avail = y_avail.to(self.device)
            sig_ok = (y_dsf == 1) & (y_avail > 0)
            sig_ok = sig_ok.unsqueeze(1).expand(-1, observed_map.shape[1], -1)
            signal_observed_map = observed_map & sig_ok
            signal_masked_map = masked_map & sig_ok

        if self.is_main_process:
            print(
                f"[NQ_DEBUG] batch_ok Nq_count={nq_count} Nq_signal={nq_signal} "
                f"k_mean={k_mean:.2f} k_max={k_max}"
            )
        
        try:
            # Forward pass through model with mixed precision
            # Model now returns 6 values: p, n, mu, scale, df, peak
            if self.use_mixed_precision:
                with autocast('cuda'):
                    if self.training_params.get('DNA', True):
                        # Model expects DNA sequence
                        output_p, output_n, output_mu, output_var, output_df, output_peak = self.model(
                            x_data_masked, x_dna, x_meta_masked, y_meta,
                            query_mask=query_mask, query_mask_signal=query_mask_signal
                        )
                    else:
                        raise ValueError("DNA must be True for CANDI_TRAINER")
            else:
                if self.training_params.get('DNA', True):
                    # Model expects DNA sequence
                    output_p, output_n, output_mu, output_var, output_df, output_peak = self.model(
                        x_data_masked, x_dna, x_meta_masked, y_meta,
                        query_mask=query_mask, query_mask_signal=query_mask_signal
                    )
                else:
                    raise ValueError("DNA must be True for CANDI_TRAINER")
                    
        except RuntimeError as e:
            local_oom = ("out of memory" in str(e).lower())
            if self._ddp_sync_skip(local_oom):
                if self.is_main_process:
                    print(f"Warning: CUDA Out of Memory! Batch size: {B}, Sequence length: {L}, Features: {F}. Skipping batch and clearing cache...")
                    print(
                        f"[NQ_DEBUG] batch_oom Nq_count={nq_count} Nq_signal={nq_signal} "
                        f"k_mean={k_mean:.2f} k_max={k_max}"
                    )
                torch.cuda.empty_cache()
                return None
            else:
                raise e

        # Query-mode structural assertions: verify strict sparse scope and scatter-back placement.
        is_query_decoder = bool(getattr(self.model, "query_decoder", False))
        emit_qreq_debug = self.is_main_process and (
            self.training_params.get('debug', False)
            or getattr(self, 'current_batch_idx', 0) < 5
            or (getattr(self, 'current_batch_idx', 0) % 200 == 0)
        )
        if is_query_decoder:
            L_out = output_p.shape[1]
            query_expand = query_mask.unsqueeze(1).expand(-1, L_out, -1)
            query_signal_expand = query_mask_signal.unsqueeze(1).expand(-1, L_out, -1)

            expected_count_tokens = int(query_expand.sum().item())
            expected_signal_tokens = int(query_signal_expand.sum().item())

            active_p = int((output_p != -1).sum().item())
            active_n = int((output_n != -1).sum().item())
            active_mu = int((output_mu != -1).sum().item())
            active_var = int((output_var != -1).sum().item())
            active_peak = int((output_peak != -1).sum().item())

            # Strict sparse scope: active lanes exactly match query-selected lanes.
            if active_p != expected_count_tokens or active_n != expected_count_tokens:
                raise ValueError(
                    f"Count sparse scope mismatch: expected={expected_count_tokens}, "
                    f"active_p={active_p}, active_n={active_n}"
                )
            if active_mu != expected_signal_tokens or active_var != expected_signal_tokens or active_peak != expected_signal_tokens:
                raise ValueError(
                    f"Signal sparse scope mismatch: expected={expected_signal_tokens}, "
                    f"active_mu={active_mu}, active_var={active_var}, active_peak={active_peak}"
                )
            if output_df is not None:
                active_df = int((output_df != -1).sum().item())
                if active_df != expected_signal_tokens:
                    raise ValueError(
                        f"StudentT df sparse scope mismatch: expected={expected_signal_tokens}, active_df={active_df}"
                    )

            # Scatter correctness: no non-sentinel values outside queried lanes.
            nonquery_count_non_sentinel = int(((~query_expand) & (output_p != -1)).sum().item())
            nonquery_signal_non_sentinel = int(((~query_signal_expand) & (output_mu != -1)).sum().item())
            if nonquery_count_non_sentinel != 0 or nonquery_signal_non_sentinel != 0:
                raise ValueError(
                    f"Scatter leak detected: count_leak={nonquery_count_non_sentinel}, "
                    f"signal_leak={nonquery_signal_non_sentinel}"
                )

            if emit_qreq_debug:
                print(
                    f"[QREQ2_SPARSE_SCOPE] expected_count={expected_count_tokens} active_p={active_p} "
                    f"active_n={active_n} expected_signal={expected_signal_tokens} active_mu={active_mu} "
                    f"active_var={active_var} active_peak={active_peak}"
                )
                print(
                    f"[QREQ3_SCATTER] count_leak={nonquery_count_non_sentinel} "
                    f"signal_leak={nonquery_signal_non_sentinel}"
                )
        
        # Validate model outputs before loss computation
        has_nan_outputs = (torch.isnan(output_p).any() or torch.isnan(output_n).any() or 
                          torch.isnan(output_mu).any() or torch.isnan(output_var).any() or 
                          torch.isnan(output_peak).any())
        if output_df is not None:
            has_nan_outputs = has_nan_outputs or torch.isnan(output_df).any()
        
        if self._ddp_sync_skip(bool(has_nan_outputs.detach().item()) if isinstance(has_nan_outputs, torch.Tensor) else bool(has_nan_outputs)):
            if self.is_main_process:
                print("Warning: NaN in model outputs! Skipping batch...")
            return None

        # Sentinel safety assertions: non-queried lanes may be -1, but they must never be supervised.
        def _assert_no_sentinel_under_mask(pred, mask, name):
            overlap = ((pred == -1) & mask).any()
            if overlap:
                raise ValueError(f"Sentinel overlap in supervised region for {name}.")
            finite_supervised = torch.isfinite(pred[mask]).all() if mask.any() else True
            if not finite_supervised:
                raise ValueError(f"Non-finite predictions in supervised region for {name}.")

        _assert_no_sentinel_under_mask(output_p, observed_map | masked_map, "count/p")
        _assert_no_sentinel_under_mask(output_n, observed_map | masked_map, "count/n")
        _assert_no_sentinel_under_mask(output_mu, signal_observed_map | signal_masked_map, "signal/mu")
        _assert_no_sentinel_under_mask(output_var, signal_observed_map | signal_masked_map, "signal/scale")
        _assert_no_sentinel_under_mask(output_peak, signal_observed_map | signal_masked_map, "peak")
        if output_df is not None:
            _assert_no_sentinel_under_mask(output_df, signal_observed_map | signal_masked_map, "signal/df")
        if emit_qreq_debug:
            print(
                "[QREQ4_SENTINEL_LOSS] passed count/signal supervised masks: "
                "no sentinel overlap, all finite under supervision."
            )
        
        # DEBUG: Check y_pval statistics
        # if self.is_main_process:
        #     y_pval_flat = y_pval.flatten()
        #     print(f"[DEBUG] y_pval stats: min={y_pval_flat.min().item():.4f}, max={y_pval_flat.max().item():.4f}, "
        #           f"mean={y_pval_flat.mean().item():.4f}, std={y_pval_flat.std().item():.4f}")
            
        #     # Check model outputs stats
        #     mu_flat = output_mu.detach().flatten()
        #     var_flat = output_var.detach().flatten()
        #     print(f"[DEBUG] output_mu stats: min={mu_flat.min().item():.4f}, max={mu_flat.max().item():.4f}, mean={mu_flat.mean().item():.4f}")
        #     print(f"[DEBUG] output_var stats: min={var_flat.min().item():.4e}, max={var_flat.max().item():.4e}, mean={var_flat.mean().item():.4e}")
            
        #     if y_pval_flat.max().item() > 100:
        #         print(f"[DEBUG] WARNING: Max y_pval > 100 ({y_pval_flat.max().item():.4f}). "
        #               "This might indicate untransformed data!")
        
        try:
            # Compute losses using CANDI_LOSS with mixed precision
            # CANDI_LOSS.forward now takes df_pred as the 5th argument
            loss_global_step = int(getattr(self, 'current_global_step', getattr(self, 'current_batch_idx', 0)))
            if self.use_mixed_precision:
                with autocast('cuda'):
                    if has_masked_regions:
                        # Normal case: compute all losses
                        obs_count_loss, imp_count_loss, obs_pval_loss, imp_pval_loss, obs_peak_loss, imp_peak_loss = self.criterion(
                            output_p, output_n, output_mu, output_var, output_df, output_peak,
                            y_data, y_pval, y_peaks, observed_map, masked_map,
                            signal_observed_map, signal_masked_map,
                            global_step=loss_global_step
                        )
                        total_loss = obs_count_loss + obs_pval_loss + obs_peak_loss + imp_count_loss + imp_pval_loss + imp_peak_loss
                    else:
                        # No masked regions: only compute observed losses
                        obs_count_loss, _, obs_pval_loss, _, obs_peak_loss, _ = self.criterion(
                            output_p, output_n, output_mu, output_var, output_df, output_peak,
                            y_data, y_pval, y_peaks, observed_map, observed_map,  # Use observed_map for count maps
                            signal_observed_map, signal_observed_map,
                            global_step=loss_global_step
                        )
                        imp_count_loss = torch.tensor(0.0, device=self.device)
                        imp_pval_loss = torch.tensor(0.0, device=self.device)
                        imp_peak_loss = torch.tensor(0.0, device=self.device)
                        total_loss = obs_count_loss + obs_pval_loss + obs_peak_loss
            else:
                if has_masked_regions:
                    # Normal case: compute all losses
                    obs_count_loss, imp_count_loss, obs_pval_loss, imp_pval_loss, obs_peak_loss, imp_peak_loss = self.criterion(
                        output_p, output_n, output_mu, output_var, output_df, output_peak,
                        y_data, y_pval, y_peaks, observed_map, masked_map,
                        signal_observed_map, signal_masked_map,
                        global_step=loss_global_step
                    )
                    total_loss = obs_count_loss + obs_pval_loss + obs_peak_loss + imp_count_loss + imp_pval_loss + imp_peak_loss
                else:
                    # No masked regions: only compute observed losses
                    obs_count_loss, _, obs_pval_loss, _, obs_peak_loss, _ = self.criterion(
                        output_p, output_n, output_mu, output_var, output_df, output_peak,
                        y_data, y_pval, y_peaks, observed_map, observed_map,  # Use observed_map for count maps
                        signal_observed_map, signal_observed_map,
                        global_step=loss_global_step
                    )
                    imp_count_loss = torch.tensor(0.0, device=self.device)
                    imp_pval_loss = torch.tensor(0.0, device=self.device)
                    imp_peak_loss = torch.tensor(0.0, device=self.device)
                    total_loss = obs_count_loss + obs_pval_loss + obs_peak_loss

            latent_kl_loss = None
            latent_beta = 0.0
            latent_kl_weighted = None
            latent_kl_loss, latent_beta = self._compute_latent_kl_term(
                global_step=int(getattr(self, 'current_global_step', getattr(self, 'current_batch_idx', 0)))
            )
            if latent_kl_loss is not None and latent_beta > 0.0:
                latent_kl_weighted = latent_kl_loss * float(latent_beta)
                total_loss = total_loss + latent_kl_weighted

            if self.is_main_process and (getattr(self, 'current_batch_idx', 0) % 200 == 0):
                dbg = self.criterion.get_debug_stats() if hasattr(self.criterion, "get_debug_stats") else {}
                if dbg:
                    debug_keys = [
                        "logvar_count", "logvar_pval", "logvar_peak",
                        "assay_w_min_h0_b0", "assay_w_max_h0_b0",
                        "assay_w_min_h1_b1", "assay_w_max_h1_b1",
                        "fgbg_used_h0_b0", "fgbg_total_h0_b0",
                        "rstable_obs_lmodel_mean", "rstable_obs_lnull_mean", "rstable_obs_loracle_mean",
                        "rstable_obs_denom_min_abs", "rstable_obs_denom_clamp_hits",
                        "rstable_imp_lmodel_mean", "rstable_imp_lnull_mean", "rstable_imp_loracle_mean",
                        "rstable_imp_denom_min_abs", "rstable_imp_denom_clamp_hits",
                        "count_rstable_enabled", "count_rstable_active", "count_rstable_warmup_steps",
                    ]
                    compact = {k: dbg[k] for k in debug_keys if k in dbg}
                    print(f"[LOSS_BAL_DEBUG] step={getattr(self, 'current_batch_idx', 0)} stats={compact}")
            
            # Check for NaN losses with detailed debugging
            local_nan_loss = bool((torch.isnan(total_loss).sum() > 0).detach().item())
            if self._ddp_sync_skip(local_nan_loss):
                if self.is_main_process:
                    print("Warning: Encountered NaN loss! Skipping batch...")
                return None
            
            # Backward pass (scaled for gradient accumulation)
            backward_loss = total_loss * float(loss_scale)
            if self.use_mixed_precision:
                self.scaler.scale(backward_loss).backward()
            else:
                backward_loss.float().backward()
                
        except RuntimeError as e:
            local_oom = ("out of memory" in str(e).lower())
            if self._ddp_sync_skip(local_oom):
                if self.is_main_process:
                    print("Warning: CUDA Out of Memory during loss computation! Skipping batch and clearing cache...")
                torch.cuda.empty_cache()
                return None
            else:
                raise e
        
        current_batch = getattr(self, 'current_batch_idx', 0)
        should_compute_full_metrics = (
            current_batch == 0
            or (((current_batch + 1) % self.full_metrics_every) == 0)
        )

        # Compute full train metrics only on the logging cadence and reuse the
        # last computed snapshot between logging steps to avoid wasted CPU work.
        if should_compute_full_metrics:
            if has_masked_regions:
                metrics = self._compute_metrics(
                    output_p, output_n, output_mu, output_var, output_peak,
                    y_data, y_pval, y_peaks, observed_map, masked_map
                )
            else:
                # Only compute observed metrics when no masking occurred
                metrics = self._compute_metrics(
                    output_p, output_n, output_mu, output_var, output_peak,
                    y_data, y_pval, y_peaks, observed_map, torch.zeros_like(observed_map)  # Empty mask for imputation metrics
                )
            self.last_full_metrics = dict(metrics)
        else:
            metrics = dict(getattr(self, 'last_full_metrics', {}))

        # DSF transition observability (available-assay scope).
        if x_dsf is not None and y_dsf is not None:
            x_dsf_cpu = x_dsf.detach().cpu()
            y_dsf_cpu = y_dsf.detach().cpu()
            y_avail_cpu = y_avail.detach().cpu()
            valid_assays = (y_avail_cpu > 0) & (x_dsf_cpu > 0) & (y_dsf_cpu > 0)
            # By convention here, lower DSF => higher effective depth.
            # So y_dsf < x_dsf is an upsampling transition (e.g., 4->1).
            up_count = int(((y_dsf_cpu < x_dsf_cpu) & valid_assays).sum().item())
            down_count = int(((y_dsf_cpu > x_dsf_cpu) & valid_assays).sum().item())
            same_count = int(((y_dsf_cpu == x_dsf_cpu) & valid_assays).sum().item())
            metrics['dsf_transitions/upsampled_count'] = up_count
            metrics['dsf_transitions/downsampled_count'] = down_count
            metrics['dsf_transitions/same_count'] = same_count
        else:
            metrics['dsf_transitions/upsampled_count'] = 0
            metrics['dsf_transitions/downsampled_count'] = 0
            metrics['dsf_transitions/same_count'] = 0

        if control_x_dsf is not None:
            control_x_dsf_cpu = control_x_dsf.detach().cpu()
            control_non1_count = int(((control_x_dsf_cpu > 0) & (control_x_dsf_cpu != 1)).sum().item())
            metrics['dsf_transitions/control_non1_count'] = control_non1_count
        else:
            metrics['dsf_transitions/control_non1_count'] = 0
        
        # Optional training-set prompt sensitivity monitor (strictly inference-only, no gradient leakage)
        if self.enable_supertrack_train_monitor:
            should_run_monitor = (
                should_compute_full_metrics
                and (
                    current_batch == 0
                    or (((current_batch + 1) % self.supertrack_train_monitor_every) == 0)
                )
            )
            if should_run_monitor:
                st_metrics = self._monitor_supertrack_on_batch(
                    x_data_masked=x_data_masked,
                    x_dna=x_dna,
                    x_meta_masked=x_meta_masked,
                    y_meta=y_meta,
                    y_avail=y_avail
                )
                metrics.update(st_metrics)

        # Loss-module running stats (EMA weights, R_stable diagnostics, uncertainty logvars).
        # These are exported under a dedicated namespace so they appear in W&B automatically.
        if hasattr(self.criterion, "get_debug_stats"):
            dbg = self.criterion.get_debug_stats() or {}
            for k, v in dbg.items():
                if isinstance(v, (int, float, np.floating, np.integer)):
                    metrics[f"loss_stats/{k}"] = float(v)
        if latent_kl_loss is not None:
            metrics["loss_stats/latent_kl"] = float(latent_kl_loss.detach().item())
            metrics["loss_stats/latent_beta"] = float(latent_beta)
            if latent_kl_weighted is not None:
                metrics["loss_stats/latent_kl_weighted"] = float(latent_kl_weighted.detach().item())
            model_obj = self._unwrap_model()
            if hasattr(model_obj, "get_last_latent_stats"):
                lstats = model_obj.get_last_latent_stats() or {}
                for k, v in lstats.items():
                    if isinstance(v, (int, float)):
                        metrics[f"loss_stats/{k}"] = float(v)

        # Return loss dictionary for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'obs_count_loss': obs_count_loss.item(),
            'imp_count_loss': imp_count_loss.item(), 
            'obs_pval_loss': obs_pval_loss.item(),
            'imp_pval_loss': imp_pval_loss.item(),
            'obs_peak_loss': obs_peak_loss.item(),
            'imp_peak_loss': imp_peak_loss.item(),
            'skipped_k0_samples': float(getattr(self, 'skipped_k0_samples', 0)),
            'skipped_k0_batches': float(getattr(self, 'skipped_k0_batches', 0)),
        }
        
        # Update progress monitoring and check for LR adjustment
        self._update_progress_monitoring(metrics, loss_dict, self.training_params.get('specific_ema_alpha', 0.005))
        
        # Log to W&B
        if self.is_main_process:
            current_epoch = getattr(self, 'current_epoch', 0)
            current_batch = getattr(self, 'current_batch_idx', 0)
            current_lr = self.optimizer.param_groups[0]['lr']
            should_log_to_wandb = (current_batch == 0) or (((current_batch + 1) % self.wandb_log_every) == 0)

            if should_log_to_wandb:
                log_data = {
                    "epoch": current_epoch + 1,
                    "batch": current_batch,
                    "lr": current_lr,
                    "grad_norm": getattr(self, 'grad_norm', 0.0),
                    **loss_dict,
                    **metrics
                }
                
                # Log EMA trends if available
                if hasattr(self, 'specific_ema'):
                    for k, v in self.specific_ema.items():
                        log_data[f"ema/{k}"] = v

                _safe_wandb_log(
                    log_data,
                    step=int(getattr(self, 'current_global_step', current_batch)),
                )
        
        # Add metrics to return dictionary
        return_dict = {**loss_dict, **metrics}
        
        return return_dict

    def _monitor_supertrack_on_batch(self, x_data_masked, x_dna, x_meta_masked, y_meta, y_avail=None):
        """
        Run training-set supertrack checks in strict inference mode.

        Sentinel and availability safety:
        - Never perturb entries with -1 (unavailable) or -2 (cloze token).
        - Apply perturbations only on valid assay entries.

        Returns:
            dict with keys:
                train_st/depth_ratio
                train_st/runtype_mse
                train_st/readlen_mse
        """
        results = {
            'train_st/depth_ratio': np.nan,
            'train_st/runtype_mse': np.nan,
            'train_st/readlen_mse': np.nan
        }

        if y_meta.ndim != 3 or y_meta.shape[1] < 4:
            return results

        subset_size = min(self.supertrack_train_monitor_max_batch, x_data_masked.shape[0])
        if subset_size <= 0:
            return results

        # Detached clones to avoid any graph reuse/side effects.
        x_sub = x_data_masked[:subset_size].detach().clone()
        x_dna_sub = x_dna[:subset_size].detach().clone()
        x_meta_sub = x_meta_masked[:subset_size].detach().clone()
        y_base = y_meta[:subset_size].detach().clone()

        if y_avail is not None and isinstance(y_avail, torch.Tensor) and y_avail.ndim == 2:
            base_available = (y_avail[:subset_size] > 0)
        else:
            base_available = torch.ones_like(y_base[:, 0, :], dtype=torch.bool)

        depth_valid = base_available & (y_base[:, 0, :] != -1) & (y_base[:, 0, :] != -2)
        runtype_valid = base_available & (y_base[:, 3, :] != -1) & (y_base[:, 3, :] != -2)
        readlen_valid = base_available & (y_base[:, 2, :] != -1) & (y_base[:, 2, :] != -2)

        def _run_nb_mean(y_prompt):
            with torch.inference_mode():
                with autocast('cuda', enabled=self.use_mixed_precision):
                    outputs_p, outputs_n, _, _, _, _ = self.model(x_sub, x_dna_sub, x_meta_sub, y_prompt)
                    return (outputs_n * (1 - outputs_p)) / torch.clamp(outputs_p, min=1e-6)

        def _expand_valid(mask_2d, output_3d):
            return mask_2d.unsqueeze(1).expand(-1, output_3d.shape[1], -1)

        was_training = self.model.training
        self.model.eval()
        try:
            # Check 1: Depth ratio
            y_low = y_base.clone()
            y_high = y_base.clone()
            y_low[:, 0, :] = torch.where(depth_valid, torch.full_like(y_low[:, 0, :], 23.0), y_low[:, 0, :])
            y_high[:, 0, :] = torch.where(depth_valid, torch.full_like(y_high[:, 0, :], 25.0), y_high[:, 0, :])

            nb_low = _run_nb_mean(y_low)
            nb_high = _run_nb_mean(y_high)
            depth_mask = _expand_valid(depth_valid, nb_low) & torch.isfinite(nb_low) & torch.isfinite(nb_high) & (nb_low > 0) & (nb_high > 0)
            if depth_mask.any():
                denom = nb_low[depth_mask].sum()
                if torch.abs(denom) > 1e-8:
                    results['train_st/depth_ratio'] = float((nb_high[depth_mask].sum() / denom).item())

            # Check 2: RunType MSE
            y_single = y_base.clone()
            y_paired = y_base.clone()
            y_single[:, 3, :] = torch.where(runtype_valid, torch.full_like(y_single[:, 3, :], 0.0), y_single[:, 3, :])
            y_paired[:, 3, :] = torch.where(runtype_valid, torch.full_like(y_paired[:, 3, :], 1.0), y_paired[:, 3, :])

            nb_single = _run_nb_mean(y_single)
            nb_paired = _run_nb_mean(y_paired)
            run_mask = _expand_valid(runtype_valid, nb_single) & torch.isfinite(nb_single) & torch.isfinite(nb_paired)
            if run_mask.any():
                results['train_st/runtype_mse'] = float(((nb_single[run_mask] - nb_paired[run_mask]) ** 2).mean().item())

            # Check 3: Read length MSE
            y_short = y_base.clone()
            y_long = y_base.clone()
            y_short[:, 2, :] = torch.where(readlen_valid, torch.full_like(y_short[:, 2, :], 36.0), y_short[:, 2, :])
            y_long[:, 2, :] = torch.where(readlen_valid, torch.full_like(y_long[:, 2, :], 100.0), y_long[:, 2, :])

            nb_short = _run_nb_mean(y_short)
            nb_long = _run_nb_mean(y_long)
            read_mask = _expand_valid(readlen_valid, nb_short) & torch.isfinite(nb_short) & torch.isfinite(nb_long)
            if read_mask.any():
                results['train_st/readlen_mse'] = float(((nb_short[read_mask] - nb_long[read_mask]) ** 2).mean().item())

        except Exception as e:
            if self.is_main_process:
                print(f"Warning: train-set supertrack monitor failed: {e}")
        finally:
            if was_training:
                self.model.train()

        return results
    
    def _estimate_batches_per_epoch(self):
        """
        Estimate the number of batches per epoch based on dataset parameters.
        
        The total number of samples per epoch is:
        num_biosamples × num_loci × num_dsf_factors × num_chromosomes
        
        Then divided by batch_size to get number of batches.
        
        Returns:
            int: Estimated number of batches per epoch, or None if cannot estimate
        """
        try:
            # Create a temporary dataset to get the actual counts
            dataset_cls = resolve_dataset_class(self.dataset_params)
            temp_dataset = dataset_cls(**self.dataset_params)
            
            # Setup the data looper to get actual counts
            temp_dataset.setup_datalooper(
                m=self.dataset_params.get('m', 1000),
                context_length=self.dataset_params.get('context_length', 30000),
                bios_batchsize=1,  # CANDIIterableDataset always uses 1
                loci_batchsize=1,  # CANDIIterableDataset always uses 1
                loci_gen_strategy=self.dataset_params.get('loci_gen_strategy', 'random'),
                ccre_fraction=self.dataset_params.get('ccre_fraction', 0.3),
                split=self.dataset_params.get('split', 'train'),
                shuffle_bios=self.dataset_params.get('shuffle_bios', True),
                dsf_list=self.dataset_params.get('dsf_list', [1, 2]),
                includes=self.dataset_params.get('includes'),
                excludes=self.dataset_params.get('excludes', []),
                must_have_chr_access=self.dataset_params.get('must_have_chr_access', False), 
                bios_min_exp_avail_threshold=self.dataset_params.get('bios_min_exp_avail_threshold', 0),
                balanced_bios_order=self.dataset_params.get('balanced_bios_order', True)
            )
            
            # Get the actual counts after setup
            num_biosamples = len(temp_dataset.navigation)
            num_loci = temp_dataset.num_regions  # This is the number of loci (m_regions)
            num_dsf_factors = len(temp_dataset.dsf_list)
            num_chromosomes = len(temp_dataset.loci.keys()) if hasattr(temp_dataset, 'loci') else 1
            
            # Account for DDP and DataLoader workers sharding
            total_samples = num_biosamples * num_loci * num_dsf_factors
            
            if self.is_ddp:
                # In DDP, samples are divided among processes
                world_size = self.world_size
                total_samples = total_samples // max(1, world_size)
            
            # Calculate batches
            batch_size = self.training_params.get('batch_size', 25)
            estimated_batches = max(1, total_samples // batch_size)

            self.estimated_batches_per_epoch = estimated_batches
            
            if self.is_main_process:
                print(f"Estimated batches per epoch: {estimated_batches} "
                      f"(samples: {total_samples}, batch_size: {batch_size})")
                print(f"Dataset composition: {num_biosamples} biosamples × "
                      f"{num_loci} loci × {num_dsf_factors} DSF factors = "
                      f"{num_biosamples * num_loci * num_dsf_factors} total samples")
            
            return estimated_batches
            
        except Exception as e:
            if self.is_main_process:
                print(f"Warning: Could not estimate batches per epoch: {e}")
            return None
    
    def _setup_cosine_scheduler(self, batches_per_epoch):
        """Setup cosine scheduler with actual batch counts."""
        epochs = self.training_params['epochs']
        inner_epochs = self.training_params['inner_epochs']
        
        # Calculate actual total steps
        grad_accum_steps = max(1, int(self.training_params.get('grad_accum_steps', 1)))
        num_total_steps = max(1, int(np.ceil((epochs * inner_epochs * batches_per_epoch) / grad_accum_steps)))
        # Use 20% of total steps for warmup 
        warmup_steps = max(1, int(0.1 * num_total_steps))
        
        # Ensure we have at least 1 step for the cosine annealing phase
        cosine_steps = max(1, num_total_steps - warmup_steps)
        
        if self.is_main_process:
            print(f"Setting up cosine scheduler: {num_total_steps} total steps, {warmup_steps} warmup steps, {cosine_steps} cosine steps")
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[
                LinearLR(self.optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_steps),
                CosineAnnealingLR(self.optimizer, T_max=cosine_steps, eta_min=0.1 * self.training_params['learning_rate'])
            ],
            milestones=[warmup_steps]
        )
        
        # Store the total steps for tracking
        self.total_scheduler_steps = num_total_steps
        self.current_scheduler_step = 0
    
    def _validate(self):
        """
        Run validation if enabled, computing metrics on validation set.
        
        Returns:
            tuple: (validation_summary_string, validation_metrics_dict) or (None, None) if validation fails
        """
        if not self.enable_validation or self.validator is None:
            return None, None
            
        if not self.is_main_process:
            return None, None  # Only run validation on main process
            
        try:
            print("Running validation...")
                
            # Set model to evaluation mode
            self.model.eval()
            
            with torch.no_grad():
                if self.validator == "simplified":
                    # Simplified validation fallback
                    validation_summary = self._run_simplified_validation()
                    validation_metrics = {"validation_type": "simplified", "status": "completed"}
                else:
                    # Use the full MONITOR_VALIDATION system
                    validation_summary, validation_metrics = self.validator.get_validation(self.model)
                
            # Set model back to training mode
            self.model.train()
            
            if validation_summary and validation_metrics:
                print("Validation completed successfully")
                return validation_summary, validation_metrics
            else:
                print("Warning: Validation returned empty results")
                return None, None
                
        except Exception as e:
            print(f"Error: Validation failed: {e}. Continuing training without validation...")
            # Set model back to training mode in case of error
            self.model.train()
            return None, None
    
    def _compute_metrics(self, output_p, output_n, output_mu, output_var, output_peak, y_data, y_pval, y_peaks, observed_map, masked_map):
        """
        Compute metrics per feature for both observed and imputed predictions.
        
        Args:
            output_p, output_n: Model outputs for negative binomial parameters [B, L, F]
            output_mu, output_var: Model outputs for Gaussian parameters [B, L, F]
            output_peak: Model outputs for peak predictions [B, L, F]
            y_data: Target count data [B, L, F]
            y_pval: Target p-value data [B, L, F]
            y_peaks: Target peak data [B, L, F]
            observed_map: Boolean mask for observed (upsampling) targets [B, L, F]
            masked_map: Boolean mask for masked (imputation) targets [B, L, F]
            
        Returns:
            dict: Dictionary containing per-feature metrics
        """
        metrics = {}
        B, L, F = y_data.shape  # Batch size, sequence length, num features
        
        # === IMPUTED (MASKED) METRICS PER FEATURE ===
        if masked_map.any():
            imp_count_r2_per_feature = []
            imp_count_spearman_per_feature = []
            imp_count_pearson_per_feature = []
            imp_count_mse_per_feature = []
            imp_count_mae_per_feature = []
            imp_count_mae_r2_per_feature = []
            
            imp_pval_r2_per_feature = []
            imp_pval_spearman_per_feature = []
            imp_pval_pearson_per_feature = []
            imp_pval_mse_per_feature = []
            imp_pval_mae_per_feature = []
            imp_pval_mae_r2_per_feature = []
            imp_count_perplexity_per_feature = []
            imp_pval_perplexity_per_feature = []
            
            imp_peak_auc_per_feature = []
            
            # Compute metrics per feature (F dimension) and per sample (B dimension)
            for f in range(F):
                # Collect all masked data points for this feature across all samples
                all_imp_count_pred = []
                all_imp_count_true = []

                all_imp_pval_pred = []
                all_imp_pval_true = []

                all_imp_count_var = []
                all_imp_pval_var = []
                
                all_imp_peak_pred = []
                all_imp_peak_true = []
                
                # Iterate over each sample in the batch
                for b in range(B):
                    # Get masked positions for this sample and feature
                    sample_masked_map = masked_map[b, :, f]
                    
                    if sample_masked_map.any():
                        # Count predictions for this sample and feature
                        sample_output_p = output_p[b, :, f][sample_masked_map]
                        sample_output_n = output_n[b, :, f][sample_masked_map]
                        neg_bin_imp = NegativeBinomial(sample_output_p.cpu().detach(), sample_output_n.cpu().detach())
                        sample_count_pred = neg_bin_imp.expect().numpy()
                        sample_count_true = y_data[b, :, f][sample_masked_map].cpu().detach().numpy()
                        
                        # P-value predictions for this sample and feature
                        sample_pval_pred = output_mu[b, :, f][sample_masked_map].cpu().detach().numpy()
                        sample_pval_true = y_pval[b, :, f][sample_masked_map].cpu().detach().numpy()
                        sample_pval_var = output_var[b, :, f][sample_masked_map].cpu().detach().numpy()
                        
                        # Peak predictions for this sample and feature
                        sample_peak_pred = output_peak[b, :, f][sample_masked_map].cpu().detach().numpy()
                        sample_peak_true = y_peaks[b, :, f][sample_masked_map].cpu().detach().numpy()
                        
                        # Collect data points
                        all_imp_count_pred.extend(sample_count_pred)
                        all_imp_count_true.extend(sample_count_true)
                        all_imp_pval_pred.extend(sample_pval_pred)
                        all_imp_pval_true.extend(sample_pval_true)
                        all_imp_pval_var.extend(sample_pval_var)
                        all_imp_peak_pred.extend(sample_peak_pred)
                        all_imp_peak_true.extend(sample_peak_true)
                
                # Compute metrics if we have enough data points across all samples
                if len(all_imp_count_true) > 1:
                    all_imp_count_pred = np.array(all_imp_count_pred)
                    all_imp_count_true = np.array(all_imp_count_true)
                    
                    # Check for zero variance to avoid division by zero in r2_score
                    if np.var(all_imp_count_true) > 1e-10:  # Avoid division by zero
                        r2_val = r2_score(all_imp_count_true, all_imp_count_pred)
                        imp_count_r2_per_feature.append(r2_val)
                    else:
                        # If variance is zero, r2 is undefined, use a default value
                        imp_count_r2_per_feature.append(0.0)
                    
                    mse_val = ((all_imp_count_true - all_imp_count_pred)**2).mean()
                    imp_count_mse_per_feature.append(mse_val)
                    
                    # MAE and MAE-R2
                    mae_val = np.abs(all_imp_count_true - all_imp_count_pred).mean()
                    imp_count_mae_per_feature.append(mae_val)
                    mae_denom = np.abs(all_imp_count_true - all_imp_count_true.mean()).sum()
                    if mae_denom > 1e-10:
                        mae_r2_val = 1.0 - np.abs(all_imp_count_true - all_imp_count_pred).sum() / mae_denom
                        imp_count_mae_r2_per_feature.append(mae_r2_val)
                    
                    # Spearman correlation
                    spearman_result = spearmanr(all_imp_count_true, all_imp_count_pred)
                    if not np.isnan(spearman_result.correlation):
                        imp_count_spearman_per_feature.append(spearman_result.correlation)
                    
                    # Pearson correlation
                    pearson_result = pearsonr(all_imp_count_true, all_imp_count_pred)
                    if not np.isnan(pearson_result[0]):
                        imp_count_pearson_per_feature.append(pearson_result[0])

                    # Compute perplexity for count predictions
                    # Reconstruct NegativeBinomial for perplexity calculation
                    # We need to collect the parameters again for perplexity
                    all_output_p = []
                    all_output_n = []
                    for b in range(B):
                        sample_masked_map = masked_map[b, :, f]
                        if sample_masked_map.any():
                            all_output_p.extend(output_p[b, :, f][sample_masked_map].cpu().detach().numpy())
                            all_output_n.extend(output_n[b, :, f][sample_masked_map].cpu().detach().numpy())
                    
                    if len(all_output_p) > 0:
                        neg_bin_imp = NegativeBinomial(torch.tensor(all_output_p), torch.tensor(all_output_n))
                        neg_bin_probs = neg_bin_imp.pmf(all_imp_count_true.astype(int))
                        perplexity = compute_perplexity(neg_bin_probs)
                        imp_count_perplexity_per_feature.append(perplexity.item())
                
                if len(all_imp_pval_true) > 1:
                    all_imp_pval_pred = np.array(all_imp_pval_pred)
                    all_imp_pval_true = np.array(all_imp_pval_true)
                    
                    # Check for zero variance to avoid division by zero in r2_score
                    if np.var(all_imp_pval_true) > 1e-10:  # Avoid division by zero
                        imp_pval_r2_per_feature.append(r2_score(all_imp_pval_true, all_imp_pval_pred))
                    else:
                        # If variance is zero, r2 is undefined, use a default value
                        imp_pval_r2_per_feature.append(0.0)
                    imp_pval_mse_per_feature.append(((all_imp_pval_true - all_imp_pval_pred)**2).mean())
                    
                    # MAE and MAE-R2 for pval
                    mae_val = np.abs(all_imp_pval_true - all_imp_pval_pred).mean()
                    imp_pval_mae_per_feature.append(mae_val)
                    mae_denom = np.abs(all_imp_pval_true - all_imp_pval_true.mean()).sum()
                    if mae_denom > 1e-10:
                        mae_r2_val = 1.0 - np.abs(all_imp_pval_true - all_imp_pval_pred).sum() / mae_denom
                        imp_pval_mae_r2_per_feature.append(mae_r2_val)
                    
                    # Spearman correlation
                    spearman_result = spearmanr(all_imp_pval_true, all_imp_pval_pred)
                    if not np.isnan(spearman_result.correlation):
                        imp_pval_spearman_per_feature.append(spearman_result.correlation)
                    
                    # Pearson correlation
                    pearson_result = pearsonr(all_imp_pval_true, all_imp_pval_pred)
                    if not np.isnan(pearson_result[0]):
                        imp_pval_pearson_per_feature.append(pearson_result[0])
                    
                    # Compute perplexity for p-value predictions
                    all_imp_pval_var = np.array(all_imp_pval_var)
                    if self.dist_type == 'laplace':
                        # all_imp_pval_var is actually log_b for Laplace
                        dist_imp = Laplace(all_imp_pval_pred, all_imp_pval_var)
                    else:
                        dist_imp = Gaussian(all_imp_pval_pred, all_imp_pval_var)
                    dist_probs = dist_imp.pdf(all_imp_pval_true)
                    perplexity = compute_perplexity(dist_probs)
                    imp_pval_perplexity_per_feature.append(perplexity.item())
                
                # Compute AUC-ROC for peak predictions
                if len(all_imp_peak_true) > 1:
                    all_imp_peak_pred = np.array(all_imp_peak_pred)
                    all_imp_peak_true = np.array(all_imp_peak_true)
                    
                    # Check if we have both positive and negative samples
                    if len(np.unique(all_imp_peak_true)) > 1:
                        try:
                            auc_score = roc_auc_score(all_imp_peak_true, all_imp_peak_pred)
                            imp_peak_auc_per_feature.append(auc_score)
                        except ValueError:
                            # Handle edge cases where AUC cannot be computed
                            pass
            
            # Aggregate imputed metrics: median only
            if imp_count_r2_per_feature:
                imp_count_r2_arr = np.array(imp_count_r2_per_feature)
                metrics.update({
                    'imp_count_r2_median': np.median(imp_count_r2_arr)
                })
                
            if imp_count_spearman_per_feature:
                imp_count_spearman_arr = np.array(imp_count_spearman_per_feature)
                metrics.update({
                    'imp_count_spearman_median': np.median(imp_count_spearman_arr)
                })
                
            if imp_count_pearson_per_feature:
                imp_count_pearson_arr = np.array(imp_count_pearson_per_feature)
                metrics.update({
                    'imp_count_pearson_median': np.median(imp_count_pearson_arr)
                })
                
            if imp_pval_r2_per_feature:
                imp_pval_r2_arr = np.array(imp_pval_r2_per_feature)
                metrics.update({
                    'imp_pval_r2_median': np.median(imp_pval_r2_arr)
                })
                
            if imp_pval_spearman_per_feature:
                imp_pval_spearman_arr = np.array(imp_pval_spearman_per_feature)
                metrics.update({
                    'imp_pval_spearman_median': np.median(imp_pval_spearman_arr)
                })
                
            if imp_pval_pearson_per_feature:
                imp_pval_pearson_arr = np.array(imp_pval_pearson_per_feature)
                metrics.update({
                    'imp_pval_pearson_median': np.median(imp_pval_pearson_arr)
                })
            
            # Aggregate perplexity metrics for imputation
            if imp_count_perplexity_per_feature:
                imp_count_perplexity_arr = np.array(imp_count_perplexity_per_feature)
                metrics.update({
                    'imp_count_perplexity_median': np.median(imp_count_perplexity_arr)
                })
            
            if imp_pval_perplexity_per_feature:
                imp_pval_perplexity_arr = np.array(imp_pval_perplexity_per_feature)
                metrics.update({
                    'imp_pval_perplexity_median': np.median(imp_pval_perplexity_arr)
                })
            
            # Aggregate MSE metrics for imputation
            if imp_count_mse_per_feature:
                imp_count_mse_arr = np.array(imp_count_mse_per_feature)
                metrics.update({
                    'imp_count_mse_median': np.median(imp_count_mse_arr)
                })
            
            if imp_pval_mse_per_feature:
                imp_pval_mse_arr = np.array(imp_pval_mse_per_feature)
                metrics.update({
                    'imp_pval_mse_median': np.median(imp_pval_mse_arr)
                })
            
            # Aggregate MAE and MAE-R2 metrics for imputation
            if imp_count_mae_per_feature:
                metrics.update({
                    'imp_count_mae_median': np.median(np.array(imp_count_mae_per_feature))
                })
            
            if imp_count_mae_r2_per_feature:
                metrics.update({
                    'imp_count_mae_r2_median': np.median(np.array(imp_count_mae_r2_per_feature))
                })
            
            if imp_pval_mae_per_feature:
                metrics.update({
                    'imp_pval_mae_median': np.median(np.array(imp_pval_mae_per_feature))
                })
            
            if imp_pval_mae_r2_per_feature:
                metrics.update({
                    'imp_pval_mae_r2_median': np.median(np.array(imp_pval_mae_r2_per_feature))
                })
            
            # Aggregate AUC metrics for imputation
            if imp_peak_auc_per_feature:
                imp_peak_auc_arr = np.array(imp_peak_auc_per_feature)
                metrics.update({
                    'imp_peak_auc_median': np.median(imp_peak_auc_arr)
                })
            
        # === OBSERVED (UPSAMPLING) METRICS PER FEATURE ===
        if observed_map.any():
            obs_count_r2_per_feature = []
            obs_count_spearman_per_feature = []
            obs_count_pearson_per_feature = []
            obs_count_mse_per_feature = []
            obs_count_mae_per_feature = []
            obs_count_mae_r2_per_feature = []
            
            obs_pval_r2_per_feature = []
            obs_pval_spearman_per_feature = []
            obs_pval_pearson_per_feature = []
            obs_pval_mse_per_feature = []
            obs_pval_mae_per_feature = []
            obs_pval_mae_r2_per_feature = []
            obs_count_perplexity_per_feature = []
            obs_pval_perplexity_per_feature = []
            
            obs_peak_auc_per_feature = []
            
            # Compute metrics per feature (F dimension) and per sample (B dimension)
            for f in range(F):
                # Collect all observed data points for this feature across all samples
                all_obs_count_pred = []
                all_obs_count_true = []
                all_obs_pval_pred = []
                all_obs_pval_true = []
                all_obs_pval_var = []
                
                all_obs_peak_pred = []
                all_obs_peak_true = []
                
                # Iterate over each sample in the batch
                for b in range(B):
                    # Get observed positions for this sample and feature
                    sample_observed_map = observed_map[b, :, f]
                    
                    if sample_observed_map.any():
                        # Count predictions for this sample and feature
                        sample_output_p = output_p[b, :, f][sample_observed_map]
                        sample_output_n = output_n[b, :, f][sample_observed_map]
                        neg_bin_obs = NegativeBinomial(sample_output_p.cpu().detach(), sample_output_n.cpu().detach())
                        sample_count_pred = neg_bin_obs.expect().numpy()
                        sample_count_true = y_data[b, :, f][sample_observed_map].cpu().detach().numpy()
                        
                        # P-value predictions for this sample and feature
                        sample_pval_pred = output_mu[b, :, f][sample_observed_map].cpu().detach().numpy()
                        sample_pval_true = y_pval[b, :, f][sample_observed_map].cpu().detach().numpy()
                        sample_pval_var = output_var[b, :, f][sample_observed_map].cpu().detach().numpy()
                        
                        # Peak predictions for this sample and feature
                        sample_peak_pred = output_peak[b, :, f][sample_observed_map].cpu().detach().numpy()
                        sample_peak_true = y_peaks[b, :, f][sample_observed_map].cpu().detach().numpy()
                        
                        # Collect data points
                        all_obs_count_pred.extend(sample_count_pred)
                        all_obs_count_true.extend(sample_count_true)
                        all_obs_pval_pred.extend(sample_pval_pred)
                        all_obs_pval_true.extend(sample_pval_true)
                        all_obs_pval_var.extend(sample_pval_var)
                        all_obs_peak_pred.extend(sample_peak_pred)
                        all_obs_peak_true.extend(sample_peak_true)
                
                # Compute metrics if we have enough data points across all samples
                if len(all_obs_count_true) > 1:
                    all_obs_count_pred = np.array(all_obs_count_pred)
                    all_obs_count_true = np.array(all_obs_count_true)
                    
                    # Check for zero variance to avoid division by zero in r2_score
                    if np.var(all_obs_count_true) > 1e-10:  # Avoid division by zero
                        obs_count_r2_per_feature.append(r2_score(all_obs_count_true, all_obs_count_pred))
                    else:
                        # If variance is zero, r2 is undefined, use a default value
                        obs_count_r2_per_feature.append(0.0)
                    obs_count_mse_per_feature.append(((all_obs_count_true - all_obs_count_pred)**2).mean())
                    
                    # MAE and MAE-R2
                    mae_val = np.abs(all_obs_count_true - all_obs_count_pred).mean()
                    obs_count_mae_per_feature.append(mae_val)
                    mae_denom = np.abs(all_obs_count_true - all_obs_count_true.mean()).sum()
                    if mae_denom > 1e-10:
                        mae_r2_val = 1.0 - np.abs(all_obs_count_true - all_obs_count_pred).sum() / mae_denom
                        obs_count_mae_r2_per_feature.append(mae_r2_val)
                    
                    # Spearman correlation
                    spearman_result = spearmanr(all_obs_count_true, all_obs_count_pred)
                    if not np.isnan(spearman_result.correlation):
                        obs_count_spearman_per_feature.append(spearman_result.correlation)
                    
                    # Pearson correlation
                    pearson_result = pearsonr(all_obs_count_true, all_obs_count_pred)
                    if not np.isnan(pearson_result[0]):
                        obs_count_pearson_per_feature.append(pearson_result[0])
                    
                    # Compute perplexity for observed count predictions
                    # Reconstruct NegativeBinomial for perplexity calculation
                    # We need to collect the parameters again for perplexity
                    all_output_p = []
                    all_output_n = []
                    for b in range(B):
                        sample_observed_map = observed_map[b, :, f]
                        if sample_observed_map.any():
                            all_output_p.extend(output_p[b, :, f][sample_observed_map].cpu().detach().numpy())
                            all_output_n.extend(output_n[b, :, f][sample_observed_map].cpu().detach().numpy())
                    
                    if len(all_output_p) > 0:
                        neg_bin_obs = NegativeBinomial(torch.tensor(all_output_p), torch.tensor(all_output_n))
                        neg_bin_probs = neg_bin_obs.pmf(all_obs_count_true.astype(int))
                        perplexity = compute_perplexity(neg_bin_probs)
                        obs_count_perplexity_per_feature.append(perplexity.item())
                
                if len(all_obs_pval_true) > 1:
                    all_obs_pval_pred = np.array(all_obs_pval_pred)
                    all_obs_pval_true = np.array(all_obs_pval_true)
                    
                    # Check for zero variance to avoid division by zero in r2_score
                    if np.var(all_obs_pval_true) > 1e-10:  # Avoid division by zero
                        obs_pval_r2_per_feature.append(r2_score(all_obs_pval_true, all_obs_pval_pred))
                    else:
                        # If variance is zero, r2 is undefined, use a default value
                        obs_pval_r2_per_feature.append(0.0)
                    obs_pval_mse_per_feature.append(((all_obs_pval_true - all_obs_pval_pred)**2).mean())
                    
                    # MAE and MAE-R2 for pval
                    mae_val = np.abs(all_obs_pval_true - all_obs_pval_pred).mean()
                    obs_pval_mae_per_feature.append(mae_val)
                    mae_denom = np.abs(all_obs_pval_true - all_obs_pval_true.mean()).sum()
                    if mae_denom > 1e-10:
                        mae_r2_val = 1.0 - np.abs(all_obs_pval_true - all_obs_pval_pred).sum() / mae_denom
                        obs_pval_mae_r2_per_feature.append(mae_r2_val)
                    
                    # Spearman correlation
                    spearman_result = spearmanr(all_obs_pval_true, all_obs_pval_pred)
                    if not np.isnan(spearman_result.correlation):
                        obs_pval_spearman_per_feature.append(spearman_result.correlation)
                    
                    # Pearson correlation
                    pearson_result = pearsonr(all_obs_pval_true, all_obs_pval_pred)
                    if not np.isnan(pearson_result[0]):
                        obs_pval_pearson_per_feature.append(pearson_result[0])
                    
                    # Compute perplexity for observed p-value predictions
                    all_obs_pval_var = np.array(all_obs_pval_var)
                    if self.dist_type == 'laplace':
                        # all_obs_pval_var is actually log_b for Laplace
                        dist_obs = Laplace(all_obs_pval_pred, all_obs_pval_var)
                    else:
                        dist_obs = Gaussian(all_obs_pval_pred, all_obs_pval_var)
                    dist_probs = dist_obs.pdf(all_obs_pval_true)
                    perplexity = compute_perplexity(dist_probs)
                    obs_pval_perplexity_per_feature.append(perplexity.item())
                
                # Compute AUC-ROC for observed peak predictions
                if len(all_obs_peak_true) > 1:
                    all_obs_peak_pred = np.array(all_obs_peak_pred)
                    all_obs_peak_true = np.array(all_obs_peak_true)
                    
                    # Check if we have both positive and negative samples
                    if len(np.unique(all_obs_peak_true)) > 1:
                        try:
                            auc_score = roc_auc_score(all_obs_peak_true, all_obs_peak_pred)
                            obs_peak_auc_per_feature.append(auc_score)
                        except ValueError:
                            # Handle edge cases where AUC cannot be computed
                            pass
            
            # Aggregate observed metrics: median only
            if obs_count_r2_per_feature:
                obs_count_r2_arr = np.array(obs_count_r2_per_feature)
                metrics.update({
                    'obs_count_r2_median': np.median(obs_count_r2_arr)
                })
                
            if obs_count_spearman_per_feature:
                obs_count_spearman_arr = np.array(obs_count_spearman_per_feature)
                metrics.update({
                    'obs_count_spearman_median': np.median(obs_count_spearman_arr)
                })
                
            if obs_count_pearson_per_feature:
                obs_count_pearson_arr = np.array(obs_count_pearson_per_feature)
                metrics.update({
                    'obs_count_pearson_median': np.median(obs_count_pearson_arr)
                })
                
            if obs_pval_r2_per_feature:
                obs_pval_r2_arr = np.array(obs_pval_r2_per_feature)
                metrics.update({
                    'obs_pval_r2_median': np.median(obs_pval_r2_arr)
                })
                
            if obs_pval_spearman_per_feature:
                obs_pval_spearman_arr = np.array(obs_pval_spearman_per_feature)
                metrics.update({
                    'obs_pval_spearman_median': np.median(obs_pval_spearman_arr)
                })
                
            if obs_pval_pearson_per_feature:
                obs_pval_pearson_arr = np.array(obs_pval_pearson_per_feature)
                metrics.update({
                    'obs_pval_pearson_median': np.median(obs_pval_pearson_arr)
                })
            
            # Aggregate perplexity metrics for observed (upsampling)
            if obs_count_perplexity_per_feature:
                obs_count_perplexity_arr = np.array(obs_count_perplexity_per_feature)
                metrics.update({
                    'obs_count_perplexity_median': np.median(obs_count_perplexity_arr)
                })
            
            if obs_pval_perplexity_per_feature:
                obs_pval_perplexity_arr = np.array(obs_pval_perplexity_per_feature)
                metrics.update({
                    'obs_pval_perplexity_median': np.median(obs_pval_perplexity_arr)
                })
            
            # Aggregate MSE metrics for observed (upsampling)
            if obs_count_mse_per_feature:
                obs_count_mse_arr = np.array(obs_count_mse_per_feature)
                metrics.update({
                    'obs_count_mse_median': np.median(obs_count_mse_arr)
                })
            
            if obs_pval_mse_per_feature:
                obs_pval_mse_arr = np.array(obs_pval_mse_per_feature)
                metrics.update({
                    'obs_pval_mse_median': np.median(obs_pval_mse_arr)
                })
            
            # Aggregate MAE and MAE-R2 metrics for observed (upsampling)
            if obs_count_mae_per_feature:
                metrics.update({
                    'obs_count_mae_median': np.median(np.array(obs_count_mae_per_feature))
                })
            
            if obs_count_mae_r2_per_feature:
                metrics.update({
                    'obs_count_mae_r2_median': np.median(np.array(obs_count_mae_r2_per_feature))
                })
            
            if obs_pval_mae_per_feature:
                metrics.update({
                    'obs_pval_mae_median': np.median(np.array(obs_pval_mae_per_feature))
                })
            
            if obs_pval_mae_r2_per_feature:
                metrics.update({
                    'obs_pval_mae_r2_median': np.median(np.array(obs_pval_mae_r2_per_feature))
                })
            
            # Aggregate AUC metrics for observed (upsampling)
            if obs_peak_auc_per_feature:
                obs_peak_auc_arr = np.array(obs_peak_auc_per_feature)
                metrics.update({
                    'obs_peak_auc_median': np.median(obs_peak_auc_arr)
                })
           
        return metrics
    
    def _print_batch_log(self, metrics, loss_dict, batch_idx, epoch, batch_processing_time=None, optimizer_step=False):
        """Print batch logging similar to old_train_candi.py format."""
        if not metrics or not loss_dict:
            return
            
        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        lr_printstatement = f"LR {current_lr:.2e}"
        
        # Get gradient norm if available
        grad_norm = 0.0
        if hasattr(self, 'grad_norm'):
            grad_norm = self.grad_norm
        
        # Get masking probabilities info
        mask_info = "N/A"
        if hasattr(self, 'last_masking_probs'):
            probs = self.last_masking_probs
            mask_info = f"A:{probs['p_full_assay']:.1f}/L:{probs['p_full_loci']:.1f}/C:{probs['p_chunks']:.1f}"
        
        # Format batch processing time
        batch_time_str = "N/A"
        batch_time = 0.0
        if batch_processing_time is not None:
            batch_time_str = f"{batch_processing_time:.2f}s"
            batch_time = batch_processing_time
        
        # Record progress for CSV logging
        self._record_progress(epoch, batch_idx, metrics, loss_dict, grad_norm, mask_info, batch_time, current_lr)
        
        # Improved and aesthetic print statement with aligned columns and section headers
        sep = " | "
        metrics_table = [
            # Header
            f"{'='*84}",
            f" Epoch: {epoch+1:<4}   Batch: {batch_idx:<4}/{self.estimated_batches_per_epoch}  ({100.0 * batch_idx / self.estimated_batches_per_epoch:.1f}%)".ljust(82),
            f"{'-'*84}",
            # Section: Losses
            f"{'Type':<8} {'nbNLL':>10} {'gNLL':>10} {'peakLoss':>12}",
            f"{'-'*84}",
            f"{'Imp':<8} "
            f"{loss_dict.get('imp_count_loss', 0.0):>10.2f} "
            f"{loss_dict.get('imp_pval_loss', 0.0):>10.2f} "
            f"{loss_dict.get('imp_peak_loss', 0.0):>12.2f}",
            f"{'Obs':<8}"
            f"{loss_dict.get('obs_count_loss', 0.0):>10.2f} "
            f"{loss_dict.get('obs_pval_loss', 0.0):>10.2f} "
            f"{loss_dict.get('obs_peak_loss', 0.0):>12.2f}",
            f"{'-'*84}",
            # Section: R2
            f"{'':<8} {'Count R2':>10} {'Pval R2':>10}",
            f"{'Imp':<8} "
            f"{metrics.get('imp_count_r2_median', 0.0):>10.2f} "
            f"{metrics.get('imp_pval_r2_median', 0.0):>10.2f}",
            f"{'Obs':<8} "
            f"{metrics.get('obs_count_r2_median', 0.0):>10.2f} "
            f"{metrics.get('obs_pval_r2_median', 0.0):>10.2f}",
            f"{'-'*84}",
            # Section: Spearman
            f"{'':<8} {'Count SRCC':>10} {'Pval SRCC':>10}",
            f"{'Imp':<8} "
            f"{metrics.get('imp_count_spearman_median', 0.0):>10.2f} "
            f"{metrics.get('imp_pval_spearman_median', 0.0):>10.2f}",
            f"{'Obs':<8} "
            f"{metrics.get('obs_count_spearman_median', 0.0):>10.2f} "
            f"{metrics.get('obs_pval_spearman_median', 0.0):>10.2f}",
            f"{'-'*84}",
            # Section: Pearson
            f"{'':<8} {'Count PCC':>10} {'Pval PCC':>10}",
            f"{'Imp':<8} "
            f"{metrics.get('imp_count_pearson_median', 0.0):>10.2f} "
            f"{metrics.get('imp_pval_pearson_median', 0.0):>10.2f}",
            f"{'Obs':<8} "
            f"{metrics.get('obs_count_pearson_median', 0.0):>10.2f} "
            f"{metrics.get('obs_pval_pearson_median', 0.0):>10.2f}",
            f"{'-'*84}",
            # Section: MSE
            f"{'':<8} {'Count MSE':>10} {'Pval MSE':>10}",
            f"{'Imp':<8} "
            f"{metrics.get('imp_count_mse_median', 0.0):>10.2f} "
            f"{metrics.get('imp_pval_mse_median', 0.0):>10.2f}",
            f"{'Obs':<8} "
            f"{metrics.get('obs_count_mse_median', 0.0):>10.2f} "
            f"{metrics.get('obs_pval_mse_median', 0.0):>10.2f}",
            f"{'-'*84}",
            # Section: Peak AUC
            f"{'':<8} {'Peak AUC':>10}",
            f"{'Imp':<8} {metrics.get('imp_peak_auc_median', 0.0):>10.2f}",
            f"{'Obs':<8} {metrics.get('obs_peak_auc_median', 0.0):>10.2f}",
            f"{'-'*84}",
        ]
        
        # Add EMA values (as additional nicely formatted section)
        if hasattr(self, 'specific_ema') and self.specific_ema:
            metrics_table.append("EMA (Exponential Moving Average):")
            metrics_table.append(f"{'':<8} {'Count_Loss':>10} {'Obs_Count_Loss':>15} {'Pval_Loss':>15} {'Obs_Pval_Loss':>15}")
            metrics_table.append(
                f"{'EMA':<8} "
                f"{self.specific_ema.get('imp_count_loss', 0.0):>10.2f} "
                f"{self.specific_ema.get('obs_count_loss', 0.0):>15.2f} "
                f"{self.specific_ema.get('imp_pval_loss', 0.0):>15.2f} "
                f"{self.specific_ema.get('obs_pval_loss', 0.0):>15.2f}"
            )
            metrics_table.append(f"{'':<8} {'Pval_R2':>10} {'Pval_SRCC':>12} {'Pval_PCC':>12} {'Obs_Peak_AUC':>14}")
            metrics_table.append(
                f"{'EMA':<8} "
                f"{self.specific_ema.get('imp_pval_r2_median', 0.0):>10.2f} "
                f"{self.specific_ema.get('imp_pval_spearman_median', 0.0):>12.2f} "
                f"{self.specific_ema.get('imp_pval_pearson_median', 0.0):>12.2f} "
                f"{self.specific_ema.get('obs_peak_auc_median', 0.0):>14.2f}"
            )
            metrics_table.append(f"{'':<8} {'Count_R2':>10} {'Count_SRCC':>12} {'Count_PCC':>12} {'Peak_AUC':>14}")
            metrics_table.append(
                f"{'EMA':<8} "
                f"{self.specific_ema.get('imp_count_r2_median', 0.0):>10.2f} "
                f"{self.specific_ema.get('imp_count_spearman_median', 0.0):>12.2f} "
                f"{self.specific_ema.get('imp_count_pearson_median', 0.0):>12.2f} "
                f"{self.specific_ema.get('imp_peak_auc_median', 0.0):>14.2f}"
            )
            metrics_table.append(f"{'-'*84}")

        # Add training dynamics and environment
        if any(k in metrics for k in ['train_st/depth_ratio', 'train_st/runtype_mse', 'train_st/readlen_mse']):
            metrics_table.append(
                f"Train-ST: depth_ratio={metrics.get('train_st/depth_ratio', np.nan):.3f} "
                f"runtype_mse={metrics.get('train_st/runtype_mse', np.nan):.3e} "
                f"readlen_mse={metrics.get('train_st/readlen_mse', np.nan):.3e}"
            )
        metrics_table.append(
            f"DSF-Trans: up={int(metrics.get('dsf_transitions/upsampled_count', 0))} "
            f"down={int(metrics.get('dsf_transitions/downsampled_count', 0))} "
            f"same={int(metrics.get('dsf_transitions/same_count', 0))} "
            f"ctrl_non1={int(metrics.get('dsf_transitions/control_non1_count', 0))}"
        )
        metrics_table.append(
            f"Gradient Norm: {grad_norm:.2f}   Mask Probs: {mask_info}   {lr_printstatement}   "
            f"step={'yes' if optimizer_step else 'no'}   Batch time: {batch_time_str}"
        )
        metrics_table.append(f"{'='*84}")

        # Then, before printing the metrics table (around line 1301), add this:
        # Clear previous table by moving cursor up and clearing lines
        # if hasattr(self, 'last_table_lines') and self.last_table_lines > 0:
        #     # Move cursor up and clear each line
        #     print('\033[F\033[K' * self.last_table_lines, end='')
        
        # Print organized metrics
        table_output = "\n".join(metrics_table)
        print(table_output)
        print("\n")

        # Store the number of lines for next time
        self.last_table_lines = len(metrics_table)
    
    def _save_progress_to_csv(self, epoch, batch_idx):
        """Save progress data to CSV file every 100 batches."""
        if not self.progress_data:
            return
        
        # Create filename if not already set (only once)
        if self.progress_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.progress_file = Path(self.progress_dir) / f"training_progress_{timestamp}.csv"
            if self.is_main_process:
                print(f"Progress will be saved to: {self.progress_file}")
            
        # Create DataFrame from progress data
        df = pd.DataFrame(self.progress_data)
        
        # Save to CSV (overwrite existing file)
        df.to_csv(self.progress_file, index=False)
        
        if self.is_main_process:
            print(f"Progress updated in {self.progress_file} ({len(df)} records)")
    
    def _record_validation_progress(self, val_results):
        """Record validation metrics for CSV logging."""
        if not val_results:
            return
        
        # Add to validation progress data
        self.validation_progress_data.append(val_results)
    
    def _save_validation_progress_csv(self, epoch=None):
        """
        Save validation progress data to CSV file.
        
        Args:
            epoch: Current epoch number (for checkpoint naming if needed)
        """
        if not self.validation_progress_data:
            return
        
        # Create filename if not already set (only once)
        if self.validation_progress_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.validation_progress_file = Path(self.progress_dir) / f"validation_progress_{timestamp}.csv"
            if self.is_main_process:
                print(f"Validation progress will be saved to: {self.validation_progress_file}")
        
        # Create DataFrame from validation progress data
        df = pd.DataFrame(self.validation_progress_data)
        
        # Order columns logically:
        # 1. iteration, progress_pct
        # 2. val_loss/* (aggregate losses)
        # 3. val_metrics/* (aggregate metrics)
        # 4. val_loss_per_assay/* (per-assay losses)
        # 5. val_metrics_per_assay/* (per-assay metrics)
        
        base_cols = ['iteration', 'progress_pct']
        
        # Categorize columns by prefix
        val_loss_cols = sorted([c for c in df.columns if c.startswith('val_loss/') and 'per_assay' not in c])
        val_metrics_cols = sorted([c for c in df.columns if c.startswith('val_metrics/') and 'per_assay' not in c])
        val_loss_per_assay_cols = sorted([c for c in df.columns if c.startswith('val_loss_per_assay/')])
        val_metrics_per_assay_cols = sorted([c for c in df.columns if c.startswith('val_metrics_per_assay/')])
        
        # Build ordered column list
        ordered_cols = []
        for col in base_cols:
            if col in df.columns:
                ordered_cols.append(col)
        ordered_cols.extend(val_loss_cols)
        ordered_cols.extend(val_metrics_cols)
        ordered_cols.extend(val_loss_per_assay_cols)
        ordered_cols.extend(val_metrics_per_assay_cols)
        
        # Add any remaining columns not yet included
        for col in df.columns:
            if col not in ordered_cols:
                ordered_cols.append(col)
        
        # Reorder DataFrame
        df = df[ordered_cols]
        
        # Save to CSV (overwrite existing file)
        df.to_csv(self.validation_progress_file, index=False)
        
        if self.is_main_process:
            print(f"Validation progress updated in {self.validation_progress_file} ({len(df)} records)")
    
    def _record_progress(self, epoch, batch_idx, metrics, loss_dict, grad_norm, mask_info, batch_time, lr):
        """Record all progress metrics for CSV logging."""
        # Get EMA values if available
        ema_values = {}
        if hasattr(self, 'specific_ema') and self.specific_ema:
            ema_values = {
                'EMA_Imp_Pval_R2': self.specific_ema.get('imp_pval_r2_median', 0.0),
                'EMA_Imp_Pval_SRCC': self.specific_ema.get('imp_pval_spearman_median', 0.0),
                'EMA_Imp_Pval_PCC': self.specific_ema.get('imp_pval_pearson_median', 0.0),
                'EMA_Imp_Pval_MAE_R2': self.specific_ema.get('imp_pval_mae_r2_median', 0.0),
                'EMA_Imp_Count_R2': self.specific_ema.get('imp_count_r2_median', 0.0),
                'EMA_Imp_Count_SRCC': self.specific_ema.get('imp_count_spearman_median', 0.0),
                'EMA_Imp_Count_PCC': self.specific_ema.get('imp_count_pearson_median', 0.0),
                'EMA_Imp_Count_MAE_R2': self.specific_ema.get('imp_count_mae_r2_median', 0.0),
                'EMA_Obs_Pval_MAE_R2': self.specific_ema.get('obs_pval_mae_r2_median', 0.0),
                'EMA_Obs_Count_MAE_R2': self.specific_ema.get('obs_count_mae_r2_median', 0.0),
                'EMA_Imp_Count_Loss': self.specific_ema.get('imp_count_loss', 0.0),
                'EMA_Obs_Count_Loss': self.specific_ema.get('obs_count_loss', 0.0),
                'EMA_Imp_Pval_Loss': self.specific_ema.get('imp_pval_loss', 0.0),
                'EMA_Obs_Pval_Loss': self.specific_ema.get('obs_pval_loss', 0.0),
                'EMA_Imp_Peak_AUC': self.specific_ema.get('imp_peak_auc_median', 0.0),
                'EMA_Obs_Peak_AUC': self.specific_ema.get('obs_peak_auc_median', 0.0)
            }
        
        # Create record with all metrics
        record = {
            'epoch': epoch + 1,
            'batch_idx': batch_idx,
            'timestamp': datetime.now().isoformat(),
            'learning_rate': lr,
            'gradient_norm': grad_norm,
            'mask_info': mask_info,
            'batch_time': batch_time,
            
            # Loss values
            'imp_count_loss': loss_dict.get('imp_count_loss', 0.0),
            'obs_count_loss': loss_dict.get('obs_count_loss', 0.0),
            'imp_pval_loss': loss_dict.get('imp_pval_loss', 0.0),
            'obs_pval_loss': loss_dict.get('obs_pval_loss', 0.0),
            'imp_peak_loss': loss_dict.get('imp_peak_loss', 0.0),
            'obs_peak_loss': loss_dict.get('obs_peak_loss', 0.0),
            'total_loss': loss_dict.get('total_loss', 0.0),
            
            # Imputation metrics
            'imp_count_r2_median': metrics.get('imp_count_r2_median', 0.0),
            'imp_count_spearman_median': metrics.get('imp_count_spearman_median', 0.0),
            'imp_count_pearson_median': metrics.get('imp_count_pearson_median', 0.0),
            'imp_count_mse_median': metrics.get('imp_count_mse_median', 0.0),
            'imp_count_mae_median': metrics.get('imp_count_mae_median', 0.0),
            'imp_count_mae_r2_median': metrics.get('imp_count_mae_r2_median', 0.0),
            'imp_count_perplexity_median': metrics.get('imp_count_perplexity_median', 0.0),
            
            'imp_pval_r2_median': metrics.get('imp_pval_r2_median', 0.0),
            'imp_pval_spearman_median': metrics.get('imp_pval_spearman_median', 0.0),
            'imp_pval_pearson_median': metrics.get('imp_pval_pearson_median', 0.0),
            'imp_pval_mse_median': metrics.get('imp_pval_mse_median', 0.0),
            'imp_pval_mae_median': metrics.get('imp_pval_mae_median', 0.0),
            'imp_pval_mae_r2_median': metrics.get('imp_pval_mae_r2_median', 0.0),
            'imp_pval_perplexity_median': metrics.get('imp_pval_perplexity_median', 0.0),
            
            'imp_peak_auc_median': metrics.get('imp_peak_auc_median', 0.0),
            
            # Upsampling metrics
            'obs_count_r2_median': metrics.get('obs_count_r2_median', 0.0),
            'obs_count_spearman_median': metrics.get('obs_count_spearman_median', 0.0),
            'obs_count_pearson_median': metrics.get('obs_count_pearson_median', 0.0),
            'obs_count_mse_median': metrics.get('obs_count_mse_median', 0.0),
            'obs_count_mae_median': metrics.get('obs_count_mae_median', 0.0),
            'obs_count_mae_r2_median': metrics.get('obs_count_mae_r2_median', 0.0),
            'obs_count_perplexity_median': metrics.get('obs_count_perplexity_median', 0.0),
            
            'obs_pval_r2_median': metrics.get('obs_pval_r2_median', 0.0),
            'obs_pval_spearman_median': metrics.get('obs_pval_spearman_median', 0.0),
            'obs_pval_pearson_median': metrics.get('obs_pval_pearson_median', 0.0),
            'obs_pval_mse_median': metrics.get('obs_pval_mse_median', 0.0),
            'obs_pval_mae_median': metrics.get('obs_pval_mae_median', 0.0),
            'obs_pval_mae_r2_median': metrics.get('obs_pval_mae_r2_median', 0.0),
            'obs_pval_perplexity_median': metrics.get('obs_pval_perplexity_median', 0.0),
            
            'obs_peak_auc_median': metrics.get('obs_peak_auc_median', 0.0),

            # DSF transition observability
            'dsf_upsampled_count': metrics.get('dsf_transitions/upsampled_count', 0),
            'dsf_downsampled_count': metrics.get('dsf_transitions/downsampled_count', 0),
            'dsf_same_count': metrics.get('dsf_transitions/same_count', 0),
            'control_non1_count': metrics.get('dsf_transitions/control_non1_count', 0),

            # Training-set supertrack monitor metrics
            'train_st_depth_ratio': metrics.get('train_st/depth_ratio', np.nan),
            'train_st_runtype_mse': metrics.get('train_st/runtype_mse', np.nan),
            'train_st_readlen_mse': metrics.get('train_st/readlen_mse', np.nan),
            
            # EMA values
            **ema_values
        }
        
        # Add to progress data
        self.progress_data.append(record)
        
        # Save to CSV every 100 batches or every batch if it's the first few batches
        if batch_idx % 100 == 0 or batch_idx < 10:
            self._save_progress_to_csv(epoch, batch_idx)
    
    def _update_progress_monitoring(self, metrics, loss_dict, specific_ema_alpha = 0.005):
        """
        Update progress monitoring with EMA tracking and check for learning rate adjustment.
        
        Args:
            metrics: Dictionary of computed metrics
            loss_dict: Dictionary of loss values
        """
            
        # Initialize specific EMA tracking for requested metrics (alpha=0.005)
        if not hasattr(self, 'specific_ema'):
            self.specific_ema = {}
            self.specific_ema_alpha = specific_ema_alpha
        
        
        # Add loss values (negated for monitoring increasing trends)
        loss_keys = ["imp_count_loss", "obs_count_loss", "imp_pval_loss", "obs_pval_loss", "imp_peak_loss", "obs_peak_loss"]
        
        # Update specific EMA tracking for requested metrics
        specific_metrics = [
            "imp_pval_r2_median", "imp_count_r2_median",
            "imp_pval_spearman_median", "imp_count_spearman_median", 
            "imp_pval_pearson_median", "imp_count_pearson_median",
            "imp_pval_mae_r2_median", "imp_count_mae_r2_median",
            "obs_pval_mae_r2_median", "obs_count_mae_r2_median",
            "imp_peak_auc_median", "obs_peak_auc_median"
        ]
        
        for key in specific_metrics + loss_keys:
            if key in metrics:
                if key not in metrics:
                    continue
                value = metrics[key]
                if key not in self.specific_ema:
                    self.specific_ema[key] = value
                else:
                    self.specific_ema[key] = self.specific_ema_alpha * value + (1 - self.specific_ema_alpha) * self.specific_ema[key]
            else:
                if key not in loss_dict:
                    continue
                value = loss_dict[key]
                if key not in self.specific_ema:
                    self.specific_ema[key] = value
                else:
                    self.specific_ema[key] = self.specific_ema_alpha * value + (1 - self.specific_ema_alpha) * self.specific_ema[key]
        

    def _log_metrics(self, metrics, loss_dict, batch_idx):
        """
        Log per-feature metrics with IQR, median, min, max, and EMA values.
        
        Args:
            metrics: Dictionary of computed per-feature metrics with aggregated statistics
            loss_dict: Dictionary of loss values  
            batch_idx: Current batch index
        """
        if not self.is_main_process:
            return
            
        # Log current batch per-feature metrics (IQR, median, min, max)
        if metrics:
            print(f"Batch {batch_idx} Per-Feature Metrics (Aggregated):")
            
            # Group metrics by type for better readability
            metric_types = ['imp_count_r2', 'imp_count_spearman', 'imp_count_pearson',
                           'imp_pval_r2', 'imp_pval_spearman', 'imp_pval_pearson',
                           'obs_count_r2', 'obs_count_spearman', 'obs_count_pearson',
                           'obs_pval_r2', 'obs_pval_spearman', 'obs_pval_pearson']
            
            for metric_type in metric_types:
                # Check if we have all aggregation statistics for this metric
                median_key = f"{metric_type}_median"
                iqr_key = f"{metric_type}_iqr"
                min_key = f"{metric_type}_min"
                max_key = f"{metric_type}_max"
                
                if all(key in metrics for key in [median_key, iqr_key, min_key, max_key]):
                    median_val = metrics[median_key]
                    iqr_val = metrics[iqr_key]
                    min_val = metrics[min_key]
                    max_val = metrics[max_key]
                    
                    if not any(np.isnan([median_val, iqr_val, min_val, max_val])):
                        print(f"  {metric_type}: median={median_val:.4f}, IQR={iqr_val:.4f}, min={min_val:.4f}, max={max_val:.4f}")
        
        # Log EMA values if available (using median values for EMA tracking)
        if hasattr(self, 'prog_mon_ema') and len(self.prog_mon_ema) > 0:
            print(f"EMA Metrics (based on median values):")
            for key, value in self.prog_mon_ema.items():
                if 'loss' in key:
                    # Show original loss value (un-negated)
                    print(f"  EMA_{key}: {-1*value:.4f}")
                else:
                    print(f"  EMA_{key}: {value:.4f}")
    
    def _validate_batch(self, batch):
        """
        Validate that the batch has the expected structure from CANDIIterableDataset.
        
        Args:
            batch: Dictionary containing batch data from CANDIIterableDataset
            
        Returns:
            bool: True if batch is valid, False otherwise
        """
        if not isinstance(batch, dict):
            return False
            
        # Expected keys from CANDIIterableDataset
        expected_keys = {'x_data', 'x_meta', 'x_avail', 'x_dna', 
                        'y_data', 'y_meta', 'y_avail', 'y_pval', 'y_peaks', 'y_dna'}
        
        if not all(key in batch for key in expected_keys):
            missing_keys = expected_keys - set(batch.keys())
            if self.is_main_process:
                print(f"Missing keys in batch: {missing_keys}")
            return False
            
        # Check that all values are tensors
        for key, value in batch.items():
            if key != 'sample_id' and not isinstance(value, torch.Tensor):
                if self.is_main_process:
                    print(f"Non-tensor value for key {key}: {type(value)}")
                return False
                
        return True
    
    def _move_batch_to_device(self, batch):
        """
        Move all tensor values in the batch to the training device.
        
        Args:
            batch: Dictionary containing batch data
            
        Returns:
            dict: Batch with tensors moved to device
        """
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device, non_blocking=True)
            else:
                device_batch[key] = value  # Keep non-tensors as is (e.g., sample_id)
        return device_batch
    
    def _log_batch_info(self, batch, batch_idx, loss_dict=None):
        """
        Log information about the batch for debugging.
        
        Args:
            batch: Dictionary containing batch data
            batch_idx: Index of the current batch
            loss_dict: Optional dictionary containing loss values
        """
        print(f"Batch {batch_idx} info:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape {value.shape}, dtype {value.dtype}, device {value.device}")
            else:
                print(f"  {key}: {type(value)} - {value}")
        
        if loss_dict is not None:
            print(f"  Losses:")
            for loss_name, loss_value in loss_dict.items():
                print(f"    {loss_name}: {loss_value:.4f}")
        
        print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB" if torch.cuda.is_available() else "  CPU mode")
    
    def _save_checkpoint(self, epoch, model_name):
        """
        Save model checkpoint after each epoch.
        
        Args:
            epoch: Current epoch number (0-indexed)
            model_name: Name of the model for checkpoint naming
        """
        if not self.is_main_process:
            return
            
        # Set up checkpoint directory if not already done
        if self.checkpoint_dir is None:
            self.checkpoint_dir = Path(self.progress_dir) / "checkpoints"
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Remove previous checkpoint if it exists
        if self.current_checkpoint_path and self.current_checkpoint_path.exists():
            self.current_checkpoint_path.unlink()
            if self.is_main_process:
                print(f"🗑️  Removed previous checkpoint: {self.current_checkpoint_path.name}")
        
        # Create new checkpoint path
        checkpoint_name = f"{model_name}_epoch_{epoch+1}.pt"
        self.current_checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save model state dict
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        try:
            torch.save(model_to_save.state_dict(), self.current_checkpoint_path)
            if self.is_main_process:
                print(f"💾 Epoch {epoch+1} checkpoint saved: {self.current_checkpoint_path.name}")
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"⚠️  Checkpoint save failed due to CUDA error: {e}")
                print("    Training will continue, but this checkpoint was not saved.")
            else:
                raise

##=========================================== Loader =====================================================##

class CANDI_LOADER(object):
    def __init__(self, model_path, hyper_parameters, DNA=False):
        self.model_path = model_path
        self.hyper_parameters = hyper_parameters
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.DNA = DNA

    @staticmethod
    def _load_state_dict_latent_compat(model, checkpoint, enable_latent_kl: bool, context: str):
        allowed_suffixes = {
            "latent_mu_head.weight",
            "latent_mu_head.bias",
            "latent_logvar_head.weight",
            "latent_logvar_head.bias",
        }
        if not enable_latent_kl:
            model.load_state_dict(checkpoint)
            return

        incompatible = model.load_state_dict(checkpoint, strict=False)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        bad_missing = [k for k in missing if not any(k.endswith(s) for s in allowed_suffixes)]
        bad_unexpected = [k for k in unexpected if not any(k.endswith(s) for s in allowed_suffixes)]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                f"{context}: state-dict mismatch beyond latent KL heads. "
                f"missing={bad_missing}, unexpected={bad_unexpected}"
            )
        if missing or unexpected:
            print(f"[latent_kl_compat] {context}: allowed missing={missing}, allowed unexpected={unexpected}")

    def load_CANDI(self):
        signal_dim = self.hyper_parameters["signal_dim"]
        dropout = self.hyper_parameters["dropout"]
        nhead = self.hyper_parameters["nhead"]
        n_sab_layers = self.hyper_parameters["n_sab_layers"]
        metadata_embedding_dim = self.hyper_parameters["metadata_embedding_dim"]
        context_length = self.hyper_parameters["context_length"]

        n_cnn_layers = self.hyper_parameters["n_cnn_layers"]
        conv_kernel_size = self.hyper_parameters["conv_kernel_size"]
        pool_size = self.hyper_parameters["pool_size"]
        separate_decoders = self.hyper_parameters["separate_decoders"]
        norm = self.hyper_parameters.get("norm", "batch")  # Default to "batch" for backward compatibility
        attention_type = self.hyper_parameters.get("attention_type", "dual")  # Default to "dual" for backward compatibility
        output_ff = self.hyper_parameters.get("output_ff", False)  # Default to False for backward compatibility
        xl_dna = self.hyper_parameters.get("xl_dna", False)  # Default to False for backward compatibility
        mask_stem = self.hyper_parameters.get("mask_stem", False)  # Default to False for backward compatibility
        decoder_type = self.hyper_parameters.get("decoder_type", "fixed")
        moe_experts = int(self.hyper_parameters.get("moe_experts", 4))
        nq_chunk_multiplier = int(self.hyper_parameters.get("nq_chunk_multiplier", 1))
        condconv_k = int(self.hyper_parameters.get("condconv_k", 3))
        condconv_routing = str(self.hyper_parameters.get("condconv_routing", "hybrid"))
        condconv_gate_activation = str(self.hyper_parameters.get("condconv_gate_activation", "sigmoid"))
        dist_type = self.hyper_parameters.get("dist_type", "gaussian")
        signal_transform = self.hyper_parameters.get("signal_transform", "arcsinh")
        enable_latent_kl = bool(self.hyper_parameters.get("enable_latent_kl", False))
        latent_std_min = float(self.hyper_parameters.get("latent_std_min", 0.01))
        latent_std_max = float(self.hyper_parameters.get("latent_std_max", 1.0))
        latent_reparam_mode = str(self.hyper_parameters.get("latent_reparam_mode", "clamp"))
        latent_sample_train_only = bool(self.hyper_parameters.get("latent_sample_train_only", True))
        num_assays = int(self.hyper_parameters.get("num_assays", signal_dim))
        # Backward compatibility: old checkpoints may carry num_runtypes=4.
        num_runtypes = int(self.hyper_parameters.get("num_runtypes", 2))
        
        if self.DNA:
            if self.hyper_parameters["unet"]:
                model = CANDI_UNET(
                    signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
                    n_sab_layers, pool_size=pool_size, dropout=dropout, context_length=context_length, 
                    separate_decoders=separate_decoders, norm=norm, attention_type=attention_type,
                    output_ff=output_ff, xl_dna=xl_dna, mask_stem=mask_stem,
                    num_assays=num_assays, num_runtypes=num_runtypes,
                    dist_type=dist_type, signal_transform=signal_transform,
                    decoder_type=decoder_type, moe_experts=moe_experts,
                    nq_chunk_multiplier=nq_chunk_multiplier, condconv_k=condconv_k,
                    condconv_routing=condconv_routing, condconv_gate_activation=condconv_gate_activation,
                    enable_latent_kl=enable_latent_kl, latent_std_min=latent_std_min,
                    latent_std_max=latent_std_max, latent_reparam_mode=latent_reparam_mode,
                    latent_sample_train_only=latent_sample_train_only)
            else:
                model = CANDI(
                    signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
                    n_sab_layers, pool_size=pool_size, dropout=dropout, context_length=context_length, 
                    separate_decoders=separate_decoders, norm=norm, attention_type=attention_type,
                    output_ff=output_ff, xl_dna=xl_dna, mask_stem=mask_stem,
                    num_assays=num_assays, num_runtypes=num_runtypes,
                    dist_type=dist_type, signal_transform=signal_transform,
                    decoder_type=decoder_type, moe_experts=moe_experts,
                    nq_chunk_multiplier=nq_chunk_multiplier, condconv_k=condconv_k,
                    condconv_routing=condconv_routing, condconv_gate_activation=condconv_gate_activation,
                    enable_latent_kl=enable_latent_kl, latent_std_min=latent_std_min,
                    latent_std_max=latent_std_max, latent_reparam_mode=latent_reparam_mode,
                    latent_sample_train_only=latent_sample_train_only)
        else:
            model = CANDI(
                signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
                n_sab_layers, pool_size=pool_size, dropout=dropout, context_length=context_length,
                separate_decoders=separate_decoders, norm=norm, attention_type=attention_type,
                output_ff=output_ff, xl_dna=xl_dna, mask_stem=mask_stem,
                num_assays=num_assays, num_runtypes=num_runtypes,
                dist_type=dist_type, signal_transform=signal_transform,
                decoder_type=decoder_type, moe_experts=moe_experts,
                nq_chunk_multiplier=nq_chunk_multiplier, condconv_k=condconv_k,
                condconv_routing=condconv_routing, condconv_gate_activation=condconv_gate_activation,
                enable_latent_kl=enable_latent_kl, latent_std_min=latent_std_min,
                latent_std_max=latent_std_max, latent_reparam_mode=latent_reparam_mode,
                latent_sample_train_only=latent_sample_train_only)

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self._load_state_dict_latent_compat(
            model,
            checkpoint,
            enable_latent_kl=enable_latent_kl,
            context=f"CANDI_LOADER({self.model_path})",
        )

        model = model.to(self.device)
        return model

##=========================================== DDP Utilities ==============================================##

def init_distributed():
    """Initialize distributed training. Returns rank, world_size, local_rank."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # Check if the requested device exists
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for DDP training")
        
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(f"Requested local_rank {local_rank} but only {torch.cuda.device_count()} CUDA devices available")
        
        # Initialize the process group
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return None, None, None

def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()

def check_gpu_availability():
    """Check GPU availability and provide guidance for DDP setup."""
    if not torch.cuda.is_available():
        print("❌ CUDA is not available on this system.")
        print("   For DDP training, you need CUDA-enabled GPUs.")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"🔍 Found {gpu_count} CUDA device(s):")
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"   GPU {i}: {gpu_name}")
    
    return gpu_count > 0

##=========================================== CLI Interface ===============================================##

def create_argument_parser():
    """Create and configure the argument parser with organized argument groups."""
    parser = argparse.ArgumentParser(
        description="CANDI: Context-Aware Neural Data Imputation - Modern Training Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Basic EIC training with default settings (full assay masking only)
            python train.py --eic --epochs 10 --batch-size 16
            
            # Training with hybrid masking (full assay + full loci)
            python train.py --eic --epochs 10 --p-full-assay 1.0 --p-full-loci 0.5 --mask-fraction 0.2
            
            # Training with all three masking strategies
            python train.py --eic --epochs 10 --p-full-assay 1.0 --p-full-loci 0.3 --p-chunks 0.3
            
            # Training with full loci masking only (no full assay masking)
            python train.py --eic --epochs 10 --p-full-assay 0.0 --p-full-loci 1.0 --mask-fraction 0.25 --chunk-size 50
            
            # Multi-GPU training with mixed precision
            python train.py --eic --ddp --mixed-precision --epochs 20 --batch-size 32
            
            # Custom model architecture with suffix
            python train.py --merged --nhead 12 --n-sab-layers 6 --expansion-factor 4 --name-suffix "experiment1"
            
            # Training with validation and checkpointing
            python train.py --eic --enable-validation --save-dir ./models --checkpoint-freq 5
            
            # U-Net model with custom loci strategy
            python train.py --eic --unet --loci-gen full_chr --num-loci 1000 --name-suffix "unet_test"
                    """)
    
    # === DATA CONFIGURATION ===
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--eic', action='store_true',
                           help='Use EIC dataset (default: merged dataset)')
    data_group.add_argument('--merged', action='store_true',
                           help='Use merged dataset (default if neither --eic nor --merged specified)')
    data_group.add_argument('--data-path', type=str, 
                           default="/project/6014832/mforooz/",
                           help='Base path to the datasets')
    data_group.add_argument('--data-backend', type=str, default='npz',
                           choices=['npz', 'h5', 'zarr'],
                           help='Data backend to use: legacy NPZ files, prepared HDF5 store, or prepared Zarr store (default: npz)')
    data_group.add_argument('--prepared-data-path', type=str, default=None,
                           help='Explicit path to the prepared data root when using --data-backend h5 or zarr')
    data_group.add_argument('--num-loci', '-m', type=int, default=5000,
                           help='Number of genomic loci to generate for training')
    data_group.add_argument('--context-length', type=int, default=3072,
                           help='Context length for genomic windows (in bins)')
    data_group.add_argument('--loci-gen', type=str, default='random', 
                           choices=['random', 'ccre', 'mixture', 'full_chr', 'gw'],
                           help='Strategy for generating genomic loci')
    data_group.add_argument('--ccre-fraction', type=float, default=0.3,
                           help='Fraction of loci drawn from cCREs when --loci-gen mixture is used (default: 0.3)')
    data_group.add_argument('--must-have-chr-access', action='store_true',
                           help='Require chromosome access for all experiments')
    data_group.add_argument('--min-avail', type=int, default=3,
                           help='Minimum number of available experiments per biosample')
    data_group.add_argument('--balanced-bios-order', dest='balanced_bios_order',
                           action=argparse.BooleanOptionalAction, default=True,
                           help='Enable train-time 5-quantile round-robin biosample ordering (use --no-balanced-bios-order to disable)')
    
    # === MODEL ARCHITECTURE ===
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--nhead', type=int, default=9,
                            help='Number of attention heads in transformer')
    model_group.add_argument('--n-sab-layers', type=int, default=4,
                            help='Number of self-attention blocks')
    model_group.add_argument('--n-cnn-layers', type=int, default=3,
                            help='Number of CNN layers in encoder/decoder')
    model_group.add_argument('--conv-kernel-size', type=int, default=3,
                            help='Convolution kernel size')
    model_group.add_argument('--pool-size', type=int, default=2,
                            help='Pooling size for CNN layers')
    model_group.add_argument('--expansion-factor', type=int, default=3,
                            help='Channel expansion factor for CNN layers')
    model_group.add_argument('--dropout', type=float, default=0.1,
                            help='Dropout rate')
    model_group.add_argument('--pos-enc', type=str, default='relative',
                            choices=['relative', 'absolute'],
                            help='Type of positional encoding')
    model_group.add_argument('--attention-type', type=str, default='xtransformers',
                            choices=['dual', 'xtransformers'],
                            help='Attention mechanism: dual (DualAttentionEncoderBlock) or xtransformers (RoPE)')
    model_group.add_argument('--separate-decoders', action='store_true', default=True,
                            help='Use separate decoders for count and p-value prediction')
    model_group.add_argument('--shared-decoders', action='store_true',
                            help='Use shared decoder (overrides --separate-decoders)')
    model_group.add_argument('--unet', action='store_true',
                           help='Use U-Net skip connections')
    model_group.add_argument('--norm-type', type=str, default='layer',
                           choices=['batch', 'layer', 'group', 'instance', 'weight', 'rms', 'none'],
                           help='Normalization type for convolutional layers')
    model_group.add_argument('--output-ff', action='store_true',
                           help='Include feed-forward layers in output heads (NegBinom, Gaussian, Peak)')
    model_group.add_argument('--xl-dna', action='store_true',
                           help='Use XL DNA encoder with wider channels and biophysical kernels (15bp initial, then 5bp)')
    model_group.add_argument('--mask-stem', action='store_true',
                           help='Use MaskStem for handling missing data. Processes (value, mask) pairs with a depth-wise 1x1 conv before the signal encoder.')
    model_group.add_argument('--decoder-type', type=str, default='fixed',
                           choices=['fixed', 'query_moe', 'query_dynconv', 'query_condconv'],
                           help='Decoder implementation: fixed (legacy default), query_moe, query_dynconv, or query_condconv')
    model_group.add_argument('--moe-experts', type=int, default=4,
                           help='Number of experts for query_moe decoder (ignored for other decoder types)')
    model_group.add_argument('--nq-chunk-multiplier', type=int, default=1,
                           help='Chunk size multiplier for query decoding: max_nq_per_chunk = multiplier * batch_size (default: 1)')
    model_group.add_argument('--condconv-k', type=int, default=3,
                           help='Number of CondConv basis kernels for query_condconv decoder (default: 3)')
    model_group.add_argument('--condconv-routing', type=str, default='hybrid',
                           choices=['query', 'feature', 'hybrid'],
                           help='Routing input for query_condconv gates: query metadata, decoded features, or hybrid concat (default: hybrid)')
    model_group.add_argument('--condconv-gate-activation', type=str, default='sigmoid',
                           choices=['sigmoid', 'softmax'],
                           help='Gate activation for query_condconv routing weights (default: sigmoid)')
    
    # === TRAINING CONFIGURATION ===
    training_group = parser.add_argument_group('Training Configuration')
    training_group.add_argument('--epochs', type=int, default=10,
                               help='Number of training epochs')
    training_group.add_argument('--batch-size', type=int, default=30,
                               help='Training batch size')
    training_group.add_argument('--learning-rate', '--lr', type=float, default=1e-3,
                               help='Initial learning rate')
    training_group.add_argument('--optimizer', type=str, default='adamax',
                               choices=['adamax', 'adam', 'adamw', 'sgd', 'radam', 'adabelief', 'muon'],
                               help='Optimizer type')
    training_group.add_argument('--weight-decay', type=float, default=None,
                               help='Weight decay (L2 penalty). If not specified, uses optimizer-specific defaults: Adam/Adamax=0.0, AdamW=0.0, SGD=0.0 (conservative)')
    training_group.add_argument('--momentum', type=float, default=0.9,
                               help='Momentum factor for SGD (default: 0.9)')
    training_group.add_argument('--beta1', type=float, default=0.9,
                               help='Beta1 for Adam/AdamW/Adamax (default: 0.9)')
    training_group.add_argument('--beta2', type=float, default=0.999,
                               help='Beta2 for Adam/AdamW/Adamax (default: 0.999)')
    # Scheduler is always cosine (linear warmup + cosine annealing)
    training_group.add_argument('--inner-epochs', type=int, default=1,
                               help='Number of inner epochs per batch')
    training_group.add_argument('--grad-accum-steps', type=int, default=1,
                               help='Number of micro-batches to accumulate before optimizer step (default: 1)')
    training_group.add_argument('--dataloader-workers', type=int, default=None,
                               help='Explicit DataLoader worker count. If unset, defaults to min(4, SLURM_CPUS_PER_TASK) when under Slurm, otherwise min(4, local CPU count).')
    training_group.add_argument('--disable-validation', action='store_true',
                               help='Disable validation monitoring during training (enabled by default)')
    training_group.add_argument('--val-freq', type=float, default=0.1,
                               help='Validation frequency as fraction of total training (e.g., 0.1 = every 10%%)')
    training_group.add_argument('--enable-supertrack-train-monitor', action='store_true',
                               help='Enable lightweight train-set supertrack prompt sensitivity monitoring')
    training_group.add_argument('--supertrack-train-monitor-every', type=int, default=100,
                               help='Run train-set supertrack monitor every N batches (default: 100)')
    training_group.add_argument('--supertrack-train-monitor-max-batch', type=int, default=8,
                               help='Max batch items to use per monitor pass to control overhead (default: 8)')
    training_group.add_argument('--wandb-log-every', type=int, default=50,
                               help='Log training metrics to W&B every N batches (default: 50)')

    training_group.add_argument('--clip-mode', type=str, default='norm',
                               choices=['norm', 'value', 'none'],
                               help='Gradient clipping mode: norm, value, or none (default: norm)')
    training_group.add_argument('--clip-value', type=float, default=5.0,
                               help='Maximum gradient norm or value for clipping (default: 5.0)')

    training_group.add_argument('--count-weight', type=float, default=0.5,
                               help='Weight for count loss in multi-task learning (default: 0.5)')
    training_group.add_argument('--pval-weight', type=float, default=1.0,
                               help='Weight for p-value loss in multi-task learning (default: 1.0)')
    training_group.add_argument('--peak-weight', type=float, default=0.25,
                               help='Weight for peak loss in multi-task learning (default: 0.25)')

    training_group.add_argument('--obs-weight', type=float, default=0.25,
                               help='Weight for observed (upsampling) losses (default: 0.25)')
    training_group.add_argument('--imp-weight', type=float, default=1.0,
                               help='Weight for imputed (masked) losses (default: 1.0)')

    training_group.add_argument('--enable-assay-ema-balance', action='store_true',
                               help='Enable per-assay x branch x head EMA frequency balancing in loss aggregation')
    training_group.add_argument('--enable-hier-reduction', action='store_true',
                               help='Enable hierarchical per-assay reduction before branch/task aggregation')

    training_group.add_argument('--assay-ema-decay', type=float, default=0.99,
                               help='EMA decay for assay availability frequency estimates')
    training_group.add_argument('--assay-ema-eps', type=float, default=1e-6,
                               help='Epsilon for inverse-frequency assay weights')
    training_group.add_argument('--assay-ema-warmup-steps', type=int, default=100,
                               help='Warmup steps before EMA assay weights are applied')
    training_group.add_argument('--assay-ema-weight-min', type=float, default=0.02,
                               help='Minimum clipped assay weight')
    training_group.add_argument('--assay-ema-weight-max', type=float, default=1.0,
                               help='Maximum clipped assay weight')

    training_group.add_argument('--enable-fg-bg-balance', action='store_true',
                               help='Enable foreground/background balancing for count and signal using GT peak')
    training_group.add_argument('--fg-weight', type=float, default=0.5,
                               help='Foreground weight when fg/bg balancing is active')
    training_group.add_argument('--fg-min-fraction', type=float, default=0.02,
                               help='Minimum foreground fraction to apply fg/bg split, else fallback to baseline reduction')

    training_group.add_argument('--enable-uncertainty-weighting', action='store_true',
                               help='Enable uncertainty-based dynamic multi-task weighting (count/pval/peak)')
    training_group.add_argument('--uncertainty-warmup-steps', type=int, default=100,
                               help='Warmup steps before uncertainty weighting activates')
    training_group.add_argument('--uncertainty-init-logvar', type=float, default=0.0,
                               help='Initial log-variance for uncertainty weighting')
                               
    training_group.add_argument('--enable-count-rstable-objective', action='store_true',
                               help='Replace count loss with stable normalized ratio objective after warmup')
    training_group.add_argument('--count-rstable-eps', type=float, default=1e-6,
                               help='Numerical epsilon added in R_stable denominator')
    training_group.add_argument('--count-rstable-ema-decay', type=float, default=0.99,
                               help='EMA decay for per-assay per-branch null/oracle baseline stats')
    training_group.add_argument('--count-rstable-warmup-steps', type=int, default=100,
                               help='Warmup steps before count R_stable objective activation')
    training_group.add_argument('--count-rstable-denom-min', type=float, default=1e-4,
                               help='Minimum absolute denominator magnitude for stable ratio')
    training_group.add_argument('--count-rstable-r-max', type=float, default=5.0,
                               help='Absolute clamp for per-assay R_stable values')
    training_group.add_argument('--count-rstable-dispersion-min', type=float, default=1e-3,
                               help='Minimum dispersion used for NB baseline/oracle builders')
    training_group.add_argument('--count-rstable-dispersion-max', type=float, default=1e4,
                               help='Maximum dispersion used for NB baseline/oracle builders')
    
    training_group.add_argument('--dist-type', type=str, default='laplace',
                               choices=['gaussian', 'laplace', 'studentst', 'mse', 'mae', 'gaussian_const', 'laplace_const', 'gamma'],
                               help='Distribution type for signal prediction: gaussian (Gaussian NLL), laplace (Laplace NLL), studentst (Student-t NLL), \
                               mse (deterministic MSE), mae (deterministic MAE), gaussian_const (Gaussian NLL with learned constant variance per assay), \
                               laplace_const (Laplace NLL with learned constant scale per assay), gamma (Gamma NLL, positive support)')
    training_group.add_argument('--signal-transform', type=str, default='arcsinh',
                               choices=['arcsinh', 'log1p', 'none'],
                               help='Signal transformation: arcsinh (inverse hyperbolic sine, default), log1p (log(1+x)), none (no transformation)')

    training_group.add_argument('--enable-latent-kl', action='store_true', default=False,
                               help='Enable latent diagonal-Gaussian KL regularization (default: disabled)')
    training_group.add_argument('--latent-kl-weight', type=float, default=1e-4,
                               help='Maximum KL coefficient beta after warmup (default: 1e-4)')
    training_group.add_argument('--latent-kl-warmup-steps', type=int, default=1000,
                               help='Warmup steps to ramp KL beta from 0 to latent-kl-weight')
    training_group.add_argument('--latent-std-min', type=float, default=0.01,
                               help='Lower bound for latent std during reparameterization')
    training_group.add_argument('--latent-std-max', type=float, default=1.0,
                               help='Upper bound for latent std during reparameterization')
    training_group.add_argument('--latent-reparam-mode', type=str, default='clamp',
                               choices=['clamp', 'softplus'],
                               help='Latent std parameterization: clamp(logvar) or softplus(raw_std)')
    training_group.add_argument('--latent-sample-train-only', action=argparse.BooleanOptionalAction, default=True,
                               help='If true, use posterior mean at eval and sample only during training')
    training_group.add_argument('--latent-deterministic-warmup-steps', type=int, default=0,
                               help='Force deterministic latent (z=mu) for first N global steps during training')
    
    training_group.add_argument('--reverse-complement-prob', type=float, default=0.5,
                               help='Probability of applying reverse complement augmentation per batch (0.0=disabled, 1.0=always, default: 0.5)')
    
    training_group.add_argument('--p-full-loci', type=float, default=0.0,
                               help='Probability of applying full loci masking (mask same loci across all assays)')
    training_group.add_argument('--p-full-assay', type=float, default=1.0,
                               help='Probability of applying full assay masking')
                               
    training_group.add_argument('--p-chunks', type=float, default=0.0,
                               help='Probability of applying independent chunk masking per assay (default: 0.0)')
    training_group.add_argument('--mask-fraction', type=float, default=0.10,
                               help='Fraction of loci to mask for full_loci and chunks strategies (default: 0.20)')
    training_group.add_argument('--chunk-size', type=int, default=40,
                               help='Size of chunks to mask for full_loci and chunks strategies (default: 40, ~1kb at 25bp)')
    
    # === SYSTEM CONFIGURATION ===
    system_group = parser.add_argument_group('System Configuration')
    system_group.add_argument('--device', type=str, default=None,
                             help='Device to use (cuda:0, cpu, etc.). Auto-detect if not specified')
    # Mixed precision is now opt-in (default: disabled) to avoid AMP instability on hard batches.
    system_group.add_argument('--mixed-precision', action='store_true', default=False,
                             help='Enable mixed precision training (default: False)')
    system_group.add_argument('--no-mixed-precision', action='store_true',
                             help='Disable mixed precision training (kept for backward compatibility)')
    system_group.add_argument('--ddp', action='store_true',
                             help='Enable Distributed Data Parallel training')
    system_group.add_argument('--rank', type=int, default=None,
                             help='Process rank for DDP (auto-detected if not specified)')
    system_group.add_argument('--world-size', type=int, default=None,
                             help='World size for DDP (auto-detected if not specified)')
    system_group.add_argument('--seed', type=int, default=42,
                             help='Random seed for reproducibility')
    system_group.add_argument('--check-gpus', action='store_true',
                             help='Check GPU availability and exit')
    
    # === MODEL SAVING/LOADING ===
    io_group = parser.add_argument_group('Model I/O')
    io_group.add_argument('--save-dir', type=str, default='./models',
                         help='Directory to save trained models')
    io_group.add_argument('--progress-dir', type=str, default='./progress',
                         help='Directory to save training progress CSV files')
    io_group.add_argument('--checkpoint', type=str, default=None,
                         help='Path to checkpoint to resume training from')
    io_group.add_argument('--checkpoint-freq', type=int, default=1,
                         help='Save checkpoint every N epochs')
    io_group.add_argument('--model-name', type=str, default=None,
                         help='Custom model name (auto-generated if not specified)')
    io_group.add_argument('--name-suffix', type=str, default=None,
                         help='Suffix to append to auto-generated model name (format: YYYYMMDD_HHMMSS_CANDI[UNET]_dataset_lociStrategy_numLoci_suffix)')
    io_group.add_argument('--no-save', action='store_true',
                         help='Do not save the trained model')
    
    # === CONFIGURATION FILE ===
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument('--config', type=str, default=None,
                             help='Path to YAML/JSON configuration file')
    config_group.add_argument('--save-config', type=str, default=None,
                             help='Save current configuration to file')
    
    # === ADVANCED OPTIONS ===
    advanced_group = parser.add_argument_group('Advanced Options')
    # Example usage in CLI: --dsf-list 1,2,4
    def parse_dsf_list(s):
        if isinstance(s, list):
            return s
        try:
            items = [x.strip() for x in s.split(',')]
            dsf_values = []
            for x in items:
                if not x.isdigit():
                    raise ValueError
                v = int(x)
                if v < 1:
                    raise ValueError
                dsf_values.append(v)
            if len(dsf_values) == 0:
                raise ValueError
            return dsf_values
        except Exception:
            raise argparse.ArgumentTypeError("dsf-list must be a comma-separated list of positive integers (e.g., 1,2,4)")
    advanced_group.add_argument(
        '--dsf-list', type=parse_dsf_list, default=[1, 2, 4],
        help='Downsampling factors to use as a comma-separated list of positive integers (e.g., --dsf-list 1,2,4)'
    )
    
    advanced_group.add_argument('--specific_ema_alpha', type=float, default=0.005,
                               help='Alpha for specific EMA tracking')
    advanced_group.add_argument('--debug', action='store_true',
                               help='Enable debug mode with extra logging')
    advanced_group.add_argument('--fill-prompt-mode', type=str, default='median',
                               choices=['sample', 'median', 'mode'],
                               help='Method for filling missing metadata in prompts: '
                                    'sample (random sampling from dataset distribution), '
                                    'median (median for numeric, mode for categorical), '
                                    'mode (mode for all fields)')
    advanced_group.add_argument('--enable-per-assay-dsf-sampling', action='store_true',
                               help='Enable per-assay DSF sampling for count X/Y and control X data.')
    advanced_group.add_argument('--per-assay-dsf-sampling-mode', type=str, default='uniform',
                               choices=['uniform'],
                               help='Sampling mode for per-assay DSF (currently only uniform).')
    
    return parser

def load_config_file(config_path):
    """Load configuration from YAML or JSON file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            try:
                import yaml
                return yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
        elif config_path.suffix.lower() == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

def save_config_file(config_dict, config_path):
    """Save configuration to YAML or JSON file."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            try:
                import yaml
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            except ImportError:
                # Fallback to JSON if PyYAML not available
                config_path = config_path.with_suffix('.json')
                with open(config_path, 'w') as jf:
                    json.dump(config_dict, jf, indent=2)
        else:
            json.dump(config_dict, f, indent=2)

def validate_arguments(args):
    """Validate argument combinations and set defaults."""
    errors = []
    
    # Dataset selection
    if args.eic and args.merged:
        errors.append("Cannot specify both --eic and --merged")
    elif not args.eic and not args.merged:
        args.merged = True  # Default to merged
    
    # Mixed precision
    if args.no_mixed_precision:
        args.mixed_precision = False
    
    
    # Decoder configuration
    if args.shared_decoders:
        args.separate_decoders = False
    
    # DDP validation
    if args.ddp:
        if args.rank is None:
            args.rank = int(os.environ.get('RANK', 0))
        if args.world_size is None:
            args.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        if args.world_size <= 1:
            print("Warning: DDP requested but world_size <= 1. Disabling DDP.")
            args.ddp = False
        elif not torch.cuda.is_available():
            print("Warning: DDP requested but CUDA is not available. Disabling DDP.")
            args.ddp = False
        elif 'LOCAL_RANK' in os.environ and int(os.environ['LOCAL_RANK']) >= torch.cuda.device_count():
            print(f"Warning: LOCAL_RANK={os.environ['LOCAL_RANK']} but only {torch.cuda.device_count()} local CUDA devices available. Disabling DDP.")
            args.ddp = False
    
    # Path validation
    data_path = Path(args.data_path)
    if not data_path.exists():
        errors.append(f"Data path does not exist: {data_path}")
    
    # Model architecture validation
    if args.nhead <= 0:
        errors.append("Number of attention heads must be positive")
    if args.n_sab_layers < 0:
        errors.append("Number of SAB layers must be zero or positive")
    if args.context_length <= 0:
        errors.append("Context length must be positive")
    if args.grad_accum_steps <= 0:
        errors.append("--grad-accum-steps must be positive")
    if getattr(args, 'moe_experts', 1) <= 0:
        errors.append("--moe-experts must be positive")
    if getattr(args, 'nq_chunk_multiplier', 1) <= 0:
        errors.append("--nq-chunk-multiplier must be positive")
    if getattr(args, 'condconv_k', 1) <= 0:
        errors.append("--condconv-k must be positive")
    if getattr(args, 'condconv_routing', 'hybrid') not in {'query', 'feature', 'hybrid'}:
        errors.append("--condconv-routing must be one of: query, feature, hybrid")
    if getattr(args, 'condconv_gate_activation', 'sigmoid') not in {'sigmoid', 'softmax'}:
        errors.append("--condconv-gate-activation must be one of: sigmoid, softmax")
    if getattr(args, 'latent_kl_weight', 0.0) < 0:
        errors.append("--latent-kl-weight must be non-negative")
    if getattr(args, 'latent_kl_warmup_steps', 0) < 0:
        errors.append("--latent-kl-warmup-steps must be >= 0")
    if getattr(args, 'latent_deterministic_warmup_steps', 0) < 0:
        errors.append("--latent-deterministic-warmup-steps must be >= 0")
    if getattr(args, 'latent_std_min', 0.0) <= 0:
        errors.append("--latent-std-min must be > 0")
    if getattr(args, 'latent_std_max', 0.0) <= 0:
        errors.append("--latent-std-max must be > 0")
    if getattr(args, 'latent_std_min', 0.0) >= getattr(args, 'latent_std_max', 0.0):
        errors.append("--latent-std-min must be < --latent-std-max")
    
    if errors:
        raise ValueError("Argument validation failed:\n" + "\n".join(f"  - {err}" for err in errors))
    
    return args

def create_model_from_args(args, signal_dim, num_assays=35, num_runtypes=2):
    """
    Create CANDI model based on CLI arguments.
    
    Args:
        num_assays: Number of distinct assay types (e.g., H3K4me3, CTCF).
                   Used for assay_embedding in MetadataEncoder.
                   Note: replaces num_sequencing_platforms per issue_supertrack.md ToDo 1.
    """
    metadata_embedding_dim = signal_dim * 4
    dist_type = getattr(args, 'dist_type', 'gaussian')
    mask_stem = getattr(args, 'mask_stem', False)
    signal_transform = getattr(args, 'signal_transform', 'arcsinh')
    decoder_type = getattr(args, 'decoder_type', 'fixed')
    moe_experts = getattr(args, 'moe_experts', 4)
    nq_chunk_multiplier = getattr(args, 'nq_chunk_multiplier', 1)
    condconv_k = getattr(args, 'condconv_k', 3)
    condconv_routing = getattr(args, 'condconv_routing', 'hybrid')
    condconv_gate_activation = getattr(args, 'condconv_gate_activation', 'sigmoid')
    enable_latent_kl = bool(getattr(args, 'enable_latent_kl', False))
    latent_std_min = float(getattr(args, 'latent_std_min', 0.01))
    latent_std_max = float(getattr(args, 'latent_std_max', 1.0))
    latent_reparam_mode = str(getattr(args, 'latent_reparam_mode', 'clamp'))
    latent_sample_train_only = bool(getattr(args, 'latent_sample_train_only', True))
    latent_deterministic_warmup_steps = int(getattr(args, 'latent_deterministic_warmup_steps', 0))
    
    if args.unet:
        model = CANDI_UNET(
            signal_dim=signal_dim,
            metadata_embedding_dim=metadata_embedding_dim,
            conv_kernel_size=args.conv_kernel_size,
            n_cnn_layers=args.n_cnn_layers,
            nhead=args.nhead,
            n_sab_layers=args.n_sab_layers,
            pool_size=args.pool_size,
            dropout=args.dropout,
            context_length=args.context_length,
            pos_enc=args.pos_enc,
            expansion_factor=args.expansion_factor,
            separate_decoders=args.separate_decoders,
            num_assays=num_assays,
            num_runtypes=num_runtypes,
            norm=args.norm_type,
            attention_type=args.attention_type,
            output_ff=args.output_ff,
            dist_type=dist_type,
            xl_dna=args.xl_dna,
            mask_stem=mask_stem,
            signal_transform=signal_transform,
            decoder_type=decoder_type,
            moe_experts=moe_experts,
            nq_chunk_multiplier=nq_chunk_multiplier,
            condconv_k=condconv_k,
            condconv_routing=condconv_routing,
            condconv_gate_activation=condconv_gate_activation,
            enable_latent_kl=enable_latent_kl,
            latent_std_min=latent_std_min,
            latent_std_max=latent_std_max,
            latent_reparam_mode=latent_reparam_mode,
            latent_sample_train_only=latent_sample_train_only,
            latent_deterministic_warmup_steps=latent_deterministic_warmup_steps
        )
    else:
        model = CANDI(
            signal_dim=signal_dim,
            metadata_embedding_dim=metadata_embedding_dim,
            conv_kernel_size=args.conv_kernel_size,
            n_cnn_layers=args.n_cnn_layers,
            nhead=args.nhead,
            n_sab_layers=args.n_sab_layers,
            pool_size=args.pool_size,
            dropout=args.dropout,
            context_length=args.context_length,
            pos_enc=args.pos_enc,
            expansion_factor=args.expansion_factor,
            separate_decoders=args.separate_decoders,
            num_assays=num_assays,
            num_runtypes=num_runtypes,
            norm=args.norm_type,
            attention_type=args.attention_type,
            output_ff=args.output_ff,
            dist_type=dist_type,
            xl_dna=args.xl_dna,
            mask_stem=mask_stem,
            signal_transform=signal_transform,
            decoder_type=decoder_type,
            moe_experts=moe_experts,
            nq_chunk_multiplier=nq_chunk_multiplier,
            condconv_k=condconv_k,
            condconv_routing=condconv_routing,
            condconv_gate_activation=condconv_gate_activation,
            enable_latent_kl=enable_latent_kl,
            latent_std_min=latent_std_min,
            latent_std_max=latent_std_max,
            latent_reparam_mode=latent_reparam_mode,
            latent_sample_train_only=latent_sample_train_only,
            latent_deterministic_warmup_steps=latent_deterministic_warmup_steps
        )
    
    return model

def setup_device(args):
    """Setup device based on arguments and availability."""
    if args.device is not None:
        device = torch.device(args.device)
    else:
        if args.ddp and 'LOCAL_RANK' in os.environ:
            device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    return device

def generate_model_name(args, timestamp=None):
    """Generate a descriptive model name based on configuration."""
    if args.model_name:
        return args.model_name
    
    # Use provided timestamp or generate new one
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    dataset_type = "eic" if args.eic else "merged"
    arch_type = "CANDI_UNET" if args.unet else "CANDI"
    loci_strategy = args.loci_gen
    num_loci = args.num_loci
    
    name_parts = [
        timestamp,
        arch_type,
        dataset_type,
        f"{loci_strategy}_{num_loci}loci"
    ]
    
    # Add suffix if provided
    if args.name_suffix:
        name_parts.append(args.name_suffix)
    
    return "_".join(name_parts)

def print_training_summary(args, model, device):
    """Print a summary of the training configuration."""

    print("=" * 80)
    print("🚀 CANDI Training Configuration")
    print("=" * 80)

    # Dataset info
    dataset_type = "EIC" if args.eic else "Merged"
    print(f"📊 Dataset: {dataset_type} ({args.data_path})")
    print(f"   Loci: {args.num_loci}, Context: {args.context_length}, Strategy: {args.loci_gen}")

    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    arch_type = "U-Net" if args.unet else "Standard"
    decoder_type = "Shared" if not args.separate_decoders else "Separate"

    print(f"🏗️  Model: CANDI-{arch_type} with {decoder_type} Decoders")
    print(f"   Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"   Architecture: {args.nhead} heads, {args.n_sab_layers} SAB layers, {args.n_cnn_layers} CNN layers")
    if args.xl_dna:
        print(f"   DNA Encoder: XL (wide channels, biophysical kernels)")
    else:
        print(f"   DNA Encoder: Standard")
    if getattr(args, 'mask_stem', False):
        print(f"   MaskStem: Enabled (depth-wise missing data handling)")

    # Print model summary
    print("\n📝 Model Summary:")
    try:
        from torchinfo import summary
        summary(model)
    except ImportError:
        print("torchinfo not installed. Printing model using print(model):")
        print(model)
    except Exception as e:
        print(f"Could not print model summary: {e}")
        print(model)

    # Training info
    print(f"🎯 Training: {args.epochs} epochs, batch size {args.batch_size}")
    world_size = args.world_size if (args.ddp and args.world_size is not None) else 1
    grad_accum_steps = max(1, int(getattr(args, 'grad_accum_steps', 1)))
    effective_batch_size = args.batch_size * world_size * grad_accum_steps
    print(f"   Effective batch size: {effective_batch_size} (= per_gpu {args.batch_size} × world_size {world_size} × accum {grad_accum_steps})")
    print(f"   Optimizer: {args.optimizer.upper()}, LR: {args.learning_rate}, Scheduler: Cosine (Linear Warmup + Cosine Annealing)")
    print(f"   Masking: Random masking")

    # System info
    print(f"💻 System: {device}")
    if args.ddp:
        print(f"   DDP: Rank {args.rank}/{args.world_size}")
    print(f"   Grad Accumulation Steps: {grad_accum_steps}")
    if args.mixed_precision:
        print(f"   Mixed Precision: Enabled")

    # I/O info
    if not args.no_save:
        print(f"💾 Output: {args.save_dir}")
        print(f"   Checkpoints: Every {args.checkpoint_freq} epochs")
    print("=" * 80)

def main():
    """Main CLI entry point for CANDI training."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Load configuration file if specified
    if args.config:
        try:
            config = load_config_file(args.config)
            # Update args with config values (CLI args take precedence)
            for key, value in config.items():
                key_attr = key.replace('-', '_')
                if not hasattr(args, key_attr) or getattr(args, key_attr) == parser.get_default(key_attr):
                    setattr(args, key_attr, value)
        except Exception as e:
            print(f"❌ Error loading configuration file: {e}")
            return 1
    
    # Save configuration if requested
    if args.save_config:
        config_dict = {k.replace('_', '-'): v for k, v in vars(args).items() if v is not None}
        try:
            save_config_file(config_dict, args.save_config)
            print(f"✅ Configuration saved to: {args.save_config}")
            return 0
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")
            return 1
    
    # Handle GPU check option
    if args.check_gpus:
        check_gpu_availability()
        return 0
    
    # Validate arguments
    try:
        args = validate_arguments(args)
    except ValueError as e:
        print(f"❌ {e}")
        return 1
        
    # Handle mixed precision logic
    if args.no_mixed_precision:
        args.mixed_precision = False

    # CUDA launch blocking is useful for debugging, but expensive in normal training.
    if args.debug:
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    device = setup_device(args)
    
    # Initialize DDP if requested
    if args.ddp:
        try:
            rank, world_size, local_rank = init_distributed()
            if rank is not None:
                args.rank, args.world_size = rank, world_size
                atexit.register(cleanup_distributed)
            else:
                print("❌ Failed to initialize DDP: Environment variables not set properly")
                print("Make sure to run with torchrun: torchrun --nproc_per_node=N train.py ...")
                return 1
        except Exception as e:
            print(f"❌ Failed to initialize DDP: {e}")
            check_gpu_availability()
            print("Falling back to single-GPU training...")
            args.ddp = False
            args.rank = None
            args.world_size = None
    
    # try:
    # Create dataset parameters
    dataset_type = "eic" if args.eic else "merged"
    base_path = args.data_path
    if not base_path.endswith('/'):
        base_path += '/'
    
    if args.eic:
        data_path = base_path + "DATA_CANDI_EIC/"
    else:
        data_path = base_path + "DATA_CANDI_MERGED/"

    prepared_data_path = None
    if args.data_backend in ("zarr", "h5"):
        if not args.eic:
            raise ValueError(f"The {args.data_backend} backend is currently implemented only for the EIC dataset.")
        if args.prepared_data_path is not None:
            prepared_data_path = args.prepared_data_path
        else:
            if args.data_backend == "zarr":
                if get_prepared_eic_path is None:
                    raise ImportError("Unable to resolve the default prepared EIC Zarr path because `data_zarr.py` is unavailable.")
                prepared_data_path = get_prepared_eic_path(data_path)
            else:
                if get_prepared_eic_h5_path is None:
                    raise ImportError("Unable to resolve the default prepared EIC HDF5 path because `data_h5.py` is unavailable.")
                prepared_data_path = get_prepared_eic_h5_path(data_path)
        if not str(prepared_data_path).endswith('/'):
            prepared_data_path = str(prepared_data_path) + '/'
    active_data_path = prepared_data_path if args.data_backend in ("zarr", "h5") else data_path
    
    dataset_params = {
        'base_path': active_data_path,
        'dataset_type': dataset_type,
        'm': args.num_loci,
        'context_length': args.context_length * 25, 
        'split': 'train',
        'loci_gen_strategy': args.loci_gen,
        'ccre_fraction': args.ccre_fraction,
        'dsf_list': args.dsf_list,
        'DNA': True,
        'must_have_chr_access': args.must_have_chr_access,
        'bios_min_exp_avail_threshold': args.min_avail,
        'shuffle_bios': True,
        'balanced_bios_order': args.balanced_bios_order,
        'fill_prompt_mode': args.fill_prompt_mode,
        'signal_transform': args.signal_transform,
        'enable_per_assay_dsf_sampling': args.enable_per_assay_dsf_sampling,
        'per_assay_dsf_sampling_mode': args.per_assay_dsf_sampling_mode,
        'seed': args.seed,
        'data_backend': args.data_backend,
    }
    
    # Create training parameters
    training_params = {
        'optimizer': args.optimizer,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'beta1': args.beta1,
        'beta2': args.beta2,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'grad_accum_steps': args.grad_accum_steps,
        'dataloader_workers': args.dataloader_workers,
        'inner_epochs': args.inner_epochs,
        'enable_validation': not args.disable_validation,  # Enabled by default unless --disable-validation
        'val_freq': args.val_freq,
        'enable_supertrack_train_monitor': args.enable_supertrack_train_monitor,
        'supertrack_train_monitor_every': args.supertrack_train_monitor_every,
        'supertrack_train_monitor_max_batch': args.supertrack_train_monitor_max_batch,
        'wandb_log_every': args.wandb_log_every,
        'use_mixed_precision': args.mixed_precision,
        'specific_ema_alpha': args.specific_ema_alpha,
        'progress_dir': args.progress_dir,
        'debug': args.debug,
        'DNA': True,
        'no_save': args.no_save,
        'count_weight': args.count_weight,
        'pval_weight': args.pval_weight,
        'peak_weight': args.peak_weight,
        'obs_weight': args.obs_weight,
        'imp_weight': args.imp_weight,
        'enable_assay_ema_balance': args.enable_assay_ema_balance,
        'enable_hier_reduction': args.enable_hier_reduction,
        'assay_ema_decay': args.assay_ema_decay,
        'assay_ema_eps': args.assay_ema_eps,
        'assay_ema_warmup_steps': args.assay_ema_warmup_steps,
        'assay_ema_weight_min': args.assay_ema_weight_min,
        'assay_ema_weight_max': args.assay_ema_weight_max,
        'enable_fg_bg_balance': args.enable_fg_bg_balance,
        'fg_weight': args.fg_weight,
        'fg_min_fraction': args.fg_min_fraction,
        'enable_uncertainty_weighting': args.enable_uncertainty_weighting,
        'uncertainty_warmup_steps': args.uncertainty_warmup_steps,
        'uncertainty_init_logvar': args.uncertainty_init_logvar,
        'enable_count_rstable_objective': args.enable_count_rstable_objective,
        'count_rstable_eps': args.count_rstable_eps,
        'count_rstable_ema_decay': args.count_rstable_ema_decay,
        'count_rstable_warmup_steps': args.count_rstable_warmup_steps,
        'count_rstable_denom_min': args.count_rstable_denom_min,
        'count_rstable_r_max': args.count_rstable_r_max,
        'count_rstable_dispersion_min': args.count_rstable_dispersion_min,
        'count_rstable_dispersion_max': args.count_rstable_dispersion_max,
        'p_full_loci': args.p_full_loci,
        'p_full_assay': args.p_full_assay,
        'p_chunks': args.p_chunks,
        'mask_fraction': args.mask_fraction,
        'chunk_size': args.chunk_size,
        'reverse_complement_prob': args.reverse_complement_prob,
        'clip_mode': args.clip_mode,
        'clip_value': args.clip_value,
        'dist_type': args.dist_type,
        'signal_transform': args.signal_transform,
        'enable_latent_kl': args.enable_latent_kl,
        'latent_kl_weight': args.latent_kl_weight,
        'latent_kl_warmup_steps': args.latent_kl_warmup_steps,
        'latent_std_min': args.latent_std_min,
        'latent_std_max': args.latent_std_max,
        'latent_reparam_mode': args.latent_reparam_mode,
        'latent_sample_train_only': args.latent_sample_train_only,
        'latent_deterministic_warmup_steps': args.latent_deterministic_warmup_steps,
    }
    
    # Create temporary dataset to get signal_dim and metadata information
    dataset_cls = resolve_dataset_class(dataset_params)
    temp_dataset = dataset_cls(**dataset_params)
    signal_dim = len(temp_dataset.aliases['experiment_aliases'])
    
    # Get metadata information from the dataset
    # Use num_assays instead of num_sequencing_platforms per issue_supertrack.md ToDo 1
    num_assays = temp_dataset.num_assays
    num_runtypes = 2  # Base run types: 0=single, 1=paired. Missing/cloze handled in MetadataEncoder.
    
    # Store signal_dim and metadata info for later use in training
    dataset_params['signal_dim'] = signal_dim
    dataset_params['num_assays'] = num_assays
    dataset_params['num_runtypes'] = num_runtypes
    
    # Create model with metadata information
    model = create_model_from_args(args, signal_dim, num_assays, num_runtypes)
    
    # Load checkpoint if specified
    if args.checkpoint:
        if Path(args.checkpoint).exists():
            print(f"📂 Loading checkpoint: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            if not bool(getattr(args, "enable_latent_kl", False)):
                model.load_state_dict(checkpoint)
            else:
                allowed_suffixes = {
                    "latent_mu_head.weight",
                    "latent_mu_head.bias",
                    "latent_logvar_head.weight",
                    "latent_logvar_head.bias",
                }
                incompatible = model.load_state_dict(checkpoint, strict=False)
                missing = list(getattr(incompatible, "missing_keys", []))
                unexpected = list(getattr(incompatible, "unexpected_keys", []))
                bad_missing = [k for k in missing if not any(k.endswith(s) for s in allowed_suffixes)]
                bad_unexpected = [k for k in unexpected if not any(k.endswith(s) for s in allowed_suffixes)]
                if bad_missing or bad_unexpected:
                    raise RuntimeError(
                        "Checkpoint/model mismatch beyond latent KL heads in train.py checkpoint load. "
                        f"missing={bad_missing}, unexpected={bad_unexpected}"
                    )
                if missing or unexpected:
                    print(f"[latent_kl_compat] train.py allowed missing={missing}, allowed unexpected={unexpected}")
        else:
            print(f"⚠️ Checkpoint not found: {args.checkpoint}") 
    
    # Create trainer
    trainer = CANDI_TRAINER(
        model=model,
        dataset_params=dataset_params,
        training_params=training_params,
        device=device,
        rank=args.rank if args.ddp else None,
        world_size=args.world_size if args.ddp else None
    )
    
    # Generate timestamp once and use it consistently for model naming
    # This ensures all files (progress CSV, checkpoints, final model) go to the same directory
    training_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = generate_model_name(args, training_timestamp)
    print(f"🏷️  Model name: {model_name}")
    
    # Create model directory and save config at start of training
    if not args.no_save and (not args.ddp or args.rank == 0):
        model_dir = Path(args.save_dir) / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize W&B
        try:
            os.environ.setdefault("WANDB_DISABLE_GIT", "true")
            os.environ.setdefault("WANDB_DISABLE_CODE", "true")
            os.environ.setdefault("WANDB_HTTP_TIMEOUT", "300")
            import wandb
            wandb_run_id = os.environ.get("WANDB_RUN_ID", model_name)
            os.environ["WANDB_RUN_ID"] = wandb_run_id
            # Only initialize W&B on the main process
            wandb_run = wandb.init(
                project="CANDI",
                name=model_name,
                id=wandb_run_id,
                resume="allow",
                dir=str(model_dir / "wandb"),
                config=vars(args)
            )
            _register_wandb_run(wandb_run)
            _install_wandb_signal_handlers()
            
            # Save W&B URL to file
            wandb_url_file = model_dir / "wandb_url.txt"
            with open(wandb_url_file, "w") as f:
                f.write(f"W&B Run URL: {wandb_run.get_url()}\n")
                f.write(f"Project URL: {wandb_run.get_project_url()}\n")
                f.write(f"Run ID: {wandb_run.id}\n")
            print(f"W&B URL saved to: {wandb_url_file}")
            print(f"W&B run configured with WANDB_RUN_ID={wandb_run_id} and resume='allow'")
            
        except ImportError:
            print("Warning: wandb not installed. Skipping W&B initialization.")
        
        # Save config at start of training
        config_dict = {k.replace('_', '-'): v for k, v in vars(args).items() if v is not None}
        config_dict['model_parameters'] = sum(p.numel() for p in model.parameters())
        config_dict['signal_dim'] = signal_dim
        config_dict['num_assays'] = num_assays
        config_dict['num_runtypes'] = num_runtypes
        config_dict['decoder_type'] = getattr(args, 'decoder_type', 'fixed')
        config_dict['moe_experts'] = int(getattr(args, 'moe_experts', 4))
        config_dict['nq_chunk_multiplier'] = int(getattr(args, 'nq_chunk_multiplier', 1))
        config_dict['condconv_k'] = int(getattr(args, 'condconv_k', 3))
        config_dict['condconv_routing'] = str(getattr(args, 'condconv_routing', 'hybrid'))
        config_dict['condconv_gate_activation'] = str(getattr(args, 'condconv_gate_activation', 'sigmoid'))
        
        config_path = model_dir / f"{model_name}_config.json"
        save_config_file(config_dict, config_path)
        print(f"📝 Configuration saved to: {config_path}")
        
        # Update trainer's progress_dir to use the model directory
        trainer.progress_dir = str(model_dir)
        trainer.progress_file = None  # Reset progress file to use new directory
        trainer.validation_progress_file = None  # Reset validation progress file to use new directory
        trainer.model_name = model_name  # Set model name for checkpoint saving
    
    # Start training
    start_time = time.time()
    
    # Print training start
    print_training_summary(args, model, device)
    
    trained_model = trainer.train()
    
    end_time = time.time()
    training_duration = end_time - start_time
    
    # Print training completion
    hours, remainder = divmod(training_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTraining Complete!")
    print(f"Duration: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    print("=" * 80)
    
    # Debug information for model saving
    print(f"🔍 Model saving debug info:")
    print(f"   args.no_save: {args.no_save}")
    print(f"   args.ddp: {args.ddp}")
    print(f"   args.rank: {args.rank}")
    print(f"   Condition result: {not args.no_save and (not args.ddp or args.rank == 0)}")
    
    # Save model if requested (with fallback for safety)
    should_save = not args.no_save and (not args.ddp or args.rank == 0)
    print(f"💾 Model saving decision: {should_save}")
    
    if should_save:
        model_dir = Path(args.save_dir) / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / f"{model_name}.pt"
        
        print(f"💾 Saving trained model to: {model_path}")
        try:
            # Handle DDP model unwrapping
            model_to_save = trained_model.module if hasattr(trained_model, 'module') else trained_model
            torch.save(model_to_save.state_dict(), model_path)
            print(f"✅ Model successfully saved to: {model_path}")
            
            # Verify the file was actually created
            if model_path.exists():
                file_size = model_path.stat().st_size
                print(f"✅ Model file verified: {file_size:,} bytes")
            else:
                print(f"❌ Model file was not created!")
                return 1
                
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            # Try to save to a fallback location
            fallback_path = Path(args.save_dir) / f"fallback_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            try:
                model_to_save = trained_model.module if hasattr(trained_model, 'module') else trained_model
                torch.save(model_to_save.state_dict(), fallback_path)
                print(f"🆘 Fallback model saved to: {fallback_path}")
            except Exception as e2:
                print(f"❌ Fallback save also failed: {e2}")
                return 1
        
        # Update config with training duration
        config_path = model_dir / f"{model_name}_config.json"
        try:
            if config_path.exists():
                # Load existing config and update it
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config_dict['training_duration'] = training_duration
                config_dict['model_parameters'] = sum(p.numel() for p in model_to_save.parameters())
                save_config_file(config_dict, config_path)
                print(f"✅ Config updated with training duration: {config_path}")
            else:
                # Create new config if it doesn't exist
                config_dict = {k.replace('_', '-'): v for k, v in vars(args).items() if v is not None}
                config_dict['training_duration'] = training_duration
                config_dict['model_parameters'] = sum(p.numel() for p in model_to_save.parameters())
                save_config_file(config_dict, config_path)
                print(f"✅ Config created: {config_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not update config file: {e}")
    else:
        if args.no_save:
            print("📝 Model saving disabled (--no-save flag)")
        elif args.ddp and args.rank != 0:
            print(f"📝 Skipping model save on non-main process (rank {args.rank})")
        else:
            print("📝 Model saving skipped for unknown reason")
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎯 Training Session Complete")
    print("=" * 80)

    _safe_wandb_finish(exit_code=0)

    if args.ddp:
        cleanup_distributed()
    
    return 0
        
    # except KeyboardInterrupt:
    #     print("\n⚠️ Training interrupted by user")
    #     return 130
    # except Exception as e:
    #     print(f"❌ Training failed: {e}")
    #     if args.debug:
    #         import traceback
    #         traceback.print_exc()
    #     return 1
    # finally:
    #     # Cleanup DDP
    #     if args.ddp:
    #         cleanup_distributed()

if __name__ == "__main__":
    sys.exit(main())