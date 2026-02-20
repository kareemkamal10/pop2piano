"""
Pop2Piano Training Script - Complete Implementation

Features:
- Mixed Precision (FP16) for faster training
- Gradient Checkpointing for memory efficiency  
- Resume from checkpoint
- TensorBoard logging
- Early stopping
- Multi-GPU support
"""

import os
import sys
import argparse
from datetime import datetime

import torch
try:
    import lightning as pl
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
    from lightning.pytorch.loggers import TensorBoardLogger
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger

from omegaconf import OmegaConf

# Import project modules
from transformer_wrapper import TransformerWrapper
from midi_tokenizer import MidiTokenizer
from dataset import Pop2PianoDataset, Pop2PianoCollator, create_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='Train Pop2Piano Model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='output_dir', help='Directory with preprocessed data')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--epochs', type=int, default=None, help='Override max epochs')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--precision', type=str, default='16-mixed', choices=['32', '16-mixed', 'bf16-mixed'], 
                        help='Training precision')
    parser.add_argument('--debug', action='store_true', help='Debug mode with small dataset')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # ─────────────────────────────────────────────────────────────
    # 1. Load Configuration
    # ─────────────────────────────────────────────────────────────
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    config = OmegaConf.load(args.config)
    print(f"✅ Configuration loaded from {args.config}")
    
    # Apply command line overrides
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.epochs:
        config.training.max_epochs = args.epochs
    if args.lr:
        config.training.lr = args.lr
    
    # Set seed for reproducibility
    pl.seed_everything(config.training.seed, workers=True)
    
    # ─────────────────────────────────────────────────────────────
    # 2. Setup Data
    # ─────────────────────────────────────────────────────────────
    data_dir = args.data_dir
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"⚠️  '{data_dir}' was missing. Created it.")
        print("💡 Run download.py and preprocess scripts first!")
        return
    
    # Create tokenizer
    tokenizer = MidiTokenizer(config.tokenizer)
    
    # Create dataloaders
    print(f"\n📦 Loading data from: {data_dir}")
    
    # Adjust workers for Windows
    num_workers = 0 if sys.platform == 'win32' else config.training.num_workers
    
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        config=config,
        tokenizer=tokenizer,
        batch_size=config.training.batch_size,
        num_workers=num_workers,
    )
    
    if len(train_loader.dataset) == 0:
        print("❌ No training data found!")
        print("💡 Make sure to download and preprocess data first.")
        return
    
    print(f"📊 Training samples: {len(train_loader.dataset)}")
    print(f"📊 Validation samples: {len(val_loader.dataset)}")
    print(f"📊 Batch size: {config.training.batch_size}")
    print(f"📊 Training batches: {len(train_loader)}")
    
    # ─────────────────────────────────────────────────────────────
    # 3. Setup Model
    # ─────────────────────────────────────────────────────────────
    print("\n🧠 Initializing Model...")
    
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Resuming from checkpoint: {args.resume}")
        model = TransformerWrapper.load_from_checkpoint(args.resume, config=config)
    else:
        model = TransformerWrapper(config)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,}")
    
    # ─────────────────────────────────────────────────────────────
    # 4. Setup Callbacks
    # ─────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        # Save best models
        ModelCheckpoint(
            dirpath=f"checkpoints/{timestamp}",
            filename="pop2piano-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            save_last=True,  # Always save last checkpoint for resume
        ),
        
        # Early stopping
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
            verbose=True,
        ),
        
        # Learning rate monitor
        LearningRateMonitor(logging_interval='step'),
    ]
    
    # ─────────────────────────────────────────────────────────────
    # 5. Setup Logger
    # ─────────────────────────────────────────────────────────────
    logger = TensorBoardLogger(
        save_dir="logs",
        name="pop2piano",
        version=timestamp,
    )
    
    # ─────────────────────────────────────────────────────────────
    # 6. Setup Trainer
    # ─────────────────────────────────────────────────────────────
    print("\n⚙️  Setting up Trainer...")
    
    # Determine accelerator and devices
    if torch.cuda.is_available():
        accelerator = "cuda"
        devices = min(config.training.num_gpu, torch.cuda.device_count())
        print(f"🎮 Using {devices} GPU(s)")
    else:
        accelerator = "cpu"
        devices = 1
        print("💻 Using CPU (training will be slow)")
    
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=args.precision,  # Mixed precision for speed
        callbacks=callbacks,
        logger=logger,
        
        # Gradient settings
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        
        # Validation
        check_val_every_n_epoch=config.training.check_val_every_n_epoch,
        
        # Performance
        enable_progress_bar=True,
        enable_model_summary=True,
        
        # Debug mode
        fast_dev_run=args.debug,
        
        # Deterministic for reproducibility
        deterministic=True,
    )
    
    # ─────────────────────────────────────────────────────────────
    # 7. Start Training
    # ─────────────────────────────────────────────────────────────
    print("\n🚀 Starting Training...")
    print(f"📊 Precision: {args.precision}")
    print(f"📊 Max epochs: {config.training.max_epochs}")
    print(f"📊 Learning rate: {config.training.lr}")
    print(f"📊 Optimizer: {config.training.optimizer}")
    print("-" * 50)
    
    try:
        trainer.fit(
            model, 
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=args.resume if args.resume and os.path.exists(args.resume) else None,
        )
        
        print("\n✅ Training completed!")
        print(f"📁 Checkpoints saved to: checkpoints/{timestamp}")
        print(f"📁 Logs saved to: logs/pop2piano/{timestamp}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print("💡 You can resume training using --resume <checkpoint_path>")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
