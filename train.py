import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

# Import our project modules
from transformer_wrapper import TransformerWrapper
from dataset import Pop2PianoDataset, Pop2PianoCollator

def main():
    # 1. Load Configuration
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError("config.yaml not found!")
    
    config = OmegaConf.load(config_path)
    print("✅ Configuration loaded.")

    # 2. Setup Data
    # Point this to your OUTPUT_DIR containing preprocessed data
    DATA_DIR = "output_dir" 
    
    # Create output_dir if it doesn't exist just to prevent crash
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print("⚠️ 'output_dir' was missing. Created it. Please put data there.")

    train_dataset = Pop2PianoDataset(DATA_DIR, config, split='train')
    
    # Check if we have data
    if len(train_dataset) == 0:
        print("❌ No data found in 'output_dir'. Cannot train.")
        print("💡 Run download.py and preprocess scripts first!")
        return

    train_loader = DataLoader(
        train_dataset, 
        batch_size=2, # Small batch for local testing 
        shuffle=True, 
        num_workers=0, # Windows often needs 0 workers
        # collate_fn=Pop2PianoCollator(None) # TODO: Initialize with tokenizer
    )

    # 3. Setup Model
    print("🧠 Initializing Model (TransformerWrapper)...")
    model = TransformerWrapper(config)
    
    # Add a dummy training_step to the model instance dynamically
    # (Since we didn't edit the original file directly to keep it clean for now)
    # Ideally, you'd add this method inside transformer_wrapper.py
    def training_step(self, batch, batch_idx):
        # ---------------------------------------------------------
        # THIS IS A PLACEHOLDER LOGIC
        # A real training step needs to:
        # 1. Take audio/spectrogram input
        # 2. Take MIDI token labels
        # 3. Pass to self.transformer(input_values=..., labels=...)
        # 4. Return loss
        # ---------------------------------------------------------
        
        # Simulating a forward pass for testing
        # T5 expects 'input_ids' or 'inputs_embeds' and 'labels'
        
        # dummy_loss = torch.tensor(0.5, requires_grad=True).to(self.device)
        # self.log("train_loss", dummy_loss)
        # return dummy_loss
        raise NotImplementedError("You need to implement the actual data mapping in `training_step` inside transformer_wrapper.py")

    # Patching the method (Python allows this!)
    # TransformerWrapper.training_step = training_step

    # 4. Setup Trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="pop2piano-{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,
        monitor="train_loss",
    )

    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="auto", # Uses GPU if available
        devices=1,
        callbacks=[checkpoint_callback],
        logger=False # Disable logger for simple local test
    )

    # 5. Start Training
    print("🚀 Starting Training Loop...")
    try:
        trainer.fit(model, train_loader)
    except NotImplementedError as e:
        print(f"\n🛑 STOPPED ESPC: {e}")
        print("To fix this, we need to modify transformer_wrapper.py to handle the batch data correctly.")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")

if __name__ == "__main__":
    main()
