import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

class Pop2PianoDataset(Dataset):
    def __init__(self, data_dir, config, split='train'):
        """
        Custom Dataset for Pop2Piano.
        Assumes data is preprocessed into the structure described in preprocess/README.md
        """
        self.data_dir = data_dir
        self.config = config
        self.split = split
        
        # Find all beat step files which serve as anchors for each sample
        self.track_dirs = glob.glob(os.path.join(data_dir, "*"))
        self.valid_tracks = []
        
        print(f"🔍 Scanning {data_dir} for processed tracks...")
        for track_path in self.track_dirs:
            if not os.path.isdir(track_path):
                continue
            
            # Basic validation: Check if essential files exist
            # Based on preprocess/README.md structure
            vid_id = os.path.basename(track_path)
            # Just checking for one essential file to confirm valid folder
            if glob.glob(os.path.join(track_path, "*.beatstep.npy")):
                self.valid_tracks.append(track_path)

        print(f"✅ Found {len(self.valid_tracks)} valid tracks for {split}.")

    def __len__(self):
        return len(self.valid_tracks)

    def __getitem__(self, idx):
        track_dir = self.valid_tracks[idx]
        vid_id = os.path.basename(track_dir)
        
        # Load preprocessed files
        # Note: In a real scenario, you iterate over segments. 
        # Here we simplify by picking a random segment or the whole file if small.
        # But Pop2Piano training usually cuts based on bars.
        
        try:
            # 1. Load Embeddings/Input Audio (MelSpectrogram is calculated in model, so we load audio)
            # Actually TransformerWrapper expects 'audio' and 'beatstep' for inference,
            # but for training it likely expects pre-calculated tokens or raw audio to be sliced.
            
            # Let's look at TransformerWrapper.forward signature (which is deprecated in the file but useful hint)
            # It expects input_ids and labels.
            # T5 (Transformer) -> Input: Mel Spectrogram (Audio) | Label: MIDI Tokens
            
            # For simplicity in this 'Mock' implementation to get fit() running:
            # We will load the .npy files.
            
            beatstep_path = glob.glob(os.path.join(track_dir, "*.beatstep.npy"))[0]
            beatstep = np.load(beatstep_path)
            
            # Audio path (pitchshifted or regular)
            # In a full impl, we load audio using librosa, but that's slow in __getitem__
            # We assume pre-computed features or load on fly. 
            # For this MVP, we return dummy tensors matching shapes to ensure dimensions work.
            
            # Real implementation would require loading utils.dsp.LogMelSpectrogram
            # and processing the audio chunk corresponding to the beatstep.
            
            return {
                "track_id": vid_id,
                "beatstep": beatstep,
                # "audio": audio_chunk 
            }
        except Exception as e:
            print(f"Error loading {track_dir}: {e}")
            return self.__getitem__(random.randint(0, len(self)-1))

class Pop2PianoCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, batch):
        # This function merges a list of samples into a mini-batch
        # It needs to pad sequences to the longest in the batch
        return torch.utils.data.dataloader.default_collate(batch)
