"""
Pop2Piano Dataset - Complete Implementation
Handles loading audio, MIDI tokens, and creating training batches.
"""

import os
import glob
import torch
import numpy as np
import librosa
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from midi_tokenizer import MidiTokenizer, EOS, PAD


class Pop2PianoDataset(Dataset):
    """
    Dataset for Pop2Piano Training.
    
    Loads preprocessed data structure:
    - *.pitchshift.wav or *.wav : Audio file
    - *.notes.npy : MIDI notes (onset_idx, offset_idx, pitch, velocity)
    - *.beatstep.npy : Beat timestamps
    - *.maqam.txt : Optional; maqam token name (e.g. hijaz, western) for maqam-aware training
    """
    
    def __init__(self, data_dir, config, tokenizer, split='train', 
                 augment=True, val_ratio=0.1):
        """
        Args:
            data_dir: Directory containing preprocessed tracks
            config: OmegaConf config object
            tokenizer: MidiTokenizer instance
            split: 'train' or 'val'
            augment: Whether to apply data augmentation
            val_ratio: Ratio of data to use for validation
        """
        self.data_dir = data_dir
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        self.augment = augment and (split == 'train')
        
        # Audio settings
        self.sample_rate = config.dataset.sample_rate
        self.n_bars = config.dataset.n_bars
        self.target_length = config.dataset.target_length
        
        # Composer tokens
        self.composer_to_token = config.composer_to_feature_token
        self.composers = list(self.composer_to_token.keys())
        
        # Find all valid tracks
        self.samples = self._scan_tracks()
        
        # Train/Val split
        random.seed(config.training.seed)
        random.shuffle(self.samples)
        split_idx = int(len(self.samples) * (1 - val_ratio))
        
        if split == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
        
        print(f"✅ Pop2PianoDataset [{split}]: {len(self.samples)} samples")
    
    def _scan_tracks(self):
        """Scan data directory for valid preprocessed tracks."""
        samples = []
        track_dirs = glob.glob(os.path.join(self.data_dir, "*"))
        
        print(f"🔍 Scanning {self.data_dir} for processed tracks...")
        
        for track_path in track_dirs:
            if not os.path.isdir(track_path):
                continue
            
            # Find required files
            beatstep_files = glob.glob(os.path.join(track_path, "*.beatstep.npy"))
            notes_files = glob.glob(os.path.join(track_path, "*.notes.npy"))
            
            # Audio file (prefer pitchshifted)
            audio_files = glob.glob(os.path.join(track_path, "*.pitchshift.wav"))
            if not audio_files:
                audio_files = glob.glob(os.path.join(track_path, "*.wav"))
            
            # Validate all required files exist
            if beatstep_files and notes_files and audio_files:
                notes_path = notes_files[0]
                # Maqam file has same base as notes: EHl_eQhgefw.notes.npy -> EHl_eQhgefw.maqam.txt
                maqam_path = notes_path.replace(".notes.npy", ".maqam.txt")
                if not os.path.isfile(maqam_path):
                    maqam_path = None
                samples.append({
                    'track_dir': track_path,
                    'track_id': os.path.basename(track_path),
                    'beatstep_path': beatstep_files[0],
                    'notes_path': notes_path,
                    'audio_path': audio_files[0],
                    'maqam_path': maqam_path,
                })
        
        print(f"📁 Found {len(samples)} valid tracks")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Load a single sample with audio and MIDI tokens."""
        sample_info = self.samples[idx]
        
        try:
            # Load preprocessed data
            beatstep = np.load(sample_info['beatstep_path'])
            notes = np.load(sample_info['notes_path'])
            
            # Load audio
            audio, _ = librosa.load(
                sample_info['audio_path'], 
                sr=self.sample_rate,
                mono=True
            )
            
            # Select random segment (n_bars)
            n_steps = self.n_bars * 4  # 4 beats per bar
            max_start = max(0, len(beatstep) - n_steps - 1)
            
            if max_start > 0:
                start_beat_idx = random.randint(0, max_start)
            else:
                start_beat_idx = 0
            
            end_beat_idx = min(start_beat_idx + n_steps, len(beatstep) - 1)
            
            # Extract audio segment
            start_time = beatstep[start_beat_idx]
            end_time = beatstep[end_beat_idx]
            
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            audio_segment = audio[start_sample:end_sample]
            
            # Apply data augmentation
            if self.augment:
                audio_segment = self._augment_audio(audio_segment)
            
            # Extract notes for this segment
            segment_notes, _ = self.tokenizer.split_notes(
                notes, beatstep, start_time, end_time
            )
            
            # Convert notes to tokens
            if len(segment_notes) > 0:
                # Shift note indices to be relative to segment start
                segment_notes = segment_notes.copy()
                segment_notes[:, 0] -= start_beat_idx  # onset
                segment_notes[:, 1] -= start_beat_idx  # offset
                segment_notes[:, 0] = np.clip(segment_notes[:, 0], 0, n_steps)
                segment_notes[:, 1] = np.clip(segment_notes[:, 1], 0, n_steps)
            
            # Composer/maqam token: use pre-computed maqam if available, else western or random
            composer_value = None
            if sample_info.get('maqam_path'):
                try:
                    with open(sample_info['maqam_path'], 'r', encoding='utf-8') as f:
                        token_name = f.read().strip().lower()
                    composer_value = self.composer_to_token.get(token_name)
                except Exception:
                    pass
            if composer_value is None:
                composer_value = self.composer_to_token.get("western")
            if composer_value is None:
                composer = random.choice(self.composers)
                composer_value = self.composer_to_token[composer]
            
            # Convert to relative tokens with composer
            tokens = self.tokenizer.notes_to_relative_tokens(
                segment_notes if len(segment_notes) > 0 else np.array([]).reshape(0, 4),
                offset_idx=0,
                add_eos=True,
                add_composer=True,
                composer_value=composer_value
            )
            
            # Truncate if too long
            if len(tokens) > self.target_length:
                tokens = tokens[:self.target_length]
            
            return {
                'audio': torch.from_numpy(audio_segment).float(),
                'labels': torch.from_numpy(tokens).long(),
                'composer': torch.tensor(composer_value).long(),
                'track_id': sample_info['track_id'],
                'beatstep': torch.from_numpy(
                    beatstep[start_beat_idx:end_beat_idx+1] - start_time
                ).float(),
            }
            
        except Exception as e:
            print(f"⚠️ Error loading {sample_info['track_id']}: {e}")
            # Return a different sample on error
            new_idx = random.randint(0, len(self) - 1)
            if new_idx == idx:
                new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)
    
    def _augment_audio(self, audio):
        """Apply data augmentation to audio."""
        # Pitch shift (-2 to +2 semitones)
        if random.random() < 0.3:
            n_steps = random.uniform(-2, 2)
            audio = librosa.effects.pitch_shift(
                audio, sr=self.sample_rate, n_steps=n_steps
            )
        
        # Time stretch (0.9 to 1.1)
        if random.random() < 0.3:
            rate = random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=rate)
        
        # Add small noise
        if random.random() < 0.2:
            noise = np.random.randn(len(audio)) * 0.005
            audio = audio + noise
        
        return audio.astype(np.float32)


class Pop2PianoCollator:
    """
    Collator for creating padded batches.
    """
    
    def __init__(self, config, pad_token_id=PAD):
        self.config = config
        self.pad_token_id = pad_token_id
        self.sample_rate = config.dataset.sample_rate
    
    def __call__(self, batch):
        """
        Collate batch of samples into padded tensors.
        
        Args:
            batch: List of dicts from dataset __getitem__
            
        Returns:
            Dict with padded tensors
        """
        # Separate components
        audios = [item['audio'] for item in batch]
        labels = [item['labels'] for item in batch]
        composers = torch.stack([item['composer'] for item in batch])
        track_ids = [item['track_id'] for item in batch]
        
        # Pad audio to max length in batch
        max_audio_len = max(a.shape[0] for a in audios)
        padded_audios = torch.zeros(len(audios), max_audio_len)
        audio_lengths = torch.zeros(len(audios), dtype=torch.long)
        
        for i, audio in enumerate(audios):
            padded_audios[i, :audio.shape[0]] = audio
            audio_lengths[i] = audio.shape[0]
        
        # Pad labels to max length in batch
        padded_labels = pad_sequence(
            labels, 
            batch_first=True, 
            padding_value=self.pad_token_id
        )
        
        # Create attention mask for labels (1 for real tokens, 0 for padding)
        label_attention_mask = (padded_labels != self.pad_token_id).long()
        
        return {
            'audio': padded_audios,
            'audio_lengths': audio_lengths,
            'labels': padded_labels,
            'label_attention_mask': label_attention_mask,
            'composer': composers,
            'track_ids': track_ids,
        }


def create_dataloaders(data_dir, config, tokenizer, batch_size=None, num_workers=None):
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Directory containing preprocessed data
        config: OmegaConf config
        tokenizer: MidiTokenizer instance
        batch_size: Override config batch size
        num_workers: Override config num_workers
        
    Returns:
        train_loader, val_loader
    """
    batch_size = batch_size or config.training.batch_size
    num_workers = num_workers or config.training.num_workers
    
    # Create datasets
    train_dataset = Pop2PianoDataset(
        data_dir=data_dir,
        config=config,
        tokenizer=tokenizer,
        split='train',
        augment=True,
    )
    
    val_dataset = Pop2PianoDataset(
        data_dir=data_dir,
        config=config,
        tokenizer=tokenizer,
        split='val',
        augment=False,
    )
    
    # Create collator
    collator = Pop2PianoCollator(config)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    
    return train_loader, val_loader
