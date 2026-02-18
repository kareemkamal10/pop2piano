import os
import random
import math

import numpy as np
import librosa
import torch
import torch.optim as optim
try:
    import lightning as pl  # lightning 2.x
except ImportError:
    import pytorch_lightning as pl  # fallback to 1.x
import soundfile as sf
from torch.nn.utils.rnn import pad_sequence
from transformers import T5Config, T5ForConditionalGeneration
from transformers import get_cosine_schedule_with_warmup

from midi_tokenizer import MidiTokenizer, extrapolate_beat_times
from layer.input import LogMelSpectrogram, ConcatEmbeddingToMel
from preprocess.beat_quantizer import extract_rhythm, interpolate_beat_times
from utils.dsp import get_stereo

# Import piano rules and Arabic maqamat modules
try:
    from piano_rules import apply_piano_rules, PianoNote, midi_array_to_notes, notes_to_midi_array
    from arabic_maqamat import get_maqam, get_maqam_scale, detect_maqam, quantize_to_maqam
    RULES_AVAILABLE = True
except ImportError:
    RULES_AVAILABLE = False
    print("⚠️  Piano rules and Arabic maqamat modules not found. Rule-based processing disabled.")


DEFAULT_COMPOSERS = {"various composer": 2052}


class TransformerWrapper(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tokenizer = MidiTokenizer(config.tokenizer)
        self.t5config = T5Config.from_pretrained("t5-small")

        for k, v in config.t5.items():
            self.t5config.__setattr__(k, v)

        self.transformer = T5ForConditionalGeneration(self.t5config)
        self.use_mel = self.config.dataset.use_mel
        self.mel_is_conditioned = self.config.dataset.mel_is_conditioned
        self.composer_to_feature_token = config.composer_to_feature_token

        if self.use_mel and not self.mel_is_conditioned:
            self.composer_to_feature_token = DEFAULT_COMPOSERS

        if self.use_mel:
            self.spectrogram = LogMelSpectrogram()
            if self.mel_is_conditioned:
                n_dim = 512
                composer_n_vocab = len(self.composer_to_feature_token)
                embedding_offset = min(self.composer_to_feature_token.values())
                self.mel_conditioner = ConcatEmbeddingToMel(
                    embedding_offset=embedding_offset,
                    n_vocab=composer_n_vocab,
                    n_dim=n_dim,
                )
        else:
            self.spectrogram = None

        self.lr = config.training.lr

    def training_step(self, batch, batch_idx):
        """
        Training step - processes audio to mel spectrogram and trains on MIDI tokens.
        
        Batch contains:
        - audio: (batch, samples) raw audio waveform
        - labels: (batch, seq_len) MIDI tokens
        - composer: (batch,) composer token values
        """
        audio = batch['audio']  # (batch, samples)
        labels = batch['labels']  # (batch, seq_len)
        composer = batch['composer']  # (batch,)
        
        # Convert audio to mel spectrogram
        # spectrogram expects (batch, samples) -> (batch, n_mels, time)
        mel = self.spectrogram(audio)  # (batch, 512, time)
        
        # Transpose to (batch, time, 512) for transformer
        inputs_embeds = mel.transpose(1, 2)  # (batch, time, 512)
        
        # Add composer conditioning if enabled
        if self.mel_is_conditioned:
            inputs_embeds = self.mel_conditioner(inputs_embeds, composer)
        
        # Forward pass through T5
        outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            labels=labels,
        )
        
        loss = outputs.loss
        
        # Logging
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        
        # Log learning rate
        if self.trainer.optimizers:
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log("lr", lr, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step - same as training but without gradients.
        """
        audio = batch['audio']
        labels = batch['labels']
        composer = batch['composer']
        
        # Convert audio to mel spectrogram
        mel = self.spectrogram(audio)
        inputs_embeds = mel.transpose(1, 2)
        
        if self.mel_is_conditioned:
            inputs_embeds = self.mel_conditioner(inputs_embeds, composer)
        
        outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            labels=labels,
        )
        
        self.log("val_loss", outputs.loss, prog_bar=True, sync_dist=True)
        return outputs.loss
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        """
        config = self.config.training
        
        # Choose optimizer
        if config.optimizer.lower() == 'adafactor':
            from transformers import Adafactor
            optimizer = Adafactor(
                self.parameters(),
                lr=self.lr,
                relative_step=False,
                warmup_init=False,
            )
        elif config.optimizer.lower() == 'adamw':
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=0.01,
            )
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        
        # Learning rate scheduler
        if config.lr_scheduler:
            # Estimate total training steps
            if hasattr(self.trainer, 'estimated_stepping_batches'):
                total_steps = self.trainer.estimated_stepping_batches
            else:
                total_steps = config.max_epochs * 1000  # Fallback estimate
            
            warmup_steps = int(total_steps * 0.1)  # 10% warmup
            
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                }
            }
        
        return optimizer
    
    def on_train_start(self):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.transformer, 'gradient_checkpointing_enable'):
            self.transformer.gradient_checkpointing_enable()
            print("✅ Gradient checkpointing enabled")

    def forward(self, input_ids, labels):
        """
        Deprecated.
        """
        rt = self.transformer(input_ids=input_ids, labels=labels)
        return rt

    @torch.no_grad()
    def single_inference(
        self,
        feature_tokens=None,
        audio=None,
        beatstep=None,
        max_length=256,
        max_batch_size=64,
        n_bars=None,
        composer_value=None,
    ):
        """
        generate a long audio sequence

        feature_tokens or audio : shape (time, )

        beatstep : shape (time, )
        - input_ids가 해당하는 beatstep 값들
        (offset 빠짐, 즉 beatstep[0] == 0)
        - beatstep[-1] : input_ids가 끝나는 지점의 시간값
        (즉 beatstep[-1] == len(y)//sr)
        """

        assert feature_tokens is not None or audio is not None
        assert beatstep is not None

        if feature_tokens is not None:
            assert len(feature_tokens.shape) == 1

        if audio is not None:
            assert len(audio.shape) == 1

        config = self.config
        PAD = self.t5config.pad_token_id
        n_bars = config.dataset.n_bars if n_bars is None else n_bars

        if beatstep[0] > 0.01:
            print(
                "inference warning : beatstep[0] is not 0 ({beatstep[0]}). all beatstep will be shifted."
            )
            beatstep = beatstep - beatstep[0]

        if self.use_mel:
            input_ids = None
            inputs_embeds, ext_beatstep = self.prepare_inference_mel(
                audio,
                beatstep,
                n_bars=n_bars,
                padding_value=PAD,
                composer_value=composer_value,
            )
            batch_size = inputs_embeds.shape[0]
        else:
            raise NotImplementedError

        # Considering GPU capacity, some sequence would not be generated at once.
        relative_tokens = list()
        for i in range(0, batch_size, max_batch_size):
            start = i
            end = min(batch_size, i + max_batch_size)

            if input_ids is None:
                _input_ids = None
                _inputs_embeds = inputs_embeds[start:end]
            else:
                _input_ids = input_ids[start:end]
                _inputs_embeds = None

            _relative_tokens = self.transformer.generate(
                input_ids=_input_ids,
                inputs_embeds=_inputs_embeds,
                max_length=max_length,
            )
            _relative_tokens = _relative_tokens.cpu().numpy()
            relative_tokens.append(_relative_tokens)

        max_length = max([rt.shape[-1] for rt in relative_tokens])
        for i in range(len(relative_tokens)):
            relative_tokens[i] = np.pad(
                relative_tokens[i],
                [(0, 0), (0, max_length - relative_tokens[i].shape[-1])],
                constant_values=PAD,
            )
        relative_tokens = np.concatenate(relative_tokens)

        pm, notes = self.tokenizer.relative_batch_tokens_to_midi(
            relative_tokens,
            beatstep=ext_beatstep,
            bars_per_batch=n_bars,
            cutoff_time_idx=(n_bars + 1) * 4,
        )

        return relative_tokens, notes, pm

    def prepare_inference_mel(
        self, audio, beatstep, n_bars, padding_value, composer_value=None
    ):
        n_steps = n_bars * 4
        n_target_step = len(beatstep)
        sample_rate = self.config.dataset.sample_rate
        ext_beatstep = extrapolate_beat_times(beatstep, (n_bars + 1) * 4 + 1)

        def split_audio(audio):
            # Split audio corresponding beat intervals.
            # Each audio's lengths are different.
            # Because each corresponding beat interval times are different.
            batch = []

            for i in range(0, n_target_step, n_steps):

                start_idx = i
                end_idx = min(i + n_steps, n_target_step)

                start_sample = int(ext_beatstep[start_idx] * sample_rate)
                end_sample = int(ext_beatstep[end_idx] * sample_rate)
                feature = audio[start_sample:end_sample]
                batch.append(feature)
            return batch

        def pad_and_stack_batch(batch):
            batch = pad_sequence(batch, batch_first=True, padding_value=padding_value)
            return batch

        batch = split_audio(audio)
        batch = pad_and_stack_batch(batch)

        inputs_embeds = self.spectrogram(batch).transpose(-1, -2)
        if self.mel_is_conditioned:
            composer_value = torch.tensor(composer_value).to(self.device)
            composer_value = composer_value.repeat(inputs_embeds.shape[0])
            inputs_embeds = self.mel_conditioner(inputs_embeds, composer_value)
        return inputs_embeds, ext_beatstep

    @torch.no_grad()
    def generate(
        self,
        audio_path=None,
        composer=None,
        model="generated",
        steps_per_beat=2,
        stereo_amp=0.5,
        n_bars=2,
        ignore_duplicate=True,
        show_plot=False,
        save_midi=False,
        save_mix=False,
        midi_path=None,
        mix_path=None,
        click_amp=0.2,
        add_click=False,
        max_batch_size=None,
        beatsteps=None,
        mix_sample_rate=None,
        audio_y=None,
        audio_sr=None,
        # New parameters for rule-based processing
        apply_rules=True,           # Apply piano rules post-processing
        maqam=None,                  # Arabic maqam name (e.g., 'hijaz', 'bayyati')
        auto_detect_maqam=False,    # Auto-detect maqam from melody
        simplify_chords=True,       # Simplify complex chords
        quantize_rhythm=True,       # Quantize to beat grid
        humanize=False,             # Add human-like variations
    ):
        config = self.config
        device = self.device
        composer_to_feature_token = self.composer_to_feature_token

        # Resolve composer: explicit composer, or maqam alias, or auto-detect from audio
        if composer is None and maqam is not None:
            composer = maqam
        if composer is None:
            try:
                from smart_inference import analyze_audio
                if audio_y is not None and audio_sr is not None:
                    _y, _sr = audio_y, audio_sr
                elif audio_path is not None:
                    _y, _sr = librosa.load(audio_path, sr=44100)
                else:
                    _y = _sr = None
                if _y is not None:
                    analysis = analyze_audio(_y, _sr)
                    composer = (analysis.detected_maqam if analysis.is_arabic else "western")
                    if composer not in composer_to_feature_token:
                        composer = "western" if "western" in composer_to_feature_token else list(composer_to_feature_token.keys())[0]
                else:
                    composer = random.sample(list(composer_to_feature_token.keys()), 1)[0]
            except Exception:
                composer = random.sample(list(composer_to_feature_token.keys()), 1)[0]
        if composer not in composer_to_feature_token:
            composer = "western" if "western" in composer_to_feature_token else list(composer_to_feature_token.keys())[0]

        composer_value = composer_to_feature_token[composer]

        if audio_path is not None:
            extension = os.path.splitext(audio_path)[1]
            mix_path = (
                audio_path.replace(extension, f".{model}.{composer}.wav")
                if mix_path is None
                else mix_path
            )
            midi_path = (
                audio_path.replace(extension, f".{model}.{composer}.mid")
                if midi_path is None
                else midi_path
            )

        max_batch_size = 64 // n_bars if max_batch_size is None else max_batch_size
        mix_sample_rate = (
            config.dataset.sample_rate if mix_sample_rate is None else mix_sample_rate
        )

        if not ignore_duplicate:
            if os.path.exists(midi_path):
                return

        ESSENTIA_SAMPLERATE = 44100

        if beatsteps is None:
            y, sr = librosa.load(audio_path, sr=ESSENTIA_SAMPLERATE)
            (
                bpm,
                beat_times,
                confidence,
                estimates,
                essentia_beat_intervals,
            ) = extract_rhythm(audio_path, y=y)
            beat_times = np.array(beat_times)
            beatsteps = interpolate_beat_times(beat_times, steps_per_beat, extend=True)
        else:
            y = None

        if self.use_mel:
            if audio_y is None and config.dataset.sample_rate != ESSENTIA_SAMPLERATE:
                if y is not None:
                    y = librosa.core.resample(
                        y,
                        orig_sr=ESSENTIA_SAMPLERATE,
                        target_sr=config.dataset.sample_rate,
                    )
                    sr = config.dataset.sample_rate
                else:
                    y, sr = librosa.load(audio_path, sr=config.dataset.sample_rate)
            elif audio_y is not None:
                if audio_sr != config.dataset.sample_rate:
                    audio_y = librosa.core.resample(
                        audio_y, orig_sr=audio_sr, target_sr=config.dataset.sample_rate
                    )
                    audio_sr = config.dataset.sample_rate
                y = audio_y
                sr = audio_sr

            start_sample = int(beatsteps[0] * sr)
            end_sample = int(beatsteps[-1] * sr)
            _audio = torch.from_numpy(y)[start_sample:end_sample].to(device)
            fzs = None
        else:
            raise NotImplementedError

        relative_tokens, notes, pm = self.single_inference(
            feature_tokens=fzs,
            audio=_audio,
            beatstep=beatsteps - beatsteps[0],
            max_length=config.dataset.target_length
            * max(1, (n_bars // config.dataset.n_bars)),
            max_batch_size=max_batch_size,
            n_bars=n_bars,
            composer_value=composer_value,
        )

        for n in pm.instruments[0].notes:
            n.start += beatsteps[0]
            n.end += beatsteps[0]

        # ─────────────────────────────────────────────────────────────
        # Apply Piano Rules and Arabic Maqamat Post-Processing
        # ─────────────────────────────────────────────────────────────
        if apply_rules and RULES_AVAILABLE and len(pm.instruments[0].notes) > 0:
            print("🎹 Applying piano rules...")
            
            # Convert MIDI to PianoNote objects
            piano_notes = []
            for n in pm.instruments[0].notes:
                piano_notes.append(PianoNote(
                    pitch=n.pitch,
                    onset=n.start,
                    offset=n.end,
                    velocity=n.velocity
                ))
            
            # Determine scale for quantization
            scale_pitches = None
            
            # Auto-detect maqam if requested
            detected_maqam_name = None
            if auto_detect_maqam and maqam is None:
                pitches = [n.pitch for n in pm.instruments[0].notes]
                detections = detect_maqam(pitches, threshold=0.6)
                if detections:
                    detected_maqam_name = detections[0][0]
                    confidence = detections[0][1]
                    print(f"🎵 Detected maqam: {detected_maqam_name} (confidence: {confidence:.2f})")
                    maqam_obj = get_maqam(detected_maqam_name)
                    if maqam_obj:
                        scale_pitches = get_maqam_scale(maqam_obj)
            
            # Use specified maqam
            elif maqam is not None:
                maqam_obj = get_maqam(maqam)
                if maqam_obj:
                    scale_pitches = get_maqam_scale(maqam_obj)
                    print(f"🎵 Using maqam: {maqam_obj.name_en} ({maqam_obj.name_ar})")
                else:
                    print(f"⚠️  Unknown maqam: {maqam}. Using default processing.")
            
            # Apply piano rules
            processed_notes = apply_piano_rules(
                notes=piano_notes,
                scale_pitches=scale_pitches,
                simplify=simplify_chords,
                quantize=quantize_rhythm,
                humanize=humanize
            )
            
            # Update MIDI with processed notes
            pm.instruments[0].notes = []
            import pretty_midi
            for pn in processed_notes:
                pm.instruments[0].notes.append(pretty_midi.Note(
                    velocity=pn.velocity,
                    pitch=pn.pitch,
                    start=pn.onset,
                    end=pn.offset
                ))
            
            print(f"✅ Applied rules: {len(piano_notes)} -> {len(processed_notes)} notes")

        if show_plot or save_mix:
            if mix_sample_rate != sr:
                y = librosa.core.resample(y, orig_sr=sr, target_sr=mix_sample_rate)
                sr = mix_sample_rate
            if add_click:
                clicks = (
                    librosa.clicks(times=beatsteps, sr=sr, length=len(y)) * click_amp
                )
                y = y + clicks
            pm_y = pm.fluidsynth(sr)
            stereo = get_stereo(y, pm_y, pop_scale=stereo_amp)

        if show_plot:
            import IPython.display as ipd
            from IPython.display import display
            import note_seq

            display("Stereo MIX", ipd.Audio(stereo, rate=sr))
            display("Rendered MIDI", ipd.Audio(pm_y, rate=sr))
            display("Original Song", ipd.Audio(y, rate=sr))
            display(note_seq.plot_sequence(note_seq.midi_to_note_sequence(pm)))

        if save_mix:
            sf.write(
                file=mix_path,
                data=stereo.T,
                samplerate=sr,
                format="wav",
            )

        if save_midi:
            pm.write(midi_path)

        return pm, composer, mix_path, midi_path
