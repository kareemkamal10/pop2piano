"""
Smart Inference Module for Pop2Piano
=====================================
Auto-detection of maqam, Arabic music features, and best composer selection.

This module adds intelligence on top of the HuggingFace Pop2Piano API:
1. Detect if a song is Arabic and which maqam it uses (from audio, before generation)
2. Try multiple composers and pick the best result automatically
3. Apply maqam post-processing to the generated MIDI (works with HuggingFace API)

Usage:
    from smart_inference import SmartPop2Piano

    smart = SmartPop2Piano(model, processor)
    result = smart.generate("song.mp3")
    result.best_midi.write("output.mid")
"""

import numpy as np
import librosa
import pretty_midi
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field

# Import project modules
from arabic_maqamat import (
    MAQAMAT, get_maqam, get_maqam_scale, detect_maqam,
    quantize_to_maqam, Maqam, list_all_maqamat
)
from piano_rules import (
    apply_piano_rules, PianoNote, clamp_to_piano_range,
    quantize_to_scale
)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AudioAnalysis:
    """Results from analyzing an audio file."""
    is_arabic: bool = False
    arabic_confidence: float = 0.0
    detected_maqam: Optional[str] = None
    maqam_confidence: float = 0.0
    all_maqam_scores: List[Tuple[str, float]] = field(default_factory=list)
    estimated_bpm: float = 0.0
    has_augmented_second: bool = False
    dominant_pitch_classes: List[int] = field(default_factory=list)
    chroma_energy: Optional[np.ndarray] = None


@dataclass
class ComposerResult:
    """Result from generating with a specific composer."""
    composer: str
    midi: pretty_midi.PrettyMIDI
    score: float = 0.0
    note_count: int = 0
    pitch_range: int = 0
    maqam_adherence: float = 0.0


@dataclass
class SmartResult:
    """Full result from smart generation."""
    best_midi: pretty_midi.PrettyMIDI
    best_composer: str
    best_score: float
    audio_analysis: AudioAnalysis
    applied_maqam: Optional[str] = None
    all_results: List[ComposerResult] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Audio Analysis (Pre-Generation Detection)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_audio(audio: np.ndarray, sr: int = 44100) -> AudioAnalysis:
    """
    Analyze audio to detect Arabic music features and maqam BEFORE generation.
    Uses chroma features to match pitch content against known maqam scales.

    Args:
        audio: Audio waveform as numpy array
        sr: Sample rate

    Returns:
        AudioAnalysis with detected features
    """
    analysis = AudioAnalysis()

    # Extract chroma features (pitch class energy distribution)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=4096, hop_length=512)
    chroma_energy = chroma.mean(axis=1)  # Average energy per pitch class (12 values)

    # Normalize to [0, 1]
    max_energy = chroma_energy.max()
    if max_energy > 0:
        chroma_energy = chroma_energy / max_energy
    analysis.chroma_energy = chroma_energy

    # Find dominant pitch classes (above threshold)
    threshold = 0.3
    analysis.dominant_pitch_classes = [
        i for i, e in enumerate(chroma_energy) if e >= threshold
    ]

    # Detect augmented second intervals (characteristic of Hijaz, Nawa Athar, etc.)
    # Augmented second = 3 semitones between consecutive scale degrees
    analysis.has_augmented_second = _detect_augmented_second(chroma_energy)

    # Match against all maqam scales
    maqam_scores = _score_all_maqamat(chroma_energy)
    analysis.all_maqam_scores = maqam_scores

    if maqam_scores:
        best_maqam_name, best_score = maqam_scores[0]
        analysis.detected_maqam = best_maqam_name
        analysis.maqam_confidence = best_score

    # Determine if song is likely Arabic
    # Arabic indicators: augmented seconds, high maqam match (non-Ajam), specific pitch patterns
    analysis.is_arabic, analysis.arabic_confidence = _detect_arabic(analysis)

    # Estimate BPM
    try:
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = tempo[0]
        analysis.estimated_bpm = float(tempo)
    except Exception:
        analysis.estimated_bpm = 120.0

    return analysis


def _detect_augmented_second(chroma_energy: np.ndarray) -> bool:
    """
    Detect the presence of augmented second intervals.
    These are characteristic of Hijaz, Hijaz Kar, Nawa Athar, and Nikriz maqamat.
    An augmented second = gap of 3 semitones in the scale.
    """
    active = [i for i, e in enumerate(chroma_energy) if e > 0.25]
    if len(active) < 3:
        return False

    active_sorted = sorted(active)
    for i in range(len(active_sorted) - 1):
        gap = (active_sorted[i + 1] - active_sorted[i]) % 12
        if gap == 3:
            # Check if the notes around the gap are strong (not just noise)
            if (chroma_energy[active_sorted[i]] > 0.3 and
                    chroma_energy[active_sorted[i + 1]] > 0.3):
                return True
    return False


def _score_all_maqamat(chroma_energy: np.ndarray) -> List[Tuple[str, float]]:
    """
    Score how well the audio's pitch content matches each maqam.
    Uses a weighted approach: energy on scale notes vs off-scale notes.
    """
    results = []
    checked = set()

    for name, maqam in MAQAMAT.items():
        if maqam.name_en in checked:
            continue
        checked.add(maqam.name_en)

        scale = get_maqam_scale(maqam)
        in_scale_energy = sum(chroma_energy[p] for p in scale)
        out_scale_energy = sum(
            chroma_energy[p] for p in range(12) if p not in scale
        )
        total = in_scale_energy + out_scale_energy

        if total > 0:
            # Weighted score: in-scale energy dominance
            score = in_scale_energy / total

            # Bonus for augmented seconds (Hijaz family)
            if maqam.name_en in ("Hijaz", "Hijaz Kar", "Nawa Athar", "Nikriz", "Athar Kurd"):
                # Check if the augmented second interval has strong energy
                aug_intervals = []
                for i in range(len(maqam.intervals) - 1):
                    if maqam.intervals[i + 1] - maqam.intervals[i] == 3:
                        lower = (maqam.root_note + maqam.intervals[i]) % 12
                        upper = (maqam.root_note + maqam.intervals[i + 1]) % 12
                        aug_intervals.append((lower, upper))

                for lower, upper in aug_intervals:
                    if chroma_energy[lower] > 0.3 and chroma_energy[upper] > 0.3:
                        score += 0.05  # Small bonus

            results.append((maqam.name_en.lower(), round(score, 3)))

    return sorted(results, key=lambda x: x[1], reverse=True)


def _detect_arabic(analysis: AudioAnalysis) -> Tuple[bool, float]:
    """
    Determine if a song is likely Arabic based on multiple indicators.

    Returns:
        (is_arabic, confidence)
    """
    confidence = 0.0

    # Indicator 1: Augmented second (strong Arabic indicator)
    if analysis.has_augmented_second:
        confidence += 0.35

    # Indicator 2: Best maqam is characteristically Arabic (not Ajam/Major)
    if analysis.detected_maqam and analysis.detected_maqam not in ("ajam",):
        # Maqamat like Hijaz, Bayyati, Saba are strongly Arabic
        strongly_arabic = {"hijaz", "bayyati", "saba", "sikah", "hijaz_kar",
                           "nawa_athar", "nikriz", "athar_kurd", "husayni"}
        if analysis.detected_maqam in strongly_arabic:
            confidence += 0.35
        else:
            confidence += 0.15

    # Indicator 3: High maqam adherence with characteristic maqam
    if analysis.maqam_confidence > 0.75:
        confidence += 0.15

    # Indicator 4: Multiple strong pitch classes (Arabic music tends to use more scale degrees)
    if len(analysis.dominant_pitch_classes) >= 6:
        confidence += 0.1

    confidence = min(confidence, 1.0)
    is_arabic = confidence >= 0.45

    return is_arabic, round(confidence, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# MIDI Post-Processing (Works with HuggingFace API output)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_maqam_to_midi(
    pm: pretty_midi.PrettyMIDI,
    maqam_name: Optional[str] = None,
    auto_detect: bool = True,
    simplify: bool = True,
    quantize: bool = True,
    humanize: bool = False,
) -> pretty_midi.PrettyMIDI:
    """
    Apply maqam post-processing to a pretty_midi object.
    This is the KEY function that bridges HuggingFace API output with Arabic music support.

    Args:
        pm: PrettyMIDI object from HuggingFace model output
        maqam_name: Maqam to apply (e.g., 'hijaz', 'bayyati'). Auto-detected if None.
        auto_detect: If True and maqam_name is None, detect from MIDI pitches
        simplify: Simplify complex chords
        quantize: Quantize rhythm to grid
        humanize: Add human-like velocity variation

    Returns:
        Post-processed PrettyMIDI object
    """
    if not pm.instruments or not pm.instruments[0].notes:
        return pm

    notes = pm.instruments[0].notes

    # Auto-detect maqam from MIDI pitches if not specified
    if maqam_name is None and auto_detect:
        pitches = [n.pitch for n in notes]
        detections = detect_maqam(pitches, threshold=0.5)
        if detections:
            maqam_name = detections[0][0].lower()

    # Get scale pitches
    scale_pitches = None
    if maqam_name:
        maqam_obj = get_maqam(maqam_name)
        if maqam_obj:
            scale_pitches = get_maqam_scale(maqam_obj)

    # Convert to PianoNote objects
    piano_notes = [
        PianoNote(
            pitch=n.pitch,
            onset=n.start,
            offset=n.end,
            velocity=n.velocity
        )
        for n in notes
    ]

    # Apply piano rules with optional scale quantization
    processed = apply_piano_rules(
        notes=piano_notes,
        scale_pitches=scale_pitches,
        simplify=simplify,
        quantize=quantize,
        humanize=humanize,
    )

    # Rebuild MIDI
    pm.instruments[0].notes = [
        pretty_midi.Note(
            velocity=pn.velocity,
            pitch=pn.pitch,
            start=pn.onset,
            end=pn.offset,
        )
        for pn in processed
    ]

    return pm


# ═══════════════════════════════════════════════════════════════════════════════
# MIDI Quality Scoring
# ═══════════════════════════════════════════════════════════════════════════════

def score_midi_quality(
    pm: pretty_midi.PrettyMIDI,
    maqam_name: Optional[str] = None,
) -> float:
    """
    Score the musical quality of a generated MIDI.
    Higher score = better quality. Range: 0.0 to 1.0.

    Evaluates: note count, pitch range, rhythmic consistency,
    velocity variation, and maqam adherence.
    """
    if not pm.instruments or not pm.instruments[0].notes:
        return 0.0

    notes = pm.instruments[0].notes
    score = 0.0

    # 1. Note count (reasonable number of notes)
    note_count = len(notes)
    if 20 <= note_count <= 400:
        score += 0.2
    elif 10 <= note_count < 20 or 400 < note_count <= 600:
        score += 0.1
    # Too few or too many = 0

    # 2. Pitch range (good piano covers use a wide range)
    pitches = [n.pitch for n in notes]
    pitch_range = max(pitches) - min(pitches)
    if pitch_range >= 24:  # 2+ octaves
        score += 0.15
    elif pitch_range >= 12:  # 1+ octave
        score += 0.1

    # 3. Note density (notes per second -- should be moderate)
    min_start = min(n.start for n in notes)
    max_end = max(n.end for n in notes)
    duration = max_end - min_start
    if duration > 0:
        density = note_count / duration
        if 2.0 <= density <= 12.0:
            score += 0.2
        elif 1.0 <= density < 2.0 or 12.0 < density <= 18.0:
            score += 0.1

    # 4. Velocity variation (humanlike playing)
    velocities = [n.velocity for n in notes]
    if len(set(velocities)) > 1:
        vel_std = float(np.std(velocities))
        if vel_std > 8:
            score += 0.1
        elif vel_std > 3:
            score += 0.05

    # 5. Rhythmic regularity (onset spacing consistency)
    onsets = sorted(set(n.start for n in notes))
    if len(onsets) > 2:
        intervals = np.diff(onsets)
        intervals = intervals[intervals > 0.01]  # Filter noise
        if len(intervals) > 1:
            mean_interval = float(np.mean(intervals))
            if mean_interval > 0:
                cv = float(np.std(intervals)) / mean_interval
                if cv < 1.5:
                    score += 0.15
                elif cv < 2.5:
                    score += 0.08

    # 6. Maqam adherence (if maqam specified)
    if maqam_name:
        maqam_obj = get_maqam(maqam_name)
        if maqam_obj:
            scale = set(get_maqam_scale(maqam_obj))
            pitch_classes = [p % 12 for p in pitches]
            in_scale = sum(1 for pc in pitch_classes if pc in scale)
            adherence = in_scale / len(pitch_classes) if pitch_classes else 0
            score += 0.2 * adherence

    return min(round(score, 3), 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Smart Generation (Main Class)
# ═══════════════════════════════════════════════════════════════════════════════

# Maqam-aware token names (used when model is fine-tuned with config composer_to_feature_token)
MAQAM_TOKEN_NAMES = [
    "rast", "bayyati", "hijaz", "saba", "nahawand", "kurd", "ajam", "nikriz",
    "hijaz_kar", "husayni", "sikah", "nawa_athar", "jiharkah", "athar_kurd",
    "western",
]
# Fallback for models that only have composer1-21 (vanilla HuggingFace)
LEGACY_FALLBACK_COMPOSER = "composer1"


class SmartPop2Piano:
    """
    Smart wrapper around HuggingFace Pop2Piano that adds:
    - Automatic maqam detection from audio
    - Arabic song detection
    - Maqam-based token selection (detected maqam or western)
    - Maqam post-processing on generated MIDI

    Usage:
        from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor
        model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
        processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")

        smart = SmartPop2Piano(model, processor)
        result = smart.generate("arabic_song.mp3")

        print(f"Detected: {result.audio_analysis.detected_maqam}")
        print(f"Best composer: {result.best_composer}")
        result.best_midi.write("output.mid")
    """

    def __init__(self, model, processor, device=None):
        """
        Args:
            model: Pop2PianoForConditionalGeneration instance
            processor: Pop2PianoProcessor instance
            device: torch device (auto-detected if None)
        """
        self.model = model
        self.processor = processor

        if device is None:
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)

    def generate(
        self,
        audio_path: Optional[str] = None,
        audio: Optional[np.ndarray] = None,
        sr: int = 44100,
        # Auto-detection settings
        auto_detect_maqam: bool = True,
        maqam: Optional[str] = None,
        # Composer settings
        composer: Optional[str] = None,
        auto_select_composer: bool = True,
        composers_to_try: Optional[List[str]] = None,
        # Post-processing settings
        apply_post_processing: bool = True,
        simplify_chords: bool = True,
        quantize_rhythm: bool = True,
        humanize: bool = False,
        # Output settings
        save_midi: Optional[str] = None,
        verbose: bool = True,
    ) -> SmartResult:
        """
        Smart generation with full auto-detection.

        Args:
            audio_path: Path to audio file (MP3/WAV)
            audio: Pre-loaded audio array (alternative to audio_path)
            sr: Sample rate for pre-loaded audio
            auto_detect_maqam: Auto-detect Arabic maqam from audio
            maqam: Force a specific maqam (e.g., 'hijaz', 'bayyati')
            composer: Force a specific composer (e.g., 'composer1')
            auto_select_composer: Try multiple composers and pick the best
            composers_to_try: List of composers to try (default: 5 diverse ones)
            apply_post_processing: Apply maqam/piano rules post-processing
            simplify_chords: Simplify complex chords
            quantize_rhythm: Quantize to beat grid
            humanize: Add human-like velocity variation
            save_midi: Path to save output MIDI (optional)
            verbose: Print progress messages

        Returns:
            SmartResult with best MIDI, analysis, and all composer results
        """
        # ── Step 1: Load Audio ──
        if audio is None:
            if audio_path is None:
                raise ValueError("Either audio_path or audio must be provided")
            if verbose:
                print(f"Loading audio: {audio_path}")
            audio, sr = librosa.load(audio_path, sr=sr)

        # ── Step 2: Analyze Audio ──
        if verbose:
            print("Analyzing audio features...")
        analysis = analyze_audio(audio, sr)

        if verbose:
            if analysis.is_arabic:
                print(f"  Arabic song detected! (confidence: {analysis.arabic_confidence:.0%})")
            else:
                print(f"  Non-Arabic song (Arabic confidence: {analysis.arabic_confidence:.0%})")

            if analysis.detected_maqam:
                print(f"  Detected maqam: {analysis.detected_maqam} "
                      f"(confidence: {analysis.maqam_confidence:.0%})")

            if analysis.has_augmented_second:
                print("  Augmented second interval detected (Hijaz-family indicator)")

            print(f"  Estimated BPM: {analysis.estimated_bpm:.0f}")

        # ── Step 3: Determine Maqam ──
        effective_maqam = maqam  # User-specified takes priority
        if effective_maqam is None and auto_detect_maqam:
            if analysis.is_arabic and analysis.detected_maqam:
                effective_maqam = analysis.detected_maqam
                if verbose:
                    print(f"  Auto-selected maqam: {effective_maqam}")

        # ── Step 4: Determine composer token(s) to use (maqam-aware) ──
        if composer is not None:
            composer_list = [composer]
        elif composers_to_try is not None:
            composer_list = composers_to_try
        elif auto_select_composer:
            # Single best token: detected maqam or western
            if effective_maqam:
                composer_list = [effective_maqam]
                if verbose:
                    print(f"  Using maqam token: {effective_maqam}")
            else:
                composer_list = ["western"]
                if verbose:
                    print("  Using western token")
        else:
            composer_list = [effective_maqam or "western"]

        # ── Step 5: Process Audio ──
        inputs = self.processor(audio=audio, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # ── Step 6: Generate with Each Composer ──
        all_results = []

        if verbose and len(composer_list) > 1:
            print(f"\nTrying {len(composer_list)} composers...")

        for comp in composer_list:
            if verbose:
                print(f"  Generating with {comp}...", end=" ")

            try:
                model_output = self.model.generate(
                    input_features=inputs["input_features"],
                    composer=comp,
                )

                midi_obj = self.processor.batch_decode(
                    token_ids=model_output.cpu(),
                    feature_extractor_output={
                        k: v.cpu() for k, v in inputs.items()
                    },
                )["pretty_midi_objects"][0]

                # Apply post-processing
                if apply_post_processing and effective_maqam:
                    midi_obj = apply_maqam_to_midi(
                        midi_obj,
                        maqam_name=effective_maqam,
                        auto_detect=False,
                        simplify=simplify_chords,
                        quantize=quantize_rhythm,
                        humanize=humanize,
                    )
                elif apply_post_processing:
                    # Apply basic piano rules without maqam
                    midi_obj = apply_maqam_to_midi(
                        midi_obj,
                        maqam_name=None,
                        auto_detect=False,
                        simplify=simplify_chords,
                        quantize=quantize_rhythm,
                        humanize=humanize,
                    )

                # Score
                quality = score_midi_quality(midi_obj, effective_maqam)
                note_count = len(midi_obj.instruments[0].notes) if midi_obj.instruments else 0
                pitches = [n.pitch for n in midi_obj.instruments[0].notes] if note_count > 0 else []
                p_range = (max(pitches) - min(pitches)) if pitches else 0

                # Maqam adherence
                maq_adh = 0.0
                if effective_maqam and pitches:
                    maqam_obj = get_maqam(effective_maqam)
                    if maqam_obj:
                        scale = set(get_maqam_scale(maqam_obj))
                        pitch_classes = [p % 12 for p in pitches]
                        maq_adh = sum(1 for pc in pitch_classes if pc in scale) / len(pitch_classes)

                result = ComposerResult(
                    composer=comp,
                    midi=midi_obj,
                    score=quality,
                    note_count=note_count,
                    pitch_range=p_range,
                    maqam_adherence=round(maq_adh, 2),
                )
                all_results.append(result)

                if verbose:
                    maq_str = f", maqam fit: {maq_adh:.0%}" if effective_maqam else ""
                    print(f"score: {quality:.2f}, notes: {note_count}, "
                          f"range: {p_range}{maq_str}")

            except Exception as e:
                # Fallback: if maqam token not in model (vanilla HuggingFace), try legacy composer
                if comp in MAQAM_TOKEN_NAMES and comp != LEGACY_FALLBACK_COMPOSER:
                    try:
                        model_output = self.model.generate(
                            input_features=inputs["input_features"],
                            composer=LEGACY_FALLBACK_COMPOSER,
                        )
                        midi_obj = self.processor.batch_decode(
                            token_ids=model_output.cpu(),
                            feature_extractor_output={k: v.cpu() for k, v in inputs.items()},
                        )["pretty_midi_objects"][0]
                        if apply_post_processing:
                            midi_obj = apply_maqam_to_midi(
                                midi_obj, maqam_name=effective_maqam or None,
                                auto_detect=False, simplify=simplify_chords,
                                quantize=quantize_rhythm, humanize=humanize,
                            )
                        quality = score_midi_quality(midi_obj, effective_maqam)
                        note_count = len(midi_obj.instruments[0].notes) if midi_obj.instruments else 0
                        pitches = [n.pitch for n in midi_obj.instruments[0].notes] if note_count > 0 else []
                        p_range = (max(pitches) - min(pitches)) if pitches else 0
                        maq_adh = 0.0
                        if effective_maqam and pitches:
                            maqam_obj = get_maqam(effective_maqam)
                            if maqam_obj:
                                scale = set(get_maqam_scale(maqam_obj))
                                pitch_classes = [p % 12 for p in pitches]
                                maq_adh = sum(1 for pc in pitch_classes if pc in scale) / len(pitch_classes)
                        all_results.append(ComposerResult(
                            composer=comp,
                            midi=midi_obj,
                            score=quality,
                            note_count=note_count,
                            pitch_range=p_range,
                            maqam_adherence=round(maq_adh, 2),
                        ))
                        if verbose:
                            print(f"score: {quality:.2f} (fallback {LEGACY_FALLBACK_COMPOSER})")
                    except Exception:
                        if verbose:
                            print(f"FAILED ({e})")
                else:
                    if verbose:
                        print(f"FAILED ({e})")

        # ── Step 7: Pick Best Result ──
        if not all_results:
            raise RuntimeError("All composer attempts failed. Check model and audio.")

        best = max(all_results, key=lambda r: r.score)

        if verbose:
            print(f"\n** Best result: {best.composer} (score: {best.score:.2f}) **")
            if effective_maqam:
                maqam_obj = get_maqam(effective_maqam)
                if maqam_obj:
                    print(f"** Applied maqam: {maqam_obj.name_en} ({maqam_obj.name_ar}) **")

        # ── Step 8: Save if requested ──
        if save_midi:
            best.midi.write(save_midi)
            if verbose:
                print(f"Saved MIDI to: {save_midi}")

        return SmartResult(
            best_midi=best.midi,
            best_composer=best.composer,
            best_score=best.score,
            audio_analysis=analysis,
            applied_maqam=effective_maqam,
            all_results=all_results,
        )

    def generate_all_composers(
        self,
        audio_path: str,
        sr: int = 44100,
        output_dir: str = ".",
        maqam: Optional[str] = None,
        auto_detect_maqam: bool = True,
        verbose: bool = True,
    ) -> List[ComposerResult]:
        """
        Generate with all maqam tokens (and western) and save each.
        Useful for finding the best maqam for a specific song.

        Args:
            audio_path: Path to audio file
            sr: Sample rate
            output_dir: Directory to save MIDI files
            maqam: Force maqam (auto-detected if None)
            auto_detect_maqam: Auto-detect maqam
            verbose: Print progress

        Returns:
            List of all ComposerResult, sorted by score (best first)
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        all_composers = list(MAQAM_TOKEN_NAMES)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]

        result = self.generate(
            audio_path=audio_path,
            sr=sr,
            auto_detect_maqam=auto_detect_maqam,
            maqam=maqam,
            auto_select_composer=True,
            composers_to_try=all_composers,
            verbose=verbose,
        )

        # Save all MIDIs
        for cr in sorted(result.all_results, key=lambda r: r.score, reverse=True):
            midi_path = os.path.join(output_dir, f"{base_name}_{cr.composer}.mid")
            cr.midi.write(midi_path)
            if verbose:
                print(f"  Saved: {midi_path} (score: {cr.score:.2f})")

        return sorted(result.all_results, key=lambda r: r.score, reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════

def quick_analyze(audio_path: str, sr: int = 44100) -> AudioAnalysis:
    """
    Quick analysis of an audio file without generation.
    Useful for checking if a song is Arabic and which maqam it's in.

    Usage:
        from smart_inference import quick_analyze
        result = quick_analyze("song.mp3")
        print(f"Arabic: {result.is_arabic}, Maqam: {result.detected_maqam}")
    """
    audio, sr = librosa.load(audio_path, sr=sr)
    return analyze_audio(audio, sr)


def post_process_midi(
    midi_path: str,
    output_path: str,
    maqam: Optional[str] = None,
    auto_detect: bool = True,
    simplify: bool = True,
    quantize: bool = True,
    humanize: bool = False,
) -> pretty_midi.PrettyMIDI:
    """
    Post-process an existing MIDI file with maqam rules.
    Works on any MIDI file, not just Pop2Piano output.

    Usage:
        from smart_inference import post_process_midi
        post_process_midi("raw_output.mid", "processed.mid", maqam="hijaz")
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    pm = apply_maqam_to_midi(
        pm,
        maqam_name=maqam,
        auto_detect=auto_detect,
        simplify=simplify,
        quantize=quantize,
        humanize=humanize,
    )
    pm.write(output_path)
    return pm


def print_analysis_report(analysis: AudioAnalysis):
    """Print a readable analysis report."""
    print("=" * 60)
    print("  Audio Analysis Report")
    print("=" * 60)
    print(f"  Arabic Song:        {'Yes' if analysis.is_arabic else 'No'} "
          f"({analysis.arabic_confidence:.0%} confidence)")
    print(f"  Detected Maqam:     {analysis.detected_maqam or 'N/A'} "
          f"({analysis.maqam_confidence:.0%} confidence)")
    print(f"  Augmented Second:   {'Yes' if analysis.has_augmented_second else 'No'}")
    print(f"  Estimated BPM:      {analysis.estimated_bpm:.0f}")
    print(f"  Active Pitch Classes: {len(analysis.dominant_pitch_classes)}")

    if analysis.all_maqam_scores:
        print("\n  Top 5 Maqam Matches:")
        for name, score in analysis.all_maqam_scores[:5]:
            maqam_obj = get_maqam(name)
            ar_name = maqam_obj.name_ar if maqam_obj else ""
            print(f"    {name:15s} ({ar_name:>8s}) : {score:.0%}")

    print("=" * 60)
