"""
Maqam detection during preprocessing.
Runs AFTER piano transcription (step 1) and BEFORE training.
Reads transcribed piano MIDI/notes, detects maqam, saves {song_ytid}.maqam.txt
for use by the dataset loader.
"""

import os
import sys
import glob
import argparse
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arabic_maqamat import detect_maqam

# Default config path
DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config.yaml",
)
MAQAM_THRESHOLD = 0.5  # Minimum confidence to assign a maqam; else "western"


def get_pitches_from_notes(notes_path: str) -> np.ndarray:
    """Load notes.npy and return pitch column (index 2)."""
    notes = np.load(notes_path)
    if notes.size == 0 or notes.ndim < 2:
        return np.array([], dtype=np.int64)
    return notes[:, 2].astype(np.int64)


def get_pitches_from_midi(midi_path: str) -> np.ndarray:
    """Load .mid and return all note pitches."""
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(midi_path)
        pitches = []
        for inst in pm.instruments:
            for n in inst.notes:
                pitches.append(n.pitch)
        return np.array(pitches, dtype=np.int64) if pitches else np.array([], dtype=np.int64)
    except Exception:
        return np.array([], dtype=np.int64)


def detect_and_save_maqam(
    track_dir: str,
    config_path: str,
    threshold: float = MAQAM_THRESHOLD,
    use_midi: bool = False,
) -> str | None:
    """
    For one track directory: find notes/midi, detect maqam, save .maqam.txt.
    Returns the token name saved, or None if skipped.
    """
    config = OmegaConf.load(config_path)
    maqam_to_token = config.get("maqam_to_token", {})
    if not maqam_to_token:
        maqam_to_token = {
            "Rast": "rast", "Bayyati": "bayyati", "Hijaz": "hijaz",
            "Saba": "saba", "Nahawand": "nahawand", "Kurd": "kurd",
            "Ajam": "ajam", "Nikriz": "nikriz", "Hijaz Kar": "hijaz_kar",
            "Husayni": "husayni", "Sikah": "sikah", "Nawa Athar": "nawa_athar",
            "Jiharkah": "jiharkah", "Athar Kurd": "athar_kurd",
        }

    if use_midi:
        midi_files = glob.glob(os.path.join(track_dir, "*.mid"))
        # Prefer quantized/synced midi (song_ytid.mid inside folder, not piano_ytid.mid at parent)
        midi_files = [f for f in midi_files if os.path.basename(f) != os.path.basename(track_dir) + ".mid"]
        if not midi_files:
            midi_files = glob.glob(os.path.join(track_dir, "*.mid"))
        if not midi_files:
            return None
        midi_path = midi_files[0]
        base = os.path.splitext(os.path.basename(midi_path))[0]
        pitches = get_pitches_from_midi(midi_path)
    else:
        notes_files = glob.glob(os.path.join(track_dir, "*.notes.npy"))
        if not notes_files:
            return None
        notes_path = notes_files[0]
        base = os.path.basename(notes_path).replace(".notes.npy", "")
        pitches = get_pitches_from_notes(notes_path)

    if len(pitches) < 5:
        token_name = "western"
    else:
        detections = detect_maqam(pitches.tolist(), threshold=threshold)
        if detections:
            name_en = detections[0][0]
            token_name = maqam_to_token.get(name_en, "western")
        else:
            token_name = "western"

    out_path = os.path.join(track_dir, base + ".maqam.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(token_name.strip().lower())
    return token_name


def main():
    parser = argparse.ArgumentParser(description="Detect maqam per track and save .maqam.txt")
    parser.add_argument("data_dir", help="Preprocessed data directory (contains track subdirs)")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to config.yaml")
    parser.add_argument("--threshold", type=float, default=MAQAM_THRESHOLD,
                        help="Min confidence for maqam (default 0.5)")
    parser.add_argument("--use-midi", action="store_true",
                        help="Use .mid files instead of .notes.npy for pitch extraction")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"Error: {args.data_dir} is not a directory")
        sys.exit(1)

    track_dirs = [
        d for d in glob.glob(os.path.join(args.data_dir, "*"))
        if os.path.isdir(d)
    ]

    counts = {}
    for track_dir in tqdm(track_dirs, desc="Maqam detection"):
        token = detect_and_save_maqam(
            track_dir,
            args.config,
            threshold=args.threshold,
            use_midi=args.use_midi,
        )
        if token:
            counts[token] = counts.get(token, 0) + 1

    print("\nMaqam token counts:")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
