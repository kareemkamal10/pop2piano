"""
Create one minimal preprocessed track so training can run (e.g. when download failed).
Usage: python create_demo_data.py [output_dir]
Default output_dir: output_dir
"""
import os
import sys
import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None

SAMPLE_RATE = 22050  # match config.dataset.sample_rate
DURATION_SEC = 4.0
TRACK_ID = "demo_track"
BASE_NAME = "demo"


def main():
    out_root = sys.argv[1] if len(sys.argv) > 1 else "output_dir"
    track_dir = os.path.join(out_root, TRACK_ID)
    os.makedirs(track_dir, exist_ok=True)

    # Beat times: 4 sec, ~2 beats/sec -> 9 beat steps (n_bars*4+1 with n_bars=2)
    n_steps = 9
    beatstep = np.linspace(0, DURATION_SEC, n_steps, dtype=np.float64)
    np.save(os.path.join(track_dir, f"{BASE_NAME}.beatstep.npy"), beatstep)

    # Notes: [onset_idx, offset_idx, pitch, velocity] (indices into beatstep)
    notes = np.array([
        [0, 1, 60, 80],
        [1, 2, 64, 80],
        [2, 3, 67, 80],
    ], dtype=np.float64)
    np.save(os.path.join(track_dir, f"{BASE_NAME}.notes.npy"), notes)

    # Audio: 4 sec at 22050 Hz (mono). Silence or very quiet tone so it's valid.
    n_samples = int(SAMPLE_RATE * DURATION_SEC)
    if sf is not None:
        audio = np.zeros(n_samples, dtype=np.float32)
        sf.write(os.path.join(track_dir, f"{BASE_NAME}.wav"), audio, SAMPLE_RATE)
    else:
        import scipy.io.wavfile
        audio = np.zeros(n_samples, dtype=np.int16)
        scipy.io.wavfile.write(
            os.path.join(track_dir, f"{BASE_NAME}.wav"),
            SAMPLE_RATE,
            audio,
        )

    print(f"✅ Created 1 demo track: {track_dir}")
    print(f"   {BASE_NAME}.beatstep.npy, {BASE_NAME}.notes.npy, {BASE_NAME}.wav")
    print("   Run training — it should see 1 valid track and train.")


if __name__ == "__main__":
    main()
