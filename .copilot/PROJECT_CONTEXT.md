# Pop2Piano - Project Context & Overview

## 1. Project Description
**Pop2Piano** is a deep learning project utilizing Transformer-based models to generate piano covers from pop audio files. The system aligns audio waveform data with MIDI events to learn the mapping between pop music acoustics and piano arrangements.

**Key Paper:** [Pop2Piano : Pop Audio-based Piano Cover Generation](https://arxiv.org/abs/2211.00895)

## 2. Directory Structure & Key Components
*   **`download/`**: Contains scripts (`download.py`) to fetch the training dataset from YouTube based on `train_dataset.csv`.
*   **`preprocess/`**: Tools for beat quantization, alignment, and audio processing.
*   **`evaluate/`**: Inference scripts to generate piano covers from new audio inputs.
*   **`pop2piano_colab.ipynb`**: A unified, Colab-ready notebook implementing the entire pipeline with necessary fixes.
*   **`.venv/`**: Local Python virtual environment (Critical: Must be Python 3.11).

## 3. Current Operational Status
*   **Environment:** Validated on **Python 3.11.0**. Newer versions (3.12+) break core dependencies (`omegaconf`, `antlr4`).
*   **Data Acquisition:** The `download.py` script has been heavily patched to work in 2026 (see `TECHNICAL_DEBT_AND_FIXES.md`).
*   **Inference:** Successfully adapted for Google Colab environments by bypassing legacy dependency requirements (`essentia`).

## 4. Quick Start Guide

### Local Development (Windows)
1.  **Activate Environment:**
    ```powershell
    .venv\Scripts\Activate.ps1
    ```
2.  **Run Downloader:**
    ```powershell
    python download/download.py train_dataset.csv output_dir/ --max_size_gb 1.5
    ```

### Google Colab (Cloud)
1.  Push the latest local changes (especially `download.py`) to GitHub.
2.  Upload `pop2piano_colab.ipynb` to Google Colab.
3.  Run the notebook steps in order. It handles cloning, dependency patching, and execution.

## 5. Next Steps / TODOs
*   [ ] Validate the preprocessing pipeline on the downloaded dataset.
*   [ ] Test local inference (not just Colab) using the mocking strategy.
*   [ ] Consider containerizing the environment (Docker) to permanently solve dependency rot.
