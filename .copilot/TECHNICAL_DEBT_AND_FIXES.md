# Technical Debt & Applied Fixes

This document details the specific technical challenges encountered in reviving the **Pop2Piano** codebase (originally from ~2022) to work in a modern 2026 environment, specifically addressing issues on Windows and Google Colab.

## 1. Dependency Hell & Python Versioning
### The Problem
The project relies on `omegaconf` (version 2.1.1) which has a strict dependency on `antlr4-python3-runtime==4.9.3`. 
*   **Issue:** `antlr4-python3-runtime` < 4.10 is **incompatible with Python 3.12+**. It causes syntax errors during installation or runtime.
*   **Symptom:** `ModuleNotFoundError: No module named 'typing.io'` or similar deep syntax errors in site-packages.

### The Fix
*   **Forced Downgrade:** We explicitly set up a **Python 3.11** environment. This is the last modern version that maintains compatibility with the legacy `antlr4` runtime required by this project's specific dependency tree.
*   **Action:**
    ```powershell
    py -3.11 -m venv .venv
    ```

## 2. The `youtube-dl` Obsolescence
### The Problem
The original code used `youtube-dl`. As of 2026, `youtube-dl` is throttled or completely blocked by YouTube's modern API/bot defenses. Downloads would fail instantly or hang.

### The Fix
*   **Migration to `yt-dlp`:** We replaced the library with `yt-dlp`, a widely maintained fork.
*   **Code Patching:**
    *   Changed command-line arguments in `download.py` to match `yt-dlp` syntax (mostly compatible, but we optimized flags).
    *   Updated `utils/demo.py` to import `yt_dlp` instead of `youtube_dl`.

## 3. Cross-Platform Execution (Windows vs. Linux)
### The Problem
The original `download.py` used `os.system()` with hardcoded command strings.
1.  **Path Spaces:** On Windows, paths like `Desktop\New folder` caused the command to break because `os.system` doesn't handle unquoted spaces well.
2.  **Executable Extension:** On Windows, Python scripts often need to call `.exe` (e.g., `yt-dlp.exe`), whereas Linux expects just `yt-dlp`.
3.  **PATH Visibility:** `subprocess` calls sometimes fail to find the executable if it's only in the virtual environment's Scripts folder but not globally in System PATH.

### The Fix
*   **Smart Executable Resolution:** Added logic in `download.py` to detect `platform.system()`:
    ```python
    if platform.system() == "Windows":
        yt_dlp_name = "yt-dlp.exe"
    else:
        yt_dlp_name = "yt-dlp"
    ```
*   **Subprocess Overhaul:** Replaced unsafe `os.system()` calls with `subprocess.call()`. This allows passing arguments as a **list**, which automatically handles escaping spaces in file paths correctly.
*   **Fallback Logic:** If the specific path to `yt-dlp` isn't found, the code falls back to calling the global command `yt-dlp`.

## 4. The "Essentia" Mock (Colab Inference)
### The Problem
The `transformers.Pop2PianoFeatureExtractor` class has a hard dependency: `import essentia`.
*   **Issue:** `essentia` is a complex C++ audio library. Installing it on Colab via `pip` often fails due to version mismatches or compilation errors, or takes extremely long.
*   **Consequence:** Converting a simple MP3 to features crashes because the library is missing.

### The Fix
We don't actually *need* `essentia` for everything. We only need the beat tracking.
*   **The Mock:** We inject a fake module into `sys.modules` before importing transformers:
    ```python
    sys.modules["essentia"] = MagicMock()
    sys.modules["essentia.standard"] = MagicMock()
    ```
*   **The Replacement:** In our manual inference script, we use `librosa.beat.beat_track` to extract the beat information and feed it into the model manually, bypassing the part of the FeatureExtractor that would use `essentia`.

## 5. Storage Management
### The Problem
The full dataset is massive (hundreds of GBs). Running `download.py` unchecked on a local machine could fill the drive.

### The Fix
*   **Size Cap:** Implemented a `--max_size_gb` argument in `download.py`. It checks the `output_dir` size recursively before starting each new download and aborts gracefully if the limit is exceeded.
