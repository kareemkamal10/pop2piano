"""
Pop2Piano Web UI - Gradio Interface
A beautiful web interface for converting pop music to piano covers.
Theme: Cyan
"""

import os
import sys
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import librosa
import soundfile as sf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# Configuration & Constants
# ============================================================

LIBRARY_DIR = os.path.join(os.path.dirname(__file__), "audio_library")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_dir")
STEMS_DIR = os.path.join(os.path.dirname(__file__), "stems_cache")

# Create directories if they don't exist
os.makedirs(LIBRARY_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STEMS_DIR, exist_ok=True)

# Library metadata file
LIBRARY_META_FILE = os.path.join(LIBRARY_DIR, "library_meta.json")

# Composer styles available
COMPOSER_STYLES = [
    "composer1", "composer2", "composer3", "composer4", "composer5",
    "composer6", "composer7", "composer8", "composer9", "composer10",
    "composer11", "composer12", "composer13", "composer14", "composer15",
    "composer16", "composer17", "composer18", "composer19", "composer20",
    "composer21"
]

# ============================================================
# Custom CSS Theme (Cyan)
# ============================================================

CUSTOM_CSS = """
/* Main Theme Colors - Cyan Palette */
:root {
    --primary-cyan: #00BCD4;
    --primary-dark: #00838F;
    --primary-light: #B2EBF2;
    --accent-cyan: #00E5FF;
    --bg-dark: #0D1B2A;
    --bg-card: #1B263B;
    --bg-input: #243B53;
    --text-primary: #E0F7FA;
    --text-secondary: #80DEEA;
    --success: #00E676;
    --warning: #FFD54F;
    --error: #FF5252;
    --gradient-1: linear-gradient(135deg, #00BCD4 0%, #00838F 100%);
    --gradient-2: linear-gradient(135deg, #0D1B2A 0%, #1B263B 100%);
}

/* Global Styles */
.gradio-container {
    background: var(--gradient-2) !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}

/* Headers */
.app-header {
    text-align: center;
    padding: 30px 20px;
    background: var(--gradient-1);
    border-radius: 16px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(0, 188, 212, 0.3);
}

.app-header h1 {
    color: white !important;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
}

.app-header p {
    color: var(--primary-light) !important;
    font-size: 1.1rem !important;
    margin-top: 8px !important;
}

/* Cards */
.card {
    background: var(--bg-card) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    border: 1px solid rgba(0, 188, 212, 0.2) !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2) !important;
}

/* Buttons */
.primary-btn {
    background: var(--gradient-1) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(0, 188, 212, 0.4) !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0, 188, 212, 0.6) !important;
}

.secondary-btn {
    background: transparent !important;
    border: 2px solid var(--primary-cyan) !important;
    color: var(--primary-cyan) !important;
    font-weight: 600 !important;
    padding: 10px 22px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
}

.secondary-btn:hover {
    background: var(--primary-cyan) !important;
    color: var(--bg-dark) !important;
}

/* Sliders */
input[type="range"] {
    accent-color: var(--primary-cyan) !important;
}

/* Tabs */
.tab-nav button {
    background: var(--bg-card) !important;
    color: var(--text-secondary) !important;
    border: none !important;
    padding: 12px 24px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

.tab-nav button.selected {
    background: var(--gradient-1) !important;
    color: white !important;
}

/* File Upload Area */
.upload-area {
    border: 2px dashed var(--primary-cyan) !important;
    border-radius: 12px !important;
    background: rgba(0, 188, 212, 0.05) !important;
    transition: all 0.3s ease !important;
}

.upload-area:hover {
    background: rgba(0, 188, 212, 0.1) !important;
    border-color: var(--accent-cyan) !important;
}

/* Library Grid */
.library-item {
    background: var(--bg-card) !important;
    border-radius: 8px !important;
    padding: 12px !important;
    border: 1px solid rgba(0, 188, 212, 0.2) !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
}

.library-item:hover {
    border-color: var(--primary-cyan) !important;
    transform: translateY(-2px) !important;
}

/* Audio Waveform */
.audio-waveform {
    background: var(--bg-input) !important;
    border-radius: 8px !important;
}

/* Status Indicators */
.status-success {
    color: var(--success) !important;
}

.status-warning {
    color: var(--warning) !important;
}

.status-error {
    color: var(--error) !important;
}

/* Progress Bar */
.progress-bar {
    background: var(--bg-input) !important;
    border-radius: 4px !important;
    overflow: hidden !important;
}

.progress-bar-fill {
    background: var(--gradient-1) !important;
    height: 100% !important;
    transition: width 0.3s ease !important;
}

/* Section Titles */
.section-title {
    color: var(--primary-cyan) !important;
    font-size: 1.3rem !important;
    font-weight: 600 !important;
    margin-bottom: 16px !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}

/* Stem Controls */
.stem-control {
    background: var(--bg-input) !important;
    border-radius: 8px !important;
    padding: 16px !important;
    margin-bottom: 12px !important;
    display: flex !important;
    align-items: center !important;
    gap: 16px !important;
}

.stem-icon {
    font-size: 24px !important;
}

/* Output Cards */
.output-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, rgba(0, 188, 212, 0.1) 100%) !important;
    border: 1px solid var(--primary-cyan) !important;
    border-radius: 12px !important;
    padding: 20px !important;
}

/* Info Box */
.info-box {
    background: rgba(0, 188, 212, 0.1) !important;
    border-left: 4px solid var(--primary-cyan) !important;
    border-radius: 0 8px 8px 0 !important;
    padding: 16px !important;
    margin: 16px 0 !important;
}

/* Labels */
label {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}

/* Dropdown */
select, .dropdown {
    background: var(--bg-input) !important;
    color: var(--text-primary) !important;
    border: 1px solid rgba(0, 188, 212, 0.3) !important;
    border-radius: 8px !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-dark);
}

::-webkit-scrollbar-thumb {
    background: var(--primary-cyan);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--accent-cyan);
}

/* Piano Animation (Optional Enhancement) */
@keyframes pianoKey {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(2px); }
}

.piano-animation {
    animation: pianoKey 0.3s ease-in-out;
}
"""

# ============================================================
# SVG Icons
# ============================================================

SVG_ICONS = {
    "piano": '''<svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-7 15.5h-1V13H9v5.5H8V13H6v5.5H5V10h12v8.5h-1V13h-2v5.5h-1V13h-1v5.5z"/>
    </svg>''',
    
    "music": '''<svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
        <path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z"/>
    </svg>''',
    
    "upload": '''<svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
        <path d="M9 16h6v-6h4l-7-7-7 7h4zm-4 2h14v2H5z"/>
    </svg>''',
    
    "folder": '''<svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
        <path d="M10 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/>
    </svg>''',
    
    "settings": '''<svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
        <path d="M19.14 12.94c.04-.31.06-.63.06-.94 0-.31-.02-.63-.06-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.04.31-.06.63-.06.94s.02.63.06.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z"/>
    </svg>''',
    
    "download": '''<svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
        <path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/>
    </svg>''',
    
    "play": '''<svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
        <path d="M8 5v14l11-7z"/>
    </svg>''',
    
    "vocals": '''<svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
        <path d="M12 14c1.66 0 2.99-1.34 2.99-3L15 5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.48 6-3.3 6-6.72h-1.7z"/>
    </svg>''',
    
    "drums": '''<svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
        <circle cx="12" cy="12" r="10"/>
    </svg>''',
    
    "bass": '''<svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </svg>''',
    
    "other": '''<svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
        <path d="M4 6H2v14c0 1.1.9 2 2 2h14v-2H4V6zm16-4H8c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-1 9h-4v4h-2v-4H9V9h4V5h2v4h4v2z"/>
    </svg>''',
    
    "waveform": '''<svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
        <path d="M7 18h2V6H7v12zm4 4h2V2h-2v20zm-8-8h2v-4H3v4zm12 4h2V6h-2v12zm4-8v4h2v-4h-2z"/>
    </svg>'''
}

# ============================================================
# Library Management Functions
# ============================================================

def load_library_metadata():
    """Load library metadata from JSON file."""
    if os.path.exists(LIBRARY_META_FILE):
        with open(LIBRARY_META_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"files": []}

def save_library_metadata(metadata):
    """Save library metadata to JSON file."""
    with open(LIBRARY_META_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def add_to_library(audio_path, original_name=None):
    """Add an audio file to the library."""
    if audio_path is None:
        return None, "No file provided"
    
    metadata = load_library_metadata()
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if original_name:
        base_name = Path(original_name).stem
    else:
        base_name = Path(audio_path).stem
    
    ext = Path(audio_path).suffix
    new_filename = f"{base_name}_{timestamp}{ext}"
    new_path = os.path.join(LIBRARY_DIR, new_filename)
    
    # Copy file to library
    shutil.copy2(audio_path, new_path)
    
    # Get audio info
    try:
        y, sr = librosa.load(audio_path, sr=None, duration=10)
        duration = librosa.get_duration(y=y, sr=sr)
    except:
        duration = 0
    
    # Add to metadata
    file_info = {
        "id": timestamp,
        "filename": new_filename,
        "original_name": original_name or base_name,
        "path": new_path,
        "added_date": datetime.now().isoformat(),
        "duration": duration
    }
    metadata["files"].append(file_info)
    save_library_metadata(metadata)
    
    return new_path, f"✅ Added to library: {base_name}"

def get_library_files():
    """Get list of files in library."""
    metadata = load_library_metadata()
    return [(f["original_name"], f["path"]) for f in metadata["files"]]

def delete_from_library(file_path):
    """Delete a file from library."""
    metadata = load_library_metadata()
    metadata["files"] = [f for f in metadata["files"] if f["path"] != file_path]
    save_library_metadata(metadata)
    
    if os.path.exists(file_path):
        os.remove(file_path)
    
    return "🗑️ Deleted from library"

# ============================================================
# Audio Processing Functions
# ============================================================

def separate_stems(audio_path, use_spleeter=True):
    """Separate audio into stems using Spleeter."""
    if audio_path is None:
        return None, None, None, None, "No audio file provided"
    
    try:
        # Create cache directory for this file
        file_hash = str(hash(audio_path))[-8:]
        cache_dir = os.path.join(STEMS_DIR, file_hash)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Check if already separated
        vocals_path = os.path.join(cache_dir, "vocals.wav")
        accompaniment_path = os.path.join(cache_dir, "accompaniment.wav")
        
        if os.path.exists(vocals_path) and os.path.exists(accompaniment_path):
            return vocals_path, accompaniment_path, None, None, "✅ Stems loaded from cache"
        
        if use_spleeter:
            from spleeter.separator import Separator
            from spleeter.audio.adapter import AudioAdapter
            
            separator = Separator("spleeter:2stems")
            audio_loader = AudioAdapter.default()
            
            # Load audio
            waveform, sr = audio_loader.load(audio_path, sample_rate=44100)
            
            # Separate
            prediction = separator.separate(waveform)
            
            # Save stems
            sf.write(vocals_path, prediction["vocals"], 44100)
            sf.write(accompaniment_path, prediction["accompaniment"], 44100)
            
            return vocals_path, accompaniment_path, None, None, "✅ Separation complete"
        else:
            return audio_path, None, None, None, "⚠️ Spleeter not enabled"
            
    except ImportError:
        return audio_path, None, None, None, "⚠️ Spleeter not installed. Using original audio."
    except Exception as e:
        return audio_path, None, None, None, f"❌ Error: {str(e)}"

def mix_stems(vocals_path, vocals_vol, vocals_on,
              accompaniment_path, accompaniment_vol, accompaniment_on,
              drums_path=None, drums_vol=0.5, drums_on=False,
              bass_path=None, bass_vol=0.5, bass_on=False):
    """Mix selected stems with specified volumes."""
    stems = []
    sr = 44100
    
    if vocals_on and vocals_path and os.path.exists(vocals_path):
        y, sr = librosa.load(vocals_path, sr=sr)
        stems.append(y * vocals_vol)
    
    if accompaniment_on and accompaniment_path and os.path.exists(accompaniment_path):
        y, sr = librosa.load(accompaniment_path, sr=sr)
        stems.append(y * accompaniment_vol)
    
    if drums_on and drums_path and os.path.exists(drums_path):
        y, sr = librosa.load(drums_path, sr=sr)
        stems.append(y * drums_vol)
    
    if bass_on and bass_path and os.path.exists(bass_path):
        y, sr = librosa.load(bass_path, sr=sr)
        stems.append(y * bass_vol)
    
    if not stems:
        return None
    
    # Make all stems same length
    max_len = max(len(s) for s in stems)
    stems = [np.pad(s, (0, max_len - len(s))) for s in stems]
    
    # Mix
    mixed = sum(stems)
    mixed = mixed / np.abs(mixed).max()  # Normalize
    
    # Save to temp file
    temp_path = os.path.join(OUTPUT_DIR, "mixed_stems.wav")
    sf.write(temp_path, mixed, sr)
    
    return temp_path

# ============================================================
# Model Functions (Placeholder - integrate with actual model)
# ============================================================

def convert_to_piano(audio_path, composer_style, n_bars, steps_per_beat, 
                     stereo_amp, add_click, progress=gr.Progress()):
    """Convert audio to piano using Pop2Piano model."""
    if audio_path is None:
        return None, None, None, "❌ No audio file provided"
    
    try:
        progress(0.1, desc="Loading model...")
        
        # Import and load model
        import torch
        from omegaconf import OmegaConf
        
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        config = OmegaConf.load(config_path)
        
        from transformer_wrapper import TransformerWrapper
        
        progress(0.3, desc="Loading weights...")
        
        # Check for model checkpoint
        model_path = os.path.join(os.path.dirname(__file__), "model.ckpt")
        
        if os.path.exists(model_path):
            model = TransformerWrapper.load_from_checkpoint(model_path, config=config)
        else:
            # Try to load from Hugging Face
            try:
                from transformers import AutoModel
                # Placeholder for HuggingFace model loading
                return None, None, None, "❌ Model checkpoint not found. Please download the model."
            except:
                return None, None, None, "❌ Model checkpoint not found at: " + model_path
        
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        
        progress(0.5, desc="Processing audio...")
        
        # Generate output paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(audio_path).stem
        midi_path = os.path.join(OUTPUT_DIR, f"{base_name}_{timestamp}.mid")
        mix_path = os.path.join(OUTPUT_DIR, f"{base_name}_{timestamp}_mix.wav")
        
        progress(0.7, desc="Generating piano...")
        
        # Generate piano
        pm, composer, mix_path, midi_path = model.generate(
            audio_path=audio_path,
            composer=composer_style,
            n_bars=n_bars,
            steps_per_beat=steps_per_beat,
            stereo_amp=stereo_amp,
            add_click=add_click,
            save_midi=True,
            save_mix=True,
            midi_path=midi_path,
            mix_path=mix_path
        )
        
        progress(1.0, desc="Done!")
        
        # Get MIDI audio for playback
        midi_audio_path = os.path.join(OUTPUT_DIR, f"{base_name}_{timestamp}_piano.wav")
        midi_y = pm.fluidsynth(44100)
        sf.write(midi_audio_path, midi_y, 44100)
        
        return midi_audio_path, mix_path, midi_path, f"✅ Conversion complete! Style: {composer}"
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n{traceback.format_exc()}"
        return None, None, None, error_msg

def download_from_youtube(url, progress=gr.Progress()):
    """Download audio from YouTube."""
    if not url:
        return None, "❌ No URL provided"
    
    try:
        progress(0.2, desc="Connecting to YouTube...")
        
        from utils.demo import download_youtube
        
        temp_dir = tempfile.mkdtemp()
        progress(0.5, desc="Downloading...")
        
        audio_path = download_youtube(url, temp_dir)
        
        progress(0.8, desc="Adding to library...")
        
        # Add to library
        library_path, msg = add_to_library(audio_path, Path(audio_path).stem)
        
        progress(1.0, desc="Done!")
        
        return library_path, f"✅ Downloaded and added to library"
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# ============================================================
# Build Gradio Interface
# ============================================================

def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(css=CUSTOM_CSS, title="Pop2Piano", theme=gr.themes.Base()) as app:
        
        # Header
        gr.HTML("""
        <div class="app-header">
            <h1>🎹 Pop2Piano</h1>
            <p>Transform any song into a beautiful piano cover</p>
        </div>
        """)
        
        with gr.Tabs() as tabs:
            
            # ================== Tab 1: Convert ==================
            with gr.TabItem("🎵 Convert", id="convert"):
                
                with gr.Row():
                    # Left Column - Input
                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-title">📁 Audio Source</div>')
                        
                        with gr.Tabs() as input_tabs:
                            # Upload Tab
                            with gr.TabItem("Upload"):
                                audio_input = gr.Audio(
                                    label="Upload Audio",
                                    type="filepath",
                                    elem_classes=["upload-area"]
                                )
                                save_to_library = gr.Checkbox(
                                    label="Save to Library",
                                    value=True
                                )
                            
                            # YouTube Tab
                            with gr.TabItem("YouTube"):
                                youtube_url = gr.Textbox(
                                    label="YouTube URL",
                                    placeholder="https://www.youtube.com/watch?v=..."
                                )
                                youtube_btn = gr.Button(
                                    "⬇️ Download",
                                    elem_classes=["primary-btn"]
                                )
                            
                            # Library Tab
                            with gr.TabItem("Library"):
                                library_dropdown = gr.Dropdown(
                                    label="Select from Library",
                                    choices=[],
                                    interactive=True
                                )
                                refresh_library_btn = gr.Button(
                                    "🔄 Refresh",
                                    elem_classes=["secondary-btn"]
                                )
                        
                        gr.HTML('<div class="section-title" style="margin-top: 24px;">⚙️ Settings</div>')
                        
                        composer_dropdown = gr.Dropdown(
                            label="Piano Style",
                            choices=COMPOSER_STYLES,
                            value="composer1",
                            interactive=True
                        )
                        
                        with gr.Row():
                            n_bars = gr.Slider(
                                label="Bars",
                                minimum=1,
                                maximum=8,
                                value=2,
                                step=1
                            )
                            steps_per_beat = gr.Slider(
                                label="Steps/Beat",
                                minimum=1,
                                maximum=4,
                                value=2,
                                step=1
                            )
                        
                        stereo_amp = gr.Slider(
                            label="Mix Balance (Original ↔ Piano)",
                            minimum=0,
                            maximum=1,
                            value=0.5,
                            step=0.05
                        )
                        
                        add_click = gr.Checkbox(
                            label="Add Click Track",
                            value=False
                        )
                        
                        convert_btn = gr.Button(
                            "🎹 Convert to Piano",
                            elem_classes=["primary-btn"],
                            size="lg"
                        )
                    
                    # Right Column - Output
                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-title">🎹 Output</div>')
                        
                        status_text = gr.Markdown("Ready to convert...")
                        
                        with gr.Group(elem_classes=["output-card"]):
                            piano_audio = gr.Audio(
                                label="🎹 Piano Only",
                                type="filepath"
                            )
                            
                            mix_audio = gr.Audio(
                                label="🎧 Stereo Mix",
                                type="filepath"
                            )
                            
                            midi_file = gr.File(
                                label="📄 MIDI File",
                                type="filepath"
                            )
            
            # ================== Tab 2: Stem Separator ==================
            with gr.TabItem("🎚️ Stem Separator", id="stems"):
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-title">🎵 Source Audio</div>')
                        
                        stem_audio_input = gr.Audio(
                            label="Audio to Separate",
                            type="filepath"
                        )
                        
                        separate_btn = gr.Button(
                            "🔀 Separate Stems",
                            elem_classes=["primary-btn"]
                        )
                        
                        stem_status = gr.Markdown("Upload audio and click Separate")
                    
                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-title">🎚️ Stem Controls</div>')
                        
                        # Vocals Control
                        with gr.Group(elem_classes=["stem-control"]):
                            with gr.Row():
                                vocals_on = gr.Checkbox(label="🎤 Vocals", value=True)
                                vocals_vol = gr.Slider(
                                    label="Volume",
                                    minimum=0,
                                    maximum=1,
                                    value=1.0,
                                    step=0.05
                                )
                            vocals_audio = gr.Audio(label="Preview", type="filepath")
                        
                        # Accompaniment Control
                        with gr.Group(elem_classes=["stem-control"]):
                            with gr.Row():
                                accompaniment_on = gr.Checkbox(label="🎸 Accompaniment", value=True)
                                accompaniment_vol = gr.Slider(
                                    label="Volume",
                                    minimum=0,
                                    maximum=1,
                                    value=1.0,
                                    step=0.05
                                )
                            accompaniment_audio = gr.Audio(label="Preview", type="filepath")
                        
                        mix_stems_btn = gr.Button(
                            "🎛️ Mix Selected Stems",
                            elem_classes=["secondary-btn"]
                        )
                        
                        mixed_audio = gr.Audio(
                            label="Mixed Output",
                            type="filepath"
                        )
                        
                        use_for_convert = gr.Button(
                            "➡️ Use for Conversion",
                            elem_classes=["primary-btn"]
                        )
            
            # ================== Tab 3: Library ==================
            with gr.TabItem("📚 Library", id="library"):
                
                gr.HTML('<div class="section-title">📚 Your Audio Library</div>')
                
                gr.HTML("""
                <div class="info-box">
                    <p>All uploaded and downloaded audio files are saved here for easy reuse.</p>
                </div>
                """)
                
                library_gallery = gr.Dataframe(
                    headers=["Name", "Path", "Actions"],
                    datatype=["str", "str", "str"],
                    label="Library Files",
                    interactive=False
                )
                
                with gr.Row():
                    refresh_lib_btn = gr.Button(
                        "🔄 Refresh Library",
                        elem_classes=["secondary-btn"]
                    )
                    
                    delete_file_path = gr.Textbox(
                        label="File path to delete",
                        visible=False
                    )
                    
                    delete_btn = gr.Button(
                        "🗑️ Delete Selected",
                        elem_classes=["secondary-btn"]
                    )
                
                library_audio_preview = gr.Audio(
                    label="Preview",
                    type="filepath"
                )
            
            # ================== Tab 4: Settings ==================
            with gr.TabItem("⚙️ Settings", id="settings"):
                
                gr.HTML('<div class="section-title">⚙️ Application Settings</div>')
                
                with gr.Group():
                    gr.Markdown("### Model Settings")
                    
                    model_path_input = gr.Textbox(
                        label="Model Checkpoint Path",
                        value="model.ckpt",
                        placeholder="Path to model.ckpt"
                    )
                    
                    use_gpu = gr.Checkbox(
                        label="Use GPU (CUDA)",
                        value=True
                    )
                
                with gr.Group():
                    gr.Markdown("### Audio Settings")
                    
                    sample_rate = gr.Dropdown(
                        label="Sample Rate",
                        choices=["22050", "44100", "48000"],
                        value="22050"
                    )
                    
                    default_composer = gr.Dropdown(
                        label="Default Piano Style",
                        choices=COMPOSER_STYLES,
                        value="composer1"
                    )
                
                with gr.Group():
                    gr.Markdown("### Stem Separation")
                    
                    separator_model = gr.Dropdown(
                        label="Separator Model",
                        choices=["spleeter:2stems", "spleeter:4stems", "spleeter:5stems"],
                        value="spleeter:2stems"
                    )
                
                save_settings_btn = gr.Button(
                    "💾 Save Settings",
                    elem_classes=["primary-btn"]
                )
                
                settings_status = gr.Markdown("")
        
        # ============================================================
        # Event Handlers
        # ============================================================
        
        # Helper function to update library dropdown
        def update_library_dropdown():
            files = get_library_files()
            choices = [f[0] for f in files]
            return gr.update(choices=choices)
        
        def get_library_dataframe():
            files = get_library_files()
            return [[f[0], f[1], "🗑️"] for f in files]
        
        # YouTube download
        youtube_btn.click(
            fn=download_from_youtube,
            inputs=[youtube_url],
            outputs=[audio_input, status_text]
        ).then(
            fn=update_library_dropdown,
            outputs=[library_dropdown]
        )
        
        # Refresh library
        refresh_library_btn.click(
            fn=update_library_dropdown,
            outputs=[library_dropdown]
        )
        
        refresh_lib_btn.click(
            fn=get_library_dataframe,
            outputs=[library_gallery]
        )
        
        # Select from library
        def on_library_select(choice):
            files = get_library_files()
            for name, path in files:
                if name == choice:
                    return path
            return None
        
        library_dropdown.change(
            fn=on_library_select,
            inputs=[library_dropdown],
            outputs=[audio_input]
        )
        
        # Save uploaded file to library
        def on_audio_upload(audio_path, save_to_lib):
            if audio_path and save_to_lib:
                add_to_library(audio_path, Path(audio_path).name)
            return audio_path
        
        audio_input.upload(
            fn=on_audio_upload,
            inputs=[audio_input, save_to_library],
            outputs=[audio_input]
        )
        
        # Convert button
        convert_btn.click(
            fn=convert_to_piano,
            inputs=[audio_input, composer_dropdown, n_bars, steps_per_beat, 
                    stereo_amp, add_click],
            outputs=[piano_audio, mix_audio, midi_file, status_text]
        )
        
        # Stem separation
        separate_btn.click(
            fn=separate_stems,
            inputs=[stem_audio_input],
            outputs=[vocals_audio, accompaniment_audio, gr.State(), gr.State(), stem_status]
        )
        
        # Mix stems
        mix_stems_btn.click(
            fn=mix_stems,
            inputs=[vocals_audio, vocals_vol, vocals_on,
                    accompaniment_audio, accompaniment_vol, accompaniment_on],
            outputs=[mixed_audio]
        )
        
        # Use mixed audio for conversion
        def use_mixed_for_conversion(mixed_path):
            return mixed_path
        
        use_for_convert.click(
            fn=use_mixed_for_conversion,
            inputs=[mixed_audio],
            outputs=[audio_input]
        ).then(
            fn=lambda: 0,  # Switch to convert tab
            outputs=[]
        )
        
        # Load library on start
        app.load(
            fn=update_library_dropdown,
            outputs=[library_dropdown]
        )
        
        app.load(
            fn=get_library_dataframe,
            outputs=[library_gallery]
        )
    
    return app

# ============================================================
# Main Entry Point
# ============================================================

if __name__ == "__main__":
    print("🎹 Starting Pop2Piano Web UI...")
    print(f"📁 Library folder: {LIBRARY_DIR}")
    print(f"📂 Output folder: {OUTPUT_DIR}")
    
    app = create_interface()
    
    # Launch with share option for Colab/Kaggle compatibility
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Creates public URL
        inbrowser=True
    )
