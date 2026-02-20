"""
Pop2Piano Benchmark & Comparison Tool

This script allows you to:
1. Compare inference results WITH and WITHOUT piano rules
2. Visualize the differences (piano roll, note distribution)
3. Test Arabic maqamat quantization
4. Measure playability metrics

Usage:
    python benchmark.py --audio sample.mp3
    python benchmark.py --demo  # Uses built-in test
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import Counter
from typing import List, Dict, Tuple

# Import our modules
from piano_rules import (
    PianoNote, apply_piano_rules, split_to_hands,
    simplify_chord, is_playable_chord, PIANO_MIN_NOTE, PIANO_MAX_NOTE,
    MAX_NOTES_TOTAL, extract_melody, extract_bass
)
from arabic_maqamat import (
    get_maqam, get_maqam_scale, detect_maqam, list_all_maqamat,
    quantize_to_maqam, MAQAM_HIJAZ, MAQAM_BAYYATI, MAQAM_RAST
)


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization Functions
# ═══════════════════════════════════════════════════════════════════════════════

def plot_piano_roll(
    notes: List[PianoNote],
    title: str = "Piano Roll",
    ax=None,
    color='blue',
    alpha=0.7
):
    """
    Plot a piano roll visualization.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    
    for note in notes:
        rect = Rectangle(
            (note.onset, note.pitch - 0.4),
            note.offset - note.onset,
            0.8,
            facecolor=color,
            edgecolor='black',
            alpha=alpha,
            linewidth=0.5
        )
        ax.add_patch(rect)
    
    if notes:
        ax.set_xlim(0, max(n.offset for n in notes) + 0.5)
        ax.set_ylim(
            min(n.pitch for n in notes) - 2,
            max(n.pitch for n in notes) + 2
        )
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('MIDI Pitch', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add piano key reference lines
    for pitch in range(21, 109, 12):  # C notes
        ax.axhline(y=pitch, color='gray', linestyle='--', alpha=0.3)
    
    return ax


def plot_comparison(
    before: List[PianoNote],
    after: List[PianoNote],
    title: str = "Before vs After Piano Rules"
):
    """
    Plot side-by-side comparison of notes before and after processing.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    plot_piano_roll(before, "Before (Original)", axes[0], color='red', alpha=0.6)
    plot_piano_roll(after, "After (With Piano Rules)", axes[1], color='green', alpha=0.6)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_note_distribution(
    before: List[PianoNote],
    after: List[PianoNote]
):
    """
    Plot pitch distribution comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pitch histogram
    before_pitches = [n.pitch for n in before]
    after_pitches = [n.pitch for n in after]
    
    bins = range(PIANO_MIN_NOTE, PIANO_MAX_NOTE + 1, 4)
    
    axes[0].hist(before_pitches, bins=bins, alpha=0.5, label='Before', color='red')
    axes[0].hist(after_pitches, bins=bins, alpha=0.5, label='After', color='green')
    axes[0].set_xlabel('MIDI Pitch')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Pitch Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Hand distribution (left vs right)
    left_before = len([n for n in before if n.pitch < 60])
    right_before = len([n for n in before if n.pitch >= 60])
    left_after = len([n for n in after if n.hand == 'left' or n.pitch < 60])
    right_after = len([n for n in after if n.hand == 'right' or n.pitch >= 60])
    
    x = np.arange(2)
    width = 0.35
    
    axes[1].bar(x - width/2, [left_before, right_before], width, label='Before', color='red', alpha=0.6)
    axes[1].bar(x + width/2, [left_after, right_after], width, label='After', color='green', alpha=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Left Hand', 'Right Hand'])
    axes[1].set_ylabel('Number of Notes')
    axes[1].set_title('Hand Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_maqam_comparison(
    original_pitches: List[int],
    maqam_name: str = 'hijaz'
):
    """
    Visualize how notes are quantized to a maqam scale.
    """
    maqam = get_maqam(maqam_name)
    if not maqam:
        print(f"Unknown maqam: {maqam_name}")
        return None
    
    scale = get_maqam_scale(maqam)
    
    # Quantize pitches
    quantized = [quantize_to_maqam(p, maqam) for p in original_pitches]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Before/After pitch classes
    original_classes = [p % 12 for p in original_pitches]
    quantized_classes = [p % 12 for p in quantized]
    
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Original distribution
    orig_counts = Counter(original_classes)
    quant_counts = Counter(quantized_classes)
    
    x = np.arange(12)
    
    axes[0].bar(x, [orig_counts.get(i, 0) for i in range(12)], color='blue', alpha=0.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(note_names)
    axes[0].set_title('Original Pitch Classes')
    axes[0].set_ylabel('Count')
    
    # Highlight scale notes
    for pc in scale:
        axes[0].axvline(x=pc, color='orange', linestyle='--', alpha=0.5)
    
    # Quantized distribution
    colors = ['green' if i in scale else 'gray' for i in range(12)]
    axes[1].bar(x, [quant_counts.get(i, 0) for i in range(12)], color=colors, alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(note_names)
    axes[1].set_title(f'Quantized to {maqam.name_en} ({maqam.name_ar})')
    axes[1].set_ylabel('Count')
    
    fig.suptitle(f'Maqam Quantization: {maqam.name_en}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics Functions
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_playability_metrics(notes: List[PianoNote]) -> Dict:
    """
    Calculate playability metrics for a set of notes.
    """
    if not notes:
        return {'total_notes': 0}
    
    # Group notes by onset time
    time_groups: Dict[float, List[PianoNote]] = {}
    for note in notes:
        onset = round(note.onset, 2)
        if onset not in time_groups:
            time_groups[onset] = []
        time_groups[onset].append(note)
    
    # Metrics
    simultaneous_notes = [len(group) for group in time_groups.values()]
    max_simultaneous = max(simultaneous_notes) if simultaneous_notes else 0
    avg_simultaneous = np.mean(simultaneous_notes) if simultaneous_notes else 0
    
    # Check hand playability
    playable_chords = 0
    unplayable_chords = 0
    
    for group in time_groups.values():
        pitches = [n.pitch for n in group]
        left, right = split_to_hands(pitches)
        
        if is_playable_chord(left, 'left') and is_playable_chord(right, 'right'):
            playable_chords += 1
        else:
            unplayable_chords += 1
    
    # Pitch range
    all_pitches = [n.pitch for n in notes]
    pitch_range = max(all_pitches) - min(all_pitches)
    
    # Notes in piano range
    in_range = sum(1 for p in all_pitches if PIANO_MIN_NOTE <= p <= PIANO_MAX_NOTE)
    out_of_range = len(all_pitches) - in_range
    
    return {
        'total_notes': len(notes),
        'total_chords': len(time_groups),
        'max_simultaneous': max_simultaneous,
        'avg_simultaneous': round(avg_simultaneous, 2),
        'playable_chords': playable_chords,
        'unplayable_chords': unplayable_chords,
        'playability_ratio': round(playable_chords / max(1, len(time_groups)), 2),
        'pitch_range': pitch_range,
        'min_pitch': min(all_pitches),
        'max_pitch': max(all_pitches),
        'notes_in_range': in_range,
        'notes_out_of_range': out_of_range,
    }


def print_metrics_comparison(before_metrics: Dict, after_metrics: Dict):
    """
    Print a formatted comparison of metrics.
    """
    print("\n" + "=" * 60)
    print("📊 PLAYABILITY METRICS COMPARISON")
    print("=" * 60)
    
    headers = ['Metric', 'Before', 'After', 'Change']
    
    rows = [
        ('Total Notes', before_metrics['total_notes'], after_metrics['total_notes']),
        ('Total Chords', before_metrics['total_chords'], after_metrics['total_chords']),
        ('Max Simultaneous', before_metrics['max_simultaneous'], after_metrics['max_simultaneous']),
        ('Avg Simultaneous', before_metrics['avg_simultaneous'], after_metrics['avg_simultaneous']),
        ('Playable Chords', before_metrics['playable_chords'], after_metrics['playable_chords']),
        ('Unplayable Chords', before_metrics['unplayable_chords'], after_metrics['unplayable_chords']),
        ('Playability %', f"{before_metrics['playability_ratio']*100:.0f}%", 
                          f"{after_metrics['playability_ratio']*100:.0f}%"),
        ('Pitch Range', before_metrics['pitch_range'], after_metrics['pitch_range']),
        ('Notes Out of Range', before_metrics['notes_out_of_range'], after_metrics['notes_out_of_range']),
    ]
    
    print(f"\n{'Metric':<25} {'Before':>10} {'After':>10} {'Change':>10}")
    print("-" * 60)
    
    for row in rows:
        name, before, after = row
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            change = after - before
            if change > 0:
                change_str = f"+{change}"
            else:
                change_str = str(change)
            
            # Color coding
            if 'Unplayable' in name or 'Out of Range' in name:
                improvement = change < 0
            else:
                improvement = change >= 0
            
            indicator = "✅" if improvement else "❌"
        else:
            change_str = "-"
            indicator = ""
        
        print(f"{name:<25} {str(before):>10} {str(after):>10} {change_str:>8} {indicator}")
    
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# Demo / Test Functions
# ═══════════════════════════════════════════════════════════════════════════════

def generate_test_notes(n_notes: int = 50, chaos_level: float = 0.5) -> List[PianoNote]:
    """
    Generate random test notes to simulate model output.
    
    Args:
        n_notes: Number of notes to generate
        chaos_level: 0 = clean, 1 = very chaotic (many out-of-range, clusters)
    """
    np.random.seed(42)
    notes = []
    
    current_time = 0
    
    for i in range(n_notes):
        # Random pitch (with some chaos)
        if np.random.random() < chaos_level * 0.3:
            # Out of range notes
            pitch = np.random.randint(10, 120)
        else:
            # Normal range
            pitch = np.random.randint(36, 96)
        
        # Duration
        duration = np.random.uniform(0.1, 0.5)
        
        # Time advance (with some clustering)
        if np.random.random() < chaos_level * 0.4:
            # Cluster - same onset as previous
            time_advance = 0
        else:
            time_advance = np.random.uniform(0.1, 0.4)
        
        current_time += time_advance
        
        # Velocity
        velocity = np.random.randint(40, 100)
        
        notes.append(PianoNote(
            pitch=pitch,
            onset=current_time,
            offset=current_time + duration,
            velocity=velocity
        ))
    
    return notes


def run_demo():
    """
    Run a demonstration of the piano rules system.
    """
    print("\n" + "🎹" * 20)
    print("  POP2PIANO - PIANO RULES DEMO")
    print("🎹" * 20)
    
    # Generate test notes
    print("\n📝 Generating test notes (simulating model output)...")
    original_notes = generate_test_notes(n_notes=60, chaos_level=0.6)
    print(f"   Generated {len(original_notes)} notes")
    
    # Calculate before metrics
    before_metrics = calculate_playability_metrics(original_notes)
    
    # Apply piano rules
    print("\n🎼 Applying piano rules...")
    processed_notes = apply_piano_rules(
        notes=original_notes,
        scale_pitches=None,
        simplify=True,
        quantize=True,
        humanize=False
    )
    print(f"   Processed: {len(original_notes)} → {len(processed_notes)} notes")
    
    # Calculate after metrics
    after_metrics = calculate_playability_metrics(processed_notes)
    
    # Print comparison
    print_metrics_comparison(before_metrics, after_metrics)
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    
    # Piano roll comparison
    fig1 = plot_comparison(original_notes, processed_notes)
    fig1.savefig('benchmark_piano_roll.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: benchmark_piano_roll.png")
    
    # Note distribution
    fig2 = plot_note_distribution(original_notes, processed_notes)
    fig2.savefig('benchmark_distribution.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: benchmark_distribution.png")
    
    # Maqam demo
    print("\n🎵 Testing Arabic Maqamat...")
    print("\nAvailable Maqamat:")
    for m in list_all_maqamat()[:8]:
        print(f"   • {m['name_en']} ({m['name_ar']}) - {m['mood'][:30]}...")
    
    # Test maqam detection
    pitches = [n.pitch for n in original_notes]
    detections = detect_maqam(pitches, threshold=0.5)
    print(f"\n🔍 Maqam Detection Results:")
    for name, conf in detections[:3]:
        print(f"   • {name}: {conf*100:.0f}% confidence")
    
    # Maqam quantization visualization
    fig3 = plot_maqam_comparison(pitches, 'hijaz')
    if fig3:
        fig3.savefig('benchmark_maqam.png', dpi=150, bbox_inches='tight')
        print("   ✅ Saved: benchmark_maqam.png")
    
    print("\n" + "=" * 60)
    print("✅ DEMO COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("   📊 benchmark_piano_roll.png - Piano roll before/after")
    print("   📊 benchmark_distribution.png - Note distribution")
    print("   📊 benchmark_maqam.png - Maqam quantization demo")
    
    # Show plots if interactive
    try:
        plt.show()
    except:
        pass
    
    return original_notes, processed_notes


def run_maqam_demo():
    """
    Interactive demo for Arabic maqamat.
    """
    print("\n" + "🎵" * 20)
    print("  ARABIC MAQAMAT DEMO")
    print("🎵" * 20)
    
    maqamat = list_all_maqamat()
    
    print("\n📜 All Available Maqamat:\n")
    for m in maqamat:
        print(f"   {m['name_en']:<15} {m['name_ar']:<10} Root: {m['root']:<3} | {m['mood']}")
    
    # Create scale comparison chart
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    for idx, m_info in enumerate(maqamat[:14]):
        maqam = get_maqam(m_info['name_en'])
        if not maqam or idx >= 14:
            continue
        
        scale = get_maqam_scale(maqam)
        
        # Create bar chart for scale
        colors = ['green' if i in scale else 'lightgray' for i in range(12)]
        axes[idx].bar(range(12), [1 if i in scale else 0.2 for i in range(12)], color=colors)
        axes[idx].set_xticks(range(12))
        axes[idx].set_xticklabels(note_names, fontsize=8)
        axes[idx].set_title(f"{m_info['name_en']}\n{m_info['name_ar']}", fontsize=10)
        axes[idx].set_ylim(0, 1.2)
        axes[idx].set_yticks([])
    
    # Hide unused subplots
    for idx in range(14, 16):
        axes[idx].axis('off')
    
    fig.suptitle('Arabic Maqamat Scale Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig('maqamat_scales.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved: maqamat_scales.png")
    
    try:
        plt.show()
    except:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Pop2Piano Benchmark Tool')
    parser.add_argument('--demo', action='store_true', help='Run demo with generated notes')
    parser.add_argument('--maqam-demo', action='store_true', help='Run Arabic maqamat demo')
    parser.add_argument('--audio', type=str, help='Path to audio file for inference comparison')
    parser.add_argument('--chaos', type=float, default=0.6, help='Chaos level for demo (0-1)')
    
    args = parser.parse_args()
    
    if args.maqam_demo:
        run_maqam_demo()
    elif args.demo or not args.audio:
        run_demo()
    else:
        # TODO: Full inference comparison with actual model
        print("Full inference comparison requires model weights.")
        print("Running demo mode instead...")
        run_demo()


if __name__ == "__main__":
    main()
