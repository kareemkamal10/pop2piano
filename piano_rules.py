"""
Piano Rules Module - Rule-based Piano Arrangement

This module contains musical rules for piano arrangement that help the model
produce better results even on unseen music. These rules are based on:
1. Piano playing techniques and constraints
2. Music theory fundamentals
3. Arrangement best practices

These rules act as a "fallback" or "enhancement" layer alongside the ML model.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

# Piano range (MIDI note numbers)
PIANO_MIN_NOTE = 21   # A0
PIANO_MAX_NOTE = 108  # C8

# Hand ranges (typical)
LEFT_HAND_MIN = 21    # A0
LEFT_HAND_MAX = 64    # E4 (middle range)
RIGHT_HAND_MIN = 48   # C3 
RIGHT_HAND_MAX = 108  # C8

# Maximum simultaneous notes per hand
MAX_NOTES_LEFT_HAND = 5
MAX_NOTES_RIGHT_HAND = 5
MAX_NOTES_TOTAL = 10

# Maximum stretch (interval) between fingers (in semitones)
MAX_HAND_STRETCH = 12  # One octave is comfortable max

# Velocity ranges
VELOCITY_MIN = 20
VELOCITY_MAX = 127
VELOCITY_DEFAULT = 77


# ═══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PianoNote:
    """Represents a single piano note."""
    pitch: int           # MIDI note number (21-108)
    onset: float         # Start time in seconds
    offset: float        # End time in seconds
    velocity: int        # MIDI velocity (0-127)
    hand: str = 'auto'   # 'left', 'right', or 'auto'


@dataclass
class Chord:
    """Represents a chord (multiple simultaneous notes)."""
    notes: List[int]     # List of MIDI pitches
    root: int            # Root note
    quality: str         # 'major', 'minor', 'dim', 'aug', etc.


# ═══════════════════════════════════════════════════════════════════════════════
# Piano Constraint Functions
# ═══════════════════════════════════════════════════════════════════════════════

def clamp_to_piano_range(pitch: int) -> int:
    """Ensure pitch is within piano range, transposing by octaves if needed."""
    while pitch < PIANO_MIN_NOTE:
        pitch += 12
    while pitch > PIANO_MAX_NOTE:
        pitch -= 12
    return pitch


def is_playable_chord(notes: List[int], hand: str = 'right') -> bool:
    """
    Check if a set of notes can be played with one hand.
    
    Args:
        notes: List of MIDI pitches
        hand: 'left' or 'right'
        
    Returns:
        True if chord is playable by one hand
    """
    if len(notes) == 0:
        return True
    if len(notes) > MAX_NOTES_RIGHT_HAND:
        return False
    
    sorted_notes = sorted(notes)
    stretch = sorted_notes[-1] - sorted_notes[0]
    
    return stretch <= MAX_HAND_STRETCH


def split_to_hands(notes: List[int]) -> Tuple[List[int], List[int]]:
    """
    Split notes between left and right hands.
    
    Rule: Lower notes go to left hand, higher to right.
    Split point is around middle C (60).
    
    Args:
        notes: List of MIDI pitches
        
    Returns:
        (left_hand_notes, right_hand_notes)
    """
    if len(notes) == 0:
        return [], []
    
    sorted_notes = sorted(notes)
    split_point = 60  # Middle C
    
    # Adjust split point if all notes are on one side
    if sorted_notes[0] >= split_point:
        split_point = sorted_notes[0]
    elif sorted_notes[-1] < split_point:
        split_point = sorted_notes[-1] + 1
    
    left_hand = [n for n in sorted_notes if n < split_point]
    right_hand = [n for n in sorted_notes if n >= split_point]
    
    # Ensure each hand doesn't have too many notes
    while len(left_hand) > MAX_NOTES_LEFT_HAND:
        # Move middle note to right hand
        note = left_hand.pop()
        right_hand.insert(0, note)
    
    while len(right_hand) > MAX_NOTES_RIGHT_HAND:
        # Move lowest note to left hand
        note = right_hand.pop(0)
        left_hand.append(note)
    
    return left_hand, right_hand


def simplify_chord(notes: List[int], max_notes: int = 4) -> List[int]:
    """
    Simplify a chord to be more playable.
    
    Keeps: root, third, fifth, seventh (in order of importance)
    
    Args:
        notes: List of MIDI pitches
        max_notes: Maximum notes to keep
        
    Returns:
        Simplified list of pitches
    """
    if len(notes) <= max_notes:
        return notes
    
    sorted_notes = sorted(notes)
    
    # Keep lowest (bass), highest (melody), and distribute rest
    if max_notes == 1:
        return [sorted_notes[-1]]  # Keep melody
    elif max_notes == 2:
        return [sorted_notes[0], sorted_notes[-1]]  # Bass + melody
    else:
        # Keep bass, melody, and evenly distributed middle notes
        result = [sorted_notes[0]]  # Bass
        
        middle_count = max_notes - 2
        middle_notes = sorted_notes[1:-1]
        
        if middle_count > 0 and len(middle_notes) > 0:
            step = len(middle_notes) / middle_count
            for i in range(middle_count):
                idx = int(i * step)
                if idx < len(middle_notes):
                    result.append(middle_notes[idx])
        
        result.append(sorted_notes[-1])  # Melody
        return sorted(result)


def quantize_to_scale(pitch: int, scale_pitches: List[int]) -> int:
    """
    Quantize a pitch to the nearest note in a scale.
    
    Args:
        pitch: MIDI pitch to quantize
        scale_pitches: List of pitches in the scale (one octave, 0-11)
        
    Returns:
        Quantized pitch
    """
    pitch_class = pitch % 12
    octave = pitch // 12
    
    # Find nearest scale pitch
    min_distance = 12
    nearest = pitch_class
    
    for scale_pitch in scale_pitches:
        # Check both directions (with wrapping)
        dist = min(
            abs(pitch_class - scale_pitch),
            12 - abs(pitch_class - scale_pitch)
        )
        if dist < min_distance:
            min_distance = dist
            nearest = scale_pitch
    
    return octave * 12 + nearest


# ═══════════════════════════════════════════════════════════════════════════════
# Melody Extraction Rules
# ═══════════════════════════════════════════════════════════════════════════════

def extract_melody(notes: List[PianoNote]) -> List[PianoNote]:
    """
    Extract melody line from a set of notes.
    
    Rule: Melody is typically the highest note at each time point.
    
    Args:
        notes: List of PianoNote objects
        
    Returns:
        Melody notes only
    """
    if not notes:
        return []
    
    # Group notes by onset time
    time_groups: Dict[float, List[PianoNote]] = {}
    for note in notes:
        onset = round(note.onset, 3)  # Quantize time
        if onset not in time_groups:
            time_groups[onset] = []
        time_groups[onset].append(note)
    
    # Take highest note at each time point
    melody = []
    for onset in sorted(time_groups.keys()):
        group = time_groups[onset]
        highest = max(group, key=lambda n: n.pitch)
        melody.append(highest)
    
    return melody


def extract_bass(notes: List[PianoNote]) -> List[PianoNote]:
    """
    Extract bass line from a set of notes.
    
    Rule: Bass is typically the lowest note at each time point.
    """
    if not notes:
        return []
    
    # Group notes by onset time
    time_groups: Dict[float, List[PianoNote]] = {}
    for note in notes:
        onset = round(note.onset, 3)
        if onset not in time_groups:
            time_groups[onset] = []
        time_groups[onset].append(note)
    
    # Take lowest note at each time point
    bass = []
    for onset in sorted(time_groups.keys()):
        group = time_groups[onset]
        lowest = min(group, key=lambda n: n.pitch)
        bass.append(lowest)
    
    return bass


# ═══════════════════════════════════════════════════════════════════════════════
# Chord Voicing Rules
# ═══════════════════════════════════════════════════════════════════════════════

def create_piano_voicing(
    root: int,
    quality: str = 'major',
    inversion: int = 0,
    spread: bool = False
) -> Tuple[List[int], List[int]]:
    """
    Create a piano voicing for a chord.
    
    Args:
        root: Root note (MIDI pitch)
        quality: 'major', 'minor', 'dim', 'aug', '7', 'm7', 'maj7'
        inversion: 0 = root position, 1 = first inversion, etc.
        spread: If True, spread chord across both hands
        
    Returns:
        (left_hand_notes, right_hand_notes)
    """
    # Build chord intervals
    intervals = {
        'major': [0, 4, 7],
        'minor': [0, 3, 7],
        'dim': [0, 3, 6],
        'aug': [0, 4, 8],
        '7': [0, 4, 7, 10],
        'm7': [0, 3, 7, 10],
        'maj7': [0, 4, 7, 11],
        'sus2': [0, 2, 7],
        'sus4': [0, 5, 7],
    }
    
    chord_intervals = intervals.get(quality, intervals['major'])
    
    # Build chord notes
    chord_notes = [root + interval for interval in chord_intervals]
    
    # Apply inversion
    for _ in range(inversion % len(chord_notes)):
        chord_notes[0] += 12
        chord_notes.sort()
    
    # Ensure notes are in piano range
    chord_notes = [clamp_to_piano_range(n) for n in chord_notes]
    
    if spread:
        # Spread voicing: bass note in left hand, rest in right
        left_hand = [chord_notes[0]]
        right_hand = chord_notes[1:]
    else:
        # Compact voicing: all in one hand position
        return split_to_hands(chord_notes)
    
    return left_hand, right_hand


# ═══════════════════════════════════════════════════════════════════════════════
# Rhythm Rules
# ═══════════════════════════════════════════════════════════════════════════════

def quantize_rhythm(
    onset: float,
    grid: float = 0.125,  # 1/8 note at 120 BPM
    swing: float = 0.0    # 0.0 = straight, 0.5 = heavy swing
) -> float:
    """
    Quantize onset time to a rhythmic grid.
    
    Args:
        onset: Original onset time in seconds
        grid: Grid resolution in seconds
        swing: Swing amount (0-1)
        
    Returns:
        Quantized onset time
    """
    # Basic quantization
    quantized = round(onset / grid) * grid
    
    # Apply swing to off-beats
    if swing > 0:
        beat_position = (quantized / grid) % 2
        if beat_position == 1:  # Off-beat
            quantized += grid * swing * 0.5
    
    return quantized


def humanize_velocity(velocity: int, amount: float = 0.1) -> int:
    """
    Add human-like variation to velocity.
    
    Args:
        velocity: Original velocity
        amount: Variation amount (0-1)
        
    Returns:
        Humanized velocity
    """
    variation = int(velocity * amount * (np.random.random() * 2 - 1))
    result = velocity + variation
    return max(VELOCITY_MIN, min(VELOCITY_MAX, result))


# ═══════════════════════════════════════════════════════════════════════════════
# Post-Processing Rules
# ═══════════════════════════════════════════════════════════════════════════════

def apply_piano_rules(
    notes: List[PianoNote],
    scale_pitches: Optional[List[int]] = None,
    simplify: bool = True,
    quantize: bool = True,
    humanize: bool = True
) -> List[PianoNote]:
    """
    Apply all piano rules to a list of notes.
    
    This is the main function to call for post-processing model output.
    
    Args:
        notes: List of PianoNote objects from model
        scale_pitches: Optional scale to quantize to (for Arabic maqamat)
        simplify: Whether to simplify complex chords
        quantize: Whether to quantize rhythm
        humanize: Whether to add humanization
        
    Returns:
        Processed list of PianoNote objects
    """
    if not notes:
        return []
    
    processed = []
    
    # Group notes by onset time for chord processing
    time_groups: Dict[float, List[PianoNote]] = {}
    for note in notes:
        onset = round(note.onset, 3)
        if onset not in time_groups:
            time_groups[onset] = []
        time_groups[onset].append(note)
    
    for onset in sorted(time_groups.keys()):
        group = time_groups[onset]
        pitches = [n.pitch for n in group]
        
        # 1. Clamp to piano range
        pitches = [clamp_to_piano_range(p) for p in pitches]
        
        # 2. Quantize to scale if provided
        if scale_pitches:
            pitches = [quantize_to_scale(p, scale_pitches) for p in pitches]
        
        # 3. Simplify if too many notes
        if simplify and len(pitches) > MAX_NOTES_TOTAL:
            pitches = simplify_chord(pitches, MAX_NOTES_TOTAL)
        
        # 4. Split to hands
        left_hand, right_hand = split_to_hands(pitches)
        
        # 5. Create processed notes
        for i, pitch in enumerate(left_hand + right_hand):
            original = group[min(i, len(group) - 1)]
            
            new_onset = onset
            if quantize:
                new_onset = quantize_rhythm(onset)
            
            new_velocity = original.velocity
            if humanize:
                new_velocity = humanize_velocity(original.velocity)
            
            processed.append(PianoNote(
                pitch=pitch,
                onset=new_onset,
                offset=original.offset,
                velocity=new_velocity,
                hand='left' if pitch in left_hand else 'right'
            ))
    
    return processed


# ═══════════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════════

def notes_to_midi_array(notes: List[PianoNote], beatstep: np.ndarray) -> np.ndarray:
    """
    Convert PianoNote objects to MIDI array format used by tokenizer.
    
    Returns:
        Array of shape (N, 4): [onset_idx, offset_idx, pitch, velocity]
    """
    if not notes:
        return np.array([]).reshape(0, 4)
    
    result = []
    for note in notes:
        # Find nearest beat indices
        onset_idx = np.searchsorted(beatstep, note.onset)
        offset_idx = np.searchsorted(beatstep, note.offset)
        
        result.append([onset_idx, offset_idx, note.pitch, note.velocity])
    
    return np.array(result, dtype=np.int32)


def midi_array_to_notes(midi_array: np.ndarray, beatstep: np.ndarray) -> List[PianoNote]:
    """
    Convert MIDI array format to PianoNote objects.
    
    Args:
        midi_array: Array of shape (N, 4): [onset_idx, offset_idx, pitch, velocity]
        beatstep: Beat timestamps
        
    Returns:
        List of PianoNote objects
    """
    notes = []
    for row in midi_array:
        onset_idx, offset_idx, pitch, velocity = row
        
        onset = beatstep[min(onset_idx, len(beatstep) - 1)]
        offset = beatstep[min(offset_idx, len(beatstep) - 1)]
        
        notes.append(PianoNote(
            pitch=int(pitch),
            onset=float(onset),
            offset=float(offset),
            velocity=int(velocity)
        ))
    
    return notes
