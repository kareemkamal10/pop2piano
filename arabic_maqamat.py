"""
Arabic Maqamat (المقامات العربية) Module

This module provides support for Arabic musical scales (Maqamat) in piano arrangement.
Each Maqam has a unique character and emotional quality.

Maqamat use quarter tones in traditional Arabic music, but for piano we approximate
using the closest Western semitones or common adaptations.

Reference: Arabic music typically uses 24 notes per octave (quarter tones),
but piano only has 12. We provide both strict and adapted versions.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Quarter Tone Approximation
# ═══════════════════════════════════════════════════════════════════════════════

# In Arabic music, E half-flat and B half-flat are common
# On piano, we typically use the natural note or alternate between natural and flat
# depending on context

class QuarterToneApproximation(Enum):
    """How to handle quarter tones on piano."""
    LOWER = 'lower'      # Use the flat/lower semitone
    HIGHER = 'higher'    # Use the natural/higher semitone  
    ALTERNATE = 'alt'    # Alternate based on direction
    ORNAMENT = 'orn'     # Use ornaments (grace notes) to suggest quarter tone


# ═══════════════════════════════════════════════════════════════════════════════
# Maqam Definitions
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Maqam:
    """
    Represents an Arabic Maqam (musical scale/mode).
    
    Attributes:
        name_ar: Arabic name
        name_en: English transliteration
        intervals: Intervals in semitones from root (for piano adaptation)
        quarter_tone_positions: Positions where quarter tones occur (0-indexed)
        root_note: Default root note (0-11, where 0=C)
        mood: Emotional character description
        common_modulations: Related maqamat for modulation
    """
    name_ar: str
    name_en: str
    intervals: List[int]
    quarter_tone_positions: List[int]
    root_note: int
    mood: str
    common_modulations: List[str]


# ─────────────────────────────────────────────────────────────────────────────
# Primary Maqamat (الأجناس الأساسية)
# ─────────────────────────────────────────────────────────────────────────────

MAQAM_RAST = Maqam(
    name_ar="راست",
    name_en="Rast",
    # Original: C D E♭ F G A B♭ C (E is quarter flat, B is quarter flat)
    # Piano adaptation: Major scale feel with b7
    intervals=[0, 2, 4, 5, 7, 9, 10],  # C D E F G A Bb
    quarter_tone_positions=[2, 6],  # E and B positions
    root_note=0,  # C
    mood="فرحة متوازنة - Balanced joy, celebratory",
    common_modulations=["Bayyati", "Sikah", "Nahawand"]
)

MAQAM_BAYYATI = Maqam(
    name_ar="بياتي",
    name_en="Bayyati",
    # Original: D E♭ F G A B♭ C D (E is quarter flat)
    # Piano adaptation: Minor-ish with flat 2
    intervals=[0, 1, 3, 5, 7, 8, 10],  # D Eb F G A Bb C
    quarter_tone_positions=[1],  # E position
    root_note=2,  # D
    mood="حنين وشجن - Nostalgic, melancholic",
    common_modulations=["Rast", "Saba", "Husayni"]
)

MAQAM_HIJAZ = Maqam(
    name_ar="حجاز",
    name_en="Hijaz",
    # D E♭ F# G A B♭ C D - distinctive augmented 2nd
    intervals=[0, 1, 4, 5, 7, 8, 10],  # D Eb F# G A Bb C
    quarter_tone_positions=[],  # No quarter tones in standard Hijaz
    root_note=2,  # D
    mood="شرقي مميز - Distinctly Eastern, mystical",
    common_modulations=["Bayyati", "Hijaz Kar", "Nahawand"]
)

MAQAM_SABA = Maqam(
    name_ar="صبا",
    name_en="Saba",
    # D E♭ F G♭ A B♭ C D - very characteristic
    intervals=[0, 1, 3, 4, 7, 8, 10],  # D Eb F Gb A Bb C
    quarter_tone_positions=[1],  # E position
    root_note=2,  # D
    mood="حزين عميق - Deep sorrow, contemplative",
    common_modulations=["Bayyati", "Hijaz"]
)

MAQAM_NAHAWAND = Maqam(
    name_ar="نهاوند",
    name_en="Nahawand",
    # C D E♭ F G A♭ B C - similar to harmonic minor
    intervals=[0, 2, 3, 5, 7, 8, 11],  # C D Eb F G Ab B
    quarter_tone_positions=[],
    root_note=0,  # C
    mood="رومانسي حزين - Romantic, bittersweet",
    common_modulations=["Rast", "Hijaz", "Kurd"]
)

MAQAM_SIKAH = Maqam(
    name_ar="سيكاه",
    name_en="Sikah",
    # E♭ F G A B♭ C D E♭ (E is quarter flat at root)
    # Piano: Start on Eb
    intervals=[0, 2, 4, 6, 7, 9, 11],  # Eb F G A Bb C D
    quarter_tone_positions=[0],  # Root is quarter tone
    root_note=3,  # Eb
    mood="روحاني هادئ - Spiritual, meditative",
    common_modulations=["Rast", "Huzam"]
)

MAQAM_AJAM = Maqam(
    name_ar="عجم",
    name_en="Ajam",
    # C D E F G A B C - equivalent to Western Major scale
    intervals=[0, 2, 4, 5, 7, 9, 11],
    quarter_tone_positions=[],
    root_note=0,  # C
    mood="فرحة صافية - Pure joy, triumphant",
    common_modulations=["Jiharkah", "Shawq Afza"]
)

MAQAM_KURD = Maqam(
    name_ar="كرد",
    name_en="Kurd",
    # D E♭ F G A B♭ C D - similar to Phrygian
    intervals=[0, 1, 3, 5, 7, 8, 10],
    quarter_tone_positions=[],
    root_note=2,  # D
    mood="قوي حاسم - Strong, decisive",
    common_modulations=["Hijaz", "Nahawand"]
)

# ─────────────────────────────────────────────────────────────────────────────
# Secondary Maqamat (فروع المقامات)
# ─────────────────────────────────────────────────────────────────────────────

MAQAM_HIJAZ_KAR = Maqam(
    name_ar="حجاز كار",
    name_en="Hijaz Kar",
    # C D♭ E F G A♭ B C
    intervals=[0, 1, 4, 5, 7, 8, 11],
    quarter_tone_positions=[],
    root_note=0,  # C
    mood="درامي مكثف - Dramatic, intense",
    common_modulations=["Hijaz", "Nahawand"]
)

MAQAM_HUSAYNI = Maqam(
    name_ar="حسيني",
    name_en="Husayni",
    # A Bayyati variant starting on A
    intervals=[0, 1, 3, 5, 7, 8, 10],
    quarter_tone_positions=[1],
    root_note=9,  # A
    mood="رقيق عاطفي - Tender, emotional",
    common_modulations=["Bayyati", "Rast"]
)

MAQAM_JIHARKAH = Maqam(
    name_ar="جهاركاه",
    name_en="Jiharkah",
    # F G A B♭ C D E F
    intervals=[0, 2, 4, 5, 7, 9, 11],
    quarter_tone_positions=[],
    root_note=5,  # F
    mood="منطلق حر - Free-spirited",
    common_modulations=["Ajam", "Rast"]
)

MAQAM_NAWA_ATHAR = Maqam(
    name_ar="نوا أثر",
    name_en="Nawa Athar",
    # C D E♭ F# G A♭ B C
    intervals=[0, 2, 3, 6, 7, 8, 11],
    quarter_tone_positions=[],
    root_note=0,
    mood="غامض ساحر - Mysterious, enchanting",
    common_modulations=["Hijaz", "Nikriz"]
)

MAQAM_NIKRIZ = Maqam(
    name_ar="نكريز",
    name_en="Nikriz",
    # C D E♭ F# G A B♭ C
    intervals=[0, 2, 3, 6, 7, 9, 10],
    quarter_tone_positions=[],
    root_note=0,
    mood="قوي ثابت - Powerful, resolute",
    common_modulations=["Nawa Athar", "Hijaz"]
)

MAQAM_ATHAR_KURD = Maqam(
    name_ar="أثر كرد",
    name_en="Athar Kurd",
    # D E♭ F# G A B♭ C D
    intervals=[0, 1, 4, 5, 7, 8, 10],
    quarter_tone_positions=[],
    root_note=2,
    mood="مثير درامي - Exciting, dramatic",
    common_modulations=["Kurd", "Hijaz"]
)


# ═══════════════════════════════════════════════════════════════════════════════
# Maqam Registry
# ═══════════════════════════════════════════════════════════════════════════════

MAQAMAT: Dict[str, Maqam] = {
    # Primary
    "rast": MAQAM_RAST,
    "راست": MAQAM_RAST,
    "bayyati": MAQAM_BAYYATI,
    "bayati": MAQAM_BAYYATI,
    "بياتي": MAQAM_BAYYATI,
    "hijaz": MAQAM_HIJAZ,
    "حجاز": MAQAM_HIJAZ,
    "saba": MAQAM_SABA,
    "صبا": MAQAM_SABA,
    "nahawand": MAQAM_NAHAWAND,
    "نهاوند": MAQAM_NAHAWAND,
    "sikah": MAQAM_SIKAH,
    "سيكاه": MAQAM_SIKAH,
    "ajam": MAQAM_AJAM,
    "عجم": MAQAM_AJAM,
    "kurd": MAQAM_KURD,
    "كرد": MAQAM_KURD,
    
    # Secondary
    "hijaz_kar": MAQAM_HIJAZ_KAR,
    "حجاز كار": MAQAM_HIJAZ_KAR,
    "husayni": MAQAM_HUSAYNI,
    "حسيني": MAQAM_HUSAYNI,
    "jiharkah": MAQAM_JIHARKAH,
    "جهاركاه": MAQAM_JIHARKAH,
    "nawa_athar": MAQAM_NAWA_ATHAR,
    "نوا أثر": MAQAM_NAWA_ATHAR,
    "nikriz": MAQAM_NIKRIZ,
    "نكريز": MAQAM_NIKRIZ,
    "athar_kurd": MAQAM_ATHAR_KURD,
    "أثر كرد": MAQAM_ATHAR_KURD,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Maqam Functions
# ═══════════════════════════════════════════════════════════════════════════════

def get_maqam(name: str) -> Optional[Maqam]:
    """
    Get a Maqam by name (Arabic or English).
    
    Args:
        name: Maqam name (case-insensitive)
        
    Returns:
        Maqam object or None if not found
    """
    return MAQAMAT.get(name.lower())


def get_maqam_scale(maqam: Maqam, root: int = None) -> List[int]:
    """
    Get the scale pitches (0-11) for a maqam.
    
    Args:
        maqam: Maqam object
        root: Optional root note override (0-11)
        
    Returns:
        List of pitch classes (0-11)
    """
    root = root if root is not None else maqam.root_note
    return [(root + interval) % 12 for interval in maqam.intervals]


def get_full_scale(maqam: Maqam, start_octave: int = 4, num_octaves: int = 2) -> List[int]:
    """
    Get full MIDI note numbers for a maqam across octaves.
    
    Args:
        maqam: Maqam object
        start_octave: Starting octave (4 = middle C octave)
        num_octaves: Number of octaves to span
        
    Returns:
        List of MIDI note numbers
    """
    scale_pitches = get_maqam_scale(maqam)
    notes = []
    
    for octave in range(start_octave, start_octave + num_octaves):
        for pitch_class in scale_pitches:
            midi_note = octave * 12 + pitch_class
            if 21 <= midi_note <= 108:  # Piano range
                notes.append(midi_note)
    
    return sorted(notes)


def detect_maqam(pitches: List[int], threshold: float = 0.7) -> List[Tuple[str, float]]:
    """
    Attempt to detect which maqam a set of pitches belongs to.
    
    Args:
        pitches: List of MIDI note numbers
        threshold: Minimum match ratio to consider
        
    Returns:
        List of (maqam_name, confidence) tuples, sorted by confidence
    """
    if not pitches:
        return []
    
    # Convert to pitch classes
    pitch_classes = set(p % 12 for p in pitches)
    
    results = []
    
    # Check unique maqamat only
    checked = set()
    for name, maqam in MAQAMAT.items():
        if maqam.name_en in checked:
            continue
        checked.add(maqam.name_en)
        
        maqam_pitches = set(get_maqam_scale(maqam))
        
        # Calculate overlap
        if len(pitch_classes) == 0:
            continue
            
        matches = len(pitch_classes & maqam_pitches)
        confidence = matches / len(pitch_classes)
        
        if confidence >= threshold:
            results.append((maqam.name_en, confidence))
    
    return sorted(results, key=lambda x: x[1], reverse=True)


def quantize_to_maqam(
    pitch: int,
    maqam: Maqam,
    direction: str = 'nearest'
) -> int:
    """
    Quantize a pitch to the nearest note in a maqam.
    
    Args:
        pitch: MIDI note number
        maqam: Target maqam
        direction: 'nearest', 'up', or 'down'
        
    Returns:
        Quantized MIDI note number
    """
    scale_pitches = get_maqam_scale(maqam)
    pitch_class = pitch % 12
    octave = pitch // 12
    
    if pitch_class in scale_pitches:
        return pitch
    
    if direction == 'nearest':
        # Find nearest scale pitch
        min_distance = 12
        nearest = pitch_class
        
        for sp in scale_pitches:
            dist = min(abs(pitch_class - sp), 12 - abs(pitch_class - sp))
            if dist < min_distance:
                min_distance = dist
                nearest = sp
        
        return octave * 12 + nearest
    
    elif direction == 'up':
        # Find next higher scale pitch
        for i in range(1, 13):
            if (pitch_class + i) % 12 in scale_pitches:
                new_pitch_class = (pitch_class + i) % 12
                new_octave = octave + (1 if pitch_class + i >= 12 else 0)
                return new_octave * 12 + new_pitch_class
    
    else:  # down
        # Find next lower scale pitch
        for i in range(1, 13):
            if (pitch_class - i) % 12 in scale_pitches:
                new_pitch_class = (pitch_class - i) % 12
                new_octave = octave - (1 if pitch_class - i < 0 else 0)
                return new_octave * 12 + new_pitch_class
    
    return pitch


def get_maqam_chords(maqam: Maqam) -> Dict[str, Tuple[int, str]]:
    """
    Get common chords used in a maqam.
    
    Returns:
        Dict mapping chord function to (scale_degree, quality)
    """
    # Basic chord qualities based on intervals
    scale = maqam.intervals
    chords = {}
    
    # Root chord
    third = scale[2] - scale[0] if len(scale) > 2 else 4
    if third == 3:
        chords['I'] = (0, 'minor')
    elif third == 4:
        chords['I'] = (0, 'major')
    else:
        chords['I'] = (0, 'sus')
    
    # More chord analysis could be added here
    # For now, return basic tonic chord
    
    return chords


def suggest_accompaniment_pattern(
    maqam: Maqam,
    tempo: float = 120,
    style: str = 'traditional'
) -> List[Dict]:
    """
    Suggest left-hand accompaniment patterns for a maqam.
    
    Args:
        maqam: Maqam to accompany
        tempo: Tempo in BPM
        style: 'traditional', 'modern', 'simple'
        
    Returns:
        List of pattern dictionaries with timing and notes
    """
    root = maqam.root_note
    fifth = (root + 7) % 12
    
    beat_duration = 60 / tempo
    
    if style == 'simple':
        # Just root notes on beats
        return [
            {'time': 0, 'pitch': root + 36, 'duration': beat_duration},  # C2
            {'time': beat_duration * 2, 'pitch': root + 36, 'duration': beat_duration},
        ]
    
    elif style == 'traditional':
        # Root-fifth pattern common in Arabic music
        return [
            {'time': 0, 'pitch': root + 36, 'duration': beat_duration * 0.5},
            {'time': beat_duration, 'pitch': fifth + 36, 'duration': beat_duration * 0.5},
            {'time': beat_duration * 2, 'pitch': root + 36, 'duration': beat_duration * 0.5},
            {'time': beat_duration * 3, 'pitch': fifth + 36, 'duration': beat_duration * 0.5},
        ]
    
    else:  # modern
        # Arpeggiated pattern
        pattern = []
        for i, interval in enumerate(maqam.intervals[:4]):
            pattern.append({
                'time': beat_duration * i * 0.5,
                'pitch': root + 36 + interval,
                'duration': beat_duration * 0.4,
            })
        return pattern


# ═══════════════════════════════════════════════════════════════════════════════
# Ornaments (زخارف)
# ═══════════════════════════════════════════════════════════════════════════════

def add_trill(
    pitch: int,
    duration: float,
    maqam: Maqam,
    speed: float = 8
) -> List[Tuple[float, int, float]]:
    """
    Add an Arabic-style trill (رعشة).
    
    Returns:
        List of (time_offset, pitch, duration) tuples
    """
    # Find upper neighbor in maqam
    upper = quantize_to_maqam(pitch + 1, maqam, direction='up')
    
    notes = []
    trill_duration = duration * 0.8
    note_duration = trill_duration / speed
    
    for i in range(int(speed)):
        t = i * note_duration
        p = pitch if i % 2 == 0 else upper
        notes.append((t, p, note_duration * 0.9))
    
    # End on main note
    notes.append((trill_duration, pitch, duration * 0.2))
    
    return notes


def add_grace_note(
    pitch: int,
    maqam: Maqam,
    direction: str = 'below'
) -> Tuple[int, float]:
    """
    Add a grace note (نغمة زخرفية).
    
    Returns:
        (grace_pitch, grace_duration)
    """
    if direction == 'below':
        grace = quantize_to_maqam(pitch - 1, maqam, direction='down')
    else:
        grace = quantize_to_maqam(pitch + 1, maqam, direction='up')
    
    return (grace, 0.05)  # Short grace note


# ═══════════════════════════════════════════════════════════════════════════════
# Export all maqam names for easy access
# ═══════════════════════════════════════════════════════════════════════════════

ALL_MAQAMAT = [
    "Rast", "Bayyati", "Hijaz", "Saba", "Nahawand", 
    "Sikah", "Ajam", "Kurd", "Hijaz Kar", "Husayni",
    "Jiharkah", "Nawa Athar", "Nikriz", "Athar Kurd"
]

def list_all_maqamat() -> List[Dict]:
    """
    Get a list of all available maqamat with their info.
    
    Returns:
        List of maqam info dicts
    """
    result = []
    seen = set()
    
    for name, maqam in MAQAMAT.items():
        if maqam.name_en in seen:
            continue
        seen.add(maqam.name_en)
        
        result.append({
            'name_en': maqam.name_en,
            'name_ar': maqam.name_ar,
            'mood': maqam.mood,
            'root': ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][maqam.root_note],
        })
    
    return result
