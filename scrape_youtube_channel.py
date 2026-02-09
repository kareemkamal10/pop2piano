"""
Script to extract video IDs from a YouTube channel or playlist.
Useful for collecting Arabic piano covers for training.

Usage:
    python scrape_youtube_channel.py "https://www.youtube.com/@pianosongsbymo/playlists"
    python scrape_youtube_channel.py "https://www.youtube.com/@pianosongsbymo/videos"
    python scrape_youtube_channel.py "PLAYLIST_URL"
"""

import os
import sys
import json
import subprocess
import platform
import csv
from datetime import datetime

def get_yt_dlp_path():
    """Get yt-dlp executable path."""
    if platform.system() == "Windows":
        yt_dlp_name = "yt-dlp.exe"
    else:
        yt_dlp_name = "yt-dlp"
    
    yt_dlp_path = os.path.join(os.path.dirname(sys.executable), yt_dlp_name)
    if os.path.exists(yt_dlp_path):
        return yt_dlp_path
    return yt_dlp_name

def extract_videos_from_channel(channel_url, yt_dlp_path):
    """Extract all video info from a YouTube channel or playlist."""
    
    print(f"🔍 Fetching videos from: {channel_url}")
    print("⏳ This may take a few minutes...")
    
    cmd = [
        yt_dlp_path,
        "--flat-playlist",
        "--dump-json",
        "--no-warnings",
        "--ignore-errors",
        channel_url
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        videos = []
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    data = json.loads(line)
                    videos.append({
                        'id': data.get('id', ''),
                        'title': data.get('title', 'Unknown'),
                        'duration': data.get('duration', 0),
                        'url': f"https://www.youtube.com/watch?v={data.get('id', '')}"
                    })
                except json.JSONDecodeError:
                    continue
        
        return videos
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def main():
    if len(sys.argv) < 2:
        # Default to the Arabic piano channel
        channel_url = "https://www.youtube.com/@pianosongsbymo/videos"
        print("ℹ️  No URL provided, using default: @pianosongsbymo")
    else:
        channel_url = sys.argv[1]
    
    yt_dlp_path = get_yt_dlp_path()
    
    # Check yt-dlp
    try:
        subprocess.run([yt_dlp_path, "--version"], capture_output=True, timeout=10)
    except:
        print("❌ yt-dlp not found! Install with: pip install yt-dlp")
        sys.exit(1)
    
    videos = extract_videos_from_channel(channel_url, yt_dlp_path)
    
    if not videos:
        print("❌ No videos found!")
        sys.exit(1)
    
    print(f"\n✅ Found {len(videos)} videos!")
    print("=" * 60)
    
    # Print first 10 as preview
    print("\n📋 Preview (first 10):")
    print("-" * 60)
    for i, video in enumerate(videos[:10], 1):
        duration_str = f"{video['duration']//60}:{video['duration']%60:02d}" if video['duration'] else "N/A"
        print(f"{i:3}. [{video['id']}] {video['title'][:50]}... ({duration_str})")
    
    if len(videos) > 10:
        print(f"    ... and {len(videos) - 10} more")
    
    # Save to files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save as CSV
    csv_filename = f"arabic_piano_videos_{timestamp}.csv"
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'title', 'duration', 'url'])
        writer.writeheader()
        writer.writerows(videos)
    print(f"\n📄 Saved to: {csv_filename}")
    
    # 2. Save just IDs (for easy copy/paste)
    ids_filename = f"arabic_piano_ids_{timestamp}.txt"
    with open(ids_filename, 'w', encoding='utf-8') as f:
        for video in videos:
            f.write(video['id'] + '\n')
    print(f"📄 IDs saved to: {ids_filename}")
    
    # 3. Print summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Total videos: {len(videos)}")
    
    total_duration = sum(v['duration'] or 0 for v in videos)
    hours = total_duration // 3600
    minutes = (total_duration % 3600) // 60
    print(f"Total duration: {hours}h {minutes}m")
    
    print("\n💡 Next steps:")
    print("   1. Find the ORIGINAL Arabic songs for each piano cover")
    print("   2. Create pairs like: piano_id,pop_id")
    print("   3. Add to train_dataset.csv or create arabic_dataset.csv")
    print("=" * 60)

if __name__ == "__main__":
    main()
