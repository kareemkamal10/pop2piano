"""
Script to estimate total dataset size from YouTube without downloading.
Uses yt-dlp to fetch video metadata and calculate file sizes.

Usage:
    python estimate_dataset_size.py
    python estimate_dataset_size.py --limit 100  # Test with first 100 videos
"""

import os
import sys
import csv
import json
import subprocess
import platform
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse

def get_yt_dlp_path():
    """Get yt-dlp executable path."""
    if platform.system() == "Windows":
        yt_dlp_name = "yt-dlp.exe"
    else:
        yt_dlp_name = "yt-dlp"
    
    # Try finding in Python's Scripts folder first
    yt_dlp_path = os.path.join(os.path.dirname(sys.executable), yt_dlp_name)
    if os.path.exists(yt_dlp_path):
        return yt_dlp_path
    
    # Assume it's in PATH
    return yt_dlp_name

def get_video_size(video_id, yt_dlp_path):
    """
    Get estimated file size for a YouTube video without downloading.
    Returns size in bytes or None if failed.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    cmd = [
        yt_dlp_path,
        "--dump-json",
        "--no-download",
        "--no-warnings",
        "-f", "bestaudio",  # We only need audio
        url
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=30,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode == 0 and result.stdout:
            data = json.loads(result.stdout)
            # filesize or filesize_approx
            size = data.get('filesize') or data.get('filesize_approx') or 0
            duration = data.get('duration') or 0
            title = data.get('title', 'Unknown')
            return {
                'id': video_id,
                'size': size,
                'duration': duration,
                'title': title,
                'status': 'ok'
            }
        else:
            return {
                'id': video_id,
                'size': 0,
                'duration': 0,
                'title': 'N/A',
                'status': 'error'
            }
    except subprocess.TimeoutExpired:
        return {
            'id': video_id,
            'size': 0,
            'duration': 0,
            'title': 'N/A',
            'status': 'timeout'
        }
    except Exception as e:
        return {
            'id': video_id,
            'size': 0,
            'duration': 0,
            'title': 'N/A',
            'status': f'error: {str(e)}'
        }

def format_size(bytes_size):
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f} PB"

def format_duration(seconds):
    """Convert seconds to human readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def main():
    parser = argparse.ArgumentParser(description='Estimate YouTube dataset size without downloading')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of videos to check')
    parser.add_argument('--workers', type=int, default=5, help='Number of parallel workers (default: 5)')
    parser.add_argument('--csv', type=str, default='train_dataset.csv', help='Path to CSV file')
    args = parser.parse_args()
    
    # Read CSV
    csv_path = args.csv
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        sys.exit(1)
    
    video_ids = set()
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_ids.add(row['piano_ids'])  # Piano covers
            video_ids.add(row['pop_ids'])    # Pop songs
    
    video_ids = list(video_ids)
    
    if args.limit:
        video_ids = video_ids[:args.limit]
    
    print(f"📊 Estimating size for {len(video_ids)} unique videos...")
    print(f"🔧 Using {args.workers} parallel workers")
    print("-" * 60)
    
    yt_dlp_path = get_yt_dlp_path()
    
    # Check if yt-dlp exists
    try:
        subprocess.run([yt_dlp_path, "--version"], capture_output=True, timeout=10)
    except Exception as e:
        print(f"❌ yt-dlp not found! Please install it: pip install yt-dlp")
        sys.exit(1)
    
    results = []
    total_size = 0
    total_duration = 0
    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(get_video_size, vid, yt_dlp_path): vid 
            for vid in video_ids
        }
        
        with tqdm(total=len(video_ids), desc="Fetching metadata") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                
                if result['status'] == 'ok':
                    total_size += result['size']
                    total_duration += result['duration']
                    success_count += 1
                else:
                    error_count += 1
                
                pbar.update(1)
                pbar.set_postfix({
                    'size': format_size(total_size),
                    'ok': success_count,
                    'err': error_count
                })
    
    # Print results
    print("\n" + "=" * 60)
    print("📈 RESULTS SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully checked: {success_count} videos")
    print(f"❌ Failed/Unavailable:   {error_count} videos")
    print("-" * 60)
    print(f"📦 Total estimated size: {format_size(total_size)}")
    print(f"⏱️  Total duration:       {format_duration(total_duration)}")
    print("-" * 60)
    
    if success_count > 0:
        avg_size = total_size / success_count
        avg_duration = total_duration / success_count
        print(f"📊 Average per video:    {format_size(avg_size)}")
        print(f"⏱️  Average duration:     {format_duration(avg_duration)}")
        
        # Extrapolate for full dataset if we used --limit
        if args.limit and args.limit < len(video_ids):
            full_count = len(video_ids) + (len(video_ids) - args.limit)  # Approximate
            estimated_full = avg_size * full_count
            print("-" * 60)
            print(f"🔮 Estimated full dataset: ~{format_size(estimated_full)}")
    
    print("=" * 60)
    
    # Save detailed results to file
    output_file = "dataset_size_report.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'size', 'duration', 'title', 'status'])
        writer.writeheader()
        writer.writerows(results)
    print(f"📄 Detailed report saved to: {output_file}")

if __name__ == "__main__":
    main()
