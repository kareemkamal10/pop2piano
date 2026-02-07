"""
Usage:
python youtube_down.py piano_covers.txt /output/dir
"""

import os
import multiprocessing

import tempfile
import shutil
import glob
import pandas as pd
import re

from tqdm import tqdm
from joblib import Parallel, delayed
import sys
import subprocess
import platform
from omegaconf import OmegaConf

def get_dir_size(path):
    total = 0
    try:
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += get_dir_size(entry.path)
    except FileNotFoundError:
        pass
    return total

def download_piano(
    url: str,
    output_dir: str,
    postprocess=True,
    dry_run=False,
    max_size_gb=None,
) -> int:
    # os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    
    if max_size_gb is not None:
        current_size = get_dir_size(output_dir)
        if current_size > max_size_gb * 1024 * 1024 * 1024:
            # print("Size limit reached.")
            return 0
    
    import imageio_ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    
    # Determine yt-dlp path based on OS
    if platform.system() == "Windows":
        yt_dlp_name = "yt-dlp.exe"
    else:
        yt_dlp_name = "yt-dlp"

    # Try finding yt-dlp in the same dir as python first
    yt_dlp_path = os.path.join(os.path.dirname(sys.executable), yt_dlp_name)
    
    # If not found there, assume it's in PATH
    if not os.path.exists(yt_dlp_path):
        yt_dlp_path = yt_dlp_name

    with tempfile.TemporaryDirectory() as tmpdir:
        output = f"{tmpdir}/%(uploader)s___%(title)s___%(id)s___%(duration)d.%(ext)s"

        cmd = [
            yt_dlp_path,
            "-o", output,
            "--ffmpeg-location", ffmpeg_path,
            "--extract-audio",
            "--audio-quality", "0",
            "--audio-format", "wav",
            "--retries", "50",
            "--prefer-ffmpeg",
            "--force-ipv4",
            "--yes-playlist",
            "--ignore-errors"
        ]

        if dry_run:
            cmd.append("--get-filename")
        
        if postprocess:
            cmd.extend(["--postprocessor-args", "-ac 1 -ar 16000"])
            
        cmd.append(url)
        
        result = subprocess.call(cmd)

        if not dry_run:

            files = os.listdir(tmpdir)

            for filename in files:
                filename_wo_ext, ext = os.path.splitext(filename)
                uploader, title, ytid, duration = filename_wo_ext.split("___")
                meta = OmegaConf.create()
                meta.piano = OmegaConf.create()
                meta.piano.uploader = uploader
                meta.piano.title = title
                meta.piano.ytid = ytid
                meta.piano.duration = int(duration)
                OmegaConf.save(meta, os.path.join(output_dir, ytid + ".yaml"))
                shutil.move(
                    os.path.join(tmpdir, filename),
                    os.path.join(output_dir, f"{ytid}{ext}"),
                )

    return result


def download_piano_main(piano_list, output_dir, dry_run=False, max_size_gb=None):
    """
    piano_list : list of youtube id
    """
    os.makedirs(output_dir, exist_ok=True)
    Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(download_piano)(
            url=f"https://www.youtube.com/watch?v={ytid}",
            output_dir=output_dir,
            postprocess=True,
            dry_run=dry_run,
            max_size_gb=max_size_gb,
        )
        for ytid in tqdm(piano_list)
    )


def download_pop(piano_id, pop_id, output_dir, dry_run, max_size_gb=None):
    
    if max_size_gb is not None:
        current_size = get_dir_size(output_dir)
        if current_size > max_size_gb * 1024 * 1024 * 1024:
            # print("Size limit reached.")
            return 0
    
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    output_file_template = "%(id)s___%(title)s___%(duration)d.%(ext)s"
    pop_output_dir = os.path.join(output_dir, piano_id)
    os.makedirs(pop_output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, piano_id, output_file_template)
    url = f"https://www.youtube.com/watch?v={pop_id}"
    
    # Determine yt-dlp path based on OS
    if platform.system() == "Windows":
        yt_dlp_name = "yt-dlp.exe"
    else:
        yt_dlp_name = "yt-dlp"

    # Try finding yt-dlp in the same dir as python first
    yt_dlp_path = os.path.join(os.path.dirname(sys.executable), yt_dlp_name)
    
    # If not found there, assume it's in PATH
    if not os.path.exists(yt_dlp_path):
        yt_dlp_path = yt_dlp_name
    
    # Needs ffmpeg path too if not in env
    import imageio_ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        yt_dlp_path,
        "-o", output_template,
        "--ffmpeg-location", ffmpeg_path,
        "--extract-audio",
        "--audio-quality", "0",
        "--audio-format", "wav",
        "--retries", "25",
        "--prefer-ffmpeg",
        "--match-filter", "duration < 300 & duration > 150",
        "--postprocessor-args", "-ac 2 -ar 44100"
    ]

    if dry_run:
        cmd.append("--get-filename")
    
    cmd.append(url)
    
    result = subprocess.call(cmd)

    if not dry_run:
        files = list(filter(lambda x: x.endswith(".wav"), os.listdir(pop_output_dir)))
        files = glob.glob(os.path.join(pop_output_dir, "*.wav"))
        for filename in files:
            filename_wo_ext, ext = os.path.splitext(os.path.basename(filename))
            ytid, title, duration = filename_wo_ext.split("___")
            yaml = os.path.join(output_dir, piano_id + ".yaml")

            meta = OmegaConf.load(yaml)
            meta.song = OmegaConf.create()
            meta.song.ytid = ytid
            meta.song.title = title
            meta.song.duration = int(duration)

            OmegaConf.save(meta, yaml)
            shutil.move(
                os.path.join(filename),
                os.path.join(pop_output_dir, f"{ytid}{ext}"),
            )


def download_pop_main(piano_list, pop_list, output_dir, dry_run=False, max_size_gb=None):
    """
    piano_list : list of youtube id
    pop_list : corresponding youtube id of pop songs
    """

    Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(download_pop)(
            piano_id=piano_id,
            pop_id=pop_id,
            output_dir=output_dir,
            dry_run=dry_run,
            max_size_gb=max_size_gb,
        )
        for piano_id, pop_id in tqdm(list(zip(piano_list, pop_list)))
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="piano cover downloader")

    parser.add_argument("dataset", type=str, default=None, help="provided csv")
    parser.add_argument("output_dir", type=str, default=None, help="output dir")
    parser.add_argument(
        "--num_audio",
        type=int,
        default=None,
        help="if specified, only {num_audio} pairs will be downloaded",
    )
    parser.add_argument(
        "--dry_run", default=False, action="store_true", help="whether dry_run"
    )
    parser.add_argument(
        "--max_size_gb",
        type=float,
        default=None,
        help="Maximum size of output directory in GB before stopping (optional)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    df = df[: args.num_audio]
    piano_list = df["piano_ids"].tolist()
    download_piano_main(piano_list, args.output_dir, args.dry_run, args.max_size_gb)

    available_piano_list = glob.glob(args.output_dir + "/**/*.yaml", recursive=True)
    df.index = df["piano_ids"]

    failed_piano = []

    available_piano_list_id = [
        os.path.splitext(os.path.basename(ap))[0] for ap in available_piano_list
    ]

    for piano_id_to_be_downloaded in tqdm(df["piano_ids"]):
        if piano_id_to_be_downloaded in available_piano_list_id:
            continue
        else:
            failed_piano.append(piano_id_to_be_downloaded)

    if len(failed_piano) > 0:
        print(f"{len(failed_piano)} of files are failed to be downloaded")
        df = df.drop(index=failed_piano)

    piano_list = df["piano_ids"].tolist()
    pop_list = df["pop_ids"].tolist()

    download_pop_main(
        piano_list, pop_list, output_dir=args.output_dir, dry_run=args.dry_run, max_size_gb=args.max_size_gb
    )
