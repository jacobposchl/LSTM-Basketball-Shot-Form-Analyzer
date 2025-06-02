#!/usr/bin/env python3
"""
Review the exact video snippets corresponding to each fixed-length sequence
that your LSTM trains on (.npz sequences), overlaying:
 - a green bar for makes, red for misses
 - a top text overlay showing shot start/end frames and times
and tile them in a grid.
"""
import os
import tempfile
import numpy as np
import pandas as pd
from moviepy.editor import (
    VideoFileClip,
    ColorClip,
    CompositeVideoClip,
    ImageClip
)
from google.cloud import storage
from PIL import Image, ImageDraw, ImageFont

from config import GCP_DOWNLOAD_DIR, GCP_BUCKET_NAME
from data_processing import (
    compute_sequence_info,
    compute_max_length,
    compute_subdataset_info
)

# -------------------- USER CONFIGURATION --------------------
CSV_PATH = (
    "/home/jakeposchl/LSTM-Basketball-Shot-Form-Analyzer/Datasets/49DA42F0-FFF6-4CE8-8284-685C2BD1D3B9-2F68BE92-9BD5-4D1D-9542-D3DA420DDE57_entire_dataset_49DA42F0-FFF6-4CE8-8284-685C2BD1D3B9.csv"
)
JOB_ID     = "49DA42F0-FFF6-4CE8-8284-685C2BD1D3B9"
NPZ_PREFIX = "dev_shot_sequences"
# ------------------------------------------------------------

def make_text_clip(text, width, duration, fontsize=14):
    """
    Render `text` into a transparent ImageClip of size (width x text_height),
    using a scalable TTF.
    """
    # 1) Load a scalable font
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    try:
        font = ImageFont.truetype(font_path, fontsize)
    except IOError:
        print(f"⚠️  Warning: could not load {font_path}, falling back to default font.")
        font = ImageFont.load_default()

    # 2) Measure text with textbbox
    dummy = Image.new("RGBA", (1,1), (0,0,0,0))
    draw  = ImageDraw.Draw(dummy)
    l, t, r, b = draw.textbbox((0,0), text, font=font)
    text_w = r - l
    text_h = b - t
    pad    = 8
    total_h = text_h + pad*2

    # 3) Create semi-transparent background
    bg = Image.new("RGBA", (width, total_h), (0, 0, 0, 180))
    draw = ImageDraw.Draw(bg)
    x = max((width - text_w)//2, 0)
    y = pad
    draw.text((x, y), text, font=font, fill="white")

    # 4) Convert to ImageClip
    return (
        ImageClip(np.array(bg))
        .set_duration(duration)
        .set_position(("center", "top"))
    )

def main():
    # 1) Load CSV
    print(f"Loading CSV from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # 2) Compute padded/truncated windows
    seq_info = compute_sequence_info(df)
    max_len  = compute_max_length(seq_info)
    sub_info = compute_subdataset_info(df, seq_info, max_len)
    # sub_info entries: (shot_id, start_frame, end_frame, seq_len)

    # 3) List .npz blobs
    client = storage.Client()
    bucket = client.bucket(GCP_BUCKET_NAME)
    blobs  = list(bucket.list_blobs(prefix=f"{NPZ_PREFIX}/{JOB_ID}_"))
    print(f"Found {len(blobs)} .npz sequences for job {JOB_ID}")

    # 4) Load video
    video_filename = df['video'].iloc[0]
    video_path     = os.path.join(GCP_DOWNLOAD_DIR, video_filename)
    print(f"Loading video: {video_path}")
    video = VideoFileClip(video_path, audio=False)
    fps   = video.fps
    print(f"Video loaded ({fps:.2f} FPS, {video.duration:.2f}s)")

    # 5) Build annotated clips
    labeled_clips = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for blob in blobs:
            shot_fname = os.path.basename(blob.name)  # e.g. JOBID_3.npz
            shot_id    = int(shot_fname.split("_")[-1].split(".")[0])

            # Download .npz
            npz_path = os.path.join(tmpdir, shot_fname)
            blob.download_to_filename(npz_path)
            data  = np.load(npz_path)
            label = int(data["label"])  # 1=make, 0=miss

            # Find matching window
            match = next((s for s in sub_info if s[0] == shot_id), None)
            if not match:
                print(f"⚠️  No window for shot {shot_id}, skipping")
                continue
            _, start_f, end_f, _ = match

            # Extract subclip (narrow width for more text room)
            start_t = start_f / fps
            end_t   = end_f   / fps
            clip    = video.subclip(start_t, end_t).resize(width=360)
            w, h    = clip.size

            # Bottom bar
            bar_col = (0,150,0) if label == 1 else (150,0,0)
            bar = (
                ColorClip((w, 30), color=bar_col)
                .set_duration(clip.duration)
                .set_position(("center", "bottom"))
            )

            # Top text
            txt = (
                f"S {shot_id}  F {start_f}-{end_f}  "
                f"T {start_t:.2f}s-{end_t:.2f}s"
            )
            txt_clip = make_text_clip(txt, w, clip.duration, fontsize=14)

            comp = CompositeVideoClip([clip, bar, txt_clip])
            labeled_clips.append(comp)
            print(f"Prepared shot {shot_id}: frames {start_f}-{end_f}, label={'Make' if label else 'Miss'}")

    if not labeled_clips:
        raise RuntimeError("No clips to display.")

    # 6) Tile in grid: more columns (6) to give each clip room
    n_per_row = 6
    cw, ch    = labeled_clips[0].size
    n_rows    = (len(labeled_clips) + n_per_row - 1) // n_per_row
    grid_w    = cw * n_per_row
    grid_h    = ch * n_rows

    positioned = []
    for idx, clip in enumerate(labeled_clips):
        row, col = divmod(idx, n_per_row)
        x, y     = col * cw, row * ch
        positioned.append(clip.set_position((x, y)))

    final = CompositeVideoClip(positioned, size=(grid_w, grid_h))

    # 7) Export
    out = "sequence_review.mp4"
    print(f"Writing {out} …")
    final.write_videofile(out, fps=fps, audio=False)
    print("Done! Check sequence_review.mp4")

if __name__ == "__main__":
    main()
