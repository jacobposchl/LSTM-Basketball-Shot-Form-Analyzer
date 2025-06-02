#!/usr/bin/env python3
"""
run_pipeline.py

For each video in the list, this script:
  1. Extracts the job_id (first GUID) from the filename.
  2. Fetches that job’s raw gesture events from Firestore and writes gestures.json.
  3. Calls main.py with the correct --video, --job_id and --gesture_events_file flags.
"""

import os
import subprocess
import json
from google.cloud import firestore

# --- Configuration — adjust if your paths ever move ---
BUCKET_PREFIX = "gs://basketball-ai-data/project/basketball-ai-data/files/Videos/"
GESTURE_FILE = "gestures.json"

# List all of your video filenames here (just the basename .mp4)
VIDEOS = [
    "5C9A6C63-21FC-4246-98EE-2E030D1D0FF5-7EDFADC8-0B71-4E4B-9575-7280A3B19CAD.mp4",
    "B12A6C74-40D6-46F7-B01C-5BD32EAB57C3-0C5EF3CE-F414-4CDE-8F7E-71A48066E1E1.mp4",
    "983FDE05-E27D-42FE-8ACF-E8575082BAC2-B78630AC-20DB-4D28-B7C7-76C6AB076B8F.mp4",
    "8B6F531F-0ED9-49CF-8C07-B282109EFBAD-4DB55C62-1F1D-4E2F-BDFC-7A8CE3D8DD6C.mp4",
    "8C3FA4AA-C31D-4EF8-BDAD-1EFB87F26F2D-B8FF9AB2-C64E-4279-A8D4-30902C7BDE39.mp4",
    "7AD8F3AC-1F53-4BC6-819E-928908A81BFE-645002CA-9479-4311-8812-E8058D24A289.mp4",
    "6A0745FA-3820-4DEB-B74D-130717AAC3B4-62C78CB5-83A1-4481-901A-AD3B14E2DD6B.mp4",
    "B796FD79-D8AD-40F1-9FA9-41EB374CD07D-488C30EF-36E9-433D-976D-682CF7D854D0.mp4",
    "024B25FB-6872-467A-A8E2-ACBCDC4522CA-AC4CFB43-80BC-43AD-B685-BDE60CCA6C50.mp4",
    "ED8E466C-0364-4037-A63A-1C97EF606DF8-5256D2A1-8031-4A31-A7B2-728C8B5E5743.mp4",
    "D1C4A126-A17B-490B-BCAE-B09ED5EB83AB-0258B6F5-58D1-452D-8123-EBBF1A8D9897.mp4",
    "4D0555E4-8812-48ED-967F-098035BB603C-D6A7DF92-B0F2-45A0-9C1F-C543BD759925.mp4",
    "EA046113-CDA3-4F3F-9A66-44675D6EEE38-A7F41E6A-4949-460F-97D1-5EFAB1B10E0F.mp4",
    "518FCA99-8D2C-4172-899F-B3A504F7F6FC-FACFD5FF-93F1-40A4-9C42-DD6BE2CE79D7.mp4",
    "739FD8BC-D695-48D7-9A7F-AE8426A2CFA7-FD37C762-9DBC-4EF6-9DE2-A2ADFD7AC72E.mp4",
    "77316F39-D64A-44AA-887B-80AA6DDC453D-5ECE9A4F-4A2C-44E6-B020-15A413224A10.mp4",
    "D1C4A126-A17B-490B-BCAE-B09ED5EB83AB-0258B6F5-58D1-452D-8123-EBBF1A8D9897.mp4",
    "EA046113-CDA3-4F3F-9A66-44675D6EEE38-A7F41E6A-4949-460F-97D1-5EFAB1B10E0F.mp4",
    "ED8E466C-0364-4037-A63A-1C97EF606DF8-5256D2A1-8031-4A31-A7B2-728C8B5E5743.mp4",
    "F62940F1-2D8A-452B-B190-DB5CF1263228-24070A0B-E011-4CA3-BB67-3C0EC93B2638.mp4",
]

def export_gestures(job_id: str) -> None:
    db = firestore.Client()
    doc = db.collection("jobs").document(job_id).get()
    if not doc.exists:
        raise RuntimeError(f"Firestore document for job '{job_id}' does not exist.")
    data = doc.to_dict()
    events = data.get("raw_gesture_payload", {}).get("gesture_events", [])
    with open(GESTURE_FILE, "w") as f:
        json.dump(events, f, indent=2)
    print(f"[{job_id}] Wrote {len(events)} gesture events → {GESTURE_FILE}")

def run_main(video: str, job_id: str) -> None:
    video_path = BUCKET_PREFIX + video
    cmd = [
        "python3", "main.py",
        "--video", video_path,
        "--job_id", job_id,
        "--gesture_events_file", os.path.abspath(GESTURE_FILE)
    ]
    print(f"[{job_id}] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[{job_id}] main.py completed.\n")

def process(video: str) -> None:
    # Extract job_id = first five dash-segments of filename
    base = os.path.splitext(video)[0]
    parts = base.split("-")
    if len(parts) < 5:
        raise ValueError(f"Filename '{video}' doesn't contain a full GUID.")
    job_id = "-".join(parts[:5])
    export_gestures(job_id)
    run_main(video, job_id)

if __name__ == "__main__":
    for vid in VIDEOS:
        try:
            process(vid)
        except Exception as e:
            print(f"ERROR processing {vid}: {e}")
            break
