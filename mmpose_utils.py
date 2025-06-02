# main.py :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}

import os, glob, sys, cv2, torch, numpy as np
from mmdet.utils import register_all_modules
register_all_modules()
from mmengine.config import Config
from mmengine.registry import init_default_scope
init_default_scope('mmdet')

from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model, inference_topdown

def find_cached(patterns, backup_path=None):
    cache_root = os.path.expanduser('~/.cache/mim')
    for pat in patterns:
        matches = glob.glob(os.path.join(cache_root, '**', pat), recursive=True)
        if matches:
            return matches[0]
    if backup_path and os.path.exists(os.path.expanduser(backup_path)):
        return os.path.expanduser(backup_path)
    raise FileNotFoundError(f"No files matching {patterns} under {cache_root}")

def download_models_if_needed():
    print("Checking MMDet & MMPose caches…")
    # Detector
    try:
        find_cached(['faster-rcnn_r50_fpn_1x_coco*.pth',
                     'faster_rcnn_r50_fpn_1x_coco*.pth'])
    except FileNotFoundError:
        print("Downloading detector model…")
        os.system("mim download mmdet --config faster-rcnn_r50_fpn_1x_coco --dest ~/.cache/mim")
    # Pose
    try:
        find_cached(['td-hm_hrnet-w32_8xb64-210e_coco-256x192*.pth'])
    except FileNotFoundError:
        print("Downloading pose model…")
        os.system("mim download mmpose --config td-hm_hrnet-w32_8xb64-210e_coco-256x192 --dest ~/.cache/mim")
    print("Models ready.")

def visualize_skeleton(img, pose_results, radius, thickness=1, kpt_score_thr=0.3):
    """Draws the COCO skeleton on the image."""
    img_copy = img.copy()
    skeleton = [
        [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
        [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
        [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
        [3, 5], [4, 6]
    ]
    palette = np.array([[255,128,0],[255,153,51],[255,178,102],[230,230,0],
                        [255,153,255],[153,204,255],[255,102,255],[255,51,255],
                        [102,178,255],[51,153,255],[255,153,153],[255,102,102],
                        [255,51,51],[153,255,153],[102,255,102],[51,255,51],
                        [0,255,0],[0,0,255],[255,0,0],[255,255,255]])
    for pose_data in pose_results:
        # extract the first instance’s keypoints & scores
        instance = pose_data.pred_instances
        kpts   = instance.keypoints[0]
        scores = instance.keypoint_scores[0]

        # draw keypoints
        for i, (x,y) in enumerate(kpts):
            if scores[i] > kpt_score_thr:
                cv2.circle(img_copy, (int(x),int(y)), radius,
                           tuple(map(int, palette[i % len(palette)])), -1)

        # draw skeleton lines
        for idx, (u,v) in enumerate(skeleton):
            if scores[u] > kpt_score_thr and scores[v] > kpt_score_thr:
                pt1 = tuple(map(int, kpts[u]))
                pt2 = tuple(map(int, kpts[v]))
                cv2.line(img_copy, pt1, pt2,
                         tuple(map(int, palette[idx % len(palette)])),
                         thickness)

    return img_copy
