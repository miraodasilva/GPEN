"""
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
"""
import os
import cv2
import glob
import argparse
import numpy as np
from face_enhancement import FaceEnhancement
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader


def read_video_opencv(filename):
    video = []
    cap = cv2.VideoCapture(filename)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            video.append(frame)
        else:
            break
    cap.release()
    video = np.array(video)
    return video


def write_lossless_video(video, path):
    fourcc = cv2.VideoWriter_fourcc("F", "F", "V", "1")
    # fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    writer = cv2.VideoWriter(path, fourcc, 25, (list(video)[0].shape[0], list(video)[0].shape[1]))
    for frame in list(video):
        writer.write(frame)
    writer.release()  # close the writer


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, dir):
        self.files = []
        for d, _, files in os.walk(dir):
            for f in files:
                if f.endswith(".mp4"):
                    self.files += [os.path.join(d, f)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        video_path = self.files[idx]
        frames = list(read_video_opencv(video_path))
        return frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="GPEN-BFR-512", help="GPEN model")
    parser.add_argument("--task", type=str, default="FaceEnhancement", help="task of GPEN model")
    parser.add_argument("--key", type=str, default=None, help="key of GPEN model")
    parser.add_argument("--in_size", type=int, default=512, help="in resolution of GPEN")
    parser.add_argument("--out_size", type=int, default=None, help="out resolution of GPEN")
    parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier of GPEN")
    parser.add_argument("--narrow", type=float, default=1, help="channel narrow scale")
    parser.add_argument("--use_sr", action="store_true", help="use sr or not")
    parser.add_argument("--use_cuda", action="store_true", help="use cuda or not")
    parser.add_argument("--aligned", action="store_true", help="input are aligned faces or not")
    parser.add_argument("--sr_model", type=str, default="realesrnet", help="SR model")
    parser.add_argument("--sr_scale", type=int, default=2, help="SR scale")
    parser.add_argument("--indir", type=str, default="examples/imgs", help="input folder")
    parser.add_argument("--outdir", type=str, default="results/outs-BFR", help="output folder")
    parser.add_argument("--file_list", type=str, default=None, help="output folder")
    # new
    # parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # dataset = VideoDataset(args.indir)
    # video_data_loader = DataLoader(
    #    dataset, batch_size=args.batch_size, shuffle=False, num_workers=50, pin_memory=True, drop_last=False
    # )

    if args.task == "FaceEnhancement":
        processer = FaceEnhancement(
            base_dir="/data/home/rs2517/GPEN/",
            in_size=args.in_size,
            model=args.model,
            use_sr=args.use_sr,
            sr_model=args.sr_model,
            sr_scale=args.sr_scale,
            channel_multiplier=args.channel_multiplier,
            narrow=args.narrow,
            key=args.key,
            device="cuda" if args.use_cuda else "cpu",
        )
    else:
        raise NotImplementedError("Not Implemented!")
    if args.file_list:
        with open(args.file_list) as f:
            files = [l.rstrip() for l in f.readlines()]
    else:
        files = sorted(glob.glob(os.path.join(args.indir, "*.mp4")))
    for n, file in enumerate(tqdm(files[:])):
        filename = file
        frames = list(read_video_opencv(filename))
        enhanced_frames = []
        for img in tqdm(frames):
            with torch.no_grad():
                img_out, orig_faces, enhanced_faces = processer.process(img, aligned=args.aligned)
            img = cv2.resize(img, img_out.shape[:2][::-1])
            enhanced_frames += [img_out]

        new_video_path = filename.replace(".mp4", ".avi").replace(args.indir, args.outdir)
        print(new_video_path)
        os.makedirs(os.path.dirname(new_video_path), exist_ok=True)
        write_lossless_video(enhanced_frames, new_video_path)
