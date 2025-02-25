import os
import subprocess
import numpy as np
import torch as t
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms._transforms_video import ToTensorVideo, NormalizeVideo

from tinyHD.src.models.fastsal3D.model import FastSalA
from tinyHD.src.my_sampler import EvalClipSampler
from tinyHD.src.dataset.utils.video_clips import VideoClips
from tinyHD.src.dataset.my_transform import resize_random_hflip_crop
from tinyHD.src.generate import post_process_png
import tinyHD.src.config as cfg

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config.config import load_config

class VideoProcessor:
    def __init__(self, config):
        self.weight_path = config.get('weight_path', 'ucf_d123s.pth')
        self.video_path = config.get('video_path', 'data/UCF101/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi')
        self.save_path = config.get('save_path', 'output')
        self.config_path, self.reduced_channel, self.mode = self.extract_config_from_weights(self.weight_path)
        self.config = cfg.get_config(self.config_path)
        self.model = self.load_model()
        self.video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.save_path_new = os.path.join(self.save_path, self.video_name)
        self.original_frames = os.path.join(self.save_path_new, "original_frames")
        self.heatmap_bboxes = os.path.join(self.save_path_new, "heatmap_with_bbox")
        self.output_bboxes = os.path.join(self.save_path_new, "original_frames_with_bbox")

    def extract_config_from_weights(self, weight_path):
        config_mappings = {
            'ucf_d123s.pth': ('tinyHD/src/config/eval_config_single.yaml', 1, 'single'),
            'ucf_d123m.pth': ('tinyHD/src/config/eval_config_multi_ucf.yaml', 1, 'multi'),
        }
        return config_mappings.get(os.path.basename(weight_path), ValueError('Unknown weight file'))

    def load_model(self):
        model = FastSalA(self.reduced_channel, self.config.MODEL_SETUP.DECODER, self.config.MODEL_SETUP.SINGLE,
                         d1_last=self.config.MODEL_SETUP.D1_LAST, force_multi=self.config.MODEL_SETUP.FORCE_MULTI,
                         n_output=self.config.MODEL_SETUP.OUTPUT_SIZE)
        model.load_state_dict(t.load(self.weight_path, map_location='cuda:0')['student_model'])
        model.cuda().eval()
        return model

    def generate_video_from_frames(self, frames_dir, output_video_path, fps=1):
        if not os.path.exists(frames_dir) or not os.listdir(frames_dir):
            raise FileNotFoundError(f"No frames found in directory: {frames_dir}")
        
        command = [
            "ffmpeg", "-framerate", str(fps), "-i", os.path.join(frames_dir, "%d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", output_video_path
        ]
        subprocess.run(command, check=True)
        print(f"Video saved at {output_video_path}")

    def extract_bounding_boxes(self, image, percentile=95, max_bboxes=5):
        threshold = np.percentile(image, percentile)
        mask = (image >= threshold).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = sorted(
            (cv2.boundingRect(contour) for contour in contours),
            key=lambda bbox: bbox[2] * bbox[3], reverse=True
        )
        return [(x, y, x + w, y + h) for x, y, w, h in bboxes[:max_bboxes]]

    def draw_bounding_boxes(self, image, bboxes, color=(0, 0, 255), thickness=2):
        for x_min, y_min, x_max, y_max in bboxes:
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

    def extract_video_frames(self, video_path, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(save_dir, f"{frame_idx}.png"), frame)
            frame_idx += 1
        cap.release()
        print(f"Frames saved in {save_dir}")

    def overlay_bboxes_on_original(self, original_dir, bbox_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for orig_frame, bbox_frame in zip(sorted(os.listdir(original_dir)), sorted(os.listdir(bbox_dir))):
            orig_img = cv2.imread(os.path.join(original_dir, orig_frame))
            bbox_img = cv2.imread(os.path.join(bbox_dir, bbox_frame), cv2.IMREAD_GRAYSCALE)
            bboxes = self.extract_bounding_boxes(bbox_img)
            self.draw_bounding_boxes(orig_img, bboxes)
            cv2.imwrite(os.path.join(output_dir, orig_frame), orig_img)
        print(f"Processed frames saved in {output_dir}")

    def process_video(self):
        """Processes the video, extracts frames, generates bounding boxes, and creates output video."""
        
        # Extract frames
        self.extract_video_frames(self.video_path, self.original_frames)

        # Get list of extracted frames
        frame_files = sorted(os.listdir(self.original_frames))

        # Check if frames were extracted successfully
        if not frame_files:
            raise RuntimeError(f"No frames were extracted in {self.original_frames}. Check if {self.video_path} is valid.")

        # Load the first frame to get its shape
        first_frame_path = os.path.join(self.original_frames, frame_files[0])
        first_frame = cv2.imread(first_frame_path)

        # Check if the frame was loaded correctly
        if first_frame is None:
            raise RuntimeError(f"Failed to read the first frame: {first_frame_path}")

        frame_shape = first_frame.shape[:2]  # Get (height, width)

        dataloader = DataLoader(VideoDataset(self.video_path, 16, 1), batch_size=20, shuffle=False, num_workers=4, pin_memory=True)

        os.makedirs(self.heatmap_bboxes, exist_ok=True)

        with t.no_grad():
            for x, frame_idx, _ in dataloader:
                pred_maps = self.model(x.cuda())[0].cpu().numpy()
                for img, fid in zip(pred_maps, frame_idx.numpy()[:, -1]):
                    processed_img = post_process_png(img[0], original_shape=(frame_shape[1], frame_shape[0]))
                    self.draw_bounding_boxes(processed_img, self.extract_bounding_boxes(processed_img))
                    cv2.imwrite(os.path.join(self.heatmap_bboxes, f'{fid}.png'), processed_img)

        self.overlay_bboxes_on_original(self.original_frames, self.heatmap_bboxes, self.output_bboxes)
        self.generate_video_from_frames(self.output_bboxes, os.path.join(self.save_path, "output_with_bbox.mp4"), fps=30)
        
        print("Processing complete.")



class VideoDataset(Dataset):
    def __init__(self, video_path, window_size, step):
        self.vr = VideoClips([video_path], clip_length_in_frames=window_size, frames_between_clips=step)
        self.to_tensor = T.Compose([ToTensorVideo(), NormalizeVideo((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.augmentation = resize_random_hflip_crop((192, 256), (192, 256), False, False, False)

    def __getitem__(self, index):
        video_data, _, _, _, clip_pts = self.vr.get_clip(index)
        original_size = np.asarray(video_data.shape[1:-1])
        video_data = self.to_tensor(video_data)
        processed_data = self.augmentation([video_data])[0]
        return processed_data, clip_pts, original_size

    def __len__(self):
        return len(self.vr)


if __name__ == '__main__':
    config = load_config("../config/config.yaml")['config']
    processor = VideoProcessor(config)
    processor.process_video()
