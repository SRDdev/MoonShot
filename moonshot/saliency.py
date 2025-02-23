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

# Paths Configuration
WEIGHT_PATH = '../weights/ucf_d123s.pth'
VIDEO_PATH = '../example_data/gen2_1.mp4'
SAVE_PATH = '../output/'


def generate_video_from_frames(frames_dir, output_video_path, fps=30):
    """
    Converts a sequence of image frames into a video using ffmpeg.
    """
    if not os.path.exists(frames_dir) or not os.listdir(frames_dir):
        raise FileNotFoundError(f"No frames found in directory: {frames_dir}")
    
    command = [
        "ffmpeg", "-framerate", str(fps), "-i", os.path.join(frames_dir, "%d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", output_video_path
    ]
    subprocess.run(command, check=True)
    print(f"Video saved at {output_video_path}")


def extract_config_from_weights(weight_path):
    """Maps weight file to configuration details."""
    config_mappings = {
        'ucf_d123s.pth': ('config/eval_config_single.yaml', 1, 'single'),
        'ucf_d123m.pth': ('config/eval_config_multi_ucf.yaml', 1, 'multi'),
    }
    return config_mappings.get(os.path.basename(weight_path), ValueError('Unknown weight file'))


class VideoDataset(Dataset):
    """Loads video frames and applies preprocessing for model input."""
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


def extract_bounding_boxes(image, percentile=95, max_bboxes=5):
    """Extracts top bounding boxes from heatmap based on intensity percentile."""
    threshold = np.percentile(image, percentile)
    mask = (image >= threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = sorted(
        (cv2.boundingRect(contour) for contour in contours),
        key=lambda bbox: bbox[2] * bbox[3], reverse=True
    )
    return [(x, y, x + w, y + h) for x, y, w, h in bboxes[:max_bboxes]]


def draw_bounding_boxes(image, bboxes, color=(0, 0, 255), thickness=2):
    """Draws bounding boxes on an image."""
    for x_min, y_min, x_max, y_max in bboxes:
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)


def extract_video_frames(video_path, save_dir):
    """Extracts frames from video and saves them as images."""
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


def overlay_bboxes_on_original(original_dir, bbox_dir, output_dir):
    """Transfers bounding boxes from heatmap images to original video frames."""
    os.makedirs(output_dir, exist_ok=True)
    for orig_frame, bbox_frame in zip(sorted(os.listdir(original_dir)), sorted(os.listdir(bbox_dir))):
        orig_img = cv2.imread(os.path.join(original_dir, orig_frame))
        bbox_img = cv2.imread(os.path.join(bbox_dir, bbox_frame), cv2.IMREAD_GRAYSCALE)
        bboxes = extract_bounding_boxes(bbox_img)
        draw_bounding_boxes(orig_img, bboxes)
        cv2.imwrite(os.path.join(output_dir, orig_frame), orig_img)
    print(f"Processed frames saved in {output_dir}")


def process_video():
    """Main function to process the video and generate output with bounding boxes."""
    config_path, reduced_channel, mode = extract_config_from_weights(WEIGHT_PATH)
    config = cfg.get_config(config_path)
    
    # Load Model
    model = FastSalA(reduced_channel, config.MODEL_SETUP.DECODER, config.MODEL_SETUP.SINGLE, 
                     d1_last=config.MODEL_SETUP.D1_LAST, force_multi=config.MODEL_SETUP.FORCE_MULTI, 
                     n_output=config.MODEL_SETUP.OUTPUT_SIZE)
    model.load_state_dict(t.load(WEIGHT_PATH, map_location='cuda:0')['student_model'])
    model.cuda().eval()
    
    # Prepare directories
    original_frames = os.path.join(SAVE_PATH, "original_frames")
    heatmap_bboxes = os.path.join(SAVE_PATH, "heatmap_with_bbox")
    output_bboxes = os.path.join(SAVE_PATH, "original_frames_with_bbox")
    
    extract_video_frames(VIDEO_PATH, original_frames)
    dataloader = DataLoader(VideoDataset(VIDEO_PATH, 16, 1), batch_size=20, shuffle=False, num_workers=4, pin_memory=True)
    
    os.makedirs(heatmap_bboxes, exist_ok=True)
    with t.no_grad():
        for x, frame_idx, _ in dataloader:
            pred_maps = model(x.cuda())[0].cpu().numpy()
            for img, fid in zip(pred_maps, frame_idx.numpy()[:, -1]):
                processed_img = post_process_png(img[0])
                draw_bounding_boxes(processed_img, extract_bounding_boxes(processed_img))
                cv2.imwrite(os.path.join(heatmap_bboxes, f'{fid}.png'), processed_img)
    
    overlay_bboxes_on_original(original_frames, heatmap_bboxes, output_bboxes)
    generate_video_from_frames(output_bboxes, os.path.join(SAVE_PATH, "output_with_bbox.mp4"), fps=30)
    print("Processing complete.")


if __name__ == '__main__':
    process_video()
