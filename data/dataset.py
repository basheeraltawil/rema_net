"""
Action Recognition Dataset
===========================
Loads video clips and returns:
    - RGB tensors    (colour frames)
    - Flow tensors   (optical flow)
    - Caption embeddings (BLIP → BERT-tiny)
    - Label          (action class index)

Results are cached to disk after the first extraction so subsequent
epochs / runs load instantly.
"""

import os
import heapq

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from config.default import CAPTION_DIM
from models.streams import CaptionStream


# ==============================================================================
# Shared caption model (loaded once, reused everywhere)
# ==============================================================================

_shared_caption_model = None


def get_caption_model():
    """Return a singleton CaptionStream so BLIP+BERT are loaded only once."""
    global _shared_caption_model
    if _shared_caption_model is None:
        _shared_caption_model = CaptionStream()
    return _shared_caption_model


# ==============================================================================
# Dataset
# ==============================================================================

class ActionRecognitionDataset(Dataset):
    """
    Args:
        data_dir        : root folder containing video files
        annotation_file : CSV with 'clip_path' and 'label' columns,
                          *or* a pandas DataFrame
        num_segments    : how many frames to sample from each video
        transform       : image transforms for RGB frames
        flow_transform  : image transforms for optical flow frames
    """

    TARGET_RGB_SHAPE  = (3, 224, 224)
    TARGET_FLOW_SHAPE = (2, 224, 224)

    def __init__(
        self,
        data_dir,
        annotation_file,
        num_segments=16,
        transform=None,
        flow_transform=None,
    ):
        self.data_dir       = data_dir
        self.num_segments   = num_segments
        self.transform      = transform
        self.flow_transform = flow_transform if flow_transform else transform

        # Accept CSV path or DataFrame
        if isinstance(annotation_file, pd.DataFrame):
            annotations = annotation_file
        else:
            annotations = pd.read_csv(annotation_file)

        # Keep only samples whose video files exist on disk
        self.valid_samples = []
        for _, row in annotations.iterrows():
            video_path = os.path.join(data_dir, row["clip_path"].lstrip("/"))
            if os.path.exists(video_path):
                self.valid_samples.append((video_path, int(row["label"])))

        print(f"Found {len(self.valid_samples)} valid video samples.")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        video_path, label = self.valid_samples[idx]

        rgb, flow, embeddings = self._extract_frames_and_flow(video_path)

        if rgb is None or flow is None:
            print(f"[Skip] Extraction failed: {video_path}")
            return self.__getitem__((idx + 1) % len(self))

        if len(rgb) != self.num_segments or len(flow) != self.num_segments:
            print(f"[Skip] Frame count mismatch: {video_path}")
            return self.__getitem__((idx + 1) % len(self))

        return rgb, flow, embeddings, label

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_zero_tensors(self):
        T = self.num_segments
        return (
            torch.zeros(T, *self.TARGET_RGB_SHAPE),
            torch.zeros(T, *self.TARGET_FLOW_SHAPE),
            torch.zeros(T, CAPTION_DIM),
        )

    def _sample_frame_indices(self, total_frames):
        return np.linspace(0, total_frames - 1, self.num_segments, dtype=int)

    def _extract_frames_and_flow(self, video_path):
        """
        1. Check cache → return if found
        2. Open video, sample N frames
        3. Extract RGB tensor + optical flow per frame
        4. Caption the highest-motion frames with BLIP
        5. Pad / trim to num_segments
        6. Save cache
        """

        # --- cache ---
        cache_path = video_path + "_cache.pt"
        if os.path.exists(cache_path):
            try:
                saved = torch.load(cache_path, weights_only=False)
                return saved["rgb_frames"], saved["flow_frames"], saved["embeddings"]
            except Exception as e:
                print(f"Cache load error ({video_path}): {e}")

        # --- open video ---
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ret, first_frame = cap.read()
        if not ret or total_frames <= 0:
            print(f"Cannot open video: {video_path}")
            cap.release()
            return self._get_zero_tensors()

        frame_indices = sorted(self._sample_frame_indices(total_frames))
        current_idx   = 0
        prev_gray     = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        rgb_frames    = []
        flow_frames   = []
        rgb_pil_list  = []
        motion_scores = []

        # --- extract frames ---
        for target_idx in frame_indices:
            while current_idx < target_idx:
                if not cap.grab():
                    break
                current_idx += 1

            ret, frame = cap.retrieve()
            if not ret:
                ret, frame = cap.read()
            if not ret:
                continue
            current_idx += 1

            # RGB
            rgb_image  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_pil    = Image.fromarray(rgb_image)
            rgb_tensor = (
                self.transform(rgb_pil)
                if self.transform
                else transforms.ToTensor()(rgb_pil)
            )
            if rgb_tensor.shape != self.TARGET_RGB_SHAPE:
                rgb_tensor = F.interpolate(
                    rgb_tensor.unsqueeze(0),
                    size=self.TARGET_RGB_SHAPE[1:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            rgb_frames.append(rgb_tensor)
            rgb_pil_list.append(rgb_pil)

            # Optical flow
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow_field = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            motion = np.sqrt(flow_field[..., 0] ** 2 + flow_field[..., 1] ** 2).mean()
            motion_scores.append(motion)

            def channel_to_tensor(channel):
                normed = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX).astype(
                    np.uint8
                )
                return transforms.ToTensor()(Image.fromarray(normed))

            flow_tensor = torch.cat(
                [
                    channel_to_tensor(flow_field[..., 0]),
                    channel_to_tensor(flow_field[..., 1]),
                ],
                dim=0,
            )

            if self.flow_transform:
                for t in self.flow_transform.transforms:
                    if isinstance(t, transforms.Normalize):
                        flow_tensor = t(flow_tensor)
                        break

            if flow_tensor.shape != self.TARGET_FLOW_SHAPE:
                flow_tensor = F.interpolate(
                    flow_tensor.unsqueeze(0),
                    size=self.TARGET_FLOW_SHAPE[1:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            flow_frames.append(flow_tensor)
            prev_gray = gray

        cap.release()

        if not rgb_frames:
            print(f"No frames extracted: {video_path}")
            return self._get_zero_tensors()

        # --- caption embeddings (top-k motion frames) ---
        k     = max(1, len(motion_scores) // 4)
        top_k = set(
            heapq.nlargest(k, range(len(motion_scores)), key=lambda i: motion_scores[i])
        )

        caption_model  = get_caption_model()
        embeddings     = []
        last_embedding = None

        for i in range(len(rgb_pil_list)):
            if i in top_k or last_embedding is None:
                with torch.no_grad():
                    embedding, _ = caption_model(rgb_pil_list[i])
                last_embedding = embedding
            embeddings.append(last_embedding)

        # --- pad / trim ---
        while len(rgb_frames) < self.num_segments:
            rgb_frames.append(torch.zeros(*self.TARGET_RGB_SHAPE))
            flow_frames.append(torch.zeros(*self.TARGET_FLOW_SHAPE))
            embeddings.append(torch.zeros_like(embeddings[0]))

        rgb_frames  = torch.stack(rgb_frames[: self.num_segments])
        flow_frames = torch.stack(flow_frames[: self.num_segments])
        embeddings  = torch.stack(embeddings[: self.num_segments])

        # --- save cache ---
        try:
            torch.save(
                {
                    "rgb_frames": rgb_frames,
                    "flow_frames": flow_frames,
                    "embeddings": embeddings,
                },
                cache_path,
            )
        except Exception as e:
            print(f"Cache save error ({video_path}): {e}")

        return rgb_frames, flow_frames, embeddings


# ==============================================================================
# Collate function  (for DataLoader)
# ==============================================================================

def custom_collate_fn(batch):
    """
    Combines individual samples into a batch.
    Drops None samples and fixes minor shape mismatches.
    """
    batch = [s for s in batch if s is not None]
    if len(batch) == 0:
        return None

    rgb_list, flow_list, emb_list, label_list = zip(*batch)

    def fix_shapes(tensor_list, spatial_dims):
        ref_shape = tensor_list[0].shape
        if all(t.shape == ref_shape for t in tensor_list):
            return list(tensor_list)

        fixed = []
        for t in tensor_list:
            if t.shape != ref_shape:
                t = F.interpolate(
                    t.unsqueeze(0),
                    size=ref_shape[1:],
                    mode="bilinear" if spatial_dims == 2 else "linear",
                    align_corners=False,
                ).squeeze(0)
            fixed.append(t)
        return fixed

    rgb_list  = fix_shapes(rgb_list, spatial_dims=2)
    flow_list = fix_shapes(flow_list, spatial_dims=2)
    emb_list  = fix_shapes(emb_list, spatial_dims=1)

    return (
        torch.stack(rgb_list),
        torch.stack(flow_list),
        torch.stack(emb_list),
        torch.tensor(label_list),
    )
