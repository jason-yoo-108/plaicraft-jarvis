import cv2
import torch
from transformers import AutoProcessor


def resize_image(img, target_resolution=(224, 224)):
    # For your sanity, do not resize with any function than INTER_LINEAR
    # Only resize if needed (target_resolution is flipped for opencv)
    if img.shape[:2] != target_resolution[::-1]:
        img = cv2.resize(img, target_resolution, interpolation=cv2.INTER_LINEAR)
    return img


class PlaicraftStateTokenizer:
    def __init__(self):
        # Call the parent constructor to initialize common configurations
        self.tokenizer = AutoProcessor.from_pretrained("CraftJarvis/JarvisVLA-Qwen2-VL-7B")

    def format_jarvis_input(self, frames, text_in):
        """
        Accept plaicraft video frames (integer array) and text input, and format them for JarvisVLA.
        """
        # Resize frames to the required resolution
        frames = [resize_image(frame) for frame in frames]

        # Convert frames to tensor and normalize
        frames_tensor = torch.tensor(frames).permute(0, 3, 1, 2).float() / 255.0

        # Tokenize the text input
        text_tokens = self.tokenizer(text_in, return_tensors="pt")

        return {
            "images": frames_tensor,
            "text": text_tokens.input_ids,
            "attention_mask": text_tokens.attention_mask,
        }
