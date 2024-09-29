import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

class FeatureExtractor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.resnet50(pretrained=True).to(self.device)
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    def extract(self, video_path, frame_step=5, num_frames=10):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.arange(0, frame_count, step=frame_step, dtype=int)
        frame_indices = frame_indices[:num_frames]
        features = []

        for idx in tqdm(range(frame_count), desc=f"Извлечение кадров из {video_path}"):
            ret, frame = cap.read()
            if not ret:
                break
            if idx in frame_indices:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                input_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = self.model(input_tensor)
                features.append(output.cpu().numpy())

        cap.release()
        if features:
            video_features = np.mean(features, axis=0)
            return video_features.flatten()
        else:
            return None
