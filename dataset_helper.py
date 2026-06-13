# Inside dataset_helper.py
import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from PIL import Image

ORIG_W = 346     
ORIG_H = 260     

class PedestrianDataset(Dataset):
    def __init__(self, frames_dir, labels_dir, transform=None):
        self.frames_dir = frames_dir
        self.labels_dir = labels_dir
        self.transform  = transform
        self.filenames  = []
        for f in sorted(os.listdir(frames_dir)):
            if f.endswith('.png'):
                base = os.path.splitext(f)[0]
                if os.path.exists(os.path.join(labels_dir, f"{base}.xml")):
                    self.filenames.append(base)

    def __len__(self):
        return len(self.filenames)

    def parse_xml(self, xml_path, idx):
        root   = ET.parse(xml_path).getroot()
        boxes, labels, areas = [], [], []
        for obj in root.findall('object'):
            if obj.find('name').text == 'person':
                labels.append(1)
                bb   = obj.find('bndbox')
                xmin = float(bb.find('xmin').text)
                ymin = float(bb.find('ymin').text)
                xmax = float(bb.find('xmax').text)
                ymax = float(bb.find('ymax').text)
                boxes.append([xmin, ymin, xmax, ymax])
                areas.append((xmax - xmin) * (ymax - ymin))
        return {
            "boxes":    torch.as_tensor(boxes,  dtype=torch.float32),
            "labels":   torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx],     dtype=torch.int64),
            "area":     torch.as_tensor(areas,  dtype=torch.float32),
            "iscrowd":  torch.zeros(len(labels), dtype=torch.int64),
        }

    def __getitem__(self, idx):
        base   = self.filenames[idx]
        image  = Image.open(os.path.join(self.frames_dir, f"{base}.png")).convert("RGB")
        target = self.parse_xml(os.path.join(self.labels_dir, f"{base}.xml"), idx)
        if self.transform:
            image = self.transform(image)
        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))