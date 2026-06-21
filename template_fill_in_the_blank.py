import os
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

# Hardware device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# System paths for dataset components
TRAIN_DIR = "data/train" 
FRAMES_DIR = os.path.join(TRAIN_DIR, "Pedestrian frame")
LABELS_DIR = os.path.join(TRAIN_DIR, "Pedestrian label")

# [START Config]
# Definition of main SNN and optimization hyperparameters
num_steps = 5           # Time steps for SNN simulation
batch_size = 128        # Batch size
num_epochs = 10         # Training epochs
learning_rate = 1e-3    # Learning rate for optimizer
# [END Config]


class PedestrianDataset(Dataset):
    def __init__(self, frames_dir, labels_dir, transform=None):
        self.frames_dir = frames_dir
        self.labels_dir = labels_dir
        self.transform = transform
        
        # Match PNGs with their corresponding XML files dynamically
        self.filenames = []
        for f in sorted(os.listdir(frames_dir)):
            if f.endswith('.png'):
                base_name = os.path.splitext(f)[0]
                xml_name = f"{base_name}.xml"
                if os.path.exists(os.path.join(labels_dir, xml_name)):
                    self.filenames.append(base_name)
                    
        print(f"Matched {len(self.filenames)} image/label pairs successfully.")
        
    def __len__(self):
        return len(self.filenames)

    def parse_xml(self, xml_path, idx):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        areas = []
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            
            # CRITICAL: Match 'person' from your XML file
            if name == 'person':
                labels.append(1)  # Class 1 = Person (Class 0 is background)
                
                # Extract Bounding Box
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                
                boxes.append([xmin, ymin, xmax, ymax])
                
                # Calculate box area (width * height)
                area = (xmax - xmin) * (ymax - ymin)
                areas.append(area)
        
        # Convert everything to standard PyTorch Tensors
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([idx], dtype=torch.int64)
        target["area"] = torch.as_tensor(areas, dtype=torch.float32)
        target["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64) # Assuming no crowd tags
        
        return target

    def __getitem__(self, idx):
        base_name = self.filenames[idx]
        
        img_path = os.path.join(self.frames_dir, f"{base_name}.png")
        xml_path = os.path.join(self.labels_dir, f"{base_name}.xml")
        
        # Your XML lists depth=1 (Grayscale). 
        # Note: Most torchvision detection models still expect 3 channels (RGB).
        # Convert to "RGB" (replicates the grayscale channel 3 times) to avoid model errors.
        image = Image.open(img_path).convert("RGB") 
        
        target = self.parse_xml(xml_path, idx)
        
        if self.transform:
            image = self.transform(image)
            
        return image, target


# [START Data loading]
# Transforms to handle 256x256 conversion
image_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Instantiate Dataset and Loader
dataset = PedestrianDataset(FRAMES_DIR, LABELS_DIR, transform=image_transforms)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

print(f"Total Images: {len(dataset)} | Training Split: {len(train_dataset)} | Testing Split: {len(test_dataset)}")

def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
# [END Data loading]


# [START Model definition]
class SpikingPedestrianDetector(nn.Module):
    def __init__(self, 
                 beta=0.9, 
                 threshold=1.0, 
                 num_steps=3, 
                 surrogate_slope=25, 
                 grid_size=32):
        super().__init__()
        
        # Store parameters
        self.num_steps = num_steps 
        self.grid_size = grid_size
        
        # Configurable spike gradient surrogate function
        spike_grad = surrogate.fast_sigmoid(slope=surrogate_slope)
        
        # 1. Spiking Convolutional Backbone
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
        
        # 2. Adaptive Pooling (The secret to a dynamic grid_size)
        # This forces the spatial feature map to exactly (grid_size x grid_size)
        # regardless of the original image dimensions or convolution strides.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.grid_size, self.grid_size))
        
        # 3. Grid-Based Detection Head
        # Outputs 5 items per cell: [objectness_logit, xmin, ymin, xmax, ymax]
        self.detector_head = nn.Conv2d(64, 5, kernel_size=1)

    def forward(self, x):
        # Initialize membrane potentials for LIF neurons
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        spk3_sum = None
        
        # Static input emulation
        for step in range(self.num_steps):
            cur1 = self.conv1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            cur3 = self.conv3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            
            # Dynamically initialize accumulator based on conv output shape
            if spk3_sum is None:
                spk3_sum = torch.zeros_like(spk3)
            spk3_sum += spk3 
            
        # Average spike feature map across time
        feat = spk3_sum / self.num_steps
        
        # Morph the feature map to the user-defined grid_size
        feat_pooled = self.adaptive_pool(feat)
        
        # Output shape: (Batch, 5, grid_size, grid_size)
        grid_predictions = self.detector_head(feat_pooled)
        
        return grid_predictions


# Model, Optimizer, and Loss functions
model = SpikingPedestrianDetector(
    beta=0.85,               # Membrane potential decay
    threshold=0.9,           # Lower threshold = easier spiking
    num_steps=num_steps,             # More time steps for accuracy
    surrogate_slope=20,      # Gradient steepness for backprop
    grid_size=64             # Change grid resolution (e.g., 16, 32, 64)
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Losses
bbox_loss_fn = nn.MSELoss() # bounding boxes
cls_loss_fn = nn.BCEWithLogitsLoss() # object classification

grid_size = model.grid_size  # Spatial width/height of the final feature map layer
orig_w, orig_h = 346, 260  # Native dataset dimensions
# [END Model definition]

# [START Training loop]
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    for images, targets in train_loader:
        if isinstance(targets, dict):
            targets = (targets,)
            
        if isinstance(images, (tuple, list)):
            images = torch.stack(images).to(device)
        elif isinstance(images, torch.Tensor):
            if images.dim() == 3:
                images = images.unsqueeze(0) 
            images = images.to(device)

        # 1. Forward Pass through SNN -> shape: (Batch, 5, 32, 32)
        pred_grid = model(images)
        
        # 2. Dynamically build the target grid map for the entire batch
        batch_size = images.size(0)
        true_grid = torch.zeros((batch_size, 5, grid_size, grid_size), device=device)
        
        for i in range(len(targets)):
            boxes = targets[i]["boxes"]
            if boxes.size(0) == 0:
                continue
                
            for box in boxes:
                # Normalize box coordinates between [0, 1]
                xmin = box[0].item() / orig_w
                ymin = box[1].item() / orig_h
                xmax = box[2].item() / orig_w
                ymax = box[3].item() / orig_h
                
                # Compute box center point
                cx = (xmin + xmax) / 2.0
                cy = (ymin + ymax) / 2.0
                
                # Determine target cell indices inside the 32x32 matrix
                gx = int(cx * grid_size)
                gy = int(cy * grid_size)
                
                # Clamp boundaries defensively
                gx = max(0, min(grid_size - 1, gx))
                gy = max(0, min(grid_size - 1, gy))
                
                # Set target presence and normalized dimensions at the mapped cell
                true_grid[i, 0, gy, gx] = 1.0
                true_grid[i, 1, gy, gx] = xmin
                true_grid[i, 2, gy, gx] = ymin
                true_grid[i, 3, gy, gx] = xmax
                true_grid[i, 4, gy, gx] = ymax
        
        # 3. Compute Grid Losses
        # A. Objectness Classification Loss (Evaluated over the whole grid layout)
        pred_logits = pred_grid[:, 0, :, :]
        true_logits = true_grid[:, 0, :, :]
        loss_cls = cls_loss_fn(pred_logits, true_logits)
        
        # B. Masked Box Regression Loss (Evaluated only where true pedestrians exist)
        mask = (true_grid[:, 0, :, :] == 1.0)
        if mask.sum() > 0:
            # Gather coordinates across grid masks -> Shape: (Num_Objects, 4)
            pred_boxes = pred_grid[:, 1:5, :, :].permute(0, 2, 3, 1)[mask]
            true_boxes = true_grid[:, 1:5, :, :].permute(0, 2, 3, 1)[mask]
            loss_box = bbox_loss_fn(pred_boxes, true_boxes)
        else:
            loss_box = torch.tensor(0.0, device=device)
            
        # Combine losses (scale loss_box slightly to normalize coordinate variations)
        total_loss = loss_cls + 2.0 * loss_box
        
        # 4. Optimization step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        epoch_loss += total_loss.item()
        
    print(f"Epoch [{epoch+1}/{num_epochs}] | Total Grid Loss: {epoch_loss:.4f}")
# [END Training loop]


def calculate_iou(boxA, boxB):
    """
    Calculates Intersection over Union (IoU) between two boxes.
    Boxes format: [xmin, ymin, xmax, ymax]
    """
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBAArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute Union area
    unionArea = boxAArea + boxBAArea - interArea

    if unionArea == 0:
        return 0.0
        
    return interArea / float(unionArea)

# [START Testing]
def test_model_accuracy(model, data_loader, device):
    model.eval() 
    
    total_images = 0
    correct_classifications = 0
    total_iou = 0.0
    total_true_boxes = 0
    
    orig_w, orig_h = 346, 260
    
    with torch.no_grad():
        for images, targets in data_loader:
            if isinstance(targets, dict):
                targets = (targets,)
                
            if isinstance(images, (tuple, list)):
                images = torch.stack(images).to(device)
            elif isinstance(images, torch.Tensor):
                if images.dim() == 3: images = images.unsqueeze(0)
                images = images.to(device)
                
            # Forward Pass through grid-based SNN
            pred_grid = model(images)
            
            for i in range(len(targets)):
                total_images += 1
                
                # Extract ground truth boxes for this image
                true_boxes = targets[i]["boxes"].cpu().numpy()
                actual_has_person = (len(true_boxes) > 0)
                
                # Parse grid predictions where logit >= 0.0 (probability >= 50%)
                logits = pred_grid[i, 0, :, :]
                boxes_map = pred_grid[i, 1:5, :, :]
                
                gy_indices, gx_indices = torch.where(logits >= 0.0)
                pred_boxes_list = []
                for gy, gx in zip(gy_indices, gx_indices):
                    pred_boxes_list.append(boxes_map[:, gy, gx].cpu().numpy())
                    
                has_person_prediction = (len(pred_boxes_list) > 0)
                
                # Frame-level classification accuracy check
                if has_person_prediction == actual_has_person:
                    correct_classifications += 1
                
                # Multi-object IoU calculation
                if actual_has_person:
                    for t_box in true_boxes:
                        total_true_boxes += 1
                        # Scale ground truth box to [0, 1] bounds
                        t_box_scaled = [
                            t_box[0] / orig_w,
                            t_box[1] / orig_h,
                            t_box[2] / orig_w,
                            t_box[3] / orig_h
                        ]
                        
                        # Find maximum IoU matching among active predicted grids
                        max_iou = 0.0
                        for p_box in pred_boxes_list:
                            iou = calculate_iou(p_box, t_box_scaled)
                            if iou > max_iou:
                                max_iou = iou
                        total_iou += max_iou
                        
    # Calculate Metrics
    cls_accuracy = (correct_classifications / total_images) * 100
    avg_iou = (total_iou / total_true_boxes) * 100 if total_true_boxes > 0 else 0.0
    
    print("\n================ TEST SPLIT RESULTS ================")
    print(f"Total Evaluated Images:       {total_images}")
    print(f"Pedestrian Presence Accuracy:  {cls_accuracy:.2f}%")
    print(f"Average Bounding Box IoU:     {avg_iou:.2f}%")
    print("====================================================")
    
    return cls_accuracy, avg_iou
# [END Testing]

test_model_accuracy(model, test_loader, device)