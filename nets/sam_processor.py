import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Import SAM related modules
from .segment_anything import sam_model_registry, SamPredictor

class SAMProcessor(nn.Module):
    def __init__(self, checkpoint_path, model_type='vit_b', device='cuda'):
        """
        SAM processor

        Args:
            checkpoint_path: SAM model checkpoint path
            model_type: Model type ('vit_h', 'vit_l', 'vit_b')
            device: Device ('cuda' or 'cpu')
        """
        super(SAMProcessor, self).__init__()
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Lazy load SAM model
        self.sam = None
        self.predictor = None

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # print(f"SAM processor initialization completed, model will be loaded on first use")
    
    def _load_model(self):
        """Lazy load SAM model"""
        if self.sam is None:
            # print(f"Loading SAM model: {self.model_type} from {self.checkpoint_path}")
            try:
                self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
                self.sam.to(device=self.device)
                self.predictor = SamPredictor(self.sam)
                print(f"SAM model loaded successfully, using device: {self.device}")
            except Exception as e:
                print(f"SAM model loading failed: {e}")
                raise e
    
    def forward(self, image, peak_points):
        """
        Use SAM to process images and generate pseudo-labels

        Args:
            image: Input image (B, 3, H, W) range [0, 1]
            peak_points: Peak point list, each element is (N, 3), where N is peak count, 3 represents (x, y, score)

        Returns:
            pseudo_labels: Pseudo-label list, each element is (N, 5), where 5 represents (x_min, y_min, w, h, class)
        """
        # Load model (if not loaded yet)
        self._load_model()

        B, C, H, W = image.shape
        pseudo_labels = []

        # Convert image to numpy format and adjust range to [0, 255]
        image_np = (image.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for b in range(B):
            try:
                # Process current batch image
                img = image_np[b]  # (H, W, 3)

                # Set image
                self.predictor.set_image(img)

                # Get current batch peak points
                points = peak_points[b]  # (N, 3)

                if points.size(0) == 0:
                    # If no peak points, add empty pseudo-labels
                    pseudo_labels.append(torch.empty((0, 5), device=image.device))
                    continue

                # Limit peak point count to reduce memory usage
                max_points = min(points.size(0), 10)  # Process at most 10 points
                if points.size(0) > max_points:
                    # Select points with highest scores
                    _, indices = torch.topk(points[:, 2], max_points)
                    points = points[indices]

                # Extract coordinates and labels
                coords = points[:, :2].cpu().numpy()  # (N, 2)

                # All points are marked as foreground points
                labels = np.ones(coords.shape[0], dtype=np.int32)  # (N,)

                # Use SAM to predict masks, reduce multimask_output to save memory
                masks, iou_predictions, _ = self.predictor.predict(
                    point_coords=coords,
                    point_labels=labels,
                    multimask_output=False,  # Output only one best mask
                    return_logits=False
                )

                # masks: (H, W) if multimask_output=False
                # iou_predictions: (1,)

                # Convert masks to bounding boxes
                batch_boxes = []
                if masks.ndim == 2:  # Single mask case
                    mask = masks  # (H, W)

                    # Find mask bounding box
                    pos = np.where(mask)
                    if len(pos[0]) > 0:
                        y_min, y_max = np.min(pos[0]), np.max(pos[0])
                        x_min, x_max = np.min(pos[1]), np.max(pos[1])

                        # Calculate width and height
                        w = x_max - x_min
                        h = y_max - y_min

                        # Keep only reasonable bounding boxes
                        if w > 5 and h > 5:  # Minimum size threshold
                            # Create pseudo-label (x_min, y_min, w, h, class)
                            pseudo_label = [float(x_min), float(y_min), float(w), float(h), 0.0]
                            batch_boxes.append(pseudo_label)
                else:  # Multi-mask case
                    for i in range(masks.shape[0]):
                        mask = masks[i]  # (H, W)

                        # Find mask bounding box
                        pos = np.where(mask)
                        if len(pos[0]) > 0:
                            y_min, y_max = np.min(pos[0]), np.max(pos[0])
                            x_min, x_max = np.min(pos[1]), np.max(pos[1])

                            # Calculate width and height
                            w = x_max - x_min
                            h = y_max - y_min

                            # Keep only reasonable bounding boxes
                            if w > 5 and h > 5:  # Minimum size threshold
                                # Create pseudo-label (x_min, y_min, w, h, class)
                                pseudo_label = [float(x_min), float(y_min), float(w), float(h), 0.0]
                                batch_boxes.append(pseudo_label)

                if len(batch_boxes) > 0:
                    pseudo_labels.append(torch.tensor(batch_boxes, dtype=torch.float32, device=image.device))
                else:
                    pseudo_labels.append(torch.empty((0, 5), device=image.device))

            except torch.cuda.OutOfMemoryError:
                pseudo_labels.append(torch.empty((0, 5), device=image.device))
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            except Exception as e:
                pseudo_labels.append(torch.empty((0, 5), device=image.device))
                continue

        return pseudo_labels

# Test code
if __name__ == "__main__":
    # Create SAM processor instance
    # Note: Need to download SAM model checkpoint first
    # processor = SAMProcessor('nets/segment_anything/pretrained/sam_vit_h_4b8939.pth', model_type='vit_h')

    # Create test image and peak points
    B, C, H, W = 1, 3, 512, 512
    image = torch.randn(B, C, H, W)
    peak_points = [torch.tensor([[100, 100, 0.9], [200, 200, 0.8]], device=image.device)]

    print(f"Image shape: {image.shape}")
    print(f"Peak points: {peak_points[0].shape}")

    # Note: Actual testing requires SAM model checkpoint
    print("SAM processor implementation completed. Actual testing requires SAM model checkpoint.")