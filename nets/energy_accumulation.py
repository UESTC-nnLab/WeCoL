import torch
import torch.nn as nn
import numpy as np
import cv2
from scipy.signal import convolve
from skimage.filters.rank import entropy
from skimage.morphology import disk

class EnergyProcessor(nn.Module):
    def __init__(self, alpha=0.8, threshold=0.2):
        """
        Infrared small target enhancement processor

        Args:
            alpha: Energy accumulation decay factor
            threshold: Background suppression threshold
        """
        super(EnergyProcessor, self).__init__()
        self.alpha = alpha
        self.threshold = threshold
    
    def butterworth_highpass(self, img, d0=30, n=2):
        """
        Perform frequency domain Butterworth high-pass filtering on single frame image

        Args:
            img: Single frame image (H, W)
            d0: Cutoff frequency
            n: Filter order

        Returns:
            Filtered image (H, W)
        """
        h, w = img.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        du, dv = u - w//2, v - h//2
        D = np.sqrt(du**2 + dv**2)
        
        # Calculate Butterworth high-pass filter
        H = 1 / (1 + (d0 / (D + 1e-5))**(2 * n))

        # Frequency domain transform
        img_dft = np.fft.fftshift(np.fft.fft2(img))
        img_hp = np.real(np.fft.ifft2(np.fft.ifftshift(img_dft * H)))
        
        return img_hp

    def highpass_filter(self, img_seq, d0=30, n=2):
        """
        Process multiple frames (frames, H, W)

        Args:
            img_seq: Image sequence (frames, H, W)

        Returns:
            Filtered image sequence (frames, H, W)
        """
        frames, h, w = img_seq.shape
        output_seq = np.zeros((frames, h, w))

        for i in range(frames):
            output_seq[i] = self.butterworth_highpass(img_seq[i], d0, n)
        
        return output_seq

    def compute_saliency(self, img):
        """
        Calculate Laplacian entropy saliency

        Args:
            img: Input image (H, W)

        Returns:
            Saliency map (H, W)
        """
        # Ensure image is uint8 type
        if img.dtype != np.uint8:
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        entropy_map = entropy(np.abs(laplacian).astype(np.uint8), disk(3))
        return entropy_map

    def energy_accumulation(self, saliency_maps):
        """
        Multi-frame energy accumulation to enhance target signals

        Args:
            saliency_maps: Saliency map sequence [frames, (H, W)]

        Returns:
            Accumulated energy map (H, W)
        """
        accumulated_energy = np.zeros_like(saliency_maps[0])
        for t, smap in enumerate(saliency_maps):
            accumulated_energy += (self.alpha**t) * (smap**2)
        return accumulated_energy / (np.max(accumulated_energy) + 1e-8)  # Normalization

    def background_suppression(self, energy_map):
        """
        Background suppression to reduce noise interference

        Args:
            energy_map: Energy map (H, W)

        Returns:
            Suppressed energy map (H, W)
        """
        energy_map[energy_map < self.threshold] = 0  # Set threshold
        return cv2.medianBlur((energy_map * 255).astype(np.uint8), 3)

    def forward(self, features):
        """
        Process feature map sequence

        Args:
            features: Feature map sequence (B, frames, C, H, W)

        Returns:
            Enhanced feature map (B, C, H, W)
        """
        B, frames, C, H, W = features.shape
        enhanced_features = []
        
        for b in range(B):
            # Process each batch
            batch_features = features[b]  # (frames, C, H, W)

            # Convert feature maps to image sequence for processing
            # Here we use the first channel for processing
            img_seq = batch_features[:, 0, :, :].detach().cpu().numpy()  # (frames, H, W)

            # 1. High-pass filtering
            filtered_seq = self.highpass_filter(img_seq)

            # 2. Calculate saliency for each frame
            saliency_maps = [self.compute_saliency(frame) for frame in filtered_seq]

            # 3. Energy accumulation
            energy_map = self.energy_accumulation(saliency_maps)

            # 4. Background suppression
            filtered_map = self.background_suppression(energy_map)

            # Convert processed energy map to tensor and adjust dimensions
            energy_tensor = torch.from_numpy(filtered_map).float().to(features.device) / 255.0
            energy_tensor = energy_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            energy_tensor = torch.nn.functional.interpolate(energy_tensor, size=(H, W), mode='bilinear', align_corners=False)
            energy_tensor = energy_tensor.squeeze(0).squeeze(0)  # (H, W)

            # Apply energy map to features of all channels
            enhanced_feat = batch_features[-1] * energy_tensor.unsqueeze(0)  # (C, H, W)
            enhanced_features.append(enhanced_feat)

        # Stack results from all batches
        enhanced_features = torch.stack(enhanced_features, dim=0)  # (B, C, H, W)
        
        return enhanced_features

# Test code
if __name__ == "__main__":
    # Create processor instance
    processor = EnergyProcessor()

    # Create test features
    B, frames, C, H, W = 2, 5, 128, 64, 64
    features = torch.randn(B, frames, C, H, W)

    # Process features
    enhanced_features = processor(features)

    print(f"Input shape: {features.shape}")
    print(f"Output shape: {enhanced_features.shape}")

    # Verify output shape
    expected_shape = (B, C, H, W)
    assert enhanced_features.shape == expected_shape, f"Expected {expected_shape}, got {enhanced_features.shape}"

    print("PSE processing test passed!")