import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PeakPointGenerator(nn.Module):
    def __init__(self, target_size=(512, 512), kernel_size=16, target_count=20):
        """
        Peak point generator

        Args:
            target_size: Target image size
            kernel_size: MaxPool kernel size
            target_count: Target peak point count
        """
        super(PeakPointGenerator, self).__init__()
        self.target_size = target_size
        self.kernel_size = kernel_size
        self.target_count = target_count
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        
    def forward(self, features, K=None):
        """
        Generate peak points

        Args:
            features: Input feature map (B, C, H, W)
            K: Real target count

        Returns:
            peaks: Peak point list, each element is (B, N, 3), where N is peak count, 3 represents (x, y, score)
        """
        B, C, H, W = features.shape
        
        # 1. Upsample to original image size
        upsampled_features = F.interpolate(features, size=self.target_size, mode='bilinear', align_corners=False)

        # 2. Apply MaxPool to each channel to find local maxima
        # First apply global average pooling to get activation values for each position
        activation_map = torch.mean(upsampled_features, dim=1)  # (B, H, W)

        # Apply MaxPool to ensure output size matches input
        pooled_map = F.max_pool2d(activation_map.unsqueeze(1),
                                 kernel_size=self.kernel_size,
                                 stride=1,
                                 padding=self.kernel_size//2,
                                 ceil_mode=False).squeeze(1)  # (B, H, W)

        # If sizes don't match, perform cropping
        if pooled_map.shape != activation_map.shape:
            min_h = min(pooled_map.shape[1], activation_map.shape[1])
            min_w = min(pooled_map.shape[2], activation_map.shape[2])
            pooled_map = pooled_map[:, :min_h, :min_w]
            activation_map = activation_map[:, :min_h, :min_w]
            local_maxima = (pooled_map == activation_map) & (activation_map > 0)  # (B, H, W)
        else:
            # Find local maxima points (where MaxPool output equals original value)
            local_maxima = (pooled_map == activation_map) & (activation_map > 0)  # (B, H, W)
        
        peaks = []
        for b in range(B):
            # Get current batch local maxima coordinates
            coords = torch.nonzero(local_maxima[b], as_tuple=False)  # (N, 2) where N is number of peaks
            if coords.size(0) == 0:
                # If no peak points found, return empty tensor
                peaks.append(torch.empty((0, 3), device=features.device))
                continue

            # Get corresponding activation values
            scores = activation_map[b, coords[:, 0], coords[:, 1]]  # (N,)

            # Combine coordinates and scores
            peak_data = torch.cat([coords.float(), scores.unsqueeze(1)], dim=1)  # (N, 3) where 3 is (y, x, score)

            # 2. Double filtering: sorting based on activation values and threshold filtering
            # Calculate dynamic threshold τ = ζ + δ (mean + standard deviation)
            mean_activation = torch.mean(activation_map[b])
            std_activation = torch.std(activation_map[b])
            threshold = mean_activation + std_activation

            # Filter points below threshold
            valid_mask = peak_data[:, 2] >= threshold
            filtered_peaks = peak_data[valid_mask]

            if filtered_peaks.size(0) == 0:
                # If no points after filtering, use original peak points
                filtered_peaks = peak_data

            # Sort by activation value in descending order
            sorted_indices = torch.argsort(filtered_peaks[:, 2], descending=True)
            sorted_peaks = filtered_peaks[sorted_indices]

            # 3. Non-maximum suppression
            nms_peaks = self._nms(sorted_peaks, self.kernel_size // 2)

            # 4. Control peak point count
            if K is not None:
                # Use real K value
                if isinstance(K, torch.Tensor):
                    final_count = K[b].item()
                elif isinstance(K, list):
                    # K is a list of tensors or scalars
                    k_val = K[b]
                    if isinstance(k_val, torch.Tensor):
                        final_count = k_val.item()
                    else:
                        final_count = int(k_val)
                else:
                    final_count = int(K)
            else:
                # Use default target count
                final_count = self.target_count

            if nms_peaks.size(0) > final_count:
                nms_peaks = nms_peaks[:final_count]

            # Adjust coordinate format to (x, y, score)
            nms_peaks = nms_peaks[:, [1, 0, 2]]  # Swap x and y coordinates

            peaks.append(nms_peaks)
            
        return peaks
    
    def _nms(self, peaks, distance_threshold):
        """
        Non-maximum suppression

        Args:
            peaks: Peak points (N, 3) where 3 is (y, x, score)
            distance_threshold: Distance threshold

        Returns:
            Filtered peak points
        """
        if peaks.size(0) == 0:
            return peaks
            
        keep = []
        removed = set()
        
        for i in range(peaks.size(0)):
            if i in removed:
                continue

            keep.append(i)

            # Check all subsequent points
            for j in range(i + 1, peaks.size(0)):
                if j in removed:
                    continue

                # Calculate spatial distance
                dist = torch.sqrt((peaks[i, 0] - peaks[j, 0])**2 + (peaks[i, 1] - peaks[j, 1])**2)

                # If distance is less than threshold, remove weaker point
                if dist <= distance_threshold:
                    removed.add(j)

        # Return kept points
        if keep:
            return peaks[torch.tensor(keep, device=peaks.device)]
        else:
            return torch.empty((0, 3), device=peaks.device)

# Test code
if __name__ == "__main__":
    # Create peak point generator instance
    generator = PeakPointGenerator(target_size=(512, 512), kernel_size=16, target_count=10)

    # Create test features
    B, C, H, W = 2, 128, 64, 64
    features = torch.randn(B, C, H, W)

    # Generate peak points
    peaks = generator(features)

    print(f"Input shape: {features.shape}")
    print(f"Number of peaks per batch: {[p.shape[0] for p in peaks]}")

    # Check peak point format
    for i, peak in enumerate(peaks):
        if peak.size(0) > 0:
            print(f"Batch {i} peak points (first 5):")
            print(f"  Coordinates and scores: {peak[:5]}")
            # Verify coordinate range
            assert torch.all(peak[:, 0] >= 0) and torch.all(peak[:, 0] < 512), "X coordinate out of range"
            assert torch.all(peak[:, 1] >= 0) and torch.all(peak[:, 1] < 512), "Y coordinate out of range"

    print("Peak point generation test passed!")