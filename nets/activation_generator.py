import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('nets/InfMAE')
from models_infmae_skip4 import infmae_vit_base_patch16
from util import misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import timm.optim.optim_factory as optim_factory
# InfMAE (Activation Map Generator)
class InfMAEFeatureProcessor(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=128):
        """
        Process InfMAE features to match the shape of backbone features

        Args:
            input_dim: InfMAE feature dimension (768)
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension (128)
        """
        super(InfMAEFeatureProcessor, self).__init__()
        
        # Convert 49 patch features to smaller spatial dimension
        self.patch_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Upsample 7x7 feature map to 64x64
        self.upsample = nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)

        # Final convolution layer to adjust channel number
        self.final_conv = nn.Conv2d(output_dim, 128, kernel_size=1)
        
    def forward(self, infmae_features):
        """
        Process InfMAE features

        Args:
            infmae_features: (B, 49, 768)

        Returns:
            processed_features: (B, 128, 64, 64)
        """
        B, N, C = infmae_features.shape  # B, 49, 768

        # Process each patch feature
        features = self.patch_processor(infmae_features)  # (B, 49, 128)

        # Reshape to 7x7 feature map
        # 49 = 7 * 7
        features = features.permute(0, 2, 1)  # (B, 128, 49)
        features = features.view(B, -1, 7, 7)  # (B, 128, 7, 7)

        # Upsample to 64x64
        features = self.upsample(features)  # (B, 128, 64, 64)

        # Final adjustment
        features = self.final_conv(features)  # (B, 128, 64, 64)

        return features

# Process activation map to match the shape of backbone features
class InfMAEFeatureExtractor(nn.Module):
    def __init__(self, checkpoint_path, device='cuda'):
        """
        InfMAE feature extractor

        Args:
            checkpoint_path: Pretrained model path
            device: Device ('cuda' or 'cpu')
        """
        super(InfMAEFeatureExtractor, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path

        # Load model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

        # Data preprocessing (consistent with training)
        self.register_buffer('mean', torch.tensor([0.425, 0.425, 0.425]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.200, 0.200, 0.200]).view(1, 3, 1, 1))

        print(f"InfMAE model loaded successfully, using device: {self.device}")
    
    def _load_model(self):
        """Load pretrained model"""
        model = infmae_vit_base_patch16(norm_pix_loss=False)

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            print("Successfully loaded pretrained weights")
        except Exception as e:
            print(f"Error loading weights: {e}")

        return model

    def preprocess(self, x):
        """
        Preprocess image

        Args:
            x: Input image (B, 3, H, W) range [0, 1]

        Returns:
            Preprocessed image (B, 3, 224, 224)
        """
        # Resize to 224x224
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        # Normalize
        x = (x - self.mean) / self.std
        return x
    
    def extract_features(self, x, mask_ratio=0.15):
        """
        Extract image features

        Args:
            x: Input image (B, 3, H, W) range [0, 1]
            mask_ratio: Mask ratio

        Returns:
            features: Extracted features (B, 49, 768)
            mask: Mask (B, 196)
        """
        # Preprocess
        x = self.preprocess(x)

        # Ensure input is on correct device
        x = x.to(self.device)

        # Inference
        with torch.no_grad():
            # Get encoder output
            latent, mask, ids_restore = self.model.forward_encoder(x, mask_ratio=mask_ratio)

        return latent, mask

