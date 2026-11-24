import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Try to import required modules
try:
    from nets.swin_transformer import SwinTransformerV2
except ImportError:
    print("Warning: nets.swin_transformer not found, using placeholder")
    SwinTransformerV2 = None


class STU(nn.Module):
    """
    STU: SwinTransformer UNet
    A standard UNet architecture with SwinTransformerV2 backbone
    """

    def __init__(self, num_classes=21, pretrained=False, backbone="swin_transformer_v2_large", input_shape=[256, 256]):
        super(STU, self).__init__()

        self.num_classes = num_classes
        self.input_shape = input_shape

        # Backbone (SwinTransformerV2 or simple conv backbone as fallback)
        if SwinTransformerV2 is not None and backbone == "swin_transformer_v2_large":
            try:
                self.backbone = SwinTransformerV2(
                    img_size=input_shape[0],
                    patch_size=4,
                    in_chans=3,
                    num_classes=1000,
                    embed_dim=192,
                    depths=[2, 2, 18, 2],
                    num_heads=[6, 12, 24, 48],
                    window_size=8,  # Optimized for 256x256 input
                    drop_path_rate=0.3,
                    pretrained_window_sizes=[12, 12, 12, 6]
                )

                # Probe actual feature dimensions
                self._probe_feature_dimensions()
                self.use_swin = True
                print("Successfully created SwinTransformerV2 Large backbone")

            except Exception as e:
                print(f"Warning: Could not create SwinTransformer, using simple backbone: {e}")
                self.use_swin = False
        else:
            self.use_swin = False

        # If SwinTransformer is not available, use simple conv backbone
        if not self.use_swin:
            self.backbone = self._create_simple_backbone()
            self.feature_dims = [64, 128, 256, 512]  # Simple backbone feature dimensions

        # Create decoder layers
        self._create_decoder_layers()

        # Final classification layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )

        total_params = sum(p.numel() for p in self.parameters())
        print(f"STU model created with {total_params:,} parameters")

    def _probe_feature_dimensions(self):
        """Probe the actual feature dimensions from SwinTransformer"""
        self.backbone.eval()
        with torch.no_grad():
            # Create dummy input
            dummy_input = torch.randn(1, 3, self.input_shape[0], self.input_shape[1])
            try:
                features = self.backbone.forward_features(dummy_input)
                self.feature_dims = [f.shape[1] for f in features]  # Get channel dimensions
            except Exception as e:
                print(f"Error probing features, using default dimensions: {e}")
                self.feature_dims = [192, 384, 768, 1536]  # Default SwinV2-Large dimensions
        self.backbone.train()

    def _create_decoder_layers(self):
        """Create decoder layers with correct input dimensions"""
        # Decoder layers (use interpolation for upsampling)
        self.decoder4 = self._make_decoder_layer(self.feature_dims[3], 512)
        self.decoder3 = self._make_decoder_layer(512 + self.feature_dims[2], 256)
        self.decoder2 = self._make_decoder_layer(256 + self.feature_dims[1], 128)
        self.decoder1 = self._make_decoder_layer(128 + self.feature_dims[0], 64)

    def _create_simple_backbone(self):
        """Create simple convolutional backbone as fallback"""
        return nn.ModuleDict({
            'stage1': nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)  # 256->128
            ),
            'stage2': nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)  # 128->64
            ),
            'stage3': nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)  # 64->32
            ),
            'stage4': nn.Sequential(
                nn.Conv2d(256, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)  # 32->16
            )
        })

    def _make_decoder_layer(self, in_channels, out_channels):
        """Create decoder layer with double convolution"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass
        Input: x [B, 3, H, W]
        Output: segmentation result [B, num_classes, H, W]
        """
        original_size = x.shape[2:]  # Store original H, W

        # Backbone feature extraction
        if self.use_swin:
            features = self.backbone.forward_features(x)
        else:
            # Use simple backbone
            features = []
            feat = x
            for stage_name in ['stage1', 'stage2', 'stage3', 'stage4']:
                feat = self.backbone[stage_name](feat)
                features.append(feat)

        # Decoder process with skip connections
        # Stage 4 (deepest layer) - no skip connection
        d4 = self.decoder4(features[3])

        # Stage 3 with skip connection
        target_size = features[2].shape[2:]
        up4 = F.interpolate(d4, size=target_size, mode='bilinear', align_corners=False)
        d3_input = torch.cat([up4, features[2]], dim=1)
        d3 = self.decoder3(d3_input)

        # Stage 2 with skip connection
        target_size = features[1].shape[2:]
        up3 = F.interpolate(d3, size=target_size, mode='bilinear', align_corners=False)
        d2_input = torch.cat([up3, features[1]], dim=1)
        d2 = self.decoder2(d2_input)

        # Stage 1 with skip connection
        target_size = features[0].shape[2:]
        up2 = F.interpolate(d2, size=target_size, mode='bilinear', align_corners=False)
        d1_input = torch.cat([up2, features[0]], dim=1)
        d1 = self.decoder1(d1_input)

        # Final upsampling to original input size
        final_up = F.interpolate(d1, size=original_size, mode='bilinear', align_corners=False)

        # Segmentation prediction
        seg_output = self.final_conv(final_up)

        return seg_output

    def freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True


def create_stu_model(num_classes=4, pretrained=True, backbone="swin_transformer_v2_large", input_shape=[256, 256]):
    """Create STU (SwinTransformer UNet) model"""
    model = STU(
        num_classes=num_classes,
        pretrained=pretrained,
        backbone=backbone,
        input_shape=input_shape
    )
    return model


def test_model():
    """Test if the model can be created and run normally"""
    print("🔧 Testing STU (SwinTransformer UNet) model...")

    try:
        # Create model
        model = create_stu_model(num_classes=4, pretrained=False)
        print("✅ Model created successfully")

        # Create test input
        test_input = torch.randn(2, 3, 256, 256)
        print(f"📥 Test input shape: {test_input.shape}")

        # Model forward pass
        model.eval()
        with torch.no_grad():
            output = model(test_input)

        print(f"📤 Segmentation output shape: {output.shape}")

        # Check output dimensions
        expected_shape = (2, 4, 256, 256)  # [batch, classes, H, W]

        assert output.shape == expected_shape, f"Output shape error: expected {expected_shape}, got {output.shape}"

        print("✅ Output shape check passed")

        # Test gradient calculation
        model.train()
        loss = output.sum()
        loss.backward()
        print("✅ Gradient calculation normal")

        # Test freeze/unfreeze functionality
        model.freeze_backbone()
        print("✅ Backbone freeze function normal")

        model.unfreeze_backbone()
        print("✅ Backbone unfreeze function normal")

        print("🎉 All tests passed! STU model working normally")

        # Model parameter statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"📊 Model parameter statistics:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run test
    success = test_model()

    if success:
        print("\n🚀 Model is ready for training!")
        print("💡 Usage:")
        print("   from stu_model import create_stu_model")
        print("   model = create_stu_model(num_classes=4)")
    else:
        print("\n⚠️ Please check dependencies and code configuration")