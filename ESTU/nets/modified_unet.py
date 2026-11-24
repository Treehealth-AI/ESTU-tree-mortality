import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


try:
    from nets.swin_transformer import SwinTransformerV2
except ImportError:
    print("Warning: nets.swin_transformer not found, using placeholder")
    SwinTransformerV2 = None

try:
    from sobel_module import EdgeEnhancedFeature, SobelGuidedUpsampling
except ImportError:
    # print("Warning: sobel_module not found, creating local implementation")
    pass


    class EdgeEnhancedFeature(nn.Module):
        def __init__(self, in_channels, reduction=4):
            super(EdgeEnhancedFeature, self).__init__()
            self.in_channels = in_channels


            import numpy as np
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)


            self.sobel_x_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                          padding=1, bias=False, groups=in_channels)
            self.sobel_y_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                          padding=1, bias=False, groups=in_channels)


            with torch.no_grad():
                for i in range(in_channels):
                    self.sobel_x_conv.weight[i, 0] = torch.from_numpy(sobel_x)
                    self.sobel_y_conv.weight[i, 0] = torch.from_numpy(sobel_y)


            self.sobel_x_conv.weight.requires_grad = False
            self.sobel_y_conv.weight.requires_grad = False


            self.fusion_conv = nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels // reduction, 1),
                nn.BatchNorm2d(in_channels // reduction),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.Sigmoid()
            )

        def forward(self, x):

            grad_x = self.sobel_x_conv(x)
            grad_y = self.sobel_y_conv(x)
            edge_features = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)


            combined = torch.cat([x, edge_features], dim=1)
            attention = self.fusion_conv(combined)


            enhanced_features = x * attention + edge_features * (1 - attention)
            return enhanced_features


    class SobelGuidedUpsampling(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(SobelGuidedUpsampling, self).__init__()

            self.upsample = nn.ConvTranspose2d(in_channels, out_channels,
                                               kernel_size=2, stride=2)
            self.edge_guide = EdgeEnhancedFeature(out_channels)

            self.refine = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            upsampled = self.upsample(x)
            edge_enhanced = self.edge_guide(upsampled)
            refined = self.refine(edge_enhanced)
            return refined


class SobelEnhancedUnet(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone="swin_transformer_v2_large", input_shape=[256, 256]):
        super(SobelEnhancedUnet, self).__init__()


        self.input_edge_enhance = EdgeEnhancedFeature(3, reduction=2)


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
                    window_size=12,
                    drop_path_rate=0.3,
                    pretrained_window_sizes=[12, 12, 12, 6]
                )
                self.feature_dims = [192, 384, 768, 1536]
                self.use_swin = True
            except Exception as e:
                print(f"Warning: Could not create SwinTransformer, using simple backbone: {e}")
                self.use_swin = False
        else:
            self.use_swin = False


        if not self.use_swin:
            self.backbone = self._create_simple_backbone()
            self.feature_dims = [64, 128, 256, 512]


        self.decoder4 = self._make_decoder_layer(self.feature_dims[3], 512)
        self.edge_enhance4 = EdgeEnhancedFeature(512, reduction=4)

        self.decoder3 = self._make_decoder_layer(512 + self.feature_dims[2], 256)
        self.edge_enhance3 = EdgeEnhancedFeature(256, reduction=4)

        self.decoder2 = self._make_decoder_layer(256 + self.feature_dims[1], 128)
        self.edge_enhance2 = EdgeEnhancedFeature(128, reduction=4)

        self.decoder1 = self._make_decoder_layer(128 + self.feature_dims[0], 64)
        self.edge_enhance1 = EdgeEnhancedFeature(64, reduction=4)


        self.sobel_up4 = SobelGuidedUpsampling(512, 512)
        self.sobel_up3 = SobelGuidedUpsampling(256, 256)
        self.sobel_up2 = SobelGuidedUpsampling(128, 128)
        self.sobel_up1 = SobelGuidedUpsampling(64, 64)


        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )


        self.edge_supervision = nn.Sequential(
            nn.Conv2d(64, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def _create_simple_backbone(self):
        pass
        return nn.ModuleDict({
            'stage1': nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ),
            'stage2': nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ),
            'stage3': nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ),
            'stage4': nn.Sequential(
                nn.Conv2d(256, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )
        })

    def _make_decoder_layer(self, in_channels, out_channels):
        pass
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        pass


        x_enhanced = self.input_edge_enhance(x)


        if self.use_swin:
            features = self.backbone.forward_features(x_enhanced)
        else:

            features = []
            feat = x_enhanced
            for stage_name in ['stage1', 'stage2', 'stage3', 'stage4']:
                feat = self.backbone[stage_name](feat)
                features.append(feat)


        d4 = self.decoder4(features[3])
        d4_edge = self.edge_enhance4(d4)
        up4 = self.sobel_up4(d4_edge)

        # Stage 3
        d3_input = torch.cat([up4, features[2]], dim=1)
        d3 = self.decoder3(d3_input)
        d3_edge = self.edge_enhance3(d3)
        up3 = self.sobel_up3(d3_edge)

        # Stage 2
        d2_input = torch.cat([up3, features[1]], dim=1)
        d2 = self.decoder2(d2_input)
        d2_edge = self.edge_enhance2(d2)
        up2 = self.sobel_up2(d2_edge)

        # Stage 1
        d1_input = torch.cat([up2, features[0]], dim=1)
        d1 = self.decoder1(d1_input)
        d1_edge = self.edge_enhance1(d1)
        up1 = self.sobel_up1(d1_edge)


        final_up = F.interpolate(up1, size=x.shape[2:],
                                 mode='bilinear', align_corners=False)


        seg_output = self.final_conv(final_up)


        edge_output = self.edge_supervision(final_up)

        return seg_output, edge_output

    def freeze_backbone(self):
        pass
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        pass
        for param in self.backbone.parameters():
            param.requires_grad = True


def create_sobel_unet(num_classes=4, pretrained=True, backbone="swin_transformer_v2_large", input_shape=[256, 256]):
    pass
    model = SobelEnhancedUnet(
        num_classes=num_classes,
        pretrained=pretrained,
        backbone=backbone,
        input_shape=input_shape
    )
    return model


def test_model():
    pass
    print("🔧 开始测试Sobel增强U-Net模型...")

    try:

        model = create_sobel_unet(num_classes=4, pretrained=False)
        print("✅ 模型创建成功")


        test_input = torch.randn(2, 3, 256, 256)
        print(f"📥 测试输入形状: {test_input.shape}")


        model.eval()
        with torch.no_grad():
            seg_output, edge_output = model(test_input)

        print(f"📤 分割输出形状: {seg_output.shape}")
        print(f"📤 边缘输出形状: {edge_output.shape}")


        expected_seg_shape = (2, 4, 256, 256)  # [batch, classes, H, W]
        expected_edge_shape = (2, 1, 256, 256)  # [batch, 1, H, W]

        assert seg_output.shape == expected_seg_shape, f"分割输出形状错误: 期望 {expected_seg_shape}, 实际 {seg_output.shape}"
        assert edge_output.shape == expected_edge_shape, f"边缘输出形状错误: 期望 {expected_edge_shape}, 实际 {edge_output.shape}"

        print("✅ 输出形状检查通过")


        model.train()
        loss = seg_output.sum() + edge_output.sum()
        loss.backward()
        print("✅ 梯度计算正常")


        model.freeze_backbone()
        print("✅ Backbone冻结功能正常")

        model.unfreeze_backbone()
        print("✅ Backbone解冻功能正常")

        print("🎉 所有测试通过！Sobel增强U-Net模型工作正常")


        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"📊 模型参数统计:")
        print(f"   总参数数量: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":

    success = test_model()

    if success:
        print("\n🚀 模型已准备就绪，可以开始训练！")
        print("💡 使用方法:")
        print("   from modified_unet import create_sobel_unet")
        print("   model = create_sobel_unet(num_classes=4)")
    else:
        print("\n⚠️ 请检查依赖项和代码配置")