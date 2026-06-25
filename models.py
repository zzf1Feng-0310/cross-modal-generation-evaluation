import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class Generator(nn.Layer):
    def __init__(self, z_dim=100, cond_dim=128, img_channels=3, feature_dim=64):#初始化参数（）
        super().__init__()
        self.feature_dim = feature_dim      # 保存以便 reshape 使用
        self.init_size = 4  #生成器初始特征图尺寸（4×4）
        self.fc = nn.Linear(z_dim + cond_dim, feature_dim * 8 * self.init_size ** 2)
#全连接层
        self.deconv = nn.Sequential(
            nn.BatchNorm2D(feature_dim * 8),
            nn.Upsample(scale_factor=2),
            nn.Conv2D(feature_dim * 8, feature_dim * 4, 3, padding=1),
            nn.BatchNorm2D(feature_dim * 4),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.Conv2D(feature_dim * 4, feature_dim * 2, 3, padding=1),
            nn.BatchNorm2D(feature_dim * 2),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.Conv2D(feature_dim * 2, feature_dim, 3, padding=1),
            nn.BatchNorm2D(feature_dim),
            nn.ReLU(),

            nn.Conv2D(feature_dim, img_channels, 3, padding=1),
            nn.Tanh()#输出 [-1,1] 的图像，和数据归一化范围一致
        )

    def forward(self, z, cond):
        x = paddle.concat([z, cond], axis=1)
        x = self.fc(x)
        # 修复：显式指定通道数，避免两个 -1
        x = x.reshape((-1, self.feature_dim * 8, self.init_size, self.init_size))
        img = self.deconv(x)
        return img


class Discriminator(nn.Layer):#Discriminator（判断器）
    def __init__(self, cond_dim=128, img_channels=3, feature_dim=64):
        super().__init__()
        self.cond_embed = nn.Linear(cond_dim, 32 * 32)

        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2D(img_channels + 1, feature_dim, 3, stride=2, padding=1)
        )
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2D(feature_dim, feature_dim * 2, 3, stride=2, padding=1)
        )
        self.conv3 = nn.utils.spectral_norm(
            nn.Conv2D(feature_dim * 2, feature_dim * 4, 3, stride=2, padding=1)
        )
        self.conv4 = nn.utils.spectral_norm(
            nn.Conv2D(feature_dim * 4, 1, 3, padding=0)
        )

        self.bn2 = nn.BatchNorm2D(feature_dim * 2)
        self.bn3 = nn.BatchNorm2D(feature_dim * 4)

    def forward(self, img, cond):
        B = img.shape[0]
        cond_map = self.cond_embed(cond).reshape((B, 1, 32, 32))
        x = paddle.concat([img, cond_map], axis=1)

        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        out = self.conv4(x)

        return out.reshape((B, -1)).mean(1)