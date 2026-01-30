import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, g_ch, x_ch, int_ch):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_ch, int_ch, 1),
            nn.BatchNorm2d(int_ch)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_ch, int_ch, 1),
            nn.BatchNorm2d(int_ch)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(int_ch, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        psi = self.psi(F.relu(self.W_g(g) + self.W_x(x)))
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = ConvBlock(3, 64)
        self.c2 = ConvBlock(64, 128)
        self.c3 = ConvBlock(128, 256)
        self.c4 = ConvBlock(256, 512)
        self.c5 = ConvBlock(512, 1024)

        self.u4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.a4 = AttentionBlock(512, 512, 256)
        self.c6 = ConvBlock(1024, 512)

        self.u3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.a3 = AttentionBlock(256, 256, 128)
        self.c7 = ConvBlock(512, 256)

        self.u2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.a2 = AttentionBlock(128, 128, 64)
        self.c8 = ConvBlock(256, 128)

        self.u1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.a1 = AttentionBlock(64, 64, 32)
        self.c9 = ConvBlock(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(F.max_pool2d(c1, 2))
        c3 = self.c3(F.max_pool2d(c2, 2))
        c4 = self.c4(F.max_pool2d(c3, 2))
        c5 = self.c5(F.max_pool2d(c4, 2))

        u4 = self.u4(c5)
        a4 = self.a4(u4, c4)
        c6 = self.c6(torch.cat([u4, a4], dim=1))

        u3 = self.u3(c6)
        a3 = self.a3(u3, c3)
        c7 = self.c7(torch.cat([u3, a3], dim=1))

        u2 = self.u2(c7)
        a2 = self.a2(u2, c2)
        c8 = self.c8(torch.cat([u2, a2], dim=1))

        u1 = self.u1(c8)
        a1 = self.a1(u1, c1)
        c9 = self.c9(torch.cat([u1, a1], dim=1))

        return self.out(c9)
