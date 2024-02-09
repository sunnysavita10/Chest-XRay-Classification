import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        """
        Creating custom CNN architecture for Image classification
        """
        super(Net, self).__init__()

        self.convolution_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=8, kernel_size=(3, 3), padding=0, bias=True
            ),
            nn.ReLU(),
            nn.BatchNorm2d(8),
        )

        self.pooling11 = nn.MaxPool2d(2, 2)

        self.convolution_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8, out_channels=20, kernel_size=(3, 3), padding=0, bias=True
            ),
            nn.ReLU(),
            nn.BatchNorm2d(20),
        )

        self.pooling22 = nn.MaxPool2d(2, 2)

        self.convolution_block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=20,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(10),
        )

        self.pooling33 = nn.MaxPool2d(2, 2)

        self.convolution_block4 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=20,
                kernel_size=(3, 3),
                padding=0,
                bias=True,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(20),
        )

        self.convolution_block5 = nn.Sequential(
            nn.Conv2d(
                in_channels=20,
                out_channels=32,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )

        self.convolution_block6 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=10,
                kernel_size=(3, 3),
                padding=0,
                bias=True,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(10),
        )

        self.convolution_block7 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(10),
        )

        self.convolution_block8 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=14,
                kernel_size=(3, 3),
                padding=0,
                bias=True,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(14),
        )

        self.convolution_block9 = nn.Sequential(
            nn.Conv2d(
                in_channels=14,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=True,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )

        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=4))

        self.convolution_block_out = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=2, kernel_size=(4, 4), padding=0, bias=True
            ),
        )

    def forward(self, x) -> float:
        x = self.convolution_block1(x)

        x = self.pooling11(x)

        x = self.convolution_block2(x)

        x = self.pooling22(x)

        x = self.convolution_block3(x)

        x = self.pooling33(x)

        x = self.convolution_block4(x)

        x = self.convolution_block5(x)

        x = self.convolution_block6(x)

        x = self.convolution_block7(x)

        x = self.convolution_block8(x)

        x = self.convolution_block9(x)

        x = self.gap(x)

        x = self.convolution_block_out(x)

        x = x.view(-1, 2)

        return F.log_softmax(x, dim=-1)