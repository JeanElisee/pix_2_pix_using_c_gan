import torch
import torch.nn as nn

class CNNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv = nn.Sequential(
            # Stride is set to be 2
            # Bias will always be false
            # padding_mode = reflect 
            nn.Conv2d(in_channels, out_channels, 4, stride, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


# Gets the input image but also the output image, x, y <- concatenate them along the channels
class Discriminator(nn.Module):
    # features is what we will be using for our CNN block 
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]): # We send a 256^2 and in the output we will have a 30^2
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels*2, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        
        # We skip the first because it was in the initialization block 
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )

            in_channels = feature

        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"),
        )   

        self.model = nn.Sequential(*layers) # We explode what we got from the list and add to the NN sequential


    def forward(self, x, y): # We get y as input (either fake of real)
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)


def test():
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    
    model = Discriminator()
    preds = model(x, y)

    # print(model)
    print(preds.shape)


if __name__ == "__main__":
    test()