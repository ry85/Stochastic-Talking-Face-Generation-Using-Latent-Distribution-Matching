import torch.nn as nn

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.audio_eocder = nn.Sequential(
            nn.Conv2d(1,64,3,1,1),
            nn.Conv2d(64,128,3,1,1),
            nn.MaxPool2d(3, stride=(1,2)),
            nn.Conv2d(128,256,3,1,1),
            nn.Conv2d(256,256,3,1,1),
            nn.MaxPool2d(3, stride=(2,2))
            )
        self.audio_eocder_fc = nn.Sequential(
            nn.Linear(6144,2048),
            nn.ReLU(True),
            nn.Linear(2048,128),
            nn.ReLU(True),
       
            )

    def forward(self, audio):  #torch.Size([13, 28, 12])
        current_feature = self.audio_eocder(audio.unsqueeze(1))
        current_feature = current_feature.view(current_feature.size(0), -1)
        current_feature = self.audio_eocder_fc(current_feature)  #torch.Size([13, 128])
        
        return current_feature