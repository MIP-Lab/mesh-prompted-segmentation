import torch.nn as nn

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class ConvModule(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 norm_type='bn3d',
                 activation_type='relu',
                 inplace=True,
                 bias=False
                 ):
        super().__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, 
                              padding, dilation, groups, bias)
        if norm_type == 'bn3d':
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.bn = None
        if activation_type == 'relu':
            self.activate = nn.ReLU(inplace=inplace)
        else:
            self.activate = None
        
        self.activation_type = activation_type
        self.norm_type = norm_type
        # Use msra init by default
        self.init_weights()

    def init_weights(self):
        kaiming_init(self.conv, a=0, nonlinearity=self.activation_type or 'relu')
        if self.bn is not None:
            constant_init(self.bn, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        x = self.conv(x)
        if norm and self.bn is not None:
            x = self.bn(x)
        if activate and self.activate is not None:
            x = self.activate(x)
        return x