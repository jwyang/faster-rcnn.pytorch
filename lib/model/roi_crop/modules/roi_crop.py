from torch.nn.modules.module import Module
from ..functions.roi_crop import RoICropFunction

class _RoICrop(Module):
    def __init__(self, layout = 'BHWD'):
        super(_RoICrop, self).__init__()
        self.f = RoICropFunction()
    def forward(self, input1, input2):
        return self.f(input1, input2)
