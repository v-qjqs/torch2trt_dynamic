# dummy converters throw warnings method encountered

from .dummy_converters import *

# supported converters will override dummy converters

from .activation import *
from .adaptive_avg_pool2d import *
from .adaptive_max_pool2d import *
from .AdaptiveAvgPool2d import *
from .add import *
from .avg_pool2d import *
from .mul import *
from .div import *
from .BatchNorm1d import *
from .BatchNorm2d import *
from .cat import *
from .clamp import *
from .Conv1d import *
from .Conv2d import *
from .ConvTranspose2d import *
from .getitem import *
from .identity import *
from .Identity import *
from .instance_norm import *
from .Linear import *
from .LogSoftmax import *
from .max_pool2d import *
from .max import *
from .min import *
from .normalize import *
from .pad import *
from .permute import *
from .pow import *
from .prelu import *
from .prod import *
from .relu import *
from .ReLU import *
from .relu6 import *
from .ReLU6 import *
from .sigmoid import *
from .sub import *
from .sum import *
from .view import *
from .tanh import *
from .transpose import *
from .mean import *
from .softmax import *
from .split import *
from .chunk import *
from .unary import *

# without plugin
from .matmul import *
from .interpolate_custom import *
from .topk import *
from .index_select import *
from .addcmul import *
from .conv2d import *
from .view_as import *
from .unsqueeze import *
from .squeeze import *
from .flatten import *
from .stack import *
from .pixel_shuffle import *
from .LayerNorm import *
from .exview import *
from .size import *
from .gather import *
from .argmax import *
from .argmin import *

try:
    # custom plugin support
    from .GroupNorm import *
    from .repeat import *
except:
    print("plugin not found.")
