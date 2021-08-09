import ctypes
import os
import os.path as osp

dir_path = osp.join(os.path.expanduser('~'), 'space/trt_plugin/build/lib/')

if not osp.exists(dir_path):
    if 'AMIRSTAN_LIBRARY_PATH' in os.environ:
        dir_path = os.environ['AMIRSTAN_LIBRARY_PATH']
    else:
        dir_path = os.path.dirname(os.path.realpath(__file__))


def load_plugin_library():
    # ctypes.CDLL(osp.join(dir_path, 'libamirstan_plugin.so'))
    dir_path = '/home/SENSETIME/liqiaofei1/Desktop/pro/torch2trt/torch2trt_dynamic/build/lib.linux-x86_64-3.7/'
    ctypes.CDLL(osp.join(dir_path, 'libtorch2trt_dynamic.cpython-37m-x86_64-linux-gnu.so'))
