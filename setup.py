import glob
import os
import re
from pkg_resources import DistributionNotFound, get_distribution, parse_version
from setuptools import find_packages, setup

EXT_TYPE = ''
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension
    EXT_TYPE = 'pytorch'
    cmd_class = {'build_ext': BuildExtension.with_options(use_ninja=True)}
except ModuleNotFoundError:
    cmd_class = {}
    print('Skip building ext ops due to the absence of torch.')


# def choose_requirement(primary, secondary):
#     """If some version of primary requirement installed, return primary, else
#     return secondary."""
#     try:
#         name = re.split(r'[!<>=]', primary)[0]
#         get_distribution(name)
#     except DistributionNotFound:
#         return secondary

#     return str(primary)


# def get_version():
#     version_file = 'mmcv/version.py'
#     with open(version_file, 'r', encoding='utf-8') as f:
#         exec(compile(f.read(), version_file, 'exec'))
#     return locals()['__version__']


# def parse_requirements(fname='requirements/runtime.txt', with_version=True):
#     """Parse the package dependencies listed in a requirements file but strips
#     specific versioning information.

#     Args:
#         fname (str): path to requirements file
#         with_version (bool, default=False): if True include version specs

#     Returns:
#         List[str]: list of requirements items

#     CommandLine:
#         python -c "import setup; print(setup.parse_requirements())"
#     """
#     import sys
#     from os.path import exists
#     require_fpath = fname

#     def parse_line(line):
#         """Parse information from a line in a requirements text file."""
#         if line.startswith('-r '):
#             # Allow specifying requirements in other files
#             target = line.split(' ')[1]
#             for info in parse_require_file(target):
#                 yield info
#         else:
#             info = {'line': line}
#             if line.startswith('-e '):
#                 info['package'] = line.split('#egg=')[1]
#             else:
#                 # Remove versioning from the package
#                 pat = '(' + '|'.join(['>=', '==', '>']) + ')'
#                 parts = re.split(pat, line, maxsplit=1)
#                 parts = [p.strip() for p in parts]

#                 info['package'] = parts[0]
#                 if len(parts) > 1:
#                     op, rest = parts[1:]
#                     if ';' in rest:
#                         # Handle platform specific dependencies
#                         # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
#                         version, platform_deps = map(str.strip,
#                                                      rest.split(';'))
#                         info['platform_deps'] = platform_deps
#                     else:
#                         version = rest  # NOQA
#                     info['version'] = (op, version)
#             yield info

#     def parse_require_file(fpath):
#         with open(fpath, 'r') as f:
#             for line in f.readlines():
#                 line = line.strip()
#                 if line and not line.startswith('#'):
#                     for info in parse_line(line):
#                         yield info

#     def gen_packages_items():
#         if exists(require_fpath):
#             for info in parse_require_file(require_fpath):
#                 parts = [info['package']]
#                 if with_version and 'version' in info:
#                     parts.extend(info['version'])
#                 if not sys.version.startswith('3.4'):
#                     # apparently package_deps are broken in 3.4
#                     platform_deps = info.get('platform_deps')
#                     if platform_deps is not None:
#                         parts.append(';' + platform_deps)
#                 item = ''.join(parts)
#                 yield item

#     packages = list(gen_packages_items())
#     return packages


# install_requires = parse_requirements()

# try:
#     # OpenCV installed via conda.
#     import cv2  # NOQA: F401
#     major, minor, *rest = cv2.__version__.split('.')
#     if int(major) < 3:
#         raise RuntimeError(
#             f'OpenCV >=3 is required but {cv2.__version__} is installed')
# except ImportError:
#     # If first not installed install second package
#     CHOOSE_INSTALL_REQUIRES = [('opencv-python-headless>=3',
#                                 'opencv-python>=3')]
#     for main, secondary in CHOOSE_INSTALL_REQUIRES:
#         install_requires.append(choose_requirement(main, secondary))


def get_extensions():
    extensions = []
    if os.getenv('TRT_WITH_OPS', '0') == '0':
        return extensions

    # prevent ninja from using too many resources
    os.environ.setdefault('MAX_JOBS', '4')
    
    # ext_name = 'mmcv._ext_trt'
    ext_name = 'libtorch2trt_dynamic'
    from torch.utils.cpp_extension import include_paths, library_paths
    library_dirs = []
    libraries = []
    include_dirs = []
    tensorrt_path = os.getenv('TENSORRT_DIR', '0')
    tensorrt_lib_path = glob.glob(
        os.path.join(tensorrt_path, 'targets', '*', 'lib'))[0]
    library_dirs += [tensorrt_lib_path]
    libraries += ['nvinfer', 'nvparsers', 'nvinfer_plugin']
    libraries += ['cudart']
    kwargs = {}
    define_macros = []
    extra_compile_args = {'cxx': []}

    include_path = os.path.abspath('./torch2trt_dynamic/ops/csrc/common/cuda')
    include_trt_path = os.path.abspath('./torch2trt_dynamic/ops/csrc/tensorrt')
    include_dirs.append(include_trt_path)
    include_dirs.append(include_path)
    include_dirs.append(os.path.join(tensorrt_path, 'include'))
    include_dirs += include_paths(cuda=True)

    op_files = glob.glob('./torch2trt_dynamic/ops/csrc/tensorrt/plugins/*')
    define_macros += [('TRT_WITH_OPS', None)]
    cuda_args = os.getenv('MMCV_CUDA_ARGS')  # TODO
    extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
    library_dirs += library_paths(cuda=True)

    kwargs['library_dirs'] = library_dirs
    kwargs['libraries'] = libraries

    from setuptools import Extension
    from torch.utils.cpp_extension import CppExtension, CUDAExtension
    ext_ops = Extension(
        name=ext_name,
        sources=op_files,
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        language='c++',
        library_dirs=library_dirs,
        libraries=libraries)
    extensions.append(ext_ops)
    return extensions


setup(
    name='torch2trt_dynamic' if os.getenv('TRT_WITH_OPS', '0') == '0' else 'torch2trt_dynamic-full',
    # version=get_version(),
    version='0.4.1',
    description='Torch to TensorRT Dynamic',
    keywords='torch2tensorrt',
    packages=find_packages(),
    include_package_data=True,
    # classifiers=[
    #     'Development Status :: 4 - Beta',
    #     'License :: OSI Approved :: Apache Software License',
    #     'Operating System :: OS Independent',
    #     'Programming Language :: Python :: 3',
    #     'Programming Language :: Python :: 3.6',
    #     'Programming Language :: Python :: 3.7',
    #     'Programming Language :: Python :: 3.8',
    #     'Topic :: Utilities',
    # ],
    # url='https://github.com/open-mmlab/mmcv',
    # author='MMCV Authors',
    # author_email='openmmlab@gmail.com',
    # setup_requires=['pytest-runner'],
    # tests_require=['pytest'],
    # install_requires=install_requires,
    ext_modules=get_extensions(),
    cmdclass=cmd_class,
    zip_safe=False)
