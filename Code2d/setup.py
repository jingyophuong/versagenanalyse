from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


# Or, if you use cythonize() to make the ext_modules list,
# include_dirs can be passed to setup()

setup(
    ext_modules=cythonize("2d Code\_texture.pyx"),
    include_dirs=[numpy.get_include()]
)    