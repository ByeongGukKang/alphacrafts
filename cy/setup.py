from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "share",  # Module name
        ["share.pyx"],  # Source file
        # extra_compile_args = ['-O3'],
        # extra_link_args = extra_link_args,
    )
]

setup(
    ext_modules = cythonize(
        extensions, 
        compiler_directives = {
            "language_level": "3", "boundscheck": False, "wraparound": False, 'nonecheck': False
        },
    ),
)