#!/usr/bin/env python3
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ["-std=c++17"]  # Updated to C++17 for PyTorch 2.7

nvcc_args = [
    "-gencode", "arch=compute_50,code=sm_50",
    "-gencode", "arch=compute_52,code=sm_52",
    "-gencode", "arch=compute_60,code=sm_60",
    "-gencode", "arch=compute_61,code=sm_61",
    "-gencode", "arch=compute_70,code=sm_70",
    "-gencode", "arch=compute_75,code=sm_75",
    "-gencode", "arch=compute_80,code=sm_80",
    "-gencode", "arch=compute_86,code=sm_86",
    "-gencode", "arch=compute_89,code=sm_89",
    "-gencode", "arch=compute_90,code=sm_90",
]

setup(
    name="correlation_cuda",
    ext_modules=[
        CUDAExtension(
            "correlation_cuda",
            ["correlation_cuda.cc", "correlation_cuda_kernel.cu"],
            extra_compile_args={
                "cxx": cxx_args,
                "nvcc": nvcc_args,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
