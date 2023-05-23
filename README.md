C++/CUDA GPU-accelerated convolution in 2D and 3D.

NVIDIA CUDA toolbox - including an nvcc compiler, supported C++ compiler, and library cuFFT - must be installed.

Run functions ``CUDAconvolution(data, kernel)`` or ``CUDAconvolution3D(data, kernel)`` analogous to matlab ``conv2``, ``convn``.

The method is convolution by FFT, pointwise multiply, and inverse FFT.

This method is much faster in the case of medium to large kernels; 
outperforms matlab starting at kernel size ~12 x 12 x 12 and speedup is more than 1000x at convolution 900x900x200 with 100x100x100 kernel (``test3d.mlx``).
Execution time should be constant and is <1s on my machine up to GPU memory limit.

Data + kernel dimensions , rounded up to nearest power of two, must fit on GPU memory.

Data should be the bigger array, as an array cut to original dimensions of data is returned.