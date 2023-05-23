function dataOut = CUDAconvolution3D(data, kernel)
%CUDAconvolution(data, kernel) GPU-accelerated 3D convolution
% The first time, compiles from mexGPUconvolution3D.cu, 
% convolutionFFT3D_common.h 
%   - .cu and .h files must be on path
%   - NVIDIA nvcc compiler must be installed
%   - NVDIA cuFFT library must be installed
% INPUTS
% data, kernel - 3D numeric matlab arrays
% OUTPUTS
% dataOut - convolution of data and kernel, matlab 3D single array on host
%
% C++/CUDA code is adapted from NVIDIA cuda-samples/convolutionFFT2D
assert(ndims(data)==3 && ndims(kernel)==3 && isnumeric(data) && isnumeric(kernel))
if ~(isfile('mexGPUconvolution3D.mexw64') || isfile ('mexGPUconvolution3D.mexmaci64') ||...
    isfile('mexGPUconvolution3D.mexa64'))
    disp("Compiling...");
    mexcuda -v -lcufft mexGPUconvolution3D.cu;
end
dataGPU=gpuArray(cast(data, 'single'));
kernelGPU=gpuArray(cast(kernel, 'single'));
[dataOutGPU] = mexGPUconvolution3D(dataGPU,kernelGPU);
dataOut=gather(dataOutGPU);
end