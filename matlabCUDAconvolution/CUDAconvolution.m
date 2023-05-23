function dataOut = CUDAconvolution(data, kernel)
%CUDAconvolution(data, kernel) GPU-accelerated 2D convolution
% the first time, compiles mex file from mexGPUconvolution.cu, 
% convolutionFFT2D_common.h 
%   - .cu and .h files must be on path
%   - NVIDIA nvcc compiler must be installed
%   - NVDIA cuFFT library must be installed
% INPUTS
% data, kernel - 2D numeric matlab arrays
% OUTPUTS
% dataOut - convolution of data and kernel, matlab 2D single array on host
%
% C++/CUDA code is adapted from NVIDIA cuda-samples/convolutionFFT2D
assert(ismatrix(data) && ismatrix(kernel) && isnumeric(data) && isnumeric(kernel))
if ~(isfile('mexGPUconvolution.mexw64') || isfile ('mexGPUconvolution.mexmaci64') ||...
    isfile('mexGPUconvolution.mexa64'))
    disp("Compiling...");
    mexcuda -v -lcufft mexGPUconvolution.cu;
end
dataGPU=gpuArray(cast(data, 'single'));
kernelGPU=gpuArray(cast(kernel, 'single'));
[dataOutGPU] = mexGPUconvolution(dataGPU,kernelGPU);
dataOut=gather(dataOutGPU);
end