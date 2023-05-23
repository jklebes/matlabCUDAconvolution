/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <helper_cuda.h>
#include "convolutionFFT3D_common.h"
#include "convolutionFFT3D.cuh"

#define USE_TEXTURE 1
#define POWER_OF_TWO 1

#if (USE_TEXTURE)
#define LOAD_FLOAT(i) tex1Dfetch<float>(texFloat, i)
#define SET_FLOAT_BASE
#else
#define LOAD_FLOAT(i) d_Src[i]
#define SET_FLOAT_BASE
#endif

////////////////////////////////////////////////////////////////////////////////
/// Position convolution kernel center at (0, 0) in the image
////////////////////////////////////////////////////////////////////////////////
__global__ void padKernel_kernel(float *d_Dst, float *d_Src, int fftD, int fftH, int fftW,
                                 int kernelD, int kernelH, int kernelW, int kernelZ, int kernelY,
                                 int kernelX
#if (USE_TEXTURE)
                                 ,
                                 cudaTextureObject_t texFloat
#endif
                                 ) {

  const int z = blockDim.z * blockIdx.z + threadIdx.z;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;
  const int x = blockDim.x * blockIdx.x + threadIdx.x;

  if (z < kernelD && y < kernelH && x < kernelW) {
    int kz = z - kernelZ;

      if (kz < 0) {
          kz += fftD;
      }

    int ky = y - kernelY;

    if (ky < 0) {
      ky += fftH;
    }

    int kx = x - kernelX;

    if (kx < 0) {
      kx += fftW;
    }

    d_Dst[kz* fftH *fftW + ky * fftW + kx] = LOAD_FLOAT(z* kernelH * kernelW + y * kernelW + x);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Prepare data for "pad to border" addressing mode
////////////////////////////////////////////////////////////////////////////////
__global__ void padDataClampToBorder_kernel(float *d_Dst, float *d_Src,
                                            int fftD, int fftH, int fftW, int dataD, int dataH,
                                            int dataW, int kernelD, int kernelH, int kernelW,
                                            int kernelZ, int kernelY, int kernelX
#if (USE_TEXTURE)
                                            ,
                                            cudaTextureObject_t texFloat
#endif
                                            ) {
  const int z = blockDim.z * blockIdx.z + threadIdx.z;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int borderD = dataD + kernelZ;
  const int borderH = dataH + kernelY;
  const int borderW = dataW + kernelX;

  if (z < fftD && y < fftH && x < fftW) {
    int dz, dy, dx;

    if (z < dataD) {
        dz = z;
    }

    if (y < dataH) {
      dy = y;
    }

    if (x < dataW) {
      dx = x;
    }

    if (z >= dataD && z < borderD) {
        dz = dataD - 1;
    }

    if (y >= dataH && y < borderH) {
      dy = dataH - 1;
    }

    if (x >= dataW && x < borderW) {
      dx = dataW - 1;
    }

    if (z >= borderD) {
        dz = 0;
    }

    if (y >= borderH) {
      dy = 0;
    }

    if (x >= borderW) {
      dx = 0;
    }

    d_Dst[z* fftH * fftW + y * fftW + x] = LOAD_FLOAT(dz * dataH * dataW + dy * dataW + dx);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Modulate Fourier image of padded data by Fourier image of padded kernel
// and normalize by FFT size
////////////////////////////////////////////////////////////////////////////////
inline __device__ void mulAndScale(fComplex &a, const fComplex &b,
                                   const float &c) {
  fComplex t = {c * (a.x * b.x - a.y * b.y), c * (a.y * b.x + a.x * b.y)};
  a = t;
}

__global__ void modulateAndNormalize_kernel(fComplex *d_Dst, fComplex *d_Src,
                                            int dataSize, float c) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= dataSize) {
    return;
  }

  fComplex a = d_Src[i];
  fComplex b = d_Dst[i];

  mulAndScale(a, b, c);

  d_Dst[i] = a;
}

////////////////////////////////////////////////////////////////////////////////
// 3D R2C / C2R post/preprocessing kernels
////////////////////////////////////////////////////////////////////////////////
#if (USE_TEXTURE)
#define LOAD_FCOMPLEX(i) tex1Dfetch<fComplex>(texComplex, i)
#define LOAD_FCOMPLEX_A(i) tex1Dfetch<fComplex>(texComplexA, i)
#define LOAD_FCOMPLEX_B(i) tex1Dfetch<fComplex>(texComplexB, i)

#define SET_FCOMPLEX_BASE
#define SET_FCOMPLEX_BASE_A
#define SET_FCOMPLEX_BASE_B
#else
#define LOAD_FCOMPLEX(i) d_Src[i]
#define LOAD_FCOMPLEX_A(i) d_SrcA[i]
#define LOAD_FCOMPLEX_B(i) d_SrcB[i]

#define SET_FCOMPLEX_BASE
#define SET_FCOMPLEX_BASE_A
#define SET_FCOMPLEX_BASE_B
#endif

inline __device__ void spPostprocessC2C(fComplex &D1, fComplex &D2,
                                        const fComplex &twiddle) {
  float A1 = 0.5f * (D1.x + D2.x);
  float B1 = 0.5f * (D1.y - D2.y);
  float A2 = 0.5f * (D1.y + D2.y);
  float B2 = 0.5f * (D1.x - D2.x);

  D1.x = A1 + (A2 * twiddle.x + B2 * twiddle.y);
  D1.y = (A2 * twiddle.y - B2 * twiddle.x) + B1;
  D2.x = A1 - (A2 * twiddle.x + B2 * twiddle.y);
  D2.y = (A2 * twiddle.y - B2 * twiddle.x) - B1;
}

// Premultiply by 2 to account for 1.0 / (DZ * DY * DX) normalization
inline __device__ void spPreprocessC2C(fComplex &D1, fComplex &D2,
                                       const fComplex &twiddle) {
  float A1 = /* 0.5f * */ (D1.x + D2.x);
  float B1 = /* 0.5f * */ (D1.y - D2.y);
  float A2 = /* 0.5f * */ (D1.y + D2.y);
  float B2 = /* 0.5f * */ (D1.x - D2.x);

  D1.x = A1 - (A2 * twiddle.x - B2 * twiddle.y);
  D1.y = (B2 * twiddle.x + A2 * twiddle.y) + B1;
  D2.x = A1 + (A2 * twiddle.x - B2 * twiddle.y);
  D2.y = (B2 * twiddle.x + A2 * twiddle.y) - B1;
}

inline __device__ void getTwiddle(fComplex &twiddle, float phase) {
  __sincosf(phase, &twiddle.y, &twiddle.x);
}

inline __device__ uint mod(uint a, uint DA) {
  //(DA - a) % DA, assuming a <= DA
  return a ? (DA - a) : a;
}

static inline uint factorRadix2(uint &log2N, uint n) {
  if (!n) {
    log2N = 0;
    return 0;
  } else {
    for (log2N = 0; n % 2 == 0; n /= 2, log2N++)
      ;

    return n;
  }
}

inline __device__ void udivmod(uint &dividend, uint divisor, uint &rem) {
#if (!POWER_OF_TWO)
  rem = dividend % divisor;
  dividend /= divisor;
#else
  rem = dividend & (divisor - 1);
  dividend >>= (__ffs(divisor) - 1);
#endif
}

__global__ void spPostprocess3D_kernel(fComplex *d_Dst, fComplex *d_Src,
                                       uint DZ, uint DY, uint DX, uint threadCount,
                                       uint padding, float phaseBase
#if (USE_TEXTURE)
                                       ,
                                       cudaTextureObject_t texComplex
#endif
                                       ) {
  const uint threadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadId >= threadCount) {
    return;
  }

  uint x, y, z, i = threadId;
  udivmod(i, DX / 2, x);
  udivmod(i, DY, y);
  udivmod(i, DZ, z);

  // Avoid overwrites in columns DX / 2 by different threads
  if ((x == 0) && (y > DY / 2)) {
    return;
  }

  const uint srcOffset = i * DZ * DY * DX;
  const uint dstOffset = i * DZ * DY * (DX + padding);

  // Process x = [0 .. DX / 2 - 1] U [DX / 2 + 1 .. DX]
  {
    const uint loadPos1 = srcOffset + z * DY * DX + y * DX + x;
    const uint loadPos2 = srcOffset + mod(z,DZ) * DY * DX + mod(y, DY) * DX + mod(x, DX);
    const uint storePos1 = dstOffset + z * DY * (DX + padding) + y * (DX + padding) + x;
    const uint storePos2 = dstOffset + mod(z , DZ) * DY * (DX + padding) + mod(y, DY) * (DX + padding) + (DX - x);

    fComplex D1 = LOAD_FCOMPLEX(loadPos1);
    fComplex D2 = LOAD_FCOMPLEX(loadPos2);

    fComplex twiddle;
    getTwiddle(twiddle, phaseBase * (float)x);
    spPostprocessC2C(D1, D2, twiddle);

    d_Dst[storePos1] = D1;
    d_Dst[storePos2] = D2;
  }

  // Process x = DX / 2
  if (x == 0) {
    const uint loadPos1 = srcOffset + z * DY * DX + y * DX + DX / 2;
    const uint loadPos2 = srcOffset + mod(z,DZ) *DY * DX + mod(y, DY) * DX + DX / 2;
    const uint storePos1 = dstOffset + z * DY * (DX + padding)  + y * (DX + padding) + DX / 2;
    const uint storePos2 = dstOffset + mod(z, DZ) * DY * ( DX + padding ) + mod(y, DY) * (DX + padding) + DX / 2;

    fComplex D1 = LOAD_FCOMPLEX(loadPos1);
    fComplex D2 = LOAD_FCOMPLEX(loadPos2);

    // twiddle = getTwiddle(phaseBase * (DX / 2)) = exp(dir * j * PI / 2)
    fComplex twiddle = {0, (phaseBase > 0) ? 1.0f : -1.0f};
    spPostprocessC2C(D1, D2, twiddle);

    d_Dst[storePos1] = D1;
    d_Dst[storePos2] = D2;
  }
}

__global__ void spPreprocess3D_kernel(fComplex *d_Dst, fComplex *d_Src, uint DZ, uint DY,
                                      uint DX, uint threadCount, uint padding,
                                      float phaseBase
#if (USE_TEXTURE)
                                      ,
                                      cudaTextureObject_t texComplex
#endif
                                      ) {
  const uint threadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadId >= threadCount) {
    return;
  }
  // ??
  uint x, y, z, i = threadId;
  udivmod(i, DX / 2, x);
  udivmod(i, DY, y);
  udivmod(i, DZ, z);

  // Avoid overwrites in columns 0 and DX / 2 by different threads (lower and
  // upper halves)
  if ((x == 0) && (y > DY / 2)) {
    return;
  }

  const uint srcOffset = i * DZ * DY * (DX + padding);
  const uint dstOffset = i * DZ * DY * DX;

  // Process x = [0 .. DX / 2 - 1] U [DX / 2 + 1 .. DX]
  {
    const uint loadPos1 = srcOffset + z * DY * (DX + padding) + y * (DX + padding) + x;
    const uint loadPos2 = srcOffset + mod(z,DZ) * DY * (DX + padding) + mod(y, DY) * (DX + padding) + (DX - x);
    const uint storePos1 = dstOffset + z * DY * DX + y * DX + x;
    const uint storePos2 = dstOffset + mod(z, DZ) * DY * DX + mod(y, DY) * DX + mod(x, DX);

    fComplex D1 = LOAD_FCOMPLEX(loadPos1);
    fComplex D2 = LOAD_FCOMPLEX(loadPos2);

    fComplex twiddle;
    getTwiddle(twiddle, phaseBase * (float)x);
    spPreprocessC2C(D1, D2, twiddle);

    d_Dst[storePos1] = D1;
    d_Dst[storePos2] = D2;
  }

  // Process x = DX / 2
  if (x == 0) {
    const uint loadPos1 = srcOffset + z * DY * (DX + padding) + y * (DX + padding) + DX / 2;
    const uint loadPos2 = srcOffset + mod(z, DZ) * DY * (DX + padding) + mod(y, DY) * (DX + padding) + DX / 2;
    const uint storePos1 = dstOffset + z* DY * DX + y * DX + DX / 2;
    const uint storePos2 = dstOffset + mod(z, DZ) * DY * DX + mod(y, DY) * DX + DX / 2;

    fComplex D1 = LOAD_FCOMPLEX(loadPos1);
    fComplex D2 = LOAD_FCOMPLEX(loadPos2);

    // twiddle = getTwiddle(phaseBase * (DX / 2)) = exp(-dir * j * PI / 2)
    fComplex twiddle = {0, (phaseBase > 0) ? 1.0f : -1.0f};
    spPreprocessC2C(D1, D2, twiddle);

    d_Dst[storePos1] = D1;
    d_Dst[storePos2] = D2;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Combined spPostprocess3D + modulateAndNormalize + spPreprocess3D
////////////////////////////////////////////////////////////////////////////////
__global__ void spProcess3D_kernel(fComplex *d_Dst, fComplex *d_SrcA,
                                   fComplex *d_SrcB, uint DZ, uint DY, uint DX,
                                   uint threadCount, float phaseBase, float c
#if (USE_TEXTURE)
                                   ,
                                   cudaTextureObject_t texComplexA,
                                   cudaTextureObject_t texComplexB
#endif
                                   ) {
  const uint threadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadId >= threadCount) {
    return;
  }

  uint x, y, z, i = threadId;
  udivmod(i, DX, x);
  udivmod(i, DY, y);
  udivmod(i, DZ/2, z);

  const uint offset = i * DZ * DY * DX;

  // Avoid overwrites in rows 0 and DY / 2 by different threads (left and right
  // halves) Otherwise correctness for in-place transformations is affected
  if ((z == 0) && (y > DY / 2)) {
    return;
  }

  fComplex twiddle;

  // Process z = [0 .. DZ / 2 - 1] U [DZ - (DZ / 2) + 1 .. DZ - 1]
  {
    const uint pos1 = offset + z * DY * DX + y * DX + x;
    const uint pos2 = offset + mod(z, DZ) * DY * DX + mod(y, DY) * DX + mod(x, DX);

    fComplex D1 = LOAD_FCOMPLEX_A(pos1);
    fComplex D2 = LOAD_FCOMPLEX_A(pos2);
    fComplex K1 = LOAD_FCOMPLEX_B(pos1);
    fComplex K2 = LOAD_FCOMPLEX_B(pos2);
    getTwiddle(twiddle, phaseBase * (float)x);

    spPostprocessC2C(D1, D2, twiddle);
    spPostprocessC2C(K1, K2, twiddle);
    mulAndScale(D1, K1, c);
    mulAndScale(D2, K2, c);
    spPreprocessC2C(D1, D2, twiddle);

    d_Dst[pos1] = D1;
    d_Dst[pos2] = D2;
  }

  if (z == 0) {
    const uint pos1 = offset + (DZ/2) * DY * DX + y * DX + x;
    const uint pos2 = offset + (DZ/2) * DY * DX + mod(y,DY) * DX + mod(x, DX);

    fComplex D1 = LOAD_FCOMPLEX_A(pos1);
    fComplex D2 = LOAD_FCOMPLEX_A(pos2);
    fComplex K1 = LOAD_FCOMPLEX_B(pos1);
    fComplex K2 = LOAD_FCOMPLEX_B(pos2);

    spPostprocessC2C(D1, D2, twiddle);
    spPostprocessC2C(K1, K2, twiddle);
    mulAndScale(D1, K1, c);
    mulAndScale(D2, K2, c);
    spPreprocessC2C(D1, D2, twiddle);

    d_Dst[pos1] = D1;
    d_Dst[pos2] = D2;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Position convolution kernel center at (0, 0) in the image
////////////////////////////////////////////////////////////////////////////////
extern "C" void padKernel(float *d_Dst, float *d_Src, int fftD, int fftH, int fftW,
                          int kernelD, int kernelH, int kernelW, int kernelZ, int kernelY, int kernelX) {
  assert(d_Src != d_Dst);
  dim3 threads(8, 8, 4);
  dim3 grid(iDivUp(kernelW, threads.x), iDivUp(kernelH, threads.y), iDivUp(kernelD, threads.z));

  SET_FLOAT_BASE;
#if (USE_TEXTURE)
  cudaTextureObject_t texFloat;
  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeLinear;
  texRes.res.linear.devPtr = d_Src;
  texRes.res.linear.sizeInBytes = sizeof(float) * kernelH * kernelW * kernelD;
  texRes.res.linear.desc = cudaCreateChannelDesc<float>();

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = false;
  texDescr.filterMode = cudaFilterModeLinear;
  texDescr.addressMode[0] = cudaAddressModeWrap;
  texDescr.readMode = cudaReadModeElementType;

  checkCudaErrors(cudaCreateTextureObject(&texFloat, &texRes, &texDescr, NULL));
#endif

  padKernel_kernel<<<grid, threads>>>(d_Dst, d_Src, fftD, fftH, fftW, kernelD,
                                      kernelH, kernelW, kernelZ, kernelY, kernelX
#if (USE_TEXTURE)
                                      ,
                                      texFloat
#endif
                                      );
  getLastCudaError("padKernel_kernel<<<>>> execution failed\n");

#if (USE_TEXTURE)
  checkCudaErrors(cudaDestroyTextureObject(texFloat));
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Prepare data for "pad to border" addressing mode
////////////////////////////////////////////////////////////////////////////////
extern "C" void padDataClampToBorder(float *d_Dst, float *d_Src, int fftD,
                                     int fftH, int fftW, int dataD, int dataH, int dataW,
                                     int kernelD, int kernelH, int kernelW, int kernelZ,
                                     int kernelY, int kernelX) {
  assert(d_Src != d_Dst);
  dim3 threads(8, 8, 4);
  dim3 grid(iDivUp(fftW, threads.x), iDivUp(fftH, threads.y), iDivUp(fftD, threads.z));

#if (USE_TEXTURE)
  cudaTextureObject_t texFloat;
  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeLinear;
  texRes.res.linear.devPtr = d_Src;
  texRes.res.linear.sizeInBytes = sizeof(float) * dataH * dataW * dataD;
  texRes.res.linear.desc = cudaCreateChannelDesc<float>();

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = false;
  texDescr.filterMode = cudaFilterModeLinear;
  texDescr.addressMode[0] = cudaAddressModeWrap;
  texDescr.readMode = cudaReadModeElementType;

  checkCudaErrors(cudaCreateTextureObject(&texFloat, &texRes, &texDescr, NULL));
#endif

  padDataClampToBorder_kernel<<<grid, threads>>>(
      d_Dst, d_Src, fftD, fftH, fftW, dataD, dataH, dataW, kernelD, kernelH, kernelW, kernelZ, kernelY, kernelX
#if (USE_TEXTURE)
      ,
      texFloat
#endif
      );
  getLastCudaError("padDataClampToBorder_kernel<<<>>> execution failed\n");

#if (USE_TEXTURE)
  checkCudaErrors(cudaDestroyTextureObject(texFloat));
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Modulate Fourier image of padded data by Fourier image of padded kernel
// and normalize by FFT size
////////////////////////////////////////////////////////////////////////////////
extern "C" void modulateAndNormalize(fComplex *d_Dst, fComplex *d_Src, int fftD,
                                     int fftH, int fftW, int padding) {
  assert(fftW % 2 == 0);
  const int dataSize = fftD * fftH *(fftW / 2 + padding);
  
  modulateAndNormalize_kernel<<<iDivUp(dataSize, 256), 256 >>>(
      d_Dst, d_Src, dataSize, 1.0f / (float)(fftW * fftH * fftD));
  getLastCudaError("modulateAndNormalize() execution failed\n");
}

////////////////////////////////////////////////////////////////////////////////
// 3D R2C / C2R post/preprocessing kernels
////////////////////////////////////////////////////////////////////////////////
static const double PI = 3.1415926535897932384626433832795;
static const uint BLOCKDIM = 256;

extern "C" void spPostprocess3D(void *d_Dst, void *d_Src, uint DZ, uint DY, uint DX,
                                uint padding, int dir) {
  assert(d_Src != d_Dst);
  assert(DX % 2 == 0);

#if (POWER_OF_TWO)
  uint log2DX, log2DY, log2DZ;
  uint factorizationRemX = factorRadix2(log2DX, DX);
  uint factorizationRemY = factorRadix2(log2DY, DY);
  uint factorizationRemZ = factorRadix2(log2DZ, DZ);
  assert(factorizationRemX == 1 && factorizationRemY == 1 && factorizationRemZ == 1);
#endif

  const uint threadCount = DZ * DY * (DX / 2);
  const double phaseBase = dir * PI / (double)DX;

#if (USE_TEXTURE)
  cudaTextureObject_t texComplex;
  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeLinear;
  texRes.res.linear.devPtr = d_Src;
  texRes.res.linear.sizeInBytes = sizeof(fComplex) * DZ * DY * (DX + padding);
  texRes.res.linear.desc = cudaCreateChannelDesc<fComplex>();

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = false;
  texDescr.filterMode = cudaFilterModeLinear;
  texDescr.addressMode[0] = cudaAddressModeWrap;
  texDescr.readMode = cudaReadModeElementType;

  checkCudaErrors(
      cudaCreateTextureObject(&texComplex, &texRes, &texDescr, NULL));
#endif

  spPostprocess3D_kernel<<<iDivUp(threadCount, BLOCKDIM), BLOCKDIM>>>(
      (fComplex *)d_Dst, (fComplex *)d_Src, DZ, DY, DX, threadCount, padding,
      (float)phaseBase
#if (USE_TEXTURE)
      ,
      texComplex
#endif
      );
  getLastCudaError("spPostprocess3D_kernel<<<>>> execution failed\n");

#if (USE_TEXTURE)
  checkCudaErrors(cudaDestroyTextureObject(texComplex));
#endif
}

extern "C" void spPreprocess3D(void *d_Dst, void *d_Src, uint DZ, uint DY, uint DX,
                               uint padding, int dir) {
  assert(d_Src != d_Dst);
  assert(DX % 2 == 0);

#if (POWER_OF_TWO)
  uint log2DX, log2DY, log2DZ;
  uint factorizationRemX = factorRadix2(log2DX, DX);
  uint factorizationRemY = factorRadix2(log2DY, DY);
  uint factorizationRemZ = factorRadix2(log2DZ, DZ);
  assert(factorizationRemX == 1 && factorizationRemY == 1 && factorizationRemZ == 1);
#endif

  const uint threadCount = DZ * DY * (DX / 2);
  const double phaseBase = -dir * PI / (double)DX;

#if (USE_TEXTURE)
  cudaTextureObject_t texComplex;
  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeLinear;
  texRes.res.linear.devPtr = d_Src;
  texRes.res.linear.sizeInBytes = sizeof(fComplex) * DZ * DY * (DX + padding);
  texRes.res.linear.desc = cudaCreateChannelDesc<fComplex>();

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = false;
  texDescr.filterMode = cudaFilterModeLinear;
  texDescr.addressMode[0] = cudaAddressModeWrap;
  texDescr.readMode = cudaReadModeElementType;

  checkCudaErrors(
      cudaCreateTextureObject(&texComplex, &texRes, &texDescr, NULL));
#endif
  spPreprocess3D_kernel<<<iDivUp(threadCount, BLOCKDIM), BLOCKDIM>>>(
      (fComplex *)d_Dst, (fComplex *)d_Src, DZ, DY, DX, threadCount, padding,
      (float)phaseBase
#if (USE_TEXTURE)
      ,
      texComplex
#endif
      );
  getLastCudaError("spPreprocess3D_kernel<<<>>> execution failed\n");

#if (USE_TEXTURE)
  checkCudaErrors(cudaDestroyTextureObject(texComplex));
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Combined spPostprocess3D + modulateAndNormalize + spPreprocess3D
////////////////////////////////////////////////////////////////////////////////
extern "C" void spProcess3D(void *d_Dst, void *d_SrcA, void *d_SrcB, uint DZ, uint DY,
                            uint DX, int dir) {
  assert(DZ % 2 == 0);

#if (POWER_OF_TWO)
  uint log2DX, log2DY, log2DZ;
  uint factorizationRemX = factorRadix2(log2DX, DX);
  uint factorizationRemY = factorRadix2(log2DY, DY);
  uint factorizationRemZ = factorRadix2(log2DZ, DZ);
  assert(factorizationRemX == 1 && factorizationRemY == 1 && factorizationRemZ == 1);
#endif

  const uint threadCount = (DZ / 2) * DY * DX; //TODO not sure why other side
  const double phaseBase = dir * PI / (double)DX;

#if (USE_TEXTURE)
  cudaTextureObject_t texComplexA, texComplexB;
  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeLinear;
  texRes.res.linear.devPtr = d_SrcA;
  texRes.res.linear.sizeInBytes = sizeof(fComplex) * DZ * DY * DX;
  texRes.res.linear.desc = cudaCreateChannelDesc<fComplex>();

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = false;
  texDescr.filterMode = cudaFilterModeLinear;
  texDescr.addressMode[0] = cudaAddressModeWrap;
  texDescr.readMode = cudaReadModeElementType;

  checkCudaErrors(
      cudaCreateTextureObject(&texComplexA, &texRes, &texDescr, NULL));

  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeLinear;
  texRes.res.linear.devPtr = d_SrcB;
  texRes.res.linear.sizeInBytes = sizeof(fComplex) * DZ * DY * DX;
  texRes.res.linear.desc = cudaCreateChannelDesc<fComplex>();

  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = false;
  texDescr.filterMode = cudaFilterModeLinear;
  texDescr.addressMode[0] = cudaAddressModeWrap;
  texDescr.readMode = cudaReadModeElementType;

  checkCudaErrors(
      cudaCreateTextureObject(&texComplexB, &texRes, &texDescr, NULL));
#endif
  spProcess3D_kernel<<<iDivUp(threadCount, BLOCKDIM), BLOCKDIM>>>(
      (fComplex *)d_Dst, (fComplex *)d_SrcA, (fComplex *)d_SrcB, DZ, DY, DX,
      threadCount, (float)phaseBase, 0.5f / (float)(DZ * DY * DX)
#if (USE_TEXTURE)
                                         ,
      texComplexA, texComplexB
#endif
      );
  getLastCudaError("spProcess3D_kernel<<<>>> execution failed\n");

#if (USE_TEXTURE)
  checkCudaErrors(cudaDestroyTextureObject(texComplexA));
  checkCudaErrors(cudaDestroyTextureObject(texComplexB));
#endif
}
