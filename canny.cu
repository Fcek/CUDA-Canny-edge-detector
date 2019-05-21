#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "CImg.h"

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

using namespace cimg_library;

__global__
void colorToGrey(float *pi){
  // int x = threadIdx.x + blockIdx.x * blockDim.x; 
  // int y = threadIdx.y + blockIdx.y * blockDim.y;
  // int tx = threadIdx.x;
  // int tbx = blockIdx.x;
  // int ty = threadIdx.y;
  // int tby = blockIdx.y;
  pi[0] = 3.14;
  printf("done");
  // for(int i = 0; i < image.width(); i++){
  //   for(int j = 0; j < image.height(); j++){
  //     int grayValueWeight = (int)(0.299*image(i, j, 0) +
  //       0.587*image(i, j, 1) + 0.114*image(i, j, 2));
  //     gray(i, j, 0) = grayValueWeight;
  //   }
  // }
}


int main(int argc, char **argv) {

  int *r, *g, *b;
  int *r_gpu, *g_gpu, *b_gpu, *a_gpu;
  int height, width;
  int h_gpu, w_gpu;
  int size;
  float *pi, pi1;

  dim3 dimBlock(32, 32);
  dim3 dimGrid(height/32, width/32);
  
  CImg<int> image("images/test1.jpg"),
    imageGPU(image.width(), image.height()),
    gray(image.width(), image.height(), 1, 1, 0),
    grayGPU(image.width(), image.height(), 1, 1, 0);
  //image.blur(2.5);

  height = image.height();
  width = image.width();

  size = sizeof(int)* width * height;

  r = (int *)malloc(size);
  g = (int *)malloc(size);
  b = (int *)malloc(size);
  // for(int i = 0; i < image.height(); i++){
  //   r[i] = (int *)malloc(sizeof(int) * image.width());
  //   g[i] = (int *)malloc(sizeof(int) * image.width());
  //   b[i] = (int *)malloc(sizeof(int) * image.width());
  // }

  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      //printf("abc");
      r[i*height+j] = image(i, j, 0);
      g[i*height+j] = image(i, j, 1);
      b[i*height+j] = image(i, j, 2);
    }
  }  

  printf(" %d",size);
  printf("w= %d, h= %d %d\n", image.width(), image.height(), sizeof(float));
  CImgDisplay local(image, "Hah");

  // This will pick the best possible CUDA capable device
  cudaDeviceProp deviceProp;
  int devID = findCudaDevice(argc, (const char **)argv);
  if (devID < 0)
  {
    printf("exiting...\n");
    exit(EXIT_SUCCESS);
  }
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

  // Statistics about the GPU device
  printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n",
  deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);
  int version = (deviceProp.major * 0x10 + deviceProp.minor);
  if (version < 0x11)
  {
      printf("%s: requires a minimum CUDA compute 1.1 capability\n", "Canny edge detector");
      cudaDeviceReset();
      exit(EXIT_SUCCESS);
  }

  printf("abc %d", size);
  fflush(stdout);
  checkCudaErrors(cudaMalloc((void **)&r_gpu, size));
  checkCudaErrors(cudaMalloc((void **)&g_gpu, size));
  checkCudaErrors(cudaMalloc((void **)&b_gpu, size));
  checkCudaErrors(cudaMalloc((void **)&a_gpu, size));
  checkCudaErrors(cudaMalloc((void **)&pi, sizeof(float)));
  // checkCudaErrors(cudaMalloc((void **)&h_gpu, sizeof(int)));

  cudaMemcpy(r_gpu, r, size, cudaMemcpyHostToDevice);
  cudaMemcpy(g_gpu, g, size, cudaMemcpyHostToDevice);
  cudaMemcpy(b_gpu, b, size, cudaMemcpyHostToDevice);
  // cudaMemcpy(w_gpu, width, sizeof(int), cudaMemcpyHostToDevice);
  // cudaMemcpy(h_gpu, height, sizeof(int), cudaMemcpyHostToDevice);

  //fflush(stdout);
  
  colorToGrey<<<1, 1>>>(pi);
  cudaMemcpy(&pi1, &pi[0], sizeof(float), cudaMemcpyDeviceToHost);
  printf("abc%f", pi1);
  fflush(stdout);
  // free(r);
  // free(g);
  // free(b);

  return 0;
}