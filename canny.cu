#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "CImg.h"

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

using namespace cimg_library;

__global__
void colorToGrey(){
  // for(int i = 0; i < image.width(); i++){
  //   for(int j = 0; j < image.height(); j++){
  //     int grayValueWeight = (int)(0.299*image(i, j, 0) +
  //       0.587*image(i, j, 1) + 0.114*image(i, j, 2));
  //     gray(i, j, 0) = grayValueWeight;
  //   }
  // }
}


int main(int argc, char **argv) {

  int threadCountX = 1000;
  int threadCountY = 1000;
  int threadCount = threadCountX*threadCountY;
  int **r, **g, **b;
  int size;
  
  CImg<int> image("images/test1.jpg"),
    imageGPU(image.width(), image.height()),
    gray(image.width(), image.height(), 1, 1, 0),
    grayGPU(image.width(), image.height(), 1, 1, 0);
  image.blur(2.5);

  size = sizeof(int)* image.width() * image.height();

  r = (int **)malloc(sizeof(int *) * image.height());
  g = (int **)malloc(sizeof(int *) * image.height());
  b = (int **)malloc(sizeof(int *) * image.height());
  for(int i = 0; i < image.height(); i++){
    r[i] = (int *)malloc(sizeof(int) * image.width());
    g[i] = (int *)malloc(sizeof(int) * image.width());
    b[i] = (int *)malloc(sizeof(int) * image.width());
  }

  for(int i = 0; i < image.height(); i++){

    for(int j = 0; j < image.width(); j++){
      //printf("abc");
      r[i][j] = image(i, j, 0);
      g[i][j] = image(i, j, 1);
      b[i][j] = image(i, j, 2);

      printf(" %d",r[i][j]);
    }
  }  

  printf("w= %d, h= %d %d\n", image.width(), image.height(), sizeof(float));
  CImgDisplay local(gray, "Hah");
  
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

  //checkCudaErrors(cudaMalloc(image, sizeof(image)));

  return 0;
}