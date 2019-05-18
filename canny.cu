#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "CImg.h"

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

using namespace cimg_library;

__global__


int main(int argc, char **argv) {

  int threadCountX = 1000;
  int threadCountY = 1000;
  int threadCount = threadCountX*threadCountY;
  
  CImg<unsigned char> image("images/test1.jpg"),
    gray(image.width(), image.height(), 1, 1, 0);
  image.blur(1);
  for(int i = 0; i < image.width(); i++){
    for(int j = 0; j < image.height(); j++){
      int grayValueWeight = (int)(0.299*image(i, j, 0) +
        0.587*image(i, j, 1) + 0.114*image(i, j, 2));
      gray(i, j, 0) = grayValueWeight;
    }
  }
  printf("w= %d, h= %d\n", image.width(), image.height());
  CImgDisplay local(gray, "Hah");
  while (true) {
    local.wait();
  }
  
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
  printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
  deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);
  
  return 0;
}