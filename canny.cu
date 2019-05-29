#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "CImg.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

using namespace cimg_library;

__global__ void
colorToGrey(int threadCount, int threadsPerBlock, int blocksAmount, int *r, int *g, int *b, int *a, int h, int w){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  // int tx = threadIdx.x;
  // int bx = blockIdx.x;
  // int ty = threadIdx.y;
  // int by = blockIdx.y;
  for(int i = 0; i < w; i = i + threadsPerBlock*blocksAmount){
    // printf("i = %d\n", i);
    for(int j = 0; j < h; j = j + threadsPerBlock*blocksAmount){
      // printf("j = %d\n", j);
      if(x + i < w && y + j < h){
        a[(x+i)*h+(y+j)] = (int)(0.299*r[(x+i)*h+(y+j)] +
          0.587*g[(x+i)*h+(y+j)] + 0.114*b[(x+i)*h+(y+j)]);
          // printf("a = %d\n", a[(x+i)*h+(y+j)]);
      }
    }
  }
}

__global__ void 
gradientCalc(int threadCount, int threadsPerBlock, int blocksAmount, int *a, float *grad, float *theta, int h, int w){
  int sobelX[3][3] = {{-1,0,1},
                      {-2,0,2},
                      {-1,0,1}};
  int sobelY[3][3] = {{-1,-2,-1},
                      {0,0,0},
                      {1,2,1}};
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int iX, iY;
  
  for(int i = 1; i < w - 1; i = i + threadsPerBlock*blocksAmount){
    // printf("i = %d\n", i);
    for(int j = 1; j < h - 1; j = j + threadsPerBlock*blocksAmount){
      // printf("j = %d\n", j);
      if(x + i < w && y + j < h){
        iX = sobelX[2][2] * a[(x+i-1)*h+(y+j-1)] + sobelX[2][1] * a[(x+i)*h+(y+j-1)] + sobelX[2][0] * a[(x+i+1)*h+(y+j-1)] +
          sobelX[1][2] * a[(x+i-1)*h+(y+j)] + sobelX[1][1] * a[(x+i)*h+(y+j)] + sobelX[1][0] * a[(x+i+1)*h+(y+j)] +
          sobelX[0][2] * a[(x+i-1)*h+(y+j+1)] + sobelX[0][1] * a[(x+i)*h+(y+j+1)] + sobelX[0][0] * a[(x+i+1)*h+(y+j+1)];
        iY = sobelY[2][2] * a[(x+i-1)*h+(y+j-1)] + sobelY[2][1] * a[(x+i)*h+(y+j-1)] + sobelY[2][0] * a[(x+i+1)*h+(y+j-1)] +
          sobelY[1][2] * a[(x+i-1)*h+(y+j)] + sobelY[1][1] * a[(x+i)*h+(y+j)] + sobelY[1][0] * a[(x+i+1)*h+(y+j)] +
          sobelY[0][2] * a[(x+i-1)*h+(y+j+1)] + sobelY[0][1] * a[(x+i)*h+(y+j+1)] + sobelY[0][0] * a[(x+i+1)*h+(y+j+1)];
        grad[(x+i)*h+(y+j)] = hypotf((float)iX, (float)iY);
        theta[(x+i)*h+(y+j)] = atan2((float)iY, (float)iX);
        // printf("%d. grad = %f   iX = %d   iY = %d\n", x, grad[(x+i)*h+(y+j)], iX, iY);
      }
    }
  }
}

__global__ void
normGrad(int threadCount, int threadsPerBlock, int blocksAmount, float *grad, int maxGrad, int *normGrad, int h, int w){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  for(int i = 1; i < w - 1; i = i + threadsPerBlock*blocksAmount){
    // printf("i = %d\n", i);
    for(int j = 1; j < h - 1; j = j + threadsPerBlock*blocksAmount){
      // printf("j = %d\n", j);
      if(x + i < w && y + j < h){
        normGrad[(x+i)*h+(y+j)] = (int)((grad[(x+i)*h+(y+j)] / grad[maxGrad]) * 255.0);
        //printf("%d. grad = %f   norm = %d   max = %f\n", x, grad[(x+i)*h+(y+j)], normGrad[(x+i)*h+(y+j)], grad[maxGrad]);
      }
    }
  }
}

// void
// nonMaxSupp(int threadCount, int threadsPerBlock, int blocksAmount, int *grad, float *theta, int *nonMax, int h, int w){
//   // int x = blockIdx.x * blockDim.x + threadIdx.x;
//   // int y = blockIdx.y * blockDim.y + threadIdx.y;
//   int x = 0;
//   int y = 0;
//   float degree;
//   int q, r;
//   for(int i = 1; i < w - 1; i = i + threadsPerBlock*blocksAmount){
//     // printf("i = %d\n", i);
//     for(int j = 1; j < h - 1; j = j + threadsPerBlock*blocksAmount){
//       // printf("j = %d\n", j);
//       if(x + i < w && y + j < h){
//         //printf("%d. grad = %f   norm = %d   max = %f\n", x, grad[(x+i)*h+(y+j)], normGrad[(x+i)*h+(y+j)], grad[maxGrad]);
//         degree = theta[(x+i)*h+(y+j)] * (180.0 / M_PI);
//         // printf("degree  %f\n", degree);
//         if(degree < 0.0){
//           degree += 180.0;
//           // printf("degree < 0 %f\n", degree);
//         }

//         q = 255;
//         r = 255;
        
//         if((degree >= 0.0 && degree < 22.5) || (degree >= 157.5 && degree <= 180.0)){
//           q = grad[(x+i)*h+(y+j+1)];
//           r = grad[(x+i)*h+(y+j-1)];
//         } else if(degree >= 22.5 && degree < 67.5){
//           q = grad[(x+i+1)*h+(y+j-1)];
//           r = grad[(x+i-1)*h+(y+j+1)];
//         } else if(degree >= 67.5 && degree < 112.5){
//           q = grad[(x+i+1)*h+(y+j)];
//           r = grad[(x+i-1)*h+(y+j)];
//         } else if(degree >= 112.5 && degree < 157.5){
//           q = grad[(x+i-1)*h+(y-j)];
//           r = grad[(x+i+1)*h+(y+j)];
//         }

//         if(grad[(x+i)*h+(y+j)] >= q && grad[(x+i)*h+(y+j)] >= r){
//           nonMax[(x+i)*h+(y+j)] = grad[(x+i)*h+(y+j)];
//           // printf("edge \n");
//         } else {
//           nonMax[(x+i)*h+(y+j)] = 0;
//         }
//         if(nonMax[(x+i)*h+(y+j)] > 0)
//           printf("nonMax = %d \n", nonMax[(x+i)*h+(y+j)]);
//       }
//       y++;
//     }
//     x++;
//   }
// }

void
nonMaxSupp(int threadCount, int threadsPerBlock, int blocksAmount, int *grad, float *theta, int *nonMax, int h, int w){
  // int x = blockIdx.x * blockDim.x + threadIdx.x;
  // int y = blockIdx.y * blockDim.y + threadIdx.y;
  float degree;
  int q, r;
  w = w - 2;
  h = h - 2;
  for(int i = 1; i < w - 1; i++){
    // printf("i = %d\n", i);
    for(int j = 1; j < h - 1; j++){
      // printf("j = %d\n", j);
      if(i < w && j < h){
        //printf("%d. grad = %f   norm = %d   max = %f\n", x, grad[(x+i)*h+(y+j)], normGrad[(x+i)*h+(y+j)], grad[maxGrad]);
        degree = theta[(i)*h+(j)] * (180.0 / M_PI);
        // printf("degree  %f\n", degree);
        if(degree < 0.0){
          degree += 180.0;
          // printf("degree < 0 %f\n", degree);
        }

        q = 255;
        r = 255;
        
        if((degree >= 0.0 && degree < 22.5) || (degree >= 157.5 && degree <= 180.0)){
          q = grad[(i)*h+(j+1)];
          r = grad[(i)*h+(j-1)];
        } else if(degree >= 22.5 && degree < 67.5){
          q = grad[(i+1)*h+(j-1)];
          r = grad[(i-1)*h+(j+1)];
        } else if(degree >= 67.5 && degree < 112.5){
          q = grad[(i+1)*h+(j)];
          r = grad[(i-1)*h+(j)];
        } else if(degree >= 112.5 && degree < 157.5){
          q = grad[(i-1)*h+(j)];
          r = grad[(i+1)*h+(j)];
        }
        // printf("%d. q = %d r = %d\n",i*j,q,r);
        if(grad[(i)*h+(j)] >= q && grad[(i)*h+(j)] >= r){
          nonMax[(i)*h+(j)] = grad[(i)*h+(j)];
          // printf("edge \n");
        } else {
          nonMax[(i)*h+(j)] = 0;
        }
        // if(nonMax[(i)*h+(j)] > 20)
        //   printf("nonMax = %d \n", nonMax[(i)*h+(j)]);
      }
    }
  }
}

__global__ void
threshold(int threadCount, int threadsPerBlock, int blocksAmount, int *nonMax, int indexMax, int h, int w){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  float lowThresholdRatio = 0.05;
  float highThresholdRatio = 0.09;
  // printf("%d", nonMax[indexMax]);
  float highThreshold = nonMax[indexMax] * highThresholdRatio;
  float lowThreshold = highThreshold * lowThresholdRatio;

  for(int i = 1; i < w - 1; i = i + threadsPerBlock*blocksAmount){
    // printf("i = %d\n", i);
    for(int j = 1; j < h - 1; j = j + threadsPerBlock*blocksAmount){
      // printf("j = %d\n", j);
      if(x + i < w && y + j < h){
        if(nonMax[(x+i)*h+(y+j)] >= highThreshold){
          nonMax[(x+i)*h+(y+j)] = 255;
        } else if(nonMax[(x+i)*h+(y+j)] <= highThreshold && nonMax[(x+i)*h+(y+j)] >= lowThreshold){
          nonMax[(x+i)*h+(y+j)] = 25;
        } else {
          nonMax[(x+i)*h+(y+j)] = 0;
        }
      }
    }
  }
}

__global__ void
hysteresis(int threadCount, int threadsPerBlock, int blocksAmount, int *nonMax, int *cannyImg, int h, int w){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  for(int i = 1; i < w - 1; i = i + threadsPerBlock*blocksAmount){
    // printf("i = %d\n", i);
    for(int j = 1; j < h - 1; j = j + threadsPerBlock*blocksAmount){
      // printf("j = %d\n", j);
      if(x + i < w && y + j < h){
        if(nonMax[(x+i)*h+(y+j)] == 0){
          cannyImg[(x+i)*h+(y+j)] = 0;
        } else if(nonMax[(x+i)*h+(y+j)] == 255){
          cannyImg[(x+i)*h+(y+j)] = 255;
        } else if(nonMax[(x+i-1)*h+(y+j+1)] == 255 || nonMax[(x+i)*h+(y+j+1)] == 255 || nonMax[(x+i+1)*h+(y+j+1)] == 255 || 
          nonMax[(x+i-1)*h+(y+j)] == 255 || nonMax[(x+i+1)*h+(y+j)] == 255 || nonMax[(x+i-1)*h+(y+j-1)] == 255 || 
          nonMax[(x+i)*h+(y+j-1)] == 255 || nonMax[(x+i+1)*h+(y+j-1)] == 255){
          cannyImg[(x+i)*h+(y+j)] = 255;
        }
      }
    }
  }
}

int main(int argc, char **argv) {

  int threadsPerBlock = 32;
  int blocksAmount = 50;
  int threadCount = threadsPerBlock*threadsPerBlock*blocksAmount*blocksAmount;
  int *r, *g, *b, *a;
  int *r_gpu, *g_gpu, *b_gpu, *a_gpu, *gradient_gpu, *nonMax, *nonMax_gpu, *cannyImg, *cannyImg_gpu;
  int height = 0, width = 0;
  int size, maxGrad;
  float *grad_gpu, *theta_gpu, *theta;
  clock_t start, end;
  double time_used;

  dim3 dimBlock(threadsPerBlock, threadsPerBlock);
  dim3 dimGrid(blocksAmount, blocksAmount);
  
  CImg<unsigned char> image("images/test2.jpg");
  image.blur(2);
  height = image.height();
  width = image.width();
  
  size = sizeof(int)* width * height;
  r = (int *)malloc(size);
  g = (int *)malloc(size);
  b = (int *)malloc(size);
  a = (int *)malloc(size);  

  for(int i = 0; i < width; i++){
    for(int j = 0; j < height; j++){
      r[i*height+j] = image(i, j, 0);
      g[i*height+j] = image(i, j, 1);
      b[i*height+j] = image(i, j, 2);
    }
  }  

  printf(" %d\n",size);
  printf("w= %d, h= %d %d\n", image.width(), image.height(), sizeof(float));
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

  printf("abc %d %d", size, width*height);
  fflush(stdout);

  start = clock();

  checkCudaErrors(cudaMalloc((void **)&r_gpu, size));
  checkCudaErrors(cudaMalloc((void **)&g_gpu, size));
  checkCudaErrors(cudaMalloc((void **)&b_gpu, size));
  checkCudaErrors(cudaMalloc((void **)&a_gpu, size));

  cudaMemcpy(r_gpu, r, size, cudaMemcpyHostToDevice);
  cudaMemcpy(g_gpu, g, size, cudaMemcpyHostToDevice);
  cudaMemcpy(b_gpu, b, size, cudaMemcpyHostToDevice);

  free(r);
  free(g);
  free(b);

  colorToGrey<<<dimGrid, dimBlock>>>(threadCount, threadsPerBlock, blocksAmount, r_gpu, g_gpu, b_gpu, a_gpu, height, width);
  
  cudaFree(r_gpu);
  cudaFree(g_gpu);
  cudaFree(b_gpu);

  nonMax = (int *)malloc(size);
  theta = (float *)malloc(sizeof(float)*width*height);

  checkCudaErrors(cudaMalloc((void **)&gradient_gpu, size));
  checkCudaErrors(cudaMalloc((void **)&grad_gpu, sizeof(float)*width*height));
  checkCudaErrors(cudaMalloc((void **)&theta_gpu, sizeof(float)*width*height));
  //checkCudaErrors(cudaMalloc((void **)&nonMax, size));

  cublasHandle_t handle;
  cublasStatus_t stat;
  stat = cublasCreate(&handle);
  checkCudaErrors(stat);
  
  gradientCalc<<<dimGrid, dimBlock>>>(threadCount, threadsPerBlock, blocksAmount, a_gpu, grad_gpu, theta_gpu, height, width);
  cublasIsamax(handle, width*height, grad_gpu, 1, &maxGrad);
  // printf("\nmax grad = %d", maxGrad);
  // fflush(stdout);

  normGrad<<<dimGrid, dimBlock>>>(threadCount, threadsPerBlock, blocksAmount, grad_gpu, maxGrad, gradient_gpu, height, width);

  // cudaMemcpy(gradd, grad_gpu, sizeof(float)*width*height, cudaMemcpyDeviceToHost);  
  // cudaDeviceSynchronize();
  // for(int i = 0; i < width*height; i= i+900000){
  //   printf("\na = %f", gradd[i]);
  // }
  // printf("\na = %f", gradd[maxGrad]);
  // fflush(stdout);  

  cudaMemcpy(a, gradient_gpu, size, cudaMemcpyDeviceToHost); 
  cudaMemcpy(theta, theta_gpu, size, cudaMemcpyDeviceToHost); 

  cudaFree(a_gpu);
  cudaFree(grad_gpu);
  cudaFree(gradient_gpu);
  cudaFree(theta_gpu);

  nonMaxSupp(threadCount, threadsPerBlock, blocksAmount, a, theta, nonMax, height, width);

  free(a);
  free(theta);

  checkCudaErrors(cudaMalloc((void **)&nonMax_gpu, sizeof(float)*width*height));

  int k = 0;
  int index = 0;
  for(int i = 0; i < height*width; i++){
    if(k < nonMax[i]){
      k = nonMax[i];
      index = i;
    }
  }
  printf("abc %d\n", nonMax[index]);
  cudaMemcpy(nonMax_gpu, nonMax, size, cudaMemcpyHostToDevice);
  
  // cublasIsamax(handle, width*height, nonMax_gpu, 1, &maxImg);
  // cudaMemcpy(a, gradient_gpu, size, cudaMemcpyDeviceToHost); 
  // printf("nonmax %d = %f\n",maxImg, fnonMax[maxImg]);

  threshold<<<dimGrid, dimBlock>>>(threadCount, threadsPerBlock, blocksAmount, nonMax_gpu, index, height, width);

  cannyImg = (int *)malloc(size);
  checkCudaErrors(cudaMalloc((void **)&cannyImg_gpu, size));

  hysteresis<<<dimGrid, dimBlock>>>(threadCount, threadsPerBlock, blocksAmount, nonMax_gpu, cannyImg_gpu, height, width);

  cudaMemcpy(cannyImg, cannyImg_gpu, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(nonMax, nonMax_gpu, size, cudaMemcpyDeviceToHost);

  end = clock();
  time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("time: %fs\n", time_used);
  //free(nonMax);
  cudaFree(cannyImg_gpu);
  CImg<unsigned char> canny(width, height, 1, 1, 0);
  for(int i = 0; i < width; i++){
    for(int j = 0; j < height; j++){
      // if(nonMax[maxGrad]< nonMax[i*height+j])
      //   printf("abc %d\n", a[i*height+j]);
      canny(i,j,0,0) = cannyImg[i*height+j];
    }
  }
  free(cannyImg);

  canny.save("images/result2.bmp");
 
  cublasDestroy(handle);

  return 0;
}