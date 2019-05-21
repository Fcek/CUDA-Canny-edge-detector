
#rm main.cu.o
#rm cgs_lab 
#/usr/local/cuda-6.0/bin/nvcc -ccbin g++ -I/usr/local/cuda/samples/common/inc  -m64     -o main.cu.o -c main.cu
#/usr/local/cuda-6.0/bin/nvcc -ccbin g++ -m64 -o cgs_lab main.cu.o  -lcublas -lcusparse

rm canny.cu.o
rm canny_project 
/usr/local/cuda-10.1/bin/nvcc -ccbin g++ -I/usr/local/cuda/samples/common/inc  -m64      -o canny.cu.o -c canny.cu
/usr/local/cuda-10.1/bin/nvcc -ccbin g++ -m64 -o canny_project canny.cu.o  -lcublas -lcusparse -lm -lpthread -lX11 



