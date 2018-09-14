#include "common.h"
#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

__constant__ int blur_radius = 2;
__constant__ int blur_diameter = 5;
__constant__ double blur_matrix[5][5] = {{0.04, 0.04, 0.04, 0.04, 0.04},
                                  {0.04, 0.04, 0.04, 0.04, 0.04},
                                  {0.04, 0.04, 0.04, 0.04, 0.04},
                                  {0.04, 0.04, 0.04, 0.04, 0.04},
                                  {0.04, 0.04, 0.04, 0.04, 0.04}};

__global__ void blur(unsigned char *original_image, unsigned char *copy_image, int width, int height, int step, int channels) {
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;

  int index = step * i + channels * j;
  copy_image[index] = 0.0;
  copy_image[index + 1] = 0.0;
  copy_image[index + 2] = 0.0;
  for (int k = 0; k < blur_radius * 2 + 1; k++) {
    for (int l = 0; l < blur_radius * 2 + 1; l++) {
      int neighbor_index = step * (i + k - blur_radius) +
                            channels * (j + l - blur_radius);
      copy_image[index] +=
          blur_matrix[k][l] * original_image[neighbor_index];
      copy_image[index + 1] +=
          blur_matrix[k][l] * original_image[neighbor_index + 1];
      copy_image[index + 2] +=
          blur_matrix[k][l] * original_image[neighbor_index + 2];
    }
  }
}

void image_blur(const cv::Mat& input, cv::Mat& output){


	size_t colorBytes = input.step * input.rows;
	size_t blurredBytes = output.step * output.rows;

	unsigned char *d_input, *d_output;

	// Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, colorBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, blurredBytes), "CUDA Malloc Failed");

	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), colorBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

  
	const dim3 block(64, 64);

	const dim3 grid((int)ceil((float)input.rows / block.x), (int)ceil((float)input.cols/ block.y));
	printf("image_blur_kernel<<<(%d, %d) , (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);

  auto start_cpu = chrono::high_resolution_clock::now();
	blur <<<grid, block >>>(d_input, d_output, input.cols, input.rows, static_cast<int>(input.step), input.channels());
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");
  auto end_cpu = chrono::high_resolution_clock::now();

	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, blurredBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
  SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");
  
  chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

  cout << "blur <<<(" << grid.x << ", " << grid.y << "), (" << block.x
       << ", " << block.y << ")>>> elapsed " << duration_ms.count()
       << "ms." << endl;
}

int main(int argc, char *argv[]){
  // Get the image path
	string imagePath;
  (argc < 2) ? imagePath = "image.jpg" : imagePath = argv[1];

	Mat input = imread(imagePath, CV_LOAD_IMAGE_COLOR);

	if (input.empty()){
		cout << "Image Not Found!" << std::endl;
		cin.get();
		return -1;
	}

	Mat output = input.clone();

	image_blur(input, output);
  cv::imwrite("output.jpg", output);

	return 0;
}
