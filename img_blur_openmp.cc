#include <chrono>
#include <iostream>
#include <math.h>
#include <string>

#include <omp.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const int blur_radius = 2;
const int blur_diameter = 5;
const double blur_matrix[5][5] = {{0.04, 0.04, 0.04, 0.04, 0.04},
                                  {0.04, 0.04, 0.04, 0.04, 0.04},
                                  {0.04, 0.04, 0.04, 0.04, 0.04},
                                  {0.04, 0.04, 0.04, 0.04, 0.04},
                                  {0.04, 0.04, 0.04, 0.04, 0.04}};

void blur(Mat *original, Mat *copy, int width, int height) {
  unsigned char *original_image = original->ptr();
  unsigned char *copy_image = copy->ptr();
  if (width > blur_diameter && height > blur_diameter) {
#pragma omp parallel for
    for (int i = blur_radius; i < height - blur_radius; i++) {
      for (int j = blur_radius; j < width - blur_radius; j++) {
        int index = copy->step * i + copy->channels() * j;
        copy_image[index] = 0.0;
        copy_image[index + 1] = 0.0;
        copy_image[index + 2] = 0.0;
        for (int k = 0; k < blur_radius * 2 + 1; k++) {
          for (int l = 0; l < blur_radius * 2 + 1; l++) {
            int neighbor_index = copy->step * (i + k - blur_radius) +
                                 copy->channels() * (j + l - blur_radius);
            copy_image[index] +=
                blur_matrix[k][l] * original_image[neighbor_index];
            copy_image[index + 1] +=
                blur_matrix[k][l] * original_image[neighbor_index + 1];
            copy_image[index + 2] +=
                blur_matrix[k][l] * original_image[neighbor_index + 2];
          }
        }
      }
    }
  }
}

int main(int argc, char **argv) {
  string path;

  if (argc == 1) {
    cerr << "You must pass an image path to the program" << endl;
    return -1;
  } else {
    path = argv[1];
  }

  int t = 4;

  if (argc > 2) {
    t = atoi(argv[2]);
  }

  omp_set_num_threads(t);

  Mat original = imread(path, IMREAD_COLOR);

  Mat copy = original.clone();

  auto start_calc = chrono::high_resolution_clock::now();

  blur(&original, &copy, original.cols, original.rows);

  auto end_calc = chrono::high_resolution_clock::now();

  chrono::duration<float, std::milli> duration_calc = end_calc - start_calc;

  cout << "Blurring image took " << duration_calc.count() << "ms" << endl;

  imwrite("blur_" + path, copy);

  imshow("ORIGINAL", original);
  imshow("BLURRED", copy);

  waitKey();

  return 0;
}