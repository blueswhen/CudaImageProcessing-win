// Copyright 2013-10 sxniu
#ifndef IMAGEPROCESSING_INCLUDE_CANNYANDEDGECONNECTION_H_
#define IMAGEPROCESSING_INCLUDE_CANNYANDEDGECONNECTION_H_

#include <cuda_runtime.h>

namespace canny_and_edge_connection {

// canny
__global__ void ImageTurnGray(int* source_image, int image_width,
                              int image_height);
__global__ void GaussSmoothX(int* source_image, float* gauss_temp_array,
                             float *gaussnum, float *weightsum,
                             int image_width, int image_height,
                             int half_window_size);
__global__ void GaussSmoothY(int* source_image, int* gauss_image,
                             float* gauss_temp_array, float *gaussnum,
                             float *weightsum, int image_width,
                             int image_height, int half_window_size);
__global__ void GradMagnitude(int *gauss_image, float *gradient_x_image,
                              float *gradient_y_image, int *magnitude_image,
                              int image_width, int image_height);
__global__ void NonmaxSuppress(int *magnitude_image,
                               float *gradient_x_image,
                               float *gradient_y_image,
                               int *probable_edge_image,
                               int image_width, int image_height);
__global__ void EdgeCreation(int *magnitude_image, int* probable_edge_image,
                             int *edge_image, int image_width,
                             int image_height, int edge_min, int edge_max);

// edge connection
__global__ void SearchBreakPoint(int* edge_image, int image_width,
                                 int image_height);
__global__ void TraceEdge(int* edge_image,
                          int* probable_edge_image,
                          int* magnitude_image,
                          int image_width,
                          int image_height,
                          int edge_min_trace,
                          int recursion_num);
__global__ void ColourToWhite(int* edge_image, int image_width,
                              int image_height);
__global__ void LonelyPointRemoved(int* edge_image, int image_width,
                                   int image_height);
__global__ void ConnectEdge(int* edge_image, int* magnitude_image,
                            int magnitude_num, int magnitude_enable,
                            int image_width, int image_height,
                            int min_length, int step, int max_length);

}  // namespace canny_and_edge_connection

#endif  // IMAGEPROCESSING_INCLUDE_CANNYANDEDGECONNECTION_H_