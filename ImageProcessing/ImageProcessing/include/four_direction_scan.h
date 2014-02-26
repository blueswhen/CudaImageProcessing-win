// Copyright 2013-10 sxniu
#ifndef IMAGEPROCESSING_INCLUDE_FOURDIRECTIONSCAN_H_
#define IMAGEPROCESSING_INCLUDE_FOURDIRECTIONSCAN_H_

#include <cuda_runtime.h>

namespace four_direction_scan {

// four direction scan algorithm
__global__ void TopToEndScan(int* edge_image, int image_width,
                             int image_height, int search_length,
                             int reference_colour, int render_colour);
__global__ void LeftToRightScan(int* edge_image, int image_width,
                                int image_height, int search_length,
                                int reference_colour, int render_colour);
__global__ void EndToTopScan(int* edge_image, int image_width,
                             int image_height, int search_length,
                             int reference_colour, int render_colour);
__global__ void RightToLeftScan(int* edge_image, int image_width,
                                int image_height, int search_length,
                                int reference_colour, int render_colour);
// keep the filling colour and remove temp colour
__global__ void RmExtraColour(int* edge_image, int image_width);

#ifdef FOURDIRECTIONSCAN_WITH_FEEDBACK
void FourDirectionScan(int* edge_image, int image_width,
                       int image_height, int s_length, int* dev_exist);
#endif

void FourDirectionScan(int* edge_image, int image_width,
                       int image_height, int s_length,
                       int repeat_num, int fill_colour);

}  // namespace four_direction_scan

#endif  // IMAGEPROCESSING_INCLUDE_FOURDIRECTIONSCAN_H_
