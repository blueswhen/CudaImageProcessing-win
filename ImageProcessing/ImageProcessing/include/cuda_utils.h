// Copyright 2013-10 sxniu
#ifndef IMAGEPROCESSING_INCLUDE_CUDAUTILS_H_
#define IMAGEPROCESSING_INCLUDE_CUDAUTILS_H_

#include <cuda_runtime.h>
#include "include/ConstValue.h"

namespace cuda_utils {

// if set_or_removed = 1, set, =0, removed
// ColourAreaEdge means the edge of colour area
__global__ void SetOrRemoveColourAreaEdge(int* edge_image, int image_width,
                                          int image_height, int area_colour,
                                          int set_or_removed,
                                          int new_edge_colour);
// remove edge in colour area
// turn the white edge in colour area to the same colour with the area
__global__ void RemoveEdgeFromColourArea(int* edge_image, int image_width,
                                         int image_height, int area_colour);
// remove some colour in image
__global__ void RemoveColour(int* edge_image_src,
                             int* edge_image_dst,
                             int image_width,
                             int image_height,
                             int remove_colour);
// turn the white edge around the colour edge to the colour edge
__global__ void FindArroundPointsFromColourEdge(
    int* edge_image, int close_colour, int image_width, int image_height);
__global__ void ChangeBreakPointColour(int* edge_image, int breakpoint_colour,
                                       int image_width, int image_height);
__global__ void SetBoard(int* source_image, int* edge_image,
                         int image_width, int image_height,
                         BoardType direction);
__global__ void RemoveBoard(int* source_image, int* edge_image,
                            int image_width, int image_height,
                            BoardType direction);
__global__ void FindCommonColourFromDifferentImage(
    int* edge_image_src, int* edge_image_dst,
    int image_width, int image_height, int reference_colour,
    int common_colour, int source_only_colour, int destination_only_colour);
__global__ void CopyImage(int* image_src, int* image_dst,
                          int image_width, int image_height);
__global__ void EdgeRecovery(int* image_src, int* edge_image,
                             int image_width, int image_height);
__global__ void WhiteEdgeRemoved(int* image_src, int image_width,
                                 int image_height);
// filling the hole of edge with the area colour after the edge removed
__global__ void FillingEdgeHole(int* image_src, int image_width,
                                int image_height);
// find the edge of the colour area and use edge_colour to draw it
__global__ void FindColourAreaEdge(int* edge_image, int area_colour,
                                   int edge_colour, int image_width,
                                   int image_height);
void MergeSort(int* test_array_dev, size_t size);
// left_pos is the max left position of array
// right_pos is the max right position of array
__global__ void MergeSort(int* test_array_dev, int* tmp_array_dev,
                          size_t left_pos, size_t right_pos);
void Sum(int* test_array_dev, size_t size);
void RegionFillingByEdgeTracing(int* edge_image, int* backup_image, int image_width, int image_height);
}  // namespace cuda_utils

#endif  // IMAGEPROCESSING_INCLUDE_CUDAUTILS_H_
