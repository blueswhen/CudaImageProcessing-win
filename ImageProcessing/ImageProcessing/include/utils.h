// Copyright 2013-10 sxniu
#ifndef IMAGEPROCESSING_INCLUDE_UTILS_H_
#define IMAGEPROCESSING_INCLUDE_UTILS_H_

namespace utils {

// counting time
void ShowTime(double *freq, double *start_time,
              int *time, int start_or_end, int count_enable);
// show number in dialogue box
void ShowNum(int num);
// thinning alogrithm
void Thinning(int* image, int image_width, int image_height);
size_t BigRand();
size_t GenRanNumFromRange(size_t begin, size_t end);
void GenRanArray(int* test_array, size_t size);
void MergeSort(int* test_array, size_t size);
// left_pos is the max left position of array
// right_pos is the max right position of array
void MergeSort(int* test_array, int* tmp_array, size_t left_pos, size_t right_pos);
void Sum(int* test_array, size_t size);
void Sobel(int* source_image, int* edge_image, int image_width,
		   int image_height, int threshold);
void Canny(int* source_image, int* edge_image, int image_width,
           int image_height, int low_threshold, int height_threshold);
}  // namespace utils

#endif  // IMAGEPROCESSING_INCLUDE_UTILS_H_