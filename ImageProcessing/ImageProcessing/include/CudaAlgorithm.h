// Copyright 2013-10 sxniu
#ifndef IMAGEPROCESSING_INCLUDE_CUDAALGORITHM_H_
#define IMAGEPROCESSING_INCLUDE_CUDAALGORITHM_H_

#include "include/ConstValue.h"

class Paremeter;
class CudaRes;

class CudaAlgorithm {
 public:
  CudaAlgorithm(Paremeter* prm, CudaRes* res)
    : m_prm(prm)
    , m_res(res) {}
  // the function to use all alogrithms
  void Main();

 private:
  void Canny(Paremeter* prm, CudaRes* res);
  void EdgeTracing(Paremeter* prm, CudaRes* res);
  void EdgeConnecting(Paremeter* prm, CudaRes* res);
  void SetOrRemoveBoardCombination(int* source_image, int* edge_image,
                                   int image_width, int image_height,
                                   int board_combination_num,
                                   int set_or_removed);
  void ImageSegmentationByBoardCombination(int* source_image, int* edge_image,
                                           int* backup_image_4,
                                           int image_width, int image_height,
                                           int* board_combination,
                                           const int* kColourArray);
  void ImageSegmentationByScanLength(int* source_image, int* edge_image,
                                     int* backup_image_4,
                                     int image_width, int image_height,
                                     int* length_parameter);
  void LenaFrontAndBackSceneSegmentation(int* source_image, int* edge_image,
                                         int* backup_image_1,
                                         int* backup_image_4, int* dev_exist,
                                         int image_width, int image_height);
  void LenaSegmentationByBoard(int* source_image, int* edge_image,
                               int* backup_image_1, int* backup_image_4,
                               int image_width, int image_height);

 private:
  Paremeter* m_prm;
  CudaRes* m_res;
};
#endif  // IMAGEPROCESSING_INCLUDE_CUDAALGORITHM_H_
