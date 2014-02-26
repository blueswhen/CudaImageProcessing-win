// Copyright 2013-10 sxniu
#ifndef IMAGEPROCESSING_INCLUDE_PAREMETER_H_
#define IMAGEPROCESSING_INCLUDE_PAREMETER_H_
#include "include/ConstValue.h"

class CudaRes;

class Paremeter {
 public:
  friend class D3DInit;
  friend class Cuda3DInit;
  friend class CudaAlgorithm;
  Paremeter();
  void ReadIniFile();
  void Kplus() {
    k += 1;
  }

 private:
  // gauss filter array, choose [-3*kSigma, 3*kSigma] data.
  float m_gaussnum[1 + 2 * THREE_SIGMA_INT];
  // the lengh of data
  int m_windowsize;
  // gauss coefficient sum
  float m_weightsum;

  // save 7 kinds of resolution in program to choose, use 0~6 number
  int m_resolution_choose;
  // screen or image width
  int m_s_width;
  // screen or image height
  int m_s_height;
  int m_block_X;
  int m_block_Y;
  int m_grid_X;
  int m_grid_Y;

  // if 1, execute canny
  int m_canny_enable;
  // canny coefficient
  int m_highthreshold;
  int m_lowthreshold;
  // the number of recursion, >= 0
  int m_recursion_num;

  // if 1, execute edge connection
  int m_edge_connection_enable;
  // the minimum half length of the searching box board
  int m_min_l;
  // the maximum half length of the searching box board
  int m_max_l;
  // the increment of the half length of the searching box board
  int m_del_l;

  // separate lena image by different board combination
  int m_lena_segmentation_by_board;
  // separate lena image to front and back scene
  int m_lena_segmentation_seperate_front_back;

  // if 1, execute four direction scan
  int m_segmentation_enable;
  int m_segmentation_by_board_or_by_length;

  // there are 16 board combinations, choose 1 to use this combination
  int m_board_combination[16];

  // the length of scan in four direction scan alogrithm
  // use different length to segment image
  int m_length_parameter[16];
  int m_exist;

  // the Paremeter of dynamic effect in window show
  int k;
};
#endif  // IMAGEPROCESSING_INCLUDE_PAREMETER_H_
