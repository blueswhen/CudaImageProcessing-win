// Copyright 2013-10 sxniu
#include "include/CudaAlgorithm.h"
#include <cuda_runtime.h>
#include <windows.h>
#include <iostream>
#include "include/canny_and_edge_connection.h"
#include "include/Paremeter.h"
#include "include/CudaRes.h"
#include "include/region_filling_by_edge_tracing.h"
#include "include/four_direction_scan.h"
#include "include/cuda_utils.h"
#include "include/utils.h"
#include "include/ConstValue.h"

using namespace canny_and_edge_connection;  // NOLINT
using namespace region_filling_by_edge_tracing;  // NOLINT
using namespace four_direction_scan;  // NOLINT
using namespace cuda_utils;  // NOLINT
using namespace utils; // NOLINT

int g_block_x;
int g_block_y;
int g_grid_x;
int g_grid_y;

const size_t kMaxNum = 10485760;

void CudaAlgorithm::Main() {
  g_block_x = m_prm->m_block_X;
  g_block_y = m_prm->m_block_Y;
  g_grid_x = m_prm->m_grid_X;
  g_grid_y = m_prm->m_grid_Y;

  dim3 block(g_block_x, g_block_y);
  dim3 grid(g_grid_x, g_grid_y);
#if 0
  int* test_array = new int[kMaxNum];
  GenRanArray(test_array, kMaxNum);

  // utils::MergeSort(test_array, kMaxNum);
  // utils::Sum(test_array, kMaxNum);

  int* test_array_dev = NULL;
  if (cudaMalloc(&test_array_dev, kMaxNum * sizeof(int)) != cudaSuccess) {
    MessageBox(NULL, "cudaMallocPitch failed", NULL, NULL);
  }
  if (cudaMemcpy(test_array_dev, test_array, kMaxNum * sizeof(int),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    MessageBox(NULL, "cudaMemcpy failed", NULL, NULL);
  }

  // cuda_utils::MergeSort(test_array_dev, kMaxNum);
  cuda_utils::Sum(test_array_dev, kMaxNum);

  if (cudaMemcpy(test_array, test_array_dev, kMaxNum * sizeof(int),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    MessageBox(NULL, "cudaMemcpy back failed", NULL, NULL);
  }
  int is_sort_finished = 1;
  for(int i = 0; i < kMaxNum; ++i) {
    if (test_array[i] != i) {
      is_sort_finished = 0;
	  break;
    }
  }
  cudaFree(test_array_dev);
  delete [] test_array;
#endif
  double freq = 0;
  double start_time = 0;
  int time = 0;
  // ShowTime(&freq, &start_time, &time, 1, COUNT_TIME_ENABLE);
  Canny(m_prm, m_res);
  EdgeTracing(m_prm, m_res);
  // ShowTime(&freq, &start_time, &time, 0, COUNT_TIME_ENABLE);
#if 1  // filling algorithm
  if (m_prm->m_edge_connection_enable) {
    EdgeConnecting(m_prm, m_res);
  }
#if 1  // four direction scan
  ShowTime(&freq, &start_time, &time, 1, COUNT_TIME_ENABLE);
#if 0  // need cuda 3.5 ability
  cuda_utils::RegionFillingByEdgeTracing(m_res->m_edge_image,
                                         m_res->m_backup_image_1,
                                         m_prm->m_s_width,
                                         m_prm->m_s_height);
#endif
  if (m_prm->m_lena_segmentation_seperate_front_back) {
#if 0  // set board
    SetBoard<<<grid, block>>>(m_res->m_source_image,
                              m_res->m_edge_image,
                              m_prm->m_s_width,
                              m_prm->m_s_height,
                              BOARD_DOWN); 
#endif
    LenaFrontAndBackSceneSegmentation(m_res->m_source_image,
                                      m_res->m_edge_image,
                                      m_res->m_backup_image_1,
                                      m_res->m_backup_image_4,
                                      m_res->m_dev_exist,
                                      m_prm->m_s_width,
                                      m_prm->m_s_height);
  }
  ShowTime(&freq, &start_time, &time, 0, COUNT_TIME_ENABLE);
#else  // REN algorithm
#if 0  // set board
  SetBoard<<<grid, block>>>(m_res->m_source_image,
                            m_res->m_edge_image,
                            m_prm->m_s_width,
                            m_prm->m_s_height,
                            BOARD_DOWN);
#endif
  int* image = new int[(m_prm->m_s_width) * (m_prm->m_s_height)];
  int* result_image = new int[(m_prm->m_s_width) * (m_prm->m_s_height)];
  if (cudaMemcpy(image, m_res->m_edge_image,
                 m_prm->m_s_width * sizeof(int) * m_prm->m_s_height,
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    MessageBox(NULL, "cudaMemcpy failed", NULL, NULL);
  }
#if 0
  for (int y = 0; y < m_prm->m_s_height; ++y) {
	  for (int x = 0; x < m_prm->m_s_width; ++x) {
        int index = y * m_prm->m_s_width + x;
		if (image[index] & WHITE == WHITE) {
		  image[index] = 0;
		} else {
		  image[index] = WHITE;
		}
	  }
  }
#endif
  // if (m_prm->m_lena_segmentation_seperate_front_back) {
    // Thinning(image, m_prm->m_s_width, m_prm->m_s_height);
    // Thinning(image, m_prm->m_s_width, m_prm->m_s_height);
  // }

  ShowTime(&freq, &start_time, &time, 1, COUNT_TIME_ENABLE);
  RegionFillingByEdgeTracing(image, m_prm->m_s_width, m_prm->m_s_height, m_prm->k);
  // utils::Sobel(image, result_image, m_prm->m_s_width, m_prm->m_s_height, 40);
  // utils::Canny(image, result_image, m_prm->m_s_width, m_prm->m_s_height, m_prm->m_lowthreshold, m_prm->m_highthreshold);
  ShowTime(&freq, &start_time, &time, 0, COUNT_TIME_ENABLE);

  if (cudaMemcpy(m_res->m_edge_image, image,
                 m_prm->m_s_width * sizeof(int) * m_prm->m_s_height,
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    MessageBox(NULL, "cudaMemcpy failed", NULL, NULL);
  }
#if 0  
  RemoveBoard<<<grid, block>>>(m_res->m_source_image,
                               m_res->m_edge_image,
                               m_prm->m_s_width,
                               m_prm->m_s_height, BOARD_ALL);

  if (cudaMemcpy(image, m_res->m_edge_image,
                 m_prm->m_s_width * sizeof(int) * m_prm->m_s_height,
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    MessageBox(NULL, "cudaMemcpy failed", NULL, NULL);
  }
  RegionFillingByEdgeTracing(image, m_prm->m_s_width,
                             m_prm->m_s_height, m_prm->k);
  if (cudaMemcpy(m_res->m_edge_image, image,
                 m_prm->m_s_width * sizeof(int) * m_prm->m_s_height,
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    MessageBox(NULL, "cudaMemcpy failed", NULL, NULL);
  }
#endif
  delete [] result_image;
  delete [] image;
#endif
#if 0
  if (m_prm->m_lena_segmentation_by_board) {
    LenaSegmentationByBoard(m_res->m_source_image, m_res->m_edge_image,
                            m_res->m_backup_image_1, m_res->m_backup_image_4,
                            m_prm->m_s_width, m_prm->m_s_height);
  }


  if (m_prm->m_segmentation_enable &&
     !m_prm->m_lena_segmentation_by_board &&
     !m_prm->m_lena_segmentation_seperate_front_back) {
    if (m_prm->m_segmentation_by_board_or_by_length) {
      ImageSegmentationByBoardCombination(m_res->m_source_image,
                                          m_res->m_edge_image,
                                          m_res->m_backup_image_4,
                                          m_prm->m_s_width, m_prm->m_s_height,
                                          m_prm->m_board_combination,
                                          kColourArray);
    } else {
      ImageSegmentationByScanLength(m_res->m_source_image,
                                    m_res->m_edge_image,
                                    m_res->m_backup_image_4,
                                    m_prm->m_s_width,
                                    m_prm->m_s_height,
                                    m_prm->m_length_parameter);
    }
  }
#endif
#endif
}

void CudaAlgorithm::Canny(Paremeter* m_prm, CudaRes* m_res) {
  dim3 block(g_block_x, g_block_y);
  dim3 grid(g_grid_x, g_grid_y);
  ImageTurnGray<<<grid, block>>>(m_res->m_source_image,
                                 m_prm->m_s_width,
                                 m_prm->m_s_height);

  GaussSmoothX<<<grid, block>>>(m_res->m_source_image,
                                m_res->m_gauss_temp_array,
                                m_res->m_gaussnum_dev,
                                m_res->m_weightsum_dev,
                                m_prm->m_s_width,
                                m_prm->m_s_height, 1);

  GaussSmoothY<<<grid, block>>>(m_res->m_source_image,
                                m_res->m_gauss_image,
                                m_res->m_gauss_temp_array,
                                m_res->m_gaussnum_dev,
                                m_res->m_weightsum_dev,
                                m_prm->m_s_width,
                                m_prm->m_s_height, 1);

  GradMagnitude<<<grid, block>>>(m_res->m_gauss_image,
                                 m_res->m_gradient_x_image,
                                 m_res->m_gradient_y_image,
                                 m_res->m_magnitude_image,
                                 m_prm->m_s_width,
                                 m_prm->m_s_height);

  NonmaxSuppress<<<grid, block>>>(m_res->m_magnitude_image,
                                  m_res->m_gradient_x_image,
                                  m_res->m_gradient_y_image,
                                  m_res->m_probable_edge_image,
                                  m_prm->m_s_width,
                                  m_prm->m_s_height);

  // 200 is some random big number
  EdgeCreation<<<grid, block>>>(m_res->m_magnitude_image,
                                m_res->m_probable_edge_image,
                                m_res->m_edge_image,
                                m_prm->m_s_width,
                                m_prm->m_s_height,
                                m_prm->m_highthreshold, 200);
}

void CudaAlgorithm::EdgeTracing(Paremeter* m_prm, CudaRes* m_res) {
  dim3 block(g_block_x, g_block_y);
  dim3 grid(g_grid_x, g_grid_y);

  SearchBreakPoint<<<grid, block>>>(m_res->m_edge_image,
                                    m_prm->m_s_width,
                                    m_prm->m_s_height);

  TraceEdge<<<grid, block>>>(m_res->m_edge_image,
                             m_res->m_probable_edge_image,
                             m_res->m_magnitude_image,
                             m_prm->m_s_width,
                             m_prm->m_s_height,
                             m_prm->m_lowthreshold,
                             m_prm->m_recursion_num);

  ColourToWhite<<<grid, block>>>(m_res->m_edge_image,
                                 m_prm->m_s_width,
                                 m_prm->m_s_height);
}

void CudaAlgorithm::EdgeConnecting(Paremeter* m_prm, CudaRes* m_res) {
  dim3 block(g_block_x, g_block_y);
  dim3 grid(g_grid_x, g_grid_y);
  SearchBreakPoint<<<grid, block>>>(m_res->m_edge_image,
                                    m_prm->m_s_width,
                                    m_prm->m_s_height);

  ConnectEdge<<<grid, block>>>(m_res->m_edge_image,
                               m_res->m_magnitude_image,
                               10, 0, m_prm->m_s_width,
                               m_prm->m_s_height,
                               m_prm->m_min_l,
                               m_prm->m_del_l,
                               m_prm->m_max_l);

  ColourToWhite<<<grid, block>>>(m_res->m_edge_image,
                                 m_prm->m_s_width,
                                 m_prm->m_s_height);
#if 1
  RemoveBoard<<<grid, block>>>(m_res->m_source_image,
                               m_res->m_edge_image,
                               m_prm->m_s_width,
                               m_prm->m_s_height, BOARD_ALL);
#endif
}

void CudaAlgorithm::SetOrRemoveBoardCombination(int *source_image,
                                                int *edge_image,
                                                int image_width,
                                                int image_height,
                                                int board_combination_num,
                                                int set_or_removed) {
  dim3 block(g_block_x, g_block_y);
  dim3 grid(g_grid_x, g_grid_y);
  switch (board_combination_num) {
  case 0:
    break;
  case 1:
    if (set_or_removed) {
      SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                image_height, BOARD_DOWN);
    } else {
      RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                   image_height, BOARD_DOWN);
    }
    break;
  case 2:
    if (set_or_removed) {
      SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                image_height, BOARD_UP);
    } else {
      RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                   image_height, BOARD_UP);
    }
    break;
  case 3:
    if (set_or_removed) {
      SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                image_height, BOARD_RIGHT);
    } else {
      RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                   image_height, BOARD_RIGHT);
    }
    break;
  case 4:
    if (set_or_removed) {
      SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                image_height, BOARD_LEFT);
    } else {
      RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                   image_height, BOARD_LEFT);
    }
    break;
  case 5:
    if (set_or_removed) {
      SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                image_height, BOARD_UP);
      SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                image_height, BOARD_DOWN);
    } else {
      RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                   image_height, BOARD_UP);
      RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                   image_height, BOARD_DOWN);
    }
    break;
  case 6:
    if (set_or_removed) {
      SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                image_height, BOARD_UP);
      SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                image_height, BOARD_LEFT);
    } else {
      RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                   image_height, BOARD_UP);
      RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                   image_height, BOARD_LEFT);
    }
    break;
  case 7:
    if (set_or_removed) {
      SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                image_height, BOARD_LEFT);
      SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                image_height, BOARD_RIGHT);
    } else {
      RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                   image_height, BOARD_LEFT);
      RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                   image_height, BOARD_RIGHT);
    }
    break;
  case 8:
    if (set_or_removed) {
      SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                image_height, BOARD_DOWN);
      SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                image_height, BOARD_LEFT);
    } else {
      RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                   image_height, BOARD_DOWN);
      RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                   image_height, BOARD_LEFT);
    }
    break;
  case 9:
    if (set_or_removed) {
      SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                image_height, BOARD_DOWN);
      SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                image_height, BOARD_RIGHT);
    } else {
      RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                   image_height, BOARD_DOWN);
      RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                   image_height, BOARD_RIGHT);
    }
    break;
  case 10:
    if (set_or_removed) {
      SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                image_height, BOARD_LEFT);
      SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                image_height, BOARD_RIGHT);
    } else {
      RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                   image_height, BOARD_LEFT);
      RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                   image_height, BOARD_RIGHT);
    }
    break;
  case 11:
    if (set_or_removed) {
      SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                image_height, BOARD_UP);
      SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                image_height, BOARD_DOWN);
      SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                image_height, BOARD_LEFT);
    } else {
      RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                   image_height, BOARD_UP);
      RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                   image_height, BOARD_DOWN);
      RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                   image_height, BOARD_LEFT);
    }
    break;
  case 12:
    if (set_or_removed) {
      SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                image_height, BOARD_UP);
      SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                image_height, BOARD_DOWN);
      SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                image_height, BOARD_RIGHT);
    } else {
      RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                   image_height, BOARD_UP);
      RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                   image_height, BOARD_DOWN);
      RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                   image_height, BOARD_RIGHT);
    }
    break;
  case 13:
    if (set_or_removed) {
      SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                image_height, BOARD_UP);
      SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                image_height, BOARD_LEFT);
      SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                image_height, BOARD_RIGHT);
    } else {
      RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                   image_height, BOARD_UP);
      RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                   image_height, BOARD_LEFT);
      RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                   image_height, BOARD_RIGHT);
    }
    break;
  case 14:
    if (set_or_removed) {
      SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                image_height, BOARD_DOWN);
      SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                image_height, BOARD_LEFT);
      SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                image_height, BOARD_RIGHT);
    } else {
      RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                   image_height, BOARD_DOWN);
      RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                   image_height, BOARD_LEFT);
      RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                                   image_height, BOARD_RIGHT);
    }
    break;
  }
}

void CudaAlgorithm::ImageSegmentationByBoardCombination(
    int *source_image, int *edge_image, int *backup_image_4,
    int image_width, int image_height, int *board_combination,
    const int *kColourArray) {
  dim3 block(g_block_x, g_block_y);
  dim3 grid(g_grid_x, g_grid_y);

  CopyImage<<<grid, block>>>(edge_image, backup_image_4,
                             image_width, image_height);
  for (int j = 0; j < 15; j++) {
    if (board_combination[j]) {
      SetOrRemoveBoardCombination(source_image, edge_image,
                                  image_width, image_height, j, 1);
      FourDirectionScan(edge_image, image_width, image_height,
                        SCAN_MAX_LENGTH, SCAN_MAX_REPEAT,
                        kColourArray[j]);

      // ensure colour edge surround colour area
      SetOrRemoveColourAreaEdge<<<grid, block>>>(
          edge_image, image_width, image_height, kColourArray[j], 1, WHITE);
      SetOrRemoveBoardCombination(source_image, edge_image,
                                  image_width, image_height, j, 0);
    }
  }
  WhiteEdgeRemoved<<<grid, block>>>(edge_image, image_width, image_height);
  FillingEdgeHole<<<grid, block>>>(edge_image, image_width, image_height);
  EdgeRecovery<<<grid, block>>>(edge_image, backup_image_4,
                                image_width, image_height);
}

void CudaAlgorithm::ImageSegmentationByScanLength(int *source_image,
                                                  int *edge_image,
                                                  int *backup_image_4,
                                                  int image_width,
                                                  int image_height,
                                                  int *length_parameter) {
  dim3 block(g_block_x, g_block_y);
  dim3 grid(g_grid_x, g_grid_y);
  CopyImage<<<grid, block>>>(edge_image, backup_image_4,
                             image_width, image_height);

  SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                            image_height, BOARD_UP);
  SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                            image_height, BOARD_DOWN);
  SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                            image_height, BOARD_LEFT);
  SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                            image_height, BOARD_RIGHT);
  for (int k = 0; k < 16; k++) {
    int s_length = length_parameter[k];
    if (s_length == 0) break;
    FourDirectionScan(edge_image, image_width, image_height,
                      s_length, SCAN_MAX_REPEAT, kColourArray[k]);

    SetOrRemoveColourAreaEdge<<<grid, block>>>(edge_image, image_width,
                                               image_height, kColourArray[k],
                                               1, WHITE);  // set

    RemoveEdgeFromColourArea<<<grid, block>>>(edge_image, image_width,
                                              image_height, kColourArray[k]);
  }
  RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                               image_height, BOARD_UP);
  RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                               image_height, BOARD_DOWN);
  RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                               image_height, BOARD_LEFT);
  RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                               image_height, BOARD_RIGHT);

  WhiteEdgeRemoved<<<grid, block>>>(edge_image, image_width, image_height);
  FillingEdgeHole<<<grid, block>>>(edge_image, image_width, image_height);
  EdgeRecovery<<<grid, block>>>(edge_image, backup_image_4,
                                image_width, image_height);
}

void CudaAlgorithm::LenaFrontAndBackSceneSegmentation(
    int* source_image, int* edge_image, int* backup_image_1,
    int* backup_image_4, int* dev_exist, int image_width, int image_height) {
  dim3 block(g_block_x, g_block_y);
  dim3 grid(g_grid_x, g_grid_y);
  // CopyImage<<<grid, block>>>(edge_image, backup_image_4,
  //                            image_width, image_height);

  // SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
  //                           image_height, BOARD_DOWN);

#ifndef FOURDIRECTIONSCAN_WITH_FEEDBACK
  FourDirectionScan(edge_image, image_width, image_height,
                    SCAN_MAX_LENGTH, 4, COLOUR_RED);
#else
  FourDirectionScan(edge_image, image_width, image_height,
                    SCAN_MAX_LENGTH, dev_exist);
#endif  // FOURDIRECTIONSCAN_WITH_FEEDBACK

#if 0
  RemoveEdgeFromColourArea<<<grid, block>>>(edge_image, image_width,
                                            image_height, COLOUR_RED);
  RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                               image_height, BOARD_DOWN);

  CopyImage<<<grid, block>>>(edge_image, backup_image_1,
                             image_width, image_height);
  RemoveColour<<<grid, block>>>(edge_image, edge_image, image_width,
                                image_height, COLOUR_RED);

  FourDirectionScan(edge_image, image_width, image_height,
                    SCAN_MAX_LENGTH, SCAN_MAX_REPEAT, COLOUR_RED);
  FindCommonColourFromDifferentImage<<<grid, block>>>(
      backup_image_1, edge_image, image_width, image_height,
      COLOUR_RED, 0, COLOUR_RED, COLOUR_PURPLE);
  SetOrRemoveColourAreaEdge<<<grid, block>>>(edge_image, image_width,
                                             image_height, COLOUR_RED,
                                             1, COLOUR_RED);
  EdgeRecovery<<<grid, block>>>(edge_image, backup_image_4,
                                image_width, image_height);
#endif
}

// segmentation by board combination
void CudaAlgorithm::LenaSegmentationByBoard(int *source_image,
                                            int *edge_image,
                                            int *backup_image_1,
                                            int *backup_image_4,
                                            int image_width,
                                            int image_height) {
  dim3 block(g_block_x, g_block_y);
  dim3 grid(g_grid_x, g_grid_y);
  CopyImage<<<grid, block>>>(edge_image, backup_image_4,
                             image_width, image_height);
  // down and left board combination
  SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                            image_height, BOARD_DOWN);
  FourDirectionScan(edge_image, image_width, image_height,
                    SCAN_MAX_LENGTH, SCAN_MAX_REPEAT, COLOUR_RED);

  RemoveEdgeFromColourArea<<<grid, block>>>(edge_image, image_width,
                                            image_height, COLOUR_RED);
  RemoveBoard<<<grid, block>>>(source_image, edge_image,
                               image_width, image_height, BOARD_DOWN);

  CopyImage<<<grid, block>>>(edge_image, backup_image_1,
                             image_width, image_height);
  RemoveColour<<<grid, block>>>(edge_image, edge_image, image_width,
                                image_height, COLOUR_RED);

  SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                            image_height, BOARD_LEFT);
  FourDirectionScan(edge_image, image_width, image_height,
                    SCAN_MAX_LENGTH, SCAN_MAX_REPEAT, COLOUR_RED);

  RemoveEdgeFromColourArea<<<grid, block>>>(edge_image, image_width,
                                            image_height, COLOUR_RED);
  RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                               image_height, BOARD_LEFT);
  FindCommonColourFromDifferentImage<<<grid, block>>>(
      backup_image_1, edge_image, image_width, image_height,
      COLOUR_RED, 0, COLOUR_RED, COLOUR_YELLOW);

  // right and up board combination
  SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                            image_height, BOARD_RIGHT);
  FourDirectionScan(edge_image, image_width, image_height,
                    SCAN_MAX_LENGTH, SCAN_MAX_REPEAT, COLOUR_GREEN);

  RemoveEdgeFromColourArea<<<grid, block>>>(edge_image, image_width,
                                            image_height, COLOUR_GREEN);
  RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                               image_height, BOARD_RIGHT);

  CopyImage<<<grid, block>>>(edge_image, backup_image_1,
                             image_width, image_height);
  RemoveColour<<<grid, block>>>(edge_image, edge_image, image_width,
                                image_height, COLOUR_GREEN);

  SetBoard<<<grid, block>>>(source_image, edge_image,
                            image_width, image_height, BOARD_UP);
  FourDirectionScan(edge_image, image_width, image_height,
                    SCAN_MAX_LENGTH, SCAN_MAX_REPEAT, COLOUR_GREEN);

  RemoveEdgeFromColourArea<<<grid, block>>>(edge_image, image_width,
                                            image_height, COLOUR_GREEN);
  RemoveBoard<<<grid, block>>>(source_image, edge_image,
                               image_width, image_height, BOARD_UP);

  FindCommonColourFromDifferentImage<<<grid, block>>>(
      backup_image_1, edge_image, image_width, image_height,
      COLOUR_GREEN, 0, COLOUR_PURPLE, COLOUR_CYAN);

  // down left and down right board combination
  SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                            image_height, BOARD_DOWN);
  SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                            image_height, BOARD_LEFT);
  FourDirectionScan(edge_image, image_width, image_height,
                    SCAN_MAX_LENGTH, SCAN_MAX_REPEAT, COLOUR_GREEN);

  RemoveEdgeFromColourArea<<<grid, block>>>(edge_image, image_width,
                                            image_height, COLOUR_GREEN);
  RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                               image_height, BOARD_DOWN);
  RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                               image_height, BOARD_LEFT);

  CopyImage<<<grid, block>>>(edge_image, backup_image_1,
                             image_width, image_height);
  RemoveColour<<<grid, block>>>(edge_image, edge_image, image_width,
                                image_height, COLOUR_GREEN);

  SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                            image_height, BOARD_DOWN);
  SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                            image_height, BOARD_RIGHT);
  FourDirectionScan(edge_image, image_width, image_height,
                    SCAN_MAX_LENGTH, SCAN_MAX_REPEAT, COLOUR_GREEN);

  RemoveEdgeFromColourArea<<<grid, block>>>(edge_image, image_width,
                                            image_height, COLOUR_GREEN);
  RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                               image_height, BOARD_DOWN);
  RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                               image_height, BOARD_RIGHT);

  FindCommonColourFromDifferentImage<<<grid, block>>>(
      backup_image_1, edge_image, image_width, image_height,
      COLOUR_GREEN, 0, COLOUR_BLUE, COLOUR_LIGHT_GRAY);

  // up down left board combination
  SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                            image_height, BOARD_UP);
  SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                            image_height, BOARD_DOWN);
  SetBoard<<<grid, block>>>(source_image, edge_image, image_width,
                            image_height, BOARD_LEFT);
  FourDirectionScan(edge_image, image_width, image_height,
                    SCAN_MAX_LENGTH, SCAN_MAX_REPEAT, COLOUR_GREEN);

  RemoveEdgeFromColourArea<<<grid, block>>>(edge_image, image_width,
                                            image_height, COLOUR_GREEN);
  RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                               image_height, BOARD_UP);
  RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                               image_height, BOARD_DOWN);
  RemoveBoard<<<grid, block>>>(source_image, edge_image, image_width,
                               image_height, BOARD_LEFT);

  WhiteEdgeRemoved<<<grid, block>>>(edge_image, image_width, image_height);
  FillingEdgeHole<<<grid, block>>>(edge_image, image_width, image_height);

  EdgeRecovery<<<grid, block>>>(edge_image, backup_image_4,
                                image_width, image_height);
}
