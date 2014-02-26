// Copyright 2013-10 sxniu
#include "include/canny_and_edge_connection.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <string>
#include <algorithm>
#include "include/Paremeter.h"
#include "include/CudaRes.h"
#include "include/ConstValue.h"
#include "include/utils.h"

namespace canny_and_edge_connection {

__global__ void ImageTurnGray(int* source_image, int image_width,
                              int image_height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * image_width + x;

  float red, green, blue;
  red = (source_image[index] & COLOUR_RED) >> 16;
  green = (source_image[index] & COLOUR_GREEN) >> 8;
  blue = source_image[index] & COLOUR_BLUE;
  int gray;
  gray = static_cast<int>(red * 0.3 + green * 0.59 + blue * 0.11);
  gray = (gray << 16) + (gray << 8) + gray;
  source_image[index] = gray;
}

__global__ void GaussSmoothX(int* source_image, float* gauss_temp_array,
                             float* gaussnum, float* weightsum,
                             int image_width, int image_height,
                             int half_window_size) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * image_width + x;

  if (x > 0 && x < image_width - 1 && y > 0 && y < image_height - 1) {
    float d_dot_mul = 0;
    *weightsum = 0;

    for (int i = -half_window_size; i <= half_window_size; i++) {
      *weightsum += gaussnum[half_window_size + i];
      int gray = source_image[y * image_width + (i + x)] & COLOUR_BLUE;
      d_dot_mul += static_cast<float>(gray) * gaussnum[half_window_size + i];
    }
    gauss_temp_array[index] = d_dot_mul / *weightsum;
  }
}

__global__ void GaussSmoothY(int* source_image, int* gauss_image,
                             float* gauss_temp_array, float* gaussnum,
                             float* weightsum, int image_width,
                             int image_height, int half_window_size) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * image_width + x;

  if (x > 0 && x < image_width - 1 && y > 0 && y < image_height - 1) {
    float d_dot_mul = 0;

    for (int i = -half_window_size; i <= half_window_size; i++) {
      d_dot_mul += gauss_temp_array[(y + i) * image_width + x] *
                   gaussnum[half_window_size + i];
    }
    int des_gray = static_cast<int>(d_dot_mul / *weightsum) & COLOUR_BLUE;
    des_gray = (des_gray << 16) + (des_gray << 8) + des_gray;
    gauss_image[index] = des_gray;
  }
}

__global__ void GradMagnitude(int* gauss_image,
                              float* gradient_x_image,
                              float* gradient_y_image,
                              int* magnitude_image,
                              int image_width, int image_height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * image_width + x;

  if (x > 0 && x < image_width - 1 && y > 0 && y < image_height - 1) {
    // calculate gradient on x direction
    int gray_x1 = gauss_image[y * image_width+ min(image_width - 1, x + 1)] &
                  COLOUR_BLUE;
    int gray_x2 = gauss_image[y * image_width + max(0, x - 1)] & COLOUR_BLUE;
    gradient_x_image[index] = static_cast<float>(gray_x1 - gray_x2);

    // calculate gradient on y direction
    int gray_y1 = gauss_image[min(image_height - 1, y + 1) * image_width + x] &
                  COLOUR_BLUE;
    int gray_y2 = gauss_image[max(0, y - 1) * image_width+ x ] & COLOUR_BLUE;
    gradient_y_image[index] = static_cast<float>(gray_y1 - gray_y2);

    magnitude_image[index] = static_cast<int>(sqrt(gradient_x_image[index] *
                                              gradient_x_image[index] +
                                              gradient_y_image[index] *
                                              gradient_y_image[index]) + 0.5);
  } else {
    magnitude_image[index] = 0;
  }
}

__global__ void NonmaxSuppress(int* magnitude_image,
                               float* gradient_x_image,
                               float* gradient_y_image,
                               int* probable_edge_image,
                               int image_width,
                               int image_height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * image_width + x;

  // gradient component on x direction
  float gx;
  float gy;

  // temp value
  int g1, g2, g3, g4;
  float weight;
  float tmp1;
  float tmp2;
  float tmp3;

  if (magnitude_image[index] == 0) {
    probable_edge_image[index] = 0;
  } else {
    tmp3 = static_cast<float>(magnitude_image[index]);

    gx = gradient_x_image[index];
    gy = gradient_y_image[index];

    if (fabs(gy) > fabs(gx)) {
      weight = fabs(gx) / fabs(gy);
      g2 = magnitude_image[index - image_width];
      g4 = magnitude_image[index + image_width];

      if (gx * gy > 0) {
        g1 = magnitude_image[index - image_width - 1];
        g3 = magnitude_image[index + image_width + 1];
      } else {
        g1 = magnitude_image[index - image_width + 1];
        g3 = magnitude_image[index + image_width - 1];
      }
    } else {
      weight = fabs(gy) / fabs(gx);

      g2 = magnitude_image[index + 1];
      g4 = magnitude_image[index - 1];

      if (gx * gy > 0) {
        g1 = magnitude_image[index + image_width + 1];
        g3 = magnitude_image[index - image_width - 1];
      } else {
        g1 = magnitude_image[index - image_width + 1];
        g3 = magnitude_image[index + image_width - 1];
      }
    }

    tmp1 = weight * static_cast<float>(g1) +
           (1 - weight) * static_cast<float>(g2);
    tmp2 = weight * static_cast<float>(g3) +
           (1 - weight) * static_cast<float>(g4);

    if (tmp3 >= tmp1 && tmp3 >= tmp2) {
      probable_edge_image[index] = COLOUR_BLUE;
    } else {
      probable_edge_image[index] = 0;
    }
  }
}

__global__ void EdgeCreation(int* magnitude_image, int* probable_edge_image,
                             int* edge_image, int image_width,
                             int image_height, int edge_min, int edge_max) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * image_width + x;

  if ((probable_edge_image[index] != 0) &&
     (magnitude_image[index] >= edge_min) &&
     (magnitude_image[index] <= edge_max)) {
    edge_image[index] = WHITE;
  } else {
    edge_image[index] = 0;
  }
}

__global__ void SearchBreakPoint(int* edge_image, int image_width,
                                 int image_height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index_cen = y * image_width + x;
  int index[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int sum = 0;
  index[0] = max(y - 1, 0) * image_width + max(x - 1, 0);
  index[1] = max(y - 1, 0) * image_width + min(x + 1, image_width);
  index[2] = min(y + 1, image_height) * image_width + min(x + 1, image_width);
  index[3] = min(y + 1, image_height) * image_width + max(x - 1, 0);
  index[4] = max(y - 1, 0) * image_width + x;
  index[5] = y * image_width + max(x - 1, 0);
  index[6] = y * image_width + min(x + 1, image_width);
  index[7] = min(y + 1, image_height) * image_width + x;

  int index_arround[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  if (x > 0 && x < image_width - 1 && y > 0 && y < image_height - 1) {
    if (edge_image[index_cen] == WHITE) {
      index_arround[0] = max(y - 2, 0) * image_width + max(x - 1, 0);
      index_arround[1] = max(y - 2, 0) * image_width + x;
      index_arround[2] = max(y - 2, 0) * image_width + min(x + 1, image_width);
      index_arround[3] = max(y - 1, 0) * image_width + min(x + 2, image_width);
      index_arround[4] = y * image_width + min(x + 2, image_width);
      index_arround[5] = min(y + 1, image_height) * image_width +
                         min(x + 2, image_width);
      index_arround[6] = min(y + 2, image_height) * image_width + max(x - 1, 0);
      index_arround[7] = min(y + 2, image_height) * image_width + x;
      index_arround[8] = min(y + 2, image_height) * image_width +
                         min(x + 1, image_width);
      index_arround[9] = max(y - 1, 0) * image_width + max(x - 2, 0);
      index_arround[10] = y * image_width + max(x - 2, 0);
      index_arround[11] = min(y + 1, image_height) * image_width +
                          max(x - 2, 0);

      for (int i = 0; i < 8; i++) {
        if (edge_image[index[i]] != 0) {
          sum++;
        }
      }
      if (sum == 3) {
        if (edge_image[index[5]] +
            edge_image[index[0]] +
            edge_image[index[4]] == 0x02fffffd) {
          if (edge_image[index_arround[9]] +
              edge_image[index_arround[10]] +
              edge_image[index_arround[11]] == 0) {
            edge_image[index[5]] = 0x0C000000 + index_cen;
          } else if (edge_image[index_arround[0]] +
                     edge_image[index_arround[1]] +
                     edge_image[index_arround[2]] == 0) {
            edge_image[index[4]] = 0x0C000000 + index_cen;
          }
        } else if (edge_image[index[4]] +
                   edge_image[index[1]] +
                   edge_image[index[6]] == 0x02fffffd) {
          if (edge_image[index_arround[3]] +
              edge_image[index_arround[4]] +
              edge_image[index_arround[5]] == 0) {
            edge_image[index_cen] = 0x0C000000 + index_cen;
          } else if (edge_image[index_arround[0]] +
                     edge_image[index_arround[1]] +
                     edge_image[index_arround[2]] == 0) {
            edge_image[index[4]] = 0x0C000000 + index_cen;
          }
        } else if (edge_image[index[6]] +
                   edge_image[index[2]] +
                   edge_image[index[7]] == 0x02fffffd) {
          if (edge_image[index_arround[3]] +
              edge_image[index_arround[4]] +
              edge_image[index_arround[5]] == 0) {
            edge_image[index_cen] = 0x0C000000 + index_cen;
          } else if (edge_image[index_arround[6]] +
                     edge_image[index_arround[7]] +
                     edge_image[index_arround[8]] == 0) {
            edge_image[index_cen] = 0x0C000000 + index_cen;
          }
        } else if (edge_image[index[7]] +
                   edge_image[index[3]] +
                   edge_image[index[5]] == 0x02fffffd) {
          if (edge_image[index_arround[6]] +
              edge_image[index_arround[7]] +
              edge_image[index_arround[8]] == 0) {
            edge_image[index_cen] = 0x0C000000 + index_cen;
          } else if (edge_image[index_arround[9]] +
                     edge_image[index_arround[10]] +
                     edge_image[index_arround[11]] == 0) {
            edge_image[index[5]] = 0x0C000000 + index_cen;
          }
        }
      } else if (sum == 1) {
        if (edge_image[index[0]] == WHITE) {
          if ((edge_image[index_arround[0]] +
               edge_image[index_arround[1]] == 0) ||
              (edge_image[index_arround[9]] +
               edge_image[index_arround[10]] == 0)) {
            edge_image[index_cen] = 0x0A000000 + index_cen;
          }
        } else if (edge_image[index[1]] == WHITE) {
          if ((edge_image[index_arround[2]] +
               edge_image[index_arround[1]] == 0) ||
              (edge_image[index_arround[3]] +
               edge_image[index_arround[4]] == 0)) {
            edge_image[index_cen] = 0x0A000000 + index_cen;
          }
        } else if (edge_image[index[2]] == WHITE) {
          if ((edge_image[index_arround[5]] +
               edge_image[index_arround[4]] == 0) ||
              (edge_image[index_arround[8]] +
               edge_image[index_arround[7]] == 0)) {
            edge_image[index_cen] = 0x0A000000 + index_cen;
          }
        } else if (edge_image[index[3]] == WHITE) {
          if ((edge_image[index_arround[6]] +
               edge_image[index_arround[7]] == 0) ||
              (edge_image[index_arround[11]] +
               edge_image[index_arround[10]] == 0)) {
            edge_image[index_cen] = 0x0A000000 + index_cen;
          }
        } else {
          edge_image[index_cen] = 0x0A000000 + index_cen;
        }
      } else if (sum == 2) {
        if (edge_image[index[0]]+edge_image[index[4]] == 0x01fffffe) {
          if ((edge_image[index_arround[1]] +
               edge_image[index_arround[2]] == 0) ||
              (edge_image[index_arround[9]] +
               edge_image[index_arround[10]] == 0)) {
            edge_image[index_cen] = 0x0B000000 + index_cen;
          }
        } else if (edge_image[index[4]]+edge_image[index[1]] == 0x01fffffe) {
          if ((edge_image[index_arround[0]] +
               edge_image[index_arround[1]] == 0) ||
              (edge_image[index_arround[3]] +
               edge_image[index_arround[4]] == 0)) {
            edge_image[index_cen] = 0x0B000000 + index_cen;
          }
        } else if (edge_image[index[1]]+edge_image[index[6]] == 0x01fffffe) {
          if ((edge_image[index_arround[4]] +
               edge_image[index_arround[5]] == 0) ||
              (edge_image[index_arround[1]] +
               edge_image[index_arround[2]] == 0)) {
            edge_image[index_cen] = 0x0B000000 + index_cen;
          }
        } else if (edge_image[index[6]]+edge_image[index[2]] == 0x01fffffe) {
          if ((edge_image[index_arround[3]] +
               edge_image[index_arround[4]] == 0) ||
              (edge_image[index_arround[7]] +
               edge_image[index_arround[8]] == 0)) {
            edge_image[index_cen] = 0x0B000000 + index_cen;
          }
        } else if (edge_image[index[2]] + edge_image[index[7]] == 0x01fffffe) {
          if ((edge_image[index_arround[7]] +
               edge_image[index_arround[8]] == 0) ||
              (edge_image[index_arround[4]] +
               edge_image[index_arround[5]] == 0)) {
            edge_image[index_cen] = 0x0B000000 + index_cen;
          }
        } else if (edge_image[index[7]]+edge_image[index[3]] == 0x01fffffe) {
          if ((edge_image[index_arround[7]] +
               edge_image[index_arround[8]] == 0) ||
              (edge_image[index_arround[10]] +
               edge_image[index_arround[11]] == 0)) {
             edge_image[index_cen] = 0x0B000000 + index_cen;
          }
        } else if (edge_image[index[3]] + edge_image[index[5]] == 0x01fffffe) {
          if ((edge_image[index_arround[9]] +
               edge_image[index_arround[10]] == 0) ||
              (edge_image[index_arround[6]] +
               edge_image[index_arround[7]] == 0)) {
            edge_image[index_cen] = 0x0B000000 + index_cen;
          }
        } else if (edge_image[index[5]] + edge_image[index[0]] == 0x01fffffe) {
          if ((edge_image[index_arround[10]] +
               edge_image[index_arround[11]] == 0) ||
              (edge_image[index_arround[0]] +
               edge_image[index_arround[1]] == 0)) {
            edge_image[index_cen] = 0x0B000000 + index_cen;
          }
        }
      } else if (sum == 0) {
        edge_image[index_cen] = 0;
      }
    }
  }
}

__device__ void BreakPointCreation(int* edge_image, int x, int y,
                                   int* point_pos, int* pos_num,
                                   int* delta_index_cen,
                                   int* breakpoint_position,
                                   int* breakpoint_finded,
                                   int image_width, int image_height) {
  int index_cen = y * image_width + x;
  point_pos[0] = index_cen;
  *pos_num = 1;
  *breakpoint_position = 0;

  int index_arr[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  index_arr[0] = max(y - 1, 0) * image_width + max(x - 1, 0);
  index_arr[1] = min(y + 1, image_height) * image_width + max(x - 1, 0);
  index_arr[2] = max(y - 1, 0) * image_width + min(x + 1, image_width);
  index_arr[3] = min(y + 1, image_height) * image_width +
                 min(x + 1, image_width);
  index_arr[4] = max(y - 1, 0) * image_width + x;
  index_arr[5] = y * image_width + max(x - 1, 0);
  index_arr[6] = y * image_width + min(x + 1, image_width);
  index_arr[7] = min(y + 1, image_height) * image_width + x;

  switch (edge_image[index_cen] & 0xff000000) {
    case 0x0A000000:
    case 0x0B000000:
    case 0x0C000000:
      // plus 0xf0000000 for distinguish the real breakpoint
      edge_image[index_cen] = edge_image[index_cen] + 0xf0000000;
      *breakpoint_position = edge_image[index_cen];
      for (int j = 7; j >= 0; j--) {
        if (edge_image[index_arr[j]] != 0) {
          point_pos[*pos_num] = index_arr[j];
          (*pos_num)++;
          *delta_index_cen = index_arr[j] - index_cen;
        }
      }
      *breakpoint_finded = true;
      break;
    default:
      *breakpoint_finded = false;
  }
}


__device__ void SetBreakPointSearchOrder(int* index, int index_cen,
                                         int delta_index_cen,
                                         int image_width) {
  if (delta_index_cen == image_width - 1) {
    index[0] = index_cen + (-image_width + 1);
    index[1] = index_cen + 1;
    index[2] = index_cen - image_width;
    index[3] = index_cen + image_width + 1;
    index[4] = index_cen + (-image_width - 1);
    index[5] = index_cen + image_width;
    index[6] = index_cen - 1;
  } else if (delta_index_cen == image_width) {
    index[0] = index_cen - image_width;
    index[1] = index_cen + (-image_width + 1);
    index[2] = index_cen + (-image_width - 1);
    index[3] = index_cen + 1;
    index[4] = index_cen - 1;
    index[5] = index_cen + image_width + 1;
    index[6] = index_cen + image_width - 1;
  } else if (delta_index_cen == image_width + 1) {
    index[0] = index_cen + (-image_width - 1);
    index[1] = index_cen - image_width;
    index[2] = index_cen - 1;
    index[3] = index_cen + (-image_width + 1);
    index[4] = index_cen + image_width - 1;
    index[5] = index_cen + 1;
    index[6] = index_cen + image_width;
  } else if (delta_index_cen == -1) {
    index[0] = index_cen + 1;
    index[1] = index_cen + image_width + 1;
    index[2] = index_cen + (-image_width + 1);
    index[3] = index_cen + image_width;
    index[4] = index_cen - image_width;
    index[5] = index_cen + image_width - 1;
    index[6] = index_cen + (-image_width - 1);
  } else if (delta_index_cen == 1) {
    index[0] = index_cen - 1;
    index[1] = index_cen + image_width - 1;
    index[2] = index_cen + (-image_width - 1);
    index[3] = index_cen + image_width;
    index[4] = index_cen - image_width;
    index[5] = index_cen + image_width + 1;
    index[6] = index_cen + (-image_width + 1);
  } else if (delta_index_cen == -image_width - 1) {
    index[0] = index_cen + image_width + 1;
    index[1] = index_cen + image_width;
    index[2] = index_cen + 1;
    index[3] = index_cen + image_width - 1;
    index[4] = index_cen + (-image_width + 1);
    index[5] = index_cen - 1;
    index[6] = index_cen - image_width;
  } else if (delta_index_cen == -image_width) {
    index[0] = index_cen + image_width;
    index[1] = index_cen + image_width + 1;
    index[2] = index_cen + image_width - 1;
    index[3] = index_cen + 1;
    index[4] = index_cen - 1;
    index[5] = index_cen + (-image_width + 1);
    index[6] = index_cen + (-image_width - 1);
  } else if (delta_index_cen == 1 - image_width) {
    index[0] = index_cen + image_width - 1;
    index[1] = index_cen + image_width;
    index[2] = index_cen - 1;
    index[3] = index_cen + image_width + 1;
    index[4] = index_cen + (-image_width - 1);
    index[5] = index_cen + 1;
    index[6] = index_cen - image_width;
  } else {
    index[0] = index_cen + image_width - 1;
    index[1] = index_cen + image_width;
    index[2] = index_cen - 1;
    index[3] = index_cen + image_width + 1;
    index[4] = index_cen + (-image_width - 1);
    index[5] = index_cen + 1;
    index[6] = index_cen - image_width;
  }
}

__device__ void DoTraceEdge(int delta_index_cen, int index_cen,
                            int* probable_edge_image, int* edge_image,
                            int* magnitude_image, int* del_index_cen_next,
                            int* index_cen_next, int* finish_trace,
                            int breakpoint_position, int* finish_no_first,
                            int image_width, int image_height,
                            int edge_min_trace, int* recursion_times,
                            int* first_recursion_num,
                            int* recursion_coordinate,
                            int* del_recursion_coordinate,
                            int* recursion_coordinate_num,
                            int recursion_num) {
  // if recursion_num == 0, no recursion will happen
  int index[7] = {0, 0, 0, 0, 0, 0, 0};

  SetBreakPointSearchOrder(index, index_cen, delta_index_cen, image_width);

  int enable = true;

  int n = 0;
  if (edge_min_trace == 0) {
    int max_magnitude_index = index[0];
    int max_magnitude = magnitude_image[max_magnitude_index];

    int index_push[2] = {index[0], index[1]};
    int k = 0;

    for (int i = 0; i <= 2; i++) {
      if (edge_image[index[i]] != 0 &&
          edge_image[index[i]] != breakpoint_position &&
          enable == true) {    // meet other edge or tracing edge
        *finish_trace = true;
        enable = false;
        break;
      }
      if (edge_image[index[i]] == 0 &&
         probable_edge_image[index[i]] != 0) {
        if (enable == true) {
          max_magnitude_index = index[i];
          max_magnitude = magnitude_image[max_magnitude_index];
          enable = false;
          *recursion_times = 0;
        }
        if (magnitude_image[index[i]] > max_magnitude) {
          max_magnitude_index = index[i];
          max_magnitude = magnitude_image[max_magnitude_index];
        } else if (recursion_num && k != 2) {
          index_push[k] = index[i];
          k++;
        }
      }
    }
    if (enable == false) {
      *index_cen_next = max_magnitude_index;
      *del_index_cen_next = index_cen - max_magnitude_index;
      edge_image[max_magnitude_index] = breakpoint_position;

      if (recursion_num) {
        if (k >= 1) {
          if (*first_recursion_num != recursion_num)
            (*first_recursion_num)++;
          recursion_coordinate[*recursion_coordinate_num] = index_push[0];
          del_recursion_coordinate[*recursion_coordinate_num] = index_cen -
                                                                index_push[0];
          if (*recursion_coordinate_num == recursion_num - 1)
            *recursion_coordinate_num = 0;
          else
            (*recursion_coordinate_num)++;
        }
        if (recursion_num != 1 && k == 2) {
          if (*first_recursion_num != recursion_num)
            (*first_recursion_num)++;
          recursion_coordinate[*recursion_coordinate_num] = index_push[1];
          del_recursion_coordinate[*recursion_coordinate_num] = index_cen -
                                                                index_push[1];
          if ((*recursion_coordinate_num) == recursion_num - 1)
            (*recursion_coordinate_num) = 0;
          else
            (*recursion_coordinate_num)++;
        }
      }
    }
  } else {
    for (int i = 0; i <= 6; i++) {
      if (edge_image[index[i]] != 0 &&
          edge_image[index[i]] != breakpoint_position &&
          enable == true) {    // meet other edge or tracing edge
        *finish_trace = true;
        enable = false;
        break;
      }
      if (edge_image[index[i]] == 0 &&
         probable_edge_image[index[i]] != 0 &&
         magnitude_image[index[i]] > edge_min_trace) {
        if (n == 0) {
          *index_cen_next = index[i];
          *del_index_cen_next = index_cen - index[i];
          edge_image[index[i]] = breakpoint_position;
          enable = false;
          n++;
          if (!recursion_num) break;
        } else if (recursion_num) {
          if (n != 3 && n != recursion_num + 1 &&
             probable_edge_image[index[i]] != COLOUR_RED) {
            if (*first_recursion_num != recursion_num)
              (*first_recursion_num)++;
            n++;
            probable_edge_image[index[i]] = COLOUR_RED;
            recursion_coordinate[*recursion_coordinate_num] = index[i];
            del_recursion_coordinate[*recursion_coordinate_num] = index_cen -
                                                                  index[i];
            if ((*recursion_coordinate_num) == recursion_num - 1)
              (*recursion_coordinate_num) = 0;
            else
              (*recursion_coordinate_num)++;
          } else {
              break;
          }
        }
      }
    }
  }
  if (enable == true) {
    *finish_trace = true;
  }

  // continue tracing "num_again" number points after edge-tracing
  if (recursion_num && *recursion_times != *first_recursion_num &&
     *finish_trace == true && enable == true) {
    if (*recursion_coordinate_num == 0)
      *recursion_coordinate_num = recursion_num - 1;
    else
      (*recursion_coordinate_num)--;

    int bp = breakpoint_position;
    edge_image[recursion_coordinate[*recursion_coordinate_num]] = bp;
    *index_cen_next = recursion_coordinate[*recursion_coordinate_num];
    *del_index_cen_next = del_recursion_coordinate[*recursion_coordinate_num];

    *finish_trace = false;
    (*recursion_times)++;
  }
}

__global__ void TraceEdge(int* edge_image, int* probable_edge_image,
                          int* magnitude_image,
                          int image_width, int image_height,
                          int edge_min_trace,
                          int recursion_num) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index_cen = y * image_width + x;

  int delta_index_cen = 0;
  int breakpoint_position = 0;
  int del_index_cen_next = 0;
  int index_cen_next = 0;
  int finish_trace = false;
  int trace_en = false;
  int point_pos[4] = {0, 0, 0, 0};
  int pos_num = 0;
  int finish_no_first = false;

  if (x - 1 > 0 && x + 2 < image_width && y - 1 > 0 && y + 2 < image_height) {
    BreakPointCreation(edge_image, x, y,
                       point_pos, &pos_num,
                       &delta_index_cen,
                       &breakpoint_position,
                       &trace_en,
                       image_width, image_height);

    if (trace_en) {
      int recursion_times = 0;
      int first_recursion_num = 0;
      int recursion_coordinate_num = 0;
      int recursion_coordinate[20];
      int del_recursion_coordinate[20];

      for (int i = 0; i < 5000; i++) {
        DoTraceEdge(delta_index_cen, index_cen,
                    probable_edge_image, edge_image,
                    magnitude_image,
                    &del_index_cen_next,
                    &index_cen_next,
                    &finish_trace,
                    breakpoint_position,
                    &finish_no_first,
                    image_width, image_height,
                    edge_min_trace,
                    &recursion_times,
                    &first_recursion_num,
                    recursion_coordinate,
                    del_recursion_coordinate,
                    &recursion_coordinate_num,
                    recursion_num);

        delta_index_cen = del_index_cen_next;
        index_cen = index_cen_next;

        if (finish_trace == true) {
          break;
        }
      }
    }
  }
}

__global__ void ColourToWhite(int* edge_image, int image_width,
                              int image_height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * image_width + x;

  if ((edge_image[index] & WHITE) != 0 && edge_image[index] != WHITE) {
    edge_image[index] = WHITE;
  } else if ((edge_image[index] & WHITE) == 0) {
    edge_image[index] = 0;
  }
}

__global__ void LonelyPointRemoved(int* edge_image, int image_width,
                                   int image_height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index_cen = y * image_width + x;

  int index_arround[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  index_arround[0] = max(y - 1, 0) * image_width + max(x - 1, 0);
  index_arround[1] = min(y + 1, image_height) * image_width + max(x - 1, 0);
  index_arround[2] = max(y - 1, 0) * image_width + min(x + 1, image_width);
  index_arround[3] = min(y + 1, image_height) * image_width +
                     min(x + 1, image_width);
  index_arround[4] = max(y - 1, 0) * image_width + x;
  index_arround[5] = y * image_width + max(x - 1, 0);
  index_arround[6] = y * image_width + min(x + 1, image_width);
  index_arround[7] = min(y + 1, image_height) * image_width + x;

  int cln = true;
  if (edge_image[index_cen] == WHITE) {
    for (int i = 0; i < 8; i++) {
      if (edge_image[index_arround[i]] != 0) {
        cln = false;
        break;
      }
    }
    if (cln) edge_image[index_cen] = 0;
  }
}

__device__ void SetPointApproximatePosition(int mir_x, int mir_y,
                                            int* position, int serch_length) {
  if (mir_x < 0 && mir_y == -serch_length) {
    // up1
    *position = 0;
  } else if (mir_x >= 0 && mir_x < serch_length && mir_y == -serch_length) {
    // up2
    *position = 1;
  } else if (mir_x == serch_length && mir_y < 0) {
    // right1
    *position = 2;
  } else if (mir_x == serch_length && mir_y >= 0 && mir_y < serch_length) {
    // right2
    *position = 3;
  } else if (mir_x > 0 && mir_y == serch_length) {
    // down2
    *position = 4;
  } else if (mir_x <= 0 && mir_x > -serch_length && mir_y == serch_length) {
    // down1
    *position = 5;
  } else if (mir_x == -serch_length && mir_y > 0) {
    // left2
    *position = 6;
  } else {
    // left1
    *position = 7;
  }
}

__device__ void SetConnectedPointSearchOrder(int x, int y,
                                             int* index,
                                             int* ind_x, int* ind_y,
                                             int mir_position,
                                             int image_width, int image_height,
                                             int serch_length) {
  switch (mir_position) {
  case 0:
    for (int i = 0; i < serch_length; i++) {
      // up1
      index[i] = max(y - serch_length, 0) * image_width +
                 max(x - serch_length + i, 0);
      ind_x[i] = -serch_length + i;
      ind_y[i] = -serch_length;
      // l1
      index[i + serch_length] = max(y - serch_length + i + 1, 0) * image_width +
                                max(x - serch_length, 0);
      ind_x[i + serch_length] = -serch_length;
      ind_y[i + serch_length] = -serch_length + i + 1;
      // up2
      index[i + 2 * serch_length] = max(y - serch_length, 0) * image_width +
                                    min(x + i, image_width);
      ind_x[i + 2 * serch_length] = i;
      ind_y[i + 2 * serch_length] = -serch_length;
      // l2
      index[i + 3 * serch_length] = min(y + 1 + i, image_height) * image_width +
                                    max(x - serch_length, 0);
      ind_x[i + 3 * serch_length] = -serch_length;
      ind_y[i + 3 * serch_length] = 1 + i;
      // r1
      index[i + 4 * serch_length] = max(y - serch_length + i, 0) * image_width +
                                    min(x + serch_length, image_width);
      ind_x[i + 4 * serch_length] = serch_length;
      ind_y[i + 4 * serch_length] = -serch_length + i;
      // d1
      index[i + 5 * serch_length] = min(y + serch_length, image_height) *
                                    image_width +
                                    max(x - serch_length + i + 1, 0);
      ind_x[i + 5 * serch_length] = -serch_length + i + 1;
      ind_y[i + 5 * serch_length] = serch_length;
      // r2
      index[i + 6 * serch_length] = min(y + i, image_height) * image_width +
                                    min(x + serch_length, image_width);
      ind_x[i + 6 * serch_length] = serch_length;
      ind_y[i + 6 * serch_length] = i;
      // d2
      index[i + 7 * serch_length] = min(y + serch_length, image_height) *
                                    image_width +
                                    min(x + 1+i, image_width);
      ind_x[i + 7 * serch_length] = 1 + i;
      ind_y[i + 7 * serch_length] = serch_length;
    }
    break;
  case 1:
    for (int i = 0; i < serch_length; i++) {
      // up2
      index[i] = max(y - serch_length, 0) * image_width +
                 min(x + serch_length - 1 - i, image_width);
      ind_x[i] = serch_length - 1-i;
      ind_y[i] = -serch_length;
      // r1
      index[i + serch_length] = max(y - serch_length + i, 0) * image_width +
                                min(x + serch_length, image_width);
      ind_x[i + serch_length] = serch_length;
      ind_y[i + serch_length] = -serch_length + i;
      // up1
      index[i + 2 * serch_length] = max(y - serch_length, 0) * image_width +
                                    max(x - i - 1, 0);
      ind_x[i + 2 * serch_length] = -i - 1;
      ind_y[i + 2 * serch_length] = -serch_length;
      // r2
      index[i + 3 * serch_length] = min(y + i, image_height) * image_width +
                                    min(x + serch_length, image_width);
      ind_x[i + 3 * serch_length] = serch_length;
      ind_y[i + 3 * serch_length] = i;
      // l1
      index[i + 4 * serch_length] = max(y - serch_length + 1 + i, 0) *
                                    image_width +
                                    max(x - serch_length, 0);
      ind_x[i + 4 * serch_length] = -serch_length;
      ind_y[i + 4 * serch_length] = -serch_length + 1 + i;
      // d2
      index[i + 5 * serch_length] = min(y + serch_length, image_height) *
                                    image_width +
                                    min(x + serch_length - i, image_width);
      ind_x[i + 5 * serch_length] = serch_length - i;
      ind_y[i + 5 * serch_length] = serch_length;
      // l2
      index[i + 6 * serch_length] = min(y + i, image_height) * image_width +
                                    max(x - serch_length, 0);
      ind_x[i + 6 * serch_length] = -serch_length;
      ind_y[i + 6 * serch_length] = i;
      // d1
      index[i + 7 * serch_length] = min(y + serch_length, image_height) *
                                    image_width +
                                    max(x - i, 0);
      ind_x[i + 7 * serch_length] = -i;
      ind_y[i + 7 * serch_length] = serch_length;
    }
    break;
  case 7:
    for (int i = 0; i < serch_length; i++) {
      // l1
      index[i] = max(y - serch_length + 1 + i, 0) * image_width +
                 max(x - serch_length, 0);
      ind_x[i] = -serch_length;
      ind_y[i] = -serch_length + 1+i;
      // up1
      index[i + serch_length] = max(y - serch_length, 0) * image_width +
                                max(x - serch_length + i, 0);
      ind_x[i + serch_length] = -serch_length + i;
      ind_y[i + serch_length] = -serch_length;
      // l2
      index[i + 2 * serch_length] = min(y + 1+i, image_height) * image_width +
                                    max(x - serch_length, 0);
      ind_x[i + 2 * serch_length] = -serch_length;
      ind_y[i + 2 * serch_length] = 1 + i;
      // up2
      index[i + 3 * serch_length] = max(y - serch_length, 0) * image_width +
                                    min(x + i, image_width);
      ind_x[i + 3 * serch_length] = i;
      ind_y[i + 3 * serch_length] = -serch_length;
      // d1
      index[i + 4 * serch_length] = min(y + serch_length, image_height) *
                                    image_width +
                                    max(x - serch_length + 1 + i, 0);
      ind_x[i + 4 * serch_length] = -serch_length + 1 + i;
      ind_y[i + 4 * serch_length] = serch_length;
      // r1
      index[i + 5 * serch_length] = max(y - serch_length + i, 0) * image_width +
                                    min(x + serch_length, image_width);
      ind_x[i + 5 * serch_length] = serch_length;
      ind_y[i + 5 * serch_length] = -serch_length + i;
      // d2
      index[i + 6 * serch_length] = min(y + serch_length, image_height) *
                                    image_width +
                                    min(x + 1+i, image_width);
      ind_x[i + 6 * serch_length] = 1 + i;
      ind_y[i + 6 * serch_length] = serch_length;
      // r2
      index[i + 7 * serch_length] = min(y + i, image_height) * image_width +
                                    min(x + serch_length, image_width);
      ind_x[i + 7 * serch_length] = serch_length;
      ind_y[i + 7 * serch_length] = i;
    }
    break;
  case 6:
    for (int i = 0; i < serch_length; i++) {
      // l2
      index[i] = min(y + serch_length - i, image_height) * image_width +
                 max(x - serch_length, 0);
      ind_x[i] = -serch_length;
      ind_y[i] = serch_length - i;
      // d1
      index[i + serch_length] = min(y + serch_length, image_height) *
                                image_width +
                                max(x - serch_length + 1 + i, 0);
      ind_x[i + serch_length] = -serch_length + 1 + i;
      ind_y[i + serch_length] = serch_length;
      // l1
      index[i + 2 * serch_length] = max(y - i, 0) * image_width +
                                    max(x - serch_length, 0);
      ind_x[i + 2 * serch_length] = -serch_length;
      ind_y[i + 2 * serch_length] = -i;
      // d2
      index[i + 3 * serch_length] = min(y + serch_length, image_height) *
                                    image_width +
                                    min(x + 1+i, image_width);
      ind_x[i + 3 * serch_length] = 1 + i;
      ind_y[i + 3 * serch_length] = serch_length;
      // up1
      index[i + 4 * serch_length] = max(y - serch_length, 0) * image_width +
                                    max(x - serch_length + i, 0);
      ind_x[i + 4 * serch_length] = -serch_length + i;
      ind_y[i + 4 * serch_length] = -serch_length;
      // r2
      index[i + 5 * serch_length] = min(y + serch_length - 1 - i,
                                        image_height) *
                                    image_width +
                                    min(x + serch_length, image_width);
      ind_x[i + 5 * serch_length] = serch_length;
      ind_y[i + 5 * serch_length] = serch_length - 1 - i;
      // up2
      index[i + 6 * serch_length] = max(y - serch_length, 0) * image_width +
                                    min(x + i, image_width);
      ind_x[i + 6 * serch_length] = i;
      ind_y[i + 6 * serch_length] = -serch_length;
      // r1
      index[i + 7 * serch_length] = max(y - 1 - i, 0) * image_width +
                                    min(x + serch_length, image_width);
      ind_x[i + 7 * serch_length] = serch_length;
      ind_y[i + 7 * serch_length] = -1 - i;
    }
    break;
  case 2:
    for (int i = 0; i < serch_length; i++) {
      // r1
      index[i] = max(y - serch_length + i, 0) * image_width +
                 min(x + serch_length, image_width);
      ind_x[i] = serch_length;
      ind_y[i] = -serch_length + i;
      // up2
      index[i + serch_length] = max(y - serch_length, 0) * image_width +
                                min(x + serch_length - 1 - i, image_width);
      ind_x[i + serch_length] = serch_length - 1-i;
      ind_y[i + serch_length] = -serch_length;
      // r2
      index[i + 2 * serch_length] = min(y + i, image_height) * image_width +
                                    min(x + serch_length, image_width);
      ind_x[i + 2 * serch_length] = serch_length;
      ind_y[i + 2 * serch_length] = i;
      // up1
      index[i + 3 * serch_length] = max(y - serch_length, 0) * image_width +
                                    max(x - 1-i, 0);
      ind_x[i + 3 * serch_length] = -1 - i;
      ind_y[i + 3 * serch_length] = -serch_length;
      // d2
      index[i + 4 * serch_length] = min(y + serch_length, image_height) *
                                    image_width +
                                    min(x + serch_length - i, image_width);
      ind_x[i + 4 * serch_length] = serch_length - i;
      ind_y[i + 4 * serch_length] = serch_length;
      // l1
      index[i + 5 * serch_length] = max(y - serch_length + 1 + i, 0) *
                                    image_width +
                                    max(x - serch_length, 0);
      ind_x[i + 5 * serch_length] = -serch_length;
      ind_y[i + 5 * serch_length] = -serch_length + 1 + i;
      // d1
      index[i + 6 * serch_length] = min(y + serch_length, image_height) *
                                    image_width +
                                    max(x - i, 0);
      ind_x[i + 6 * serch_length] = -i;
      ind_y[i + 6 * serch_length] = serch_length;
      // l2
      index[i + 7 * serch_length] = min(y + 1 + i, image_height) * image_width +
                                    max(x - serch_length, 0);
      ind_x[i + 7 * serch_length] = -serch_length;
      ind_y[i + 7 * serch_length] = 1 + i;
    }
    break;
  case 3:
    for (int i = 0; i < serch_length; i++) {
      // r2
      index[i] = min(y + serch_length - 1 - i, image_height) * image_width +
                 min(x + serch_length, image_width);
      ind_x[i] = serch_length;
      ind_y[i] = serch_length - 1 - i;
      // d2
      index[i + serch_length] = min(y + serch_length, image_height) *
                                image_width +
                                min(x + serch_length - i, image_width);
      ind_x[i + serch_length] = serch_length - i;
      ind_y[i + serch_length] = serch_length;
      // r1
      index[i + 2 * serch_length] = max(y - 1 - i, 0) * image_width +
                                    min(x + serch_length, image_width);
      ind_x[i + 2 * serch_length] = serch_length;
      ind_y[i + 2 * serch_length] = -1 - i;
      // d1
      index[i + 3 * serch_length] = min(y + serch_length, image_height) *
                                    image_width +
                                    max(x - i, 0);
      ind_x[i + 3 * serch_length] = -i;
      ind_y[i + 3 * serch_length] = serch_length;
      // up2
      index[i + 4 * serch_length] = max(y - serch_length, 0) * image_width +
                                    min(x + serch_length - 1 - i, image_width);
      ind_x[i + 4 * serch_length] = serch_length - 1 - i;
      ind_y[i + 4 * serch_length] = -serch_length;
      // l2
      index[i + 5 * serch_length] = min(y + serch_length - i, image_height) *
                                    image_width +
                                    max(x - serch_length, 0);
      ind_x[i + 5 * serch_length] = -serch_length;
      ind_y[i + 5 * serch_length] = serch_length - i;
      // up1
      index[i + 6 * serch_length] = max(y - serch_length, 0) * image_width +
                                    max(x - 1 - i, 0);
      ind_x[i + 6 * serch_length] = -1 - i;
      ind_y[i + 6 * serch_length] = -serch_length;
      // l1
      index[i + 7 * serch_length] = max(y - i, 0) * image_width +
                                    max(x - serch_length, 0);
      ind_x[i + 7 * serch_length] = -serch_length;
      ind_y[i + 7 * serch_length] = -i;
    }
    break;
  case 5:
    for (int i = 0; i < serch_length; i++) {
      // d1
      index[i] = min(y + serch_length, image_height) * image_width +
                 max(x - serch_length + 1 + i, 0);
      ind_x[i] = -serch_length + 1 + i;
      ind_y[i] = serch_length;
      // l2
      index[i + serch_length] = min(y + serch_length - i, image_height) *
                                image_width +
                                max(x - serch_length, 0);
      ind_x[i + serch_length] = -serch_length;
      ind_y[i + serch_length] = serch_length - i;
      // d2
      index[i + 2 * serch_length] = min(y + serch_length, image_height) *
                                    image_width +
                                    min(x + 1 + i, image_width);
      ind_x[i + 2 * serch_length] = 1 + i;
      ind_y[i + 2 * serch_length] = serch_length;
      // l1
      index[i + 3 * serch_length] = max(y - i, 0) * image_width +
                                    max(x - serch_length, 0);
      ind_x[i + 3 * serch_length] = -serch_length;
      ind_y[i + 3 * serch_length] = -i;
      // r2
      index[i + 4 * serch_length] = min(y + serch_length - 1 - i,
                                        image_height) *
                                    image_width +
                                    min(x + serch_length, image_width);
      ind_x[i + 4 * serch_length] = serch_length;
      ind_y[i + 4 * serch_length] = serch_length - 1-i;
      // up1
      index[i + 5 * serch_length] = max(y - serch_length, 0) * image_width +
                                    max(x - serch_length + i, 0);
      ind_x[i + 5 * serch_length] = -serch_length + i;
      ind_y[i + 5 * serch_length] = -serch_length;
      // r1
      index[i + 6 * serch_length] = max(y - 1-i, 0) * image_width +
                                    min(x + serch_length, image_width);
      ind_x[i + 6 * serch_length] = serch_length;
      ind_y[i + 6 * serch_length] = -1 - i;
      // up2
      index[i + 7 * serch_length] = max(y - serch_length, 0) * image_width +
                                    min(x + i, image_width);
      ind_x[i + 7 * serch_length] = i;
      ind_y[i + 7 * serch_length] = -serch_length;
    }
    break;
  case 4:
    for (int i = 0; i < serch_length; i++) {
      // d2
      index[i] = min(y + serch_length, image_height) * image_width +
                 min(x + serch_length - i, image_width);
      ind_x[i] = serch_length - i;
      ind_y[i] = serch_length;
      // r2
      index[i + serch_length] = min(y + serch_length - 1 - i, image_height) *
                                image_width +
                                min(x + serch_length, image_width);
      ind_x[i + serch_length] = serch_length;
      ind_y[i + serch_length] = serch_length - 1 - i;
      // d1
      index[i + 2 * serch_length] = min(y + serch_length, image_height) *
                                    image_width +
                                    max(x - i, 0);
      ind_x[i + 2 * serch_length] = -i;
      ind_y[i + 2 * serch_length] = serch_length;
      // r1
      index[i + 3 * serch_length] = max(y - 1 - i, 0) * image_width +
                                    min(x + serch_length, image_width);
      ind_x[i + 3 * serch_length] = serch_length;
      ind_y[i + 3 * serch_length] = -1 - i;
      // l2
      index[i + 4 * serch_length] = min(y + serch_length - i, image_height) *
                                    image_width +
                                    max(x - serch_length, 0);
      ind_x[i + 4 * serch_length] = -serch_length;
      ind_y[i + 4 * serch_length] = serch_length - i;
      // up2
      index[i + 5 * serch_length] = max(y - serch_length, 0) * image_width +
                                    min(x + serch_length - 1 - i, image_width);
      ind_x[i + 5 * serch_length] = serch_length - 1-i;
      ind_y[i + 5 * serch_length] = -serch_length;
      // l1
      index[i + 6 * serch_length] = max(y - i, 0) * image_width +
                                    max(x - serch_length, 0);
      ind_x[i + 6 * serch_length] = -serch_length;
      ind_y[i + 6 * serch_length] = -i;
      // up1
      index[i + 7 * serch_length] = max(y - serch_length, 0) * image_width +
                                    max(x - 1 - i, 0);
      ind_x[i + 7 * serch_length] = -1 - i;
      ind_y[i + 7 * serch_length] = -serch_length;
    }
  }
}

__device__ void ConnectPoint(int* edge_image,
                             int search_position,
                             int index_cen, int target_point,
                             int target_point_x, int target_point_y,
                             int breakpoint_position,
                             int image_width,
                             int serch_length,
                             int* magnitude_image,
                             int magnitude_num,
                             int magnitude_enable) {
  int index;
  int not_eight_points = false;
  // deal with 8 points in 8 special place
  if (target_point == ((-serch_length) * image_width - serch_length)) {
    // upper left
    for (int i = 1; i < serch_length; i++) {
      index = index_cen + (-i) * image_width - i;
      edge_image[index] = breakpoint_position;
      if (magnitude_enable) magnitude_image[index] = magnitude_num;
    }
  } else if (target_point == ((-serch_length) * image_width)) {
    // up
    for (int i = 1; i < serch_length; i++) {
      index = index_cen + (-i) * image_width;
      edge_image[index] = breakpoint_position;
      if (magnitude_enable) magnitude_image[index] = magnitude_num;
    }
  } else if (target_point == ((-serch_length) * image_width + serch_length)) {
    // upper right
    for (int i = 1; i < serch_length; i++) {
      index = index_cen + (-i) * image_width + i;
      edge_image[index] = breakpoint_position;
      if (magnitude_enable) magnitude_image[index] = magnitude_num;
    }
  } else if (target_point == (-serch_length)) {
    // left
    for (int i = 1; i < serch_length; i++) {
      index = index_cen - i;
      edge_image[index] = breakpoint_position;
      if (magnitude_enable) magnitude_image[index] = magnitude_num;
    }
  } else if (target_point == (serch_length)) {
    // right
    for (int i = 1; i < serch_length; i++) {
      index = index_cen + i;
      edge_image[index] = breakpoint_position;
      if (magnitude_enable) magnitude_image[index] = magnitude_num;
    }
  } else if (target_point == (serch_length * image_width - serch_length)) {
    // bottom left
    for (int i = 1; i < serch_length; i++) {
      index = index_cen + i * image_width - i;
      edge_image[index] = breakpoint_position;
      if (magnitude_enable) magnitude_image[index] = magnitude_num;
    }
  } else if (target_point == (serch_length * image_width)) {
    // bottom
    for (int i = 1; i < serch_length; i++) {
      index = index_cen + i * image_width;
      edge_image[index] = breakpoint_position;
      if (magnitude_enable) magnitude_image[index] = magnitude_num;
    }
  } else if (target_point == (serch_length + serch_length * image_width)) {
    // bottom right
    for (int i = 1; i < serch_length; i++) {
      index = index_cen + i * image_width + i;
      edge_image[index] = breakpoint_position;
      if (magnitude_enable) magnitude_image[index] = magnitude_num;
    }
  } else {
    not_eight_points = true;
  }

  int index1 = index_cen;
  int index2 = index_cen;
  int index3 = index_cen;

  if (not_eight_points) {
    switch (search_position) {
    case 0:
      for (int i = 1; i < serch_length; i++) {
        if ((2 * i) <= -target_point_y && i <= -target_point_x) {
          // turn up two points then turn left one points to go
          index1 = index3 - image_width;
          index2 = index3 - 2 * image_width;
          index3 = index_cen - 2 * i * image_width - i;
          edge_image[index1] = breakpoint_position;
          edge_image[index2] = breakpoint_position;
          edge_image[index3] = breakpoint_position;
          if (magnitude_enable) {
            magnitude_image[index1] = magnitude_num;
            magnitude_image[index2] = magnitude_num;
            magnitude_image[index3] = magnitude_num;
          }
        } else {   // point by point tracing when target closed
          for (int j = 1; j < -target_point_y - 2 * (i - 1); j++) {
            index3 = index3 - image_width;
            edge_image[index3] = breakpoint_position;
            if (magnitude_enable) magnitude_image[index3] = magnitude_num;
          }
          for (int j = 1; j < -target_point_x - (i - 1); j++) {
            index3 = index3 - 1;
            edge_image[index3] = breakpoint_position;
            if (magnitude_enable) magnitude_image[index3] = magnitude_num;
          }
          break;
        }
      }
      break;
    case 1:
      for (int i = 1; i < serch_length; i++) {
        if ((2 * i) <= -target_point_y && i <= target_point_x) {
          // turn up two points then turn right one point to go
          index1 = index3 - image_width;
          index2 = index3 - 2 * image_width;
          index3 = index_cen - 2 * i * image_width + i;
          edge_image[index1] = breakpoint_position;
          edge_image[index2] = breakpoint_position;
          edge_image[index3] = breakpoint_position;
          if (magnitude_enable) {
            magnitude_image[index1] = magnitude_num;
            magnitude_image[index2] = magnitude_num;
            magnitude_image[index3] = magnitude_num;
          }
        } else {   // point by point tracing when target closed
          for (int j = 1; j < -target_point_y - 2 * (i - 1); j++) {
            index3 = index3 - image_width;
            edge_image[index3] = breakpoint_position;
            if (magnitude_enable) magnitude_image[index3] = magnitude_num;
          }
          for (int j = 1; j < target_point_x - (i - 1); j++) {
            index3 = index3 + 1;
            edge_image[index3] = breakpoint_position;
            if (magnitude_enable) magnitude_image[index3] = magnitude_num;
          }
          break;
        }
      }
      break;
    case 2:
      for (int i = 1; i < serch_length; i++) {
        if (i <= -target_point_y && 2 * i <= target_point_x) {
          // turn right two points then turn up one point to go
          index1 = index3 + 1;
          index2 = index3 + 2;
          index3 = index_cen - i * image_width + 2 * i;
          edge_image[index1] = breakpoint_position;
          edge_image[index2] = breakpoint_position;
          edge_image[index3] = breakpoint_position;
          if (magnitude_enable) {
            magnitude_image[index1] = magnitude_num;
            magnitude_image[index2] = magnitude_num;
            magnitude_image[index3] = magnitude_num;
          }
        } else {    // point by point tracing when target closed
          for (int j = 1; j < -target_point_y - (i - 1); j++) {
            index3 = index3 - image_width;
            edge_image[index3] = breakpoint_position;
            if (magnitude_enable) magnitude_image[index3] = magnitude_num;
          }
          for (int j = 1; j < target_point_x - 2 * (i - 1); j++) {
            index3 = index3 + 1;
            edge_image[index3] = breakpoint_position;
            if (magnitude_enable) magnitude_image[index3] = magnitude_num;
          }
          break;
        }
      }
      break;
    case 3:
      for (int i = 1; i < serch_length; i++) {
        if (i <= target_point_y && 2 * i <= target_point_x) {
          // turn right two points then turn down one point to go
          index1 = index3 + 1;
          index2 = index3 + 2;
          index3 = index_cen + i * image_width + 2 * i;
          edge_image[index1] = breakpoint_position;
          edge_image[index2] = breakpoint_position;
          edge_image[index3] = breakpoint_position;
          if (magnitude_enable) {
            magnitude_image[index1] = magnitude_num;
            magnitude_image[index2] = magnitude_num;
            magnitude_image[index3] = magnitude_num;
          }
        } else {     // point by point tracing when target closed
          for (int j = 1; j < target_point_y - (i - 1); j++) {
            index3 = index3 + image_width;
            edge_image[index3] = breakpoint_position;
            if (magnitude_enable) magnitude_image[index3] = magnitude_num;
          }
          for (int j = 1; j < target_point_x - 2 * (i - 1); j++) {
            index3 = index3 + 1;
            edge_image[index3] = breakpoint_position;
            if (magnitude_enable) magnitude_image[index3] = magnitude_num;
          }
          break;
        }
      }
      break;
    case 4:
      for (int i = 1; i < serch_length; i++) {
        if (2 * i <= target_point_y && i <= target_point_x) {
          // turn down two points then turn right one point to go
          index1 = index3 + image_width;
          index2 = index3 + 2 * image_width;
          index3 = index_cen + 2 * i * image_width + i;
          edge_image[index1] = breakpoint_position;
          edge_image[index2] = breakpoint_position;
          edge_image[index3] = breakpoint_position;
          if (magnitude_enable) {
            magnitude_image[index1] = magnitude_num;
            magnitude_image[index2] = magnitude_num;
            magnitude_image[index3] = magnitude_num;
          }
        } else {      // point by point tracing when target closed
          for (int j = 1; j < target_point_y - 2 * (i - 1); j++) {
            index3 = index3 + image_width;
            edge_image[index3] = breakpoint_position;
            if (magnitude_enable) magnitude_image[index3] = magnitude_num;
          }
          for (int j = 1; j < target_point_x-(i - 1); j++) {
            index3 = index3 + 1;
            edge_image[index3] = breakpoint_position;
            if (magnitude_enable) magnitude_image[index3] = magnitude_num;
          }
          break;
        }
      }
      break;
    case 5:
      for (int i = 1; i < serch_length; i++) {
        if (2 * i <= target_point_y && i <= -target_point_x) {
          // turn down two points then turn left one point to go
          index1 = index3 + image_width;
          index2 = index3 + 2 * image_width;
          index3 = index_cen + 2 * i * image_width - i;
          edge_image[index1] = breakpoint_position;
          edge_image[index2] = breakpoint_position;
          edge_image[index3] = breakpoint_position;
          if (magnitude_enable) {
            magnitude_image[index1] = magnitude_num;
            magnitude_image[index2] = magnitude_num;
            magnitude_image[index3] = magnitude_num;
          }
        } else {       // point by point tracing when target closed
          for (int j = 1; j < target_point_y - 2 * (i - 1); j++) {
            index3 = index3 + image_width;
            edge_image[index3] = breakpoint_position;
            if (magnitude_enable) magnitude_image[index3] = magnitude_num;
          }
          for (int j = 1; j < -target_point_x - (i - 1); j++) {
            index3 = index3 - 1;
            edge_image[index3] = breakpoint_position;
            if (magnitude_enable) magnitude_image[index3] = magnitude_num;
          }
          break;
        }
      }
      break;
    case 6:
      for (int i = 1; i < serch_length; i++) {
        if (i <= target_point_y && (2 * i) <= -target_point_x) {
          // turn left two points then turn down one point to go
          index1 = index3 - 1;
          index2 = index3 - 2;
          index3 = index_cen + i * image_width - 2 * i;
          edge_image[index1] = breakpoint_position;
          edge_image[index2] = breakpoint_position;
          edge_image[index3] = breakpoint_position;
          if (magnitude_enable) {
            magnitude_image[index1] = magnitude_num;
            magnitude_image[index2] = magnitude_num;
            magnitude_image[index3] = magnitude_num;
          }
        } else {        // point by point tracing when target closed
          for (int j = 1; j < target_point_y - (i - 1); j++) {
            index3 = index3 + image_width;
            edge_image[index3] = breakpoint_position;
            if (magnitude_enable) magnitude_image[index3] = magnitude_num;
          }
          for (int j = 1; j < -target_point_x - 2 * (i - 1); j++) {
            index3 = index3 - 1;
            edge_image[index3] = breakpoint_position;
            if (magnitude_enable) magnitude_image[index3] = magnitude_num;
          }
          break;
        }
      }
      break;
    case 7:
      for (int i = 1; i < serch_length; i++) {
        if (i <= -target_point_y && (2 * i) <= -target_point_x) {
          // turn left two points then ture up one point to go
          index1 = index3 - 1;
          index2 = index3 - 2;
          index3 = index_cen - i * image_width - 2 * i;
          edge_image[index1] = breakpoint_position;
          edge_image[index2] = breakpoint_position;
          edge_image[index3] = breakpoint_position;
          if (magnitude_enable) {
            magnitude_image[index1] = magnitude_num;
            magnitude_image[index2] = magnitude_num;
            magnitude_image[index3] = magnitude_num;
          }
        } else {
          for (int j = 1; j < -target_point_y - (i - 1); j++) {
            // point by point tracing when target closed
            index3 = index3 - image_width;
            edge_image[index3] = breakpoint_position;
            if (magnitude_enable) magnitude_image[index3] = magnitude_num;
          }
          for (int j = 1; j < -target_point_x - 2 * (i - 1); j++) {
            index3 = index3 - 1;
            edge_image[index3] = breakpoint_position;
            if (magnitude_enable) magnitude_image[index3] = magnitude_num;
          }
          break;
        }
      }
    }
  }
}


__global__ void ConnectEdge(int* edge_image, int* magnitude_image,
                            int magnitude_num, int magnitude_enable,
                            int image_width, int image_height,
                            int min_length, int step, int max_length) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index_cen = y * image_width + x;
  int point_pos[4];
  int pos_num;
  int breakpoint_position;
  int breakpoint_other = 0xffffffff;
  int breakpoint_finded = false;
  int delta_index_cen;
  int index_arround[7];

  int next_index_cen = 0;

  int serch_length;
  int serch_length_max = max_length;
  int index[8 * 20];
  int ind_x[8 * 20];
  int ind_y[8 * 20];

  int point_in_edge_finded;
  int connect_point_finded;
  int target;

  int target_x;
  int target_y;
  int target_position;
  int search_target_position;
  int target_point_x;
  int target_point_y;
  int target_point;

  BreakPointCreation(edge_image, x, y,
                     point_pos, &pos_num,
                     &delta_index_cen,
                     &breakpoint_position, &breakpoint_finded,
                     image_width, image_height);
  // back tracing from new breakpoint for preparing connection
  if (breakpoint_finded) {
    index_cen = delta_index_cen + index_cen;
    delta_index_cen = -delta_index_cen;
    edge_image[index_cen] = breakpoint_position;

    int del_target;

    for (serch_length = min_length;
         serch_length <= serch_length_max;
         serch_length+=step) {
      // serch_length is a half of the board length of searching box
      point_in_edge_finded = false;
      connect_point_finded = false;

      if (next_index_cen != 8) {
        // back tracing to find the edge with the breakpoint
        for (int j = 0; j < 2 * step; j++) {
          next_index_cen = 8;
          SetBreakPointSearchOrder(index_arround,
                                   index_cen,
                                   delta_index_cen,
                                   image_width);

          for (int i = 6; i >= 0; i--) {
            if (edge_image[index_arround[i]] == WHITE) {
              edge_image[index_arround[i]] = breakpoint_position;
            } else if (edge_image[index_arround[i]] != 0 &&
                      edge_image[index_arround[i]] != breakpoint_position) {
              breakpoint_other = edge_image[index_arround[i]];
            }

            // to avoid the loop break when surrounding points
            // are all breakpoint_position
            if (edge_image[index_arround[i]] == breakpoint_position) {
              next_index_cen = i;
            }
          }

          if (next_index_cen == 8) break;
          delta_index_cen = index_cen - index_arround[next_index_cen];
          index_cen = index_arround[next_index_cen];
        }
      }

      index_cen = y * image_width + x;

      for (int i = 0; i < serch_length; i++) {
        // up
        index[i] = max(y - serch_length, 0) * image_width +
                   max(x - serch_length + i, 0);
        ind_x[i] = -serch_length + i;
        ind_y[i] = -serch_length;
        index[i + serch_length] = max(y - serch_length, 0) * image_width +
                                  min(x + i, image_width);
        ind_x[i + serch_length] = i;
        ind_y[i + serch_length] = -serch_length;
        // right
        index[i + 2 * serch_length] = max(y - serch_length + i, 0) *
                                      image_width +
                                      min(x + serch_length, image_width);
        ind_x[i + 2 * serch_length] = serch_length;
        ind_y[i + 2 * serch_length] = -serch_length + i;
        index[i + 3 * serch_length] = min(y + i, image_height) * image_width +
                                      min(x + serch_length, image_width);
        ind_x[i + 3 * serch_length] = serch_length;
        ind_y[i + 3 * serch_length] = i;
        // down
        index[i + 4 * serch_length] = min(y + serch_length, image_height) *
                                      image_width +
                                      min(x + serch_length - i, image_width);
        ind_x[i + 4 * serch_length] = serch_length - i;
        ind_y[i + 4 * serch_length] = serch_length;
        index[i + 5 * serch_length] = min(y + serch_length, image_height) *
                                      image_width +
                                      max(x - i, 0);
        ind_x[i + 5 * serch_length] = -i;
        ind_y[i + 5 * serch_length] = serch_length;
        // left
        index[i + 6 * serch_length] = min(y + serch_length - i, image_height) *
                                      image_width +
                                      max(x - serch_length, 0);
        ind_x[i + 6 * serch_length] = -serch_length;
        ind_y[i + 6 * serch_length] = serch_length - i;
        index[i + 7 * serch_length] = max(y - i, 0) * image_width +
                                      max(x - serch_length, 0);
        ind_x[i + 7 * serch_length] = -serch_length;
        ind_y[i + 7 * serch_length] = -i;
      }

      int first_connnected_position;
      int first_connected_x;
      int first_connected_y;

      for (int i = 0; i < 8 * serch_length; i++) {
        if (connect_point_finded && point_in_edge_finded) break;
        if (edge_image[index[i]] != 0 && edge_image[index[i]] != COLOUR_RED &&
           edge_image[index[i]] != breakpoint_position &&
           edge_image[index[i]] != breakpoint_other) {
          connect_point_finded = true;
          first_connnected_position = index[i];
          first_connected_x = ind_x[i];
          first_connected_y = ind_y[i];
        } else if (edge_image[index[i]] == breakpoint_position ||
                  edge_image[index[i]] == breakpoint_other) {
          target = index_cen - ind_y[i] * image_width - ind_x[i];
          del_target = -ind_y[i] * image_width - ind_x[i];
          target_x = -ind_x[i];
          target_y = -ind_y[i];
          point_in_edge_finded = true;
        }
      }

      if (connect_point_finded) {
        if (point_in_edge_finded == false) {
          target_point_x = first_connected_x;
          target_point_y = first_connected_y;
          SetPointApproximatePosition(target_point_x, target_point_y,
                                      &search_target_position, serch_length);
          target_point = first_connnected_position - (y * image_width + x);
        } else {
          SetPointApproximatePosition(target_x, target_y,
                                      &target_position, serch_length);
          if (edge_image[target] != 0 && edge_image[target] != COLOUR_RED &&
             edge_image[target] != breakpoint_position &&
             edge_image[target] != breakpoint_other) {
            target_point_x = target_x;
            target_point_y = target_y;
            search_target_position = target_position;
            target_point = target - (y * image_width + x);
            break;
          } else {
            SetConnectedPointSearchOrder(x, y, index, ind_x, ind_y,
                                         target_position, image_width,
                                         image_height, serch_length);
            for (int i = 0; i < 8 * serch_length; i++) {
              if (edge_image[index[i]] != 0 &&
                  edge_image[index[i]] != COLOUR_RED &&
                  edge_image[index[i]] != breakpoint_position &&
                  edge_image[index[i]] != breakpoint_other) {
                target_point_x = ind_x[i];
                target_point_y = ind_y[i];
                SetPointApproximatePosition(ind_x[i], ind_y[i],
                                            &search_target_position,
                                            serch_length);
                target_point = index[i] - (y * image_width + x);
                break;
              }
            }
          }
        }
        break;
      } else {
          index_cen = index_arround[next_index_cen];
      }
    }

    index_cen = y * image_width + x;
    if (connect_point_finded == true) {
      ConnectPoint(edge_image,
                   search_target_position,
                   index_cen, target_point,
                   target_point_x, target_point_y,
                   COLOUR_RED,
                   image_width,
                   serch_length,
                   magnitude_image,
                   magnitude_num,
                   magnitude_enable);
    }
  }
}

}  // namespace canny_and_edge_connection
