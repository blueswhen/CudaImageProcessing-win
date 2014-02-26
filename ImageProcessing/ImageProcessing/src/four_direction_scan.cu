// Copyright 2013-10 sxniu
#include "include/four_direction_scan.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <string>
#include "include/ConstValue.h"
#include "include/utils.h"

namespace four_direction_scan {

__global__ void TopToEndScan(int* edge_image, int image_width,
                             int image_height, int search_length,
                             int reference_colour, int render_colour) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index_cen = y * image_width + x;
  int enable_render = false;
  int pos_num = 0;

  if (x > 1 && x < image_width - 2 && y > 1 && y < image_height - 2) {
    if (edge_image[index_cen] == WHITE &&
        edge_image[index_cen + image_width] == reference_colour) {
      for (int i = 0; i < search_length; i++) {
        if (y + i + 1 > image_height - 3) {
          enable_render = true;
          pos_num = i;
          break;
        }
        if (edge_image[index_cen+(i + 1) * image_width] == WHITE) {
          enable_render = true;
          pos_num = i;
          break;
        }
        if (edge_image[index_cen+(i + 1) * image_width] != reference_colour)
          break;
      }
      if (enable_render) {
        for (int i = 0; i < pos_num; i++) {
          edge_image[index_cen+(i + 1) * image_width] = render_colour;
        }
      }
    }
  }
}

__global__ void LeftToRightScan(int* edge_image, int image_width,
                                int image_height, int search_length,
                                int reference_colour, int render_colour) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index_cen = y * image_width + x;
  int enable_render = false;
  int pos_num = 0;

  if (x > 1 && x < image_width - 2 && y > 1 && y < image_height - 2) {
    if (edge_image[index_cen] == WHITE &&
        edge_image[index_cen + 1] == reference_colour) {
      for (int i = 0; i < search_length; i++) {
        if (x + i + 1 > image_width - 3) {
          enable_render = true;
          pos_num = i;
          break;
        }
        if (edge_image[index_cen + i + 1] == WHITE) {
          enable_render = true;
          pos_num = i;
          break;
        }
        if (edge_image[index_cen + i + 1] != reference_colour) break;
      }
      if (enable_render) {
        for (int i = 0; i < pos_num; i++) {
          edge_image[index_cen + i + 1] = render_colour;
        }
      }
    }
  }
}

__global__ void EndToTopScan(int* edge_image, int image_width,
                             int image_height, int search_length,
                             int reference_colour, int render_colour) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index_cen = y * image_width + x;
  int enable_render = false;
  int pos_num = 0;

  if (x > 1 && x < image_width - 2 && y > 1 && y < image_height - 2) {
    if (edge_image[index_cen] == WHITE &&
        edge_image[index_cen - image_width] == reference_colour) {
      for (int i = 0; i < search_length; i++) {
        if (y-(i + 1) < 2) {
          enable_render = true;
          pos_num = i;
          break;
        }
        if (edge_image[index_cen-(i + 1) * image_width] == WHITE) {
          enable_render = true;
          pos_num = i;
          break;
        }
        if (edge_image[index_cen-(i + 1) * image_width] != reference_colour)
          break;
      }
      if (enable_render) {
        for (int i = 0; i < pos_num; i++) {
          edge_image[index_cen-(i + 1) * image_width] = render_colour;
        }
      }
    }
  }
}

__global__ void RightToLeftScan(int* edge_image, int image_width,
                                int image_height, int search_length,
                                int reference_colour, int render_colour) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index_cen = y * image_width + x;
  int enable_render = false;
  int pos_num = 0;

  if (x > 1 && x < image_width - 2 && y > 1 && y < image_height - 2) {
    if (edge_image[index_cen] == WHITE &&
        edge_image[index_cen - 1] == reference_colour) {
      for (int i = 0; i < search_length; i++) {
        if (x - i - 1 < 2) {
          enable_render = true;
          pos_num = i;
          break;
        }
        if (edge_image[index_cen - i - 1] == WHITE) {
          enable_render = true;
          pos_num = i;
          break;
        }
        if (edge_image[index_cen - i - 1] != reference_colour) break;
      }
      if (enable_render) {
        for (int i = 0; i < pos_num; i++) {
          edge_image[index_cen - i - 1] = render_colour;
        }
      }
    }
  }
}

__global__ void RmExtraColour(int* edge_image, int image_width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index_cen = y * image_width + x;

  if (edge_image[index_cen] != WHITE &&
      edge_image[index_cen] != 0 &&
      edge_image[index_cen] != COLOUR_RED &&
      edge_image[index_cen] != COLOUR_YELLOW &&
      edge_image[index_cen] != COLOUR_PURPLE &&
      edge_image[index_cen] != COLOUR_CYAN &&
      edge_image[index_cen] != COLOUR_GREEN &&
      edge_image[index_cen] != COLOUR_BLUE &&
      edge_image[index_cen] != COLOUR_LIGHT_GRAY &&
      edge_image[index_cen] != COLOUR_LIGHT_BLUE &&
      edge_image[index_cen] != COLOUR_PURPLE_RED &&
      edge_image[index_cen] != COLOUR_LIGHT_PURPLE &&
      edge_image[index_cen] != COLOUR_DARK_YELLOW &&
      edge_image[index_cen] != COLOUR_LIGHT_GREEN &&
      edge_image[index_cen] != COLOUR_DARK_GREEN &&
      edge_image[index_cen] != COLOUR_DARK_BLUE &&
      edge_image[index_cen] != COLOUR_DARK_RED &&
      edge_image[index_cen] != COLOUR_GRAY) {
    edge_image[index_cen] = 0;
  }
}

void FourDirectionScan(int* edge_image, int image_width,
                       int image_height, int s_length,
                       int repeat_num, int fill_colour) {
  dim3 block(g_block_x, g_block_y);
  dim3 grid(g_grid_x, g_grid_y);

  int _host_exist = 8;

  int  a = 0;
  int b = 0;

  for (int i = 0; i < repeat_num; i++) {
    a = b;
    b = 2000;  // random big number
    TopToEndScan<<<grid, block>>>(edge_image, image_width,
                                  image_height, s_length, a, b);
    a = b;
    b++;
    LeftToRightScan<<<grid, block>>>(edge_image, image_width,
                                     image_height, s_length, a, b);
    a = b;
    b++;
    EndToTopScan<<<grid, block>>>(edge_image, image_width,
                                  image_height, s_length, a, b);
    a = b;
    if (i == repeat_num - 1) {
      b = fill_colour;
    } else {
      b++;
    }
    RightToLeftScan<<<grid, block>>>(edge_image, image_width,
                                     image_height, s_length, a, b);
  }
  RmExtraColour<<<grid, block>>>(edge_image, image_width);
}

#ifdef FOURDIRECTIONSCAN_WITH_FEEDBACK
__global__ void exist_identify(int* exist) {
  if (threadIdx.x == 0 && threadIdx.y == 0 &&
      blockIdx.x == 0 && blockIdx.y == 0) {
    if (*exist > 1)
      *exist = 1;
    else
      *exist = 0;
  }
}

__global__ void exist_set(int* exist) {
  if (threadIdx.x == 0 && threadIdx.y == 0 &&
      blockIdx.x == 0 && blockIdx.y == 0) {
    *exist = 1;
  }
}

__global__ void colour_exist(int* des_mem, int ref_col, int image_width,
                             int image_height, int* exist) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index_cen = y * image_width + x;

  if (des_mem[index_cen] == ref_col) {
    atomicAdd(exist, 1);
  }
}

void FourDirectionScan(int* edge_image, int image_width,
                       int image_height, int s_length, int* dev_exist) {
  dim3 block(g_block_x, g_block_y);
  dim3 grid(g_grid_x, g_grid_y);

  int _host_exist = 8;

  exist_set<<<grid, block>>>(dev_exist);

  int  a = 0;
  int b = 0;

  for (int i = 0; i < 500; i++) {
    a = b;
    b = 2000;
    TopToEndScan<<<grid, block>>>(edge_image, image_width,
                                  image_height, s_length, a, b);
    a = b;
    b++;
    LeftToRightScan<<<grid, block>>>(edge_image, image_width,
                                     image_height, s_length, a, b);
    a = b;
    b++;
    EndToTopScan<<<grid, block>>>(edge_image, image_width,
                                  image_height, s_length, a, b);
    a = b;
    b = 0x0000ffff * i;  // kColourArray[i];
    RightToLeftScan<<<grid, block>>>(edge_image, image_width,
                                     image_height, s_length, a, b);
    // RmExtraColour<<<grid, block>>>(edge_image, image_width);

    if (i > 0) {
      colour_exist<<<grid, block>>>(edge_image, 0x0000ffff * (i - 1),  // kColourArray[i - 1],
                                    image_width, image_height, dev_exist);
      exist_identify<<<grid, block>>>(dev_exist);
      cudaMemcpy(&_host_exist, dev_exist, sizeof(int), cudaMemcpyDeviceToHost);
	  if (_host_exist == 0) {
		utils::ShowNum(i+1);
        break;
      }
    }
  }
}
#endif

}  // namespace four_direction_scan
