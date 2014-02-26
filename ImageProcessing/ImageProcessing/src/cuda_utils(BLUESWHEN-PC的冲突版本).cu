// Copyright 2013-10 sxniu
#include "include/cuda_utils.h"
#include <cuda_runtime.h>
#include <windows.h>
#include <algorithm>
#include "include/ConstValue.h"

namespace cuda_utils {

__global__ void SetOrRemoveColourAreaEdge(int* edge_image,
                                          int image_width,
                                          int image_height,
                                          int area_colour,
                                          int set_or_removed,
                                          int new_edge_colour) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index_cen = y * image_width + x;
  int index[8];
  index[0] = max(y - 1, 0) * image_width + max(x - 1, 0);
  index[1] = max(y - 1, 0) * image_width + min(x + 1, image_width);
  index[2] = min(y + 1, image_height) * image_width + min(x + 1, image_width);
  index[3] = min(y + 1, image_height) * image_width + max(x - 1, 0);
  index[4] = max(y - 1, 0) * image_width + x;
  index[5] = y * image_width + max(x - 1, 0);
  index[6] = y * image_width + min(x + 1, image_width);
  index[7] = min(y + 1, image_height) * image_width + x;

  int enable = 0;
  int once1 = true;
  int once2 = true;
  if (edge_image[index_cen] == WHITE) {
    for (int i = 0; i < 8; i++) {
      if (edge_image[index[i]] == area_colour && once1) {
        enable++;
        once1 = false;
      }
      if (edge_image[index[i]] == 0 && once2) {
        enable++;
        once2 = false;
      }
      if (enable == 2) {
        if (set_or_removed)
          edge_image[index_cen] = new_edge_colour;
        else
          edge_image[index_cen] = 0;
        break;
      }
    }
  }
  if (edge_image[index_cen] == area_colour && set_or_removed) {
    for (int i = 4; i < 8; i++) {
      if (edge_image[index[i]] == 0) {
        edge_image[index[i]] = new_edge_colour;
        break;
      }
    }
  }
}

__global__ void RemoveEdgeFromColourArea(int* edge_image,
                                         int image_width,
                                         int image_height,
                                         int area_colour) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index_cen = y * image_width + x;
  int index[8];
  index[0] = max(y - 1, 0) * image_width + max(x - 1, 0);
  index[1] = max(y - 1, 0) * image_width + min(x + 1, image_width);
  index[2] = min(y + 1, image_height) * image_width + min(x + 1, image_width);
  index[3] = min(y + 1, image_height) * image_width + max(x - 1, 0);
  index[4] = max(y - 1, 0) * image_width + x;
  index[5] = y * image_width + max(x - 1, 0);
  index[6] = y * image_width + min(x + 1, image_width);
  index[7] = min(y + 1, image_height) * image_width + x;

  int remove_en = true;
  if (edge_image[index_cen] == WHITE) {
    for (int i = 0; i < 8; i++) {
      if (edge_image[index[i]] == 0) {
        remove_en = false;
        break;
      }
    }
    if (remove_en) edge_image[index_cen] = area_colour;
  }
}

__global__ void RemoveColour(int* edge_image_src, int* edge_image_dst,
                             int image_width, int image_height,
                             int remove_colour) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index_cen = y * image_width + x;

  if (edge_image_src[index_cen] == remove_colour) {
    edge_image_dst[index_cen] = 0;
  }
}

__global__ void FindArroundPointsFromColourEdge(int* edge_image,
                                                int close_colour,
                                                int image_width,
                                                int image_height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index_cen = y * image_width + x;

  int index[8];
  index[0] = max(y - 1, 0) * image_width + max(x - 1, 0);
  index[1] = max(y - 1, 0) * image_width + min(x + 1, image_width);
  index[2] = min(y + 1, image_height) * image_width + min(x + 1, image_width);
  index[3] = min(y + 1, image_height) * image_width + max(x - 1, 0);
  index[4] = max(y - 1, 0) * image_width + x;
  index[5] = y * image_width + max(x - 1, 0);
  index[6] = y * image_width + min(x + 1, image_width);
  index[7] = min(y + 1, image_height) * image_width + x;

  if (edge_image[index_cen] == close_colour) {
    for (int i = 0; i < 8; i++) {
      if (edge_image[index[i]] == WHITE)
        edge_image[index[i]] = close_colour;
    }
  }
}

__global__ void ChangeBreakPointColour(int* edge_image,
                                       int breakpoint_colour,
                                       int image_width,
                                       int image_height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index_cen = y * image_width + x;

  if (edge_image[index_cen] != 0 && edge_image[index_cen] != WHITE)
    edge_image[index_cen] = breakpoint_colour;
}

__global__ void SetBoard(int* source_image, int* edge_image,
                         int image_width, int image_height,
                         BoardType direction) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index_cen = y * image_width + x;

  if (y > 0 && y < image_height - 1) {
    if (direction == BOARD_UP) {
      if (y == 2 && (source_image[index_cen] & WHITE) != 0 ||
          (source_image[index_cen - image_width] & WHITE) == 0 &&
          (source_image[index_cen] & WHITE) != 0) {
        edge_image[index_cen + image_width] = WHITE;
      }
    } else if (direction == BOARD_DOWN) {
      if (y == image_height - 3 &&
          (source_image[index_cen] & WHITE) != 0 ||
          (source_image[index_cen + image_width] & WHITE) == 0 &&
          (source_image[index_cen] & WHITE) != 0) {
        edge_image[index_cen - image_width] = WHITE;
      }
    } else if (direction == BOARD_LEFT) {
      if (x == 2) edge_image[index_cen + 1] = WHITE;
    } else {
      if (x == image_width - 3) edge_image[index_cen - 1] = WHITE;
    }
  }
}

__global__ void RemoveBoard(int* source_image, int* edge_image,
                            int image_width, int image_height,
                            BoardType direction) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index_cen = y * image_width + x;

  if (y > 0 && y < image_height - 1) {
    if (direction == BOARD_UP) {
      if (y == 2 && (source_image[index_cen] & WHITE) != 0 ||
          (source_image[index_cen - image_width] & WHITE) == 0 &&
          (source_image[index_cen] & WHITE) != 0) {
        edge_image[index_cen - image_width] = 0;
        edge_image[index_cen] = 0;
        edge_image[index_cen + image_width] = 0;
      }
    } else if (direction == BOARD_DOWN) {
      if (y == image_height - 3 &&
          (source_image[index_cen] & WHITE) != 0 ||
          (source_image[index_cen + image_width] & WHITE) == 0 &&
          (source_image[index_cen] & WHITE) != 0) {
        edge_image[index_cen - image_width] = 0;
        edge_image[index_cen] = 0;
        edge_image[index_cen + image_width] = 0;
      }
    } else if (direction == BOARD_LEFT) {
      if (x == 2) {
        edge_image[index_cen - 1] = 0;
        edge_image[index_cen] = 0;
        edge_image[index_cen + 1] = 0;
      }
    } else if (direction == BOARD_RIGHT) {
      if (x == image_width - 3) {
        edge_image[index_cen - 1] = 0;
        edge_image[index_cen] = 0;
        edge_image[index_cen + 1] = 0;
      }
    } else {
      if (y == 2 && (source_image[index_cen] & WHITE) != 0 ||
          (source_image[index_cen - image_width] & WHITE) == 0 &&
          (source_image[index_cen] & WHITE) != 0) {
        edge_image[index_cen - image_width] = 0;
        edge_image[index_cen] = 0;
        edge_image[index_cen + image_width] = 0;
      } else if (y == image_height - 3 &&
                 (source_image[index_cen] & WHITE) != 0 ||
                 (source_image[index_cen + image_width] & WHITE) == 0 &&
                 (source_image[index_cen] & WHITE) != 0) {
        edge_image[index_cen - image_width] = 0;
        edge_image[index_cen] = 0;
        edge_image[index_cen + image_width] = 0;
      } else if (x == 2) {
        edge_image[index_cen - 1] = 0;
        edge_image[index_cen] = 0;
        edge_image[index_cen + 1] = 0;
      } else if (x == image_width - 3) {
        edge_image[index_cen - 1] = 0;
        edge_image[index_cen] = 0;
        edge_image[index_cen + 1] = 0;
      }
    }
  }
}

// area1 is common colour, source image has area2 and destination not
// destination image has area3 and source not
__global__ void FindCommonColourFromDifferentImage(
    int* edge_image_src, int* edge_image_dst, int image_width,
    int image_height, int reference_colour, int common_colour,
    int source_only_colour, int destination_only_colour) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index_cen = y * image_width + x;

  if (edge_image_src[index_cen] == reference_colour &&
      edge_image_dst[index_cen] == reference_colour)
    edge_image_dst[index_cen] = common_colour;
  else if (edge_image_src[index_cen] == reference_colour)
    edge_image_dst[index_cen] = source_only_colour;
  else if (edge_image_dst[index_cen] == reference_colour)
    edge_image_dst[index_cen] = destination_only_colour;
}

__global__ void CopyImage(int* image_src, int* image_dst,
                          int image_width, int image_height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index_cen = y * image_width + x;

  image_dst[index_cen] = image_src[index_cen];
}

__global__ void EdgeRecovery(int* image_src, int* edge_image,
                             int image_width, int image_height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index_cen = y * image_width + x;

  if (edge_image[index_cen] == WHITE && image_src[index_cen] != WHITE) {
    image_src[index_cen] = WHITE;
  }
}

__global__ void WhiteEdgeRemoved(int* image_src,
                                 int image_width,
                                 int image_height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index_cen = y * image_width + x;

  if (image_src[index_cen] == WHITE) image_src[index_cen] = 0;
}

__global__ void FillingEdgeHole(int* image_src,
                                int image_width,
                                int image_height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index_cen = y * image_width + x;

  if (x > 1 && x < image_width - 2 && y > 1 && y < image_height - 2) {
    if (image_src[index_cen] == 0 && image_src[index_cen - 1] != 0) {
      int render_colour = image_src[index_cen - 1];
      for (int i = 0; i < 5000; i++) {
        if (image_src[index_cen + i] == 0)
          image_src[index_cen + i] = render_colour;
        else
          break;
      }
    }
  }
}

__global__ void FindColourAreaEdge(int* edge_image, int area_colour,
                                   int edge_colour, int image_width,
                                   int image_height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index_cen = y * image_width + x;

  int index[8];
  index[0] = max(y - 1, 0) * image_width + max(x - 1, 0);
  index[1] = max(y - 1, 0) * image_width + min(x + 1, image_width);
  index[2] = min(y + 1, image_height) * image_width + min(x + 1, image_width);
  index[3] = min(y + 1, image_height) * image_width + max(x - 1, 0);
  index[4] = max(y - 1, 0) * image_width + x;
  index[5] = y * image_width + max(x - 1, 0);
  index[6] = y * image_width + min(x + 1, image_width);
  index[7] = min(y + 1, image_height) * image_width + x;

  int edge_enable = 0;
  if (edge_image[index_cen] == area_colour) {
    for (int i = 4; i < 8; i++) {
      if (edge_image[index[i]] != 0 && edge_image[index[i]] != WHITE)
        edge_enable++;
    }
    if (edge_enable != 4)
      edge_image[index_cen] = edge_colour;
  }
}

__device__ void MergeArray(int* test_array_dev, int* tmp_array_dev,
                           size_t left_pos, size_t center_pos,
                           size_t right_pos) {
  int* array_front_half = test_array_dev + left_pos;
  int* array_after_half = test_array_dev + center_pos;

  size_t front_half_size = center_pos - left_pos;
  size_t after_half_size = right_pos - center_pos + 1;

  size_t ptr_front_array = 0;
  size_t ptr_after_array = 0;
  size_t ptr_tmp_array = 0;

  while (ptr_front_array <= front_half_size - 1 &&
         ptr_after_array <= after_half_size - 1) {
    if (array_front_half[ptr_front_array] <
        array_after_half[ptr_after_array]) {
      tmp_array_dev[ptr_tmp_array++] = array_front_half[ptr_front_array++];
    } else {
      tmp_array_dev[ptr_tmp_array++] = array_after_half[ptr_after_array++];
    }
  }

  while (ptr_front_array < front_half_size) {
    tmp_array_dev[ptr_tmp_array++] = array_front_half[ptr_front_array++];
  }

  while (ptr_after_array < after_half_size) {
    tmp_array_dev[ptr_tmp_array++] = array_after_half[ptr_after_array++];
  }

  for (size_t i = left_pos; i < right_pos + 1; i++) {
    test_array_dev[i] = tmp_array_dev[i - left_pos];
  }
}

void MergeSort(int* test_array_dev, size_t size) {
  dim3 block(g_block_x * g_block_y);
  dim3 grid(g_grid_x * g_grid_y);
  int* temp_array_dev = NULL;
  if (cudaMalloc(&temp_array_dev, size * sizeof(int)) != cudaSuccess) {
    MessageBox(NULL, "cudaMallocPitch failed", NULL, NULL);
  }
  MergeSort<<<grid, block>>>(test_array_dev, temp_array_dev, 0, size - 1);
  cudaFree(temp_array_dev);
}

__global__ void MergeSort(int* test_array_dev, int* tmp_array_dev,
                          size_t left_pos, size_t right_pos) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  size_t array_size = right_pos - left_pos + 1;
  size_t small_array_size = array_size / (blockDim.x * gridDim.x) + 1;

  if (left_pos < right_pos) {
	size_t array_start = index * small_array_size;
	size_t array_center = array_start + (small_array_size - 1) / 2;
	size_t array_end = array_start + small_array_size - 1;
	if (array_end < array_size) {
      MergeArray(test_array_dev, tmp_array_dev, array_start,
                 array_center + 1, array_end);
	} else if (array_start < array_size) {
		size_t array_center = array_start + (array_size - array_start - 1) / 2;
        MergeArray(test_array_dev, tmp_array_dev, array_start,
                   array_center + 1, array_size);
	}
  }
}

void Sum(int* test_array_dev, size_t size) {
  dim3 block(g_block_x * g_block_y);
  dim3 grid(g_grid_x * g_grid_y);
  DoSum<<<grid, block>>>(test_array_dev, size);
}

__global__ void DoSum(int* test_array_dev, size_t size) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ sum_block;
  size_t size_for_thread = size / (blockDim.x * gridDim.x) + 1;
  for (int i = 0; i < size_for_thread; ++i) {
    int index_array = index * size_for_thread + i;
    if (index_array >= size)
      break;
    sum_block += test_array_dev[index_array];
  }
  sum_block = 
}

}  // namespace cuda_utils
