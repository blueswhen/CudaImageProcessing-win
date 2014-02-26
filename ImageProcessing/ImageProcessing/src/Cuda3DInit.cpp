// Copyright 2013-10 sxniu
#include "include/Cuda3DInit.h"
#include <cuda_runtime.h>
#include <cuda_d3d9_interop.h>
#include <string>
#include "include/D3DRes.h"
#include "include/CudaRes.h"
#include "include/ConstValue.h"
#include "include/Paremeter.h"

void Cuda3DInit::CreateResources() {
  if (cudaD3D9SetDirect3DDevice(m_d3d_res->m_d3d_device) != cudaSuccess) {
    MessageBox(NULL, "cudaD3D9SetDirect3DDevice SHIT", NULL, NULL);
  }
  if (cudaGraphicsD3D9RegisterResource(
          &(m_cuda_res->m_surface_cuda), m_d3d_res->m_back_surface,
          cudaGraphicsRegisterFlagsNone) != cudaSuccess) {
    MessageBox(NULL, "cudaGraphicsD3D9RegisterResource SHIT", NULL, NULL);
  }
  if (cudaMalloc(&(m_cuda_res->m_gaussnum_dev),
                (1 + 2 * THREE_SIGMA_INT) * sizeof(float)) != cudaSuccess) {
    MessageBox(NULL, "cudaMallocPitch SHIT", NULL, NULL);
  }
  if (cudaMalloc(&(m_cuda_res->m_weightsum_dev),
                sizeof(float)) != cudaSuccess) {
    MessageBox(NULL, "cudaMallocPitch SHIT", NULL, NULL);
  }
  if (cudaMemcpy(m_cuda_res->m_gaussnum_dev, m_prm->m_gaussnum,
                (1 + 2 * THREE_SIGMA_INT) * sizeof(float),
                cudaMemcpyHostToDevice) != cudaSuccess) {
    MessageBox(NULL, "cudaMemcpy1 SHIT", NULL, NULL);
  }
  if (cudaMemcpy(m_cuda_res->m_weightsum_dev, &(m_prm->m_weightsum),
                sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
    MessageBox(NULL, "cudaMemcpy2 SHIT", NULL, NULL);
  }
  if (cudaMallocPitch(&(m_cuda_res->m_source_image),
                     &(m_cuda_res->m_cuda_pitch),
                     m_prm->m_s_width * sizeof(int),
                     m_prm->m_s_height) != cudaSuccess) {
    MessageBox(NULL, "cudaMallocPitch SHIT", NULL, NULL);
  }
  if (cudaMallocPitch(&(m_cuda_res->m_probable_edge_image),
                     &(m_cuda_res->m_cuda_pitch),
                     m_prm->m_s_width * sizeof(int),
                     m_prm->m_s_height) != cudaSuccess) {
    MessageBox(NULL, "cudaMallocPitch SHIT", NULL, NULL);
  }
  if (cudaMallocPitch(&(m_cuda_res->m_edge_image),
                     &(m_cuda_res->m_cuda_pitch),
                     m_prm->m_s_width * sizeof(int),
                     m_prm->m_s_height) != cudaSuccess) {
    MessageBox(NULL, "cudaMallocPitch SHIT", NULL, NULL);
  }
  if (cudaMallocPitch(&(m_cuda_res->m_gauss_image),
                     &(m_cuda_res->m_cuda_pitch),
                     m_prm->m_s_width * sizeof(int),
                     m_prm->m_s_height) != cudaSuccess) {
    MessageBox(NULL, "cudaMallocPitch SHIT", NULL, NULL);
  }
  if (cudaMallocPitch(&(m_cuda_res->m_gauss_temp_array),
                     &(m_cuda_res->m_cuda_pitch),
                     m_prm->m_s_width * sizeof(float),
                     m_prm->m_s_height) != cudaSuccess) {
    MessageBox(NULL, "cudaMallocPitch SHIT", NULL, NULL);
  }
  if (cudaMallocPitch(&(m_cuda_res->m_backup_image_1),
                     &(m_cuda_res->m_cuda_pitch),
                     m_prm->m_s_width * sizeof(int),
                     m_prm->m_s_height) != cudaSuccess)  {
    MessageBox(NULL, "cudaMallocPitch SHIT", NULL, NULL);
  }
  if (cudaMallocPitch(&(m_cuda_res->m_backup_image_2),
                     &(m_cuda_res->m_cuda_pitch),
                     m_prm->m_s_width * sizeof(int),
                     m_prm->m_s_height) != cudaSuccess)  {
    MessageBox(NULL, "cudaMallocPitch SHIT", NULL, NULL);
  }
  if (cudaMallocPitch(&(m_cuda_res->m_backup_image_3),
                     &(m_cuda_res->m_cuda_pitch),
                     m_prm->m_s_width * sizeof(int),
                     m_prm->m_s_height) != cudaSuccess)  {
    MessageBox(NULL, "cudaMallocPitch SHIT", NULL, NULL);
  }
  if (cudaMallocPitch(&(m_cuda_res->m_backup_image_4),
                     &(m_cuda_res->m_cuda_pitch),
                     m_prm->m_s_width * sizeof(int),
                     m_prm->m_s_height) != cudaSuccess)  {
    MessageBox(NULL, "cudaMallocPitch SHIT", NULL, NULL);
  }
  if (cudaMallocPitch(&(m_cuda_res->m_gradient_x_image),
                     &(m_cuda_res->m_cuda_pitch),
                     m_prm->m_s_width * sizeof(float),
                     m_prm->m_s_height) != cudaSuccess) {
    MessageBox(NULL, "cudaMallocPitch SHIT", NULL, NULL);
  }
  if (cudaMallocPitch(&(m_cuda_res->m_gradient_y_image),
                     &(m_cuda_res->m_cuda_pitch),
                     m_prm->m_s_width * sizeof(float),
                     m_prm->m_s_height) != cudaSuccess) {
    MessageBox(NULL, "cudaMallocPitch SHIT", NULL, NULL);
  }
  if (cudaMallocPitch(&(m_cuda_res->m_magnitude_image),
                     &(m_cuda_res->m_cuda_pitch),
                     m_prm->m_s_width * sizeof(int),
                     m_prm->m_s_height) != cudaSuccess) {
    MessageBox(NULL, "cudaMallocPitch SHIT", NULL, NULL);
  }

#ifdef FOURDIRECTIONSCAN_WITH_FEEDBACK
  if (cudaMalloc(&(m_cuda_res->m_dev_exist), sizeof(int)) != cudaSuccess) {
    MessageBox(NULL, "cudaMalloc SHIT", NULL, NULL);
  }
#endif
}

void Cuda3DInit::FreeResources() {
  cudaFree(m_cuda_res->m_gaussnum_dev);
  cudaFree(m_cuda_res->m_weightsum_dev);
  cudaFree(m_cuda_res->m_source_image);
  cudaFree(m_cuda_res->m_probable_edge_image);
  cudaFree(m_cuda_res->m_edge_image);
  cudaFree(m_cuda_res->m_gauss_image);
  cudaFree(m_cuda_res->m_gauss_temp_array);
  cudaFree(m_cuda_res->m_backup_image_1);
  cudaFree(m_cuda_res->m_backup_image_2);
  cudaFree(m_cuda_res->m_backup_image_3);
  cudaFree(m_cuda_res->m_backup_image_4);
  cudaFree(m_cuda_res->m_gradient_x_image);
  cudaFree(m_cuda_res->m_gradient_y_image);
  cudaFree(m_cuda_res->m_magnitude_image);

#ifdef FOURDIRECTIONSCAN_WITH_FEEDBACK
  cudaFree(m_cuda_res->m_dev_exist);
#endif
}

void Cuda3DInit::MapResources() {
  if (cudaGraphicsMapResources(
          1, &(m_cuda_res->m_surface_cuda), 0) != cudaSuccess) {
    MessageBox(NULL, "cudaGraphicsMapResources SHIT", NULL, NULL);
  }
  if (cudaGraphicsSubResourceGetMappedArray(
          &(m_cuda_res->m_surface_array),
          (m_cuda_res->m_surface_cuda), 0, 0) != cudaSuccess) {
    MessageBox(NULL, "cudaGraphicsSubResourceGetMappedArray SHIT", NULL, NULL);
  }
  if (cudaMemcpyFromArray(
          m_cuda_res->m_source_image, m_cuda_res->m_surface_array,
          0, 0, m_prm->m_s_width * sizeof(int) * m_prm->m_s_height,
          cudaMemcpyDeviceToDevice) != cudaSuccess) {
    MessageBox(NULL, "cudaMemcpyFromArray SHIT", NULL, NULL);
  }
}

void Cuda3DInit::UnMapResources() {
  if (cudaMemcpy2DToArray(m_cuda_res->m_surface_array, 0, 0,
                          m_cuda_res->m_edge_image,
                          m_prm->m_s_width * sizeof(int),
                          m_prm->m_s_width * sizeof(int),
                          m_prm->m_s_height,
                          cudaMemcpyDeviceToDevice) != cudaSuccess) {
    MessageBox(NULL, "cudaMemcpy2DToArray SHIT", NULL, NULL);
  }

  if (cudaGraphicsUnmapResources(
          1, &(m_cuda_res->m_surface_cuda), 0) != cudaSuccess) {
    MessageBox(NULL, "cudaGraphicsUnmapResources SHIT", NULL, NULL);
  }
}
