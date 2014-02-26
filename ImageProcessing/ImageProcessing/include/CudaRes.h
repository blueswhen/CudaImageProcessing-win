// Copyright 2013-10 sxniu
#ifndef IMAGEPROCESSING_INCLUDE_CUDARES_H_
#define IMAGEPROCESSING_INCLUDE_CUDARES_H_

class Paremeter;
struct cudaGraphicsResource;
struct cudaArray;

class CudaRes {
 public:
  friend class D3DInit;
  friend class Cuda3DInit;
  friend class CudaAlgorithm;
  CudaRes();

 private:
  struct cudaGraphicsResource* m_surface_cuda;
  cudaArray* m_surface_array;
  size_t m_cuda_pitch;

  // create an array in graphic memory to save gauss data
  float* m_gaussnum_dev;
  // create value in graphic memory to save weight sum
  float* m_weightsum_dev;
  // create array in graphic memory to save original image
  int* m_source_image;
  // create array in graphic memory to save probable edge image
  int* m_probable_edge_image;
  // create array in graphic memory to save edge image
  int* m_edge_image;
  // create array in graphic memory to save gauss image
  int* m_gauss_image;
  // create array in graphic memory to save gauss temp image
  float* m_gauss_temp_array;
  // back up array in graphic memory
  int* m_backup_image_1;
  int* m_backup_image_2;
  int* m_backup_image_3;
  int* m_backup_image_4;
  float* m_gradient_x_image;
  float* m_gradient_y_image;
  int* m_magnitude_image;

  // create value in graphic memory to
  // feedback whether four direction scan finish
  int* m_dev_exist;
};
#endif  // IMAGEPROCESSING_INCLUDE_CUDARES_H_
