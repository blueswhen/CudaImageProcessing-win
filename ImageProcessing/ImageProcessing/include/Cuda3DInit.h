// Copyright 2013-10 sxniu
#ifndef IMAGEPROCESSING_INCLUDE_CUDA3DINIT_H_
#define IMAGEPROCESSING_INCLUDE_CUDA3DINIT_H_

class D3DRes;
class CudaRes;
class Paremeter;

class Cuda3DInit {
 public:
  Cuda3DInit(D3DRes* d3d_res, CudaRes* cuda_res, Paremeter* prm)
    : m_d3d_res(d3d_res)
    , m_cuda_res(cuda_res)
    , m_prm(prm) {}

  void CreateResources();
  void FreeResources();
  void MapResources();
  void UnMapResources();

 private:
  D3DRes* m_d3d_res;
  CudaRes* m_cuda_res;
  Paremeter* m_prm;
};
#endif  // IMAGEPROCESSING_INCLUDE_CUDA3DINIT_H_
