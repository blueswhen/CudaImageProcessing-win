// Copyright 2013-10 sxniu
#include "include/CudaRes.h"
#include <stdlib.h>

CudaRes::CudaRes()
  : m_surface_cuda(NULL)
  , m_surface_array(NULL)
  , m_cuda_pitch(0)
  , m_gaussnum_dev(NULL)
  , m_weightsum_dev(NULL)
  , m_source_image(NULL)
  , m_probable_edge_image(NULL)
  , m_edge_image(NULL)
  , m_gauss_image(NULL)
  , m_gauss_temp_array(NULL)
  , m_backup_image_1(NULL)
  , m_backup_image_2(NULL)
  , m_backup_image_3(NULL)
  , m_backup_image_4(NULL)
  , m_gradient_x_image(NULL)
  , m_gradient_y_image(NULL)
  , m_magnitude_image(NULL)
  , m_dev_exist(NULL) {}
