// Copyright 2013-10 sxniu
#include "include/D3DRes.h"

D3DRes::D3DRes()
  : m_d3d_device(NULL)
  , m_vb(NULL)
  , m_texture(NULL)
  , m_d3d(NULL)
  , m_back_surface(NULL)
  , m_temp_surface(NULL)
  , m_result_surface(NULL)
  , m_temp_surface_array(NULL)
  , m_result_surface_array(NULL)
  , m_surface_width(0)
  , m_surface_height(0) {}
