// Copyright 2013-10 sxniu
#ifndef IMAGEPROCESSING_INCLUDE_D3DRES_H_
#define IMAGEPROCESSING_INCLUDE_D3DRES_H_

#include <d3dx9.h>

class D3DRes {
 public:
  friend class D3DInit;
  friend class Cuda3DInit;
  friend void RegionFilling(D3DRes* res);
  D3DRes();

 private:
  LPDIRECT3DDEVICE9 m_d3d_device;
  LPDIRECT3DVERTEXBUFFER9 m_vb;
  LPDIRECT3DTEXTURE9 m_texture;
  LPDIRECT3D9 m_d3d;
  LPDIRECT3DSURFACE9 m_back_surface;  // backbuffer
  LPDIRECT3DSURFACE9 m_temp_surface;
  LPDIRECT3DSURFACE9 m_result_surface;

  int* m_temp_surface_array;
  int* m_result_surface_array;
  int m_surface_width;
  int m_surface_height;
};
#endif  // IMAGEPROCESSING_INCLUDE_D3DRES_H_
