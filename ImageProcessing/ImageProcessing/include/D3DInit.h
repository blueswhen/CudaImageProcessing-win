// Copyright 2013-10 sxniu
#ifndef IMAGEPROCESSING_INCLUDE_D3DINIT_H_
#define IMAGEPROCESSING_INCLUDE_D3DINIT_H_

#include <d3dx9.h>
#include <windows.h>
#include <string>

class D3DRes;
class Paremeter;

class D3DInit {
 public:
  D3DInit(char* filename, D3DRes* res, Paremeter* prm, HWND hwnd);
  HRESULT Initialization();
  // draw a image to show
  void Render();
  void CleanUp();
  // show image to window
  void Present();
  // save image with bmp file
  void ScreenShot(char* filename);
  void LockSurfaces();
  void UnlockSurfaces();
  // copy surface to backbuffer
  void UpdateSurface();

 private:
  struct CUSTOMVERTEX {
    D3DXVECTOR3 position;
    FLOAT tu, tv;
  };

  HRESULT InitD3D();
  HRESULT InitGeometry();
  void SetupMatrices();

  std::string m_filename;
  HWND m_hwnd;
  D3DRes* m_res;
  Paremeter* m_prm;
};
#endif  // IMAGEPROCESSING_INCLUDE_D3DINIT_H_
