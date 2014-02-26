// Copyright 2013-10 sxniu
#include "include/D3DInit.h"
#include <windows.h>
#include "include/LoadLib.h"
#include "include/D3DRes.h"
#include "include/Paremeter.h"

#define D3DFVF_CUSTOMVERTEX (D3DFVF_XYZ | D3DFVF_TEX1)

D3DInit::D3DInit(char* filename, D3DRes* res, Paremeter* prm, HWND m_hwnd)
  : m_filename(filename)
  , m_hwnd(m_hwnd)
  , m_res(res)
  , m_prm(prm) {}

HRESULT D3DInit::InitD3D() {
  if (NULL == (m_res->m_d3d = Direct3DCreate9(D3D_SDK_VERSION)))
    return E_FAIL;
  D3DPRESENT_PARAMETERS d3dpp;
  ZeroMemory(&d3dpp, sizeof(d3dpp));
  d3dpp.Windowed = 1;
  d3dpp.BackBufferCount = 1;
  d3dpp.BackBufferHeight = m_prm->m_s_height;
  d3dpp.BackBufferWidth = m_prm->m_s_width;
  d3dpp.hDeviceWindow = m_hwnd;
  d3dpp.Flags = D3DPRESENTFLAG_LOCKABLE_BACKBUFFER;
  d3dpp.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;
  d3dpp.SwapEffect = D3DSWAPEFFECT_DISCARD;
  d3dpp.BackBufferFormat = D3DFMT_A8R8G8B8;
  d3dpp.EnableAutoDepthStencil = TRUE;
  d3dpp.AutoDepthStencilFormat = D3DFMT_D32F_LOCKABLE;

  m_res->m_surface_width = m_prm->m_s_width;
  m_res->m_surface_height = m_prm->m_s_height;

  if (FAILED(m_res->m_d3d->CreateDevice(
                              D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL,
                              m_hwnd, D3DCREATE_HARDWARE_VERTEXPROCESSING,
                              &d3dpp, &(m_res->m_d3d_device)))) {
    return E_FAIL;
  }

  m_res->m_d3d_device->SetRenderState(D3DRS_LIGHTING, FALSE);
  D3DCAPS9 caps;
  m_res->m_d3d_device->GetDeviceCaps(&caps);
  int g_maxAnisotropic = caps.MaxAnisotropy;
  m_res->m_d3d_device->SetSamplerState(0, D3DSAMP_MAGFILTER,
                                       D3DTEXF_ANISOTROPIC);
  m_res->m_d3d_device->SetSamplerState(0, D3DSAMP_MINFILTER,
                                       D3DTEXF_ANISOTROPIC);
  m_res->m_d3d_device->SetSamplerState(0, D3DSAMP_MAXANISOTROPY,
                                       g_maxAnisotropic);

  if (FAILED(m_res->m_d3d_device->GetBackBuffer(
                                      0, 0, D3DBACKBUFFER_TYPE_MONO,
                                      &m_res->m_back_surface))) {
    MessageBox(NULL, "GetRenderTarget fail", NULL, NULL);
  }
#ifdef PROCESS_IMAGE_BY_LOCK_SURFACE
  if (FAILED(m_res->m_d3d_device->CreateOffscreenPlainSurface(
                                      m_prm->m_s_height, m_prm->m_s_width,
                                      D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM,
                                      &(m_res->m_temp_surface), NULL))) {
    MessageBox(NULL, "m_temp_surface fail", NULL, NULL);
  }
  if (FAILED(m_res->m_d3d_device->CreateOffscreenPlainSurface(
                                      m_prm->m_s_height, m_prm->m_s_width,
                                      D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM,
                                      &(m_res->m_result_surface), NULL))) {
    MessageBox(NULL, "m_result_surface fail", NULL, NULL);
  }
#endif  // PROCESS_IMAGE_BY_LOCK_SURFACE
  return S_OK;
}


HRESULT D3DInit::InitGeometry() {
  if (FAILED(D3DXCreateTextureFromFileEx(m_res->m_d3d_device,
                                        m_filename.c_str(),
                                        D3DX_DEFAULT,
                                        D3DX_DEFAULT,
                                        D3DX_FROM_FILE,
                                        D3DUSAGE_RENDERTARGET,
                                        D3DFMT_A8R8G8B8,
                                        D3DPOOL_DEFAULT,
                                        D3DX_FILTER_TRIANGLE|D3DX_FILTER_DITHER,
                                        D3DX_FILTER_BOX,
                                        0, NULL, NULL,
                                        &(m_res->m_texture)))) {
    MessageBox(NULL, "Could not find source image", NULL, NULL);
    return E_FAIL;
  }
  if (FAILED(m_res->m_d3d_device->CreateVertexBuffer(
                                     2 * 4 * sizeof(CUSTOMVERTEX), 0,
                                     D3DFVF_CUSTOMVERTEX, D3DPOOL_DEFAULT,
                                     &(m_res->m_vb), NULL))) {
    return E_FAIL;
  }

  CUSTOMVERTEX* pVertices;
  if (FAILED(m_res->m_vb->Lock(0, 0, reinterpret_cast<void**>(&pVertices), 0)))
    return E_FAIL;

  pVertices[0].position = D3DXVECTOR3(0, 0, 5);
  pVertices[0].tu = 0.0f;
  pVertices[0].tv = 1.0f;

  pVertices[1].position = D3DXVECTOR3(0, 10, 5);
  pVertices[1].tu = 0.0f;
  pVertices[1].tv = 0.0f;

  pVertices[2].position = D3DXVECTOR3(10, 0, 5);
  pVertices[2].tu = 1.0f;
  pVertices[2].tv = 1.0f;

  pVertices[3].position = D3DXVECTOR3(10, 10, 5);
  pVertices[3].tu = 1.0f;
  pVertices[3].tv = 0.0f;

  m_res->m_vb->Unlock();

  return S_OK;
}

void D3DInit::CleanUp() {
  if (m_res->m_texture != NULL)
    m_res->m_texture->Release();

  if (m_res->m_vb != NULL)
    m_res->m_vb->Release();

  if (m_res->m_d3d_device != NULL)
    m_res->m_d3d_device->Release();

  if (m_res->m_d3d != NULL)
    m_res->m_d3d->Release();
}

void D3DInit::SetupMatrices() {
  D3DXMATRIX matWorld;
  D3DXMatrixIdentity(&matWorld);
  m_res->m_d3d_device->SetTransform(D3DTS_WORLD, &matWorld);

  D3DXVECTOR3 vEyePt(5.0f, 5.0f, 0.0f);
  D3DXVECTOR3 vLookatPt(5.0f, 5.0f, 5.0f);
  D3DXVECTOR3 vUpVec(0.f, 1.0f, 0.0f);
  D3DXMATRIX matView;
  D3DXMatrixLookAtLH(&matView, &vEyePt, &vLookatPt, &vUpVec);

  m_res->m_d3d_device->SetTransform(D3DTS_VIEW, &matView);
  m_res->m_d3d_device->GetTransform(D3DTS_VIEW, &matView);

  D3DXMATRIX matProj;
  D3DXMatrixPerspectiveFovLH(&matProj, D3DX_PI/2, 1.0f, 1.0f, 100.0f);
  m_res->m_d3d_device->SetTransform(D3DTS_PROJECTION, &matProj);
}

void D3DInit::Render() {
  m_res->m_d3d_device->Clear(0, NULL, D3DCLEAR_TARGET|D3DCLEAR_ZBUFFER,
                             D3DCOLOR_ARGB(0, 0, 0, 0), 1.0f, 0);

  if (SUCCEEDED(m_res->m_d3d_device->BeginScene())) {
    SetupMatrices();
    m_res->m_d3d_device->SetTexture(0, m_res->m_texture);
    m_res->m_d3d_device->SetStreamSource(0, m_res->m_vb, 0,
                                         sizeof(CUSTOMVERTEX));
    m_res->m_d3d_device->SetFVF(D3DFVF_CUSTOMVERTEX);
    m_res->m_d3d_device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);
    m_res->m_d3d_device->EndScene();
  }
#ifdef PROCESS_IMAGE_BY_LOCK_SURFACE
  if (FAILED(m_res->m_d3d_device->GetRenderTargetData(
                                      m_res->m_back_surface,
                                      m_res->m_temp_surface))) {
    MessageBox(NULL, "GetRenderTargetData m_temp_surface fail", NULL, NULL);
  }
#endif  // PROCESS_IMAGE_BY_LOCK_SURFACE
}

void D3DInit::Present() {
  m_res->m_d3d_device->Present(NULL, NULL, NULL, NULL);
}

HRESULT D3DInit::Initialization() {
  if (SUCCEEDED(InitD3D())) {
    if (SUCCEEDED(InitGeometry())) {
      return S_OK;
    }
  }
  return E_FAIL;
}

void D3DInit::LockSurfaces() {
  D3DLOCKED_RECT lockRect_temp;
  D3DLOCKED_RECT lockRect_result;
  if (FAILED(m_res->m_temp_surface->LockRect(&lockRect_temp, NULL, NULL))) {
      MessageBox(NULL, "m_temp_surface LOCK FAIL", 0, 0);
  }
  if (FAILED(m_res->m_result_surface->LockRect(&lockRect_result, NULL, NULL))) {
      MessageBox(NULL, "m_result_surface LOCK FAIL", 0, 0);
  }
  m_res->m_temp_surface_array = static_cast<int*>(lockRect_temp.pBits);
  m_res->m_result_surface_array = static_cast<int*>(lockRect_result.pBits);
}

void D3DInit::UnlockSurfaces() {
  m_res->m_temp_surface->UnlockRect();
  m_res->m_result_surface->UnlockRect();
}

void D3DInit::UpdateSurface() {
  if (FAILED(m_res->m_d3d_device->UpdateSurface(
                                      m_res->m_result_surface,
                                      NULL, m_res->m_back_surface, NULL))) {
    MessageBox(NULL, "update FAIL", 0, 0);
  }
}

void D3DInit::ScreenShot(char* filename) {
    /// Save the screen date to file
    if (FAILED(D3DXSaveSurfaceToFile(filename, D3DXIFF_BMP,
                                     m_res->m_back_surface,
                                     NULL, NULL))) {
    MessageBox(NULL, "picture can not be gotten", NULL, NULL);
  }
}
