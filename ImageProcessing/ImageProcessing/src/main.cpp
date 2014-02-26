// Copyright 2013-10 sxniu
#include <string>
#include "include/LoadLib.h"
#include "include/D3DRes.h"
#include "include/CudaRes.h"
#include "include/Paremeter.h"
#include "include/D3DInit.h"
#include "include/Cuda3DInit.h"
#include "include/utils.h"
#include "include/CudaAlgorithm.h"

using utils::ShowTime;

LRESULT WINAPI MsgProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
  switch (msg) {
  case WM_DESTROY:
    PostQuitMessage(0);
    return 0;
  }
  return DefWindowProc(hwnd, msg, wparam, lparam);
}

int main(int argc, char** argv) {
  char* filename = "source.bmp";
  std::string argv1;
  int count_time = 0;
  if (argc == 2) {
    argv1 = argv[1];
    if (argv1 == "time")
      count_time = 1;
  }

  Paremeter prm;
  prm.ReadIniFile();

  WNDCLASSEX wc;
  wc.cbSize = sizeof(WNDCLASSEX);
  wc.style = CS_CLASSDC;
  wc.lpfnWndProc = MsgProc;
  wc.cbClsExtra = 0L;
  wc.cbWndExtra = 0L;
  wc.hInstance = GetModuleHandle(NULL);
  wc.hIcon = NULL;
  wc.hCursor = NULL;
  wc.hbrBackground = NULL;
  wc.lpszMenuName = NULL;
  wc.lpszClassName = "FourDirectionScan";
  wc. hIconSm = NULL;
  RegisterClassEx(&wc);

  HWND hwnd = CreateWindow("FourDirectionScan", "FourDirectionScan",
                           WS_OVERLAPPEDWINDOW, 100, 100, 512, 512,
                           GetDesktopWindow(), NULL, wc.hInstance, NULL);

  D3DRes d3d_res;
  D3DInit d3d(filename, &d3d_res, &prm, hwnd);

  double freq = 0;
  double start_time = 0;
  int time = 0;

  if (SUCCEEDED(d3d.Initialization())) {
    CudaRes cuda_res;
    Cuda3DInit cuda(&d3d_res, &cuda_res, &prm);
    cuda.CreateResources();

    ShowWindow(hwnd, SW_SHOWDEFAULT);
    UpdateWindow(hwnd);

    MSG msg;
    ZeroMemory(&msg, sizeof(msg));
    int num = 0;
    while (msg.message != WM_QUIT) {
      if (num < 100000)  // create dynamic effect
        num++;
      // if (num % 5 == 0)
        prm.Kplus();

      if (PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
      } else {
        d3d.Render();

#ifndef PROCESS_IMAGE_BY_LOCK_SURFACE
        cuda.MapResources();
        CudaAlgorithm ca(&prm, &cuda_res);

        // counting time
        ShowTime(&freq, &start_time, &time, 1, count_time);

        ca.Main();  // cuda algorithm

        ShowTime(&freq, &start_time, &time, 0, count_time);
        cuda.UnMapResources();
#else
        d3d.LockSurfaces();
        RegionFilling(&d3d_res);
        d3d.UnlockSurfaces();
        d3d.UpdateSurface();
#endif  // PROCESS_IMAGE_BY_LOCK_SURFACE
        d3d.Present();
      }
    }
    d3d.ScreenShot("result.bmp");
    d3d.CleanUp();
    cuda.FreeResources();
  }

  UnregisterClass("FourDirectionScan", wc.hInstance);
  return 0;
}
