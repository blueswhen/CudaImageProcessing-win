// Copyright 2013-10 sxniu
#include "include/Paremeter.h"
#include <math.h>
#include <windows.h>
#include <string>

using std::string;

void MakeGauss(double sigma, float *gaussnum, float *weightsum) {
  int array_center;
  double distence;

  double PI = 3.1415926535897932384626;

  double value;
  double sum;
  sum = 0;

  array_center = kWindowSize / 2;

  for (int i = 0; i < kWindowSize; i++) {
    distence = static_cast<double>(i - array_center);
    value = exp(-(1 / 2) * distence * distence / (sigma*sigma)) /
            (sqrt(2 * PI) * sigma);
    gaussnum[i] = static_cast<float>(value);
    sum += value;
  }

  // normalization
  for (int i = 0; i < kWindowSize; i++) {
    gaussnum[i] /= static_cast<float>(sum);
  }

  for (int i = 0; i < kWindowSize; i++) {
    (*weightsum) += gaussnum[i];
  }
}

Paremeter::Paremeter()
  : m_windowsize(0)
  , m_weightsum(0)
  , m_resolution_choose(0)
  , m_s_width(0)
  , m_s_height(0)
  , m_block_X(0)
  , m_block_Y(0)
  , m_grid_X(0)
  , m_grid_Y(0)
  , m_canny_enable(0)
  , m_highthreshold(0)
  , m_lowthreshold(0)
  , m_recursion_num(0)
  , m_edge_connection_enable(0)
  , m_min_l(0)
  , m_max_l(0)
  , m_del_l(0)
  , m_lena_segmentation_by_board(0)
  , m_lena_segmentation_seperate_front_back(0)
  , m_segmentation_enable(0)
  , m_segmentation_by_board_or_by_length(0)
  , m_exist(100)
  , k(0) {
  MakeGauss(kSigma, m_gaussnum, &m_weightsum);
}

void Paremeter::ReadIniFile() {
  char app_path[256];
  GetCurrentDirectory(256, app_path);
  string file_path = app_path;
  file_path += "\\parameter_setup.ini";

  string str_section = "lena";
  string str_sectionKey = "lena_segmentation_seperate_front_back";
  char in_buf[80];
  GetPrivateProfileString(str_section.c_str(),
                          str_sectionKey.c_str(),
                          NULL, in_buf, 80, file_path.c_str());
  m_lena_segmentation_seperate_front_back = atoi(in_buf);

  str_sectionKey = "lena_segmentation_by_board";
  GetPrivateProfileString(str_section.c_str(),
                          str_sectionKey.c_str(),
                          NULL, in_buf, 80, file_path.c_str());
  m_lena_segmentation_by_board = atoi(in_buf);

  str_section = "resolution";
  str_sectionKey = "resolution_choose";
  GetPrivateProfileString(&str_section[0], &str_sectionKey[0],
                          NULL, in_buf, 80, &file_path[0]);
  m_resolution_choose = atoi(in_buf);

  switch (m_resolution_choose) {
  case 0:
    m_s_width = 1680;
    m_s_height = 1050;
    m_block_X = 16;
    m_block_Y = 25;
    m_grid_X = 105;
    m_grid_Y = 42;
    break;
  case 1:
    m_s_width = 1600;
    m_s_height = 900;
    m_block_X = 64;
    m_block_Y = 4;
    m_grid_X = 25;
    m_grid_Y = 225;
    break;
  case 2:
    m_s_width = 1440;
    m_s_height = 900;
    m_block_X = 32;
    m_block_Y = 4;
    m_grid_X = 45;
    m_grid_Y = 225;
    break;
  case 3:
    m_s_width = 1024;
    m_s_height = 1024;
    m_block_X = 32;
    m_block_Y = 8;
    m_grid_X = 32;
    m_grid_Y = 128;
    break;
  case 4:
    m_s_width = 640;
    m_s_height = 480;
    m_block_X = 32;
    m_block_Y = 8;
    m_grid_X = 20;
    m_grid_Y = 40;
    break;
  case 5:
    m_s_width = 512;
    m_s_height = 512;
    m_block_X = 16;
    m_block_Y = 16;
    m_grid_X = 32;
    m_grid_Y = 32;
    break;
  case 6:
    m_s_width = 2048;
    m_s_height = 2048;
    m_block_X = 32;
    m_block_Y = 8;
    m_grid_X = 64;
    m_grid_Y = 256;
    break;
  default:
    m_s_width = 256;
    m_s_height = 256;
    m_block_X = 32;
    m_block_Y = 4;
    m_grid_X = 8;
    m_grid_Y = 64;
  }

  str_section = "improved_canny";
  str_sectionKey = "canny_enable";
  GetPrivateProfileString(str_section.c_str(), str_sectionKey.c_str(),
                          NULL, in_buf, 80, file_path.c_str());
  m_canny_enable = atoi(in_buf);

  str_sectionKey = "heighthreshold";
  GetPrivateProfileString(&str_section[0], &str_sectionKey[0], NULL,
                          in_buf, 80, &file_path[0]);
  m_highthreshold = atoi(in_buf);

  str_sectionKey = "lowthreshold";
  GetPrivateProfileString(&str_section[0], &str_sectionKey[0], NULL,
                          in_buf, 80, &file_path[0]);
  m_lowthreshold = atoi(in_buf);

  str_sectionKey = "recursion_num";
  GetPrivateProfileString(&str_section[0], &str_sectionKey[0], NULL,
                          in_buf, 80, &file_path[0]);
  m_recursion_num = atoi(in_buf);

  str_section = "edge_connection";
  str_sectionKey = "edge_connection_enable";
  GetPrivateProfileString(&str_section[0], &str_sectionKey[0], NULL,
                          in_buf, 80, &file_path[0]);
  m_edge_connection_enable = atoi(in_buf);

  str_sectionKey = "min_L";
  GetPrivateProfileString(&str_section[0], &str_sectionKey[0], NULL,
                          in_buf, 80, &file_path[0]);
  m_min_l = atoi(in_buf);

  str_sectionKey = "max_L";
  GetPrivateProfileString(&str_section[0], &str_sectionKey[0], NULL,
                          in_buf, 80, &file_path[0]);
  m_max_l = atoi(in_buf);

  str_sectionKey = "del_L";
  GetPrivateProfileString(&str_section[0], &str_sectionKey[0], NULL,
                          in_buf, 80, &file_path[0]);
  m_del_l = atoi(in_buf);

  str_section = "segmentation_choose";
  str_sectionKey = "segmentation_enable";
  GetPrivateProfileString(&str_section[0], &str_sectionKey[0], NULL,
                          in_buf, 80, &file_path[0]);
  m_segmentation_enable = atoi(in_buf);

  str_sectionKey = "segmentation_by_board_or_by_length";
  GetPrivateProfileString(&str_section[0], &str_sectionKey[0], NULL,
                          in_buf, 80, &file_path[0]);
  m_segmentation_by_board_or_by_length = atoi(in_buf);

  str_section = "segmentation_by_board";
  string str_sectionKey_original("board_combination_element_");
  char num[5];
  for (int i = 0; i < 16; i++) {
    _itoa_s(i, num, 5, 10);
    string temp(num);
    str_sectionKey = str_sectionKey_original + temp;
    GetPrivateProfileString(&str_section[0], &str_sectionKey[0], NULL,
                            in_buf, 80, &file_path[0]);
    m_board_combination[i] = atoi(in_buf);
  }

  str_section = "segmentation_by_length";
  str_sectionKey_original = "length_parameter_";
  for (int i = 0; i < 16; i++) {
    _itoa_s(i, num, 5, 10);
    string temp(num);
    str_sectionKey = str_sectionKey_original + temp;
    GetPrivateProfileString(&str_section[0], &str_sectionKey[0], NULL,
                            in_buf, 80, &file_path[0]);
    m_length_parameter[i] = atoi(in_buf);
  }
}
