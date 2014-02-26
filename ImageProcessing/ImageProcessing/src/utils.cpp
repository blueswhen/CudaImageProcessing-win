// Copyright 2013-10 sxniu
#include "include/utils.h"
#include <stdlib.h>
#include <windows.h>
#include <time.h>
#include <math.h>
#include <string>
#include <algorithm>
#include "include/ConstValue.h"

namespace utils {

// start_or_end = 1 begin, = 0 end
void ShowTime(double *freq, double *start_time,
              int *time, int start_or_end, int count_enable) {
  LARGE_INTEGER T;
  if (start_or_end && count_enable) {
    QueryPerformanceFrequency(&T);  // get frequence
    *freq = static_cast<double>(T.QuadPart);
    QueryPerformanceCounter(&T);  // count begin
    *start_time = static_cast<double>(T.QuadPart);
  } else if (count_enable) {
    QueryPerformanceCounter(&T);  // count end
    double end_time = static_cast<double>(T.QuadPart);
    *time = static_cast<int>(1000000 * (end_time - *start_time) / *freq);
    // char num[32];
#if 1
    std::string time_num;
    char num[32];
    _itoa_s(*time, num, 32, 10);
    time_num = num;
    time_num = time_num + "us";
    MessageBox(NULL, &time_num[0], "TIME", NULL);
#endif
  }
}

void ShowNum(int num) {
  std::string i_str;
  char str[1024];
  _itoa_s(num, str, 1024, 10);
  i_str = str;
  MessageBox(NULL, i_str.c_str(), NULL, NULL);
}

// Thinning alogrithm
int CountAroundPoints(int* imageIn, int* index_arround,
                      int* arroundPoints, int* arroundBlacks) {
  int num = 0;
  int k = 0;
  for (int i = 0; i < 8; i++) {
    if (imageIn[index_arround[i]] == WHITE) {
      arroundPoints[num] = index_arround[i];
      num++;
    } else {
      arroundBlacks[k] = index_arround[i];
      k++;
    }
  }
  return num;
}

int CenterRemovedArrs(int point0, int point1, int image_width) {
  int x0 = point0 % image_width;
  int y0 = point0 / image_width;
  int x1 = point1 % image_width;
  int y1 = point1 / image_width;

  int dx = x0 - x1 > 0 ? x0 - x1 : x1 - x0;
  int dy = y0 - y1 > 0 ? y0 - y1 : y1 - y0;

  if (dx < 2 && dy < 2)
    return 1;
  else
    return 0;
}

int CenterRemovedArrs(int point0, int point1, int point2, int image_width) {
  if (CenterRemovedArrs(point0, point1, image_width)) {
    if (CenterRemovedArrs(point0, point2, image_width))
      return 1;
    else if (CenterRemovedArrs(point1, point2, image_width))
      return 1;
    else
      return 0;
  } else if (CenterRemovedArrs(point0, point2, image_width)) {
    if (CenterRemovedArrs(point1, point2, image_width))
      return 1;
    else
      return 0;
  } else {
    return 0;
  }
}

int CenterRemovedArrs(int point0, int point1, int point2,
                      int point3, int image_width) {
  if (CenterRemovedArrs(point0, point1, point2, image_width)) {
    if (CenterRemovedArrs(point0, point3, image_width))
      return 1;
    else if (CenterRemovedArrs(point1, point3, image_width))
      return 1;
    else if (CenterRemovedArrs(point2, point3, image_width))
      return 1;
    else
      return 0;
  } else if (CenterRemovedArrs(point0, point1, point3, image_width)) {
    if (CenterRemovedArrs(point0, point2, image_width))
      return 1;
    else if (CenterRemovedArrs(point1, point2, image_width))
      return 1;
    else if (CenterRemovedArrs(point3, point2, image_width))
      return 1;
    else
      return 0;
  } else if (CenterRemovedArrs(point0, point2, point3, image_width)) {
    if (CenterRemovedArrs(point0, point1, image_width))
      return 1;
    else if (CenterRemovedArrs(point2, point1, image_width))
      return 1;
    else if (CenterRemovedArrs(point3, point1, image_width))
      return 1;
    else
      return 0;
  } else {
    return 0;
  }
}

int CenterRemovedArrsB(int black0, int black1, int index_cen, int image_width) {
  if (black0 == index_cen - image_width) {
    if (black1 == index_cen + image_width)
      return 0;
    else if (black1 == index_cen + 1)
      return 0;
    else if (black1 == index_cen - 1)
      return 0;
    else
      return 1;
  } else if (black0 == index_cen + image_width) {
    if (black1 == index_cen - image_width)
      return 0;
    else if (black1 == index_cen + 1)
      return 0;
    else if (black1 == index_cen - 1)
      return 0;
    else
      return 1;
  } else if (black0 == index_cen + 1) {
    if (black1 == index_cen + image_width)
      return 0;
    else if (black1 == index_cen - image_width)
      return 0;
    else if (black1 == index_cen - 1)
      return 0;
    else
      return 1;
  } else if (black0 == index_cen -1) {
    if (black1 == index_cen + image_width)
      return 0;
    else if (black1 == index_cen - image_width)
      return 0;
    else if (black1 == index_cen + 1)
      return 0;
    else
      return 1;
  } else {
    return 1;
  }
}

int CenterRemovedArrsB(int black0, int black1, int black2,
                       int index_cen, int image_width) {
  if (!CenterRemovedArrsB(black0, black1, index_cen, image_width)) {
    int dy = (black0 - black1 > 0 ? black0 - black1 : black1 - black0) /
             image_width;
    if (dy == 1) {
      int dx0 = (black2 - black0 > 0 ? black2 - black0 : black0 - black2) %
                image_width;
      int dx1 = (black2 - black1 > 0 ? black2 - black1 : black1 - black2) %
                image_width;
      int dy0 = (black2 - black0 > 0 ? black2 - black0 : black0 - black2) /
                image_width;
      int dy1 = (black2 - black1 > 0 ? black2 - black1 : black1 - black2) /
                image_width;
      if (dx0 < 2 && dx1 < 2 && dy0 < 2 && dy0 < 2)
        return 1;
      else
        return 0;
    } else {
      return 0;
    }
  } else if (!CenterRemovedArrsB(black0, black2, index_cen, image_width)) {
    int dy = (black0 - black2 > 0 ? black0 - black2 : black2 - black0) /
             image_width;
    if (dy == 1) {
      int dx0 = (black1 - black0 > 0 ? black1 - black0 : black0 - black1) %
                image_width;
      int dx1 = (black1 - black2 > 0 ? black1 - black2 : black2 - black1) %
                image_width;
      int dy0 = (black1 - black0 > 0 ? black1 - black0 : black0 - black1) /
                image_width;
      int dy1 = (black1 - black2 > 0 ? black1 - black2 : black2 - black1) /
                image_width;
      if (dx0 < 2 && dx1 < 2 && dy0 < 2 && dy0 < 2)
        return 1;
      else
        return 0;
    } else {
      return 0;
    }
  } else if (!CenterRemovedArrsB(black1, black2, index_cen, image_width)) {
    int dy = (black1 - black2 > 0 ? black1 - black2 : black2 - black1) /
             image_width;
    if (dy == 1) {
      int dx0 = (black0 - black1 > 0 ? black0 - black1 : black1 - black0) %
                image_width;
      int dx1 = (black0 - black2 > 0 ? black0 - black2 : black2 - black0) %
                image_width;
      int dy0 = (black0 - black1 > 0 ? black0 - black1 : black1 - black0) /
                image_width;
      int dy1 = (black0 - black2 > 0 ? black0 - black2 : black2 - black0) /
                image_width;
      if (dx0 < 2 && dx1 < 2 && dy0 < 2 && dy0 < 2)
        return 1;
      else
        return 0;
    } else {
      return 0;
    }
  } else {
    return 1;
  }
}

int CenterRemovedWithArrs(int* arroundPoints, int* arroundBlacks,
                          int size, int index_cen, int image_width) {
  if (size < 1 || size > 6)
    return 1;
  if (size == 1)
    return 0;
  switch (size) {
    case 2:
      return CenterRemovedArrs(arroundPoints[0], arroundPoints[1], image_width);
    case 3:
      return CenterRemovedArrs(arroundPoints[0], arroundPoints[1],
                               arroundPoints[2], image_width);
    case 4:
      return CenterRemovedArrs(arroundPoints[0], arroundPoints[1],
                               arroundPoints[2], arroundPoints[3], image_width);
    case 5:
      return CenterRemovedArrsB(arroundBlacks[0], arroundBlacks[1],
                                arroundBlacks[2], index_cen, image_width);
    case 6:
      return CenterRemovedArrsB(arroundBlacks[0], arroundBlacks[1],
                                index_cen, image_width);
    default:
      return 1;
  }
}

void DoThinning(int* image, int x, int y, int image_width, int image_height) {
  int index_cen = y * image_width + x;
  int index[8];
  index[0] = max(y - 1, 0) * image_width + max(x - 1, 0);
  index[1] = max(y - 1, 0) * image_width + x;
  index[2] = max(y - 1, 0) * image_width + min(x + 1, image_width);
  index[3] = y * image_width + min(x + 1, image_width);
  index[4] = min(y + 1, image_height) * image_width + min(x + 1, image_width);
  index[5] = min(y + 1, image_height) * image_width + x;
  index[6] = min(y + 1, image_height) * image_width + max(x - 1, 0);
  index[7] = y * image_width + max(x - 1, 0);

  if (image[index_cen] == WHITE) {
    int arroundPoints[8];
    int arroundBlacks[8];
    int pointsNum = 0;
    pointsNum = CountAroundPoints(image, index, arroundPoints, arroundBlacks);
    int removed = 0;
    switch (pointsNum) {
      case 0:
      case 1:
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
        removed = CenterRemovedWithArrs(arroundPoints, arroundBlacks,
                                        pointsNum, index_cen, image_width);
        if (removed) {
          image[index_cen] = 0;
          return;
        }
        break;
      default:
        image[index_cen] = 0;
        return;
    }
  }
}

void Thinning(int* image, int image_width, int image_height) {
  for (int y = 1; y < image_height - 1; y++) {
    for (int x = 1; x < image_width - 1; x++) {
      DoThinning(image, x, y, image_width, image_height);
    }
  }
}

size_t BigRand() {
  srand(static_cast<size_t>(time(NULL)));
  return static_cast<size_t>(rand() * RAND_MAX + rand());
}

// [begin, end)
size_t GenRanNumFromRange(size_t begin, size_t end) {
  if (begin < end) {
      return BigRand() % (end - begin) + begin;
  } else {
      return 0; // error happened
  }
}

void GenRanArray(int* test_array, size_t size) {
  for (size_t i = 0; i < size; i++) {
      test_array[i] = i;
  }
  for (size_t i = 0; i < size; i++) {
      size_t pos = GenRanNumFromRange(i, size);
      std::swap(test_array[i], test_array[pos]);
  }
}

void MergeArray(int* test_array, int* tmp_array,
                size_t left_pos, size_t center_pos, size_t right_pos) {
  int* array_front_half = test_array + left_pos;
  int* array_after_half = test_array + center_pos;

  size_t front_half_size = center_pos - left_pos;
  size_t after_half_size = right_pos - center_pos + 1;

  size_t ptr_front_array = 0;
  size_t ptr_after_array = 0;
  size_t ptr_tmp_array = 0;

  while (ptr_front_array <= front_half_size - 1 &&
         ptr_after_array <= after_half_size - 1) {
    if (array_front_half[ptr_front_array] <
        array_after_half[ptr_after_array]) {
      tmp_array[ptr_tmp_array++] = array_front_half[ptr_front_array++];
    } else {
      tmp_array[ptr_tmp_array++] = array_after_half[ptr_after_array++];
    }
  }

  while (ptr_front_array < front_half_size) {
    tmp_array[ptr_tmp_array++] = array_front_half[ptr_front_array++];
  }

  while (ptr_after_array < after_half_size) {
    tmp_array[ptr_tmp_array++] = array_after_half[ptr_after_array++];
  }

  for (size_t i = left_pos; i < right_pos + 1; i++) {
    test_array[i] = tmp_array[i - left_pos];
  }
}

void MergeSort(int* test_array, int* tmp_array, size_t left_pos, size_t right_pos) {
  if (left_pos < right_pos) {
    size_t center = (right_pos + left_pos) / 2;
    MergeSort(test_array, tmp_array, left_pos, center);
    // must center + 1, otherwise segment fault
    MergeSort(test_array, tmp_array, center + 1, right_pos);
    MergeArray(test_array, tmp_array, left_pos, center + 1, right_pos);
  }
}

void MergeSort(int* test_array, size_t size) {
  int* tmp_array = new int[size];
  MergeSort(test_array, tmp_array, 0, size - 1);
  delete [] tmp_array;
}

void Sum(int* test_array, size_t size) {
  int sum = 0;
  double freq = 0;
  double start_time = 0;
  int time = 0;
  ShowTime(&freq, &start_time, &time, 1, COUNT_TIME_ENABLE);
  for (size_t i = 0; i < size; ++i) {
    sum += test_array[i];
  }
  ShowTime(&freq, &start_time, &time, 0, COUNT_TIME_ENABLE);
  printf("sum = %d\n", sum);
}

void TraceEdge (int y, int x, int nLowThd,int *pUnchEdge, int *pnMag, int nWidth) 
{ 
  // 对8邻域象素进行查询
  int xNb[8] = {1, 1, 0,-1,-1,-1, 0, 1};
  int yNb[8] = {0, 1, 1, 1,0 ,-1,-1,-1};
  int yy;
  int xx;
  for(int k=0; k<8; k++)
  {
    yy = y + yNb[k] ;
    xx = x + xNb[k] ;
    // 如果该象素为可能的边界点，又没有处理过
    // 并且梯度大于阈值
    if(pUnchEdge[yy*nWidth+xx] == 0x000000ff  && pnMag[yy*nWidth+xx]>=nLowThd)
    {
      // 把该点设置成为边界点
      pUnchEdge[yy*nWidth+xx] = 0x00ffffff;

      // 以该点为中心进行跟踪
      TraceEdge(yy, xx, nLowThd, pUnchEdge, pnMag, nWidth);
    }
  }
}

void Sobel(int* source_image, int* edge_image, int image_width,
		   int image_height, int threshold) {
  for (int y = 1; y < image_height - 1; ++y) {
    for (int x = 1; x < image_width - 1; ++x) {
      int index_cen = y * image_width + x;
      int index[8];
      index[0] = (y - 1) * image_width + x - 1;
      index[1] = (y - 1) * image_width + x + 1;
      index[2] = (y + 1) * image_width + x + 1;
      index[3] = (y + 1) * image_width + x - 1;
      index[4] = (y - 1) * image_width + x;
      index[5] = y * image_width + x - 1;
      index[6] = y * image_width + x + 1;
      index[7] = (y + 1) * image_width + x;

      double gx = -(source_image[index[0]] & 0x000000ff) + (source_image[index[1]] & 0x000000ff) -
		       2 * (source_image[index[5]] & 0x000000ff) + 2 * (source_image[index[6]] & 0x000000ff) -
               (source_image[index[3]] & 0x000000ff) + (source_image[index[2]] & 0x000000ff);

      double gy = (source_image[index[0]] & 0x000000ff) + 2 * (source_image[index[4]] & 0x000000ff) +
               (source_image[index[1]] & 0x000000ff) - (source_image[index[3]] & 0x000000ff) -
               2 * (source_image[index[7]] & 0x000000ff) - (source_image[index[2]] & 0x000000ff);

      double sum_of_squares = pow(gx, 2) + pow(gy, 2);
	  int dst_gray = static_cast<int>(sqrt(sum_of_squares));
      if (dst_gray > threshold)
        edge_image[index_cen] = WHITE;
	  else
        edge_image[index_cen] = 0;
    }
  }
}

void Canny(int* source_image, int* edge_image, int image_width,
           int image_height, int low_threshold, int height_threshold) {
#if 0
  // turn gray
  for (int y = 0; y < image_width; ++y) 
  {
    for (int x = 0; x < image_width; ++x) 
    {
      int index = y * image_width + x;  ///原图像素坐标
      int red,green,blue;
      red = (source_image[index]&0x00ff0000)>>16;
      green = (source_image[index]&0x0000ff00)>>8;
      blue = source_image[index]&0x000000ff;
      int gray;
      gray = (int)(red*0.3 + green*0.59 + blue*0.11);
      gray = (gray<<16) + (gray<<8) + gray;
      source_image[index] = gray;
     }    
  }
#endif
#if 1
  static int* gauss_mem = new int[image_width * image_height];
  static double* pdTmp = new double[image_width * image_height];

  int nCenter = kWindowSize / 2;;
  double  dSum = 0;
  double gaussnum[1 + 2 * THREE_SIGMA_INT];
  for(int i=0; i< kWindowSize; i++)
  {
    const double PI = 3.1415926535897932384626;
    double dDis = (double)(i - nCenter);
    double dValue = exp(-(1/2)*dDis*dDis/(kSigma*kSigma))/(sqrt(2 * PI) * kSigma );
    gaussnum[i] = (double)dValue;
    dSum += dValue;
  }
  for(int i=0; i<kWindowSize; i++)
  {
    gaussnum[i] /= (double)dSum;
  }
#if 0
  for(int i=0; i<windowsize; i++)
  {
    (*dWeightSum) += gaussnum[i];
  }
#endif
  double dWeightSum = 0;
  // x方向进行滤波
  for(int y=1; y<image_height-1; y++)
  {
    for(int x=1; x<image_width-1; x++)
    {
      double dDotMul  = 0;
      dWeightSum = 0;
      for(int i=(-nCenter); i<=nCenter; i++)
      {
        dWeightSum += gaussnum[nCenter+i];
        int gray = source_image[y*image_width + (i+x)]&0x000000ff;  ///提取灰度值在进行高斯平滑
        dDotMul += (double)gray * gaussnum[nCenter+i];
      }
      pdTmp[y*image_width + x] = (dDotMul/dWeightSum);  ///将x方向平滑的数据保存在零时数组中
    }
  }

  // y方向进行滤波
  for(int y=1; y<image_height - 1; y++)
  {
    for(int x=1; x<image_width - 1; x++)
    {
      double dDotMul  = 0;
      for(int i=(-nCenter); i<=nCenter; i++)
      {
        dDotMul += pdTmp[(y+i)*image_width + x] * gaussnum[nCenter+i];
      }
      int des_gray = ((int)(dDotMul/dWeightSum))&0x000000ff;  ///将最终的数据转换成灰度数据
      des_gray = (des_gray<<16) + (des_gray<<8) + des_gray;
      gauss_mem[y*image_width + x] = des_gray;
    }
  }
#endif
  // int* gauss_mem = source_image;
  static double *gradX_mem = new double[image_width * image_height];
  static double *gradY_mem = new double[image_width * image_height];
  static int *mag_mem = new int[image_width * image_height];
  int pixel_width = image_width;
  int height = image_height;
  for (int y = 1; y < image_height-1; ++y ) 
  {
    for (int x = 1; x < image_width-1; ++x ) 
    {
        int index = y * pixel_width + x;
        //计算x方向的方向导数，在边界出进行了处理，防止要访问的象素出界
        int gray_x1 = (gauss_mem[y*pixel_width+min(pixel_width-1,x+1)]&0x000000ff);
        int gray_x2 = (gauss_mem[y*pixel_width+max(0,x-1)]&0x000000ff);
        gradX_mem[index] = (double)(gray_x1 - gray_x2);
      
        //计算y方向的方向导数，在边界出进行了处理，防止要访问的象素出界
        int gray_y1 = (gauss_mem[min(height-1,y+1)*pixel_width + x]&0x000000ff);
        int gray_y2 = (gauss_mem[max(0,y-1)*pixel_width+ x ]&0x000000ff);
        gradY_mem[index] = (double)(gray_y1 - gray_y2);
      
        mag_mem[index] = (int)(sqrt(gradX_mem[index]*gradX_mem[index] + gradY_mem[index]*gradY_mem[index]) + 0.5);
    //    mag_mem[index] = (mag_mem[index]<<16) + (mag_mem[index]<<8 + mag_mem[index]);
    //    mag_mem[index] = gray_x1>gray_x2 ? 0x00ffffff:0;//gradX_mem[index];
    }
  }

  int* pre_edge_mem = edge_image;
  for (int y = 0; y < image_height; ++y ) 
  {
    for (int x = 0; x < image_width; ++x ) 
    {
      int index = y * pixel_width + x;
      
      // x方向梯度分量
      double gx  ;
      double gy  ;
      
      // 临时变量
      int g1, g2, g3, g4 ;
      double weight  ;
      double dTmp1   ;
      double dTmp2   ;
      double dTmp    ;
   
          // 如果当前象素的梯度幅度为0，则不是边界点
      if(mag_mem[index] == 0 )
      {
        pre_edge_mem[index] = 0 ;
      }
      else
      {
        // 当前象素的梯度幅度
        dTmp = (double)mag_mem[index] ;
        
        // x，y方向导数
        gx = (double)gradX_mem[index] ;
        gy = (double)gradY_mem[index] ;
      
        // 如果方向导数y分量比x分量大，说明导数的方向更加“趋向”于y分量。
        if (fabs(gy) > fabs(gx)) 
        {
          // 计算插值的比例
          weight = fabs(gx)/fabs(gy); 
          g2 = mag_mem[index-pixel_width] ; 
          g4 = mag_mem[index+pixel_width] ;
          
          // 如果x，y两个方向的方向导数的符号相同
          // C是当前象素，与g1-g4的位置关系为：
          //   g1 g2
          //     C         
          //     g4 g3 
          if (gx*gy > 0) 
          {           
            g1 = mag_mem[index-pixel_width-1] ;
            g3 = mag_mem[index+pixel_width+1] ;
          } 
          // 如果x，y两个方向的方向导数的符号相反
          // C是当前象素，与g1-g4的位置关系为：
          //       g2 g1
          //     C         
          //  g3 g4  
          else 
          { 
            g1 = mag_mem[index-pixel_width+1] ;
            g3 = mag_mem[index+pixel_width-1] ;
          } 
        }            
        // 如果方向导数x分量比y分量大，说明导数的方向更加“趋向”于x分量
        // 这个判断语句包含了x分量和y分量相等的情况
        else
        {
          // 计算插值的比例
          weight = fabs(gy)/fabs(gx); 
          
          g2 = mag_mem[index+1] ; 
          g4 = mag_mem[index-1] ;
        
          // 如果x，y两个方向的方向导数的符号相同
          // C是当前象素，与g1-g4的位置关系为：
          //  g3   
          //  g4 C g2       
          //       g1
          if (gx*gy > 0) 
          {        
            g1 = mag_mem[index+pixel_width+1] ;
            g3 = mag_mem[index-pixel_width-1] ;
          } 
          // 如果x，y两个方向的方向导数的符号相反
          // C是当前象素，与g1-g4的位置关系为：
          //       g1
          //  g4 C g2       
          //  g3     
          else 
          { 
            g1 = mag_mem[index-pixel_width+1] ;
            g3 = mag_mem[index+pixel_width-1] ;
          }
        }
        // 下面利用g1-g4对梯度进行插值
        
        dTmp1 = weight * (double)g1 + (1-weight) * (double)g2 ;
        dTmp2 = weight * (double)g3 + (1-weight) * (double)g4 ;
          
        // 当前象素的梯度是局部的最大值
        // 该点可能是个边界点
        if(dTmp>=dTmp1 && dTmp>=dTmp2)
        {
          pre_edge_mem[index] = 0x000000ff;
        }
        else
        {
          // 不可能是边界点
          pre_edge_mem[index] = 0;
        }
      }
    }
  }

  int edge_min = height_threshold;
  int edge_max = 200;
  for (int y = 0; y < image_height; ++y ) 
  {
    for (int x = 0; x < image_width; ++x ) 
    {
      int index = y*pixel_width + x;
       
      if((pre_edge_mem[index] == 0x000000ff) && (mag_mem[index] >= edge_min) && (mag_mem[index] <= edge_max))
      {
        pre_edge_mem[index] = 0x00ffffff;
        TraceEdge (y,x,low_threshold,pre_edge_mem,mag_mem,pixel_width);
      }
    }
  }
  for (int y = 0; y < image_height; ++y ) 
  {
    for (int x = 0; x < image_width; ++x ) 
    {
      int index = y*pixel_width + x;
       
      if(pre_edge_mem[index] != 0x00ffffff)
      {
        pre_edge_mem[index] = 0;
      }
    }
  }
}

}  // namespace utils
