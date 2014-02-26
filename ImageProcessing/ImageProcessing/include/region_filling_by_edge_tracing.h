// Copyright 2013-10 sxniu
#ifndef IMAGEPROCESSING_INCLUDE_REGIONFILLINGBYEDGETRACING_H_
#define IMAGEPROCESSING_INCLUDE_REGIONFILLINGBYEDGETRACING_H_

#include <vector>

namespace region_filling_by_edge_tracing {

// set the order of searching for edge tracing
void SetSearchOrder(int* index, int index_cen,
                    int previous_index_cen, int image_width);
// return the change of the angle when edge tracing
int GetAngle(int index_cen, int arround_index, int image_width);
int FindNextPoint(int* image, int previous_index_cen,
                  int index_cen, int image_width,
                  int* next_index_cen, int* delt_angle);

// paint the colour for start points and end points
void SetColourForPoints(int* image, int* tmp_array,
                        int previous_index_cen, int index_cen,
                        int next_index_cen, int image_width,
                        std::vector<int>* start_points,
                        std::vector<int>* end_points);
void Filling(int* image, const std::vector<int>& start_points,
             const std::vector<int>& end_points, int clockwise);
void DoRegionFillingByEdgeTracing(int* image, int x, int y,
                                  int image_width, int image_height,
                                  int* k);
void RegionFillingByEdgeTracing(int* image, int image_width,
                                int image_height, int k_size);

}  // namespace region_filling_by_edge_tracing

#endif  // IMAGEPROCESSING_INCLUDE_REGIONFILLINGBYEDGETRACING_H_
