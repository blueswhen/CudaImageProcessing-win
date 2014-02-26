// Copyright 2013-10 sxniu
#ifndef IMAGEPROCESSING_INCLUDE_CONSTVALUE_H_
#define IMAGEPROCESSING_INCLUDE_CONSTVALUE_H_

#define WHITE 0x00ffffff
// 16 colours
#define COLOUR_RED 0x00ff0000
#define COLOUR_YELLOW 0x00ffff00
#define COLOUR_PURPLE 0x00ff00ff
#define COLOUR_CYAN 0x0000ffff
#define COLOUR_GREEN 0x0000ff00
#define COLOUR_BLUE 0x000000ff
#define COLOUR_LIGHT_GRAY 0x00f0f0f0
#define COLOUR_LIGHT_BLUE 0x000080c0
#define COLOUR_PURPLE_RED 0x00800040
#define COLOUR_LIGHT_PURPLE 0x008000ff
#define COLOUR_DARK_YELLOW 0x00ff8000
#define COLOUR_LIGHT_GREEN 0x00008040
#define COLOUR_DARK_GREEN 0x00004000
#define COLOUR_DARK_BLUE 0x00000080
#define COLOUR_DARK_RED 0x00800000
#define COLOUR_GRAY 0x00808080

// using these colour in edge description region filling
#define FILLING_COL COLOUR_RED
#define BEGIN_SCAN COLOUR_YELLOW
#define SKIP_SCAN COLOUR_PURPLE
#define END_SCAN COLOUR_CYAN
#define SCANED COLOUR_GREEN
#define INSIDE_EDGE COLOUR_BLUE
// #define POINTS_NUM 1000

// the minimum integer which bigger than 3*kSigma
#define THREE_SIGMA_INT 1

// paremeters in four direction scan
#define SCAN_MAX_LENGTH 5000
#define SCAN_MAX_REPEAT 5

#define COUNT_TIME_ENABLE 0

// thansport the "exist" variable in device to host to identify
// whether the filling operation finished.
// the disadvantage of the method is the transportion spending too much time
// #define FOURDIRECTIONSCAN_WITH_FEEDBACK

// #define PROCESS_IMAGE_BY_LOCK_SURFACE

const int kColourArray[16] = {COLOUR_RED, COLOUR_YELLOW, COLOUR_PURPLE,
                              COLOUR_CYAN, COLOUR_GREEN, COLOUR_BLUE,
                              COLOUR_LIGHT_GRAY, COLOUR_LIGHT_BLUE,
                              COLOUR_PURPLE_RED, COLOUR_LIGHT_PURPLE,
                              COLOUR_DARK_YELLOW, COLOUR_LIGHT_GREEN,
                              COLOUR_DARK_GREEN, COLOUR_DARK_BLUE,
                              COLOUR_DARK_RED, COLOUR_GRAY};

const double kSigma = 0.3;
const int kWindowSize = 1 + 2 * THREE_SIGMA_INT;

// the four board of image
enum BoardType {
  BOARD_UP = 0,
  BOARD_DOWN,
  BOARD_LEFT,
  BOARD_RIGHT,
  BOARD_ALL,
};

extern int g_block_x;
extern int g_block_y;
extern int g_grid_x;
extern int g_grid_y;
#endif  // IMAGEPROCESSING_INCLUDE_CONSTVALUE_H_
