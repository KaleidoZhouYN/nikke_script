#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <winuser.h>
#include <windef.h>
#include <wingdi.h>

#ifndef _WIN_H
#define _WIN_H
BOOL CALLBACK enumWindowsProc(HWND, LPARAM);

HWND GetHWndByName(const char*);

cv::Mat GetScreenshotByHWnd(HWND,bool);

#endif