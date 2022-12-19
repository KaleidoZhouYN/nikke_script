#include "win.h"

int main(){
    const char[] keywords = "MuMu";
    auto hWnd = GetHWndByName(windowName = keywords);

    cv::Mat img = GetScreenshotByHWnd(hWnd,1);
    cv::imshow('screenshot',img);

    cv::waitKey(0);
    return 1;
}