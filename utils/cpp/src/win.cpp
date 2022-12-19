#ifndef _WIN_H
#define _WIN_H
#endif

#include "win.h"

BOOL CALLBACK enumWindowsProc(HWND hWnd, LPARAM lParam){
	// convert lparam back to vector<string>
	std::vector<std::string>& windowTitles = *reinterpret_cast<std::vector<std::string>*>(lParam);

	int length = GetWindowTextLength(hWnd);
	char* buffer = new char[length + 1];
	GetWindowText(hWnd, buffer, length + 1);
	std::string windowTitle(buffer);
	
	delete[] buffer;

	if (IsWindowVisible(hWnd) && length != 0) {
		windowTitles.push_back(windowTitle);
	}
	return TRUE;
}

HWND GetHWndByName(const char* windowName) {
	std::vector<std::string> windowTitles;
	EnumWindows(&enumWindowProc, reinterpret_cast<LPARAM>(&windowTitles));

	HWND hWnd = -1; 
	for (auto windowTitle : windowTitles) {
		if (windowTitle.find(windowName) != std::string::npos) {
			hWnd = FindWindow(windowTitle);
		}
	}
	if (hWnd == -1) {
		MessageBox(0, "your app with keyword" + windowName + " not found", "Error", 0);
	}

	return hWnd
}

cv::Mat GetScreenshotByHWnd(HWND hWnd,bool isBackground) {
	SetForegroundWindow(hWnd);
	LPRECT lp_rect = new RECT;
	GetWindowRect(hWnd,lp_rect);
	int w = lp_rect->right - lp_rect->left; 
	int h = lp_rect->buttom - lp_rect->top; 

	auto hWndDC = GetWindowDC(hWnd);
	auto mfcDC = CreateDCFromHandle(hWndDC);
	auto saveDC = CreateCompatibleDC(mfcDC);

	/***
	auto hSaveBitMap = CreateBitmap(nWidth = w, nHeight = h,
		nPlanes = 3, nBitCount = 24,
		lpBits = NULL);
	***/

	// ususally 24 bits for mfcDC
	auto hSaveBitMap = CreateCompatibleBitmap(hdc = mfcDC, cx = w, cy = h);
	hSaveBitMap = SelectObject(saveDC, hSaveBitMap);

	PrintWindow(hWnd, saveDC, PW_CLIENTONLY);

	LPVOID lpvBits;
	GetBitmapBits(hSaveBitMap, h * w * 3 * sizeof(unsigned char), lpvBits);

	cv::Mat img = cv::Mat(h, w, CV_8UC3, (void*)lpvBits);
	cv::resize(img, img, dsize = Size(w, h), interpolation = cv::INTER_LINEAR);
	DeleteObject(hSaveBitMap);
	DeleteDC(saveDC);
	DeleteDC(mfcDC);
	ReleaseDC(hWnd, hWndDC);
}
