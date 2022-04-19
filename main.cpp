#include "opencv2/opencv.hpp"
#include "common.h"
#include "slider.h"
#include <cmath>
#include <fstream>
#include <ranges>
#include <functional>
#include <tuple>

#include "spaces.h"
#include "skin_detector.h"

#define CAMERA "Camera"
#define OUTPUT "Processed"
#define OPEN_CLOSE_KERNEL "Open/Close Kernel"
#define GAUSSIAN_BLUR_KERNEL "Blur Kernel"
#define EROSION_KERNEL "Erode Kernel"

SkinDetector* fromRaw(void *rawDetector)
{
  ASSERT(rawDetector != nullptr, "There is no skin detector around here");
  return (SkinDetector *)rawDetector;
}

int main() {
  Logger::init();

  // Create the skin detector
  SkinDetector detect;

  // Create the windows
  cv::namedWindow(CAMERA);
  cv::namedWindow(OUTPUT);

  // Assign slider controls for the skin detector controls
  cv::createTrackbar(
    OPEN_CLOSE_KERNEL, OUTPUT,
    &detect.openCloseKernel, 10,
    [](int pos, void *r) {
      SkinDetector *d = fromRaw(r);
      d->openCloseKernel = pos;
    },
    (void*) &detect
  );
  cv::createTrackbar(
    GAUSSIAN_BLUR_KERNEL, OUTPUT,
    &detect.blurKernel, 3,
    [](int pos, void *r) {
      SkinDetector *d = fromRaw(r);
      d->blurKernel = pos;
    },
    (void*) &detect
  );
  cv::createTrackbar(
    EROSION_KERNEL, OUTPUT,
    &detect.erodeKernel, 3,
    [](int pos, void *r) {
      SkinDetector *d = fromRaw(r);
      d->erodeKernel = pos;
    },
    (void*) &detect
  );

  // Create instance of VideoCapture
  // cv::VideoCapture capture(VIDEO("test.mp4"));
  // cv::VideoCapture capture(VIDEO("test2.mp4"));
  cv::VideoCapture capture(VIDEO("test3.mp4"));

  if(!capture.isOpened()) {
    FATAL("Could not open video file via {}", capture.getBackendName());
    return -1;
  }

  KEY operation = KEY::NONE;
  while (operation != KEY::ESC) {
    // Extract frame from video
    cv::Mat frame;
    capture >> frame;

    if (frame.empty()) {
      DEBUG("Received empty frame, returning...");
      break;
    }

    // Resize frame to ease processing
    cv::Mat resized(cv::Size(frame.cols / 2, frame.rows / 2), CV_8UC3, cv::Scalar(0));
    cv::resize(frame, resized, cv::Size(frame.cols / 2, frame.rows / 2), 0.5, 0.5, cv::INTER_CUBIC);

    cv::Mat processed(resized.size(), CV_8UC3, cv::Scalar(0));

    detect(resized, processed);

    cv::imshow(CAMERA, resized);
    cv::imshow(OUTPUT, processed);

    // Process frame

    operation = WaitKey(25);
    switch (operation) {
      default:
        break;
    }
  }

  capture.release();

  cv::destroyAllWindows();

  Logger::destroy();
  return 0;
}
