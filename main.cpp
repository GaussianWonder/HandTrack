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
#define CTHRESH "Combined Threshold"
#define EKERNEL "Erode Kernel"

static void getHsvThreshTrack(int pos, void *rawDetector) {
  ASSERT(rawDetector != nullptr, "There is no skin detector around here");

  SkinDetector *detect = (SkinDetector *)rawDetector;
  detect->addHsvMask = (pos + 1);
}

static void getErodeKernTrack(int pos, void *rawDetector){
  ASSERT(rawDetector != nullptr, "There is no skin detector around here");

  SkinDetector *detect = (SkinDetector *)rawDetector;
  detect->erosionKernel = (pos + 1);
}

int main() {
  Logger::init();

  // Create the skin detector
  SkinDetector detect;

  // Create the windows
  cv::namedWindow(CAMERA);
  cv::namedWindow(OUTPUT);

  // Assign slider controls for the skin detector
  cv::createTrackbar(CTHRESH, OUTPUT, &detect.addHsvMask, 1000, getHsvThreshTrack, (void*) &detect);
  cv::createTrackbar(EKERNEL, OUTPUT, &detect.erosionKernel, 10, getHsvThreshTrack, (void*) &detect);

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

    cv::imshow(CAMERA, frame);
    cv::Mat processed(frame.size(), CV_8UC3, cv::Scalar(0));

    detect(frame, processed);

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
