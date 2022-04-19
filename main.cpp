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
#define SKIN_MASK "Skin Mask"
#define HAND_EDGE "Hand edge"

#define OPEN_CLOSE_KERNEL "Open/Close Kernel"
#define GAUSSIAN_BLUR_KERNEL "Blur Kernel"
#define EROSION_KERNEL "Erode Kernel"

#define CANNY_LOW "LOW Canny Threshold"
#define CANNY_HIGH "HIGH Canny Threshold"

cv::RNG RNG((int) time(0));

SkinDetector* fromRaw(void *rawDetector)
{
  ASSERT(rawDetector != nullptr, "There is no skin detector around here");
  return (SkinDetector *)rawDetector;
}

int canny_low = 0;
int canny_high = 100;

int main() {
  Logger::init();

  // Create the skin detector
  SkinDetector detect;

  // Create the windows
  cv::namedWindow(CAMERA);
  cv::namedWindow(SKIN_MASK);
  cv::namedWindow(HAND_EDGE);

  // Assign slider controls for the skin detector controls
  cv::createTrackbar(
    OPEN_CLOSE_KERNEL, SKIN_MASK,
    &detect.openCloseKernel, 10,
    [](int pos, void *r) {
      SkinDetector *d = fromRaw(r);
      d->openCloseKernel = pos;
    },
    (void*) &detect
  );
  cv::createTrackbar(
    GAUSSIAN_BLUR_KERNEL, SKIN_MASK,
    &detect.blurKernel, 3,
    [](int pos, void *r) {
      SkinDetector *d = fromRaw(r);
      d->blurKernel = pos;
    },
    (void*) &detect
  );
  cv::createTrackbar(
    EROSION_KERNEL, SKIN_MASK,
    &detect.erodeKernel, 3,
    [](int pos, void *r) {
      SkinDetector *d = fromRaw(r);
      d->erodeKernel = pos;
    },
    (void*) &detect
  );

  cv::createTrackbar(
    CANNY_LOW, HAND_EDGE,
    &canny_low, 100,
    [](int pos, void *r) {
      *((int*)r) = pos;
    },
    (void*) &canny_low
  );
  cv::createTrackbar(
    CANNY_HIGH, HAND_EDGE,
    &canny_high, 200,
    [](int pos, void *r) {
      *((int*)r) = pos;
    },
    (void*) &canny_high
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

    // Process frame
    // Resize frame to ease processing
    cv::Mat resized(cv::Size(frame.cols / 2, frame.rows / 2), CV_8UC3, cv::Scalar(0));
    cv::resize(frame, resized, cv::Size(frame.cols / 2, frame.rows / 2), 0.5, 0.5, cv::INTER_CUBIC);

    cv::Mat skin_mask(resized.size(), CV_8UC3, cv::Scalar(0));

    detect(resized, skin_mask);

    cv::imshow(CAMERA, resized);
    cv::imshow(SKIN_MASK, skin_mask);

    // Canny edge detection and Convex Hull
    cv::Mat edges(skin_mask.size(), CV_8UC1, cv::Scalar::all(0));
    cv::Mat detected_edges = skin_mask.clone();
    cv::Canny(skin_mask, detected_edges, canny_low, canny_high);
    detected_edges.copyTo(edges);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(detected_edges, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point>> hull(contours.size());
    for(std::size_t i = 0; i < contours.size(); ++i)
    {
      cv::convexHull(contours[i], hull[i]);
    }

    cv::Mat drawing = cv::Mat::zeros(detected_edges.size(), CV_8UC3);
    // for(std::size_t i = 0; i < contours.size(); ++i)
    // {
    //   if (contours[i].size() >= 500) {
    //     cv::Scalar color = cv::Scalar(RNG.uniform(0, 256), RNG.uniform(0,256), RNG.uniform(0,256));
    //     cv::drawContours(drawing, contours, (int)i, color);
    //     cv::drawContours(drawing, hull, (int)i, color);
    //   }
    // }
    if (contours.size()) {
      std::size_t maxI = 0;
      std::size_t maxSize = contours[0].size();
      for(std::size_t i = 1; i < contours.size(); ++i)
      {
        if (contours[i].size() > maxSize) {
          maxI = i;
          maxSize = contours[i].size();
        } 
      }
      cv::Scalar color = cv::Scalar(RNG.uniform(0, 256), RNG.uniform(0,256), RNG.uniform(0,256));
      cv::drawContours(drawing, contours, (int)maxI, color);
      cv::drawContours(drawing, hull, (int)maxI, color);
      cv::imshow(HAND_EDGE, detected_edges);
    }

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
