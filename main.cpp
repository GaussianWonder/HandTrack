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
#include "motion_detector.h"

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

// canny edge detection controls
int canny_low = 0;
int canny_high = 100;

using SkinMask = cv::Mat;
using HandEdges = cv::Mat;
using HandHullEdges = cv::Mat;
using ProcessedFrame = std::tuple<SkinMask, HandEdges, HandHullEdges>;

/**
 * @brief Processes a cv::Mat frame and returns (for debug purposes) a tuple of images that resemble steps of the algorithm
 * 
 * @param frame the cv::Mat image to process
 * @param scaleTarget The desired scaling of the image before processing. If smaller, this is ignored.
 * @param skinDetector see SkinDetector. This should be replaced with an Adapter for any backround substraction algorithms
 */
ProcessedFrame processFrame(const cv::Mat &frame, SkinDetector &detect) {
  // Detect skin in frame
  cv::Mat skin_mask = newGray(frame.size());
  detect(frame, skin_mask);

  // Canny Edge Detection and Convex Hull
  cv::Mat detected_edges = newGray(skin_mask.size());
  cv::Canny(skin_mask, detected_edges, canny_low, canny_high);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(detected_edges, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

  std::vector<std::vector<cv::Point>> hull(contours.size());
  for(std::size_t i = 0; i < contours.size(); ++i) {
    cv::convexHull(contours[i], hull[i]);
  }

  cv::Mat drawing = newColor(detected_edges.size());
  for(std::size_t i = 0; i < contours.size(); ++i) {
    if (contours[i].size() >= 500) {
      cv::Scalar color = cv::Scalar(RNG.uniform(0, 256), RNG.uniform(0,256), RNG.uniform(0,256));
      cv::drawContours(drawing, contours, (int)i, color);
      cv::drawContours(drawing, hull, (int)i, color);
    }
  }

  return std::make_tuple(skin_mask, detected_edges, drawing);
}

int main() {
  Logger::init();

  // Create instance of VideoCapture
  // cv::VideoCapture capture(VIDEO("test.mp4"));
  // cv::VideoCapture capture(VIDEO("test2.mp4"));
  cv::VideoCapture capture(VIDEO("test3.mp4"));

  if(!capture.isOpened()) {
    FATAL("Could not open video file via {}", capture.getBackendName());
    return -1;
  }

  // Spread video into cv::Mat frames
  std::vector<cv::Mat> frames = getVideoFrames(capture);
  capture.release();

  // Convert all frames to grayscale
  std::vector<cv::Mat> gray_frames = convertAll(frames, cv::COLOR_BGR2GRAY);

  // Create the skin detector
  SkinDetector detect;
  // Create the motion detector
  // MotionDetector inMotion(gray_frames);

  // Slider to control the current video frame rendering
  Slider slider(frames.size());
  // Current key pressed
  KEY operation = KEY::NONE;
 
  // On Event Changed handler
  auto onChangeHandler = [&](){
    ProcessedFrame result = processFrame(frames[slider.getCurrentIndex()], detect);

    cv::imshow(CAMERA, frames[slider.getCurrentIndex()]);
    cv::imshow(SKIN_MASK, std::get<0>(result));
    cv::imshow(HAND_EDGE, std::get<2>(result));
  };

  // Before entering the event loop create the windows and assign control callbacks
  { // Create the windows and assign slider controls
    cv::namedWindow(CAMERA);
    cv::namedWindow(SKIN_MASK);
    cv::namedWindow(HAND_EDGE);

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
  }

  // Event Loop
  while (operation != KEY::ESC) {
    // Input handling
    switch (operation)
    {
    case KEY::LEFT_ARROW:
      slider.previous();
      onChangeHandler();
      break;
    case KEY::RIGHT_ARROW:
      slider.next();
      onChangeHandler();
      break;
    case KEY::SPACE:
      onChangeHandler();
      break;
    }

    operation = WaitKey(25);
  }

  cv::destroyAllWindows();

  Logger::destroy();
  return 0;
}
