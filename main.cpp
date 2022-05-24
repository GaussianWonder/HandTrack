#include "opencv2/opencv.hpp"
#include "common.h"
#include "slider.h"
#include <cmath>
#include <fstream>
#include <ranges>
#include <functional>
#include <tuple>

#include "video_window.h"
#include "spaces.h"
#include "skin_detector.h"
#include "motion_detector.h"

#define CAMERA "Camera"
#define HAND_MASK "Skin Mask"
#define HAND_EDGE "Hand edge"

// Skin detector controls
#define OPEN_CLOSE_KERNEL "Open/Close Kernel"
#define GAUSSIAN_BLUR_KERNEL "Blur Kernel"
#define EROSION_KERNEL "Erode Kernel"

SkinDetector* fromRaw(void *rawDetector)
{
  ASSERT(rawDetector != nullptr, "There is no skin detector around here");
  return (SkinDetector *)rawDetector;
}

using HandEdges = cv::Mat;
using HandDebugDraw = cv::Mat;
using ProcessedFrame = std::tuple<HandEdges, HandDebugDraw>;

/**
 * @brief Processes a cv::Mat frame and returns (for debug purposes) a tuple of images that resemble steps of the algorithm
 * 
 * @param detected frame run through a detector
 */
ProcessedFrame processFrame(const cv::Mat &detected) {
  ObjectTrace trace(detected);
  cv::Mat drawing = trace.draw(detected.size());

  cv::RNG RNG((int) time(0));

  cv::circle(drawing, trace.hullCenter, 3, cv::Scalar(RNG.uniform(0, 256), RNG.uniform(0,256), RNG.uniform(0,256)));
  cv::circle(drawing, trace.contourCenter, 2, cv::Scalar(RNG.uniform(0, 256), RNG.uniform(0,256), RNG.uniform(0,256)));

  cv::Rect boundingBox = cv::boundingRect(trace.hull);
  cv::rectangle(drawing, boundingBox, cv::Scalar(RNG.uniform(0, 255), RNG.uniform(0, 255), RNG.uniform(0, 255)));

  cv::Point center = cv::Point(boundingBox.x + boundingBox.width / 2, boundingBox.y + boundingBox.height / 2);
  cv::circle(drawing, center, 5, cv::Scalar(RNG.uniform(0, 255), RNG.uniform(0, 255), RNG.uniform(0, 255)));

  cv::Scalar fingertipColor(RNG.uniform(0, 256), RNG.uniform(0,256), RNG.uniform(0,256));
  for (auto &point : trace.hullDefects) {
    cv::Point ptStart(trace.contour[point[0]]);
    cv::Point ptEnd(trace.contour[point[1]]);
    cv::Point ptFar(trace.contour[point[2]]);

    double angle = std::atan2(center.y - ptStart.y, center.x - ptStart.x) * 180 / CV_PI;
    double inAngle = innerAngle(ptStart.x, ptStart.y, ptEnd.x, ptEnd.y, ptFar.x, ptFar.y);
    double length = std::sqrt(std::pow(ptStart.x - ptFar.x, 2) + std::pow(ptStart.y - ptFar.y, 2));
    if (angle > -30 && angle < 160 && std::abs(inAngle) > 20 && std::abs(inAngle) < 120 && length > 0.1 * boundingBox.height) {
      cv::circle(drawing, ptStart, 9, fingertipColor, cv::FILLED);
    }
  }

  return std::make_tuple(detected, drawing);
}

ProcessedFrame skinDetect(const cv::Mat &frame, SkinDetector &detect) {
  cv::Mat skinMask = newGray(frame.size());
  detect(frame, skinMask);
  return processFrame(skinMask);
}

int main() {
  Logger::init();

  // Create instance of VideoCapture
  // cv::VideoCapture capture(VIDEO("test.mp4"));
  // cv::VideoCapture capture(VIDEO("test2.mp4"));
  // cv::VideoCapture capture(VIDEO("test3.mp4"));
  // cv::VideoCapture capture(VIDEO("test_clean.mp4"));
  cv::VideoCapture capture(VIDEO("smaller_test.mp4"));

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
  MotionDetector<8> inMotion(gray_frames);

  // Slider to control the current video frame rendering
  Slider slider(frames.size() - inMotion.windowSize);
  // Current key pressed
  KEY operation = KEY::NONE;
 
  // On Event Changed handler
  auto onChangeHandler = [&]() {
    ProcessedFrame result = skinDetect(frames[slider.getCurrentIndex()], detect);

    cv::imshow(CAMERA, frames[slider.getCurrentIndex()]);
    cv::imshow(HAND_MASK, std::get<0>(result));
    cv::imshow(HAND_EDGE, std::get<1>(result));
  };

  // Before entering the event loop create the windows and assign control callbacks
  { // Create the windows and assign slider controls
    cv::namedWindow(CAMERA);
    cv::namedWindow(HAND_MASK);
    cv::namedWindow(HAND_EDGE);

    cv::createTrackbar(
      OPEN_CLOSE_KERNEL, HAND_MASK,
      &detect.openCloseKernel, 10,
      [](int pos, void *r) {
        SkinDetector *d = fromRaw(r);
        d->openCloseKernel = pos;
      },
      (void*) &detect
    );
    cv::createTrackbar(
      GAUSSIAN_BLUR_KERNEL, HAND_MASK,
      &detect.blurKernel, 3,
      [](int pos, void *r) {
        SkinDetector *d = fromRaw(r);
        d->blurKernel = pos;
      },
      (void*) &detect
    );
    cv::createTrackbar(
      EROSION_KERNEL, HAND_MASK,
      &detect.erodeKernel, 3,
      [](int pos, void *r) {
        SkinDetector *d = fromRaw(r);
        d->erodeKernel = pos;
      },
      (void*) &detect
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
