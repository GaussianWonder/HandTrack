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

  cv::Scalar targetBoxColor = cv::Scalar(RNG.uniform(0, 255), RNG.uniform(0, 255), RNG.uniform(0, 255));

  cv::Rect boundingBox = cv::boundingRect(trace.hull);
  cv::rectangle(drawing, boundingBox, targetBoxColor);

  cv::Point center = cv::Point(boundingBox.x + boundingBox.width / 2, boundingBox.y + boundingBox.height / 2);
  cv::circle(drawing, center, 5, targetBoxColor);

  std::vector<cv::Point> fingertipCandidates;
  double minDist = detected.rows + detected.cols + 1;
  double maxDist = -1;
  for (auto &point : trace.hullDefects) {
    cv::Point ptStart(trace.contour[point[0]]);
    cv::Point ptEnd(trace.contour[point[1]]);
    cv::Point ptFar(trace.contour[point[2]]);

    double angle = std::atan2(center.y - ptStart.y, center.x - ptStart.x) * 180 / CV_PI;
    double inAngle = innerAngle(ptStart.x, ptStart.y, ptEnd.x, ptEnd.y, ptFar.x, ptFar.y);
    double length = std::sqrt(std::pow(ptStart.x - ptFar.x, 2) + std::pow(ptStart.y - ptFar.y, 2));

    const bool isInnerFinger = angle > -30 && angle < 160 && std::abs(inAngle) > 20 && std::abs(inAngle) < 120 && length > 0.1 * boundingBox.height;
    if (isInnerFinger) {
      fingertipCandidates.push_back(ptStart);
      fingertipCandidates.push_back(ptEnd);

      const double dist = cv::norm(ptStart - ptEnd);
      if (dist < minDist)
        minDist = dist;
      if (dist > maxDist)
        maxDist = dist;
    }
  }

  std::size_t fingertipCandidatesSize = fingertipCandidates.size();
  std::vector<cv::Point> fingertips;
  auto addToFingertipsIfUnique = [&](const cv::Point &point) {
    if (std::find(fingertips.begin(), fingertips.end(), point) == fingertips.end())
      fingertips.push_back(point);
  };

  for (std::size_t i = 0; i < fingertipCandidatesSize; ++i) {
    const cv::Point &prevPoint = fingertipCandidates[(i + fingertipCandidatesSize - 1) % fingertipCandidatesSize];
    const cv::Point &point = fingertipCandidates[i];
    const cv::Point &nextPoint = fingertipCandidates[(i + 1) % fingertipCandidatesSize];

    const double d1 = cv::norm(point - prevPoint);
    const double d2 = cv::norm(point - nextPoint);

    const bool leftCloser = d1 < minDist;
    const bool rightCloser = d2 < minDist;

    if (leftCloser) {
      addToFingertipsIfUnique(midpoint(point, prevPoint));
    }

    if (rightCloser) {
      addToFingertipsIfUnique(midpoint(point, nextPoint));
    }

    if (!leftCloser && !rightCloser) {
      addToFingertipsIfUnique(point);
    }
  }

  cv::Scalar fingertipColor(RNG.uniform(0, 256), RNG.uniform(0,256), RNG.uniform(0,256));
  for (std::size_t i = 0; i < fingertips.size(); ++i) {
    const cv::Point &point = fingertips[i];

    cv::circle(drawing, point, 9, fingertipColor, cv::FILLED);

    const cv::Point fromCenter = (point - trace.contourCenter) * 1.15f;
    const cv::Point extendedEnd = trace.contourCenter + fromCenter;
    cv::line(drawing, trace.contourCenter, extendedEnd, fingertipColor);

    cv::putText(drawing, std::to_string(i), extendedEnd, cv::FONT_HERSHEY_SIMPLEX, 0.75, fingertipColor);
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

    cv::Mat overlay = frames[slider.getCurrentIndex()].clone();
    cv::Mat &drawing = std::get<1>(result);
    for (std::size_t i = 0; i < drawing.rows; ++i) {
      for (std::size_t j = 0; j < drawing.cols; ++j) {
        if (drawing.at<cv::Vec3b>(i, j) != cv::Vec3b(0, 0, 0)) {
          overlay.at<cv::Vec3b>(i, j) = drawing.at<cv::Vec3b>(i, j);
        }
      }
    }
    cv::imshow("Overlay", overlay);
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
