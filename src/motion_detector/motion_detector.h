#ifndef __MOTION_DETECTOR_H__
#define __MOTION_DETECTOR_H__

#include "video_window.h"
#include "misc.h"
#include <vector>
#include <string>
#include <array>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

#include <opencv2/highgui.hpp>

cv::Mat getBackground(const std::vector<cv::Mat> &frames, const std::size_t medianFrameCount);

template<std::size_t WSZ>
class MotionDetector
{
public:
  MotionDetector(const std::vector<cv::Mat> &frames, const std::size_t medianFrameCount = 50)
    :background(getBackground(frames, medianFrameCount))
  {}

  void operator()(const VideoWindow<WSZ> &frameWindow, cv::Mat &dst)
  {
    // video window frames are valid and correctly initialized
    if (frameWindow.isValid() && WSZ > 0) {
      cv::Size windowFrameSize(frameWindow.frames[0].size());

      cv::Mat concatDiff = newGray(windowFrameSize);
      for (std::size_t i = 0; i < WSZ; ++i) {
        cv::Mat diff = newGray(windowFrameSize);
        cv::absdiff(frameWindow.frames[i], this->background, diff);

        cv::Mat diffBinary = newGray(windowFrameSize);
        cv::threshold(diff, diffBinary, 50, 255, cv::THRESH_BINARY);

        cv::Mat dilated = newGray(windowFrameSize);
        cv::dilate(diffBinary, dilated, cv::Mat());

        cv::Mat nextMerge = newGray(windowFrameSize);
        cv::bitwise_or(dilated, concatDiff, nextMerge);
        concatDiff = nextMerge.clone();
      }

      dst = concatDiff.clone();
    }
  }

  constexpr static std::size_t windowSize = WSZ;
private:
  // result of getBackground()
  cv::Mat background;
};

#endif // __MOTION_DETECTOR_H__