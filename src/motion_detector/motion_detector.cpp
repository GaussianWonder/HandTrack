#include "motion_detector.h"
#include "logger.h"
#include "misc.h"
#include <algorithm>

MotionDetector::MotionDetector(const std::vector<cv::Mat> &frames, const std::size_t medianFrameCount = 60, const std::size_t frameDifferenceCount = 8)
  :medianFrameCount(medianFrameCount)
  ,frameDifferenceCount(frameDifferenceCount)
  ,background(getBackground(frames))
{}

cv::Mat MotionDetector::getBackground(const std::vector<cv::Mat> &frames)
{
  std::size_t framesLen = frames.size();

  if (framesLen == 0) {
    FATAL("no frames provided, can't get background, returning 700x400 grayscale black image");
    return newGray(cv::Size(700, 400));
  }

  if (this->medianFrameCount > framesLen) {
    WARN("medianFrameCount is greater than the number of frames provided!");
    return frames[0];
  }

  cv::Mat bg = newGray(frames[0].size());
  std::size_t rows = frames[0].rows;
  std::size_t cols = frames[0].cols;
  std::size_t medianIndex = this->medianFrameCount / 2;

  for (int i=0; i<rows; ++i) {
    for (int j=0; j<cols; ++j) {
      std::vector<uchar> values;

      for (std::size_t f = 0; f < this->medianFrameCount; ++f)
        values.push_back(frames[f].at<uchar>(i, j));

      std::sort(values.begin(), values.end());

      bg.at<uchar>(i, j) = values[medianIndex];
    }
  }
}
