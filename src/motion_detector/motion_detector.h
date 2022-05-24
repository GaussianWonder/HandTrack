#ifndef __MOTION_DETECTOR_H__
#define __MOTION_DETECTOR_H__

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
  MotionDetector(const std::vector<cv::Mat> &frames, const std::size_t medianFrameCount = 60)
    :background(getBackground(frames, medianFrameCount))
  {}

  void operator()(const std::vector<cv::Mat> &frames)
  {
    
  }

  constexpr static std::size_t windowSize = WSZ;
private:
  // result of getBackground()
  cv::Mat background;
};

#endif // __MOTION_DETECTOR_H__