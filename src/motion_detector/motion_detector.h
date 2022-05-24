#ifndef __MOTION_DETECTOR_H__
#define __MOTION_DETECTOR_H__

#include <vector>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

#include <opencv2/highgui.hpp>

class MotionDetector
{
public:
  // TODO maybe use this
  enum class MedianPickDistribution {
    ASCENDING_START,
    RANDOM,
  };

  MotionDetector(const std::vector<cv::Mat> &frames, const std::size_t medianFrameCount = 60, const std::size_t frameDifferenceCount = 8);

  void operator()(cv::Mat &src, cv::Mat &dst)
  {
    
  }

  // used to getBackgrund()
  std::size_t medianFrameCount;
  // frame window to detect motion body
  std::size_t frameDifferenceCount;
private:
  // result of getBackground()
  cv::Mat background;

  cv::Mat getBackground(const std::vector<cv::Mat> &frames);
};

#endif // __MOTION_DETECTOR_H__