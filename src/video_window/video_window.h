#ifndef __IMAGE_WINDOW_H__
#define __IMAGE_WINDOW_H__

#include "common.h"
#include "logger.h"
#include <array>

template<std::size_t I>
class VideoWindow {
public:
  VideoWindow(const std::vector<cv::Mat> &frames, const std::size_t offset = 0)
  {
    const std::size_t startI = offset;
    const std::size_t endI = offset + I;
    if (endI > frames.size()) {
      FATAL("Cannot return a valid window frame, the number of frames {} is not big enough to fit {} frames from {} to {}", frames.size(), I, offset, offset + I);
      this->valid = false;
    } else {
      for (std::size_t i = startI; i < endI; ++i) {
        this->frames[i] = frames[i].clone();
      }
    }
  }

  bool isValid()
  {
    return this->valid;
  }

private:
  std::array<cv::Mat, I> frames;
  bool valid = true;
};

#endif // __IMAGE_WINDOW_H__
