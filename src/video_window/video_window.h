#ifndef __IMAGE_WINDOW_H__
#define __IMAGE_WINDOW_H__

#include "common.h"
#include "logger.h"
#include <array>

/**
 * @brief Helper class for extracting a window of frames from a collection of frames
 * 
 * @tparam I std::size_t > 0. should fit in the vector provided as input for constructors
 */
template<std::size_t I>
class VideoWindow {
public:
  VideoWindow(const std::vector<cv::Mat> &f, const std::size_t offset = 0)
  {
    const std::size_t startI = offset;
    const std::size_t endI = offset + I;

    if (endI > f.size()) {
      FATAL("Cannot return a valid window frame, the number of frames {} is not big enough to fit {} frames from {} to {}", f.size(), I, offset, offset + I);
      this->valid = false;
    } else {
      for (std::size_t i = startI, k = 0; i < endI; ++i, ++k) {
        this->frames[k] = f[i].clone();
      }
      this->valid = true;
    }
  }

  bool isValid()
  {
    return this->valid;
  }

  std::array<cv::Mat, I> frames;
  bool valid = true;
};

#endif // __IMAGE_WINDOW_H__
