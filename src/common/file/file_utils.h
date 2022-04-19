#ifndef __FILE_UTILS_H__
#define __FILE_UTILS_H__

#include <string>
#include "opencv2/opencv.hpp"

class FileUtils {
public:
  static std::string readFile(const std::string &fileName);
  static cv::Mat readImage(const std::string &fileName, const cv::ImreadModes mode);
  static void saveImage(const cv::Mat &img, const std::string &fileName);
  static void quickSave(const cv::Mat &img);
  static std::vector<std::string> nestedFilesOf(const std::string &path);
  static std::vector<std::string> filesOf(const std::string &path);
};

#endif // __FILE_UTILS_H__
