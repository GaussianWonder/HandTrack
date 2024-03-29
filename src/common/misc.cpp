#include "misc.h"
#include "logger.h"
#include <algorithm>
#include <math.h>

/**
 * @brief Convert form opencv int key to enum of named key
 */
KEY resolvedKey(const int key)
{
  switch (key) {
    case KEY_ESC:
    case KEY_SPACE:
    case KEY_ENTER:
    case KEY_DOWN_ARROW:
    case KEY_RIGHT_ARROW:
    case KEY_UP_ARROW:
    case KEY_LEFT_ARROW:
      return static_cast<KEY>(key);
    default:
      return KEY::NONE;
  }
}

/**
 * @brief Scale image proportional to image resolution
 * 
 * @param image Image to scale
 * @param scale number between (0, 1]
 * @param type cv::Mat pixel type
 * @return cv::Mat scaled (resized) image
 */
cv::Mat scaleImage(const cv::Mat &image, const double scale, const int type)
{
  const cv::Size scaledSize(image.cols * scale, image.rows * scale);
  cv::Mat scaled(scaledSize, type, cv::Scalar::all(0));
  cv::resize(image, scaled, scaledSize, scale, scale, cv::INTER_CUBIC);
  return scaled;
}

/**
 * @brief Make a new gray (CV_8UC1) image of a given size
 */
cv::Mat newGray(const cv::Size &size, const cv::Scalar &color)
{
  cv::Mat gray(size, CV_8UC1, color);
  return gray;
}

/**
 * @brief Make a new RGB (CV_8UC3) image of a given size
 */
cv::Mat newColor(const cv::Size &size, const cv::Scalar &color)
{
  cv::Mat colorImage(size, CV_8UC3, color);
  return colorImage;
}

/**
 * @brief Get a vector of frames form a given video capture
 * This also rescales the input to a given target resolution (rescale to fit)
 * 
 * @param capture Video capture to extract from
 * @param widthTarget rescaling width target
 * @param heightTarget rescalinig height target
 * @return std::vector<cv::Mat> rescaled video frames
 */
std::vector<cv::Mat> getVideoFrames(cv::VideoCapture &capture, const int widthTarget, const int heightTarget)
{
  std::vector<cv::Mat> frames;
  cv::Mat frame;

  capture >> frame;
  while (!frame.empty()) {
    const double widthScaleFactor = frame.cols > widthTarget ? ((double) widthTarget) / frame.cols : 1.0;
    const double heightScaleFactor = frame.rows > heightTarget ? ((double) heightTarget) / frame.rows : 1.0;
    const double scaleFactor = MIN(widthScaleFactor, heightScaleFactor);

    cv::Mat resized = scaleImage(frame, scaleFactor);
    frames.push_back(resized.clone());

    capture >> frame;
  }

  return frames;
}

/**
 * @brief Accepts a vector of frames and applies a color conversion to each frame
 * 
 * @param frames frames array, usually from getVideoFrames()
 * @param conversionType What conversion to apply
 * @return std::vector<cv::Mat> color converted frames
 */
std::vector<cv::Mat> convertAll(const std::vector<cv::Mat> &frames, const cv::ColorConversionCodes conversionType)
{
  std::vector<cv::Mat> convertedFrames;
  for (const auto &frame : frames) {
    cv::Mat converted;
    cv::cvtColor(frame, converted, conversionType);
    convertedFrames.push_back(converted);
  }
  return convertedFrames;
}

/**
 * @brief Construct a new Object Trace:: Object Trace object
 * 
 * @param binaryImage trace from the contour of a binary image
 */
ObjectTrace::ObjectTrace(const cv::Mat &binaryImage)
{
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;

  cv::findContours(binaryImage, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
  std::size_t largestContour = 0; // largest contour === largest area for this usecase
  double maxArea = -1;
  for (std::size_t i = 0; i < contours.size(); ++i) {
    const double area = cv::contourArea(contours[i]);
    if (area > maxArea) {
      maxArea = area;
      largestContour = i;
    }
  }

  if (maxArea > 0) {
    std::copy(contours[largestContour].begin(), contours[largestContour].end(), std::back_inserter(this->contour));
    cv::Mat contourMat(contours[largestContour]);

    std::vector<cv::Point> hull;
    cv::convexHull(contourMat, hull, true);
    std::copy(hull.begin(), hull.end(), std::back_inserter(this->hull));
    this->area = cv::contourArea(hull);

    std::vector<int> hullIndexes;
    cv::convexHull(contourMat, hullIndexes, true);

    std::vector<cv::Vec4i> convexityDefects;
    cv::convexityDefects(contourMat, hullIndexes, convexityDefects);
    std::copy(convexityDefects.begin(), convexityDefects.end(), std::back_inserter(this->hullDefects));

    {
      cv::Moments m = cv::moments(this->contour);
      this->contourCenter = cv::Point(m.m10 / m.m00, m.m01 / m.m00);
    }

    {
      cv::Moments m = cv::moments(this->hull);
      this->hullCenter = cv::Point(m.m10 / m.m00, m.m01 / m.m00);
    }
  }
}

/**
 * @brief Draw the traced object
 * 
 * @param size size of the desired drawing
 * @return cv::Mat the drawing
 */
cv::Mat ObjectTrace::draw(const cv::Size &size)
{
  cv::Scalar dullGray(68, 68, 68);

  cv::Mat drawing = newColor(size);
  if (this->area <= 0)
    return drawing;

  cv::RNG RNG((int) time(0));

  std::vector<std::vector<cv::Point>> contours;
  std::vector<std::vector<cv::Point>> hulls;

  cv::Scalar contourColor = cv::Scalar(RNG.uniform(0, 256), RNG.uniform(0,256), RNG.uniform(0,256));
  cv::Scalar hullColor = cv::Scalar(RNG.uniform(0, 256), RNG.uniform(0,256), RNG.uniform(0,256));

  contours.emplace_back(std::vector<cv::Point>(this->contour));
  hulls.push_back(std::vector<cv::Point>(this->hull));

  cv::circle(drawing, this->hullCenter, 3, hullColor);
  cv::circle(drawing, this->contourCenter, 2, contourColor);

  cv::drawContours(drawing, contours, 0, contourColor);
  cv::drawContours(drawing, hulls, 0, hullColor);

  cv::Scalar hullPtColor(RNG.uniform(0, 256), RNG.uniform(0,256), RNG.uniform(0,256));
  for (auto &point : this->hull)
    cv::circle(drawing, point, 2, hullPtColor, -1);

  cv::Scalar defectColor(RNG.uniform(0, 256), RNG.uniform(0,256), RNG.uniform(0,256));
  for(auto &defect : this->hullDefects) {
    cv::Point ptStart(this->contour[defect[0]]);
    cv::Point ptEnd(this->contour[defect[1]]);
    cv::Point ptFar(this->contour[defect[2]]);

    cv::line(drawing, ptStart, ptFar, dullGray);
    cv::line(drawing, ptFar, ptEnd, dullGray);

    cv::circle(drawing, ptFar, 5, defectColor);
  }

  return drawing;
}

// Math helpers

float innerAngle(float px1, float py1, float px2, float py2, float cx1, float cy1)
{
  float dist1 = std::sqrt( (px1-cx1)*(px1-cx1) + (py1-cy1)*(py1-cy1) );
  float dist2 = std::sqrt( (px2-cx1)*(px2-cx1) + (py2-cy1)*(py2-cy1) );

  float Ax, Ay;
  float Bx, By;
  float Cx, Cy;

  // find closest point to C  
  // DEBUG("dist1: {}, dist2: {}", dist1, dist2);

  Cx = cx1;
  Cy = cy1;
  if(dist1 < dist2) {
    Bx = px1;
    By = py1;
    Ax = px2;
    Ay = py2;
  } else {
    Bx = px2;
    By = py2;
    Ax = px1;
    Ay = py1;
  }

  float Q1 = Cx - Ax;
  float Q2 = Cy - Ay;
  float P1 = Bx - Ax;
  float P2 = By - Ay;

  float A = std::acos( (P1*Q1 + P2*Q2) / ( std::sqrt(P1*P1+P2*P2) * std::sqrt(Q1*Q1+Q2*Q2) ) );

  A = A * 180 / CV_PI;

  return A;
}

float innerAngle(const cv::Point &p1, const cv::Point &p2, const cv::Point &p3)
{
  innerAngle(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);
}

cv::Point midpoint(const cv::Point& a, const cv::Point& b) {
  cv::Point ret;
  ret.x = (a.x + b.x) / 2;
  ret.y = (a.y + b.y) / 2;
  return ret;
}

// float innerAngle(const cv::Point &p1, const cv::Point &p2, const cv::Point &p3)
// {
//   const cv::Point v1 = p2 - p1;
//   const cv::Point v2 = p3 - p2;
//   const float angle = std::acos(cv::norm(v1) * cv::norm(v2) + cv::dotProduct(v1, v2)) / (cv::norm(v1) * cv::norm(v2));
//   return angle;
// }
