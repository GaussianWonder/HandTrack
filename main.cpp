#include "opencv2/opencv.hpp"
#include "common.h"
#include "slider.h"
#include <cmath>
#include <fstream>
#include <ranges>
#include <functional>
#include <tuple>

#include "spaces.h"

// Processing functions

void bi_level_color_map(const cv::Mat &src, cv::Mat &dst, const uchar threshold = 127)
{ // black and white
  for(int i = 0; i < src.rows; ++i)
    for(int j = 0; j < src.cols; ++j)
      dst.at<uchar>(i,j) = src.at<uchar>(i,j) > threshold ? 255 : 0;
  imshow("input image", src);
  imshow("black&white", dst);
}

void negative(const cv::Mat &src, cv::Mat &dst)
{
  for(int i = 0; i < src.rows; ++i)
    for(int j = 0; j < src.cols; ++j)
      dst.at<uchar>(i,j) = 255 - src.at<uchar>(i,j);
  imshow("input image", src);
  imshow("negative", dst);
}

void additive(const cv::Mat &src, cv::Mat &dst)
{
  for(int i = 0; i < src.rows; ++i)
    for(int j = 0; j < src.cols; ++j) {
      auto val = src.at<uchar>(i,j);
      dst.at<uchar>(i,j) = val <= 155 ? val + 100 : 255;
    }
  imshow("input image", src);
  imshow("additive grayscale", dst);
}

void multiplicative(const cv::Mat &src, cv::Mat &dst)
{
  for(int i = 0; i < src.rows; ++i)
    for(int j = 0; j < src.cols; ++j) {
      auto val = src.at<uchar>(i,j) * 2.0f;
      dst.at<uchar>(i,j) = val <= 255.0f ? (uchar) val : 255;
    }
  imshow("input image", src);
  imshow("multiplicative grayscale", dst);
}

void four_squares(cv::Mat &dst)
{
  auto halfCols = dst.cols / 2;
  auto halfRows = dst.rows / 2;
  cv::Rect rectangles[] = {
    cv::Rect(0, 0, halfCols, halfRows),
    cv::Rect(halfCols, 0, halfCols, halfRows),
    cv::Rect(0, halfRows, halfCols, halfRows),
    cv::Rect(halfCols, halfRows, halfCols, halfRows)
  };
  cv::Scalar scalars[] = {
    cv::Scalar(255, 255, 255),
    cv::Scalar(0, 0, 255),
    cv::Scalar(0, 255, 0),
    cv::Scalar(0, 255, 255)
  };
  for (int i = 0; i < 4; ++i)
    cv::rectangle(dst, rectangles[i], scalars[i], -1);
  imshow("generated cv::Mat", dst);
}

void split_channels(const cv::Mat &src)
{
  cv::Mat red(src.rows, src.cols, CV_8UC3);
  cv::Mat green(src.rows, src.cols, CV_8UC3);
  cv::Mat blue(src.rows, src.cols, CV_8UC3);

  for (int i = 0; i < src.rows; ++i)
    for (int j = 0; j < src.cols; ++j) {
      auto val = src.at<cv::Vec3b>(i,j);
      red.at<cv::Vec3b>(i,j) = cv::Vec3b(0, 0, val[2]);
      green.at<cv::Vec3b>(i,j) = cv::Vec3b(0, val[1], 0);
      blue.at<cv::Vec3b>(i,j) = cv::Vec3b(val[0], 0, 0);
    }

  imshow("input image", src);
  imshow("red", red);
  imshow("green", green);
  imshow("blue", blue);
}

void split_hsv(const cv::Mat &src)
{
  cv::Mat hue(src.rows, src.cols, CV_8UC1);
  cv::Mat saturation(src.rows, src.cols, CV_8UC1);
  cv::Mat value(src.rows, src.cols, CV_8UC1);

  for (int i = 0; i < src.rows; ++i)
    for (int j = 0; j < src.cols; ++j) {
      auto val = src.at<cv::Vec3b>(i,j);
      HSV hsv(val[2], val[1], val[0]);

      hue.at<uchar>(i,j) = hsv.h * 0.7f;
      saturation.at<uchar>(i,j) = hsv.s * 2.55f;
      value.at<uchar>(i,j) = hsv.v * 2.55f;
    }

  imshow("input image", src);
  imshow("hue", hue);
  imshow("saturation", saturation);
  imshow("value", value);
}

void color_to_grayscale(const cv::Mat &src)
{
  cv::Mat dst(src.rows, src.cols, CV_8UC1);
  for (int i = 0; i < src.rows; ++i)
    for (int j = 0; j < src.cols; ++j) {
      auto val = src.at<cv::Vec3b>(i,j);
      dst.at<uchar>(i,j) = (uchar) (val[0] + val[1] + val[2]) / 3;
    }
  imshow("input image", src);
  imshow("grayscale", dst);
}

void compute_historgram(const cv::Mat &src)
{
  int histSize = 361;
  float range[] = { 0, 256 }; //the upper boundary is exclusive
  const float* histRange[] = { range };

  std::vector<cv::Mat> hsvHists = {
    cv::Mat(src.rows, src.cols, CV_8UC1),
    cv::Mat(src.rows, src.cols, CV_8UC1),
    cv::Mat(src.rows, src.cols, CV_8UC1)
  };

  for (int i = 0; i < src.rows; ++i)
    for (int j = 0; j < src.cols; ++j) {
      auto val = src.at<cv::Vec3b>(i,j);
      HSV hsv(val[2], val[1], val[0]);

      hsvHists[0].at<uchar>(i,j) = hsv.h * 0.7f;
      hsvHists[1].at<uchar>(i,j) = hsv.s * 2.55f;
      hsvHists[2].at<uchar>(i,j) = hsv.v * 2.55f;
    }

  cv::Mat hueHist;
  cv::calcHist(hsvHists.data(), 1, 0, cv::Mat(), hueHist, 1, &histSize, histRange, true, false);

  int hist_w = 360, hist_h = 400;
  int bin_w = cvRound((double) hist_w / histSize);

  cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0,0,0));

  cv::normalize(hueHist, hueHist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

  for(int i=0; i<=histSize; ++i)
  {
    RGB rgb = hsv_to_rgb((float) i, 100.0f, 100.0f);
    auto color = cv::Scalar(rgb.b, rgb.g, rgb.r);
    auto currentPoint = cv::Point(
      bin_w * (i),
      hist_h - cvRound(hueHist.at<float>(i))
    );
    auto toBottom = cv::Point(
      bin_w * (i),
      hist_h
    );
    cv::line(
      histImage,
      currentPoint,
      toBottom,
      color,
      2, 8, 0
    );
  }

  imshow("Source", src);
  imshow("Hue Histogram", histImage);

  std::vector<float> maxLocals = {};

  const int WH = 5;
  const int windowSize = 2*WH+1;
  const float TH = 0.0003;
  for (int mid = WH + 1; mid<=histSize - WH - 1; ++mid) {
    const float midVal = hueHist.at<float>(mid);
    int sum = 0;
    bool applicable = true;

    const int start = mid - WH;
    const int stop = mid + WH;
    for (int i=start; i<stop; ++i) {
      float currentVal = hueHist.at<float>(i);
      sum += currentVal;

      if (currentVal > midVal) {
        applicable = false;
        break;
      }
    }

    float avg = (float) sum / (float) windowSize;
    if (applicable && midVal > avg + TH) {
      maxLocals.push_back(midVal);
    }
  }

  // cv::Mat maxLocalsHist(hist_h, hist_w, CV_8UC3, cv::Scalar(0,0,0));
  // for (auto maxLocal: maxLocals) {
  //   RGB rgb = hsv_to_rgb((float) maxLocal, 100.0f, 100.0f);
  //   auto color = cv::Scalar(rgb.b, rgb.g, rgb.r);
  //   auto currentPoint = cv::Point(
  //     bin_w * (maxLocal),
  //     hist_h - cvRound(hueHist.at<float>(maxLocal))
  //   );
  //   auto toBottom = cv::Point(
  //     bin_w * (maxLocal),
  //     hist_h
  //   );
  //   cv::line(
  //     maxLocalsHist,
  //     currentPoint,
  //     toBottom,
  //     color,
  //     2, 8, 0
  //   );
  // }
  // imshow("Max local histogram", maxLocalsHist);

  const float errorDistribution[3][3] = {
    {0, 0, 0},
    {0, 1, 7/16},
    {3/16, 5/16, 1/16}
  };

  float errorAccumulation[(int) src.rows][(int) src.cols] = { 0 };
  cv::Mat reducedColor(src.rows, src.cols, CV_8UC3);
  const int maxLocalSz = maxLocals.size();
  for (int i = 0; i < src.rows; ++i)
    for (int j = 0; j < src.cols; ++j) {
      auto val = src.at<cv::Vec3b>(i,j);
      HSV hsv(val[2], val[1], val[0]);
      const uchar grayHue = hsv.h * 0.7f;

      float minDiff = 500.0f;
      float maxLocalMinDiff = 0.0f;
      for (float maxLocal : maxLocals) {
        float diff = std::abs(grayHue - maxLocal);
        if (minDiff > diff) {
          minDiff = diff;
          maxLocalMinDiff = maxLocal;
        }
      }

      float maxLocalHue = maxLocalMinDiff * 1.41f;
      RGB newColor = hsv_to_rgb(maxLocalHue, hsv.s, hsv.v);

      reducedColor.at<cv::Vec3b>(i, j) = cv::Vec3b(newColor.B(), newColor.G(), newColor.R());

      float error = hsv.h - maxLocalHue;
      for (int ii; ii<3; ++ii) {
        for (int jj; jj<3; ++jj) {
          int errorI = i + ii;
          int errorJ = j + jj;
          if (errorI > 0 && errorI < src.rows && errorJ > 0 && errorJ < src.cols) {
            errorAccumulation[errorI][errorJ] += error * errorDistribution[ii][jj];
          }
        }
      }
    }

  cv::Mat ditheredColor(src.rows, src.cols, CV_8UC3);
  for (int i = 0; i < src.rows; ++i)
    for (int j = 0; j < src.cols; ++j) {
      auto val = reducedColor.at<cv::Vec3b>(i,j);
      HSV hsv(val[2], val[1], val[0]);
      auto error = errorAccumulation[i][j];
      auto newHue = float(hsv.h + error);
      if (newHue > 360.0f)
        newHue -= 360.0f;
      auto replaceWith = hsv_to_rgb(newHue, hsv.s, hsv.v);
      reducedColor.at<cv::Vec3b>(i,j) = cv::Vec3b(replaceWith.B(), replaceWith.G(), replaceWith.R());
    }

  imshow("Thresholded", reducedColor);
  imshow("Dithered", ditheredColor);
}

void on_mouse_click(int event, int x, int y, int flags, void* ptr)
{
  cv::Mat *src = (cv::Mat *) ptr;
  if (event == cv::EVENT_LBUTTONUP) {
    RGB clickedColor(
      src->at<cv::Vec3b>(y, x)[2],
      src->at<cv::Vec3b>(y, x)[1],
      src->at<cv::Vec3b>(y, x)[0]
    );
    DEBUG(
      "Clicked at ({}, {}) RGB: {} {} {}",
      x,
      y,
      clickedColor.r,
      clickedColor.g,
      clickedColor.b
    );

    int pixelCount = 0; // area
    cv::Point pixelMass(0, 0); // sum of coords on each axis

    for(int i = 0; i < src->rows; ++i) {
      for(int j = 0; j < src->cols; ++j) {
        RGB currentColor(
          src->at<cv::Vec3b>(i, j)[2],
          src->at<cv::Vec3b>(i, j)[1],
          src->at<cv::Vec3b>(i, j)[0]
        );

        if (clickedColor == currentColor) {
          pixelMass += cv::Point(j, i);
          ++pixelCount;
        }
      }
    }

    cv::Point centerOfMass(
      pixelMass.x / pixelCount,
      pixelMass.y / pixelCount
    );

    DEBUG(
      "Pixel count (Area): {}",
      pixelCount
    );

    DEBUG(
      "Center of mass: {} {}",
      centerOfMass.x,
      centerOfMass.y
    );

    int numerator = 0;
    int denominator = 0;
    int minRow = -1, minCol = -1, maxRow = -1, maxCol = -1;
    auto unset = [](int a){ return a == -1; };
    cv::Mat elongation(src->rows, src->cols, CV_8UC3);
    for (int i = 0; i < src->rows; ++i) {
      for (int j = 0; j < src->cols; ++j) {
        RGB currentColor(
          src->at<cv::Vec3b>(i, j)[2],
          src->at<cv::Vec3b>(i, j)[1],
          src->at<cv::Vec3b>(i, j)[0]
        );
        elongation.at<cv::Vec3b>(i, j) = cv::Vec3b(currentColor.B(), currentColor.G(), currentColor.R());

        if (clickedColor == currentColor) {
          cv::Point currentPoint(j, i);
          cv::Point diff = currentPoint - centerOfMass;
          numerator += diff.x * diff.y;
          denominator += diff.x * diff.x - diff.y * diff.y;

          if (unset(minRow) || i < minRow)
            minRow = i;
          
          if (unset(minCol) || j < minCol)
            minCol = j;
          
          if (unset(maxRow) || i > maxRow)
            maxRow = i;

          if (unset(maxCol) || j > maxCol)
            maxCol = j;
        }
      }
    }

    float aspectRatio = (maxCol - minCol + 1.0f) / (maxRow - minRow + 1.0f);
    DEBUG(
      "Aspect ratio: {}",
      aspectRatio
    );

    // float thinness = 4.0f * CV_PI * (pixelCount / permieter);

    float phi = std::atan2(numerator * 2.0f, (float) denominator);
    phi /= 2.0f;

    phi = (phi * 180.0f) / CV_PI;

    if (phi < 0)
      phi += 180;

    DEBUG("Elongation angle: {}", phi);

    float weightX = std::sin(2 * CV_PI * (phi / 360.0f));
    float weightY = std::sin(2 * CV_PI * (phi / 360.0f));

  }
}

void geometricalFeatures(const cv::Mat &src)
{
  imshow("GeomFeature", src);
}

cv::Point firstBlackPixel(const cv::Mat &src) {
  for(int i=0; i<src.rows; ++i) {
    for (int j=0; j<src.cols; ++j) {
      if (src.at<uchar>(i, j) < 127) {
        return cv::Point(j, i);
      }
    }
  }
  return cv::Point(-1, -1);
}

void traceBorder(const cv::Mat &src)
{
  cv::Mat dst(src.rows, src.cols, CV_8UC3);
  for(int i = 0; i < src.rows; ++i)
    for(int j = 0; j < src.cols; ++j)
      dst.at<cv::Vec3b>(i,j) = src.at<uchar>(i,j) > 127 ? cv::Vec3b(255, 255, 255) : cv::Vec3b(0, 0, 0);

  const cv::Point directions[8] = {
    cv::Point(0, 1),
    cv::Point(-1, 1),
    cv::Point(-1, 0),
    cv::Point(-1, -1),
    cv::Point(0, -1),
    cv::Point(1, -1),
    cv::Point(1, 0),
    cv::Point(1, 1)
  };
  std::size_t dir = 7;
  cv::Point firstPx = firstBlackPixel(src);
  cv::Point currentPx = cv::Point(firstPx.x, firstPx.y);
  cv::Point secondPx = cv::Point(-1, -1);
  cv::Point nextPx = cv::Point(firstPx.x, firstPx.y);
  std::vector<std::size_t> AC, DC;
  int n = 0;

  do {
    currentPx = nextPx;
    ++n;
    const bool isOdd = (dir & 1);
    const std::size_t nextEven = (dir + 7) % 8;
    const std::size_t nextOdd = (dir + 6) % 8;
    dir = isOdd * nextOdd + !isOdd * nextEven;

    nextPx = currentPx + directions[dir];
    while (src.at<uchar>(nextPx.y, nextPx.x) > 127) {
      dir = (dir + 1) % 8;
      nextPx = currentPx + directions[dir];
    }
    AC.push_back(dir);

    dst.at<cv::Vec3b>(nextPx) = cv::Vec3b(0, 0, 255);

    if (n < 2) {
      firstPx = currentPx;
      secondPx = nextPx;
    }
  } while (!(n >= 2 && currentPx == firstPx && nextPx == secondPx));

  for (int i=0; i<AC.size(); ++i)
    DC.push_back((AC[i+1] - AC[i] + 8) % 8);

  imshow("edge traced", dst);
}

void reconstruct(const cv::Point &start, const std::vector<std::size_t> &AC) {
  cv::Mat mat(400, 600, CV_8UC1);
  for(int i = 0; i < mat.rows; ++i)
    for(int j = 0; j < mat.cols; ++j)
      mat.at<uchar>(i,j) = 255;

  const cv::Point directions[8] = {
    cv::Point(0, 1),
    cv::Point(-1, 1),
    cv::Point(-1, 0),
    cv::Point(-1, -1),
    cv::Point(0, -1),
    cv::Point(1, -1),
    cv::Point(1, 0),
    cv::Point(1, 1)
  };

  mat.at<uchar>(start) = 0;
  cv::Point currentPx = cv::Point(start.y, start.x);
  std::size_t n = AC.size();
  for (int i=0; i<n; ++i) {
    cv::Point nextPx = currentPx + directions[AC[i]];
    mat.at<uchar>(nextPx.x, nextPx.y) = 0;
    currentPx = nextPx;
  }

  imshow("reconstruction", mat);
}

using StructuringElement = std::tuple<cv::Mat, cv::Point>;

StructuringElement structuringElement(const std::size_t n = 3) {
  cv::Mat mat(n, n, CV_8UC1);
  for(int i = 0; i < mat.rows; ++i)
    for(int j = 0; j < mat.cols; ++j)
      mat.at<uchar>(i,j) = 255;

  // TODO generate from n
  cv::Point center = cv::Point(n / 2, n / 2);
  mat.at<uchar>(center) = 0;
  mat.at<uchar>(0, 1) = 0;
  mat.at<uchar>(1, 2) = 0;
  mat.at<uchar>(1, 0) = 0;
  mat.at<uchar>(2, 1) = 0;

  return std::make_tuple(mat, center);
}

cv::Mat dilation(const cv::Mat &src, const std::size_t n = 1)
{
  if (n == 0) return src;

  cv::Mat dst = src.clone();
  StructuringElement se = structuringElement(3);
  cv::Mat &seMate = std::get<0>(se);
  cv::Point &seAnchor = std::get<1>(se);

  for (int i=0; i<src.rows; ++i) {
    for (int j=0; j<src.cols; ++j) {
      if (src.at<uchar>(i, j) == 0) {
        for (int k=0; k<seMate.rows; ++k) {
          for (int l=0; l<seMate.cols; ++l) {
            if (seMate.at<uchar>(k, l) == 0) {
              int x = j + l - seAnchor.x;
              int y = i + k - seAnchor.y;
              if (x >= 0 && x < src.cols && y >= 0 && y < src.rows) {
                dst.at<uchar>(y, x) = 0;
              }
            }
          }
        }
      }
    }
  }

  return dilation(dst, n-1);
}

cv::Mat erosion(const cv::Mat &src, const std::size_t n = 1)
{
  if (n == 0) return src;

  cv::Mat dst = src.clone();
  StructuringElement se = structuringElement(3);
  cv::Mat &seMate = std::get<0>(se);
  cv::Point &seAnchor = std::get<1>(se);

  for (int i=0; i<src.rows; ++i) {
    for (int j=0; j<src.cols; ++j) {
      if (src.at<uchar>(i, j) == 0) {
        bool applicable = true;
        for (int k=0; k<seMate.rows; ++k) {
          for (int l=0; l<seMate.cols; ++l) {
            if (seMate.at<uchar>(k, l) == 0) {
              int x = j + l - seAnchor.x;
              int y = i + k - seAnchor.y;
              if (x >= 0 && x < src.cols && y >= 0 && y < src.rows) {
                if (src.at<uchar>(y, x) == 255) {
                  applicable = false;
                }
              }
            }
          }
        }
        if (applicable) {
          dst.at<uchar>(i, j) = 0;
        } else {
          dst.at<uchar>(i, j) = 255;
        }
      }
    }
  }

  return erosion(dst, n-1);
}

cv::Mat opening(const cv::Mat &src, const std::size_t n = 1)
{
  if (n == 0) return src;
  // return dilation(erosion(src, n), n);
  return opening(dilation(erosion(src)), n-1);
}

cv::Mat closing(const cv::Mat &src, const std::size_t n = 1)
{
  // return erosion(dilation(src, n), n);
  if (n == 0) return src;
  // return dilation(erosion(src, n), n);
  return closing(erosion(dilation(src)), n-1);
}

cv::Mat boundaryExtraction(const cv::Mat &src, const std::size_t n = 1)
{
  cv::Mat op = erosion(src, n);
  cv::Mat dst(src.rows, src.cols, CV_8UC1);

  for (int i=0; i<src.rows; ++i) {
    for (int j=0; j<src.cols; ++j) {
      // difference
      if (src.at<uchar>(i, j) != op.at<uchar>(i, j)) {
        dst.at<uchar>(i, j) = 255;
      } else {
        dst.at<uchar>(i, j) = 0;
      }
    }
  }

  return dst;
}

bool areIdentical(const cv::Mat &a, const cv::Mat &b)
{
  if (a.rows != b.rows || a.cols != b.cols) {
    return false;
  }

  for (int i=0; i<a.rows; ++i) {
    for (int j=0; j<a.cols; ++j) {
      if (a.at<uchar>(i, j) != b.at<uchar>(i, j)) {
        return false;
      }
    }
  }

  return true;
}

void regionFilling(const cv::Mat &src)
{
  cv::Mat dst = src.clone();
  cv::Mat predDst = src.clone();
  cv::Point center(src.rows / 2, src.cols / 2);

  dst.at<uchar>(center) = 0;

  while(!areIdentical(predDst, dst)) {
    cv::Mat dillated = dilation(dst);
    
  }
}

void morphologicPreview(const cv::Mat &src, const std::size_t n)
{
  cv::Mat dil = dilation(src, n);
  cv::Mat ero = erosion(src, n);
  cv::Mat opn = opening(src, n);
  cv::Mat cls = closing(src, n);
  cv::Mat border = boundaryExtraction(src, n);

  imshow("Input", src);
  imshow("Dilation", dil);
  imshow("Erosion", ero);
  imshow("Opening", opn);
  imshow("Closing", cls);
  imshow("Boundary", border);
}

int main() {
  Logger::init();

  DEBUG("Opening file {}", IMAGE("saturn.bmp"));
  cv::Mat img = FileUtils::readImage(IMAGE("saturn.bmp"), cv::IMREAD_GRAYSCALE);
  DEBUG("image loaded with {} {}", img.rows, img.cols);
  cv::Mat outImg(img.rows, img.cols, CV_8UC1);
  cv::Mat genRGB(img.rows, img.cols, CV_8UC3);

  cv::Mat flowers = FileUtils::readImage(IMAGE("flowers_24bits.bmp"), cv::IMREAD_COLOR);
  cv::Mat hueSpectrum = FileUtils::readImage(IMAGE("huespectrum.png"), cv::IMREAD_COLOR);
  cv::Mat objects = FileUtils::readImage(IMAGE("multiple/trasaturi_geometrice.bmp"), cv::IMREAD_COLOR);
  cv::Mat traceableBorder = FileUtils::readImage(IMAGE("border-tracing/object_holes.bmp"), cv::IMREAD_GRAYSCALE);
  cv::Mat testMorph = FileUtils::readImage(IMAGE("morphological_operations/3_Open/cel4thr3_bw.bmp"), cv::IMREAD_GRAYSCALE);

  std::ifstream input(IMAGE("border-tracing/reconstruct.txt"));
  int borderXStart, borderYStart, borderACLength;
  std::vector<std::size_t> borderAC;
  input >> borderXStart >> borderYStart >> borderACLength;
  for (int i=0; i<borderACLength; ++i) {
    std::size_t d;
    input >> d;
    borderAC.push_back(d);
  }
  input.close();

  INFO("Press the arrow keys to cycle through execution slides");

  cv::Mat roundRobinColor;
  cv::Mat roundRobinGray;
  std::vector<std::string> roundRobinNames = FileUtils::nestedFilesOf(IMAGE("morphological_operations/"));
  std::vector<std::function<void()>> roundRobinExecutors;
  std::transform(
    roundRobinNames.begin(),
    roundRobinNames.end(),
    std::back_inserter(roundRobinExecutors),
    [&](std::string path) -> std::function<void()> {
      return [path, &roundRobinColor, &roundRobinGray]() {
        DEBUG("Opening file {} in both color and gray scale", path);
        roundRobinGray = FileUtils::readImage(path, cv::IMREAD_GRAYSCALE);
        roundRobinColor = FileUtils::readImage(path, cv::IMREAD_COLOR);
      };
    }
  );
  Slider roundRobinSlider(roundRobinExecutors);
  roundRobinSlider.exec();

  // Processing functions that the user can slide through
  uchar thresh = 127;
  std::size_t counter = 1;
  Slider slider(
    { [&](){ bi_level_color_map(img, outImg, thresh); }
    , [&](){ traceBorder(traceableBorder); }
    , [&](){ reconstruct(cv::Point(borderYStart, borderXStart), borderAC); }
    , [&](){ morphologicPreview(roundRobinGray, counter); }
    // , [&](){ morphologicPreview(testMorph, counter); }
    }
  );

  // Loop
  std::string exportName = "";
  KEY operation = KEY::NONE;
  do {
    // Execute the current processing function
    slider.exec();

    // Input handling
    switch (operation)
    {
    case KEY::LEFT_ARROW:
      cv::destroyAllWindows();
      slider.previous();
      break;
    case KEY::RIGHT_ARROW:
      cv::destroyAllWindows();
      slider.next();
      break;
    case KEY::ENTER:
      FileUtils::quickSave(outImg);
      break;
    case KEY::SPACE:
      roundRobinSlider.next();
      roundRobinSlider.exec();
      break;
    case KEY::UP_ARROW:
      thresh += 10;
      ++counter;
      DEBUG("Counter {}", counter)
      break;
    case KEY::DOWN_ARROW:
      thresh -= 10;
      if (counter > 0)
        --counter;
      DEBUG("Counter {}", counter)
      break;
    default:
      break;
    }

    // Handle keypresses
    operation = WaitKey(200);
  } while (operation != KEY::ESC);

  Logger::destroy();
  return 0;
}
