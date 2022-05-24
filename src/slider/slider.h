#ifndef __SLIDER_H__
#define __SLIDER_H__

#include <cstddef>
#include <memory>
#include <functional>

class Slider {
  using Executor = std::function<void()>;
  using Executors = std::vector<Executor>;
  Executors executors;

  std::size_t currentIndex = 0;
  std::size_t size = 0;
public:
  Slider(Executors executors);
  Slider(std::size_t size);
  ~Slider() = default;

  void next();
  void previous();
  void exec();

  std::size_t getCurrentIndex() const;
};

#endif