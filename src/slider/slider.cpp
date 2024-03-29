#include "slider.h"
#include "logger.h"

using Executor = std::function<void()>;
using Executors = std::vector<Executor>;

Slider::Slider(Executors executors)
  :executors(executors)
{
  size = executors.size();
}

Slider::Slider(std::size_t size)
  :size(size)
  ,executors({})
{}

void Slider::next()
{
  currentIndex = currentIndex == size - 1 ? 0 : currentIndex + 1;
  DEBUG("Next");
}

void Slider::previous()
{
  currentIndex = currentIndex == 0 ? size - 1 : currentIndex - 1;
  DEBUG("Previous");
}

void Slider::exec()
{
  executors[currentIndex]();
}

std::size_t Slider::getCurrentIndex() const
{
  return currentIndex;
}
