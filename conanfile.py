from conans import ConanFile, tools
from conan.tools.cmake import CMake
import shutil
import os

class ImageProcessingTemplate(ConanFile):
  name = "hand_track"
  settings = "os", "compiler", "build_type", "arch"
  requires = (
    "spdlog/1.9.2",
    # "opencv/4.5.3", # insane build times, this is included separately, make sure find_package works for opencv
  )
  generators = "CMakeToolchain", "CMakeDeps"

  # External sources to be included in the project
  def source(self):
    pass

  # Bindings to be included in the project
  def imports(self):
    pass

  def build(self):
    cmake = CMake(self)
    cmake.configure()
    cmake.build()
