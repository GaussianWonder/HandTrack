# Project definition
cmake_minimum_required(VERSION 3.21)
project(opengl_template)

# Search all directories that contain headers
macro(HEADER_DIRECTORIES return_list)
  file(GLOB_RECURSE new_list *.h)
  set(dir_list "")
  foreach(file_path ${new_list})
    get_filename_component(dir_path ${file_path} PATH)
    set(dir_list ${dir_list} ${dir_path})
  endforeach()
  list(REMOVE_DUPLICATES dir_list)
  set(${return_list} ${dir_list})
endmacro()

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE STRING "" FORCE)
set(CMAKE_BUILD_TYPE Debug CACHE STRING "")
if(NOT CMAKE_BUILD_TYPE AND NOT (MSVC_IDE OR XCODE))
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type, one of: Release, Debug" FORCE)
endif()

add_definitions("-std=c++20")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -fcoroutines -fconcepts -fpermissive")

# Detect 64 or 32 bit
if (CMAKE_SIZEOF_VOID_P EQUAL 8)
  # 64-bit project
  SET(64_BIT_OS TRUE)
  message( STATUS "A 64-bit OS detected" )
else()
  SET(64_BIT_OS FALSE)
  message( STATUS "A 32-bit OS detected" )
endif()

if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
   message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
   file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/v0.16.1/conan.cmake"
                 "${CMAKE_BINARY_DIR}/conan.cmake")
endif()
include(${CMAKE_BINARY_DIR}/conan.cmake)

conan_cmake_configure(REQUIRES spdlog/1.9.2 GENERATORS CMakeDeps)
conan_cmake_autodetect(settings)
conan_cmake_install(PATH_OR_REFERENCE . BUILD missing SETTINGS ${settings})

set(CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR} ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR} ${CMAKE_PREFIX_PATH})

find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${CMAKE_CURRENT_LIST_DIR})

# Get all *.c files
file(GLOB_RECURSE Sources RELATIVE ${CMAKE_SOURCE_DIR} "src/*.cpp")
file(GLOB_RECURSE SBindings RELATIVE ${CMAKE_SOURCE_DIR} "bindings/*.cpp")

# Get all *.h files
file(GLOB_RECURSE Headers *.h)

if (DLIB_NO_GUI_SUPPORT)
  message("No GUI support, so we won't build the example.")
else()
  # Add everything to executable
  add_executable(${PROJECT_NAME} ${Headers} main.cpp ${Sources} ${SBindings})

  target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})

  # Get all header dirs to include
  header_directories(header_dir_list)

  list(LENGTH header_dir_list header_dir_list_count)
  message(STATUS "[INFO] Found ${header_dir_list_count} header directories.")

  target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    ${header_dir_list}
  )

  # Packages
  # Fast and Reliable logging
  find_package(spdlog CONFIG REQUIRED)
  target_link_libraries(${PROJECT_NAME} PUBLIC spdlog::spdlog)
endif()
