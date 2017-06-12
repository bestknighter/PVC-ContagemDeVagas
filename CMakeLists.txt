cmake_minimum_required (VERSION 3.0)

# Getting project folder to set as the project name and executable name
get_filename_component (CURR_DIR "${CMAKE_CURRENT_SOURCE_DIR}" NAME)

# Setting and configuring build types
set (CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Build types possible" FORCE)
set (CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type")
if (DEFINED CMAKE_BUILD_TYPE)
  set_property (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
endif ()

if (CMAKE_BUILD_TYPE MATCHES Debug)
    message ("## Configuring DEBUG version")
    set (Project_isDebug 1)
elseif (CMAKE_BUILD_TYPE MATCHES Release)
    set (Project_isDebug 0)
    message ("## Configuring RELEASE version")
endif (CMAKE_BUILD_TYPE MATCHES Debug)

set (CMAKE_CXX_FLAGS_DEBUG "-pg -ggdb")

# Setting project name and version
project (${CURR_DIR} VERSION 1.0.0.0)

# Passing some variables to the code
configure_file (
    "${CMAKE_SOURCE_DIR}/include/CMakeVars.hpp.in"
    "${CMAKE_SOURCE_DIR}/include/CMakeVars.hpp"
)

project (${PROJECT_NAME})

find_package (OpenCV REQUIRED)
if (OPENCV_NOT_FOUND) 
    message("OpenCV is required but was not found!")
endif ()

# Setting all include directories
include_directories ("./include" ${OpenCV_INCLUDE_DIRS})

# Getting all source files
file (GLOB SOURCES "src/*.cpp")

# Adding all necessary flags
list (APPEND CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++11 -fdiagnostics-color ${CMAKE_CXX_FLAGS}")

# Creating executable and linker settings
add_executable (${PROJECT_NAME} ${SOURCES})
target_link_libraries (${PROJECT_NAME} ${OpenCV_LIBS})

# Install is not used, hiding from the GUI
mark_as_advanced (CMAKE_INSTALL_PREFIX)
