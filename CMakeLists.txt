cmake_minimum_required(VERSION 3.0.0)
project(chRMT VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(Eigen3 REQUIRED)
MESSAGE( [Main] " EIGEN3_INCLUDE_DIRS = ${EIGEN3_INCLUDE_DIRS}") 
# EIGEN3_INCLUDE_DIRS = /usr/include/eigen3
include_directories(${EIGEN3_INCLUDE_DIRS})

# SET(CMAKE_CXX_FLAGS "-lgsl -lgslcblas -lm -O3")
find_package(GSL REQUIRED)
link_libraries(GSL::gsl GSL::gslcblas)

if (MSVC)
  string(REGEX REPLACE "/W[0-9]" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
endif (MSVC)

# list(APPEND CMAKE_PREFIX_PATH /home/pesznyak/GSL)

add_executable(chRMT chRMT.cpp)
add_executable(chRMT_contour chRMT_contour.cpp)
add_executable(chRMT_contour_AdaDelta chRMT_contour_AdaDelta.cpp)

# target_link_libraries(chRMT PRIVATE Eigen3::Eigen)

set_target_properties(chRMT PROPERTIES CXX_STANDARD 17
                                                 CXX_STANDARD_REQUIRED ON
                                                 CXX_EXTENSIONS OFF)

target_compile_options(chRMT PRIVATE $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wall -Wextra -pedantic>
                                               $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->)

set_target_properties(chRMT_contour PROPERTIES CXX_STANDARD 17
                                               CXX_STANDARD_REQUIRED ON
                                               CXX_EXTENSIONS OFF)

target_compile_options(chRMT_contour PRIVATE $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wall -Wextra -pedantic>
                                             $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->)

set_target_properties(chRMT_contour_AdaDelta PROPERTIES CXX_STANDARD 17
                                                    CXX_STANDARD_REQUIRED ON
                                                    CXX_EXTENSIONS OFF)

target_compile_options(chRMT_contour_AdaDelta PRIVATE $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wall -Wextra -pedantic>
                                                      $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)