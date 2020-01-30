cmake_minimum_required(VERSION 2.8.12)
project(simulator)

if (NOT TARGET pybind11)
  add_subdirectory(pybind11)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z -Wall -march=native -O3")

FIND_PACKAGE(Eigen3 REQUIRED)

if (${EIGEN3_VERSION_STRING} VERSION_LESS 3.3)
  set (EIGEN_OLD 1)
else ()
  set (EIGEN_OLD 0)
endif()

INCLUDE_DIRECTORIES(
    include
    ${EIGEN3_INCLUDE_DIRS}
    )

#list all source files
file(GLOB_RECURSE sources LIST_DIRECTORIES true src/*.cpp)
file(GLOB_RECURSE headers LIST_DIRECTORIES true include/*.h include/*.hpp)

if (NOT output_dir)
  set(output_dir ${CMAKE_SOURCE_DIR}/bin)
endif ()
message(STATUS "output dir = ${output_dir}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${output_dir})

PYBIND11_ADD_MODULE(simulator SHARED ${sources} ${headers})
if (NOT ${EIGEN_OLD})
  TARGET_LINK_LIBRARIES(simulator PRIVATE Eigen3::Eigen)
endif()
SET_TARGET_PROPERTIES(simulator PROPERTIES PREFIX "")

# write name of the file to MANIFEST file
# it allows the setup.py to use all modules
set(bin_list_file ${CMAKE_BINARY_DIR}/MANIFEST)
add_custom_command(TARGET simulator
    COMMAND ${CMAKE_COMMAND} -E
    echo $<TARGET_FILE:simulator> >> ${bin_list_file})
