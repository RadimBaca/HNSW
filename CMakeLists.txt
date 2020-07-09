cmake_minimum_required (VERSION 2.6)
project (hnsw)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -fno-rtti -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -fsanitize=address -static-libasan")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -flto -ftree-vectorize -march=native")

set(SOURCE_FILES src/settings.h src/hnsw.h src/layer.h src/node.h src/csw.h src/csw.cpp src/node_group.h src/node_group.cpp)

add_executable(hnsw src/main.cpp src/hnsw.cpp src/layer.cpp src/node.cpp ${SOURCE_FILES})
