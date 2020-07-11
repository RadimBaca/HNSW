#pragma once

///////////////// Select one from these options
//#define MAIN_CONVERT_HDF_TO_BIN
//#define MAIN_UNIT_TESTING
//#define MAIN_RUN_CREATE_AND_QUERY
//#define MAIN_RUN_CREATE_AND_QUERY_WITHOUT_HDF5
//#define MAIN_UNIT_TESTING_CSW
#define MAIN_RUN_CSW

#define LOAD_GRAPH
constexpr char load_file[] = "sift_1M.bin";

//#define DEBUG_NET // TODO implement assert
#define COLLECT_STAT

#define VISIT_HASH // visited_ is implemented using a hash map

#define COMPUTE_APPROXIMATE_VECTOR // it basicaly means that we are going to use uint8 instead of float
//#define APR_DEBUG


#ifdef COMPUTE_APPROXIMATE_VECTOR

///////////////// Select one from these options
#define USE_PLAIN_CHAR // summary is not used at all
//#define USE_TRESHOLD_SUMMARY
//#define USE_TWO_FIXED_MAX


constexpr int DISTANCE_TRESHOLD = 16;

constexpr int EXPLORE_COUNT_TRESHOLD = 10;

constexpr float VECTOR_FRAGMENT1 = 0.1;
constexpr float VECTOR_FRAGMENT2 = 0.15;
#endif