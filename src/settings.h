#pragma once

//#define MAIN_CONVERT_HDF_TO_BIN
//#define MAIN_UNIT_TESTING
//#define MAIN_RUN_CREATE_AND_QUERY
#define MAIN_RUN_CREATE_AND_QUERY_WITHOUT_HDF5

//#define DEBUG_NET // TODO implement assert
#define COLLECT_STAT

#define VISIT_HASH // visited is implemented using a hash map

//#define SELECT_NEIGHBORS1 // uses the simplest algorithm to select neighbors (just according to a distance)
//#define DISTANCE_OPTIMIZATION

#define COMPUTE_APPROXIMATE_VECTOR
//#define APR_DEBUG

#ifdef COMPUTE_APPROXIMATE_VECTOR

// one of these variants should be selected
#define USE_PLAIN_CHAR
//#define USE_TRESHOLD_SUMMARY
//#define USE_TWO_FIXED_MAX


constexpr uint8_t DISTANCE_TRESHOLD = 16;

constexpr float VECTOR_FRAGMENT1 = 0.1;
constexpr float VECTOR_FRAGMENT2 = 0.15;
#endif