#pragma once


#include "hdf5.h"
#include "H5Cpp.h"

using namespace H5;

class hdfReader
{
public:
	// HDF5 functions
	static void getDimensions(const char* filename, const char* datasetname, hsize_t(*dimensions)[2]);
	static void readData(const char* filename, const char* datasetname, float* data);
	static void readData(const char* filename, const char* datasetname, int* data);

};

