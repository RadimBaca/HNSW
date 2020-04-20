#pragma once

#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>

#include "hdf5.h"
#include "H5Cpp.h"

#include "Node.h"
#include "Layer.h"
#include "settings.h"

using namespace H5;



class HNSW
{
public:
	int M;
	int Mmax;
	int Mmax0;
	int efConstruction;
	float ml;

	uint32_t vector_count;
	uint32_t vector_size;
	int node_count;
	std::vector<std::unique_ptr<Layer>> layers;

	int visit_id;
	std::vector<Node*> W;

	HNSW(int M, int Mmax, int efConstruction, float ml)
		:M(M),
		Mmax(Mmax),
		Mmax0(Mmax*2),
		efConstruction(efConstruction), 
		ml(ml), 
		node_count(0), 
		visit_id(-1)
	{ }

	void create(const char* filename, const char* datasetname);
	void query(const char* filename, const char* querydatasetname, const char* resultdatasetname);

	void insert(float* q);
	void knn(float* q, int k, int ef);

	void printInfo();

private: 
	void search_layer(float* q, int ef);

	double distance(float* q, float* node)
	{
		double result = 0;
		for (unsigned int i = 0; i < vector_size; i++)
		{
			result += (q[i] - node[i]) * (q[i] - node[i]);
		}
		return result;
	}

	void change_layer()
	{
		for (int i = 0; i < W.size(); i++)
		{
			auto node = W[i];
			W[i] = W[i]->lower_layer;
			W[i]->copyInsertValues(*node);
		}
	}

	// HDF5 functions
	void getDimensions(const char* filename, const char* datasetname, hsize_t(*dimensions)[2]);
	void readData(const char* filename, const char* datasetname, float* data);
	void readData(const char* filename, const char* datasetname, int* data);

};

