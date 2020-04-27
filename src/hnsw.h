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


class linearHash
{
	uint32_t actual_size;
	uint32_t mask;
	uint32_t* hasharray;
	uint32_t item_count;

public:
	linearHash(uint32_t asize = 16384)
	{
		item_count = 0;
		actual_size = asize;
		mask = asize - 1;
		hasharray = new uint32_t[actual_size];
	}

	void clear()
	{
		item_count = 0;
		memset(hasharray, -1, sizeof(uint32_t) * actual_size);
	}

	void insert(uint32_t index)
	{
		//if (item_count > (actual_size >> 1))
		//{
		//	std::cout << "Hash array should be resized!" << "\n";
		//}

		uint32_t hash = index & mask;
		while (hasharray[hash] != -1)
		{
			hash++;
			if (hash >= actual_size) {
				hash = 0;
			}
		}
		item_count++;
		hasharray[hash] = index;
	}

	bool get(uint32_t index)
	{
		uint32_t hash = index & mask;
		while (hasharray[hash] != -1)
		{
			if (hasharray[hash] == index)
			{
				return true;
			}
			hash++;
			if (hash >= actual_size) {
				hash = 0;
			}
		}
		return false;
	}
};


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
#ifdef VISIT_HASH
	linearHash visited;
#endif

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
	void query(const char* filename, const char* querydatasetname, const char* resultdatasetname, int ef);

	void insert(float* q);
	void knn(float* q, int k, int ef);

	void printInfo();

private: 
	void search_layer(float* q, int ef);
	void select_neighbors(std::vector<Node*>& R, int M, bool keepPruned);
	inline bool is_distant_node(std::vector<Neighbors>& neighbors, float* node, float dist_from_q)
	{
		for (int i = 0; i < neighbors.size(); i++)
		{
			float dist = distance(node, neighbors[i].node->vector);
			if (dist < dist_from_q)
			{
				return false;
			}
		}
		return true;
	}

	inline float distance(float* q, float* node)
	{
		//size_t s = vector_size >> 2;
		//float result = 0;
		//for (unsigned int i = 0; i < vector_size; i+=4)
		//{
		//	float t = q[i] - node[i];
		//	float r1 = t * t;
		//	t = q[i + 1] - node[i + 1];
		//	float r2 = t * t;
		//	t = q[i + 2] - node[i + 2];
		//	float r3 = t * t;
		//	t = q[i + 3] - node[i + 3];
		//	float r4 = t * t;
		//	result += r1 + r2 + r3 + r4;
		//}
		//return result;

		float result = 0;
		for (unsigned int i = 0; i < vector_size; i++)
		{
			float t = q[i] - node[i];
			result += t * t;
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
#ifdef VISIT_HASH
			visited.insert(W[i]->uniqueId);
#endif
		}
	}

	// HDF5 functions
	void getDimensions(const char* filename, const char* datasetname, hsize_t(*dimensions)[2]);
	void readData(const char* filename, const char* datasetname, float* data);
	void readData(const char* filename, const char* datasetname, int* data);

};

