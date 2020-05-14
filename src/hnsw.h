#pragma once

#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <assert.h>


#include "Node.h"
#include "Layer.h"
#include "settings.h"
#include "hdfReader.h"

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
#ifdef COLLECT_STAT
	class HNSW_Stat
	{
	public:
		int distinct_computations;
		int distinct_computations_false;
		int distinct_computations_ended; // TODO - remove
		int aproximate_computations;

		HNSW_Stat()
		{
			clear();
		}

		void clear()
		{
			distinct_computations = 0;
			distinct_computations_false = 0;
			distinct_computations_ended = 0;
			aproximate_computations = 0;
		}

		void print()
		{
			std::cout << "No. distance computations: " << distinct_computations << "\n";
			std::cout << "No. false distance computations: " << distinct_computations_false << "\n";
			std::cout << "No. approximate distance computations: " << aproximate_computations << "\n";
		}
	};
#endif

public:


	int M;
	int Mmax;
	int Mmax0;
	int efConstruction;
	float ml;

	int node_count;
	std::vector<std::unique_ptr<Layer>> layers;

	int visit_id;
	std::vector<Node*> W;
#ifdef VISIT_HASH
	linearHash visited;
#endif
#ifdef COLLECT_STAT
	HNSW_Stat stat;
#endif
#ifdef COMPUTE_APPROXIMATE_VECTOR
	float* min_vector;
	float* max_vector;
	float min_value;
	float max_value;
	uint8_t* apr_q;
#endif

	uint32_t vector_count;
	uint32_t vector_size;


	HNSW(int M, int Mmax, int efConstruction, float ml)
		:M(M),
		Mmax(Mmax),
		Mmax0(Mmax*2),
		efConstruction(efConstruction), 
		ml(ml), 
		node_count(0), 
		visit_id(-1)
	{ 
	}

	void create(const char* filename, const char* datasetname);
	void query(const char* filename, const char* querydatasetname, const char* resultdatasetname, int ef);

	void insert(float* q);
	void knn(float* q, int k, int ef);
#ifdef COMPUTE_APPROXIMATE_VECTOR
	void computeApproximateVector();
#endif
	void printInfo(bool all);

	void setVectorSize(uint32_t vsize)
	{
		vector_size = vsize;
	}

private: 

	void search_layer_one(float* q);
	void search_layer(float* q, int ef);
	void select_neighbors(std::vector<Node*>& R, int M, bool keepPruned);

#ifdef COMPUTE_APPROXIMATE_VECTOR
	void apr_search_layer(uint8_t* q, int ef);
#endif
#ifndef SELECT_NEIGHBORS1
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
#endif

	inline float distance(float* q, float* node)
	{
#ifdef COLLECT_STAT
		stat.distinct_computations++;
#endif

		//size_t s = vector_size >> 3;
		//float result = 0;
		//int i = 0;
		//for (i = 0; i < vector_size; i+=8)
		//{
		//	float t = q[i] - node[i];
		//	float r1 = t * t;
		//	t = q[i + 1] - node[i + 1];
		//	float r2 = t * t;
		//	t = q[i + 2] - node[i + 2];
		//	float r3 = t * t;
		//	t = q[i + 3] - node[i + 3];
		//	float r4 = t * t;
		//	t = q[i + 4] - node[i + 4];
		//	float r5 = t * t;
		//	t = q[i + 5] - node[i + 5];
		//	float r6 = t * t;
		//	t = q[i + 6] - node[i + 6];
		//	float r7 = t * t;
		//	t = q[i + 7] - node[i + 7];
		//	float r8 = t * t;
		//	result += r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8;
		//}
		//for (; i < vector_size; i++)
		//{
		//	float t = q[i] - node[i];
		//	result += t * t;
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

#ifdef COMPUTE_APPROXIMATE_VECTOR
	inline uint32_t apr_distance(uint8_t* q, uint8_t* node)
	{
#ifdef COLLECT_STAT
		stat.aproximate_computations++;
#endif
		uint32_t result = 0;
		for (unsigned int i = 0; i < vector_size; i++)
		{
			int32_t t = q[i] - node[i];
			result += t * t;
		}
		return result;
	}
#endif


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


};

