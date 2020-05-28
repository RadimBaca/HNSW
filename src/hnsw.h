#pragma once

#include <stdio.h>
#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <random>
#include <cmath>
#include <chrono>
#include <assert.h>
#include <intrin.h>

#include "Node.h"
#include "Layer.h"
#include "settings.h"
#include "hdfReader.h"

/*
Naive implementation of linear hash array.
Fixed size, without overflow check.
TODO - Should be replaced by a bloom filter or cuckoo filter.
*/
class linearHash
{
	uint32_t actual_size;
	uint32_t mask;
	uint32_t* hasharray;

public:

	uint32_t item_count;

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


#ifdef COLLECT_STAT
class HNSW_Stat
{
public:
	int precise_distance_computations;
	int distance_computations;
	int distance_computations_false;
	int explored_nodes;
	int overflows;
	int neighbor_count;
	int node_count;

	HNSW_Stat()
	{
		clear();
	}

	void clear()
	{
		precise_distance_computations = 0;
		distance_computations = 0;
		distance_computations_false = 0;
		explored_nodes = 0;
		overflows = 0;
		neighbor_count = 0;
		node_count = 0;
	}

	void print()
	{
		std::cout << "No. precise distance computations: " << precise_distance_computations << "\n";
		std::cout << "No. distance computations: " << distance_computations << "\n";
		std::cout << "No. explored node: " << explored_nodes << "\n";
		std::cout << "No. overflows: " << overflows << "\n";
		std::cout << "No. neighbors (avg. neighbor): " << neighbor_count << " (" << (float)neighbor_count / node_count << ")\n";
	}
};
#endif


struct CompareByFirst {
	constexpr bool operator()(std::pair<uint32_t, Node*> i1, std::pair<uint32_t, Node*> i2) const noexcept
	{
		return i1.first < i2.first;
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

	int node_count;
	long long index_memory;
	std::vector<std::unique_ptr<Layer>> layers;

	int visit_id;
	std::vector<Node*> W;
	std::priority_queue<std::pair<uint32_t, Node*>, std::vector<std::pair<uint32_t, Node*>>, CompareByFirst> WP;
	std::unordered_map<pointer_t, uint32_t> distances;
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
	int shift;
	uint8_t* apr_q;
	int one_vector_mask_size;
	int* q_delta;


	// toto neni potreba (muze byt soucasti pocitani v distance)
	uint8_t query_positive_avg[2]; // average distance of regions (LPQ, SPQ)
	uint8_t query_negative_avg[2]; // average distance of regions (LNQ, SNQ)
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
	void select_neighbors(std::vector<Node*>& R, int M, bool keepPruned, bool considerOverTreshold);

#ifdef COMPUTE_APPROXIMATE_VECTOR
	void apr_search_layer_one(uint8_t* q);
	void apr_search_layer(uint8_t* q, int ef);
	void apr_search_layer_summary(uint8_t* q, int ef);
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
		stat.precise_distance_computations++;
#endif

		float result = 0;
		for (unsigned int i = 0; i < vector_size; i++)
		{
			float t = q[i] - node[i];
			result += t * t;
		}
		return result;
	}

	inline float distance_treshold_counting(float* q, float* node, uint32_t& no_over_treshold)
	{
#ifdef COLLECT_STAT
		stat.precise_distance_computations++;
#endif

		float result = 0;
		no_over_treshold = 0;
		for (unsigned int i = 0; i < vector_size; i++)
		{
			float t = q[i] - node[i];
			result += t * t;
			no_over_treshold += t > DISTANCE_TRESHOLD || -t > DISTANCE_TRESHOLD;
		}
		return result;
	}

#ifdef COMPUTE_APPROXIMATE_VECTOR
	inline uint32_t apr_distance(uint8_t* q, uint8_t* node)
	{
#ifdef COLLECT_STAT
		stat.distance_computations++;
#endif
		uint32_t result = 0;
		for (unsigned int i = 0; i < vector_size; i++)
		{
			int32_t t = q[i] - node[i];
			result += t * t;
		}
		return result;
	}

	inline uint32_t apr_distance(uint8_t* q, uint8_t* node, int* q_delta)
	{
#ifdef COLLECT_STAT
		stat.distance_computations++;
#endif
		uint32_t result = 0;
		for (unsigned int i = 0; i < vector_size; i++)
		{
			int32_t t = q[i] - node[i];
			result += t * t;
			q_delta[i] = t;
		}
		return result;
	}


	inline uint32_t apr_distance_summary(std::vector<int8_t>&node_summary, int& node_summary_position, int* q_delta, uint32_t pivot_distance)
	{
#ifdef COLLECT_STAT
		stat.distance_computations++;
#endif
		uint32_t result = pivot_distance;
		int c = node_summary[node_summary_position] * 2 + 1 + node_summary_position;
		for (int i = node_summary_position + 1; i < c; i+=2)
		{
			int8_t position = node_summary[i];
			int z = node_summary[i + 1];
			result -= 2 * q_delta[position] * z - z * z;
		}
		node_summary_position = c;
		
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

