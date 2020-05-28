#pragma once

#include <vector>
#include <stdio.h>
#include <iostream>
#include <queue>

#include "settings.h"

class Node;
class HNSW_Stat;

using pointer_t = uint32_t;


//////////////////////////////////////////////// Struct Neighbors
struct Neighbors
{
	float distance;
	pointer_t uniqueId;
	Node* node;

	Neighbors(Node* node, float distance, pointer_t uniqueID)
	{
		this->node = node;
		this->distance = distance;
		this->uniqueId = uniqueID;
	}


};

struct neighborcmp_nearest {
	bool operator()(const Neighbors& a, const Neighbors& b) const {
		return a.distance < b.distance;
	}
};

struct neighborcmp_nearest_heap {
	bool operator()(const Neighbors& a, const Neighbors& b) const {
		return a.distance > b.distance;
	}
};


struct CompareByDelta {
	constexpr bool operator()(std::pair<int, uint8_t> i1, std::pair<int, uint8_t> i2) const noexcept
	{
		return i1.first < i2.first;
	}
};

//////////////////////////////////////////////// Class Node
class Node
{
public:
	static uint32_t maxid;

	pointer_t uniqueId;
	uint32_t node_order;
	float* vector;
#ifdef COMPUTE_APPROXIMATE_VECTOR
	uint8_t* apr_vector; // aproximated vector
	int32_t apr_distance; // aproximated distance
	uint32_t apr_precise_computation; // if false, then the distance was computed just from summary
	uint32_t no_over_treshold; // number of over treshold values in the vector

	std::vector<int8_t> summary;
#endif
	std::vector<Neighbors> neighbors;
	Node* lower_layer;

	// runtime variables
	float distance;
	bool neighbors_sorted;
#ifndef VISIT_HASH
	int visit_id;
#endif

	Node(int node_order, const int vector_size, const int neighbor_size, Node* lower_layer)
		: lower_layer(lower_layer)
#ifndef VISIT_HASH
		,visit_id(0)
#endif
	{
		this->node_order = node_order;
		uniqueId = maxid++;
		vector = new float[vector_size];
#ifdef COMPUTE_APPROXIMATE_VECTOR
		apr_vector = new uint8_t[vector_size];
#endif
		neighbors.reserve(neighbor_size);
		neighbors_sorted = false;
	}


	Node(int node_order, const int vector_size, const int neighbor_size, Node* lower_layer, float* v)
		: lower_layer(lower_layer)
	{
		this->node_order = node_order;
		uniqueId = maxid++;
		vector = new float[vector_size];
		memcpy(vector, v, sizeof(float) * vector_size);
#ifdef COMPUTE_APPROXIMATE_VECTOR
		apr_vector = new uint8_t[vector_size];
#endif
		neighbors.reserve(neighbor_size);
		neighbors_sorted = false;
	}

	Node(const Node& node) = delete;

	~Node()
	{
		delete[] vector;
#ifdef COMPUTE_APPROXIMATE_VECTOR
		delete[] apr_vector;
#endif

	}

	void copyInsertValues(const Node& node)
	{
		distance = node.distance;
#ifdef COMPUTE_APPROXIMATE_VECTOR
		apr_distance = node.apr_distance;
#endif
#ifndef VISIT_HASH
		visit_id = node.visit_id;
#endif
	}

#ifdef COMPUTE_APPROXIMATE_VECTOR
	void computeApproximateVector( int shift, uint32_t vector_size)
	{
		for (int i = 0; i < vector_size; i++)
		{
			unsigned int aux = vector[i];
			apr_vector[i] = aux >> shift;
		}
	}


	int computeSummaries(uint8_t* pivot, uint32_t vector_size, int* overflows)
	{
		//neighbors_all = new uint8_t[neighbors.size() * vector_size];
		//int i = 0;
		//for (auto ng : neighbors)
		//{
		//	memcpy(&neighbors_all[i * vector_size], ng.node->apr_vector, vector_size);
		//	i++;
		//}

#ifdef USE_JUST_MAX
		auto partition = (vector_size >> COUNT_SHIFT) == 0 ? 1 : vector_size >> COUNT_SHIFT;
		std::priority_queue<std::pair<int, uint8_t>, std::vector<std::pair<int, uint8_t>>, CompareByDelta> delta;
		int byteSum = neighbors.size();
		for (auto ng : neighbors)
		{
			auto n = ng.node;
			uint8_t over_treshold = 0;
			for (int i = 0; i < vector_size; i++)
			{
				int diff = n->apr_vector[i] - pivot[i];
				if (diff > DISTANCE_TRESHOLD)
				{
					if (diff > 127)
					{
						diff = 127;
						(*overflows)++;
					}
					delta.emplace(diff, i);
					over_treshold++;
				}
				if (-diff > DISTANCE_TRESHOLD)
				{
					diff = -diff;
					if (diff < -127)
					{
						diff = -127;
						(*overflows)++;
					}
					delta.emplace(diff, i);
					over_treshold++;
				}
			}

			int8_t node_count = over_treshold < partition ? over_treshold : partition;
			byteSum += node_count * 2;
			summary.push_back(node_count);
			int i = 0;
			while (i < partition && !delta.empty())
			{
				auto item = delta.top();
				delta.pop();
				summary.push_back(item.second);
				int8_t diff = n->apr_vector[item.second] - pivot[item.second];
				summary.push_back(diff);
				i++;
			}
			while (!delta.empty())
			{
				delta.pop();
			}
		}
		return byteSum;
#else
		int neighbor_array_size = 0;
		for (auto ng : neighbors)
		{
			auto n = ng.node;

			uint8_t over_treshold = 0;
			for (int i = 0; i < vector_size; i++)
			{
				neighbor_array_size += n->apr_vector[i] - pivot[i] > DISTANCE_TRESHOLD || pivot[i] - n->apr_vector[i] > DISTANCE_TRESHOLD;
			}
		}
		summary.reserve(neighbors.size() + 2 * neighbor_array_size); // TODO - can be dangerous for higher dimensions than 256


		for (auto ng : neighbors)
		{
			auto n = ng.node;

			uint8_t over_treshold = 0;
			for (int i = 0; i < vector_size; i++)
			{
				over_treshold += n->apr_vector[i] - pivot[i] > DISTANCE_TRESHOLD || pivot[i] - n->apr_vector[i] > DISTANCE_TRESHOLD;
			}

			summary.push_back(over_treshold);
			for (int i = 0; i < vector_size; i++)
			{
				int diff = (int)n->apr_vector[i] - pivot[i];
				if (diff > 127)
				{
					diff = 127;
					(*overflows)++;
				}

				if (diff < -127)
				{
					diff = -127;
					(*overflows)++;
				}

				if (diff > DISTANCE_TRESHOLD || -diff > DISTANCE_TRESHOLD)
				{
					summary.push_back(i);
					summary.push_back(diff);
				}
			}
		}

		return neighbors.size() + 2 * neighbor_array_size;
#endif
	}	

	static void computeApproximateVector(uint8_t *out, float* in, int shift, uint32_t vector_size)
	{
		for (int i = 0; i < vector_size; i++)
		{
#ifdef APR_DEBUG
			if (in[i] < min) std::cout << "Query value is under min!!\n";
			if (in[i] > max) std::cout << "Query value is above max!!\n";
#endif
			unsigned int aux = in[i];
			out[i] = aux >> shift;
		}
	}
#endif

	void print(int size)
	{
		for (int i = 0; i < size; i++)
		{
			std::cout << vector[i] << ", ";
		}
		std::cout << "D: " << distance << ", Uid: " << uniqueId << ", Node order: " << node_order;
#ifdef COMPUTE_APPROXIMATE_VECTOR
		std::cout << "\n";
		for (int i = 0; i < size; i++)
		{
			std::cout << (uint32_t)apr_vector[i] << ", ";
		}
		std::cout << "D: " << apr_distance << ", Uid: " << uniqueId;
#endif
		std::cout << "\n\n";
	}

};


struct nodecmp_farest {
	bool operator()(const Node* a, const Node* b) const {
		return a->distance < b->distance;
	}
};

struct nodecmp_farest_with_overflow {
	bool operator()(const Node* a, const Node* b) const {
		return a->no_over_treshold < b->no_over_treshold ||
			(a->no_over_treshold == b->no_over_treshold && a->distance < b->distance);
	}
};



struct nodecmp_nearest {
	bool operator()(const Node* a, const Node* b) const {
		return a->distance > b->distance;
	}
};

#ifdef COMPUTE_APPROXIMATE_VECTOR
struct apr_nodecmp_farest {
	bool operator()(const Node* a, const Node* b) const {
		return a->apr_distance < b->apr_distance;
	}
};

struct apr_nodecmp_nearest {
	bool operator()(const Node* a, const Node* b) const {
		return a->apr_distance > b->apr_distance;
	}
};
#endif