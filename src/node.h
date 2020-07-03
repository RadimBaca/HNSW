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
	pointer_t node_order;
	Node* node;

	Neighbors(const Neighbors& n)
	{
		this->distance = n.distance;
		this->node_order = n.node_order;
		this->node = n.node;
	}

	Neighbors(Node* n, float distance, pointer_t uniqueID)
	{
		this->node = n;
		this->distance = distance;
		this->node_order = uniqueID;
	}

//#define USE_SELECTED_TRESHOLD_SUMMARY
//    bool has_summary;
//#endif
};

struct neighborcmp_nearest {
	bool operator()(const Neighbors& a, const Neighbors& b) const {
		return a.distance < b.distance;
	}
};

struct neighborcmp_farest_heap {
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
	std::vector<Neighbors> neighbors;
#ifdef COMPUTE_APPROXIMATE_VECTOR
	std::vector<int8_t> summary;
#endif
	Node* lower_layer;
#ifdef COUNT_INWARD_DEGREE
	int inward_count;
#endif

	// runtime variables
	bool neighbors_sorted;
#ifndef VISIT_HASH
	int actual_node_count_;
#endif
//	int explored_count;

	Node(const int neighbor_size, Node* lower_layer)
		: lower_layer(lower_layer)
#ifndef VISIT_HASH
		, actual_node_count_(0)
#endif
#ifdef COUNT_INWARD_DEGREE
        , inward_count(0)
#endif
//        , explored_count(0)
	{
		neighbors.reserve(neighbor_size);
		neighbors_sorted = false;
	}

	Node(const Node& node) = delete;

	~Node()
	{

	}

	void copyInsertValues(const Node& node)
	{
		//distance = node.distance;
#ifndef VISIT_HASH
		actual_node_count_ = node.actual_node_count_;
#endif
	}

#ifdef COMPUTE_APPROXIMATE_VECTOR


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


};

