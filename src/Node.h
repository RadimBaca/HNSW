#pragma once

#include <vector>
#include "settings.h"

class Node;

struct Neighbors
{
	float distance;
	Node* node;

	Neighbors(Node* node, float distance)
	{
		this->node = node;
		this->distance = distance;
	}
};


struct neighborcmp_nearest {
	bool operator()(const Neighbors& a, const Neighbors& b) const {
		return a.distance < b.distance;
	}
};

class Node
{
public:
	static uint32_t maxid;

	uint32_t uniqueId;
	uint32_t node_order;
	float* vector;
	std::vector<Neighbors> neighbors;
	Node* lower_layer;

#ifdef DEBUG_NET
	int layer;
#endif

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
		neighbors.reserve(neighbor_size);
		neighbors_sorted = false;
	}

	Node(const Node& node) = delete;

	~Node()
	{
		delete[] vector;
	}

	void copyInsertValues(const Node& node)
	{
		distance = node.distance;
#ifndef VISIT_HASH
		visit_id = node.visit_id;
#endif
	}

};


struct nodecmp_farest {
	bool operator()(const Node* a, const Node* b) const {
		return a->distance < b->distance;
	}
};

struct nodecmp_nearest {
	bool operator()(const Node* a, const Node* b) const {
		return a->distance > b->distance;
	}
};