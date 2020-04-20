#pragma once

#include <vector>
#include "settings.h"

class Node;

struct Neighbors
{
	double distance;
	Node* node;

	Neighbors(Node* node, double distance)
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
	double distance;
	int visit_id;

	Node(int node_order, const int vector_size, const int neighbor_size, Node* lower_layer)
		: lower_layer(lower_layer), visit_id(0)
	{
		this->node_order = node_order;
		uniqueId = maxid++;
		vector = new float[vector_size];
		neighbors.reserve(neighbor_size);
	}


	Node(int node_order, const int vector_size, const int neighbor_size, Node* lower_layer, float* v)
		: lower_layer(lower_layer)
	{
		this->node_order = node_order;
		uniqueId = maxid++;
		vector = new float[vector_size];
		memcpy(vector, v, sizeof(float) * vector_size);
		neighbors.reserve(neighbor_size);
	}

	Node(const Node& node) = delete;

	~Node()
	{
		delete[] vector;
	}

	void copyInsertValues(const Node& node)
	{
		distance = node.distance;
		visit_id = node.visit_id;
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