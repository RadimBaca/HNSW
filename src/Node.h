#pragma once

#include <vector>
#include <stdio.h>
#include <iostream>

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
#ifdef COMPUTE_APPROXIMATE_VECTOR
	uint8_t* apr_vector; // aproximated vector
	uint32_t apr_distance; // aproximated distance
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
	void computeApproximateVector(float min, float max, uint32_t vector_size)
	{
		float diff = max - min + 1;
		for (int i = 0; i < vector_size; i++)
		{
			apr_vector[i] =  ((vector[i] - min) / diff) * 256;
		}

	}

	static void computeApproximateVector(uint8_t *out, float* in, float min, float max, uint32_t vector_size)
	{
		float diff = max - min + 1;
		for (int i = 0; i < vector_size; i++)
		{
#ifdef APR_DEBUG
			if (in[i] < min) std::cout << "Query value is under min!!\n";
			if (in[i] > max) std::cout << "Query value is above max!!\n";
#endif
			out[i] = ((in[i] - min) / diff) * 256;
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