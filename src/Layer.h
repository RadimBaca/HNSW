#pragma once

#include <vector>

#include "Node.h"
#include "settings.h"


class Layer
{
	static int max_layer;
public:	
	int layer_id;
	Node* enter_point;
	uint32_t node_count;
	std::vector<Node*> nodes;

	Layer(Node* n)
	{
		enter_point = n;
		node_count = 1;
		layer_id = Layer::max_layer++;
		nodes.push_back(n);
	}

	Layer(const Layer& layer) = delete;
};

