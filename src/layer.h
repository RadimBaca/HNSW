#pragma once

#include <vector>

#include "node.h"
#include "settings.h"


class Layer
{
	static int max_layer;
public:	
	int layer_id;
	Node* enter_point;
	pointer_t ep_node_order;
	uint32_t node_count;
	std::vector<Node*> nodes;

	Layer(Node* n, pointer_t n_node_order)
	{
		enter_point = n;
		ep_node_order = n_node_order;
		node_count = 1;
		layer_id = Layer::max_layer++;
		nodes.push_back(n);
	}

	Layer() {}

	~Layer() {
	    for(auto& n: nodes)
        {
	        delete n;
        }
	}

	Layer(const Layer& layer) = delete;
};

