//
// Created by bac027 on 03.07.20.
//

#ifndef HNSW_NODE_GROUP_H
#define HNSW_NODE_GROUP_H

#include <vector>
#include "settings.h"
#include "node.h"

struct GuideGroup {
    int summary;
    int local;
    uint8_t local_count;
    int global;
    uint8_t global_count;
};

class NodeGroup
{
public:
    static int max_count;

    int group_id_;
    std::vector<pointer_t> nodes_;
    uint8_t center_node_index_;
    GuideGroup* guidepost_;
    std::vector<int8_t> summary_; // TODO vector can be easily replaced by a simple array
    uint8_t* local_neighbors_;
    std::pair<int,int> *global_neighbors_; // group_id + node_index, node_order

    NodeGroup() :group_id_(max_count++),
                 guidepost_(nullptr),
                 local_neighbors_(nullptr),
                 global_neighbors_(nullptr)
    {

    }

    ~NodeGroup() {
        if (guidepost_ != nullptr) {
            delete[] guidepost_;
        }
        if (local_neighbors_ != nullptr) {
            delete[] local_neighbors_;
        }
        if (global_neighbors_ != nullptr) {
            delete[] global_neighbors_;
        }
    }
};

#endif //HNSW_NODE_GROUP_H
