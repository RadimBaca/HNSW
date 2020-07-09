//
// Created by Radim Baca on 01.07.20.
//

#ifndef HNSW_CSW_H
#define HNSW_CSW_H

#include "hnsw.h"
#include "node_group.h"
#include <iostream>

constexpr int kMaxGroupsPerNode = 10;
constexpr int kMaxNodesPerGroup = 255;

// Clustered small world data structure
class Csw
{
    std::vector<NodeGroup> groups_;

    HNSW *hnsw;
    int *group_assignment_;
    int *node_group_;
    bool index_loaded_;

#ifdef COLLECT_STAT
    class Stat
    {
    public:
        Stat() : no_unassigned_nodes_(0), no_local_references_(0), no_global_references_(0) {}
        int no_unassigned_nodes_;
        int no_local_references_;
        int no_global_references_;
    };

    Stat statistics_;
#endif


    bool assignIntoGroup(int *group_assignment, NodeGroup &group, const int node_order);
    void groupMaterialization();
    void sortingNodesIntoGroups(const int N, const int X);

    inline int mergeGlobalId(int group_id, int local_order);
    inline int splitGlobalId(uint8_t & local_order, int global_id);
public:
    Csw();
    ~Csw();

    void load_index(const char* file_name, const int N, const int X);


    void printInfo();


};

int Csw::mergeGlobalId(int group_id, int local_order) {
    return (group_id << 8) + local_order;
}

int Csw::splitGlobalId(uint8_t & local_order, int global_id) {
    local_order = global_id;
    return global_id >> 8;
}

#endif //HNSW_CSW_H
