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
    struct SearchItem {
        int group_id_;
        uint8_t node_index_;
        //int node_order_;
        uint32_t distance_;

        SearchItem(int group_id, uint8_t node_order_in_group, uint32_t distance)
            : group_id_(group_id)
            , node_index_(node_order_in_group)
            //, node_order_(node_order)
            , distance_(distance)
        {
        }
    };
    struct CompareSearchItemHeap {
        constexpr bool operator()(SearchItem& i1, SearchItem& i2) const noexcept {
            return i1.distance_ > i2.distance_;
        }
    };
    struct CompareSearchItemHeapReverse {
        constexpr bool operator()(SearchItem& i1, SearchItem& i2) const noexcept {
            return i1.distance_ < i2.distance_;
        }
    };

    std::vector<NodeGroup> groups_;

    HNSW *hnsw;
    int *group_assignment_;
    int *node_group_; // for each node
    bool index_loaded_;

    // query variables
    std::vector<int> result_;
    std::vector<SearchItem> sq_;

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
    void groupSearchLayer(uint8_t* q, int ef);

    inline int mergeGlobalId(int group_id, int node_index);
    inline int splitGlobalId(uint8_t & node_index, int global_id);
public:
    Csw();
    ~Csw();

    void load_index(const char* file_name, const int N, const int X);
    std::vector<int>& knn(float* q, int ef);

    void printInfo();


};

int Csw::mergeGlobalId(int group_id, int node_index) {
    return (group_id << 8) + node_index;
}

int Csw::splitGlobalId(uint8_t & node_index, int global_id) {
    node_index = global_id;
    return global_id >> 8;
}

#endif //HNSW_CSW_H
