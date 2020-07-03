//
// Created by Radim Baca on 01.07.20.
//
#include "csw.h"

csw::csw()
{
}

void csw::load_index(const char* file_name)
{
    HNSW hnsw(16, 16, 200);

    hnsw.loadGraph(file_name);
    hnsw.computeApproximateVector();

    int *group_assignement = new int[hnsw.actual_node_count_];

}