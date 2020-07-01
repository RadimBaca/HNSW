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

    hnsw.loadKNNG(file_name);
    hnsw.computeApproximateVector();

}