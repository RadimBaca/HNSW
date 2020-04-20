#include <chrono>

#include "HNSW.h"

#define FILE_NAME       "sift-128-euclidean.hdf5"


int main(void)
{
    //std::vector<Node*> W;
    //W.push_back(new Node(1, 1, nullptr));
    //W.push_back(new Node(1, 1, nullptr));
    //W.push_back(new Node(1, 1, nullptr));
    //W[0]->distance = 1;
    //W[1]->distance = 3;
    //W[2]->distance = 2;
    //std::make_heap(W.begin(), W.end(), nodecmp_farest());
    //Node* farest = W.front();

    //W.push_back(new Node(1, 1, nullptr));
    //W.back()->distance = 10;
    //std::push_heap(W.begin(), W.end(), nodecmp_farest());
    //farest = W.front();
    //
    //std::pop_heap(W.begin(), W.end(), nodecmp_farest());
    //W.pop_back();
    //farest = W.front();

    //return 0;

    auto start = std::chrono::system_clock::now();

    HNSW hnsw(20, 40, 100, 0.5);
    hnsw.create(FILE_NAME, "train");

    auto end = std::chrono::system_clock::now();
    double dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Insert time " << dur / 1000 << " [s] \n";
    hnsw.printInfo();
    hnsw.query(FILE_NAME, "test", "neighbors");

    return 0;
}

