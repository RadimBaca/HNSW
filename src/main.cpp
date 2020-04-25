#include <chrono>

#include "HNSW.h"

#define FILE_NAME       "sift-128-euclidean.hdf5"

int main(void)
{

    //linearHash a;

    //a.insert(119);
    //a.insert(120);
    //a.insert(4216);
    //a.insert(120 + 4096 + 4096);
    //a.insert(88);
    //a.insert(511);
    //a.insert(4096 - 1);
    //a.insert(2 * 4096 - 1);

    //if (a.get(120)) std::cout << "120 nalezen!\n";
    //if (!a.get(121)) std::cout << "121 nenalezen!\n";
    //if (a.get(4216)) std::cout << "4216 nalezen!\n";
    //if (a.get(119)) std::cout << "119 nalezen!\n";
    //if (!a.get(3 * 4096 - 1)) std::cout << "3 * 4096 - 1 nenalezen!\n";
    //return 0;

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

    HNSW hnsw(12, 12, 200, 0.75);
    hnsw.create(FILE_NAME, "train");

    auto end = std::chrono::system_clock::now();
    double dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Insert time " << dur / 1000 << " [s] \n";
    hnsw.printInfo();
    hnsw.query(FILE_NAME, "test", "neighbors", 75);

    return 0;
}

