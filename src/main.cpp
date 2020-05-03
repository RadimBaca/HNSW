#include <chrono>

#include "HNSW.h"

#define FILE_NAME       "sift-128-euclidean.hdf5"


int main(void)
{
 

#ifdef UNIT_TESTING
    constexpr int vcount = 1000;
    constexpr int vsize = 3;
    HNSW hnsw(4, 4, 20, 0.3);
    hnsw.vector_size = vsize;
    hnsw.vector_count = vcount;

#ifdef COMPUTE_APPROXIMATE_VECTOR
    hnsw.min_vector = new float[vsize];
    hnsw.max_vector = new float[vsize];
    hnsw.apr_q = new uint8_t[vsize];
#endif

    float* v = new float[3];
    for (int i = 0; i < vcount; i++)
    {
        for (int k = 0; k < vsize; k++)
        {
            v[k] = ((float)(rand() % 10000)) / 100;
        }
        hnsw.insert(v);
    }

#ifdef COMPUTE_APPROXIMATE_VECTOR
    hnsw.min_value = 0;
    hnsw.max_value = 100;
    hnsw.computeApproximateVector();
#endif
    hnsw.printInfo(false);

    for (int i = 0; i < 1000; i++)
    {
        //std::cout << "-------------------\n";
        for (int k = 0; k < vsize; k++)
        {
            v[k] = ((float)(rand() % 10000)) / 100;
        }
        hnsw.knn(v, 2, 3);

        //for (int k = 0; k < 3; k++)
        //{
        //    hnsw.W[k]->print(vsize);
        //}
    }
#else

    auto start = std::chrono::system_clock::now();

    HNSW hnsw(16, 16, 200, 0.5);
    hnsw.create(FILE_NAME, "train");

    auto end = std::chrono::system_clock::now();
    double dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Insert time " << dur / 1000 << " [s] \n";
    hnsw.printInfo(false);
    for (int i = 40; i <= 140; i += 30)
    {
        hnsw.query(FILE_NAME, "test", "neighbors", i);
    }
#endif
    return 0;
}

