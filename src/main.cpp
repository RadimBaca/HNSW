#include <chrono>

#include "HNSW.h"
#include <fstream>

#define FILE_NAME       "sift-128-euclidean.hdf5"


int main(void)
{
#ifdef MAIN_CONVERT_HDF_TO_BIN
    {
        hsize_t dimensions[2];
        hdfReader::getDimensions(FILE_NAME, "train", &dimensions);
        float* data = new float[dimensions[0] * dimensions[1]];
        hdfReader::readData(FILE_NAME, "train", data);
        //for (int i = 0; i < 10; i++) std::cout << data[i] << "\n";
        std::ofstream input("sift1M.bin", std::ios::binary);
        //ifstream input("../../1M_d=4.bin", ios::binary);
        input.write((char*)data, dimensions[0] * dimensions[1] * sizeof(float));
        input.close();
        delete[] data;
    }

    {
        hsize_t dimensions_query[2];
        hdfReader::getDimensions(FILE_NAME, "test", &dimensions_query);
        float* data_query = new float[dimensions_query[0] * dimensions_query[1]];
        std::cout << "Query set: " << dimensions_query[0] << " x " << dimensions_query[1] << "\n";
        hdfReader::readData(FILE_NAME, "test", data_query);
        std::ofstream qinput("siftQ1M.bin", std::ios::binary);
        //ifstream input("../../1M_d=4.bin", ios::binary);
        qinput.write((char*)data_query, dimensions_query[0] * dimensions_query[1] * sizeof(float));
        qinput.close();
        delete[] data_query;
    }

    {
        hsize_t dimensions_result[2];
        hdfReader::getDimensions(FILE_NAME, "neighbors", &dimensions_result);
        int* data_result = new int[dimensions_result[0] * dimensions_result[1]];
        std::cout << "Result set: " << dimensions_result[0] << " x " << dimensions_result[1] << "\n";
        hdfReader::readData(FILE_NAME, "neighbors", data_result);
        //for (int i = 0; i < 10; i++) std::cout << data_result[i] << "\n";
        std::ofstream kinput("knnQA1M.bin", std::ios::binary);
        //ifstream input("../../1M_d=4.bin", ios::binary);
        kinput.write((char*)data_result, dimensions_result[0] * dimensions_result[1] * sizeof(int));
        kinput.close();
        delete[] data_result;
    }
#endif

#ifdef MAIN_UNIT_TESTING
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
#endif

#ifdef MAIN_RUN_CREATE_AND_QUERY

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

