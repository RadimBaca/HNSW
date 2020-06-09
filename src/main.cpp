#include <chrono>

#include "HNSW.h"
#include <fstream>

#define FILE_NAME       "sift-128-euclidean.hdf5"

void sift_test();

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
    constexpr int vcount = 10000;
    constexpr int vsize = 128;
    HNSW hnsw(10, 10, 20, 0.3);
    hnsw.init(vsize, vcount);

    float* v = new float[vsize];
    for (int i = 0; i < vcount; i++)
    {
        for (int k = 0; k < vsize; k++)
        {
            v[k] = ((float)(rand() % 10000)) / 100;
        }
        if (i == 789)
        {
            int i = 0;
        }
        hnsw.insert(v);
    }

#ifdef COMPUTE_APPROXIMATE_VECTOR
    hnsw.min_value = 0;
    hnsw.max_value = 100;
    hnsw.computeApproximateVector();
#endif
    hnsw.printInfo(false);

    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < 20000; i++)
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
    auto end = std::chrono::system_clock::now(); 
    std::cout << (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000 << " [ms]";
    hnsw.stat.print();
#endif

#ifdef MAIN_RUN_CREATE_AND_QUERY

    auto start = std::chrono::system_clock::now();

    HNSW hnsw(16, 20, 200, 0.5);
    hnsw.create(FILE_NAME, "train");

    auto end = std::chrono::system_clock::now();
    double dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Insert time " << dur / 1000 << " [s] \n";
    hnsw.printInfo(false);
    //hnsw.visited.reduce(1);
    for (int i = 40; i <= 140; i += 30)
    {
        hnsw.query(FILE_NAME, "test", "neighbors", i);
    }
#endif

#ifdef MAIN_RUN_CREATE_AND_QUERY_WITHOUT_HDF5
    sift_test();
#endif
    return 0;
}



void sift_test() {
    size_t node_count = 1000000;
    size_t qsize = 10000;
    //size_t qsize = 1000;
    //size_t vecdim = 4;
    size_t vecdim = 128;
    size_t answer_size = 100;
    uint32_t k = 10;

    float* mass = new float[node_count * vecdim];
    std::ifstream input("sift1M/sift1M.bin", std::ios::binary);
    if (!input.is_open()) std::runtime_error("Input data file not opened!");
    input.read((char*)mass, node_count * vecdim * sizeof(float));
    input.close();

    float* massQ = new float[qsize * vecdim];
    //ifstream inputQ("../siftQ100k.bin", ios::binary);
    std::ifstream inputQ("sift1M/siftQ1M.bin", std::ios::binary);
    if (!input.is_open()) std::runtime_error("Input query file not opened!");
    //ifstream inputQ("../../1M_d=4q.bin", ios::binary);
    inputQ.read((char*)massQ, qsize * vecdim * sizeof(float));
    inputQ.close();

    unsigned int* massQA = new unsigned int[qsize * answer_size];
    //ifstream inputQA("../knnQA100k.bin", ios::binary);
    std::ifstream inputQA("sift1M/knnQA1M.bin", std::ios::binary);
    if (!input.is_open()) std::runtime_error("Input result file not opened!");
    inputQA.read((char*)massQA, qsize * answer_size * sizeof(int));
    inputQA.close();


    HNSW hnsw(16, 20, 200, 0.5);
    hnsw.init(vecdim, node_count);


    /////////////////////////////////////////////////////// INSERT PART
    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < node_count; i++)
    {
        hnsw.insert(&mass[i * vecdim]);
    }
    auto end = std::chrono::system_clock::now();
    double dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Insert time " << dur / 1000 << " [s] \n";
#ifdef COMPUTE_APPROXIMATE_VECTOR
    hnsw.stat.clear();
    hnsw.computeApproximateVector();
#ifdef COLLECT_STAT
    hnsw.stat.print();
#endif
#endif

    /////////////////////////////////////////////////////// QUERY PART
    hnsw.printInfo(false);
    //hnsw.visited.reduce(1);
    for (int ef = 40; ef <= 140; ef += 30)
    {
        double positive = 0;
        for (int i = 0; i < qsize; i++)
        {
            hnsw.knn(&massQ[i * vecdim], k, ef);

            int c1 = 0;
            int c2 = 0;
            while (c2 < k)
            {
#ifdef COMPUTE_APPROXIMATE_VECTOR
                if (std::get<2>(hnsw.apr_W[c1]) == massQA[i * answer_size + c2])
#else
                if (hnsw.W[c1].node_order == massQA[i * answer_size + c2])
#endif
                {
                    positive++;
                    c1++;
                }
                c2++;
            }

            //if (i < 10)
            //{
            //    std::cout << "\nFinded  : ";
            //    for (int m = 0; m < 10; m++)
            //    {
            //        std::cout << W[m].node_order << "(" << W[m].distance << ")  ";
            //    }
            //    std::cout << "\nExpected: ";
            //    for (int m = 0; m < 10; m++)
            //    {
            //        std::cout << data_result[i * dimensions_result[1] + m] << "  ";
            //    }
            //}
        }
        std::cout << "Precision: " << positive / (node_count * k) << "\n";

        double sum = 0;
        double min_time;
        for (int i = 0; i < 3; i++)
        {
#ifdef COLLECT_STAT
            hnsw.stat.clear();
#endif
            auto start = std::chrono::system_clock::now();
            for (int i = 0; i < qsize; i++)
            {
                hnsw.knn(&massQ[i * vecdim], k, ef);
            }
            auto end = std::chrono::system_clock::now();
            auto time = (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            sum += time;
            min_time = i == 0 ? time : std::min(min_time, time);
            std::cout << (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / node_count << " [ms]; ";
        }
        std::cout << "avg: " << sum / (3 * node_count) << " [ms]; " << "min: " << min_time / node_count << " [ms]; \n";
#ifdef COLLECT_STAT
        hnsw.stat.print();
#endif    
    }

    delete[] mass;
    delete[] massQ;
    delete[] massQA;
}