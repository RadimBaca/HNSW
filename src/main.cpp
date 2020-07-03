#include <chrono>

#include "hnsw.h"
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

    {
        HNSW hnsw(5, 5, 20);
        hnsw.init(vsize, vcount);

        float *v = new float[vsize];
        for (int i = 0; i < vcount; i++) {
            for (int k = 0; k < vsize; k++) {
                v[k] = ((float) (rand() % 10000)) / 100;
            }
            hnsw.insert(v);
        }
        hnsw.saveKNNG("test_100.bin");


#ifdef COMPUTE_APPROXIMATE_VECTOR
        hnsw.min_value = 0;
        hnsw.max_value = 100;
        hnsw.computeApproximateVector();
#endif
        for (int k = 0; k < vsize; k++) {
            v[k] = 50;
        }
        hnsw.knn(v, 2, 10);
        for (auto r : hnsw.apr_W)
        {
            int x = std::get<2>(r);
            int y = std::get<1>(r);
            std::cout << x << "(" << y << ") ";
        }
        std::cout << "\n";
    }

    {
        float *v = new float[vsize];
        HNSW hnsw(10, 10, 20);
        hnsw.loadKNNG("test_100.bin");


#ifdef COMPUTE_APPROXIMATE_VECTOR
        hnsw.min_value = 0;
        hnsw.max_value = 100;
        hnsw.computeApproximateVector();
#endif
        hnsw.printInfo(false);
#ifdef COLLECT_STAT
        hnsw.stat_.print_tree_info();
#endif

        for (int k = 0; k < vsize; k++) {
            v[k] = 50;
        }
        hnsw.knn(v, 2, 10);
        for (auto r : hnsw.apr_W)
        {
            int x = std::get<2>(r);
            int y = std::get<1>(r);
            std::cout << x << "(" << y << ") ";
        }
        std::cout << "\n";

        auto start = std::chrono::system_clock::now();
        for (int i = 0; i < 100; i++) {
            //std::cout << "-------------------\n";
            for (int k = 0; k < vsize; k++) {
                v[k] = ((float) (rand() % 10000)) / 100;
            }
            hnsw.knn(v, 2, 3);

            //for (int k = 0; k < 3; k++)
            //{
            //    hnsw.W_[k]->print(vsize);
            //}
        }
        auto end = std::chrono::system_clock::now();
        std::cout << (double) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000
                  << " [ms]";
#ifdef COLLECT_STAT
        hnsw.stat_.print();
#endif
    }
#endif

#ifdef MAIN_RUN_CREATE_AND_QUERY

    auto start = std::chrono::system_clock::now();

    HNSW hnsw(16, 20, 200, 0.5);
    hnsw.create(FILE_NAME, "train");

    auto end = std::chrono::system_clock::now();
    double dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Insert time " << dur / 1000 << " [s] \n";
    hnsw.printInfo(false);
    //hnsw.visited_.reduce(1);
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
    std::ifstream input("../sift1M/sift1M.bin", std::ios::binary);
    if (!input.is_open()) std::runtime_error("Input data file not opened!");
    input.read((char*)mass, node_count * vecdim * sizeof(float));
    input.close();

    float* massQ = new float[qsize * vecdim];
    //ifstream inputQ("../siftQ100k.bin", ios::binary);
    std::ifstream inputQ("../sift1M/siftQ1M.bin", std::ios::binary);
    if (!input.is_open()) std::runtime_error("Input query file not opened!");
    //ifstream inputQ("../../1M_d=4q.bin", ios::binary);
    inputQ.read((char*)massQ, qsize * vecdim * sizeof(float));
    inputQ.close();

    unsigned int* massQA = new unsigned int[qsize * answer_size];
    //ifstream inputQA("../knnQA100k.bin", ios::binary);
    std::ifstream inputQA("../sift1M/knnQA1M.bin", std::ios::binary);
    if (!input.is_open()) std::runtime_error("Input result file not opened!");
    inputQA.read((char*)massQA, qsize * answer_size * sizeof(int));
    inputQA.close();

    HNSW hnsw(16 , 16, 200);
#ifdef LOAD_GRAPH
    hnsw.loadGraph(load_file);
#else
    hnsw.init(vecdim, node_count );


    /////////////////////////////////////////////////////// INSERT PART
    std::cout << "Start inserting\n";
    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < node_count; i++)
    {
        hnsw.insert(&mass[i * vecdim]);
    }
    auto end = std::chrono::system_clock::now();
    double dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Insert time " << dur / 1000 << " [s] \n";
    hnsw.saveGraph(load_file);
#endif

#ifdef COMPUTE_APPROXIMATE_VECTOR
#ifdef COLLECT_STAT
    hnsw.stat_.clear();
#endif
    hnsw.computeApproximateVector();
#ifdef COLLECT_STAT
    hnsw.stat_.print_tree_info();
#endif
#endif


    /////////////////////////////////////////////////////// QUERY PART
    hnsw.printInfo(false);
    std::cout << "Start querying\n";
    std::vector<std::pair<float, float>> precision_time;
    for (int ef = 20; ef <= 200; ef += 10)
    {
        if (ef > 100) ef += 10;

        float positive = 0;
        for (int i = 0; i < qsize; i++)
        {
            hnsw.aproximateKnn(&massQ[i * vecdim], k, ef);

            std::vector<int> result;
            int c1 = 0;
#ifdef COMPUTE_APPROXIMATE_VECTOR
            for (auto item : hnsw.apr_W)
            {
                result.push_back(std::get<2>(item));
#else
            for (auto item : hnsw.W_)
            {
                result.push_back(item.node_order);
#endif

                if (c1++ >= k) break;
            }

            int c2 = 0;
            while (c2 < k)
            {
                if (std::find(result.begin(), result.end(), massQA[i * answer_size + c2]) != result.end())
                {
                    positive++;
                }
                c2++;
            }

//            if (i < 10)
//            {
//                std::cout << "\nFinded  : ";
//                for (int m = 0; m < 10; m++)
//                {
//                    std::cout << hnsw.W_[m].node_order << "(" << hnsw.W_[m].distance << ")  ";
//                }
//                std::cout << "\nExpected: ";
//                for (int m = 0; m < 10; m++)
//                {
//                    std::cout << massQA[i * answer_size + m] << "  ";
//                }
//            }
        }
        std::cout << "Precision: " << positive / (qsize * k) << ", ";

        int sum = 0;
        int min_time;
        std::cout << "ef: " << ef << ", ";
        for (int i = 0; i < 3; i++)
        {
#ifdef COLLECT_STAT
            hnsw.stat_.clear();
#endif
            auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < qsize; i++)
            {
                hnsw.aproximateKnn(&massQ[i * vecdim], k, ef);
            }
            auto end = std::chrono::steady_clock::now();
            int time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            sum += time;
            min_time = i == 0 ? time : std::min(min_time, time);
#ifdef COLLECT_STAT
            std::cout << (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / qsize << " [us]; ";
#endif
        }
        std::cout << "avg: " << (float)sum / (qsize * 3) << " [us]; " << "min: " << min_time / qsize<< " [us]; \n";
        precision_time.emplace_back((float)positive / (qsize * k), (float)min_time / qsize);
#ifdef COLLECT_STAT
        hnsw.stat_.print();
#endif    
    }
    std::cout << "\nPrecision Time [us]\n";
    for(auto item: precision_time)
    {
        std::cout << item.first << " " << item.second << "\n";
    }


    delete[] mass;
    delete[] massQ;
    delete[] massQA;
}