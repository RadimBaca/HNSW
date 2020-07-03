#pragma once
#ifndef NO_MANUAL_VECTORIZATION
#ifdef __SSE__
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#endif
#endif
#endif

#include <stdio.h>
#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <random>
#include <cmath>
#include <chrono>
#include <assert.h>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <memory>
#include <fstream>


#if defined(USE_AVX) || defined(USE_SSE)
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>
#else
#include <x86intrin.h>
#endif

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif
#endif

#include "node.h"
#include "layer.h"
#include "settings.h"

#ifdef MAIN_RUN_CREATE_AND_QUERY
#include "hdfReader.h"
#endif

/*
Naive implementation of linear hash array.
Fixed size, without overflow check.
TODO - Should be replaced by a bloom filter or cuckoo filter.
*/
class linearHash
{
	uint32_t actual_size;
	uint32_t mask;
	uint32_t* hasharray;

public:

	uint32_t item_count;

	linearHash(uint32_t asize = 16384)
	{
		item_count = 0;
		actual_size = asize;
		mask = asize - 1;
		hasharray = new uint32_t[actual_size];
	}

	~linearHash()
	{
		delete[] hasharray;
	}

	void clear()
	{
		item_count = 0;
		memset(hasharray, -1, sizeof(uint32_t) * actual_size);
	}

	void reduce(int shift)
	{
		actual_size = actual_size >> shift;
		mask = actual_size - 1;
	}

	void insert(uint32_t index)
	{
		//if (item_count > (actual_size >> 1))
		//{
		//	std::cout << "Hash array should be resized!" << "\n";
		//}

		uint32_t hash = index & mask;
		while (hasharray[hash] != -1)
		{
			hash++;
			if (hash >= actual_size) {
				hash = 0;
			}
		}
		item_count++;
		hasharray[hash] = index;
	}

	bool get(uint32_t index)
	{
		uint32_t hash = index & mask;
		while (hasharray[hash] != -1)
		{
			if (hasharray[hash] == index)
			{
				return true;
			}
			hash++;
			if (hash >= actual_size) {
				hash = 0;
			}
		}
		return false;
	}
};


#ifdef COLLECT_STAT
class HNSW_Stat
{
public:
	int precise_distance_computations;
	int distance_computations;
	int distance_computations_false;
	int explored_nodes;
	int overflows;
	int neighbor_count;
	int node_count;

	int histogram[200];

	HNSW_Stat()
	{
		clear();
	}

	void clear()
	{
		precise_distance_computations = 0;
		distance_computations = 0;
		distance_computations_false = 0;
		explored_nodes = 0;
		overflows = 0;
		neighbor_count = 0;
		node_count = 0;
        for (int i = 0; i < 200; i++)
        {
            histogram[i] = 0;
        }
	}

	void print()
	{
		std::cout << "No. precise distance computations: " << precise_distance_computations << "\n";
		std::cout << "No. distance computations: " << distance_computations << "\n";
		std::cout << "No. explored nodes: " << explored_nodes << "\n";
	}

	void print_tree_info()
    {
#ifdef COUNT_INWARD_DEGREE
	    for (int i = 0; i < 200; i++)
        {
	        std::cout << i << ": " << histogram[i] << "\n";
        }
#endif
        if (overflows > 0)
        {
            std::cout << "No. overflows: " << overflows << "\n";
            std::cout << "No. neighbors (avg. neighbor): " << neighbor_count << " (" << (float)neighbor_count / node_count << ")\n";
        }
    }
};
#endif


struct CompareByFirst {
	constexpr bool operator()(std::pair<uint32_t, Node*> i1, std::pair<uint32_t, Node*> i2) const noexcept
	{
		return i1.first < i2.first;
	}
};


struct CompareByFirstInTuple {
	constexpr bool operator()(std::tuple<uint32_t, uint32_t, Node*, pointer_t> i1, std::tuple<uint32_t, uint32_t, Node*, pointer_t> i2) const noexcept
	{
		return std::get<0>(i1) < std::get<0>(i2);
	}
};

struct CompareByDistanceInTuple {
	constexpr bool operator()(std::tuple<Node*, int32_t, pointer_t> i1, std::tuple<Node*, int32_t, pointer_t> i2) const noexcept
	{
		return std::get<1>(i1) < std::get<1>(i2);
	}
};


struct CompareByDistanceInTupleHeap {
	constexpr bool operator()(std::tuple<Node*, int32_t> i1, std::tuple<Node*, int32_t> i2) const noexcept
	{
		return std::get<1>(i1) > std::get<1>(i2);
	}
};

class HNSW
{
public:

	int M_;
	int Mmax_;
	int Mmax0_;
	int efConstruction_;
	float ml_;
	int max_node_count_;
	int min_M_;

	long long index_memory_;
	std::vector<std::unique_ptr<Layer>> layers_;
	float* vector_data_;

	bool data_cleaned_;

	int actual_node_count_;
    uint32_t vector_size_;

	std::vector<Neighbors> W_;
	std::unordered_map<pointer_t, uint32_t> distances_;
#ifdef VISIT_HASH
	linearHash visited_; // TODO replace by some different hashing technique (or cuckoo filter)
#endif
#ifdef COLLECT_STAT
	HNSW_Stat stat_;
#endif
#ifdef COMPUTE_APPROXIMATE_VECTOR
	float* min_vector;
	float* max_vector;
	float min_value;
	float max_value;
	int shift;
	uint8_t* apr_q;
	int one_vector_mask_size;
	int* q_delta;
	uint8_t* apr_vector_data;

	int row1_size;
	int row2_size;

	std::vector<std::tuple<Node*, int32_t, pointer_t>> apr_W;

	// toto neni potreba (muze byt soucasti pocitani v distance)
	uint8_t query_positive_avg[2]; // average distance of regions (LPQ, SPQ)
	uint8_t query_negative_avg[2]; // average distance of regions (LNQ, SNQ)
#endif

	uint32_t vector_parts;      // vector_size / number of 16-bit values in vector register

	HNSW(int M, int Mmax, int efConstruction)
		: M_(M),
          Mmax_(Mmax),
          Mmax0_(Mmax * 2),
          efConstruction_(efConstruction),
          actual_node_count_(-1),
          max_node_count_(0),
          data_cleaned_(true),
          min_M_(M / 2)
	{
        ml_ = 1 / log(0.8 * M);
	}

	~HNSW()
	{
		if (!data_cleaned_)
		{
			clean();
		}
	}

	void init(uint32_t vector_size, uint32_t max_node_count);
	void clean();


#ifdef MAIN_RUN_CREATE_AND_QUERY
	void create(const char* filename, const char* datasetname);
	void query(const char* filename, const char* querydatasetname, const char* resultdatasetname, int ef);
#endif

	void insert(float* q);
	void aproximateKnn(float* q, int k, int ef);
#ifdef COMPUTE_APPROXIMATE_VECTOR
	void computeApproximateVector();
	int computeSummaries(Node* node, pointer_t node_order, uint32_t vector_size, int* overflows);
#endif
	void printInfo(bool all);
	void print(int max_count);

    void saveGraph(const char* filename);
    void loadGraph(const char* filename);
private:

    void knn(float* q, int ef);
	void searchLayerOne(float* q);
	void searchLayer(float* q, int ef);
	void selectNeighbors(std::vector<Neighbors>& W, std::vector<Neighbors>& R, int M, bool keepPruned);

	inline float* getNodeVector(pointer_t node_order) { return vector_data_ + node_order * vector_size_; }

#ifdef COMPUTE_APPROXIMATE_VECTOR
	void aprSearchLayerOne(uint8_t* q);
	void aprSearchLayer(uint8_t* q, int ef);
	void aprSearchLayerSummary(uint8_t* q, int ef);
	void aprSearchLayerDoubleSummary(uint8_t* q, int ef);

	inline uint8_t* aprGetNodeVector(pointer_t node_order) { return apr_vector_data + node_order * vector_size_; }

#endif




#if defined(USE_SSE)

    float distance(float *pVect1v, float *pVect2v) {
        float *pVect1 = pVect1v;
        float *pVect2 = pVect2v;
        size_t qty = vector_size_;
        float PORTABLE_ALIGN32 TmpRes[8];
        // size_t qty4 = qty >> 2;
        size_t qty16 = qty >> 4;

        const float *pEnd1 = pVect1 + (qty16 << 4);
        // const float* pEnd2 = pVect1 + (qty4 << 2);
        // const float* pEnd3 = pVect1 + qty;

        __m128 diff, v1, v2;
        __m128 sum = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }
        _mm_store_ps(TmpRes, sum);
        float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

        return (res);
    }
#else


    inline float distance(float* q, float* node)
	{
#ifdef COLLECT_STAT
		stat_.precise_distance_computations++;
#endif

		float result = 0;
		for (unsigned int i = 0; i < vector_size_; i++)
		{
			float t = q[i] - node[i];
			result += t * t;
		}
		return result;
	}
#endif

//	inline float distance_treshold_counting(float* q, float* node, uint32_t& no_over_treshold)
//	{
//#ifdef COLLECT_STAT
//		stat_.precise_distance_computations++;
//#endif
//
//		float result = 0;
//		no_over_treshold = 0;
//		for (unsigned int i = 0; i < vector_size_; i++)
//		{
//			float t = q[i] - node[i];
//			result += t * t;
//			no_over_treshold += t > DISTANCE_TRESHOLD || -t > DISTANCE_TRESHOLD;
//		}
//		return result;
//	}

#ifdef COMPUTE_APPROXIMATE_VECTOR
#ifdef USE_AVX
    inline uint32_t apr_distance(uint8_t* __restrict__ q, uint8_t* __restrict__ node)
    {
        __m256i zero = _mm256_set1_epi64x(0); // zero = 0
        __m256i result = zero; // result = 0
        for (uint32_t i = 0; i < vector_parts; i++) {
            __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q + 16 * i));     // a = q[i]
            __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(node + 16 * i));  // b = node[i]
            __m256i ca = _mm256_cvtepu8_epi16(a);       // zero-extend 8-bit to 16-bit
            __m256i cb = _mm256_cvtepu8_epi16(b);       // zero-extend 8-bit to 16-bit
            auto sub = _mm256_sub_epi16(ca, cb);        // sub = a - b
            auto pow = _mm256_mullo_epi16(sub, sub);    // pow = sub * sub
            result = _mm256_add_epi16(result, pow);     // result += pow
        }

        auto x = _mm256_hadd_epi16(result, zero);   // first reduction pass, data is at [0:4] and [8:12]
        auto y = _mm256_hadd_epi16(x, zero);        // second reduction pass, data is at [0:2] and [8:10]
        uint32_t output = 0;
        uint16_t data[4];
        memcpy(data, reinterpret_cast<uint16_t*>(&y), sizeof(uint16_t) * 2);
        memcpy(data + 2, reinterpret_cast<uint16_t*>(&y) + 8, sizeof(uint16_t) * 2);

        for (int i = 0; i < 4; i++) {
            output += data[i];
        }

        for (unsigned int i = vector_parts * 16; i < vector_size; i++)
        {
            int32_t t = q[i] - node[i];
            output += t * t;
        }
        return output;
    }
#else
	inline uint32_t aprDistance(uint8_t* q, uint8_t* node)
	{
#ifdef COLLECT_STAT
		stat_.distance_computations++;
#endif
		uint32_t result = 0;
		for (unsigned int i = 0; i < vector_size_; i++)
		{
			int32_t t = q[i] - node[i];
			result += t * t;
		}
		return result;
	}
#endif

	inline uint32_t aprDistance(uint8_t* q, uint8_t* node, int* q_delta)
	{
#ifdef COLLECT_STAT
		stat_.distance_computations++;
#endif
		uint32_t result = 0;
		for (unsigned int i = 0; i < vector_size_; i++)
		{
			int32_t t = q[i] - node[i];
			result += t * t;
			q_delta[i] = t;
		}
		return result;
	}


//    inline uint32_t aprDistance(uint8_t* q, uint8_t* node, int* q_delta)
//    {
//#ifdef COLLECT_STAT
//        stat_.distance_computations++;
//#endif
//        uint8_t* a = q;
//        uint8_t* b = node;
//        int* d = q_delta;
//        uint32_t result = 0;
//        auto s = vector_size_ >> 2;
//        for (unsigned int i = 0; i < s; i++)
//        {
//            int32_t t = (*a) - (*b);
//            result += t * t;
//            (*d) = t;
//            a++;
//            b++;
//            d++;
//            t = (*a) - (*b);
//            result += t * t;
//            (*d) = t;
//            a++;
//            b++;
//            d++;
//            t = (*a) - (*b);
//            result += t * t;
//            (*d) = t;
//            a++;
//            b++;
//            d++;
//            t = (*a) - (*b);
//            result += t * t;
//            (*d) = t;
//            a++;
//            b++;
//            d++;
//        }
//        return result;
//    }

	inline uint32_t aprDistanceSummary(std::vector<int8_t>&node_summary, int& node_summary_position, int* q_delta, uint32_t pivot_distance)
	{
#ifdef COLLECT_STAT
		stat_.distance_computations++;
#endif
		uint32_t result = pivot_distance;
		int c = node_summary[node_summary_position] * 2 + 1 + node_summary_position;
		for (int i = node_summary_position + 1; i < c; i+=2)
		{
			int8_t position = node_summary[i];
			int z = node_summary[i + 1];
			result -= 2 * q_delta[position] * z - z * z;
		}
		node_summary_position = c;
		
		return result;
	}

	inline uint32_t aprDistanceSummaryFixedSize(std::vector<int8_t>& node_summary, int node_summary_position, int* q_delta, uint32_t pivot_distance, int row_size)
	{
#ifdef COLLECT_STAT
		stat_.distance_computations++;
#endif
		uint32_t result = pivot_distance;
		for (int i = node_summary_position; i < node_summary_position + row_size; i += 2)
		{
			int8_t position = node_summary[i];
			int z = node_summary[i + 1];
			result -= 2 * q_delta[position] * z - z * z;
		}

		return result;
	}

	void aprChangeLayer()
	{
		for (int i = 0; i < apr_W.size(); i++)
		{
			auto n = apr_W[i];
			auto node = std::get<0>(n);
			std::get<0>(apr_W[i]) = node->lower_layer;
			std::get<0>(apr_W[i])->copyInsertValues(*node);
			//#ifdef VISIT_HASH
			//			visited_.insert(W_[i]->uniqueId);
			//#endif
		}
	}
#endif

	void changeLayer()
	{
		for (int i = 0; i < W_.size(); i++)
		{
			auto n = W_[i];
			auto node = n.node;
            W_[i].node = W_[i].node->lower_layer;
			W_[i].node->copyInsertValues(*node);
//#ifdef VISIT_HASH
//			visited_.insert(W_[i]->uniqueId);
//#endif
		}
	}


    void setVectorSize(uint32_t vsize)
    {
        vector_size_ = vsize;
        vector_parts = vsize / 16;
    }

//    void clear_explored_count()
//    {
//	    for (auto& n: layers_[0]->nodes)
//        {
//	        n->explored_count = 0;
//        }
//    }
};
