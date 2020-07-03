#include "hnsw.h"

void HNSW::init(uint32_t vector_size, uint32_t p_max_node_count)
{
    data_cleaned_ = false;
    max_node_count_ = p_max_node_count;
    setVectorSize(vector_size);
    vector_data_ = new float[p_max_node_count * vector_size];
#ifdef COMPUTE_APPROXIMATE_VECTOR
    apr_vector_data = new uint8_t[p_max_node_count * vector_size];
    min_vector = new float[vector_size];
    max_vector = new float[vector_size];
    apr_q = new uint8_t[vector_size];
    one_vector_mask_size = (vector_size >> 5) == 0 ? 1 : vector_size >> 5;
    q_delta = new int[vector_size];
    row1_size = (int)(VECTOR_FRAGMENT1 * vector_size) * 2;
    row2_size = (int)(VECTOR_FRAGMENT2 * vector_size) * 2;

#endif
}

void HNSW::clean()
{
    delete[] vector_data_;
#ifdef COMPUTE_APPROXIMATE_VECTOR
    delete[] apr_vector_data;
    delete[] min_vector;
    delete[] max_vector;
    delete[] apr_q;
    delete[] q_delta;
#endif
}


#ifdef MAIN_RUN_CREATE_AND_QUERY
void HNSW::create(const char* filename, const char* datasetname)
{

    hsize_t dimensions[2];

    hdfReader::getDimensions(filename, datasetname, &dimensions);
    std::clog << "dimensions " <<
        (unsigned long)(dimensions[0]) << " x " <<
        (unsigned long)(dimensions[1]) << std::endl;
 
    init(dimensions[1], dimensions[0]);
    float* data = new float[max_node_count_ * vector_size_];
    hdfReader::readData(filename, datasetname, data);

    for (int i = 0; i < max_node_count_; i++)
    {
        insert(&data[i * vector_size_]);
    }
    delete[] data;
#ifdef COMPUTE_APPROXIMATE_VECTOR


    for (int i = 0; i < vector_size_; i++)
    {
        max_vector[i] = (max_vector[i] - min_vector[i]) * 1.1;
    }
    max_value = max_value * 1.05;

#ifdef APR_DEBUG
    std::cout << "Min\tMax\n";
    for (int i = 0; i < vector_size_; i++)
    {
        std::cout << min_vector[i] << "\t" << max_vector[i] << "\n";
    }
#endif
    stat_.clear();
    computeApproximateVector();
#endif
#ifdef COLLECT_STAT
    stat_.print();
#endif
}


void HNSW::query(const char* filename, const char* querydatasetname, const char* resultdatasetname, int ef)
{
    hsize_t dimensions_query[2];
    hsize_t dimensions_result[2];

    hdfReader::getDimensions(filename, querydatasetname, &dimensions_query);
    float* data_query = new float[dimensions_query[0] * dimensions_query[1]];
    hdfReader::readData(filename, querydatasetname, data_query);

    hdfReader::getDimensions(filename, resultdatasetname, &dimensions_result);
    int* data_result = new int[dimensions_result[0] * dimensions_result[1]];
    hdfReader::readData(filename, resultdatasetname, data_result);

    uint32_t k = 10;
    double positive = 0;
    for (int i = 0; i < dimensions_query[0]; i++)
    {
        knn(&data_query[i * dimensions_query[1]], k, ef);

        int c1 = 0;
        int c2 = 0;
        while (c2 < k)
        {
#ifdef COMPUTE_APPROXIMATE_VECTOR
            if (std::get<2>(apr_W[c1]) == data_result[i * dimensions_result[1] + c2])
#else
            if (W_[c1].node_order == data_result[i * dimensions_result[1] + c2])
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
        //        std::cout << W_[m].node_order << "(" << W_[m].distance << ")  ";
        //    }
        //    std::cout << "\nExpected: ";
        //    for (int m = 0; m < 10; m++)
        //    {
        //        std::cout << data_result[i * dimensions_result[1] + m] << "  ";
        //    }
        //}
    }
    std::cout << "Precision: " << positive / (dimensions_query[0] * k) << ", ";

    double sum = 0;
    double min_time;
    for (int i = 0; i < 3; i++)
    {
#ifdef COLLECT_STAT
        stat_.clear();
#endif
        auto start = std::chrono::system_clock::now();
        for (int i = 0; i < dimensions_query[0]; i++)
        {
            knn(&data_query[i * dimensions_query[1]], k, ef);
        }
        auto end = std::chrono::system_clock::now();
        auto time = (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        sum += time;
        min_time = i == 0 ? time : std::min(min_time, time);
#ifdef COLLECT_STAT
        std::cout << (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / dimensions_query[0] << " [ms]; ";
#endif
    }
    std::cout << "avg: " << sum / (3 * dimensions_query[0]) << " [ms]; " << "min: " << min_time / dimensions_query[0] << " [ms]; \n";
#ifdef COLLECT_STAT
    stat_.print();
#endif    

}
#endif

//////////////////////////////////////////////////////////////
///////////////////////   DML op    //////////////////////////
//////////////////////////////////////////////////////////////

void HNSW::insert(float* q)
{
    int32_t l = -log(((float)(rand() % 10000 + 1)) / 10000) * ml_;
    int32_t L = layers_.size() - 1;
    Node* down_node = nullptr;
    Node* prev = nullptr;
    Node* ep = nullptr;
    pointer_t ep_node_order;
    std::vector<Neighbors> R;
    std::vector<Neighbors> Woverflow;
    std::vector<Neighbors> Roverflow;

    actual_node_count_++;
    if (actual_node_count_ >= max_node_count_)
    {
        throw std::runtime_error("HNSW: exceeded maximal number of nodes\n");
    }
    memcpy(getNodeVector(actual_node_count_), q, sizeof(float) * vector_size_);

#ifdef VISIT_HASH
    visited_.clear();
#endif

//      this code checks whether the new multi-level node is not close to other multi-level nodes
//    if (actual_node_count_ > 100 && l > 0)
//    {
//        knn(q, Mmax0_);
//        actual_node_count_--;
//        bool is_multi_level_close = false;
//        for(auto& n : W_){
//            if (n.node->pivot)
//            {
//                is_multi_level_close = true;
//                break;
//            }
//        }
//        if (is_multi_level_close)
//        {
//            l = 0;
//        }
//    }

#ifdef COMPUTE_APPROXIMATE_VECTOR
    if (actual_node_count_ == 0)
    {
        memcpy(min_vector, q, sizeof(float) * vector_size_);
        memcpy(max_vector, q, sizeof(float) * vector_size_);
        min_value = q[0];
        max_value = q[0];
    }
    for (int i = 0; i < vector_size_; i++)
    {
        min_vector[i] = q[i] < min_vector[i] ? q[i] : min_vector[i];
        max_vector[i] = q[i] > max_vector[i] ? q[i] : max_vector[i];
        min_value = std::min(min_value, q[i]);
        max_value = std::max(max_value, q[i]);
    }
#endif

    if ((actual_node_count_ % 10000) == 0)
    {
        std::cout << actual_node_count_ << "\n";
    }

    W_.clear();
    if (L >= 0)
    {
        ep = layers_[L]->enter_point;
        ep_node_order = layers_[L]->ep_node_order;
        auto dist = distance(getNodeVector(ep_node_order), q);

#ifdef VISIT_HASH
        visited_.insert(ep_node_order);
#else
        ep->actual_node_count_ = actual_node_count_;
#endif
        W_.emplace_back(ep, dist, ep_node_order);
    }
    for (int32_t i = L; i > l; i--)
    {
        searchLayerOne(q);
        changeLayer();
    }
    for (int32_t i = std::min(L, l); i >= 0; i--)
    {
        Node* new_node = new Node(Mmax_, nullptr);
        layers_[i]->node_count++;
        layers_[i]->nodes.push_back(new_node);

        auto actualMmax = (i == 0 ? Mmax0_ : Mmax_);

        if (l > L && i == L)
        {
            down_node = new_node; // remember the top node, if we are going to add layers_
        }

        // the main logic
#ifdef VISIT_HASH
        visited_.clear();
#endif
        searchLayer(q, efConstruction_);
        selectNeighbors(W_, R, M_, false);

        for (int i = 0;i < R.size(); i++)
        {
            Neighbors &e = R[i];
            Node* enode = e.node;
#ifdef COUNT_INWARD_DEGREE
            enode->inward_count++;
#endif
            new_node->neighbors.emplace_back(enode, e.distance, e.node_order);

            bool insert_new = true;
            if (enode->neighbors.size() == actualMmax)
            {
                if (!enode->neighbors_sorted)
                {
                    std::sort(enode->neighbors.begin(), enode->neighbors.end(), neighborcmp_nearest());
                    enode->neighbors_sorted = true;
                }

                Roverflow.clear();
                Woverflow.clear();
                Woverflow = enode->neighbors;
                Woverflow.emplace_back(new_node, e.distance, actual_node_count_);
                selectNeighbors(Woverflow, Roverflow, actualMmax, false);
#ifdef COUNT_INWARD_DEGREE
                for (auto item : enode->neighbors)
                {
                    item.node->inward_count--;
                }
#endif
                enode->neighbors.clear();
                for (auto r : Roverflow)
                {
#ifdef COUNT_INWARD_DEGREE
                    r.node->inward_count++;
#endif
                    enode->neighbors.emplace_back(r);
                }

            }
            else
            {
#ifdef COUNT_INWARD_DEGREE
                new_node->inward_count++;
#endif
                enode->neighbors.emplace_back(new_node, e.distance, actual_node_count_);
            }
        }

        // connecting the nodes in different layers_
        if (prev != nullptr)
        {
            prev->lower_layer = new_node;
        }
        prev = new_node;

        // switch the actual layer (W_ nodes are replaced by their bottom counterparts)
        if (i > 0) changeLayer();
    }
    if (l > L)
    {
        // adding layers_
        for (int32_t i = L + 1; i <= l; i++)
        {
            Node* node = new Node(Mmax_, down_node);
            layers_.emplace_back(std::make_unique<Layer>(node, actual_node_count_));
            down_node = node;
        }
    }
}


void HNSW::knn(float* q, int ef)
{
    int32_t L = layers_.size() - 1;
    Node* down_node = nullptr;
    Node* prev = nullptr;
    Node* ep = nullptr;
    int ep_node_order;
    actual_node_count_++;

#ifdef VISIT_HASH
    visited_.clear();
#endif
    ep = layers_[L]->enter_point;
    ep_node_order = layers_[L]->ep_node_order;
    W_.clear();
    auto dist = distance(getNodeVector(ep_node_order), q);
    W_.emplace_back(ep, dist, ep_node_order);

#ifdef VISIT_HASH
    visited_.insert(ep_node_order);
#else
    ep->actual_node_count_ = actual_node_count_;
#endif

    for (int32_t i = L; i > 0; i--)
    {
        searchLayerOne(q);
        changeLayer();
    }
#ifdef VISIT_HASH
    visited_.clear();
#endif
    searchLayer(q, ef);
    sort(W_.begin(), W_.end(), neighborcmp_nearest());
}

void HNSW::aproximateKnn(float* q, int k, int ef)
{
    int32_t L = layers_.size() - 1;
    Node* down_node = nullptr;
    Node* prev = nullptr;
    Node* ep = nullptr;
    int ep_node_order;
    actual_node_count_++;

#ifdef VISIT_HASH
    visited_.clear();
#endif
    ep = layers_[L]->enter_point;
    ep_node_order = layers_[L]->ep_node_order;
#ifdef COMPUTE_APPROXIMATE_VECTOR
    apr_W.clear();
    Node::computeApproximateVector(apr_q, q, shift, vector_size_);
    auto dist = aprDistance(aprGetNodeVector(layers_[L]->ep_node_order), apr_q);
    apr_W.emplace_back(ep, dist, ep_node_order);
#else
    W_.clear();
    auto dist = distance(getNodeVector(ep_node_order), q);
    W_.emplace_back(ep, dist, ep_node_order);
#endif
#ifdef VISIT_HASH
    visited_.insert(ep_node_order);
#else
    ep->actual_node_count_ = actual_node_count_;
#endif

    for (int32_t i = L; i > 0; i--)
    {
#ifdef COMPUTE_APPROXIMATE_VECTOR
        aprSearchLayerOne(apr_q);
        aprChangeLayer();
#else
        search_layer_one(q);
        change_layer();
#endif
    }
#ifdef VISIT_HASH
    visited_.clear();
#endif
#ifdef COMPUTE_APPROXIMATE_VECTOR
#ifdef USE_TWO_FIXED_MAX
    apr_search_layer_double_summary(apr_q, ef);
#endif
#ifdef USE_PLAIN_CHAR
    apr_search_layer(apr_q, ef);
#endif
#ifdef USE_TRESHOLD_SUMMARY
    aprSearchLayerSummary(apr_q, ef);
#endif
#ifdef USE_SELECTED_TRESHOLD_SUMMARY
    apr_search_layer_summary(apr_q, ef);
#endif
#ifndef USE_PLAIN_CHAR
    // TODO - throw runtime error if W_.size() is less then ef
    for (int i = 0; i < apr_W.size(); i++)
    {
        auto item = distances_.find(std::get<2>(apr_W[i]));
        if (item == distances_.end())
        {
            std::get<1>(apr_W[i]) = aprDistance(apr_q, aprGetNodeVector(std::get<2>(apr_W[i])));
        }
        else
        {
            std::get<1>(apr_W[i]) = item->second;
        }
    }
#endif
    sort(apr_W.begin(), apr_W.end(), CompareByDistanceInTuple());
#else
    search_layer(q, ef);
    sort(W_.begin(), W_.end(), neighborcmp_nearest());
#endif
}

#ifdef COMPUTE_APPROXIMATE_VECTOR
void HNSW::computeApproximateVector()
{
    shift = log(max_value) / log(2) - 8 < 0 ? 0 : log(max_value) / log(2) - 8 < 0;
    std::cout << "Compute approximate vectors (shift value " << shift << ")\n";
    auto start = std::chrono::system_clock::now();
    index_memory_ = 0;
    for (int k = 0; k <= actual_node_count_; k++)
    {
        auto vector = getNodeVector(k);
        auto apr_vector = aprGetNodeVector(k);
        for (int i = 0; i < vector_size_; i++)
        {
            unsigned int aux = vector[i];
            apr_vector[i] = aux >> shift;
        }
    }

//    clear_explored_count();
//    float *v = new float[vector_size_];
//    uint8_t  shifted_max_value = (int)max_value >> shift;
//    std::cout << "max value " << (int)shifted_max_value << "\n";
//    for (int i = 0; i < 100000; i++) {
//        //std::cout << "-------------------\n";
//        int i1 = rand() % actual_node_count_;
//        int i2 = rand() % actual_node_count_;
//        for (int k = 0; k < vector_size_; k++) {
//            v[k] =  (get_node_vector(i1)[k] + getNodeVector(i2)[k]) / 2;
//        }
//        knn(v, 50);
//    }
//
//    int histogram[500];
//    for (int i = 0; i < 200; i++) histogram[i] = 0;
//    for (auto& n: layers_[0]->nodes)
//    {
//        if (n->explored_count < 500)
//            histogram[n->explored_count]++;
//    }
//    for (int i = 0; i < 200; i++) std::cout << histogram[i] << "\n";

    int overflows = 0;
    for (auto& l : layers_)
    {
        // mask version
        if (l->layer_id == 0)
        {
            int i = 0;
            for (auto n : l->nodes)
            {
                //if (n->explored_count > EXPLORE_COUNT_TRESHOLD)
                    index_memory_ += computeSummaries(n, i++, vector_size_, &overflows);
#ifdef COLLECT_STAT
                stat_.neighbor_count += n->neighbors.size();
                stat_.node_count++;

//                for (auto neig: n->neighbors)
//                {
//                    neig.node->inward_count++;
//                }
#endif
            }
        }
    }
    auto end = std::chrono::system_clock::now();
    std::cout << "Compute approximate values: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " [s]\n";
#ifdef COLLECT_STAT
    stat_.overflows = overflows;
#ifdef COUNT_INWARD_DEGREE
    for (auto n : layers_[0]->nodes)
    {
        if (n->inward_count >= 200)
        {
            stat_.histogram[199]++;
        } else
        {
            stat_.histogram[n->inward_count]++;
        }
    }
#endif
#endif
}


int HNSW::computeSummaries(Node* node, pointer_t node_order, uint32_t vector_size, int* overflows)
{
    //neighbors_all = new uint8_t[neighbors.size() * vector_size_];
    //int i = 0;
    //for (auto ng : neighbors)
    //{
    //	memcpy(&neighbors_all[i * vector_size_], ng.node->apr_vector, vector_size_);
    //	i++;
    //}

#ifdef USE_TWO_FIXED_MAX
    std::priority_queue<std::pair<int, uint8_t>, std::vector<std::pair<int, uint8_t>>, CompareByDelta> delta;
    std::vector<Neighbors> &neighbors = node->neighbors;
    std::vector<int8_t>& summary = node->summary;
    int row1_size = (int)(VECTOR_FRAGMENT1 * vector_size_) * 2;
    int row2_size = (int)(VECTOR_FRAGMENT2 * vector_size_) * 2;
    int byte_size = (row1_size + row2_size) * neighbors.size();
    int first_block = row1_size * neighbors.size();
    auto pivot = apr_get_node_vector(node_order);

    summary.resize(byte_size);
    int neighbor_order = 0;
    for (auto ng : neighbors)
    {
        auto n = ng.node;
        auto n_apr_vector = aprGetNodeVector(ng.node_order);
        uint8_t over_treshold = 0;
        for (int i = 0; i < vector_size_; i++)
        {
            int diff = n_apr_vector[i] - pivot[i];

            //if (diff > DISTANCE_TRESHOLD)
            if (diff >= 0)
            {
                if (diff > 127)
                {
                    diff = 127;
                    (*overflows)++;
                }
                delta.emplace(diff, i);
            }
            //if (-diff > DISTANCE_TRESHOLD)
            if (-diff > 0)
            {
                diff = -diff;
                if (diff < -127)
                {
                    diff = -127;
                    (*overflows)++;
                }
                delta.emplace(diff, i);
            }
        }

        int i = 0;
        while (i < row1_size)
        {
            auto item = delta.top();
            delta.pop();
            summary[neighbor_order * row1_size + i] = item.second;
            int8_t diff = n_apr_vector[item.second] - pivot[item.second];
            summary[neighbor_order * row1_size + i + 1] = diff;
            i += 2;
        }

        i = 0;
        while (i < row2_size)
        {
            auto item = delta.top();
            delta.pop();
            summary[first_block + neighbor_order * row2_size + i] = item.second;
            int8_t diff = n_apr_vector[item.second] - pivot[item.second];
            summary[first_block + neighbor_order * row2_size + i + 1] = diff;
            i += 2;
        }

        while (!delta.empty())
        {
            delta.pop();
        }
        neighbor_order++;
    }
    return byte_size;
#else
    int neighbor_array_size = 0;
    auto pivot = aprGetNodeVector(node_order);
    auto &summary = node->summary;
    for (auto ng : node->neighbors)
    {
        auto n = ng.node;

        uint8_t over_treshold = 0;
        auto n_apr_vector = aprGetNodeVector(ng.node_order);
        for (int i = 0; i < vector_size; i++)
        {
            neighbor_array_size += n_apr_vector[i] - pivot[i] > DISTANCE_TRESHOLD || pivot[i] - n_apr_vector[i] > DISTANCE_TRESHOLD;
        }
    }
    summary.reserve(node->neighbors.size() + 2 * neighbor_array_size); // TODO - can be dangerous for higher dimensions than 256


    for (auto ng : node->neighbors)
    {
        auto n = ng.node;

        uint8_t over_treshold = 0;
        auto n_apr_vector = aprGetNodeVector(ng.node_order);
        for (int i = 0; i < vector_size; i++)
        {
            over_treshold += n_apr_vector[i] - pivot[i] > DISTANCE_TRESHOLD || pivot[i] - n_apr_vector[i] > DISTANCE_TRESHOLD;
        }

        summary.push_back(over_treshold);
        for (int i = 0; i < vector_size; i++)
        {
            int diff = (int)n_apr_vector[i] - pivot[i];
            if (diff > 127)
            {
                diff = 127;
                (*overflows)++;
            }

            if (diff < -127)
            {
                diff = -127;
                (*overflows)++;
            }

            if (diff > DISTANCE_TRESHOLD || -diff > DISTANCE_TRESHOLD)
            {
                summary.push_back(i);
                summary.push_back(diff);
            }
        }
    }

    return node->neighbors.size() + 2 * neighbor_array_size;
#endif
}
#endif

void HNSW::searchLayerOne(float* q)
{
    bool change = true;
    Node* actual;
    float actual_dist;
    pointer_t actual_node_order;

    actual = W_.front().node;
    actual_dist = W_.front().distance;
    actual_node_order = W_.front().node_order;
    W_.pop_back();
    while (change)
    {
        change = false;
        for (auto ne : actual->neighbors)
        {
            auto e = ne.node;
#ifdef VISIT_HASH
            if (!visited_.get(ne.node_order))
            {
                visited_.insert(ne.node_order);
#else
            if (e->actual_node_count_ < actual_node_count_)
            {
                e->actual_node_count_ = actual_node_count_;

#endif
                auto dist = distance(getNodeVector(ne.node_order), q);

                if (dist < actual_dist)
                {
                    change = true;
                    actual = e;
                    actual_dist = dist;
                    actual_node_order = ne.node_order;
                }
            }
        }
    }
    W_.clear();
    W_.emplace_back(actual, actual_dist, actual_node_order);
}



//////////////////////////////////////////////////////////////
////////////////////   Search Layer    ///////////////////////
//////////////////////////////////////////////////////////////

void HNSW::searchLayer(float* q, int ef)
{
    std::vector<std::tuple<Node*, int32_t>> C;
    
    //W_.push_back(ep);
    for (auto n : W_)
    {
        C.emplace_back(n.node, n.distance);
#ifdef VISIT_HASH
        visited_.insert(n.node_order);
#else
        n->actual_node_count_ = actual_node_count_;

#endif
    }
    std::make_heap(W_.begin(), W_.end(), neighborcmp_farest_heap());
    std::make_heap(C.begin(), C.end(), CompareByDistanceInTupleHeap());
    auto f = W_.front().distance;
    while (!C.empty())
    {
        auto c = C.front();
        std::pop_heap(C.begin(), C.end(), CompareByDistanceInTupleHeap());
        C.pop_back();
        if (std::get<1>(c) > f) break;
        for (auto ne : std::get<0>(c)->neighbors)
        {
            auto e = ne.node;
#ifdef VISIT_HASH
            if (!visited_.get(ne.node_order))
            { 
                visited_.insert(ne.node_order);
#else
            if (e->actual_node_count_ < actual_node_count_)
            {
                e->actual_node_count_ = actual_node_count_;

#endif
                //e->distance = distance_treshold_counting(q, e->vector, e->no_over_treshold);
                auto dist = distance(q, getNodeVector(ne.node_order));

                if (dist < f || W_.size() < ef)
                {
                    C.emplace_back(e, dist);
                    std::push_heap(C.begin(), C.end(), CompareByDistanceInTupleHeap());
                    W_.emplace_back(e, dist, ne.node_order);
                    std::push_heap(W_.begin(), W_.end(), neighborcmp_farest_heap());
                    if (W_.size() > ef)
                    {
                        std::pop_heap(W_.begin(), W_.end(), neighborcmp_farest_heap());
                        W_.pop_back();
                    }
                    f = W_.front().distance;
                }
#ifdef COLLECT_STAT
                else
                {
                    stat_.distance_computations_false++;
                }
#endif
            }
        }
    }
}

#ifdef COMPUTE_APPROXIMATE_VECTOR


void HNSW::aprSearchLayerOne(uint8_t* q)
{
    bool change = true;
    Node *actual;
    int32_t actual_dist;
    pointer_t actual_node_order;

    actual = std::get<0>(apr_W.front());
    actual_dist = std::get<1>(apr_W.front());
    actual_node_order = std::get<2>(apr_W.front());

    apr_W.pop_back();
    while (change)
    {
        change = false;
        for (auto ne : actual->neighbors)
        {
            auto e = ne.node;
#ifdef VISIT_HASH
            if (!visited_.get(ne.node_order))
            {
                visited_.insert(ne.node_order);
#else
            if (e->actual_node_count_ < actual_node_count_)
            {
                e->actual_node_count_ = actual_node_count_;
#endif
                auto dist = aprDistance(aprGetNodeVector(ne.node_order), q);

                if (dist < actual_dist)
                {
                    change = true;
                    actual = e;
                    actual_dist = dist;
                    actual_node_order = ne.node_order;
                }
#ifdef COLLECT_STAT
                else
                {
                    stat_.distance_computations_false++;
                }
#endif
            }
        }
    }
    apr_W.clear();
    apr_W.emplace_back(actual, actual_dist, actual_node_order);
}


void HNSW::aprSearchLayer(uint8_t* q, int ef)
{
    std::vector<std::tuple<Node*, int32_t>> C;

    //apr_W.push_back(ep);
    for (auto& n : apr_W)
    {
        C.emplace_back(std::get<0>(n), std::get<1>(n));
#ifdef VISIT_HASH
        visited_.insert(std::get<2>(n));
#else
        n->actual_node_count_ = actual_node_count_;

#endif
    }
    std::make_heap(apr_W.begin(), apr_W.end(), CompareByDistanceInTuple());
    std::make_heap(C.begin(), C.end(), CompareByDistanceInTupleHeap());
    auto f = std::get<1>(apr_W.front());
    while (!C.empty())
    {
        auto c = C.front();
        std::pop_heap(C.begin(), C.end(), CompareByDistanceInTupleHeap());
        C.pop_back();

        if (std::get<1>(c) > f) break;
        for (auto& ne : std::get<0>(c)->neighbors)
        {
            auto e = ne.node;
#ifdef VISIT_HASH
            if (!visited_.get(ne.node_order))
            {
                visited_.insert(ne.node_order);
#else
            if (e->actual_node_count_ < actual_node_count_)
            {
                e->actual_node_count_ = actual_node_count_;

#endif
                auto dist = aprDistance(q, aprGetNodeVector(ne.node_order));

                if (dist < f || apr_W.size() < ef)
                {
                    C.emplace_back(e, dist);
                    std::push_heap(C.begin(), C.end(), CompareByDistanceInTupleHeap());
                    apr_W.emplace_back(e, dist, ne.node_order);
                    std::push_heap(apr_W.begin(), apr_W.end(), CompareByDistanceInTuple());
                    if (apr_W.size() > ef)
                    {
                        std::pop_heap(apr_W.begin(), apr_W.end(), CompareByDistanceInTuple());
                        apr_W.pop_back();
                    }
                    f = std::get<1>(apr_W.front());
                }
#ifdef COLLECT_STAT
                else
                {
                    stat_.distance_computations_false++;
                }
#endif
            }
        }
    }
}


void HNSW::aprSearchLayerSummary(uint8_t* q, int ef)
{
    std::vector<std::tuple<Node*, int32_t, pointer_t>> C;

    distances_.clear();
    //W_.push_back(ep);
    for (auto& n : apr_W)
    {
        C.emplace_back(std::get<0>(n), -std::get<1>(n), std::get<2>(n));
    }
    std::make_heap(C.begin(), C.end(), CompareByDistanceInTuple());
    auto f = std::get<1>(apr_W.front());
#ifdef VISIT_HASH
    visited_.insert(std::get<2>(apr_W.front()));
#else
    std::get<0>(apr_W.front())->actual_node_count_ = actual_node_count_;
#endif

    while (!C.empty())
    {
        auto c = C.front();
        std::pop_heap(C.begin(), C.end(), CompareByDistanceInTuple());
        C.pop_back();
        Node* nc = std::get<0>(c);

        if (-std::get<1>(c) > f) break;
#ifdef COLLECT_STAT
        stat_.explored_nodes++;
#endif
        uint32_t c_distance = aprDistance(q, aprGetNodeVector(std::get<2>(c)), q_delta);
        distances_[std::get<2>(c)] = c_distance;
        int node_summary_position = 0;
        for (auto& ne : nc->neighbors)
        {
            auto e = ne.node;
#ifdef VISIT_HASH
            if (!visited_.get(ne.node_order))
            {
                visited_.insert(ne.node_order);
#else
            if (e->actual_node_count_ < actual_node_count_)
            {
                e->actual_node_count_ = actual_node_count_;
#endif
                int32_t distance;
//                if (nc->explored_count > EXPLORE_COUNT_TRESHOLD) {
//                    distance = aprDistanceSummary(nc->summary, node_summary_position, q_delta, c_distance);
//                } else {
//                    distance = aprDistance(apr_q, aprGetNodeVector(ne.node_order));
//                }
                distance = aprDistanceSummary(nc->summary, node_summary_position, q_delta, c_distance);


                if (distance < f || apr_W.size() < ef)
                {
                    C.emplace_back(e, -distance, ne.node_order);
                    std::push_heap(C.begin(), C.end(), CompareByDistanceInTuple());
                    apr_W.emplace_back(e, distance, ne.node_order);
                    std::push_heap(apr_W.begin(), apr_W.end(), CompareByDistanceInTuple());
                    if (apr_W.size() > ef)
                    {
                        std::pop_heap(apr_W.begin(), apr_W.end(), CompareByDistanceInTuple());
                        apr_W.pop_back();
                    }
                    f = std::get<1>(apr_W.front());
                }
            }
            else
            {
//                if (nc->explored_count > EXPLORE_COUNT_TRESHOLD)
                    node_summary_position = nc->summary[node_summary_position] * 2 + 1 + node_summary_position;
            }
        }
    }
}


void HNSW::aprSearchLayerDoubleSummary(uint8_t* q, int ef)
{
    std::vector<std::tuple<Node*, int32_t, pointer_t>> C;
    std::vector<std::tuple<int32_t, int32_t, Node*, pointer_t >> W1;

    distances_.clear();
    //W_.push_back(ep);
    for (auto& n : apr_W)
    {
        C.emplace_back(std::get<0>(n), -std::get<1>(n), std::get<2>(n));
    }
    std::make_heap(C.begin(), C.end(), CompareByDistanceInTuple());
    auto f = std::get<1>(apr_W.front());
#ifdef VISIT_HASH
    visited_.insert(std::get<2>(apr_W.front()));
#else
    std::get<0>(apr_W.front())->actual_node_count_ = actual_node_count_;
#endif

    while (!C.empty())
    {
        auto c = C.front();
        std::pop_heap(C.begin(), C.end(), CompareByDistanceInTuple());
        C.pop_back();
        Node* nc = std::get<0>(c);

        if (-std::get<1>(c) > f) break;
#ifdef COLLECT_STAT
        stat_.explored_nodes++;
#endif
        uint32_t c_distance = aprDistance(q, aprGetNodeVector(std::get<2>(c)), q_delta);
        distances_[std::get<2>(c)] = c_distance;
        int node_summary_position = 0;
        int row = 0;
        for (auto& ne : nc->neighbors)
        {
            auto e = ne.node;
#ifdef VISIT_HASH
            if (!visited_.get(ne.node_order))
            {
                visited_.insert(ne.node_order);
#else
            if (e->actual_node_count_ < actual_node_count_)
            {
                e->actual_node_count_ = actual_node_count_;
#endif
                int32_t distance = aprDistanceSummaryFixedSize(nc->summary, node_summary_position, q_delta, c_distance,
                                                               row1_size);
                W1.emplace_back(-distance, row, e, ne.node_order);
            }
            node_summary_position = row1_size + node_summary_position;
            row++;
        }


        if (W1.size() > 0)
        {
            std::make_heap(W1.begin(), W1.end(), CompareByFirstInTuple());
            int block1_size = row1_size * nc->neighbors.size();
            int node_counter = W1.size() < 6 ? std::min((int)(W1.size() - 1), 3) : W1.size() / 2;
            while (node_counter >= 0)
            {
                auto n = W1.front();
                int32_t distance = aprDistanceSummaryFixedSize(nc->summary, block1_size + std::get<1>(n) * row2_size,
                                                               q_delta, -std::get<0>(n), row2_size);


                if (distance < f || apr_W.size() < ef)
                {
                    C.emplace_back(std::get<2>(n), -distance, std::get<3>(n));
                    std::push_heap(C.begin(), C.end(), CompareByDistanceInTuple());
                    apr_W.emplace_back(std::get<2>(n), distance, std::get<3>(n));
                    std::push_heap(apr_W.begin(), apr_W.end(), CompareByDistanceInTuple());
                    if (apr_W.size() > ef)
                    {
                        std::pop_heap(apr_W.begin(), apr_W.end(), CompareByDistanceInTuple());
                        apr_W.pop_back();
                    }
                    f = std::get<1>(apr_W.front());
                }

                std::pop_heap(W1.begin(), W1.end(), CompareByFirstInTuple());
                W1.pop_back();
                node_counter--;
            }
            W1.clear();
        }
    }
}

#endif


// Algorithm 4
void HNSW::selectNeighbors(std::vector<Neighbors>& W, std::vector<Neighbors>& R, int M, bool keepPruned)
{
    //std::vector<Node*> Wd;
    //Wd.reserve(M_);

    if (W.size() < M)
    {
        R = W;
        return;
    }
    int s = 2;
    int i = 2;
    R.clear();
    std::sort(W.begin(), W.end(), neighborcmp_nearest());  // TODO - use heap instead of sort
    R.emplace_back(W[0]);
    R.emplace_back(W[1]);

    while(i < W.size() && s < M)
    {
        bool q_is_close = true;
        for (int j = 0; j < R.size(); j++)
        {
            float dist = distance(getNodeVector(W[i].node_order), getNodeVector(R[j].node_order));
#ifdef NSG
            if (dist < W_[i].distance && R[j].distance < W_[i].distance)
#else
            if (dist < W[i].distance)
#endif
            {
                q_is_close = false;
                break;
            }

        }
#ifdef COUNT_INWARD_DEGREE
        if (q_is_close || W_[i].node->inward_count <= M_/4)
#else
        if (q_is_close)
#endif
        {
            R.emplace_back(W[i]);
            s++;
        }
        //else
        //{
        //    Wd.push_back(W_[i]);
        //}
        i++;
    }
    //if (keepPruned)
    //{

    //    //std::sort(Wd.begin(), Wd.end(), nodecmp_farest_with_overflow());
    //    int Wdi = 0;
    //    while (s < M_ && Wdi < Wd.size())
    //    {
    //        R.push_back(Wd[Wdi++]);
    //        s++;
    //    }
    //}



    /////////////////////////////////////////////// using heap instead of sort, not working quite well

    //std::priority_queue<std::pair<int32_t, Node*>, std::vector<std::pair<int32_t, Node*>>, CompareByFirst> A;
    //for (int i = 1; i < W_.size(); i++)
    //{
    //    A.emplace(-W_[i]->distance, W_[i]);
    //}

    //R.clear();
    //R.push_back(W_[0]);
    //while (!A.empty() && R.size() < M_)
    //{
    //    auto n = A.top();
    //    auto ndist = -n.first;
    //    A.pop();
    //    bool q_is_close = true;
    //    for (int j = 0; j < R.size(); j++)
    //    {
    //        float dist = distance(n.second->vector, R[j]->vector);
    //        if (dist < ndist)
    //        {
    //            q_is_close = false;
    //            break;
    //        }
    //    }
    //    if (q_is_close)
    //    {
    //        R.push_back(n.second);
    //    }
    //}
}


//////////////////////////////////////////////////////////////
////////////////////   Serialization    ///////////////////////
//////////////////////////////////////////////////////////////

void HNSW::saveGraph(const char* filename)
{
    try {
        std::ofstream f(filename, std::ios::binary);

        if (!f.is_open())
        {
            std::cout << "Save file " << filename << "was not sucesfully opened\n";
            return;
        }

        f.write(reinterpret_cast<const char *>(&max_node_count_), sizeof(max_node_count_));
        f.write(reinterpret_cast<const char *>(&vector_size_), sizeof(vector_size_));
        f.write(reinterpret_cast<const char *>(&actual_node_count_), sizeof(actual_node_count_));
        f.write(reinterpret_cast<const char *>(&max_value), sizeof(max_value));
        f.write(reinterpret_cast<const char *>(vector_data_), sizeof(float) * max_node_count_ * vector_size_);
        auto l_size = layers_.size();
        f.write(reinterpret_cast<const char *>(&l_size), sizeof(l_size));

        std::unordered_map<Node*, int> node_map;
        std::vector<Node*> node_vector;
        for (auto& l : layers_)
        {
            auto l_size = l->nodes.size();
            f.write(reinterpret_cast<const char *>(&l_size), sizeof(l_size));
            for (auto& n : l->nodes)
            {
                auto n_size = n->neighbors.size();
                f.write(reinterpret_cast<const char *>(&n_size), sizeof(n_size));
                node_map[n] = node_vector.size();
                node_vector.emplace_back(n);
            }
        }

        for (auto& l : layers_)
        {
            f.write(reinterpret_cast<const char *>(&l->ep_node_order), sizeof(l->ep_node_order));
            auto order = node_map[l->enter_point];
            f.write(reinterpret_cast<const char *>(&order), sizeof(order));
            for (auto& n : l->nodes)
            {
                order = node_map[n->lower_layer];
                f.write(reinterpret_cast<const char *>(&order), sizeof(order));
                auto n_size = n->neighbors.size();
                f.write(reinterpret_cast<const char *>(&n_size), sizeof(n_size));
                for (auto& nb : n->neighbors)
                {
                    order = node_map[nb.node];
                    f.write(reinterpret_cast<const char *>(&order), sizeof(order));
                    f.write(reinterpret_cast<const char *>(&nb.node_order), sizeof(nb.node_order));
                }
            }
        }

        f.close();
    }
    catch (std::exception e) {
        std::cout << e.what() << "\n";
        exit(0);
    }
}



void HNSW::loadGraph(const char* filename)
{
    try {
        std::ifstream f(filename, std::ios::binary);
        if (!f.is_open()) {
            std::cout << "Save file " << filename << "was not sucesfully opened\n";
            return;
        }

        max_value = 220; // TODO write and read
        f.read(reinterpret_cast<char *>(&max_node_count_), sizeof(max_node_count_));
        f.read(reinterpret_cast<char *>(&vector_size_), sizeof(vector_size_));
        f.read(reinterpret_cast<char *>(&actual_node_count_), sizeof(actual_node_count_));
        f.read(reinterpret_cast<char *>(&max_value), sizeof(max_value));
        init(vector_size_, max_node_count_);
        f.read(reinterpret_cast<char *>(vector_data_), sizeof(float) * max_node_count_ * vector_size_);
        auto l_size = layers_.size();
        f.read(reinterpret_cast<char *>(&l_size), sizeof(l_size));

        for (int i = 0; i < l_size; i++)
        {
            layers_.emplace_back(std::make_unique<Layer>());
        }

        std::vector<Node *> node_vector;
        for (auto &l : layers_) {
            std::size_t l_size;
            f.read(reinterpret_cast<char *>(&l_size), sizeof(l_size));
            //l->nodes.resize(l_size);
            for (int i = 0; i < l_size; i++) {
                std::size_t n_size;
                f.read(reinterpret_cast<char *>(&n_size), sizeof(n_size));
                auto new_node = new Node(n_size, nullptr);
                node_vector.push_back(new_node);
                l->nodes.emplace_back(new_node);
            }
        }


        for (auto &l : layers_) {
            f.read(reinterpret_cast<char *>(&l->ep_node_order), sizeof(l->ep_node_order));
            int order;
            f.read(reinterpret_cast<char *>(&order), sizeof(order));
            l->enter_point = node_vector[order];
            for (auto &n : l->nodes) {
                f.read(reinterpret_cast<char *>(&order), sizeof(order));
                n->lower_layer = node_vector[order];
                size_t n_size;
                f.read(reinterpret_cast<char *>(&n_size), sizeof(n_size));
                for (int i = 0; i < n_size; i++) {
                    pointer_t node_order;
                    f.read(reinterpret_cast<char *>(&order), sizeof(order));
                    f.read(reinterpret_cast<char *>(&node_order), sizeof(node_order));
                    n->neighbors.emplace_back(node_vector[order], 0, node_order);
                }
            }
        }
        f.close();

    }
    catch (std::exception e) {
        std::cout << e.what() << "\n";
        exit(0);
    }

}


//////////////////////////////////////////////////////////////
////////////////////     Rest      ///////////////////////////
//////////////////////////////////////////////////////////////


void HNSW::printInfo(bool all)
{
    std::cout << "HNSW info\n";
    for (int i = 0; i < layers_.size(); i++)
    {
        std::cout << "Layer " << i << " " << layers_[i]->nodes.size() << "\n";
    }
    std::cout << "Index memory: " << index_memory_ / (1024 * 1024) << " [MB]\n";
    std::cout << "M_: " << M_ << "\n";
    std::cout << "Mmax_: " << Mmax_ << "\n";
    std::cout << "ml_: " << ml_ << "\n";
    std::cout << "efConstruction_: " << efConstruction_ << "\n";
#ifdef COMPUTE_APPROXIMATE_VECTOR   
#ifdef USE_TWO_FIXED_MAX
    std::cout << "VECTOR_FRAGMENT1: " << (float)VECTOR_FRAGMENT1 << "\n";
    std::cout << "VECTOR_FRAGMENT2: " << (float)VECTOR_FRAGMENT2 << "\n";
#else
    std::cout << "DISTANCE_TRESHOLD: " << (int)DISTANCE_TRESHOLD << "\n";
#endif
#endif
}


void HNSW::print(int max_count)
{

        for (int i = 0; i < max_count; i++) {
            std::cout << i << ": ";
            auto node = layers_[0]->nodes[i];
            std::vector<int> d;
            for (auto ne : node->neighbors)
            {
                d.push_back(ne.node_order);
            }
            std::sort(d.begin(), d.end());
            for (auto item : d) {
                std::cout << item << "  ";
            }
            std::cout << "\n";
        }
        std::cout << "\n\n";
}

