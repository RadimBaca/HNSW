#include "HNSW.h"

void HNSW::init(uint32_t vector_size, uint32_t p_max_node_count)
{
    data_cleaned = false;
    max_node_count = p_max_node_count;
    setVectorSize(vector_size);
    vector_data = new float[p_max_node_count * vector_size];
    vector_data_row_size = sizeof(float) * vector_size;
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
    delete[] vector_data;
#ifdef COMPUTE_APPROXIMATE_VECTOR
    delete[] apr_vector_data;
    delete[] min_vector;
    delete[] max_vector;
    delete[] apr_q;
    delete[] q_delta;
#endif
}

void HNSW::create(const char* filename, const char* datasetname)
{

    hsize_t dimensions[2];

    hdfReader::getDimensions(filename, datasetname, &dimensions);
    std::clog << "dimensions " <<
        (unsigned long)(dimensions[0]) << " x " <<
        (unsigned long)(dimensions[1]) << std::endl;
 
    init(dimensions[1], dimensions[0]);
    float* data = new float[max_node_count * vector_size];
    hdfReader::readData(filename, datasetname, data);

    for (int i = 0; i < max_node_count; i++)
    {
        insert(&data[i * vector_size]);
    }
    delete[] data;
#ifdef COMPUTE_APPROXIMATE_VECTOR


    for (int i = 0; i < vector_size; i++)
    {
        max_vector[i] = (max_vector[i] - min_vector[i]) * 1.1;
    }
    max_value = max_value * 1.05;

#ifdef APR_DEBUG
    std::cout << "Min\tMax\n";
    for (int i = 0; i < vector_size; i++)
    {
        std::cout << min_vector[i] << "\t" << max_vector[i] << "\n";
    }
#endif
    stat.clear();
    computeApproximateVector();
#endif
#ifdef COLLECT_STAT
    stat.print();
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
            if (std::get<0>(apr_W[c1])->node_order == data_result[i * dimensions_result[1] + c2])
#else
            if (W[c1].node_order == data_result[i * dimensions_result[1] + c2])
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
    std::cout << "Precision: " << positive / (dimensions_query[0] * k) << "\n";

    double sum = 0;
    double min_time;
    for (int i = 0; i < 3; i++)
    {
#ifdef COLLECT_STAT
        stat.clear();
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
        std::cout << (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / dimensions_query[0] << " [ms]; ";
    }
    std::cout << "avg: " << sum / (3 * dimensions_query[0]) << " [ms]; " << "min: " << min_time / dimensions_query[0] << " [ms]; \n";
#ifdef COLLECT_STAT
    stat.print();
#endif    

}


//////////////////////////////////////////////////////////////
///////////////////////   DML op    //////////////////////////
//////////////////////////////////////////////////////////////

void HNSW::insert(float* q)
{
    int32_t l = -log(((float)(rand() % 10000 + 1)) / 10000) * ml;
    int32_t L = layers.size() - 1;
    Node* down_node = nullptr;
    Node* prev = nullptr;
    Node* ep = nullptr;
    pointer_t ep_node_order;
    std::vector<Neighbors> R;
    std::vector<Neighbors> Woverflow;
    std::vector<Neighbors> Roverflow;

    visit_id++;
    if (visit_id >= max_node_count)
    {
        throw std::runtime_error("HNSW: exceeded maximal number of nodes\n");
    }
    memcpy(get_node_vector(visit_id), q, sizeof(float) * vector_size);

#ifdef VISIT_HASH
    visited.clear();
#endif


#ifdef COMPUTE_APPROXIMATE_VECTOR
    if (visit_id == 0)
    {
        memcpy(min_vector, q, sizeof(float) * vector_size);
        memcpy(max_vector, q, sizeof(float) * vector_size);
        min_value = q[0];
        max_value = q[0];
    }
    for (int i = 0; i < vector_size; i++)
    {
        min_vector[i] = q[i] < min_vector[i] ? q[i] : min_vector[i];
        max_vector[i] = q[i] > max_vector[i] ? q[i] : max_vector[i];
        min_value = std::min(min_value, q[i]);
        max_value = std::max(max_value, q[i]);
    }
#endif

    if ((visit_id % 1000) == 0)
    {
        std::cout << visit_id << "\n";
    }

    W.clear();
    if (L >= 0)
    {
        ep = layers[L]->enter_point;
        ep_node_order = layers[L]->ep_node_order;
        auto dist = distance(get_node_vector(ep_node_order), q);

#ifdef VISIT_HASH
        visited.insert(ep_node_order);
#else
        ep->visit_id = visit_id;
#endif
        W.emplace_back(ep, dist, ep_node_order);
    }
    for (int32_t i = L; i > l; i--)
    {
        search_layer_one(q);
        change_layer();
    }
    for (int32_t i = std::min(L, l); i >= 0; i--)
    {
        Node* new_node = new Node(visit_id, vector_size, Mmax, nullptr);
        layers[i]->node_count++;
        layers[i]->nodes.push_back(new_node);

        auto actualMmax = (i == 0 ? Mmax0 : Mmax);
#ifdef DEBUG_NET
        node->layer = i;
#endif

        if (l > L && i == L)
        {
            down_node = new_node; // remember the top node, if we are going to add layers
        }

        // the main logic
#ifdef VISIT_HASH
        visited.clear();
#endif
        search_layer(q, efConstruction);
        select_neighbors(W, R, M, false, true, true);

        for (int i = 0;i < R.size(); i++)
        {
            auto e = &R[i];
            auto enode = e->node;
            new_node->neighbors.emplace_back(enode, e->distance, e->node_order);

            bool insert_new = true;
            if (enode->neighbors.size() == actualMmax)
            {
                if (!enode->neighbors_sorted)
                {
                    std::sort(enode->neighbors.begin(), enode->neighbors.end(), neighborcmp_nearest());
                    enode->neighbors_sorted = true;
                }
#ifdef SELECT_NEIGHBORS1
                if (e->neighbors[actualMmax - 1].distance > e->distance)
                {
                    e->neighbors.pop_back();
                    e->neighbors.emplace_back(node, e->distance, e->node_order);
                    std::sort(e->neighbors.begin(), e->neighbors.end(), neighborcmp_nearest()); // TODO - insert instead of sort
                }
#else
                if (e->distance < enode->neighbors[enode->neighbors.size() - 1].distance)
                {
                    Roverflow.clear();
                    Woverflow.clear();
                    int pos = 0;
                    while (enode->neighbors[pos].distance < e->distance)
                    {
                        Roverflow.push_back(enode->neighbors[pos]);
                        pos++;
                    }
                    Woverflow.emplace_back(new_node, e->distance, visit_id);
                    while (pos < enode->neighbors.size())
                    {
                        Woverflow.emplace_back(enode->neighbors[pos]);
                        pos++;
                    }
                    select_neighbors(Woverflow, Roverflow, actualMmax - Roverflow.size(), false, true, false);
                    enode->neighbors.clear();
                    for (auto r : Roverflow)
                    {
                        enode->neighbors.emplace_back(r);
                    }
                }
#endif
            }
            else {
                //if (is_distant_node(e->neighbors, node->vector, e->distance))
                {
                    enode->neighbors.emplace_back(new_node, e->distance, visit_id);
                }
            }            
        }

        // connecting the nodes in different layers
        if (prev != nullptr)
        {
            prev->lower_layer = new_node;
        }
        prev = new_node;

        // switch the actual layer (W nodes are replaced by their bottom counterparts)
        if (i > 0) change_layer();
    }
    if (l > L)
    {
        // adding layers
        for (int32_t i = L + 1; i <= l; i++)
        {
            Node* node = new Node(visit_id, vector_size, Mmax, down_node); // TODO resolve memory leak
            layers.emplace_back(std::make_unique<Layer>(node, visit_id));
            down_node = node;
        }
    }
}


void HNSW::knn(float* q, int k, int ef)
{
    int32_t L = layers.size() - 1;
    Node* down_node = nullptr;
    Node* prev = nullptr;
    Node* ep = nullptr;
    int ep_node_order;
    visit_id++;

#ifdef VISIT_HASH
    visited.clear();
#endif
    ep = layers[L]->enter_point;
    ep_node_order = layers[L]->ep_node_order;
#ifdef COMPUTE_APPROXIMATE_VECTOR
    apr_W.clear();
    Node::computeApproximateVector(apr_q, q, shift, vector_size);
    auto dist = apr_distance(apr_get_node_vector(layers[L]->ep_node_order), apr_q);
    apr_W.emplace_back(ep, dist, ep->node_order);
#else
    W.clear();
    auto dist = distance(get_node_vector(ep_node_order), q);
    W.emplace_back(ep, dist, ep_node_order);
#endif
#ifdef VISIT_HASH
    visited.insert(ep_node_order);
#else
    ep->visit_id = visit_id;
#endif

    for (int32_t i = L; i > 0; i--)
    {
#ifdef COMPUTE_APPROXIMATE_VECTOR
        apr_search_layer_one(apr_q);
        apr_change_layer();
#else
        search_layer_one(q);
        change_layer();
#endif
    }
#ifdef VISIT_HASH
    visited.clear();
#endif
#ifdef COMPUTE_APPROXIMATE_VECTOR
#ifdef USE_TWO_FIXED_MAX
    apr_search_layer_double_summary(apr_q, ef);
#else
    apr_search_layer_summary(apr_q, ef);
#endif
    // TODO - throw runtime error if W.size() is less then ef
    for (int i = 0; i < apr_W.size(); i++)
    {
        auto item = distances.find(std::get<2>(apr_W[i]));
        if (item == distances.end())
        {
            std::get<1>(apr_W[i]) = apr_distance(apr_q, apr_get_node_vector(std::get<2>(apr_W[i])));
        }
        else
        {
            std::get<1>(apr_W[i]) = item->second;
        }
    }
    sort(apr_W.begin(), apr_W.end(), CompareByDistanceInTuple());
#else
    search_layer(q, ef);
    sort(W.begin(), W.end(), neighborcmp_nearest());
#endif
}

#ifdef COMPUTE_APPROXIMATE_VECTOR
void HNSW::computeApproximateVector()
{
    shift = log(max_value) / log(2) - 8 < 0 ? 0 : log(max_value) / log(2) - 8 < 0;
    std::cout << "Compute approximate vectors (shift value " << shift << ")\n";
    auto start = std::chrono::system_clock::now();
    index_memory = 0;
    for (int k = 0; k <= visit_id; k++)
    {
        auto vector = get_node_vector(k);
        auto apr_vector = apr_get_node_vector(k);
        for (int i = 0; i < vector_size; i++)
        {
            unsigned int aux = vector[i];
            apr_vector[i] = aux >> shift;
        }
    }

    for (auto& l : layers)
    {
        // mask version
        if (l->layer_id == 0)
        {
            for (auto n : l->nodes)
            {               

                index_memory += computeSummaries(n, vector_size, &stat.overflows);
#ifdef COLLECT_STAT
                stat.neighbor_count += n->neighbors.size();
                stat.node_count++;
#endif
            }
        }
    }
    auto end = std::chrono::system_clock::now();
    std::cout << "Compute approximate values: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " [s]\n";
}


int HNSW::computeSummaries(Node* node, uint32_t vector_size, int* overflows)
{
    //neighbors_all = new uint8_t[neighbors.size() * vector_size];
    //int i = 0;
    //for (auto ng : neighbors)
    //{
    //	memcpy(&neighbors_all[i * vector_size], ng.node->apr_vector, vector_size);
    //	i++;
    //}

#ifdef USE_TWO_FIXED_MAX
    std::priority_queue<std::pair<int, uint8_t>, std::vector<std::pair<int, uint8_t>>, CompareByDelta> delta;

    int row1_size = (int)(VECTOR_FRAGMENT1 * vector_size) * 2;
    int row2_size = (int)(VECTOR_FRAGMENT2 * vector_size) * 2;
    int byte_size = (row1_size + row2_size) * neighbors.size();
    int first_block = row1_size * neighbors.size();
    summary.resize(byte_size);
    int node_order = 0;
    for (auto ng : neighbors)
    {
        auto n = ng.node;
        uint8_t over_treshold = 0;
        for (int i = 0; i < vector_size; i++)
        {
            int diff = n->apr_vector[i] - pivot[i];

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
            summary[node_order * row1_size + i] = item.second;
            int8_t diff = n->apr_vector[item.second] - pivot[item.second];
            summary[node_order * row1_size + i + 1] = diff;
            i += 2;
        }

        i = 0;
        while (i < row2_size)
        {
            auto item = delta.top();
            delta.pop();
            summary[first_block + node_order * row2_size + i] = item.second;
            int8_t diff = n->apr_vector[item.second] - pivot[item.second];
            summary[first_block + node_order * row2_size + i + 1] = diff;
            i += 2;
        }

        while (!delta.empty())
        {
            delta.pop();
        }
        node_order++;
    }
    return byte_size;
#else
    int neighbor_array_size = 0;
    auto pivot = apr_get_node_vector(node->node_order);
    auto &summary = node->summary;
    for (auto ng : node->neighbors)
    {
        auto n = ng.node;

        uint8_t over_treshold = 0;
        auto n_apr_vector = apr_get_node_vector(ng.node_order);
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
        auto n_apr_vector = apr_get_node_vector(ng.node_order);
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

void HNSW::search_layer_one(float* q)
{
    bool change = true;
    Node* actual;
    float actual_dist;
    pointer_t actual_node_order;

    actual = W.front().node;
    actual_dist = W.front().distance;
    actual_node_order = W.front().node_order;
    W.pop_back();
    while (change)
    {
        change = false;
        for (auto ne : actual->neighbors)
        {
            auto e = ne.node;
#ifdef VISIT_HASH
            if (!visited.get(ne.node_order))
            {
                visited.insert(ne.node_order);
#else
            if (e->visit_id < visit_id)
            {
                e->visit_id = visit_id;

#endif
                auto dist = distance(get_node_vector(ne.node_order), q);

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
    W.clear();
    W.emplace_back(actual, actual_dist, actual_node_order);
}

void HNSW::search_layer(float* q, int ef)
{
    std::vector<std::tuple<Node*, int32_t>> C;
    
    //W.push_back(ep);
    for (auto n : W)
    {
        C.emplace_back(n.node, n.distance);
#ifdef VISIT_HASH
        visited.insert(n.node_order);
#else
        n->visit_id = visit_id;

#endif
    }
    std::make_heap(W.begin(), W.end(), neighborcmp_farest_heap());
    std::make_heap(C.begin(), C.end(), CompareByDistanceInTupleHeap());
    auto f = W.front().distance;
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
            if (!visited.get(ne.node_order))
            { 
                visited.insert(ne.node_order);
#else
            if (e->visit_id < visit_id)
            {
                e->visit_id = visit_id;

#endif
                //e->distance = distance_treshold_counting(q, e->vector, e->no_over_treshold);
                auto dist = distance(q, get_node_vector(ne.node_order));

                if (dist < f || W.size() < ef)
                {
                    C.emplace_back(e, dist);
                    std::push_heap(C.begin(), C.end(), CompareByDistanceInTupleHeap());
                    W.emplace_back(e, dist, ne.node_order);
                    std::push_heap(W.begin(), W.end(), neighborcmp_farest_heap());
                    if (W.size() > ef)
                    {
                        std::pop_heap(W.begin(), W.end(), neighborcmp_farest_heap());
                        W.pop_back();
                    }
                    f = W.front().distance;
                }
#ifdef COLLECT_STAT
                else
                {
                    stat.distance_computations_false++;
                }
#endif
            }
        }
    }
}

#ifdef COMPUTE_APPROXIMATE_VECTOR


void HNSW::apr_search_layer_one(uint8_t* q)
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
            if (!visited.get(ne.node_order))
            {
                visited.insert(ne.node_order);
#else
            if (e->visit_id < visit_id)
            {
                e->visit_id = visit_id;
#endif
                auto dist = apr_distance(apr_get_node_vector(ne.node_order), q);

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
                    stat.distance_computations_false++;
                }
#endif
            }
        }
    }
    apr_W.clear();
    apr_W.emplace_back(actual, actual_dist, actual_node_order);
}

//
//void HNSW::apr_search_layer(uint8_t* apr_q, int ef)
//{
//    std::vector<Node*> C;
//
//    //W.push_back(ep);
//    for (auto n : apr_W)
//    {
//        C.push_back(std::get<0>(n));
//    }
//    std::make_heap(apr_W.begin(), apr_W.end(), CompareByDistanceInTuple());
//    std::make_heap(C.begin(), C.end(), apr_nodecmp_nearest());
//    while (!C.empty())
//    {
//#ifdef APR_DEBUG
//        if (C.size() > 1 && C[0]->apr_distance > C[1]->apr_distance)
//        {
//            std::cout << "HNSW::apr_search_layer - C not ordered!!\n";
//        }
//#endif
//        auto c = C.front();
//        auto f = std::get<1>(apr_W.front());
//        std::pop_heap(C.begin(), C.end(), apr_nodecmp_nearest());
//        C.pop_back();
//
//        if (c->apr_distance > f) break;
//        for (auto ne : c->neighbors)
//        {
//            auto e = ne.node;
//#ifdef VISIT_HASH
//            if (!visited.get(e->node_order))
//            {
//                visited.insert(e->node_order);
//#else
//            if (e->visit_id < visit_id)
//            {
//                e->visit_id = visit_id;
//#endif
//                auto dist = e->apr_distance = apr_distance(e->apr_vector, apr_q);
//               
//                if (e->apr_distance < f || apr_W.size() < ef)
//                {
//                    C.push_back(e);
//                    std::push_heap(C.begin(), C.end(), apr_nodecmp_nearest());
//                    apr_W.emplace_back(e, dist, e->node_order);
//                    std::push_heap(apr_W.begin(), apr_W.end(), CompareByDistanceInTuple());
//                    if (apr_W.size() > ef)
//                    {
//                        std::pop_heap(apr_W.begin(), apr_W.end(), CompareByDistanceInTuple());
//                        apr_W.pop_back();
//                    }
//                    f = std::get<1>(apr_W.front());
//                }
//            }
//        }
//    }
//}


void HNSW::apr_search_layer_summary(uint8_t* apr_q, int ef)
{
    std::priority_queue<std::pair<int32_t, Node*>, std::vector<std::pair<int32_t, Node*>>, CompareByFirst> C;

    distances.clear();
    //W.push_back(ep);
    for (auto n : apr_W)
    {
        C.emplace(-std::get<1>(n), std::get<0>(n));
    }
    auto f = std::get<1>(apr_W.front());
#ifdef VISIT_HASH
    visited.insert(std::get<2>(apr_W.front()));
#else
    std::get<0>(apr_W.front())->visit_id = visit_id;
#endif

    while (!C.empty())
    {
        auto c = C.top();
        C.pop();
        Node* nc = c.second;

        if (-c.first > f) break;
#ifdef COLLECT_STAT
        stat.explored_nodes++;
#endif
        uint32_t c_distance = apr_distance(apr_q, apr_get_node_vector(nc->node_order), q_delta);
        distances[nc->node_order] = c_distance;
        int node_summary_position = 0;
        for (auto ne : nc->neighbors)
        {
            auto e = ne.node;
#ifdef VISIT_HASH
            if (!visited.get(ne.node_order))
            {
                visited.insert(ne.node_order);
#else
            if (e->visit_id < visit_id)
            {
                e->visit_id = visit_id;
#endif

                int32_t distance = apr_distance_summary(nc->summary, node_summary_position, q_delta, c_distance);

                if (distance < f || apr_W.size() < ef)
                {
                    C.emplace(-distance, e);
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
                node_summary_position = nc->summary[node_summary_position] * 2 + 1 + node_summary_position;
            }
        }
    }
}

//
//void HNSW::apr_search_layer_double_summary(uint8_t* apr_q, int ef)
//{
//    std::priority_queue<std::pair<int32_t, Node*>, std::vector<std::pair<int32_t, Node*>>, CompareByFirst> C;
//    std::vector<std::tuple<int32_t, int32_t, Node*>> W1;
//
//    distances.clear();
//    //W.push_back(ep);
//    for (auto n : W)
//    {
//        C.emplace(-n->apr_distance, n);
//        WP.emplace(n->apr_distance, n);
//    }
//    W.clear();
//    auto f = WP.top().first;
//
//    while (!C.empty())
//    {
//#ifdef APR_DEBUG
//        if (C.size() > 1 && C[0]->apr_distance > C[1]->apr_distance)
//        {
//            std::cout << "HNSW::apr_search_layer - C not ordered!!\n";
//        }
//#endif
//        auto c = C.top();
//        C.pop();
//        Node* nc = c.second;
//
//        if (-c.first > f) break;
//#ifdef COLLECT_STAT
//        stat.explored_nodes++;
//#endif
//        uint32_t c_distance = apr_distance(apr_q, nc->apr_vector, q_delta);
//        distances[nc->uniqueId] = c_distance;
//        int node_summary_position = 0;
//        int row = 0;
//        for (auto ne : nc->neighbors)
//        {
//            auto e = ne.node;
//#ifdef VISIT_HASH
//            if (!visited.get(ne.uniqueId))
//            {
//                visited.insert(ne.uniqueId);
//#else
//            if (e->visit_id < visit_id)
//            {
//                e->visit_id = visit_id;
//#endif
//
//                int32_t distance = apr_distance_summary_fixed(nc->summary, node_summary_position, q_delta, c_distance, row1_size);
//                W1.emplace_back(-distance, row, e);
//            }
//            node_summary_position = row1_size + node_summary_position;            
//            row++;
//        }
//
//        if (W1.size() > 0)
//        {
//            std::make_heap(W1.begin(), W1.end(), CompareByFirstInTuple());
//            int block1_size = row1_size * nc->neighbors.size();
//            int node_counter = W1.size() < 4 ? W1.size() : W1.size() / 2;
//            while (node_counter >= 0)
//            {
//                auto n = W1.front();
//                int32_t distance = apr_distance_summary_fixed(nc->summary, block1_size + std::get<1>(n) * row2_size, q_delta, -std::get<0>(n), row2_size);
//
//                if (distance < f || WP.size() < ef)
//                {
//                    C.emplace(-distance, std::get<2>(n));
//                    WP.emplace(distance, std::get<2>(n));
//                    if (WP.size() > ef)
//                    {
//                        WP.pop();
//                    }
//                    f = WP.top().first;
//                }
//
//                std::pop_heap(W1.begin(), W1.end(), CompareByFirstInTuple());
//                W1.pop_back();
//                node_counter--;
//            }
//            W1.clear();
//        }
//    }
//
//    while (!WP.empty())
//    {
//        W.push_back(WP.top().second);
//        WP.pop();
//    }
//}

#endif

#ifdef SELECT_NEIGHBORS1

// Algorithm 3 - the simplest algorithm select just the M most closest
void HNSW::select_neighbors( std::vector<Node*>& R, int M, bool keepPruned, bool considerOverTreshold)
{
    R.clear();
    std::sort(W.begin(), W.end(), nodecmp_farest()); 
    for (int i = 0; i < M && i < W.size(); i++)
    {
        R.push_back(W[i]);
    }
}



#else

// Algorithm 4
void HNSW::select_neighbors(std::vector<Neighbors>& W, std::vector<Neighbors>& R, int M, bool keepPruned, bool considerOverTreshold, bool sort)
{
    //std::vector<Node*> Wd;
    //Wd.reserve(M);


    //if (W.size() < M) -- TODO
    int s = 0;
    int i = 0;
    if (sort)
    {
        R.clear();
        std::sort(W.begin(), W.end(), neighborcmp_nearest());  // TODO - use heap instead of sort
        R.emplace_back(W[0]);
        s = 1;
        i = 1;
    }
    while(i < W.size() && s < M)
    {
        bool q_is_close = true;
        for (int j = 0; j < R.size(); j++)
        {
            float dist = distance(get_node_vector(W[i].node_order), get_node_vector(R[j].node_order));
            if (dist < W[i].distance)
            {
                q_is_close = false;
                break;
            }
        }
        if (q_is_close)
        {
            R.emplace_back(W[i]);
            s++;
        }
        //else
        //{
        //    Wd.push_back(W[i]);
        //}
        i++;
    }
    //if (keepPruned)
    //{

    //    //std::sort(Wd.begin(), Wd.end(), nodecmp_farest_with_overflow());
    //    int Wdi = 0;
    //    while (s < M && Wdi < Wd.size())
    //    {
    //        R.push_back(Wd[Wdi++]);
    //        s++;
    //    }
    //}



    /////////////////////////////////////////////// using heap instead of sort, not working quite well

    //std::priority_queue<std::pair<int32_t, Node*>, std::vector<std::pair<int32_t, Node*>>, CompareByFirst> A;
    //for (int i = 1; i < W.size(); i++)
    //{
    //    A.emplace(-W[i]->distance, W[i]);
    //}

    //R.clear();
    //R.push_back(W[0]);
    //while (!A.empty() && R.size() < M)
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
#endif

void HNSW::printInfo(bool all)
{
    std::cout << "HNSW info\n";
    for (int i = 0; i < layers.size(); i++)
    {
        std::cout << "Layer " << i << " " << layers[i]->node_count << "\n";
        //if (all)
        //{
        //    for (auto n : layers[i]->nodes)
        //    {
        //        n->print(vector_size);
        //    }
        //}
    }
    std::cout << "Index memory: " << index_memory / (1024 * 1024) << " [MB]\n";
    std::cout << "M: " << M << "\n";
    std::cout << "Mmax: " << Mmax << "\n";
    std::cout << "ml: " << ml << "\n";
    std::cout << "efConstruction: " << efConstruction << "\n";
#ifdef COMPUTE_APPROXIMATE_VECTOR   
#ifdef USE_TWO_FIXED_MAX
    std::cout << "VECTOR_FRAGMENT1: " << (float)VECTOR_FRAGMENT1 << "\n";
    std::cout << "VECTOR_FRAGMENT2: " << (float)VECTOR_FRAGMENT2 << "\n";
#else
    std::cout << "DISTANCE_TRESHOLD: " << (int)DISTANCE_TRESHOLD << "\n";
#endif
#endif
}


