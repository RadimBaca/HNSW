#include "HNSW.h"

void HNSW::create(const char* filename, const char* datasetname)
{

    hsize_t dimensions[2];

    node_count = 0;
    hdfReader::getDimensions(filename, datasetname, &dimensions);
    std::clog << "dimensions " <<
        (unsigned long)(dimensions[0]) << " x " <<
        (unsigned long)(dimensions[1]) << std::endl;
    vector_count = dimensions[0];
    setVectorSize(dimensions[1]);
    float* data = new float[vector_count * vector_size];
#ifdef COMPUTE_APPROXIMATE_VECTOR
    min_vector = new float[vector_size];
    max_vector = new float[vector_size];
    apr_q = new uint8_t[vector_size];
    one_vector_mask_size = (vector_size >> 5) == 0 ? 1 : vector_size >> 5;
    q_delta = new int[vector_size];
#endif

    hdfReader::readData(filename, datasetname, data);

    for (int i = 0; i < vector_count; i++)
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
            if (W[c1]->node_order == data_result[i * dimensions_result[1] + c2])
            {
                positive++;
                c1++;
            }
            c2++;
        }
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
        std::cout << (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 10000 << " [ms]; ";
    }
    std::cout << "avg: " << sum / 3 << " [ms]; " << "min: " << min_time << " [ms]; \n";
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
    std::vector<Node*> R;

#ifdef COLLECT_STAT
    node_count++;
#endif

#ifdef VISIT_HASH
    visited.clear();
#endif

    visit_id++;
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
        ep->distance = distance(ep->vector, q);

#ifdef VISIT_HASH
        visited.insert(ep->uniqueId);
#else
        ep->visit_id = visit_id;
#endif
        W.push_back(ep);
    }
    for (int32_t i = L; i > l; i--)
    {
        search_layer_one(q);
        change_layer();
    }
    for (int32_t i = std::min(L, l); i >= 0; i--)
    {
        Node* node = new Node(visit_id, vector_size, Mmax, nullptr, q);
        layers[i]->node_count++;
        layers[i]->nodes.push_back(node);

        auto actualMmax = (i == 0 ? Mmax0 : Mmax);
#ifdef DEBUG_NET
        node->layer = i;
#endif

        if (l > L && i == L)
        {
            down_node = node; // remember the top node, if we are going to add layers
        }

        // the main logic
        search_layer(q, efConstruction);
        select_neighbors(R, M, false, true);

        for (int i = 0;i < R.size(); i++)
        {
            auto e = R[i];
            node->neighbors.emplace_back(e, e->distance, e->uniqueId);

            bool insert_new = true;
            if (e->neighbors.size() == actualMmax)
            {
                if (!e->neighbors_sorted)
                {
                    std::sort(e->neighbors.begin(), e->neighbors.end(), neighborcmp_nearest());
                    e->neighbors_sorted = true;
                }
#ifdef SELECT_NEIGHBORS1
                if (e->neighbors[actualMmax - 1].distance > e->distance)
                {
                    e->neighbors.pop_back();
                    e->neighbors.emplace_back(node, e->distance, e->uniqueId);
                    std::sort(e->neighbors.begin(), e->neighbors.end(), neighborcmp_nearest()); // TODO - insert instead of sort
                }
#else
                if (is_distant_node(e->neighbors, node->vector, e->distance))
                {
                    e->neighbors.pop_back();
                    e->neighbors.emplace_back(node, e->distance, node->uniqueId);
                    std::sort(e->neighbors.begin(), e->neighbors.end(), neighborcmp_nearest()); // TODO - insert instead of sort
                }
#endif
            }
            else {
                //if (is_distant_node(e->neighbors, node->vector, e->distance))
                {
                    e->neighbors.emplace_back(node, e->distance, node->uniqueId);
                }
            }            
        }

        // connecting the nodes in different layers
        if (prev != nullptr)
        {
            prev->lower_layer = node;
        }
        prev = node;

        // switch the actual layer (W nodes are replaced by their bottom counterparts)
        if (i > 0) change_layer();
    }
    if (l > L)
    {
        // adding layers
        for (int32_t i = L + 1; i <= l; i++)
        {
            Node* node = new Node(visit_id, vector_size, Mmax, down_node, q); // TODO resolve memory leak
            layers.emplace_back(std::make_unique<Layer>(node));
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
    visit_id++;

#ifdef VISIT_HASH
    visited.clear();
#endif
    W.clear();
    ep = layers[L]->enter_point;
#ifdef COMPUTE_APPROXIMATE_VECTOR
    Node::computeApproximateVector(apr_q, q, shift, vector_size);
    ep->apr_distance = apr_distance(ep->apr_vector, apr_q);
#else
    ep->distance = distance(ep->vector, q);
#endif
#ifdef VISIT_HASH
    visited.insert(ep->uniqueId);
#else
    ep->visit_id = visit_id;
#endif
    W.push_back(ep);

    for (int32_t i = L; i > 0; i--)
    {
#ifdef COMPUTE_APPROXIMATE_VECTOR
        apr_search_layer_one(apr_q);
#else
        search_layer_one(q);
#endif
        change_layer();
    }
#ifdef COMPUTE_APPROXIMATE_VECTOR
    apr_search_layer_summary(apr_q, ef);
    // TODO - throw runtime error if W.size() is less then ef
    for (int i = 0; i < ef; i++)
    {
        auto item = distances.find(W[i]->uniqueId);
        if (item == distances.end())
        {
            W[i]->apr_distance = apr_distance(apr_q, W[i]->apr_vector);
        }
        else
        {
            W[i]->apr_distance = item->second;
        }
    }
    sort(W.begin(), W.end(), apr_nodecmp_farest());
    //sort(W.begin(), W.end(), nodecmp_farest());
#else
    search_layer_one(q);
    change_layer();
    search_layer(q, ef);
    sort(W.begin(), W.end(), nodecmp_farest());
#endif
}

#ifdef COMPUTE_APPROXIMATE_VECTOR
void HNSW::computeApproximateVector()
{
    shift = log(max_value) / log(2) - 8 < 0 ? 0 : log(max_value) / log(2) - 8 < 0;
    std::cout << "Compute approximate vectors (shift value " << shift << ")\n";
    auto start = std::chrono::system_clock::now();
    index_memory = 0;
    for (auto& l : layers)
    {
        for (auto n : l->nodes)
        {
            n->computeApproximateVector(shift, vector_size);
        }
        // mask version
        if (l->layer_id == 0)
        {
            for (auto n : l->nodes)
            {
                //n->computeMasks2(shift, vector_size);
                index_memory +=n->computeSummaries(n->apr_vector, vector_size, &stat.overflows);
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
#endif

void HNSW::search_layer_one(float* q)
{
    bool change = true;
    Node* actual;
    actual = W.front();
    W.pop_back();
    while (change)
    {
        change = false;
        for (auto ne : actual->neighbors)
        {
            auto e = ne.node;
#ifdef VISIT_HASH
            if (!visited.get(ne.uniqueId))
            {
                visited.insert(ne.uniqueId);
#else
            if (e->visit_id < visit_id)
            {
                e->visit_id = visit_id;
#endif
                e->distance = distance(e->vector, q);

                if (e->distance < actual->distance)
                {
                    change = true;
                    actual = e;
                }
            }
        }
    }
    W.clear();
    W.push_back(actual);
}

void HNSW::search_layer(float* q, int ef)
{
    std::vector<Node*> C;

    
    //W.push_back(ep);
    for (auto n : W)
    {
        C.push_back(n);
    }
    std::make_heap(W.begin(), W.end(), nodecmp_farest());
    std::make_heap(C.begin(), C.end(), nodecmp_nearest());
    while (!C.empty())
    {
        auto c = C.front();
        auto f = W.front();
        std::pop_heap(C.begin(), C.end(), nodecmp_nearest());
        C.pop_back();

        if (c->distance > f->distance) break;
        for (auto ne : c->neighbors)
        {
            auto e = ne.node;
#ifdef VISIT_HASH
            if (!visited.get(ne.uniqueId))
            { 
                visited.insert(ne.uniqueId);
#else
            if (e->visit_id < visit_id)
            {
                e->visit_id = visit_id;
#endif
                auto f = W.front();
                //e->distance = distance_treshold_counting(q, e->vector, e->no_over_treshold);
                e->distance = distance(q, e->vector);

                if (e->distance < f->distance || W.size() < ef)
                {
                    C.push_back(e);
                    std::push_heap(C.begin(), C.end(), nodecmp_nearest());
                    W.push_back(e);
                    std::push_heap(W.begin(), W.end(), nodecmp_farest());
                    if (W.size() > ef)
                    {
                        std::pop_heap(W.begin(), W.end(), nodecmp_farest());
                        W.pop_back();
                    }
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
    Node *a1;
    a1 = W.front();
    W.pop_back();
    while (change)
    {
        change = false;
        for (auto ne : a1->neighbors)
        {
            auto e = ne.node;
#ifdef VISIT_HASH
            if (!visited.get(ne.uniqueId))
            {
                visited.insert(ne.uniqueId);
#else
            if (e->visit_id < visit_id)
            {
                e->visit_id = visit_id;
#endif
                e->apr_distance = apr_distance(e->apr_vector, q);

                if (e->apr_distance < a1->apr_distance)
                {
                    change = true;
                    a1 = e;
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
    W.clear();
    W.push_back(a1);
}


void HNSW::apr_search_layer(uint8_t* apr_q, int ef)
{
    std::vector<Node*> C;


    //W.push_back(ep);
    for (auto n : W)
    {
        C.push_back(n);
    }
    std::make_heap(W.begin(), W.end(), apr_nodecmp_farest());
    std::make_heap(C.begin(), C.end(), apr_nodecmp_nearest());
    while (!C.empty())
    {
#ifdef APR_DEBUG
        if (C.size() > 1 && C[0]->apr_distance > C[1]->apr_distance)
        {
            std::cout << "HNSW::apr_search_layer - C not ordered!!\n";
        }
#endif
        auto c = C.front();
        auto f = W.front();
        std::pop_heap(C.begin(), C.end(), apr_nodecmp_nearest());
        C.pop_back();

        if (c->apr_distance > f->apr_distance) break;
        for (auto ne : c->neighbors)
        {
            auto e = ne.node;
#ifdef VISIT_HASH
            if (!visited.get(e->uniqueId))
            {
                visited.insert(e->uniqueId);
#else
            if (e->visit_id < visit_id)
            {
                e->visit_id = visit_id;
#endif
                auto f = W.front();
                e->apr_distance = apr_distance(e->apr_vector, apr_q);
               
                if (e->apr_distance < f->apr_distance || W.size() < ef)
                {
                    C.push_back(e);
                    std::push_heap(C.begin(), C.end(), apr_nodecmp_nearest());
                    W.push_back(e);
                    std::push_heap(W.begin(), W.end(), apr_nodecmp_farest());
                    if (W.size() > ef)
                    {
                        std::pop_heap(W.begin(), W.end(), apr_nodecmp_farest());
                        W.pop_back();
                    }
                }
            }
        }
    }
}


void HNSW::apr_search_layer_summary(uint8_t* apr_q, int ef)
{
    std::priority_queue<std::pair<int32_t, Node*>, std::vector<std::pair<int32_t, Node*>>, CompareByFirst> C;

    distances.clear();
    //W.push_back(ep);
    for (auto n : W)
    {
        C.emplace(-n->apr_distance, n);
        WP.emplace(n->apr_distance, n);
    }
    W.clear();
    auto f = WP.top().first;

    while (!C.empty())
    {
#ifdef APR_DEBUG
        if (C.size() > 1 && C[0]->apr_distance > C[1]->apr_distance)
        {
            std::cout << "HNSW::apr_search_layer - C not ordered!!\n";
        }
#endif
        auto c = C.top();
        C.pop();
        Node* nc = c.second;

        if (-c.first > f) break;
#ifdef COLLECT_STAT
        stat.explored_nodes++;
#endif
        uint32_t c_distance = apr_distance(apr_q, nc->apr_vector, q_delta);
        distances[nc->uniqueId] = c_distance;
        int node_summary_position = 0;
        for (auto ne : nc->neighbors)
        {
            auto e = ne.node;
#ifdef VISIT_HASH
            if (!visited.get(ne.uniqueId))
            {
                visited.insert(ne.uniqueId);
#else
            if (e->visit_id < visit_id)
            {
                e->visit_id = visit_id;
#endif

                int32_t distance = apr_distance_summary(nc->summary, node_summary_position, q_delta, c_distance);

                if (distance < f || WP.size() < ef)
                {
                    C.emplace(-distance, e);
                    WP.emplace(distance, e);
                    if (WP.size() > ef)
                    {
                        WP.pop();
                    }
                    f = WP.top().first;
                }
            }
            else
            {
                node_summary_position = nc->summary[node_summary_position] * 2 + 1 + node_summary_position;
            }
        }
    }

    while (!WP.empty())
    {
        W.push_back(WP.top().second);
        WP.pop();
    }
}
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
void HNSW::select_neighbors(std::vector<Node*>& R, int M, bool keepPruned, bool considerOverTreshold)
{
    //std::vector<Node*> Wd;
    //Wd.reserve(M);


    //if (W.size() < M) -- TODO

    R.clear();
    std::sort(W.begin(), W.end(), nodecmp_farest());  // TODO - use heap instead of sort
    R.push_back(W[0]);
    int s = 1;
    int i = 1;
    while(i < W.size() && s < M)
    {
        bool q_is_close = true;
        for (int j = 0; j < R.size(); j++)
        {
            float dist = distance(W[i]->vector, R[j]->vector);
            if (dist < W[i]->distance)
            {
                q_is_close = false;
                break;
            }
        }
        if (q_is_close)
        {
            R.push_back(W[i]);
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



    ///////////////////////////////////////////////

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
        if (all)
        {
            for (auto n : layers[i]->nodes)
            {
                n->print(vector_size);
            }
        }
    }
    std::cout << "Index memory: " << index_memory / (1024 * 1024) << " [MB]\n";
    std::cout << "M: " << M << "\n";
    std::cout << "Mmax: " << Mmax << "\n";
    std::cout << "ml: " << ml << "\n";
    std::cout << "distance_treshold: " << (int)DISTANCE_TRESHOLD << "\n";
}


