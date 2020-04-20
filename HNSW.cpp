#include "HNSW.h"

void HNSW::create(const char* filename, const char* datasetname)
{

    hsize_t dimensions[2];

    getDimensions(filename, datasetname, &dimensions);
    std::clog << "dimensions " <<
        (unsigned long)(dimensions[0]) << " x " <<
        (unsigned long)(dimensions[1]) << std::endl;
    vector_count = dimensions[0];
    vector_size = dimensions[1];
    float* data = new float[vector_count * vector_size];

    readData(filename, datasetname, data);

    for (int i = 0; i < vector_count; i++)
    {
        insert(&data[i * vector_size]);
    }
    delete[] data;
}


void HNSW::query(const char* filename, const char* querydatasetname, const char* resultdatasetname)
{
    hsize_t dimensions_query[2];
    hsize_t dimensions_result[2];

    getDimensions(filename, querydatasetname, &dimensions_query);
    float* data_query = new float[dimensions_query[0] * dimensions_query[1]];
    readData(filename, querydatasetname, data_query);

    getDimensions(filename, resultdatasetname, &dimensions_result);
    int* data_result = new int[dimensions_result[0] * dimensions_result[1]];
    readData(filename, resultdatasetname, data_result);

    uint32_t k = 10;
    double positive = 0;
    for (int i = 0; i < dimensions_query[0]; i++)
    {
        knn(&data_query[i * dimensions_query[1]], k, 1000);

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


    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < dimensions_query[0]; i++)
    {
        knn(&data_query[i * dimensions_query[1]], k, 1000);
    }
    auto end = std::chrono::system_clock::now();
    std::cout << (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 10000 << " [ms] \n";
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
    visit_id++;

    if ((visit_id % 10000) == 0)
    {
        std::cout << visit_id << "\n";
    }

    W.clear();
    if (L >= 0)
    {
        ep = layers[L]->enter_point;
        ep->distance = distance(ep->vector, q);
        ep->visit_id = visit_id;
        W.push_back(ep);
    }
    for (int32_t i = L; i > l; i--)
    {
        search_layer(q, 2);
        change_layer();
    }
    for (int32_t i = std::min(L, l); i >= 0; i--)
    {
        layers[i]->node_count++;
        Node* node = new Node(visit_id, vector_size, Mmax, nullptr, q);
#ifdef DEBUG_NET
        node->layer = i;
#endif

        if (l > L && i == L)
        {
            down_node = node; // remember the top node, if we are going to add layers
        }

        // the main logic
        search_layer(q, efConstruction);
        std::sort(W.begin(), W.end(), nodecmp_farest()); // TODO, replace by SELECT-NEIGHBORS
        for (int i = 0; i < M && i < W.size(); i++)
        {
            node->neighbors.emplace_back(W[i], W[i]->distance);
            W[i]->neighbors.emplace_back(node, W[i]->distance);
        }
        for (auto ne : node->neighbors)
        {
            // IDEA - the shrink could be initiated after some delta
            auto e = ne.node;
            if (i > 0 && e->neighbors.size() > Mmax)
            {
                // remove the most distant node
                std::sort(e->neighbors.begin(), e->neighbors.end(), neighborcmp_nearest()); // TODO - do not perform sorting if already sorted
                e->neighbors.pop_back();
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
#ifdef DEBUG_NET
            node->layer = i;
#endif
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

    if ((visit_id % 10000) == 0)
    {
        std::cout << visit_id << "\n";
    }

    W.clear();
    ep = layers[L]->enter_point;
    ep->distance = distance(ep->vector, q);
    ep->visit_id = visit_id;
    W.push_back(ep);

    for (int32_t i = L; i > 0; i--)
    {
        search_layer(q, 2);
        change_layer();
    }
    search_layer(q, ef);
    sort(W.begin(), W.end(), nodecmp_farest());
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
        std::pop_heap(C.begin(), C.end(), nodecmp_farest());
        C.pop_back();

        if (c->distance > f->distance) break;
        for (auto ne : c->neighbors)
        {
            auto e = ne.node;
            if (e->visit_id < visit_id)
            {
                e->visit_id = visit_id;
                auto f = W.front();
                e->distance = distance(e->vector, q);
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
            }
        }
    }
}


void HNSW::printInfo()
{
    std::cout << "HNSW info\n";
    for (int i = 0; i < layers.size(); i++)
    {
        std::cout << "Layer " << i << " " << layers[i]->node_count << "\n";
    }
}


//////////////////////////////////////////////////////////////
///////////////////////   HDF5 fun    ////////////////////////
//////////////////////////////////////////////////////////////


void HNSW::getDimensions(const char* filename, const char* datasetname, hsize_t(*dimensions)[2])
{
    try
    {
        H5File file(filename, H5F_ACC_RDONLY);
        DataSet dataset = file.openDataSet(datasetname);
        DataSpace dataspace = dataset.getSpace(); // Get dataspace of the dataset.
        int rank = dataspace.getSimpleExtentNdims(); // Get the number of dimensions in the dataspace.
        int ndims = dataspace.getSimpleExtentDims(*dimensions, NULL);
    }  // end of try block
    catch (FileIException error) { error.printErrorStack();        return; }
    catch (DataSetIException error) { error.printErrorStack();        return; }
    catch (DataSpaceIException error) { error.printErrorStack();        return; }
    catch (DataTypeIException error) { error.printErrorStack();        return; }
}


void HNSW::readData(const char* filename, const char* datasetname, float* data)
{
    hid_t           file, dset;           /* Handle */
    herr_t          status;
    /*
    * Open file and initialize the operator data structure.
    */
    file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    dset = H5Dopen(file, datasetname, H5P_DEFAULT);
    status = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    /*
    * Close and release resources.
    */
    status = H5Dclose(dset);
    status = H5Fclose(file);
}


void HNSW::readData(const char* filename, const char* datasetname, int* data)
{
    hid_t           file, dset;           /* Handle */
    herr_t          status;
    /*
    * Open file and initialize the operator data structure.
    */
    file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    dset = H5Dopen(file, datasetname, H5P_DEFAULT);
    status = H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    /*
    * Close and release resources.
    */
    status = H5Dclose(dset);
    status = H5Fclose(file);
}