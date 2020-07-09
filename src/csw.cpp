//
// Created by Radim Baca on 01.07.20.
//
#include "csw.h"

Csw::Csw() : index_loaded_(false)
{
}

Csw::~Csw()
{
    if (index_loaded_) {
        delete[] group_assignment_;
        delete[] node_group_;
        delete hnsw;
        index_loaded_ = false;
    }
}


void Csw::load_index(const char* file_name, const int N, const int X)
{
#ifndef COMPUTE_APPROXIMATE_VECTOR
    throw std::runtime_error("Csw::load_index This implementation is designed for uint8_t approximation of the float data");
    return;
#else

    hnsw = new HNSW(16, 16, 200);

    hnsw->loadGraph(file_name);
    hnsw->computeApproximateVector();
    index_loaded_ = true;
    group_assignment_ = new int[kMaxGroupsPerNode * (hnsw->actual_node_count_ + 1)];
    memset(group_assignment_, 0, kMaxGroupsPerNode * sizeof(hnsw->actual_node_count_) * (hnsw->actual_node_count_ + 1));

    sortingNodesIntoGroups(N, X);
    groupMaterialization();
#endif

}

void Csw::sortingNodesIntoGroups(const int N, const int X)
{
#ifdef COMPUTE_APPROXIMATE_VECTOR
    pointer_t node_order = 0;
    int expected_unassigned_count = 3 * N / 4;
    // creation of basic groups
    std::cout << "finding the basic groups\n";
    for (auto& n : hnsw->layers_[0]->nodes)
    {
        if (group_assignment_[node_order * kMaxGroupsPerNode] == 0) {
            hnsw->searchLayer(hnsw->getNodeVector(node_order), N, n, node_order);
            //hnsw->knn(hnsw->getNodeVector(node_order), 200);
            // TODO rewrite to aproximate (using uint8_t)
            int unassigned = 0, counter = 0;
            for (auto &nr : hnsw->W_) {
                if (group_assignment_[nr.node_order * kMaxGroupsPerNode] == 0) {
                    unassigned++;
                }
                if (++counter >= N) break;
            }
            counter = 0;
            if (unassigned >= expected_unassigned_count) {
                groups_.emplace_back();
                auto &group = groups_.back();
                for (auto &nr : hnsw->W_) {
                    if (assignIntoGroup(group_assignment_, group, nr.node_order)) {
                        // if the number of groups per node node is not exceeded
                        if (++counter >= N) break;
                    }
                }
            }
        }
        node_order++;
    }

    // expanding part
    std::cout << "expanding the groups\n";
    std::unordered_map<pointer_t, int> aspiring_nodes;
    for (auto& group: groups_)    {
        while (group.nodes_.size() < X) {
            aspiring_nodes.clear();
            std::sort(group.nodes_.begin(), group.nodes_.end());
            // counting how many times is a node referenced from the group
            for (auto &n : group.nodes_) {
                for (auto &nb : hnsw->layers_[0]->nodes[n]->neighbors) {
                    // check whether the nb.node_order is in the current_nodes set
                    auto lower = std::lower_bound(group.nodes_.begin(), group.nodes_.end(), nb.node_order);
                    if (lower == group.nodes_.end() || *lower != nb.node_order) {
                        auto asp = aspiring_nodes.find(nb.node_order);
                        if (asp == aspiring_nodes.end()) {
                            aspiring_nodes[nb.node_order] = 0;
                        } else {
                            aspiring_nodes[nb.node_order]++;
                        }
                    }
                }
            }
            std::vector<std::pair<int, pointer_t >> aspiring_nodes_sorting;
            for (auto &n : aspiring_nodes) aspiring_nodes_sorting.emplace_back(n.second, n.first);
            std::sort(aspiring_nodes_sorting.begin(), aspiring_nodes_sorting.end(), [](auto &i1, auto &i2) {
                return i1.first > i2.first;
            });
            int adding_count = group.nodes_.size() > X - 10 ? X - group.nodes_.size() : 10;
            bool at_least_one_added = false;
            for (auto &n : aspiring_nodes_sorting) {
                if (n.first == 0 || adding_count-- == 0) break;
                if (assignIntoGroup(group_assignment_, group, n.second)) {
                    at_least_one_added = true;
                }
            }
            if (!at_least_one_added) break;
        }
    }


    // assigning the remainders
    std::cout << "assigning the remainders\n";
    node_order = 0;
    int *closest_node_groups = new int[groups_.size()];
    for (auto& n : hnsw->layers_[0]->nodes) {
        memset(closest_node_groups, 0, sizeof(int) * groups_.size());
        if (group_assignment_[node_order * kMaxGroupsPerNode] == 0) {
#ifdef COLLECT_STAT
            statistics_.no_unassigned_nodes_++;
#endif
            //hnsw->searchLayer(hnsw->getNodeVector(node_order), 100, n, node_order);
            // count the groups of finded nodes
            //for (auto &nr : hnsw->W_) {
            for (auto& nr: n->neighbors) {
                for (int i = 1; i <= group_assignment_[nr.node_order * kMaxGroupsPerNode]; i++) {
                    closest_node_groups[group_assignment_[nr.node_order * kMaxGroupsPerNode + i]]++;
                }
            }
            while (true) {
                // find the most frequent group
                int max_frequent = 0;
                int max_group = -1;
                for (int i = 0; i < groups_.size(); i++) {
                    if (closest_node_groups[i] > max_frequent && groups_[i].nodes_.size() < kMaxNodesPerGroup) {
                        max_frequent = closest_node_groups[i];
                        max_group = i;
                    }
                }
                if (max_group == -1) throw std::runtime_error("CSW::load_index - all groups are full!\n");
                // try to assign the node n into the most frequent group
                if (assignIntoGroup(group_assignment_, groups_[max_group], node_order))
                {
                    break;
                }
                closest_node_groups[max_group] = 0;
            }
        }
        node_order++;
    }
    // final sorting of node ids in groups
    for (auto& group: groups_) std::sort(group.nodes_.begin(), group.nodes_.end());

    delete[] closest_node_groups;
#endif
}

void Csw::groupMaterialization() {
#ifdef COMPUTE_APPROXIMATE_VECTOR
    int *aux_group_pivot = new int[hnsw->vector_size_];
    uint8_t *group_pivot = new uint8_t[hnsw->vector_size_];
    node_group_ = new int[hnsw->actual_node_count_ + 1];
    std::vector<uint8_t> local_neigh;
    std::vector<int> global_neigh;

    uint32_t *group_assignment_distance_order = new uint32_t[kMaxGroupsPerNode * (hnsw->actual_node_count_ + 1)];
    memset(group_assignment_distance_order, 0, kMaxGroupsPerNode * sizeof(uint32_t) * (hnsw->actual_node_count_ + 1));

    //center selection and distance counting
    std::cout << "center selection and distance counting\n";
    for (auto &group : groups_) {
        // compute pivot
        memset(aux_group_pivot, 0, hnsw->vector_size_);
        for (auto &n : group.nodes_) {
            for (int i = 0; i < hnsw->vector_size_; i++) {
                aux_group_pivot[i] += hnsw->aprGetNodeVector(n)[i];
            }
        }
        for (int i = 0; i < hnsw->vector_size_; i++) {
            group_pivot[i] /= group.nodes_.size();
        }

        uint32_t min_distance = -1;
        group.node_center_order_ = -1;
        int node_index = 0;
        for (auto &n : group.nodes_) {
            auto dist = hnsw->aprDistance(group_pivot, hnsw->aprGetNodeVector(n));
            if (dist < min_distance) {
                group.node_center_order_ = node_index;
                min_distance = dist;
            }
            node_index++;
        }
        // TODO remove group.node_center_order_ from nodes_
        // start compute distances from center and compute summaries
        group.guidepost_ = new GuideGroup[group.nodes_.size()];
        auto vector = hnsw->aprGetNodeVector(group.node_center_order_);
        for (int i = 0; i < hnsw->vector_size_; i++) {
            group.summary_.push_back(vector[i]); // TODO insert into vector and later copy into array (at the end)
        }
        group.guidepost_[group.node_center_order_ ].summary = 0;
        int node_group_index = 0;
        for (auto &n_order : group.nodes_) {
            uint32_t dist = 0;
            if (node_group_index != group.node_center_order_) {
                dist = hnsw->aprDistance(hnsw->aprGetNodeVector(group.node_center_order_),
                                         hnsw->aprGetNodeVector(n_order));
                group.guidepost_[node_group_index].summary = group.summary_.size();
            }
            for (int i = 1; i <= group_assignment_[n_order * kMaxGroupsPerNode]; i++) {
                if (group_assignment_[n_order * kMaxGroupsPerNode + i] == group.group_id_) {
                    group_assignment_distance_order[n_order * kMaxGroupsPerNode + i] = dist;
                    break;
                }
            }
            vector = hnsw->aprGetNodeVector(n_order);
            for (int i = 0; i < hnsw->vector_size_; i++) {
                group.summary_.push_back(vector[i]); // TODO insert into vector and later copy into array
            }
            node_group_index++;
        }
    }
    // for each node find a group where the node is closest to its pivot
    std::cout << "search for groups where the node is closest to pivot\n";
    int node_order = 0;
    for (auto &n : hnsw->layers_[0]->nodes) {
        int min_group_index = group_assignment_[node_order * kMaxGroupsPerNode + 1];
        uint32_t min_dist = group_assignment_distance_order[node_order * kMaxGroupsPerNode + 1];
        for (int i = 2; i <= group_assignment_[node_order * kMaxGroupsPerNode]; i++) {
            if (group_assignment_distance_order[node_order * kMaxGroupsPerNode + i] < min_dist) {
                min_group_index = group_assignment_[node_order * kMaxGroupsPerNode + i];
                min_dist = group_assignment_distance_order[node_order * kMaxGroupsPerNode + i];
            }
        }
        node_group_[node_order] = min_group_index;
        node_order++;
    }
    // resolve group references
    std::cout << "resolving the group references\n";
    for (auto &group : groups_) {
        local_neigh.clear();
        global_neigh.clear();
        int node_group_index = 0;
        for (auto &n : group.nodes_) {
            group.guidepost_[node_group_index].local = local_neigh.size();
            group.guidepost_[node_group_index].global = global_neigh.size();
            auto &node = hnsw->layers_[0]->nodes[n];
            for (auto &neig : node->neighbors) {
                // for each neig decide whether it is a localgroup node or not (i.e. whether it is in the group)
                auto localgroup = std::lower_bound(group.nodes_.begin(), group.nodes_.end(), neig.node_order);
                if (localgroup != group.nodes_.end() && *localgroup == neig.node_order) {
                    int position = localgroup - group.nodes_.begin();
                    local_neigh.push_back(localgroup - group.nodes_.begin());
                } else {
                    int remote_group_id = node_group_[neig.node_order];
                    auto remote_group = std::lower_bound(groups_[remote_group_id].nodes_.begin(),
                                                         groups_[remote_group_id].nodes_.end(), neig.node_order);
                    if (remote_group == groups_[remote_group_id].nodes_.end() || *remote_group != neig.node_order)
                        throw std::runtime_error(
                                "Csw::groupMaterialization - the node id was not founded in the group!");
                    int remote_local_node_order = remote_group - groups_[remote_group_id].nodes_.begin();
                    global_neigh.push_back(mergeGlobalId(remote_group_id, remote_local_node_order));
                }
            }
            group.guidepost_[node_group_index].local_count =
                    local_neigh.size() - group.guidepost_[node_group_index].local;
            group.guidepost_[node_group_index].global_count =
                    global_neigh.size() - group.guidepost_[node_group_index].global;
            node_group_index++;
        }
        // we use auxiliary vectors, therefore, we need to copy them into group
        group.local_neighbors_ = new uint8_t[local_neigh.size()];
        std::copy(local_neigh.begin(), local_neigh.end(), group.local_neighbors_);
        group.global_neighbors_ = new int[global_neigh.size()];
        std::copy(global_neigh.begin(), global_neigh.end(), group.global_neighbors_);
#ifdef COLLECT_STAT
        statistics_.no_local_references_ += local_neigh.size();
        statistics_.no_global_references_ += global_neigh.size();
#endif
    }

    delete[] group_assignment_distance_order;
    delete[] group_pivot;
    delete[] aux_group_pivot;
#endif
}

bool Csw::assignIntoGroup(int *group_assignment, NodeGroup &group, const int node_order) {
    if (group_assignment[node_order * kMaxGroupsPerNode] < kMaxGroupsPerNode - 1) {
        group.nodes_.push_back(node_order);
        group_assignment[node_order * kMaxGroupsPerNode]++;
        group_assignment[node_order * kMaxGroupsPerNode +
                         group_assignment[node_order * kMaxGroupsPerNode]] = group.group_id_;
        return true;
    }
    return false;
}

#ifdef COLLECT_STAT
void Csw::printInfo()
{
    std::cout << "\nClustered small world statistics\n";
    std::cout << "Group count: " << groups_.size() << "\n";
    std::cout << "Unassigned nodes count after group expand " << statistics_.no_unassigned_nodes_ << "\n";

    double group_count = 0;
    int max_group_count = 0;
    int max_group_count_count = 0;
    for (int i = 0; i < hnsw->actual_node_count_; i++) {
        //std::cout << i << ": " << group_assignment_[i * kMaxGroupsPerNode] << "\n";
        group_count += group_assignment_[i * kMaxGroupsPerNode];
        if (group_assignment_[i * kMaxGroupsPerNode] == max_group_count) {
            max_group_count_count++;
        }
        if (group_assignment_[i * kMaxGroupsPerNode] > max_group_count) {
            max_group_count = group_assignment_[i * kMaxGroupsPerNode];
            max_group_count_count = 1;
        }
    }
    std::cout << "Average number of groups per node " << group_count / hnsw->actual_node_count_ << "\n";
    std::cout << "Maximum number of groups per node " << max_group_count << "\n";
    std::cout << "Number of nodes having the maximum number of groups per node " << max_group_count_count << "\n";

    int max_group_size = 0;
    for (auto& group : groups_) {
        if (group.nodes_.size() > max_group_size) {
            max_group_size = group.nodes_.size();
        }
    }
    std::cout << "Maximum number of nodes in a group " << max_group_size << "\n";
    std::cout << "Number of local references " << statistics_.no_local_references_ << " (" << 100 * (long long)statistics_.no_local_references_ / (statistics_.no_local_references_ + statistics_.no_global_references_)  << "%)\n";
    std::cout << "Number of global references " << statistics_.no_global_references_ << " (" << 100 * (long long)statistics_.no_global_references_ / (statistics_.no_local_references_ + statistics_.no_global_references_) << "%)\n";
}
#else
void Csw::printInfo() { }
#endif