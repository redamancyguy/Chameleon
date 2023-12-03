//
// Created by redamancyguy on 23-5-30.
//

#ifndef INDEX_H
#define INDEX_H

#include <thread>
#include <stack>
#include "Configuration.hpp"
#include "RL_network.hpp"

#define DATA_NODE_SIZE 5
const float min_density = 0.35;
const float default_density = 0.65;
const float max_density = 0.95;


//const float min_density = 0.5;
//const float default_density = 0.7;
//const float max_density = 0.9;

//float calculate_density(float size, float probability = 0.9){
//    return size * (size - 1) / -(std::log(1-probability));
//}
double hash_factor = 10231023.353;
//double  hash_factor = 131.3;
//#define inner_cost_weight 3.85329847
//#define leaf_cost_weight 1.83443234
#define inner_cost_weight (float(6.01213))
#define leaf_cost_weight (float(1.35074178))

inline double forward_with_parameters(double lower, double upper, double capacity, double x) {
    return capacity * ((x - lower) / (upper - lower));
}

inline bool get_bitmap(const unsigned int *bitmap, const unsigned int position) {
    unsigned int index = position >> 5;
    unsigned int bit_index = position - (index << 5);
    return (bitmap[index] >> bit_index) & 1;
}

inline void de_set_bitmap(unsigned int *bitmap, const unsigned int position) {
    unsigned int index = position >> 5;
    unsigned int bit_index = position - (index << 5);
    bitmap[index] &= ~(1 << bit_index);
}

inline void set_bitmap(unsigned int *bitmap, const unsigned int position) {
    unsigned int index = position >> 5;
    unsigned int bit_index = position - (index << 5);
    bitmap[index] |= 1 << bit_index;
}

namespace CHA {
    unsigned long long node_memory_count = 0;
    long long inner_cost = 0;
    long long leaf_cost = 0;
    long long leaf_max_cost = 0;

    template<class key_T, class value_T>
    class DataNode {
    public:
        using self_type = DataNode;
        using slot_type = std::pair<key_T, value_T>;
    public:
        double lower{0};
        double upper{0};
        int capacity{-1};

        [[nodiscard]] inline double forward(double key) const {
            return std::min(double(capacity - 1), std::max(0.0, forward_with_parameters(lower, upper, capacity, key)));
        }

        ///////
        int size{};
        int max_offset{};
        slot_type array[0];


        inline unsigned int *bitmap_start() {
            return (unsigned int *) (array + capacity);
        }

        inline
        static self_type *
        new_segment(int capacity, double lower, double upper) {
            int t = ((capacity >> 5) + 1);
            auto memory_size = sizeof(self_type) + sizeof(slot_type) * capacity + sizeof(unsigned int) * t;
            auto node = (self_type *) std::malloc(memory_size);
            node->capacity = capacity;
            node->lower = lower;
            node->upper = upper;
            node->size = 0;
            node->max_offset = 0;
            unsigned int *bit_map = node->bitmap_start();
            for (int i = 0; i < t; ++i) {
                bit_map[i] = 0;
            }
            for(int i = 0;i<capacity;++i){
                node->array[i].first = 0;
            }
            node_memory_count += memory_size;
            return node;
        }

        inline
        static void
        delete_segment(self_type *node) {
            node_memory_count -= sizeof(self_type) + sizeof(slot_type) * node->capacity +
                                 sizeof(unsigned int) * ((node->capacity >> 5) + 1);
            std::free(node);
        }

        inline int hash(double key) {
            auto predict_hash = (long long) (forward(key) * hash_factor);
            return int(predict_hash % (long long) capacity);
        }

        int find(double key) {
            auto position = hash(key);
            auto left = position;
            auto right = position;
            for (int i = 0; i <= max_offset ; ++i,--left,++right) {
                if (left < 0) {
                    left += capacity;
                }
                if (right >= capacity) {
                    right -= capacity;
                }
                if (get_bitmap(bitmap_start(), left) && array[left].first == key) {
                    return left;
                }
                if (get_bitmap(bitmap_start(), right) && array[right].first == key) {
                    return right;
                }
            }
            return -1;
        }

        int find_with_cost(double key) {
            auto position = hash(key);
            auto left = position;
            auto right = position;
            for (int i = 0; i <= max_offset; ++i,--left,++right) {
                if (left < 0) {
                    left += capacity;
                }
                if (right >= capacity) {
                    right -= capacity;
                }
                if (get_bitmap(bitmap_start(), left) && array[left].first == key) {
                    return left;
                }
                if (get_bitmap(bitmap_start(), right) && array[right].first == key) {
                    return right;
                }
                ++leaf_cost;
                leaf_max_cost = std::max<long long>(leaf_max_cost,i);
            }
            return -1;
        }

        int find_insert(double key) {
            int position = hash(key);
            int length = (capacity >> 1) + 1;
            int i = 0;
            int left = position;
            int right = position;
            for (; i <= max_offset; ++i,--left,++right) {
                if (left < 0) {
                    left += capacity;
                }
                if (right >= capacity) {
                    right -= capacity;
                }
                if (!get_bitmap(bitmap_start(), left)) {
                    return left;
                } else {
                    if (array[left].first == key) {
                        return -1;
                    }
                }
                if (!get_bitmap(bitmap_start(), right)) {
                    return right;
                } else {
                    if (array[right].first == key) {
                        return -1;
                    }
                }
            }
            for (; i <= length; ++i,--left,++right) {
                if (left < 0) {
                    left += capacity;
                }
                if (right >= capacity) {
                    right -= capacity;
                }
                if (!get_bitmap(bitmap_start(), left)) {
                    max_offset = i;
                    return left;
                }
                if (!get_bitmap(bitmap_start(), right)) {
                    max_offset = i;
                    return right;
                }
            }
            return -1;
        }
    };


    template<class key_T, class value_T>
    class InnerNode {
    public:
        struct Slot {
            union {
                InnerNode *inner_node;
                DataNode<key_T, value_T> *data_node;
            };
        };
        using self_type = InnerNode;
        using slot_type = Slot;
    public:
        double lower{0};
        double upper{0};
        int capacity{-1};

        [[nodiscard]] inline auto forward(double key) const {
            return std::min(int(capacity - 1), std::max(0, int(forward_with_parameters(lower, upper, capacity, key))));
        }

        [[nodiscard]] inline
        std::pair<double, double> sub_interval(int position) const {
            auto slot_interval = (upper - lower) / double(capacity);
            return {slot_interval * double(position) + lower,
                    slot_interval * double(position + 1) + lower};
        }

        ///////
        slot_type array[0];

        inline unsigned int *bitmap_start() {
            return (unsigned int *) (array + capacity);
        }

        inline
        static self_type *
        new_segment(int capacity, double lower, double upper) {
            int t = ((capacity >> 5) + 1);
            auto memory_size = sizeof(self_type) + sizeof(slot_type) * capacity + sizeof(unsigned int) * t;
            auto node = (self_type *) std::malloc(memory_size);
            node->capacity = capacity;
            node->lower = lower;
            node->upper = upper;
            auto bit_map = node->bitmap_start();
            for (int i = 0; i < t; ++i) {
                bit_map[i] = 0;
            }
            node_memory_count += memory_size;
            return node;
        }

        inline
        static void
        delete_segment(self_type *node) {
            node_memory_count -= sizeof(self_type) + sizeof(slot_type) * node->capacity +
                                 sizeof(unsigned int) * ((node->capacity >> 5) + 1);
            std::free(node);
        }
    };

    class rebuild_lock {
        std::mutex position_lock;
        int frontend_root = -1;
        int frontend_inner = -1;
        int backend_root = -1;
        int backend_inner = -1;
    public:
        void set_frontend_position(int root, int inner) {
            while (true) {
                position_lock.lock();
                if (root != backend_root || inner != backend_inner) {
                    frontend_root = root;
                    frontend_inner = inner;
                    position_lock.unlock();
                    break;
                }
                position_lock.unlock();
            }
        }

        void set_backend_position(int root, int inner) {
            while (true) {
                position_lock.lock();
                if (root != frontend_root || inner != frontend_inner) {
                    backend_root = root;
                    backend_inner = inner;
                    position_lock.unlock();
                    break;
                }
                position_lock.unlock();
            }
        }

        void de_set_frontend_position() {
            position_lock.lock();
            frontend_root = -1;
            frontend_inner = -1;
            position_lock.unlock();
        }

        void de_set_backend_position() {
            position_lock.lock();
            backend_root = -1;
            backend_inner = -1;
            position_lock.unlock();
        }
    };

    template<class key_T, class value_T>
    std::vector<std::pair<std::pair<int, int>, std::pair<double, double>>> split_dataset(
            typename std::vector<std::pair<key_T, value_T>>::const_iterator begin,
            typename std::vector<std::pair<key_T, value_T>>::const_iterator end,
            CHA::InnerNode<key_T, value_T> *inner_node) {
        std::vector<std::pair<double, double>> intervals;
        std::vector<int> bucket(inner_node->capacity);
        for (auto i = begin; i < end; ++i) {
            ++bucket[inner_node->forward(i->first)];
        }
        std::vector<std::pair<std::pair<int, int>, std::pair<double, double>>> result;
        int left = 0, right = 0;
        for (int i = 0; i < inner_node->capacity; ++i) {
            left = right;
            right += bucket[i];
            result.push_back({{left, right}, inner_node->sub_interval(i)});
        }
        return result;
    }
    template<class key_T, class value_T>
    class Index {
    public:
        using self_type = Index<key_T, value_T>;
        using inner_node_type = InnerNode<key_T, value_T>;
        using data_node_type = DataNode<key_T, value_T>;

        double lower{};
        double upper{};
        Configuration conf;
        inner_node_type *root{};
        bool is_leaf{false};
#ifdef CB
        int rebuild_step = 0;
        rebuild_lock lock;
#endif
#ifdef using_small_network
        std::shared_ptr<Small_Q_network> q_network;
#endif
    public:

        float memory_occupied() {
            return ((float) node_memory_count + sizeof(self_type));
        }

        std::pair<bool, void *> build_(typename std::vector<std::pair<key_T, value_T>>::const_iterator begin,
                                       typename std::vector<std::pair<key_T, value_T>>::const_iterator end,
                                       std::pair<double, double> interval) {
#ifndef using_small_network
            auto data_node = DataNode<key_T,value_T>::new_segment(
                    std::max(int(float(end - begin)/default_density),DATA_NODE_SIZE),
                    interval.first,interval.second);
            for(auto i = begin;i<end;++i){
                auto slot_data = *i;
                auto position = data_node->find_insert(slot_data.first);
                if(position < 0){
                    throw MyException("bulk load insertion bad position!");
                }
                data_node->array[position] = slot_data;
                set_bitmap(data_node->bitmap_start(),position);
                ++data_node->size;
            }
            return {false,data_node};
#else
            auto pointer_result = std::pair<bool, void *>({false, nullptr});
            auto data_count = int(end - begin);
            int fanout = 1;
            if (data_count > DATA_NODE_SIZE) {
                auto pdf = get_pdf<key_T, value_T>(begin, end, interval.first, interval.second, SMALL_PDF_SIZE);
                auto a = torch::tensor(pdf).view({1, SMALL_PDF_SIZE}).mul(
                        torch::ones({int(action_space.size()), SMALL_PDF_SIZE})).to(GPU_DEVICE);
                auto b = torch::tensor(data_count).view({1, 1}).mul(torch::ones({int(action_space.size()), 1})).to(GPU_DEVICE);
                auto c = torch::tensor(action_space).to(torch::kFloat32).view({int(action_space.size()), 1}).to(GPU_DEVICE);
                auto pred_c = q_network->forward(a, b, c).to(CPU_DEVICE);
                auto ac_id = torch::argmax(pred_c,0).item().toInt();
                fanout = 1 << ac_id;
            }
            if (fanout > 1) {
                pointer_result.first = true;
                auto inner_node = Hits::InnerNode<key_T, value_T>::new_segment(fanout, interval.first,interval.second);
                pointer_result.second = inner_node;
                std::vector<std::pair<std::pair<int, int>, std::pair<double, double>>> result = split_dataset<key_T, value_T>(begin, end, inner_node);
                assert(result.size() == inner_node->capacity);
                for (int ii = 0; ii < int(result.size()); ++ii) {
                    auto tmp_result = build_(begin + result[ii].first.first,
                                             begin + result[ii].first.second, result[ii].second);
                    if (tmp_result.first) {
                        set_bitmap(inner_node->bitmap_start(), ii);
                        inner_node->array[ii].inner_node = (InnerNode<key_T, value_T> *) tmp_result.second;
                    } else {
                        de_set_bitmap(inner_node->bitmap_start(), ii);
                        inner_node->array[ii].data_node = (DataNode<key_T, value_T> *) tmp_result.second;
                    }
                }
            } else {
                auto data_node = Hits::DataNode<key_T, value_T>::new_segment(
                        int(double(std::max(DATA_NODE_SIZE, data_count)) / default_density), interval.first,interval.second);
                pointer_result.second = data_node;
                for (auto ii = begin; ii < end; ++ii) {
                    auto position = data_node->find_insert(ii->first);
                    if (position < 0) { std::cout <<"data_count:"<<data_count<<"  size:"<<data_node->size<<"  capacity:"<<data_node->capacity<< std::endl; throw MyException("bad position!"); }
                    data_node->array[position].first = ii->first;
                    data_node->array[position].second = ii->second;
                    set_bitmap(data_node->bitmap_start(), position);
                    ++data_node->size;
                }
            }
            return pointer_result;
#endif
        }

        void bulk_load(
                typename std::vector<std::pair<key_T,value_T>>::const_iterator begin,
                typename std::vector<std::pair<key_T,value_T>>::const_iterator end){
#ifdef CB
            stop_rebuild();
            rebuild_step = 0;
#endif
            delete_tree(root);
            inner_node_type::delete_segment(root);
            root = inner_node_type::new_segment(this->conf.root_fan_out, lower, upper);
            auto root_result = split_dataset(begin,end,root);
            for (int root_slot_id = 0; root_slot_id < root->capacity; ++root_slot_id) {
                auto inner_fanout = get_fanout(0,(root_result[root_slot_id].second.first + root_result[root_slot_id].second.second)/2);
                InnerNode<key_T,value_T>*  inner_node = inner_node_type::new_segment(inner_fanout, root_result[root_slot_id].second.first, root_result[root_slot_id].second.second);
                root->array[root_slot_id].inner_node = inner_node;
                set_bitmap(root->bitmap_start(),root_slot_id);
                auto inner_result = split_dataset(begin + root_result[root_slot_id].first.first,begin + root_result[root_slot_id].first.second,inner_node);
                for (int inner_slot_id = 0; inner_slot_id < inner_node->capacity; ++inner_slot_id) {
                    auto data_node = DataNode<key_T,value_T>::new_segment(
                            std::max(int(float(inner_result[inner_slot_id].first.second - inner_result[inner_slot_id].first.first)/default_density),DATA_NODE_SIZE),
                            inner_result[inner_slot_id].second.first, inner_result[inner_slot_id].second.second);
                    for(auto i = begin + root_result[root_slot_id].first.first + inner_result[inner_slot_id].first.first,
                            end_tmp = begin + root_result[root_slot_id].first.first + inner_result[inner_slot_id].first.second
                            ;i<end_tmp;++i){
                        auto position = data_node->find_insert(i->first);
                        if(position < 0){
                            throw MyException("bulk load insertion bad position!");
                        }
                        data_node->array[position] = *i;
                        set_bitmap(data_node->bitmap_start(),position);
                        ++data_node->size;
                    }
                    de_set_bitmap(inner_node->bitmap_start(), inner_slot_id);
                    inner_node->array[inner_slot_id].data_node = data_node;
                }
            }
#ifdef CB
            start_rebuild();
#endif
        }

#ifdef CB
    private:
        void stop_rebuild() {
            auto status = rebuild_step;
            ++rebuild_step;
            while (status + 1 == rebuild_step) { sleep(1); }
        }

        void start_rebuild() {
            rebuild_step = 0;
            auto status = rebuild_step;
            std::thread(std::bind(&Index::random_rebuild, this)).detach();
            while (status == rebuild_step);
        }

        std::vector<typename DataNode<key_T, value_T>::slot_type> dataset_for_rebuild;

        void get_data(InnerNode<key_T, value_T> *node) {
            for (int i = 0; i < node->capacity; ++i) {
                if (get_bitmap(node->bitmap_start(), i)) {
                    get_data(node->array[i].inner_node);
                } else {
                    auto data_node = node->array[i].data_node;
                    for (int j = 0; j < data_node->capacity; ++j) {
                        if (get_bitmap(data_node->bitmap_start(), j)) {
                            dataset_for_rebuild.push_back(data_node->array[j]);
                        }
                    }
                }
            }
        }

        void random_rebuild() {
            ++rebuild_step;
            auto status = rebuild_step;
//            puts("rebuild start!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
            while (true) {
//                sleep(1);
//                if (status != rebuild_step) {
//                    ++rebuild_step;
////                            puts("rebuild stop!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
//                    return;
//                }
                puts("rebuild!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
                for (int root_slot_id = 0; root_slot_id < root->capacity; ++root_slot_id) {
                    auto inner_node = root->array[root_slot_id].inner_node;
                    for (int inner_slot_id = 0; inner_slot_id < inner_node->capacity; ++inner_slot_id) {
                        lock.set_backend_position(root_slot_id, inner_slot_id);
                        auto min_max = std::pair<double, double>();
                        dataset_for_rebuild.clear();
                        auto data_node = inner_node->array[inner_slot_id].data_node;
                        min_max = {data_node->lower, data_node->upper};
                        for (int k = 0; k < data_node->capacity; ++k) {
                            if (get_bitmap(data_node->bitmap_start(), k)) {
                                dataset_for_rebuild.push_back(data_node->array[k]);
                            }
                        }
                        DataNode<key_T, value_T>::delete_segment(data_node);
                        data_node = DataNode<key_T,value_T>::new_segment(
                                std::max(int(float(dataset_for_rebuild.size())/default_density),DATA_NODE_SIZE),
                                min_max.first,min_max.second);
                        for(auto &i:dataset_for_rebuild){
                            auto position = data_node->find_insert(i.first);
                            if(position < 0){
                                throw MyException("bulk load insertion bad position!");
                            }
                            data_node->array[position] = i;
                            set_bitmap(data_node->bitmap_start(),position);
                            ++data_node->size;
                        }
                        de_set_bitmap(inner_node->bitmap_start(), inner_slot_id);
                        inner_node->array[inner_slot_id].data_node = data_node;
                        lock.de_set_backend_position();
                        if (status != rebuild_step) {
                            ++rebuild_step;
//                            puts("rebuild stop!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
                            return;
                        }
                    }
                    usleep(sleep_time);
//                    usleep(11190);
//                    sleep(1);
                }
            }
        }
#endif
    public:
        Index(Configuration conf, double lower, double upper) : conf(conf) {
            this->lower = lower;
            this->upper = upper;
            root = inner_node_type::new_segment(this->conf.root_fan_out, lower, upper);
            for (int i = 0; i < root->capacity; ++i) {
                auto child_interval = root->sub_interval(i);
                auto fanout = get_fanout(0, (child_interval.first + child_interval.second) / 2);
                auto new_inner_node = inner_node_type::new_segment(fanout, child_interval.first, child_interval.second);
                root->array[i].inner_node = new_inner_node;
                set_bitmap(root->bitmap_start(), i);
                for (int j = 0; j < new_inner_node->capacity; ++j) {
                    auto data_child_interval = new_inner_node->sub_interval(j);
                    auto new_data_node = data_node_type::new_segment(DATA_NODE_SIZE, data_child_interval.first,
                                                                     data_child_interval.second);
                    new_inner_node->array[j].data_node = new_data_node;
                    de_set_bitmap(new_inner_node->bitmap_start(), j);
                }
            }
#ifdef using_small_network
            q_network = std::make_shared<Small_Q_network>();
            torch::load(q_network, model_father_path + "tmp.pt");
            q_network->to(GPU_DEVICE);
            q_network->eval();
#endif
#ifdef CB
            start_rebuild();
#endif
        }

        ~Index() {
#ifdef CB
            stop_rebuild();
#endif
            if (is_leaf) {
                data_node_type::delete_segment((DataNode<key_T, value_T> *) root);
                return;
            }
            delete_tree(root);
            inner_node_type::delete_segment(root);
            root = nullptr;
        }


        inline int get_fanout(int layer, double node_interval_point) {
            auto pred = std::min(double(INNER_FANOUT_COLUMN - 1),
                                 std::max(0.0, double(INNER_FANOUT_COLUMN - 1) *
                                               (node_interval_point - lower) / (upper - lower)));
            auto left = int(pred);
            pred -= left;
            return int(0.5 + this->conf.fan_outs[std::min(INNER_FANOUT_ROW - 1, layer)][left] * (1 - pred) +
                       this->conf.fan_outs[std::min(INNER_FANOUT_ROW - 1, layer)][left + 1] * pred);
        }

        bool add(key_T key, const value_T &value) {

            auto root_position = root->forward(key);
            auto node = root->array[root_position].inner_node;
            auto position = node->forward(key);
#ifdef CB
            lock.set_frontend_position(root_position, position);
#endif
            while (get_bitmap(node->bitmap_start(), position)) {
                node = node->array[position].inner_node;
                position = node->forward(key);
            }
            auto data_node = node->array[position].data_node;
            if (data_node->size > data_node->capacity * max_density) {
#ifdef count_insert_time
                tc.synchronization();
#endif

                DataNode<key_T, value_T> *new_data_node = data_node_type::new_segment(
                        data_node->size * (1.0 / default_density), data_node->lower, data_node->upper);
                for (int i = 0; i < data_node->capacity; ++i) {
                    if (get_bitmap(data_node->bitmap_start(), i)) {
                        auto new_position = new_data_node->find_insert(data_node->array[i].first);
                        new_data_node->array[new_position] = data_node->array[i];
                        set_bitmap(new_data_node->bitmap_start(), new_position);
                        ++new_data_node->size;
                    }
                }
                data_node_type::delete_segment(data_node);
                node->array[position].data_node = new_data_node;
                data_node = new_data_node;
#ifdef count_insert_time
                insert_time_retrain += tc.get_timer_second();
#endif
            }
            auto data_position = data_node->find_insert(key);
            if (data_position < 0) {
#ifdef CB
                lock.de_set_frontend_position();
#endif
                return false;
            }
            data_node->array[data_position].first = key;
            data_node->array[data_position].second = value;
            set_bitmap(data_node->bitmap_start(), data_position);
            ++data_node->size;
#ifdef CB
            lock.de_set_frontend_position();
#endif
            return true;
        }

        bool get(key_T key, value_T &value) {

            auto root_position = root->forward(key);
            auto node = root->array[root_position].inner_node;
            auto position = node->forward(key);
#ifdef CB
            lock.set_frontend_position(root_position, position);
#endif
            while (get_bitmap(node->bitmap_start(), position)) {
                node = node->array[position].inner_node;
                position = node->forward(key);
            }
            auto data_node = node->array[position].data_node;
            auto data_position = data_node->find(key);
            if (data_position < 0 || data_node->array[data_position].first != key) {
#ifdef CB
                lock.de_set_frontend_position();
#endif
                return false;
            } else {
                value = data_node->array[data_position].second;
#ifdef CB
                lock.de_set_frontend_position();
#endif
                return true;
            }
        }

//        void range_query_(InnerNode<key_T,value_T> *node,std::vector<std::pair<key_T,value_T>> &result,double upper_bound){
//            for(int j = 0;j < node->capacity;++j){
//                if(get_bitmap(node->bitmap_start(), j)) {
//                    range_query_(result->array[j].inner_node,result);
//                }
//                else{
//                    auto data_node = result->array[j].data_node;
//                    for(auto l = 0;l < data_node->capacity;++l) {
//                        if(get_bitmap(data_node->bitmap_start(),l)){
//                            if(data_node->array[l].first < upper_bound) {
//                                result.push_back(data_node->array[l]);
//                            }
//                        }
//                    }
//                }
//                if(result->sub_interval(j).second >= upper_bound){
//                    return;
//                }
//            }
//        }
//
//        std::vector<std::pair<key_T,value_T>> range_query(key_T lower_bound, key_T upper_bound) {
//            std::vector<std::pair<key_T,value_T>> result;
//            auto key = lower_bound;
//            auto root_position = root->forward(key);
//            auto node = root->array[root_position].inner_node;
//            auto position = node->forward(key);
//            for(int i = root_position;i<root->capacity;++i){
//                auto sub_root = root->array[i].inner_node;
//                for(int j = position;j < sub_root->capacity;++j){
//#ifdef CB
//                    lock.set_frontend_position(i, j);
//#endif
//                    if(get_bitmap(node->bitmap_start(), position)) {
//                        range_query_(sub_root->array[j].inner_node,result,upper_bound);
//                    }
//                    else{
//                        auto data_node = sub_root->array[j].data_node;
//                        for(auto l = 0;l < data_node->capacity;++l) {
//                            if(get_bitmap(data_node->bitmap_start(),l)){
//                                if(data_node->array[l].first < upper_bound) {
//                                    result.push_back(data_node->array[l]);
//                                }
//                            }
//                        }
//                    }
//#ifdef CB
//                    lock.de_set_frontend_position();
//#endif
//                    if(sub_root->sub_interval(j).second >= upper_bound){
//                        return result;
//                    }
//                }
//                if(root->sub_interval(i).second >= upper_bound){
//                    return result;
//                }
//            }
//            return result;
//        }


        bool get_with_root_leaf(key_T key, value_T &value) {
            if (is_leaf) {
                auto data_node = (DataNode<key_T, value_T> *) root;
                ++inner_cost;
                auto data_position = data_node->find_with_cost(key);
                if (data_position < 0 || data_node->array[data_position].first != key) {
                    return false;
                } else {
                    value = data_node->array[data_position].second;
                    return true;
                }
            }
            auto node = root;
            auto position = node->forward(key);
            ++inner_cost;
            while (get_bitmap(node->bitmap_start(), position)) {
                node = node->array[position].inner_node;
                position = node->forward(key);
                ++inner_cost;
            }
            auto data_node = node->array[position].data_node;
            ++inner_cost;
            auto data_position = data_node->find_with_cost(key);
            if (data_position < 0 || data_node->array[data_position].first != key) {
                return false;
            } else {
                value = data_node->array[data_position].second;
                return true;
            }
        }

        bool get_with_cost(key_T key, value_T &value) {
            auto node = root;
            auto position = node->forward(key);
            ++inner_cost;
            while (get_bitmap(node->bitmap_start(), position)) {
                node = node->array[position].inner_node;
                position = node->forward(key);
                ++inner_cost;
            }
            auto data_node = node->array[position].data_node;
            ++inner_cost;
            auto data_position = data_node->find_with_cost(key);
            if (data_position < 0 || data_node->array[data_position].first != key) {
                return false;
            } else {
                value = data_node->array[data_position].second;
                return true;
            }
        }
        std::vector<long long> count_node_of_each_layer(){
            std::vector<long long> result;
            std::vector<inner_node_type*> q1;
            std::vector<inner_node_type*> q2;
            q1.push_back(root);
            while(true) {
                long long this_layer_count = 0;
                for(auto n:q1) {
                    this_layer_count += n->capacity;
                    for(int i = 0;i<n->capacity;++i) {
                        if(get_bitmap(n->bitmap_start(),i)) {
                            q2.push_back(n->array[i].inner_node);
                        }
                    }
                }
                result.push_back(this_layer_count);
                if(q2.empty()) {
                    break;
                }else {
                    q1 = q2;
                    q2.clear();
                }
            }
            return result;
        }

        bool erase(key_T key) {
            auto root_position = root->forward(key);
            auto node = root->array[root_position].inner_node;
            auto position = node->forward(key);
#ifdef CB
            lock.set_frontend_position(root_position, position);
#endif
            while (get_bitmap(node->bitmap_start(), position)) {
                node = node->array[position].inner_node;
                position = node->forward(key);
            }
            auto data_node = node->array[position].data_node;
            if (data_node->capacity > DATA_NODE_SIZE && data_node->size < data_node->capacity * min_density) {
                data_node_type *new_data_node = data_node_type::new_segment(
                        std::max(int(data_node->size * (1.0 / default_density)),DATA_NODE_SIZE),
                        data_node->lower, data_node->upper);
                for (int i = 0; i < data_node->capacity; ++i) {
                    if (get_bitmap(data_node->bitmap_start(), i)) {
                        auto new_position = new_data_node->find_insert(data_node->array[i].first);
                        new_data_node->array[new_position] = data_node->array[i];
                        set_bitmap(new_data_node->bitmap_start(), new_position);
                        ++new_data_node->size;
                    }
                }
                data_node_type::delete_segment(data_node);
                node->array[position].data_node = new_data_node;
                data_node = new_data_node;
            }
            auto data_position = data_node->find(key);
            if (data_position < 0 || data_node->array[data_position].first != key) {
#ifdef CB
                lock.de_set_frontend_position();
#endif
                return false;
            }
            de_set_bitmap(data_node->bitmap_start(), data_position);
            --data_node->size;
#ifdef CB
            lock.de_set_frontend_position();
#endif
            return true;
        }

        void delete_tree(InnerNode<key_T, value_T> *node) {
            for (int i = 0; i < node->capacity; ++i) {
                if (get_bitmap(node->bitmap_start(), i)) {
                    delete_tree(node->array[i].inner_node);
                    inner_node_type::delete_segment(node->array[i].inner_node);
                } else {
                    data_node_type::delete_segment(node->array[i].data_node);
                }
            }
        }
    };
}

#endif //INDEX_H
