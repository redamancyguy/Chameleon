//
// Created by redamancyguy on 23-9-7.
//
#include "../index/include/Index.hpp"
#include "../others/other_indexes.h"
template<class key_T, class value_T>
class DataNode {
public:
    using self_type = DataNode;
    using slot_type = std::pair<key_T, value_T>;
public:
    long cost = 0;
    double lower{0};
    double upper{0};
    int capacity{-1};

    [[nodiscard]] inline double forward(double key) const {
        return std::min(double((capacity) - 1), std::max(0.0, forward_with_parameters(lower, upper, (capacity), key)));
    }

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
        auto bit_map = node->bitmap_start();
        for (int i = 0; i < t; ++i) {
            bit_map[i] = 0;
        }
        for(int i = 0;i<capacity;++i){
            node->array[i].first = 0;
        }
        return node;
    }

    inline
    static void
    delete_segment(self_type *node) {
        std::free(node);
    }

    inline int hash(double key) {
        auto predict_hash = (long long) (forward(key) * hash_factor);
        return int(predict_hash % (long long) (capacity));
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
            ++cost;
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
class alex_node{
public:
    int capacity;
    double lower;
    double upper;
    struct Slot{
        bool is_using;
        std::pair<key_T,value_T> data;
    };
    Slot *array;
    alex_node(int capacity,double lower,double upper):capacity(capacity),lower(lower),upper(upper){
        array = new Slot[capacity];
    }
    ~alex_node(){
        delete []array;
    }

    int forward(double key){
        return int((key - lower) / (upper - lower) * double(capacity));
    }

};

int main(){
    auto dataset = dataset_source::get_dataset<std::pair<double,double>>(data_father_path + "face.data");
    int all_size = 200000000;
//    int all_size = 200000;
    std::shuffle(dataset.begin(), dataset.end(),e);
    auto min_max  = get_min_max<double,double>(dataset.begin(),dataset.end());
    auto datanode = DataNode<double,double>::new_segment(int(double(all_size) / 0.9 ),min_max.first,min_max.second);
    int step = 12;
    int step_size = all_size/step;
    for(int i = 0;i<step;++i){
        tc.synchronization();
        for(int j = 0;j<step_size;++j){
            int id = i * step_size + j;
            auto position = datanode->find_insert(dataset[id].first);
            if(position < 0){
                throw MyException("???? -1");
            }
            set_bitmap(datanode->bitmap_start(),position);
            datanode->array[position] = dataset[id];
            ++datanode->size;
        }
        std::cout <<i<<"add latency:"<<tc.get_timer_nanoSec() / step_size<< std::endl;
        tc.synchronization();
        datanode->cost = 0;
        for(int j = 0;j<step_size;++j){
            auto id = e() % step_size * (i + 1);
            datanode->find_with_cost(dataset[id].first);
        }
        std::cout <<i<<"get latency:"<<tc.get_timer_nanoSec() / step_size<< std::endl;
        std::cout <<"avg offset:"<<double(datanode->cost)/double((i+1)*step_size)<<std::endl;
    }

}