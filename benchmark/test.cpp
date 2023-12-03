//
// Created by wenli on 2023/1/25.
//
#include <iostream>
#include <random>
#include<functional>
#include <iomanip>
#include <torch/torch.h>


//#define CB
#define using_small_network

#include "../include/DEFINE.h"
#include "../index/include/Parameter.h"
//#define count_insert_time
//#ifdef count_insert_time
//double insert_time_retrain = 0;
//#endif

#include "../index/include/Index.hpp"
#include "../include/DataSet.hpp"
#include "../index/include/experience.hpp"
#include "../index/include/Controller.hpp"
#include "../others/other_indexes.h"

double train_proportion = 0.1;
auto train_size = 0;


GlobalController controller;

class EvaluationTask {
public:
    std::string dataset_name;
    int start = 0;
    int length = 0;
};

template<typename T>
std::pair<std::vector<T>, std::vector<T>> split_dataset(std::vector<T> dataset, std::pair<double, double> proportion) {
    auto train_size = std::size_t(double(dataset.size()) * proportion.first / (proportion.first + proportion.second));
    std::vector<std::size_t> indices(dataset.size());
    for (std::size_t i = 0; i < dataset.size(); i++) {
        indices[i] = i;
    }
    std::shuffle(indices.begin(), indices.end(), e);
    std::vector<T> train_dataset;
    train_dataset.reserve(train_size);
    std::vector<T> test_dataset;
    test_dataset.reserve(dataset.size() - train_size);
    for (std::size_t i = 0; i < train_size; i++) {
        train_dataset.push_back(dataset[indices[i]]);
    }
    for (std::size_t i = train_size; i < dataset.size(); i++) {
        test_dataset.push_back(dataset[indices[i]]);
    }
    return {train_dataset, test_dataset};
}

enum IndexType {
    B_PLUS_TREE,
    ALEX,
    PGM,
    LIPP_,
    HITS,
};

int steps = 4;

auto evaluate_b_tree(
        std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> bulkload_dataset,
        std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset) {
    std::vector<float> get_result;
    std::vector<float> add_result;
    std::vector<float> memory_result;
    std::vector<float> erase_result;
    std::vector<float> memory_result2;
    std::vector<float> get_result2;
    auto index = new stx::btree<KEY_TYPE, VALUE_TYPE>();
    index->bulk_load(bulkload_dataset.begin(), bulkload_dataset.end());
    int step_size = dataset.size() / steps;
    for (int i = 0; i < steps; i++) {
        tc.synchronization();
        for (int j = 0; j < step_size; j++) {
            auto id = i * step_size + j;
            if (!index->insert(dataset[id].first, dataset[id].second).second) {
                puts("b+tree add error");
            }
        }
        add_result.push_back(double(tc.get_timer_nanoSec()) / double(step_size));
        auto memory =
                float(sizeof(stx::btree<KEY_TYPE, VALUE_TYPE>) +
                      ((stx::btree<KEY_TYPE, VALUE_TYPE> *) index)->m_stats.innernodes *
                      sizeof(stx::btree<KEY_TYPE, VALUE_TYPE>::inner_node)
                      + ((stx::btree<KEY_TYPE, VALUE_TYPE> *) index)->m_stats.leaves *
                        sizeof(stx::btree<KEY_TYPE, VALUE_TYPE>::leaf_node)) / (1024 * 1024);
        memory_result.push_back(memory);
        VALUE_TYPE v;
        tc.synchronization();
        for (int j = 0; j < step_size; j++) {
            auto id = e() % ((i + 1) * step_size);
            if (index->find(dataset[id].first)->second != dataset[id].second) {
                puts("b+tree get error 1");
            }
        }
        get_result.push_back(double(tc.get_timer_nanoSec()) / double(step_size));
    }
/////////////////
    for (int i = 0; i < steps; i++) {
        VALUE_TYPE v;
        tc.synchronization();
        for (int j = 0; j < step_size; j++) {
            auto max = steps * step_size;
            auto min = (i * step_size);
            auto id = (e() % (max - min)) + min;
            if (index->find(dataset[id].first)->second != dataset[id].second) {
                puts("b+tree get error");
                std::cout << "i:" << i << "  j:" << j << std::endl;
            }
        }
        get_result2.push_back(double(tc.get_timer_nanoSec()) / double(step_size));
        tc.synchronization();
        for (int j = 0; j < step_size; j++) {
            auto id = i * step_size + j;
            if (index->erase(dataset[id].first) == 0) {
                puts("b+tree erase error");
            }
        }
        erase_result.push_back(double(tc.get_timer_nanoSec()) / double(step_size));
        auto memory = float(sizeof(stx::btree<KEY_TYPE, VALUE_TYPE>) +
                            ((stx::btree<KEY_TYPE, VALUE_TYPE> *) index)->m_stats.innernodes *
                            sizeof(stx::btree<KEY_TYPE, VALUE_TYPE>::inner_node)
                            + ((stx::btree<KEY_TYPE, VALUE_TYPE> *) index)->m_stats.leaves *
                              sizeof(stx::btree<KEY_TYPE, VALUE_TYPE>::leaf_node)) / (1024 * 1024);
        memory_result2.push_back(memory);
    }
    delete index;
    return std::vector<std::vector<float>>(
            {add_result, get_result, memory_result, erase_result, get_result2, memory_result2});
}


auto evaluate_cha(
        std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> bulkload_dataset,
        std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset) {
    std::vector<float> get_result;
    std::vector<float> add_result;
    std::vector<float> memory_result;
    std::vector<float> erase_result;
    std::vector<float> memory_result2;
    std::vector<float> get_result2;
    auto min_max = get_min_max<KEY_TYPE, VALUE_TYPE>(dataset.begin(), dataset.end());
    auto min_max2 = get_min_max<KEY_TYPE, VALUE_TYPE>(bulkload_dataset.begin(), bulkload_dataset.end());
    min_max.first = std::min(min_max.first,min_max2.first);
    min_max.second = std::min(min_max.second,min_max2.second);
    auto pdf = get_pdf<KEY_TYPE, VALUE_TYPE>(bulkload_dataset.begin(), bulkload_dataset.end(), min_max.first, min_max.second,
                                             BUCKET_SIZE);
    experience_t exp_chosen;
    std::copy(pdf.begin(), pdf.end(), exp_chosen.distribution);
    exp_chosen.data_size = float(bulkload_dataset.size() + dataset.size());
    auto best_gen = controller.get_best_action_GA(exp_chosen);
//    std::cout << "best_gen.conf.root_fan_out:" << best_gen.conf.root_fan_out << std::endl;
//    std::cout << "best_gen.conf.fan_outs[100]:" << best_gen.conf.fan_outs[0][100] << std::endl;
//    best_gen.conf = Hits::Configuration::default_configuration();
    auto index = new CHA::Index<KEY_TYPE, VALUE_TYPE>(best_gen.conf, min_max.first, min_max.second);
    index->bulk_load(bulkload_dataset.begin(), bulkload_dataset.end());
    int step_size = dataset.size() / steps;
    for (int i = 0; i < steps; i++) {
        tc.synchronization();
        for (int j = 0; j < step_size; j++) {
            auto id = i * step_size + j;
            if (!index->add(dataset[id].first, dataset[id].second)) {
                puts("hits add error");
            }
        }
        add_result.push_back(double(tc.get_timer_nanoSec()) / double(step_size));
        memory_result.push_back(((CHA::Index<KEY_TYPE, VALUE_TYPE> *) index)->memory_occupied() / (1024.0 * 1024.0));
        VALUE_TYPE v;
        tc.synchronization();
        for (int j = 0; j < step_size; j++) {
            auto id = e() % ((i + 1) * step_size);
            if (!index->get(dataset[id].first, v) ||
                v != dataset[id].second) {
                puts("hits get error 1");
            }
        }
        get_result.push_back(double(tc.get_timer_nanoSec()) / double(step_size));
    }
/////////////////
    for (int i = 0; i < steps; i++) {
        VALUE_TYPE v;
        tc.synchronization();
        for (int j = 0; j < step_size; j++) {
            auto max = steps * step_size;
            auto min = (i * step_size);
            auto id = (e() % (max - min)) + min;
            if (!index->get(dataset[id].first, v) ||
                v != dataset[id].second) {
                puts("hits get error");
            }
        }
        get_result2.push_back(double(tc.get_timer_nanoSec()) / double(step_size));
        tc.synchronization();
        for (int j = 0; j < step_size; j++) {
            auto id = i * step_size + j;
            if (!index->erase(dataset[id].first)) {
                puts("hits erase error");
            }
        }
        erase_result.push_back(double(tc.get_timer_nanoSec()) / double(step_size));
        memory_result2.push_back(((CHA::Index<KEY_TYPE, VALUE_TYPE> *) index)->memory_occupied() / (1024.0 * 1024.0));
    }
    delete index;
    return std::vector<std::vector<float>>(
            {add_result, get_result, memory_result, erase_result, get_result2, memory_result2});
}


auto evaluate_alex(
        std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> bulkload_dataset,
        std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset) {
    std::vector<float> get_result;
    std::vector<float> add_result;
    std::vector<float> memory_result;
    std::vector<float> erase_result;
    std::vector<float> memory_result2;
    std::vector<float> get_result2;
    auto index = new alex::Alex<KEY_TYPE, VALUE_TYPE>();
    index->bulk_load(bulkload_dataset.data(), bulkload_dataset.size());
    int step_size = dataset.size() / steps;
    for (int i = 0; i < steps; i++) {
        tc.synchronization();
        for (int j = 0; j < step_size; j++) {
            auto id = i * step_size + j;
            if (!index->insert(dataset[id].first,
                               dataset[id].second).second) {
                puts("alex add error");
            }
        }
        add_result.push_back(double(tc.get_timer_nanoSec()) / double(step_size));
        auto memory =
                float(sizeof(alex::Alex<KEY_TYPE, VALUE_TYPE>) +
                      ((alex::Alex<KEY_TYPE, VALUE_TYPE> *) index)->data_size() +
                      ((alex::Alex<KEY_TYPE, VALUE_TYPE> *) index)->model_size()) / (1024 * 1024);
        memory_result.push_back(memory);
        VALUE_TYPE v;
        tc.synchronization();
        for (int j = 0; j < step_size; j++) {
            auto id = e() % ((i + 1) * step_size);
            if (*index->get_payload(dataset[id].first) != dataset[id].second) {
                puts("alex get error 1");
            }
        }
        get_result.push_back(double(tc.get_timer_nanoSec()) / double(step_size));
    }
/////////////////
    for (int i = 0; i < steps; i++) {
        VALUE_TYPE v;
        tc.synchronization();
        for (int j = 0; j < step_size; j++) {
            auto max = steps * step_size;
            auto min = (i * step_size);
            auto id = (e() % (max - min)) + min;
            if (*index->get_payload(dataset[id].first) !=
                dataset[id].second) {
                puts("alex get error");
            }
        }
        get_result2.push_back(double(tc.get_timer_nanoSec()) / double(step_size));
        tc.synchronization();
        for (int j = 0; j < step_size; j++) {
            auto id = i * step_size + j;
            if (index->erase(dataset[id].first) == 0) {
                puts("alex erase error");
            }
        }
        erase_result.push_back(double(tc.get_timer_nanoSec()) / double(step_size));
        auto memory =
                float(sizeof(alex::Alex<KEY_TYPE, VALUE_TYPE>) +
                      ((alex::Alex<KEY_TYPE, VALUE_TYPE> *) index)->data_size() +
                      ((alex::Alex<KEY_TYPE, VALUE_TYPE> *) index)->model_size()) / (1024 * 1024);
        memory_result2.push_back(memory);
    }
    delete index;
    return std::vector<std::vector<float>>(
            {add_result, get_result, memory_result, erase_result, get_result2, memory_result2});
}


auto evaluate_pgm(
        std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> bulkload_dataset,
        std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset) {
    std::vector<float> get_result;
    std::vector<float> add_result;
    std::vector<float> memory_result;
    std::vector<float> erase_result;
    std::vector<float> memory_result2;
    std::vector<float> get_result2;
    auto index = new pgm::DynamicPGMIndex<KEY_TYPE, VALUE_TYPE>(bulkload_dataset.begin(), bulkload_dataset.end());
    int step_size = dataset.size() / steps;
    for (int i = 0; i < steps; i++) {
        tc.synchronization();
        for (int j = 0; j < step_size; j++) {
            auto id = i * step_size + j;
            index->insert_or_assign(dataset[id].first, dataset[id].second);
        }
        add_result.push_back(double(tc.get_timer_nanoSec()) / double(step_size));
        memory_result.push_back(
                float(((pgm::DynamicPGMIndex<KEY_TYPE, VALUE_TYPE> *) index)->size_in_bytes()) / (1024 * 1024));
        VALUE_TYPE v;
        tc.synchronization();
        for (int j = 0; j < step_size; j++) {
            auto id = e() % ((i + 1) * step_size);
            if (((pgm::DynamicPGMIndex<KEY_TYPE, VALUE_TYPE> *) index)->find(dataset[id].first)->second !=
                dataset[id].second) {
                puts("pgm get error 1");
            }
        }
        get_result.push_back(double(tc.get_timer_nanoSec()) / double(step_size));
    }
/////////////////
    for (int i = 0; i < steps; i++) {
        VALUE_TYPE v;
        tc.synchronization();
        for (int j = 0; j < step_size; j++) {
            auto max = steps * step_size;
            auto min = (i * step_size);
            auto id = (e() % (max - min)) + min;
            if (((pgm::DynamicPGMIndex<KEY_TYPE, VALUE_TYPE> *) index)->find(dataset[id].first)->second !=
                dataset[id].second) {
                puts("pgm get error");
            }
        }
        get_result2.push_back(double(tc.get_timer_nanoSec()) / double(step_size));
        tc.synchronization();
        for (int j = 0; j < step_size; j++) {
            auto id = i * step_size + j;
            ((pgm::DynamicPGMIndex<KEY_TYPE, VALUE_TYPE> *) index)->erase(dataset[id].first);
        }
        erase_result.push_back(double(tc.get_timer_nanoSec()) / double(step_size));
        memory_result2.push_back(
                float(((pgm::DynamicPGMIndex<KEY_TYPE, VALUE_TYPE> *) index)->size_in_bytes()) / (1024 * 1024));
    }
    delete ((pgm::DynamicPGMIndex<KEY_TYPE, VALUE_TYPE> *) index);
    return std::vector<std::vector<float>>(
            {add_result, get_result, memory_result, erase_result, get_result2, memory_result2});
}


auto evaluate_lipp(
        std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> bulkload_dataset,
        std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset) {
    std::vector<float> get_result;
    std::vector<float> add_result;
    std::vector<float> memory_result;
    std::vector<float> erase_result;
    std::vector<float> memory_result2;
    std::vector<float> get_result2;
    auto index = new LIPP<KEY_TYPE, VALUE_TYPE>();
    ((LIPP<KEY_TYPE, VALUE_TYPE> *) index)->bulk_load(bulkload_dataset.data(), bulkload_dataset.size());
    int step_size = dataset.size() / steps;
    for (int i = 0; i < steps; i++) {
        tc.synchronization();
        for (int j = 0; j < step_size; j++) {
            auto id = i * step_size + j;
            ((LIPP<KEY_TYPE, VALUE_TYPE> *) index)->insert(dataset[id].first, dataset[id].second);
        }
        add_result.push_back(double(tc.get_timer_nanoSec()) / double(step_size));
        memory_result.push_back(float(((LIPP<KEY_TYPE, VALUE_TYPE> *) index)->index_size()) / (1024 * 1024));
        VALUE_TYPE v;
        tc.synchronization();
        for (int j = 0; j < step_size; j++) {
            auto id = e() % ((i + 1) * step_size);
            if (((LIPP<KEY_TYPE, VALUE_TYPE> *) index)->at(dataset[id].first, false) != dataset[id].second) {
                puts("lipp get error 1");
            }
        }
        get_result.push_back(double(tc.get_timer_nanoSec()) / double(step_size));
    }
/////////////////
    for (int i = 0; i < steps; i++) {
        VALUE_TYPE v;
        tc.synchronization();
        for (int j = 0; j < step_size; j++) {
            auto max = steps * step_size;
            auto min = (i * step_size);
            auto id = (e() % (max - min)) + min;
            if (((LIPP<KEY_TYPE, VALUE_TYPE> *) index)->at(dataset[id].first, false) != dataset[id].second) {
                puts("lipp get error");
            }
        }
        get_result2.push_back(double(tc.get_timer_nanoSec()) / double(step_size));
        tc.synchronization();
        for (int j = 0; j < step_size; j++) {
            auto id = i * step_size + j;
            if (!((LIPP<KEY_TYPE, VALUE_TYPE> *) index)->erase(dataset[id].first)) {
                puts("lipp earse error !");
            }
        }
        erase_result.push_back(double(tc.get_timer_nanoSec()) / double(step_size));
        memory_result2.push_back(float(((LIPP<KEY_TYPE, VALUE_TYPE> *) index)->index_size()) / (1024 * 1024));
    }
    delete ((LIPP<KEY_TYPE, VALUE_TYPE> *) index);
    return std::vector<std::vector<float>>(
            {add_result, get_result, memory_result, erase_result, get_result2, memory_result2});
}


//int all_size = 40000000;
int all_size = 200000000;
int bulkload_size = 4000000;


auto evaluate_cha_none_exist_key(std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> bulkload_dataset,std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset) {
    Cost cost;
    auto min_max = get_min_max<KEY_TYPE, VALUE_TYPE>(dataset.begin(), dataset.end());
    auto pdf = get_pdf<KEY_TYPE, VALUE_TYPE>(dataset.begin(), dataset.end(), min_max.first,
                                             min_max.second,BUCKET_SIZE);
    experience_t exp_chosen;
    std::copy(pdf.begin(), pdf.end(), exp_chosen.distribution);
    exp_chosen.data_size = float(dataset.size() + bulkload_dataset.size());
    auto best_gen = controller.get_best_action_GA(exp_chosen);
    auto index = new CHA::Index<KEY_TYPE, VALUE_TYPE>(best_gen.conf, min_max.first, min_max.second);
    long long opt_count = 0;
    VALUE_TYPE v;
    tc.synchronization();
    for(auto i:dataset){
        if(!index->get(i.first,v)){
            index->add(i.first,i.second);
            opt_count++;
        }
        opt_count++;
    }
    delete index;
    cost.add = opt_count / tc.get_timer_second();
    return cost;
}

auto evaluate_alex_none_exist_key(std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset) {
    Cost cost;
    auto index = new alex::Alex<KEY_TYPE, VALUE_TYPE>();
    auto min_max = get_min_max<KEY_TYPE,VALUE_TYPE>(dataset.begin(),dataset.end());
    index->insert(min_max.first,min_max.first);
    index->insert(min_max.second,min_max.second);
    long long opt_count = 0;
    tc.synchronization();
    for(auto i:dataset){
        if(index->find(i.first) == index->end()){
            index->insert(i);
            opt_count++;
        }
        opt_count++;
    }
    delete index;
    cost.add = opt_count / tc.get_timer_second();
    return cost;
}
auto evaluate_pgm_none_exist_key(std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset) {
    Cost cost;
    auto index = new pgm::DynamicPGMIndex<KEY_TYPE, VALUE_TYPE>();
    long long opt_count = 0;
    tc.synchronization();
    for(auto i:dataset){
        if(index->find(i.first) == index->end()){
            index->insert_or_assign(i.first,i.second);
            opt_count++;
        }
        opt_count++;
    }
    delete index;
    cost.add = opt_count / tc.get_timer_second();
    return cost;
}

auto evaluate_lipp_none_exist_key(std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset) {
    Cost cost;
    auto index = new LIPP<KEY_TYPE, VALUE_TYPE>();
    long long opt_count = 0;
    tc.synchronization();
    for(auto i:dataset){
        if(!index->exists(i.first)){
            index->insert(i.first,i.second);
            opt_count++;
        }
        opt_count++;
    }
    delete index;
    cost.add = opt_count / tc.get_timer_second();
    return cost;
}

auto evaluate_b_tree_none_exist_key(std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset) {
    Cost cost;
    auto index = new stx::btree<KEY_TYPE, VALUE_TYPE>();
    long long opt_count = 0;
    tc.synchronization();
    for(auto i:dataset){
        if(index->find(i.first) == index->end()){
            index->insert(i);
            opt_count++;
        }
        opt_count++;
    }
    delete index;
    cost.add = opt_count / tc.get_timer_second();
    return cost;
}

std::vector<std::pair<double,long>> osmc_dimensionality_reduction(){
    auto osm_data = load_osm_binary_data<std::pair<std::pair<double, double>, long>>(
            "/home/redamancyguy/Desktop/buffer/data/osm_data/asia-latest.bin",0,600e6);
    double x_min = std::numeric_limits<double>::max(),x_max=-std::numeric_limits<double>::max();
    double y_min = std::numeric_limits<double>::max(),y_max = -std::numeric_limits<double>::max();
    for(auto &i:osm_data){
        x_min = std::min(x_min,i.first.first);
        x_max = std::max(x_max,i.first.first);
        y_min = std::min(y_min,i.first.second);
        y_max = std::max(y_max,i.first.second);
    }
}

//max_error
void exp6(){

}


void exp2(){
    controller.load_in();

//    std::cout <<"ALEX:"<<test_get_performance_with_increasing_dataset_size(ALEX)<< std::endl;
//    std::cout <<"PGM:"<<test_get_performance_with_increasing_dataset_size(PGM)<< std::endl;
//    std::cout <<"LIPP:"<<test_get_performance_with_increasing_dataset_size(LIPP_)<< std::endl;
//    std::cout <<"Hits:"<<test_get_performance_with_increasing_dataset_size(CHA)<< std::endl;
//    controller.query_weight *= 30;
//    controller.query_weight *= 10;

//    auto osm_data = load_osm_binary_data<std::pair<std::pair<double, double>, long>>(
//            "/home/redamancyguy/Desktop/buffer/data/osm_data/asia-latest.bin",0,40e6);
//    std::unordered_map<double,double> unique_bucket;
//    unique_bucket.reserve(osm_data.size());
//    for(auto dd:osm_data){
//        unique_bucket[dd.first.first+0.5*dd.first.second] = dd.first.second;
//    }
//    std::cout <<"unique_bucket.size:"<<unique_bucket.size()<< std::endl;

//    for(auto batch:std::vector<int>({333,1000,2222,3333,10000,10,33,66,100,133,200,})){
//        for(auto skew:std::vector<double>({333,1000,3333,10000,33333,10,33,100,222,})){
//            std::cout <<"batch:"<<batch<<" skew:"<<skew<< std::endl;
//            auto dataset = create_dataset(10e6,skew,batch);
//            std::shuffle(dataset.begin(), dataset.end(), e);
//            bulkload_size = dataset.size() * 0.5;
//            auto bulkload_dataset = std::vector(dataset.begin(), dataset.begin() + bulkload_size);
//            std::sort(bulkload_dataset.begin(), bulkload_dataset.end(),
//                      [&](std::pair<KEY_TYPE, VALUE_TYPE> &a, std::pair<KEY_TYPE, VALUE_TYPE> &b) {
//                          return a.first < b.first; });
//            dataset.erase(dataset.begin(), dataset.begin() + bulkload_size);
//            auto result_ = evaluate_cha(bulkload_dataset, dataset);
//            puts("cha");
//            for (int i = 0; i < steps; ++i) {
//                std::cout << "step:" << i << " add:" << result_[0][i] << " get:" << result_[1][i] << " memory:"
//                          << result_[2][i] << " erase:" << result_[3][i] << " get:" << result_[4][i] << " memory:"
//                          << result_[5][i] << std::endl;
//            }
//            result_ = evaluate_alex(bulkload_dataset, dataset);
//            puts("alex");
//            for (int i = 0; i < steps; ++i) {
//                std::cout << "step:" << i << " add:" << result_[0][i] << " get:" << result_[1][i] << " memory:"
//                          << result_[2][i] << " erase:" << result_[3][i] << " get:" << result_[4][i] << " memory:"
//                          << result_[5][i] << std::endl;
//            }
//        }
//    }
//    return;
    for (const auto &dataset_name: std::vector<std::string>(
//            { "uden.data","osmc.data","local_skew.data","face.data", })) {//"osmc.data",  "wiki.data","logn.data",
            {"logn.data", })) {//"osmc.data",  "wiki.data","logn.data",
        std::cout << " dataset_name:" << dataset_name << std::endl;
        auto dataset = dataset_source::get_dataset<std::pair<KEY_TYPE, VALUE_TYPE>>(data_father_path + dataset_name);
//        e.seed(1000000);
//        std::shuffle(dataset.begin(), dataset.end(), e);
        if(dataset.size() > all_size){
            dataset.resize(all_size);
        }
        bulkload_size = dataset.size() * 0.25;
        std::cout <<"bulkload_size:"<<bulkload_size<< std::endl;
        auto bulkload_dataset = std::vector(dataset.begin(), dataset.begin() + bulkload_size);
        std::sort(bulkload_dataset.begin(), bulkload_dataset.end(),
                  [&](std::pair<KEY_TYPE, VALUE_TYPE> &a, std::pair<KEY_TYPE, VALUE_TYPE> &b) {
                      return a.first < b.first; });
        dataset.erase(dataset.begin(), dataset.begin() + bulkload_size);
        auto result = evaluate_b_tree(bulkload_dataset, dataset);
        puts("b+tree");
        TimerClock tc;
        for (int i = 0; i < steps; ++i) {
            std::cout << "step:" << i << " add:" << result[0][i] << " get:" << result[1][i]
                      << " erase:" << result[3][i] << " get:" << result[4][i] << std::endl;
        }
        //
        result = evaluate_alex(bulkload_dataset, dataset);
        puts("alex");
        for (int i = 0; i < steps; ++i) {
            std::cout << "step:" << i << " add:" << result[0][i] << " get:" << result[1][i]
                      << " erase:" << result[3][i] << " get:" << result[4][i] << std::endl;
        }
        //
        result = evaluate_pgm(bulkload_dataset, dataset);
        puts("pgm");
        for (int i = 0; i < steps; ++i) {
            std::cout << "step:" << i << " add:" << result[0][i] << " get:" << result[1][i]
                      << " erase:" << result[3][i] << " get:" << result[4][i] << std::endl;
        }
        //
        result = evaluate_lipp(bulkload_dataset, dataset);
        puts("lipp");
        for (int i = 0; i < steps; ++i) {
            std::cout << "step:" << i << " add:" << result[0][i] << " get:" << result[1][i]
                      << " erase:" << result[3][i] << " get:" << result[4][i] << std::endl;
        }

        result = evaluate_cha(bulkload_dataset, dataset);
        puts("cha");
        for (int i = 0; i < steps; ++i) {
            std::cout << "step:" << i << " add:" << result[0][i] << " get:" << result[1][i]
                      << " erase:" << result[3][i] << " get:" << result[4][i] << std::endl;
        }
    }
}



void Scalability_test(){
    for (const auto &dataset_name: std::vector<std::string>(
            {"wiki.data","logn.data","logn.data","osmc.data", "face.data"})) {
        std::cout << " dataset_name:" << dataset_name << std::endl;
        auto dataset = dataset_source::get_dataset<std::pair<KEY_TYPE, VALUE_TYPE>>(data_father_path + dataset_name);
        std::shuffle(dataset.begin(), dataset.end(), e);
        for(auto dataset_size:std::vector({20*1e6,40*1e6,80*1e6,140*1e6,200*1e6,})){
            auto bulkload_dataset = std::vector(dataset.begin() + all_size,dataset.begin()+all_size + dataset_size);
            dataset.resize(all_size);
            controller.query_weight = 52428.8;
            for(int i = 0;i<100;i++){
                auto cost = evaluate_cha_none_exist_key(bulkload_dataset,dataset);
                std::cout <<"cha"<<cost<<"  controller.query_weight:"<<controller.query_weight<< std::endl;
                controller.query_weight *= 2;
            }
        }

    }
}

int main() {
    controller.load_in();
    e.seed(1000);
//    auto new_dd = create_dataset(200e6,1000000,1000);
//    auto new_dd = create_dataset(200e6,1000000,333);
//    dataset_source::set_dataset(data_father_path + "local_skew.data",new_dd);
    exp2();
    return 0;
//    std::cout <<"ALEX:"<<test_get_performance_with_increasing_dataset_size(ALEX)<< std::endl;
//    std::cout <<"PGM:"<<test_get_performance_with_increasing_dataset_size(PGM)<< std::endl;
//    std::cout <<"LIPP:"<<test_get_performance_with_increasing_dataset_size(LIPP_)<< std::endl;
//    std::cout <<"Hits:"<<test_get_performance_with_increasing_dataset_size(CHA)<< std::endl;
//    controller.query_weight *= 10;

    for (const auto &dataset_name: std::vector<std::string>(
            {"wiki.data","logn.data","logn.data","osmc.data", "face.data"})) {
        std::cout << " dataset_name:" << dataset_name << std::endl;
        auto dataset = dataset_source::get_dataset<std::pair<KEY_TYPE, VALUE_TYPE>>(data_father_path + dataset_name);
        std::shuffle(dataset.begin(), dataset.end(), e);
        auto bulkload_dataset = std::vector(dataset.begin() + all_size,dataset.begin()+all_size + 4000000);
        dataset.resize(all_size);
        dataset.insert(dataset.end(),dataset.begin(),dataset.end());
//        auto cost = evaluate_b_tree_none_exist_key(dataset);
//        std::cout <<"b+tree"<<cost<< std::endl;
//        cost = evaluate_alex_none_exist_key(dataset);
//        std::cout <<"alex"<<cost<< std::endl;
//        cost = evaluate_pgm_none_exist_key(dataset);
//        std::cout <<"pgm"<<cost<< std::endl;
//        if(dataset_name != "wiki.data"){
//            cost = evaluate_lipp_none_exist_key(dataset);
//            std::cout <<"lipp"<<cost<< std::endl;
//        }

//        auto cost = evaluate_cha_none_exist_key(bulkload_dataset,dataset);
//        std::cout <<"cha"<<cost<<"  controller.query_weight:"<<controller.query_weight<< std::endl;
//        continue;
        controller.query_weight = 52428.8;
        for(int i = 0;i<100;i++){
            auto cost = evaluate_cha_none_exist_key(bulkload_dataset,dataset);
            std::cout <<"cha"<<cost<<"  controller.query_weight:"<<controller.query_weight<< std::endl;
            controller.query_weight *= 2;
        }
    }
    return 0;
}
