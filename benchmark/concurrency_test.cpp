
// Created by wenli on 2023/1/25.
//
#include <iostream>
#include <random>
#include<functional>
#include <iomanip>
#include <torch/torch.h>


#define CB
//#define using_small_network

#include "../include/DEFINE.h"
#include "../index/include/Parameter.h"
//#define count_insert_time
#ifdef count_insert_time
double insert_time_retrain = 0;
#endif
int sleep_time = 0;
#include "../index/include/Index.hpp"
#include "../include/DataSet.hpp"
#include "../index/include/experience.hpp"
#include "../index/include/Controller.hpp"

int bulkload_size = 40000000;
int batch_query = 100000;

GlobalController controller;

Cost hits_evaluation(std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset,
                     bool using_model = true) {
    experience_t exp_chosen;
    auto min_max = get_min_max<KEY_TYPE,VALUE_TYPE>(dataset.begin() , dataset.end());
    auto pdf = get_pdf<KEY_TYPE,VALUE_TYPE>(dataset.begin() , dataset.end(), min_max.first, min_max.second,BUCKET_SIZE);
    std::copy(pdf.begin(),pdf.end(),exp_chosen.distribution);
    exp_chosen.data_size = float(batch_query);
    CHA::Configuration conf;
    if (using_model) {
        auto best_gen = controller.get_best_action_GA(exp_chosen);
        conf = best_gen.conf;
    } else {
        conf = CHA::Configuration::default_configuration();
    }
    exp_chosen.conf = conf;
    auto index = new CHA::Index<KEY_TYPE, VALUE_TYPE>(conf, min_max.first, min_max.second);
    std::sort(dataset.begin(),dataset.begin()+bulkload_size,
              [=](std::pair<KEY_TYPE, VALUE_TYPE> &a,std::pair<KEY_TYPE, VALUE_TYPE> &b){return a.first < b.first;});
    index->bulk_load(dataset.begin(),dataset.begin()+bulkload_size);
    std::shuffle(dataset.begin(),dataset.begin()+bulkload_size,e);
    tc.synchronization();
    std::ofstream file("/home/redamancyguy/Desktop/buffer/Hits/programes/experiments/concurrency/result/batch_data_cc_"+std::to_string(sleep_time)+".data");
    for(int i = bulkload_size;i<dataset.size();++i){
        if(!index->add(dataset[i].first,dataset[i].second)){
            puts("index add error");
        }
        if(!index->erase(dataset[i-bulkload_size].first)){
            puts("index erase error");
        }
        if(i % batch_query == 0){
            file <<tc.get_timer_nanoSec() / double(batch_query)<< std::endl;
            std::cout <<tc.get_timer_nanoSec() / double(batch_query)<< std::endl;
            tc.synchronization();
        }
    }

    delete index;
    return exp_chosen.cost;
}


int main() {
    controller.load_in();
    controller.query_weight *= 30;

    for (const auto &dataset_name: std::vector<std::string>(
            { "face.data","osmc.data","wiki.data", "logn.data", })) {
        std::cout << " dataset_name:" << dataset_name << std::endl;
        auto dataset = dataset_source::get_dataset<std::pair<KEY_TYPE, VALUE_TYPE>>(data_father_path + dataset_name);
//        std::shuffle(dataset.begin(), dataset.end(), e);
        dataset.insert(dataset.end(),dataset.begin(),dataset.end());
//        dataset.insert(dataset.end(),dataset.begin(),dataset.end());
        for(auto slp_t:std::vector<int>({0,1,10,100,1000,10000,100000})){
            sleep_time = slp_t;
            auto cost = hits_evaluation(dataset, true);
            std::cout <<cost<< std::endl;
        }
        return 0;
    }
    return 0;
}
