//
// Created by wenli on 2023/1/25.
//
#include <iostream>
#include <random>
#include<functional>
#include <iomanip>
#include <torch/torch.h>
#include "../include/DEFINE.h"
#include "../index/include/Index.hpp"
#include "../include/DataSet.hpp"
#include "../index/include/experience.hpp"
#include "../index/include/Controller.hpp"
#include "../others/other_indexes.h"

double train_proportion = 0.33;
auto train_size = 0;

GlobalController controller;

Cost hits_basic_evaluation(const std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> &dataset,
                           bool using_model = true, float hw = 0.05) {
    experience_t exp_chosen;
    auto min_max = get_min_max<KEY_TYPE,VALUE_TYPE>(dataset.begin(),dataset.end());
    auto pdf = get_pdf<KEY_TYPE,VALUE_TYPE>(dataset.begin() , dataset.end(), min_max.first, min_max.second,BUCKET_SIZE);
    std::copy(pdf.begin(),pdf.end(),exp_chosen.distribution);
    exp_chosen.data_size = float(dataset.size());
    CHA::Configuration conf;
    if (using_model) {
        controller.query_weight = hw;
        auto best_gen = controller.get_best_action_GA(exp_chosen);
        conf = best_gen.conf;
    } else {
        conf = CHA::Configuration::default_configuration();
        conf.root_fan_out = float(dataset.size());
    }
    exp_chosen.conf = conf;
    auto index = new CHA::Index<KEY_TYPE, VALUE_TYPE>(conf, min_max.first, min_max.second);
    TimerClock tc;
    for (int i = 0; i < train_size; ++i) {
        if (!index->add(dataset[i].first, dataset[i].second)) {
            puts("hits add error !");
        }
    }
    tc.synchronization();
    for (int i = train_size; i < int(dataset.size()); ++i) {
        if (!index->add(dataset[i].first, dataset[i].second)) {
            puts("hits add error !");
        }
    }
    exp_chosen.cost.add = (float) ((double) tc.get_timer_nanoSec() / ((double) (dataset.size() - train_size)));
    exp_chosen.cost.memory = index->memory_occupied() /  (1024 * 1024);
    tc.synchronization();
    VALUE_TYPE value;
    for (int i = train_size; i < int(dataset.size()); ++i) {
        if (!index->get(dataset[i].first, value) || value != dataset[i].second) {
            std::cout << "hits get error:" << dataset[i].first << std::endl;
        }
    }
    exp_chosen.cost.get = (float) ((double) tc.get_timer_nanoSec() / ((double) (dataset.size() - train_size)));
    tc.synchronization();
    for (int i = train_size; i < int(dataset.size()); ++i) {
        if (!index->erase(dataset[i].first)) {
            puts("hits erase error !");
        }
    }
    exp_chosen.cost.erase = (float) ((double) tc.get_timer_nanoSec() / ((double) (dataset.size() - train_size)));
    delete index;
    return exp_chosen.cost;
}


Cost cha_read_only(std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset,bool using_model = true) {
    experience_t exp_chosen;
    auto min_max = get_min_max<KEY_TYPE,VALUE_TYPE>(dataset.begin(),dataset.end());
    auto pdf = get_pdf<KEY_TYPE,VALUE_TYPE>(dataset.begin(),dataset.end(),min_max.first,min_max.second,BUCKET_SIZE);
    std::copy(pdf.begin(),pdf.end(),exp_chosen.distribution);
    exp_chosen.data_size = float(dataset.size());
    CHA::Configuration conf;
    if (using_model) {
        auto best_gen = controller.get_best_action_GA(exp_chosen);
        std::cout <<"                                                                               pred-cost:"<<"  get:"<<best_gen.cost.get<<"  memory:"<<(best_gen.cost.memory * sizeof(std::pair<double,double>) * dataset.size())/(1024.0 * 1024.0)<< std::endl;
        conf = best_gen.conf;
    } else {
        conf = CHA::Configuration::default_configuration();
    }
    exp_chosen.conf = conf;
    auto index = new CHA::Index<KEY_TYPE, VALUE_TYPE>(conf, min_max.first, min_max.second);

    tc.synchronization();
    index->bulk_load(dataset.begin(),dataset.end());
    exp_chosen.cost.add = tc.get_timer_nanoSec() / double(dataset.size());
    exp_chosen.cost.memory = float(index->memory_occupied() / (1024. * 1024.));
    auto tmp_e = e;
    tmp_e.seed(1000);
    dataset.resize(10e6);
    tc.synchronization();
    VALUE_TYPE value;
    for (int i=0;i<10e6;++i) {
        auto random_index=e() %  dataset.size();
        if (!index->get( dataset[random_index].first,value) || dataset[random_index].second != value) {
            std::cout << "cha get error:" << dataset[random_index] << std::endl;
        }
    }
    exp_chosen.cost.get = (float) ((double) tc.get_timer_nanoSec() / ((double) dataset.size()));
    tc.synchronization();
    exp_chosen.cost.erase = (float) ((double) tc.get_timer_nanoSec() / ((double) dataset.size()));
    delete index;
    return exp_chosen.cost;
}


Cost alex_read_only(std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset) {
    experience_t exp_chosen;
    auto min_max = get_min_max<KEY_TYPE,VALUE_TYPE>(dataset.begin(),dataset.end());
    auto pdf = get_pdf<KEY_TYPE,VALUE_TYPE>(dataset.begin(),dataset.end(),min_max.first,min_max.second,BUCKET_SIZE);
    std::copy(pdf.begin(),pdf.end(),exp_chosen.distribution);
    exp_chosen.data_size = float(dataset.size());
    CHA::Configuration conf;
    exp_chosen.conf = conf;
    auto index = new alex::Alex<KEY_TYPE, VALUE_TYPE>();
    tc.synchronization();
    index->bulk_load(dataset.data(),dataset.size());
    exp_chosen.cost.add = tc.get_timer_nanoSec() / double(dataset.size());
    exp_chosen.cost.memory =  float(sizeof(alex::Alex<KEY_TYPE, VALUE_TYPE>) + index->data_size() + index->model_size()) / (1024 * 1024);
    auto tmp_e = e;
    tmp_e.seed(1000);
    dataset.resize(10e6);
    tc.synchronization();
    VALUE_TYPE value;
    for (int i=0;i<10e6;++i) {
        auto random_index=e() %  dataset.size();
        if (index->find( dataset[random_index].first).payload() != dataset[random_index].second) {
            std::cout << "alex get error:" << dataset[random_index] << std::endl;
        }
    }
    exp_chosen.cost.get = (float) ((double) tc.get_timer_nanoSec() / ((double) dataset.size()));
    tc.synchronization();
    exp_chosen.cost.erase = (float) ((double) tc.get_timer_nanoSec() / ((double) dataset.size()));
    delete index;
    return exp_chosen.cost;
}


Cost pgm_read_only(std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset) {
    experience_t exp_chosen;

    auto index = new pgm::DynamicPGMIndex<KEY_TYPE, VALUE_TYPE>(dataset.begin(), dataset.end());
    tc.synchronization();
    exp_chosen.cost.add = tc.get_timer_nanoSec() / double(dataset.size());
    exp_chosen.cost.memory = float(index->size_in_bytes()) / (1024 * 1024);
    auto tmp_e = e;
    tmp_e.seed(1000);
    dataset.resize(10e6);
    tc.synchronization();
    VALUE_TYPE value;
    for (int i=0;i<10e6;++i) {
        auto random_index=e() %  dataset.size();
        if (dataset[random_index].second != index->find(dataset[random_index].first)->second) {
            std::cout << "pgm get error:" << dataset[random_index].first << std::endl;
        }
    }
    exp_chosen.cost.get = (float) ((double) tc.get_timer_nanoSec() / ((double) dataset.size()));
    tc.synchronization();
    exp_chosen.cost.erase = (float) ((double) tc.get_timer_nanoSec() / ((double) dataset.size()));
    delete index;
    return exp_chosen.cost;
}

Cost lipp_read_only(std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset) {
    experience_t exp_chosen;
    auto index = new LIPP<KEY_TYPE, VALUE_TYPE>();
    tc.synchronization();
    index->bulk_load(dataset.data(), int(dataset.size()));
    exp_chosen.cost.add = tc.get_timer_nanoSec() / double(dataset.size());
    exp_chosen.cost.memory = float(index->index_size()) / (1024 * 1024);
    auto tmp_e = e;
    tmp_e.seed(1000);
    dataset.resize(10e6);
    tc.synchronization();
    VALUE_TYPE value;
    for (int i=0;i<10e6;++i) {
        auto random_index=e() %  dataset.size();
        if (dataset[random_index].second != index->at(dataset[random_index].first, false)) {
            std::cout << "pgm get error:" << dataset[random_index].first << std::endl;
        }
    }
    exp_chosen.cost.get = (float) ((double) tc.get_timer_nanoSec() / ((double) dataset.size()));
    tc.synchronization();
    exp_chosen.cost.erase = (float) ((double) tc.get_timer_nanoSec() / ((double) dataset.size()));
    delete index;
    return exp_chosen.cost;
}

Cost b_tree_read_only(std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset) {
    experience_t exp_chosen;
    auto index = new stx::btree<KEY_TYPE, VALUE_TYPE>();
    index->bulk_load(dataset.begin(), dataset.end());
    exp_chosen.cost.memory =
            float(sizeof(stx::btree<KEY_TYPE, VALUE_TYPE>) +
                  index->m_stats.innernodes * sizeof(stx::btree<KEY_TYPE, VALUE_TYPE>::inner_node)
                  + index->m_stats.leaves *
                    sizeof(stx::btree<KEY_TYPE, VALUE_TYPE>::leaf_node)) / (1024 * 1024);
    tc.synchronization();
    exp_chosen.cost.add = tc.get_timer_nanoSec() / double(dataset.size());
    auto tmp_e = e;
    tmp_e.seed(1000);
    dataset.resize(10e6);
    tc.synchronization();
    VALUE_TYPE value;
    for (int i=0;i<10e6;++i) {
        auto random_index=e() %  dataset.size();
        if (dataset[random_index].second != index->find(dataset[random_index].first)->second) {
            std::cout << "b+tree get error:" << dataset[random_index].first << std::endl;
        }
    }
    exp_chosen.cost.get = (float) ((double) tc.get_timer_nanoSec() / ((double) dataset.size()));
    tc.synchronization();
    exp_chosen.cost.erase = (float) ((double) tc.get_timer_nanoSec() / ((double) dataset.size()));
    delete index;
    return exp_chosen.cost;
}


int length = 40e6;
int main() {
    controller.load_in();
    std::vector<std::string> dataset_names = scanFiles(data_father_path);
    std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset;
    auto exp_chosen = experience_t();
    std::ifstream tsk_file(father_path + "tsk_parameters.txt");
    std::ofstream result(father_path + "parameters_result.txt",std::ios::out| std::ios::binary);
    for(const auto& dataset_name:std::vector<std::string>({"uden.data","logn.data",})){
        result << "dataset_name:"<<dataset_name << std::endl;
        std::cout << "dataset_name:"<<dataset_name << std::endl;
        dataset = dataset_source::get_dataset<std::pair<KEY_TYPE, VALUE_TYPE>>(data_father_path + dataset_name);
        std::shuffle(dataset.begin(), dataset.end(),e);
        if(dataset.size() < length){
            throw MyException();
        }
        dataset.resize(length);
        std::sort(dataset.begin(), dataset.end(),
                  [&](std::pair<KEY_TYPE, VALUE_TYPE> &a,std::pair<KEY_TYPE, VALUE_TYPE> &b){return a.first < b.first;});
        controller.memory_weight = 1;
        controller.query_weight = 0.02;
        exp_chosen.cost = b_tree_read_only(dataset);
        std::cout << "b+tree query_weight:" << controller.query_weight << " cost :" << exp_chosen.cost << std::endl;
        exp_chosen.cost = alex_read_only(dataset);
        std::cout << "alex query_weight:" << controller.query_weight << " cost :" << exp_chosen.cost << std::endl;
        exp_chosen.cost = pgm_read_only(dataset);
        std::cout << "pgm query_weight:" << controller.query_weight << " cost :" << exp_chosen.cost << std::endl;
        exp_chosen.cost = lipp_read_only(dataset);
        std::cout << "lipp query_weight:" << controller.query_weight << " cost :" << exp_chosen.cost << std::endl;
//        continue;
        for(int i = 0;i<200;++i){
            exp_chosen.cost = cha_read_only(dataset, true);
            std::cout << "query_weight:" << controller.query_weight << " cost :" << exp_chosen.cost << std::endl;
            result << "hits:" << exp_chosen.cost << " query_weight:" << controller.query_weight<< " memory_weight:" << controller.memory_weight << std::endl;
            if(controller.query_weight > 20){
                controller.query_weight *= 1.1;
            }else{
                controller.query_weight *= 1.1;
            }
            controller.memory_weight *= 0.99;
        }
        puts("============================");
        result << "============================" << std::endl;
    }
}
