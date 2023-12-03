
//
// Created by wenli on 2023/1/25.
//
#include <iostream>
#include<functional>
#include <iomanip>
#include "../include/DEFINE.h"
#include "../include/DataSet.hpp"
#include "../index/include/experience.hpp"
#include "../index/include/Controller.hpp"
#include "../others/alex/core/alex.h"
#include "../others/lipp/lipp.h"
#include "../others/pgm/pgm_index_dynamic.hpp"
#include "../others/b+tree/btree.h"
#include "../index/include/Index.hpp"
double train_proportion = 0.1;
auto train_size = 0;


GlobalController controller;

Cost hits_evaluation(const std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> &dataset, bool using_model = true) {
    experience_t exp_chosen;
    auto min_max = get_min_max<KEY_TYPE, VALUE_TYPE>(dataset.begin(), dataset.end());
    auto pdf = get_pdf<KEY_TYPE, VALUE_TYPE>(dataset.begin(), dataset.end(), min_max.first, min_max.second, PDF_SIZE);
    exp_chosen.data_size = float(dataset.size());
    CHA::Configuration conf;
    if (using_model) {
        auto best_gen = controller.get_best_action_GA(exp_chosen);
        conf = best_gen.conf;
    } else {
        conf = CHA::Configuration::default_configuration();
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
    exp_chosen.cost.memory = index->memory_occupied() / float(1024 * 1024);
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


Cost alex_evaluation(const std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> &dataset) {
    experience_t exp_chosen;
    auto index = new alex::Alex<KEY_TYPE, VALUE_TYPE>();
    TimerClock tc;
    index->bulk_load(dataset.data(), train_size);
    tc.synchronization();
    for (int i = train_size; i < int(dataset.size()); ++i) {
        if (!index->insert(dataset[i].first, dataset[i].second).second) {
            puts("alex add error !");
        }
    }
    exp_chosen.cost.add = (float) ((double) tc.get_timer_nanoSec() / ((double) (dataset.size() - train_size)));
    exp_chosen.cost.memory = float( sizeof(alex::Alex<KEY_TYPE, VALUE_TYPE>) + index->data_size() + index->model_size()) / (1024 * 1024);
    tc.synchronization();
    for (int i = train_size; i < int(dataset.size()); ++i) {
        if (dataset[i].second != *index->get_payload(dataset[i].first)) {
            std::cout << "alex get error:" << dataset[i].first << std::endl;
        }
    }
    exp_chosen.cost.get = (float) ((double) tc.get_timer_nanoSec() / ((double) (dataset.size() - train_size)));
    tc.synchronization();
    for (int i = train_size; i < int(dataset.size()); ++i) {
        if (!index->erase(dataset[i].first)) {
            puts("alex erase error !");
        }
    }
    exp_chosen.cost.erase = (float) ((double) tc.get_timer_nanoSec() / ((double) (dataset.size() - train_size)));
    tc.synchronization();
    delete index;
    return exp_chosen.cost;
}



int main() {
    controller.load_in();
    TimerClock tc;
    std::vector<std::string> dataset_names = scanFiles(data_father_path);
    std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset;
    auto exp_chosen = experience_t();
    auto osm_data = load_osm_binary_data<std::pair<std::pair<double, double>, long>>(
            "/home/redamancyguy/Desktop/buffer/data/osm_data/asia-latest.bin",400e6,600e6);
    std::unordered_map<double,long> set;
    for(int i = 0;i<osm_data.size();++i){
        set[osm_data[i % 100000].first.first * 1e-6 + osm_data[i].first.second] = osm_data[i].second;
        if(set.size() > 200e6){
            puts("OK");
            break;
        }
    }

    e.seed(1000);
    for(const auto& dataset_name:std::vector<std::string>({"osmc.data","uden.data","wiki.data","face.data","logn.data",})){
        for(auto dataset_size:std::vector<float>({40e6,80e6,120e6,160e6,200e6})){

//            dataset = dataset_source::get_dataset<std::pair<KEY_TYPE, VALUE_TYPE>>(data_father_path + "local_skew.data");
            dataset.clear();
            for(auto &i:set){
                dataset.push_back({i.first,i.second});
            }
//            dataset.clear();
//            for(auto &dd:osm_data){
//                dataset.push_back({dd.first.first,dd.first.second});
//            }

            auto tmp_e = e;
            tmp_e.seed(1000);
            std::shuffle(dataset.begin(), dataset.end(), tmp_e);
            if(dataset.size() >= dataset_size){
                dataset.resize(dataset_size);
            }
            std::cout << dataset_name <<"  dataset_size:"<< dataset.size() << std::endl;
            std::cout << MAGENTA << "test_count:" << std::setw(4)
                      << double(dataset.size()) / double(1000000) << "*10**6" << RESET << std::endl;
            train_size = int(train_proportion * int(dataset.size()));
            std::sort(dataset.begin(), dataset.begin() + train_size);

            exp_chosen.cost = hits_evaluation(dataset, true);
            std::cout << "hits:" << exp_chosen.cost << std::endl;
            exp_chosen.cost = alex_evaluation(dataset);
            std::cout << "alex:" << exp_chosen.cost << std::endl;
            puts("============");
        }
    }
}
