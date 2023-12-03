//
// Created by wenli on 2023/1/25.
//
#include <iostream>
#include<functional>
#include <iomanip>
#define count_insert_time
#ifdef count_insert_time
double insert_time_retrain = 0;
#endif
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

Cost hits_evaluation(const std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> &dataset,
                           bool using_model = true) {
    experience_t exp_chosen;
    auto min_max = get_min_max<KEY_TYPE,VALUE_TYPE>(dataset.begin(),dataset.end());
    auto pdf = get_pdf<KEY_TYPE,VALUE_TYPE>(dataset.begin(),dataset.end(),min_max.first,min_max.second,PDF_SIZE);
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
    TimerClock tcc;
    for (int i = 0; i < train_size; ++i) {
        if (!index->add(dataset[i].first, dataset[i].second)) {
            puts("hits add error !");
        }
    }
    insert_time_retrain = 0;
    tcc.synchronization();
    for (int i = train_size; i < int(dataset.size()); ++i) {
        if (!index->add(dataset[i].first, dataset[i].second)) {
            puts("hits add error !");
        }
    }
    exp_chosen.cost.add = (float) ((double) tcc.get_timer_nanoSec() / ((double) (dataset.size() - train_size)));
    std::cout <<"add:"<<exp_chosen.cost.add<< std::endl;
    std::cout <<"retrain:"<<insert_time_retrain * 1e9/ ((double) (dataset.size() - train_size))<< std::endl;
    return exp_chosen.cost;
    exp_chosen.cost.memory = index->memory_occupied()/float(1024*1024);
    tcc.synchronization();
    VALUE_TYPE value;
    for (int i = train_size; i < int(dataset.size()); ++i) {
        if (!index->get(dataset[i].first, value) || value != dataset[i].second) {
            std::cout << "hits get error:" << dataset[i].first << std::endl;
        }
    }
    exp_chosen.cost.get = (float) ((double) tcc.get_timer_nanoSec() / ((double) (dataset.size() - train_size)));
    tcc.synchronization();
    for (int i = train_size; i < int(dataset.size()); ++i) {
        if (!index->erase(dataset[i].first)) {
            puts("hits erase error !");
        }
    }
    exp_chosen.cost.erase = (float) ((double) tcc.get_timer_nanoSec() / ((double) (dataset.size() - train_size)));
    delete index;
    return exp_chosen.cost;
}


Cost alex_evaluation(const std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> &dataset) {
    experience_t exp_chosen;
    auto index = new alex::Alex<KEY_TYPE, VALUE_TYPE>();
    TimerClock tcc;
    index->bulk_load(dataset.data(), train_size);
    tcc.synchronization();
    insert_time_retrain = 0;
    for (int i = train_size; i < int(dataset.size()); ++i) {
        if (!index->insert(dataset[i].first, dataset[i].second).second) {
            puts("alex add error !");
        }
    }
    exp_chosen.cost.add = (float) ((double) tcc.get_timer_nanoSec() / ((double) (dataset.size() - train_size)));
    std::cout <<"add:"<<exp_chosen.cost.add<< std::endl;
    std::cout <<"retrain:"<<insert_time_retrain * 1e9/ ((double) (dataset.size() - train_size))<< std::endl;
    return exp_chosen.cost;
    exp_chosen.cost.memory = float( sizeof(alex::Alex<KEY_TYPE, VALUE_TYPE>) + index->data_size() + index->model_size()) / (1024 * 1024);
    tcc.synchronization();
    for (int i = train_size; i < int(dataset.size()); ++i) {
        if (dataset[i].second != *index->get_payload(dataset[i].first)) {
            std::cout << "alex get error:" << dataset[i].first << std::endl;
        }
    }
    exp_chosen.cost.get = (float) ((double) tcc.get_timer_nanoSec() / ((double) (dataset.size() - train_size)));
    tcc.synchronization();
    for (int i = train_size; i < int(dataset.size()); ++i) {
        if (!index->erase(dataset[i].first)) {
            puts("alex erase error !");
        }
    }
    exp_chosen.cost.erase = (float) ((double) tcc.get_timer_nanoSec() / ((double) (dataset.size() - train_size)));
    tcc.synchronization();
    delete index;
    return exp_chosen.cost;
}

Cost lipp_evaluation(const std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> &dataset) {
    experience_t exp_chosen;
    auto index = new LIPP<KEY_TYPE, VALUE_TYPE>();
    TimerClock tcc;
    index->bulk_load(dataset.data(), train_size);
    tcc.synchronization();
    insert_time_retrain = 0;
    for (int i = train_size; i < int(dataset.size()); ++i) {
        index->insert(dataset[i].first, dataset[i].second);
    }
    exp_chosen.cost.add = (float) ((double) tcc.get_timer_nanoSec() / ((double) (dataset.size() - train_size)));
    std::cout <<"add:"<<exp_chosen.cost.add<< std::endl;
    std::cout <<"retrain:"<<insert_time_retrain* 1e9 / ((double) (dataset.size() - train_size))<< std::endl;
    return exp_chosen.cost;
//    exp_chosen.cost.memory = float(index->index_size(true, false)) / (1024 * 1024);
    exp_chosen.cost.memory = float(index->index_size()) / (1024 * 1024);
    tcc.synchronization();
    for (int i = train_size; i < int(dataset.size()); ++i) {
        if (dataset[i].second != index->at(dataset[i].first,false)) {
            std::cout << "lipp get error:" << dataset[i].first << std::endl;
        }
    }
    exp_chosen.cost.get = (float) ((double) tcc.get_timer_nanoSec() / ((double) (dataset.size() - train_size)));
    tcc.synchronization();
    for (int i = train_size; i < int(dataset.size()); ++i) {
        if (!index->erase(dataset[i].first)) {
            puts("erase error !");
        }
    }
    exp_chosen.cost.erase = (float) ((double) tcc.get_timer_nanoSec() / ((double) (dataset.size() - train_size)));
    tcc.synchronization();
    delete index;
    return exp_chosen.cost;
}


Cost pgm_evaluation(const std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> &dataset) {
    experience_t exp_chosen;
    auto index = new pgm::DynamicPGMIndex<KEY_TYPE, VALUE_TYPE>(dataset.begin(), dataset.begin() + train_size);
    TimerClock tcc;
    tcc.synchronization();
    insert_time_retrain = 0;
    for (int i = train_size; i < int(dataset.size()); ++i) {
        index->insert_or_assign(dataset[i].first, dataset[i].second);
    }
    exp_chosen.cost.add = (float) ((double) tcc.get_timer_nanoSec() / ((double) (dataset.size() - train_size)));
    std::cout <<"add:"<<exp_chosen.cost.add<< std::endl;
    std::cout <<"retrain:"<<insert_time_retrain* 1e9 / ((double) (dataset.size() - train_size))<< std::endl;
    return exp_chosen.cost;
    exp_chosen.cost.memory = float(index->size_in_bytes()) / (1024 * 1024);
    tcc.synchronization();
    for (int i = train_size; i < int(dataset.size()); ++i) {
        if (dataset[i].second != index->find(dataset[i].first)->second) {
            std::cout << "pgm get error:" << dataset[i].first << std::endl;
        }
    }
    exp_chosen.cost.get = (float) ((double) tcc.get_timer_nanoSec() / ((double) (dataset.size() - train_size)));
    tcc.synchronization();
    for (int i = train_size; i < int(dataset.size()); ++i) {
        index->erase(dataset[i].first);
    }
    exp_chosen.cost.erase = (float) ((double) tcc.get_timer_nanoSec() / ((double) (dataset.size() - train_size)));
    tcc.synchronization();
    delete index;
    return exp_chosen.cost;
}

Cost b_tree_evaluation(const std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> &dataset) {
    experience_t exp_chosen;
    auto index = new stx::btree<KEY_TYPE, VALUE_TYPE>();
    TimerClock tcc;
    index->bulk_load(dataset.begin(), dataset.begin() + train_size);
    tcc.synchronization();
    insert_time_retrain = 0;
    for (int i = train_size; i < int(dataset.size()); ++i) {
        if (!index->insert(dataset[i].first, dataset[i].second).second) {
            std::cout << "b+tree insert error:" << dataset[i].first << std::endl;
        }
    }
    exp_chosen.cost.add = (float) ((double) tcc.get_timer_nanoSec() / ((double) (dataset.size() - train_size)));
    std::cout <<"add:"<<exp_chosen.cost.add<< std::endl;
    std::cout <<"retrain:"<<insert_time_retrain * 1e9 / ((double) (dataset.size() - train_size))<< std::endl;
    return exp_chosen.cost;
    exp_chosen.cost.memory =
            float(sizeof(stx::btree<KEY_TYPE, VALUE_TYPE>) + index->m_stats.innernodes * sizeof(stx::btree<KEY_TYPE, VALUE_TYPE>::inner_node)
                                   + index->m_stats.leaves *
                                     sizeof(stx::btree<KEY_TYPE, VALUE_TYPE>::leaf_node)) / (1024 * 1024);
    tcc.synchronization();
    for (int i = train_size; i < int(dataset.size()); ++i) {
        if (dataset[i].second != index->find(dataset[i].first)->second) {
            std::cout << "b+tree get error:" << dataset[i].first << std::endl;
        }
    }
    exp_chosen.cost.get = (float) ((double) tcc.get_timer_nanoSec() / ((double) (dataset.size() - train_size)));
    tcc.synchronization();
    for (int i = train_size; i < int(dataset.size()); ++i) {
        if (index->erase(dataset[i].first) != 1) {
            std::cout << "b+tree erase error !" << std::endl;
        }
    }
    exp_chosen.cost.erase = (float) ((double) tcc.get_timer_nanoSec() / ((double) (dataset.size() - train_size)));
    tcc.synchronization();
    delete index;
    return exp_chosen.cost;
}

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

int main() {
    TimerClock tcc;
    std::vector<std::string> dataset_names = scanFiles(data_father_path);
    std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset;
    auto exp_chosen = experience_t();
    if(IsFileExist((father_path + "add_get_erase/add_get_erase_result.txt").c_str())){
        std::time_t t = std::time(nullptr);
        std::stringstream string_s;
        string_s << std::put_time(std::localtime(&t), "%Y-%m-%d %H.%M.%S");
        rename((father_path + "add_get_erase/add_get_erase_result.txt").c_str(),(father_path +  "add_get_erase/add_get_erase_result"+string_s.str()+".txt").c_str());
    }
    std::ofstream result(father_path + "add_get_erase/add_get_erase_result.txt",std::ios::out| std::ios::binary);
    result << "train_proportion:" << train_proportion << "============================" << std::endl;
    std::vector<EvaluationTask> tasks;
    tasks.push_back({"uden.data",0,int(100e6)});
    tasks.push_back({"logn.data",0,int(100e6)});
    tasks.push_back({"osmc.data",0,int(100e6)});
    tasks.push_back({"face.data",0,int(100e6)});
    tasks.push_back({"local_skew.data",0,int(100e6)});
    for (auto tsk:tasks) {
        result << tsk.dataset_name <<"  start:"<< tsk.start <<"  length:"<< tsk.length << std::endl;
        dataset = dataset_source::get_dataset<std::pair<KEY_TYPE, VALUE_TYPE>>(data_father_path+tsk.dataset_name);
        auto tmp_e = e;
        tmp_e.seed(1000);
        std::shuffle(dataset.begin(), dataset.end(), tmp_e);
        dataset.erase(dataset.begin(), dataset.begin() + tsk.start);
        dataset.erase(dataset.begin() + tsk.length, dataset.end());
        std::cout << MAGENTA <<"dataset:"<<tsk.dataset_name<< " test_count:" << std::setw(4)
                  << double(dataset.size()) / double(1000000) << "*10**6" << RESET << std::endl;
        train_size = int(train_proportion * int(dataset.size()));
//        train_size = 0;
        std::sort(dataset.begin(), dataset.begin() + train_size);
        exp_chosen.cost = b_tree_evaluation(dataset);
        std::cout << "b+tree:" << exp_chosen.cost << std::endl;
        result << "b+tree:" << exp_chosen.cost << std::endl;
        exp_chosen.cost = alex_evaluation(dataset);
        std::cout << "alex:" << exp_chosen.cost << std::endl;
        result << "alex:" << exp_chosen.cost << std::endl;
        exp_chosen.cost = pgm_evaluation(dataset);
        std::cout << "pgm:" << exp_chosen.cost << std::endl;
        result << "pgm:" << exp_chosen.cost << std::endl;
        exp_chosen.cost = lipp_evaluation(dataset);
        std::cout << "lipp:" << exp_chosen.cost << std::endl;
        result << "lipp:" << exp_chosen.cost << std::endl;
        exp_chosen.cost = hits_evaluation(dataset, true);
        std::cout << "cha:" << exp_chosen.cost <<std::endl;
        result << "cha:" << exp_chosen.cost << std::endl;
        puts("============================");
        result << "============================" << std::endl;
    }
}
