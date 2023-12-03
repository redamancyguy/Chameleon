//
// Created by redamancyguy on 23-7-27.
//

#ifndef HITS_DYNAMIC_HPP
#define HITS_DYNAMIC_HPP
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
#include "../others/alex/core/alex.h"
#include "../others/lipp/lipp.h"
#include "../others/pgm/pgm_index_dynamic.hpp"
#include "../others/b+tree/btree.h"
#include "../others/hpli/PreciseIndex.hpp"

double train_proportion = 0.1;
auto train_size = 0;
double write_times = 1;
double read_times = 10;

GlobalController controller;

Cost hits_basic_evaluation(const std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> &dataset,
                           bool using_model = true) {
    experience_t exp_chosen{};
    auto min_max = get_min_max<KEY_TYPE,VALUE_TYPE>(dataset.begin(),dataset.end());
    auto pdf = get_pdf<KEY_TYPE,VALUE_TYPE>(dataset.begin(), dataset.end(), min_max.first, min_max.second,BUCKET_SIZE);
    std::copy(pdf.begin(),pdf.end(),exp_chosen.distribution);
    exp_chosen.data_size = float(train_size);
    CHA::Configuration conf;
    if (using_model) {
        auto best_gen = controller.get_best_action_GA(exp_chosen);
        conf = best_gen.conf;
    } else {
        conf = CHA::Configuration::default_configuration();
        conf.root_fan_out = float(train_size);
    }
    exp_chosen.conf = conf;
    auto index = new CHA::Index<KEY_TYPE, VALUE_TYPE>(conf, min_max.first, min_max.second);
    TimerClock tc;
    index->bulk_load(dataset.begin(),dataset.begin() + train_size);
//    for (int i = 0; i < train_size; ++i) {
//        if (!index->add(dataset[i].first, dataset[i].second)) {
//            puts("hits add error !");
//        }
//    }
    tc.synchronization();
    long long opt_count = 0;
    int insert_cursor = train_size;
    while (insert_cursor < int(dataset.size())) {
        VALUE_TYPE value;
        for (int i = 0; i < read_times; ++i) {
            opt_count++;
            auto random_index = int(e() % train_size) + (insert_cursor - train_size);
            if (!index->get(dataset[random_index].first, value)
                || value != dataset[random_index].second) {
                std::cout << "hits get error:" << i << std::endl;
            }
        }
        for (int i = 0; i < write_times; ++i) {
            opt_count++;
            if (!index->erase(dataset[insert_cursor - train_size + i].first)) {
                puts("hits erase error !");
            }
        }
        for (int i = 0; i < write_times && insert_cursor < int(dataset.size()); ++i, ++insert_cursor) {
            opt_count++;
            if (!index->add(dataset[insert_cursor].first, dataset[insert_cursor].second)) {
                puts("hits add error !");
            }
        }
    }
    exp_chosen.cost.add = float(opt_count / (double) tc.get_timer_second());
    exp_chosen.cost.memory = index->memory_occupied() / (1024 * 1024);
    delete index;
    return exp_chosen.cost;
}


Cost alex_basic_evaluation(const std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> &dataset) {
    experience_t exp_chosen;
    auto index = new alex::Alex<KEY_TYPE, VALUE_TYPE>();
    TimerClock tc;
    index->bulk_load(dataset.data(), train_size);
    tc.synchronization();
    int insert_cursor = train_size;
    while (insert_cursor < int(dataset.size())) {
        for (int i = 0; i < read_times; ++i) {
            auto random_index = int(e() % train_size) + (insert_cursor - train_size);
            if (dataset[random_index].second != *index->get_payload(dataset[random_index].first)) {
                std::cout << "alex get error:" << i << std::endl;
            }
        }
        for (int i = 0; i < write_times; ++i) {
            if (!index->erase(dataset[insert_cursor - train_size + i].first)) {
                puts("alex erase error !");
            }
        }
        for (int i = 0; i < write_times && insert_cursor < int(dataset.size()); ++i, ++insert_cursor) {
            if (!index->insert(dataset[insert_cursor].first, dataset[insert_cursor].second).second) {
                puts("alex add error !");
            }
        }
    }    exp_chosen.cost.add = (float) ((2 + (double(read_times) / double(write_times))) *
                                        double(dataset.size() - train_size) / (double) tc.get_timer_second());
    exp_chosen.cost.memory = float( sizeof(alex::Alex<KEY_TYPE, VALUE_TYPE>) + index->data_size() + index->model_size()) / (1024 * 1024);

    delete index;
    return exp_chosen.cost;
}

Cost lipp_basic_evaluation(const std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> &dataset) {
    experience_t exp_chosen;
    auto index = new LIPP<KEY_TYPE, VALUE_TYPE>();
    TimerClock tc;
    index->bulk_load(dataset.data(), train_size);
    tc.synchronization();
    int insert_cursor = train_size;
    while (insert_cursor < int(dataset.size())) {
        for (int i = 0; i < read_times; ++i) {
            auto random_index = int(e() % train_size) + (insert_cursor - train_size);
            if (index->at(dataset[random_index].first, false) != dataset[random_index].second) {
                std::cout << "lipp get error:" << i << std::endl;
            }
        }
        for (int i = 0; i < write_times; ++i) {
            if (!index->erase(dataset[insert_cursor - train_size + i].first)) {
                puts("lipp erase error !");
            }
        }
        for (int i = 0; i < write_times && insert_cursor < int(dataset.size()); ++i, ++insert_cursor) {
            index->insert(dataset[insert_cursor].first, dataset[insert_cursor].second);
        }
    }  exp_chosen.cost.add = (float) ((2 + (double(read_times) / double(write_times))) *
                                      double(dataset.size() - train_size) / (double) tc.get_timer_second());
    exp_chosen.cost.memory = float(index->index_size()) / (1024 * 1024);
    delete index;
    return exp_chosen.cost;
}


Cost pgm_basic_evaluation(const std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> &dataset) {
    experience_t exp_chosen;
    auto index = new pgm::DynamicPGMIndex<KEY_TYPE, VALUE_TYPE>(dataset.begin(), dataset.begin() + train_size);
    TimerClock tc;
    tc.synchronization();
    int insert_cursor = train_size;
    while (insert_cursor < int(dataset.size())) {
        for (int i = 0; i < read_times; ++i) {
            auto random_index = int(e() % train_size) + (insert_cursor - train_size);
            if (dataset[random_index].second != index->find(dataset[random_index].first)->second) {
                std::cout << "pgm get error:" << i << std::endl;
            }
        }
        for (int i = 0; i < write_times; ++i) {
            index->erase(dataset[insert_cursor - train_size + i].first);
        }
        for (int i = 0; i < write_times && insert_cursor < int(dataset.size()); ++i, ++insert_cursor) {
            index->insert_or_assign(dataset[insert_cursor].first, dataset[insert_cursor].second);
        }
    }    exp_chosen.cost.add = (float) ((2 + (double(read_times) / double(write_times))) *
                                        double(dataset.size() - train_size) / (double) tc.get_timer_second());
    exp_chosen.cost.memory = float(index->size_in_bytes()) / (1024 * 1024);
    delete index;
    return exp_chosen.cost;
}

Cost b_tree_basic_evaluation(const std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> &dataset) {
    experience_t exp_chosen;
    auto index = new stx::btree<KEY_TYPE, VALUE_TYPE>();
    TimerClock tc;
    index->bulk_load(dataset.begin(), dataset.begin() + train_size);
    tc.synchronization();
    int insert_cursor = train_size;
    while (insert_cursor < int(dataset.size())) {
        for (int i = 0; i < read_times; ++i) {
            auto random_index = int(e() % train_size) + (insert_cursor - train_size);
            if (dataset[random_index].second != index->find(dataset[random_index].first)->second) {
                std::cout << "b+tree get error:" << i << std::endl;
            }
        }
        for (int i = 0; i < write_times; ++i) {
            if (index->erase(dataset[insert_cursor - train_size + i].first) != 1) {
                std::cout << "b+tree erase error !" << std::endl;
            }
        }
        for (int i = 0; i < write_times && insert_cursor < int(dataset.size()); ++i, ++insert_cursor) {
            if (!index->insert(dataset[insert_cursor].first, dataset[insert_cursor].second).second) {
                std::cout << "b+tree insert error:" << dataset[i].first << std::endl;
            }
        }
    }  exp_chosen.cost.add = (float) ((2 + (double(read_times) / double(write_times))) *
                                      double(dataset.size() - train_size) / (double) tc.get_timer_second());
    exp_chosen.cost.memory =
            float(sizeof(stx::btree<KEY_TYPE, VALUE_TYPE>) + index->m_stats.innernodes * sizeof(stx::btree<KEY_TYPE, VALUE_TYPE>::inner_node)
                  + index->m_stats.leaves *
                    sizeof(stx::btree<KEY_TYPE, VALUE_TYPE>::leaf_node)) / (1024 * 1024);

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
    auto ts = std::size_t(double(dataset.size()) * proportion.first / (proportion.first + proportion.second));
    std::vector<std::size_t> indices(dataset.size());
    for (std::size_t i = 0; i < dataset.size(); i++) {
        indices[i] = i;
    }
    std::shuffle(indices.begin(), indices.end(), e);
    std::vector<T> train_dataset;
    train_dataset.reserve(ts);
    std::vector<T> test_dataset;
    test_dataset.reserve(dataset.size() - ts);
    for (std::size_t i = 0; i < ts; i++) {
        train_dataset.push_back(dataset[indices[i]]);
    }
    for (std::size_t i = ts; i < dataset.size(); i++) {
        test_dataset.push_back(dataset[indices[i]]);
    }
    return {train_dataset, test_dataset};
}

#endif //HITS_DYNAMIC_HPP
