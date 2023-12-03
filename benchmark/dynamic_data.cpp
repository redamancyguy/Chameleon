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
#include "../others/alex/core/alex.h"
#include "../others/lipp/lipp.h"
#include "../others/pgm/pgm_index_dynamic.hpp"
#include "../others/b+tree/btree.h"

int train_size;

GlobalController controller;
int add_times, erase_times;

Cost hits_basic_evaluation(std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset, bool using_model = true) {
    experience_t exp_chosen{};
    auto min_max = get_min_max<KEY_TYPE, VALUE_TYPE>(dataset.begin(), dataset.end());
    auto pdf = get_pdf<KEY_TYPE, VALUE_TYPE>(dataset.begin(), dataset.end(), min_max.first, min_max.second,
                                             BUCKET_SIZE);
    std::copy(pdf.begin(), pdf.end(), exp_chosen.distribution);
    exp_chosen.data_size = float(train_size);
    CHA::Configuration conf;
    if (using_model) {
        auto best_gen = controller.get_best_action_GA(exp_chosen);
        conf = best_gen.conf;
    } else {
        conf = CHA::Configuration::default_configuration();
    }
    exp_chosen.conf = conf;
    auto index = new CHA::Index<KEY_TYPE, VALUE_TYPE>(conf, min_max.first, min_max.second);
    index->bulk_load(dataset.begin(), dataset.begin() + train_size);
    std::shuffle(dataset.begin(), dataset.begin() + train_size, e);
    tc.synchronization();
    long long opt_count = 0;
    int insert_cursor = train_size;
    int erase_cursor = 0;
    int read_times = erase_times + add_times;
    while (insert_cursor < int(dataset.size())) {
        VALUE_TYPE value;
        for (int i = 0; i < read_times; ++i) {
            opt_count++;
            auto random_index = int(e() % (insert_cursor - erase_cursor)) + erase_cursor;
            if (!index->get(dataset[random_index].first, value)
                || value != dataset[random_index].second) {
                std::cout << "hits get error:" << i << std::endl;
            }
        }
        for (int i = 0; i < erase_times; ++i) {
            opt_count++;
            if (!index->erase(dataset[erase_cursor++].first)) {
                puts("hits erase error !");
            }
            if (erase_cursor >= insert_cursor) {
                goto END;
            }
        }
        for (int i = 0; i < add_times && insert_cursor < int(dataset.size()); ++i, ++insert_cursor) {
            opt_count++;
            if (!index->add(dataset[insert_cursor].first, dataset[insert_cursor].second)) {
                puts("hits add error !");
            }
        }
    }
    END:
    exp_chosen.cost.add = float(opt_count / (double) tc.get_timer_second());
    exp_chosen.cost.memory = index->memory_occupied() / (1024 * 1024);
    delete index;
    return exp_chosen.cost;
}



Cost alex_basic_evaluation(const std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> &dataset) {
    experience_t exp_chosen;
    auto index = new alex::Alex<KEY_TYPE, VALUE_TYPE>();
    index->bulk_load(dataset.data(), train_size);
    tc.synchronization();
    long long opt_count = 0;
    int insert_cursor = train_size;
    int erase_cursor = 0;
    int read_times = erase_times + add_times;
    while (insert_cursor < int(dataset.size())) {
        for (int i = 0; i < read_times; ++i) {
            opt_count++;
            auto random_index = int(e() % (insert_cursor - erase_cursor)) + erase_cursor;
            if (index->find(dataset[random_index].first).payload() != dataset[random_index].second) {
                std::cout << "alex get error:" << i << std::endl;
            }
        }
        for (int i = 0; i < erase_times; ++i) {
            opt_count++;
            if (!index->erase(dataset[erase_cursor++].first)) {
                puts("alex erase error !");
            }
            if (erase_cursor >= insert_cursor) {
                goto END;
            }
        }
        for (int i = 0; i < add_times && insert_cursor < int(dataset.size()); ++i, ++insert_cursor) {
            opt_count++;
            if (!index->insert(dataset[insert_cursor].first, dataset[insert_cursor].second).second) {
                puts("alex add error !");
            }
        }
    }
    END:
    exp_chosen.cost.add = float(opt_count / (double) tc.get_timer_second());
    exp_chosen.cost.memory = float( sizeof(alex::Alex<KEY_TYPE, VALUE_TYPE>) + index->data_size() + index->model_size()) / (1024 * 1024);

    delete index;
    return exp_chosen.cost;
}


Cost lipp_basic_evaluation(const std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> &dataset) {
    experience_t exp_chosen;
    auto index = new LIPP<KEY_TYPE, VALUE_TYPE>();
    index->bulk_load(dataset.data(), train_size);
    tc.synchronization();
    long long opt_count = 0;
    int insert_cursor = train_size;
    int erase_cursor = 0;
    int read_times = erase_times + add_times;
    while (insert_cursor < int(dataset.size())) {
        VALUE_TYPE value;
        for (int i = 0; i < read_times; ++i) {
            opt_count++;
            auto random_index = int(e() % (insert_cursor - erase_cursor)) + erase_cursor;
            if (index->at(dataset[random_index].first) != dataset[random_index].second) {
                std::cout << "hits get error:" << i << std::endl;
            }
        }
        for (int i = 0; i < erase_times; ++i) {
            opt_count++;
            if (!index->erase(dataset[erase_cursor++].first)) {
                puts("hits erase error !");
            }
            if (erase_cursor >= insert_cursor) {
                goto END;
            }
        }
        for (int i = 0; i < add_times && insert_cursor < int(dataset.size()); ++i, ++insert_cursor) {
            opt_count++;
            index->insert(dataset[insert_cursor].first, dataset[insert_cursor].second);
        }
    }
    END:
    exp_chosen.cost.add = float(opt_count / (double) tc.get_timer_second());
    exp_chosen.cost.memory = float(index->index_size()) / (1024 * 1024);

    delete index;
    return exp_chosen.cost;
}


Cost pgm_basic_evaluation(const std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> &dataset) {
    experience_t exp_chosen;
    auto index = new pgm::DynamicPGMIndex<KEY_TYPE, VALUE_TYPE>(dataset.begin(), dataset.begin() + train_size);
    tc.synchronization();
    long long opt_count = 0;
    int insert_cursor = train_size;
    int erase_cursor = 0;
    int read_times = erase_times + add_times;
    while (insert_cursor < int(dataset.size())) {
        VALUE_TYPE value;
        for (int i = 0; i < read_times; ++i) {
            opt_count++;
            auto random_index = int(e() % (insert_cursor - erase_cursor)) + erase_cursor;
            if (index->find(dataset[random_index].first)->second != dataset[random_index].second) {
                std::cout << "hits get error:" << i << std::endl;
            }
        }
        for (int i = 0; i < erase_times; ++i) {
            opt_count++;
            index->erase(dataset[erase_cursor++].first);
            if (erase_cursor >= insert_cursor) {
                goto END;
            }
        }
        for (int i = 0; i < add_times && insert_cursor < int(dataset.size()); ++i, ++insert_cursor) {
            opt_count++;
            index->insert_or_assign(dataset[insert_cursor].first, dataset[insert_cursor].second);
        }
    }
    END:
    exp_chosen.cost.add = float(opt_count / (double) tc.get_timer_second());
    exp_chosen.cost.memory = float(index->size_in_bytes()) / (1024 * 1024);

    delete index;
    return exp_chosen.cost;
}


Cost b_tree_basic_evaluation(const std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> &dataset) {
    experience_t exp_chosen;
    auto index = new stx::btree<KEY_TYPE, VALUE_TYPE>();
    index->bulk_load(dataset.begin(), dataset.begin() + train_size);
    tc.synchronization();
    long long opt_count = 0;
    int insert_cursor = train_size;
    int erase_cursor = 0;
    int read_times = erase_times + add_times;
    while (insert_cursor < int(dataset.size())) {
        VALUE_TYPE value;
        for (int i = 0; i < read_times; ++i) {
            opt_count++;
            auto random_index = int(e() % (insert_cursor - erase_cursor)) + erase_cursor;
            if (index->find(dataset[random_index].first)->second != dataset[random_index].second) {
                std::cout << "hits get error:" << i << std::endl;
            }
        }
        for (int i = 0; i < erase_times; ++i) {
            opt_count++;
            if(index->erase(dataset[erase_cursor++].first) == 0){
                puts("b+tree insert error !");
            }
            if (erase_cursor >= insert_cursor) {
                goto END;
            }
        }
        for (int i = 0; i < add_times && insert_cursor < int(dataset.size()); ++i, ++insert_cursor) {
            opt_count++;
            if(!index->insert(dataset[insert_cursor].first, dataset[insert_cursor].second).second){
                puts("b+tree insert error!");
            }
        }
    }
    END:
    exp_chosen.cost.add = float(opt_count / (double) tc.get_timer_second());
    exp_chosen.cost.memory =float(sizeof(stx::btree<KEY_TYPE, VALUE_TYPE>) + index->m_stats.innernodes * sizeof(stx::btree<KEY_TYPE, VALUE_TYPE>::inner_node)
                                  + index->m_stats.leaves *
                                    sizeof(stx::btree<KEY_TYPE, VALUE_TYPE>::leaf_node)) / (1024 * 1024);

    delete index;
    return exp_chosen.cost;
}


int main() {
    controller.load_in();
    std::vector<std::string> dataset_names = scanFiles(data_father_path);
    std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset;
    auto exp_chosen = experience_t();
    std::vector<std::pair<int, int>> add_erase_proportions = {
            {0, 5},
            {1, 4},
            {2, 3},
            {3, 2},
            {4, 1},
            {5, 0},

    };
    std::ofstream result(father_path + "dynamic_2_result/add_erase_result_2.txt");
//    for (const auto &dataset_name: std::vector<std::string>({ "osmc.data", "wiki.data","face.data", "logn.data",})) {
    for (const auto &dataset_name: std::vector<std::string>({  "osmc.data","face.data", "logn.data","uden.data",})) {
        dataset = dataset_source::get_dataset<std::pair<KEY_TYPE, VALUE_TYPE>>(data_father_path + dataset_name);
        std::shuffle(dataset.begin(), dataset.end(), e);
        dataset.erase(dataset.begin() + 40000000, dataset.end());
        train_size = int(0.5 * int(dataset.size()));
        std::sort(dataset.begin(), dataset.begin() + train_size);
        for (auto pp: add_erase_proportions) {
            add_times = pp.first;
            erase_times = pp.second;
            std::cout << MAGENTA << "dataset:" << dataset_name << " train_rate:" << std::setw(4) << pp << RESET << std::endl;
            result << "dataset:" << dataset_name << " train_rate:"<< pp << std::endl;
//            exp_chosen.cost = b_tree_basic_evaluation(dataset);
//            std::cout << "b+tree:" << exp_chosen.cost << std::endl;
//            result << "b+tree:" << exp_chosen.cost << std::endl;
            exp_chosen.cost = alex_basic_evaluation(dataset);
            std::cout << "alex:" << exp_chosen.cost << std::endl;
            result << "alex:" << exp_chosen.cost << std::endl;
//            exp_chosen.cost = pgm_basic_evaluation(dataset);
//            std::cout << "pgm:" << exp_chosen.cost << std::endl;
//            result << "pgm:" << exp_chosen.cost << std::endl;
//            exp_chosen.cost = lipp_basic_evaluation(dataset);
//            std::cout << "lipp:" << exp_chosen.cost << std::endl;
//            result << "lipp:" << exp_chosen.cost << std::endl;
//            exp_chosen.cost = hits_basic_evaluation(dataset, true);
//            std::cout << "cha:" << exp_chosen.cost << std::endl;
//            result << "cha:" << exp_chosen.cost << std::endl;
        }
    }
}
