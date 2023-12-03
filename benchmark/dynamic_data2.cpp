//
// Created by redamancyguy on 23-7-27.
//
#include "dynamic.hpp"


int main() {
    controller.load_in();
    std::vector <std::string> dataset_names = scanFiles(data_father_path);
    std::vector <std::pair<KEY_TYPE, VALUE_TYPE>> dataset;
    auto exp_chosen = experience_t();
    std::vector <std::pair<int, int>> proportions = {{1, 4},
                                                     {2, 3},
                                                     {3, 2},
                                                     {4, 1},
                                                     {5, 0}};

    std::ofstream result(father_path + "dynamic_2_result/dynamic_data_result_2.txt");
    for (const auto& dataset_name:std::vector<std::string>({ "uden.data","local_skew.data"})) {
        for (auto pp: proportions) {
            write_times = pp.first;
            read_times = pp.second;
            train_proportion = 0.3;
            result << dataset_name << " w/r ratio:"<< (float(write_times) / float(write_times + read_times)) << std::endl;
            dataset = dataset_source::get_dataset < std::pair < KEY_TYPE, VALUE_TYPE >> (data_father_path + dataset_name);
            std::shuffle(dataset.begin(), dataset.end(),e);
            std::cout << MAGENTA << "this train_rate:" << std::setw(4)
                      << double(train_proportion) << RESET << " " <<dataset_name<< "  w/r ratio:"
                      << (float(write_times) / float(write_times + read_times)) << std::endl;
            train_size = int(train_proportion * int(dataset.size()));
            std::sort(dataset.begin(), dataset.begin() + train_size);
            exp_chosen.cost = b_tree_basic_evaluation(dataset);
            std::cout << "b+tree:" << exp_chosen.cost << std::endl;
            result << "b+tree:" << exp_chosen.cost << std::endl;
            exp_chosen.cost = alex_basic_evaluation(dataset);
            std::cout << "alex:" << exp_chosen.cost << std::endl;
            result << "alex:" << exp_chosen.cost << std::endl;
            exp_chosen.cost = pgm_basic_evaluation(dataset);
            std::cout << "pgm:" << exp_chosen.cost << std::endl;
            result << "pgm:" << exp_chosen.cost << std::endl;
            exp_chosen.cost = lipp_basic_evaluation(dataset);
            std::cout << "lipp:" << exp_chosen.cost << std::endl;
            result << "lipp:" << exp_chosen.cost << std::endl;
            exp_chosen.cost = hits_basic_evaluation(dataset, true);
            std::cout << "cha:" << exp_chosen.cost << std::endl;
            result << "cha:" << exp_chosen.cost << std::endl;
            puts("============================");
            result << "============================" << std::endl;
        }
    }
    return 0;
}