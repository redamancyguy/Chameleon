//
// Created by redamancyguy on 23-3-23.
//
#include <iostream>
#include <functional>
#include <vector>
#include <iomanip>
#include <torch/torch.h>
#include "../../include/DEFINE.h"
#include "../include/experience.hpp"
#include "../../include/TimerClock.hpp"
#include "../../include/StandardScalar.hpp"
#include "../include/Controller.hpp"
#include "train.hpp"


class RunningStatus {
public:
    double random_rate = 1;
    long long exp_num = 0;
    char buffer[4096]{};
};

#include "../include/Index.hpp"
void gen_exp(GlobalController &controller, std::FILE *exp_file, int exp_batch, double random_rate,int pid) {
    TimerClock tc_speed;
    static std::vector<std::string> dataset_names = scanFiles(train_dataset_path);
    auto exp_chosen = experience_t();
    for (int _ = 0; _ < exp_batch; ++_) {
        Hits::inner_cost =0;
        Hits::leaf_cost =0;
        Hits::node_memory_count =0;
        int random_length = shrink_dataset_size(
                int(std::pow(random_u_0_1(), 1) *
                    double(max_data_set_size - min_data_set_size)) + min_data_set_size);
        auto dataset_name = dataset_names[e() % dataset_names.size()];
        auto dataset = dataset_source::get_dataset<std::pair<KEY_TYPE, VALUE_TYPE>>(
                train_dataset_path + dataset_name);
        random_length = std::min(random_length, int(float(dataset.size()) * float(0.9 + random_u_0_1() * 0.1)));
        auto random_start = (int) (e() % (dataset.size() - random_length));
        dataset.erase(dataset.begin(), dataset.begin() + random_start);
        dataset.erase(dataset.begin() + random_length, dataset.end());
        std::sort(dataset.begin(), dataset.end(),[=](std::pair<KEY_TYPE, VALUE_TYPE> &a,std::pair<KEY_TYPE, VALUE_TYPE> &b){return a.first < b.first;});
        auto min_max = get_min_max<KEY_TYPE,VALUE_TYPE>(dataset.begin(),dataset.end());
        auto pdf = get_pdf<KEY_TYPE,VALUE_TYPE>(dataset.begin(),dataset.end(), min_max.first, min_max.second,BUCKET_SIZE);
        std::copy(pdf.begin(),pdf.end(),exp_chosen.distribution);
        exp_chosen.data_size = float(dataset.size());
        exp_chosen.conf.root_fan_out = float(dataset.size());
        auto conf = controller.get_best_action_GA(exp_chosen).conf * float(1 - random_rate)
                    + Hits::Configuration::random_configuration() * float(random_rate);
        exp_chosen.conf = conf;
        std::cout << BLUE <<"pid:"<<pid<< "  speed : " << "   conf.fan_outs[0]:" << conf.fan_outs[0][0]
        << "   "<< 3600 / tc_speed.get_timer_second() << "/hour  " << RESET;
        tc_speed.synchronization();
        std::cout << MAGENTA << "  length:" << std::setw(4) << double(dataset.size()) / double(1000000) << "*10**6"
                  << "  root size:" << std::setw(10) << conf.root_fan_out << RESET << std::endl;
        auto index = new Hits::Index<KEY_TYPE, VALUE_TYPE>(conf, min_max.first, min_max.second);
        index->bulk_load(dataset.begin(),dataset.end());
        auto memory = index->memory_occupied() / float(float(dataset.size()) *(sizeof(Hits::DataNode<KEY_TYPE,VALUE_TYPE>::slot_type )));
        tc.synchronization();
        VALUE_TYPE value;
        for (auto &i: dataset) {
            if (!index->get_with_cost(i.first, value)) {
                std::cout << "get error:" << i.first << std::endl;
            }
        }
        auto get_cost = float(float(Hits::inner_cost) * inner_cost_weight + float(Hits::leaf_cost) * leaf_cost_weight) / ((float) dataset.size());
        exp_chosen.cost.memory = memory;
        exp_chosen.cost.get = get_cost;
        std::fwrite(&exp_chosen, sizeof(experience_t), 1, exp_file);
        delete index;
        std::cout <<RED<<"Cost:"<<exp_chosen.cost<<RESET<< "   memory:"<<Hits::node_memory_count<< std::endl;
    }
}


[[noreturn]] void
gen(int pid, int exp_batch) {
    auto *rs = get_shared_memory<RunningStatus>();
    int file_count = 0;
    GlobalController controller;
    auto file_num = rs->exp_num++;
    std::cout << "file num : " << file_num << std::endl;
    if (!access((experience_father_path + std::to_string(file_num) + ".exp").c_str(), F_OK)) {
        throw MyException("file exist !");
    }
    auto file = std::fopen((experience_father_path + std::to_string(file_num) + ".exp").c_str(), "w");
    for (int i = 0;;++i) {
#ifdef using_semaphore
        sem_wait(pai_semaphore);
#endif
        controller.load_in();

#ifdef using_semaphore
        usleep(1000);
        sem_post(pai_semaphore);
#endif
        gen_exp(controller, file, exp_batch, rs->random_rate,pid);
        std::cout << "exp_count:" << exp_batch * (file_count++) << std::endl;
    }
    std::fclose(file);
}
