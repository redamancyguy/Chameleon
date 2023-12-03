//
// Created by redamancyguy on 23-5-18.
//
#include <iostream>
#include <unistd.h>
#include <semaphore.h>
#include <fcntl.h>
#include <sys/mman.h>

class RunningStatus {
public:
    double random_rate = 1;
    long long exp_num = 0;
    char buffer[4096]{};
};

#include "../../include/DEFINE.h"
#include "../include/Parameter.h"
#include "../include/Controller.hpp"
#include "../include/Index.hpp"
#include "../include/experience.hpp"
#include <c10/cuda/CUDAGuard.h>
//#define CB
//#define using_small_network

#define sample_batch 5
#define using_model

[[noreturn]] void sample(int pid){
    auto *rs = get_shared_memory<RunningStatus>();
    std::vector<experience_t> exps_local;
#ifdef using_model
    GlobalController controller;
#endif
    auto exp_chosen = experience_t();
    TimerClock tc_speed;
    std::vector<std::string> dataset_names = scanFiles(data_father_path);
    auto file = std::fopen((experience_father_path + std::to_string(rs->exp_num++) + ".exp").c_str(), "w");
    while(true){
        double random_rate = rs->random_rate;
#ifdef using_model
        controller.load_in();
#endif
        for (int _ = 0; _ < sample_batch; ++_) {
#ifdef using_model
            controller.random_weight();
#endif
            CHA::inner_cost = 0;
            CHA::leaf_cost = 0;
            int random_length = shrink_dataset_size(
                    int(std::pow(random_u_0_1(), 1) *
                        double(max_data_set_size - min_data_set_size)) + min_data_set_size);
            auto dataset_name = dataset_names[e() % dataset_names.size()];
            auto dataset = dataset_source::get_dataset<std::pair<KEY_TYPE, VALUE_TYPE>>(
                    data_father_path + dataset_name);
            std::shuffle(dataset.begin(), dataset.end(),e);
            random_length = std::min(random_length, int(float(dataset.size()) * float(0.9 + random_u_0_1() * 0.1)));
            auto random_start = (int) (e() % (dataset.size() - random_length));
            std::sort(dataset.begin() + random_start, dataset.begin() + random_start + random_length,
                      [=](std::pair<KEY_TYPE, VALUE_TYPE> &a,std::pair<KEY_TYPE, VALUE_TYPE> &b){return a.first < b.first;});
            auto min_max = get_min_max<KEY_TYPE,VALUE_TYPE>(dataset.begin() + random_start, dataset.begin() + random_start + random_length);
            auto pdf = get_pdf<KEY_TYPE,VALUE_TYPE>(dataset.begin() + random_start, dataset.begin() + random_start + random_length, min_max.first, min_max.second,BUCKET_SIZE);
            std::copy(pdf.begin(),pdf.end(),exp_chosen.distribution);
            exp_chosen.data_size = float(random_length);
#ifdef using_model
//            auto conf = controller.get_best_action_GA(exp_chosen).conf * float(1 - random_rate)
//                        + Hits::Configuration::random_configuration() * float(random_rate);
            auto conf = CHA::Configuration::random_configuration();
#else
            auto conf = Hits::Configuration::random_configuration();
#endif
            exp_chosen.conf = conf;
            auto index = new CHA::Index<KEY_TYPE, VALUE_TYPE>(conf, min_max.first, min_max.second);
            index->bulk_load(dataset.begin() + random_start, dataset.begin() + random_start + random_length);
            auto memory = index->memory_occupied() / float(float(random_length) *(sizeof(CHA::DataNode<KEY_TYPE,VALUE_TYPE>::slot_type)));
            VALUE_TYPE value;
            CHA::inner_cost = 0;
            for (auto i=dataset.begin() + random_start,
                    end=dataset.begin() + random_start + random_length;i < end;++i) {
                if (!index->get_with_cost(i->first, value)) {
                    std::cout << "get error:" << i->first << std::endl;
                }
            }
            auto get_cost = float(float(CHA::inner_cost) * inner_cost_weight + float(CHA::leaf_cost) * leaf_cost_weight) / ((float) random_length);
            exp_chosen.cost.memory = memory;
            exp_chosen.cost.get = get_cost;
            delete index;
            std::cout <<"memory:query:"<<controller.memory_weight<<":"<<controller.query_weight;
            std::cout << BLUE <<" pid:"<<pid << " speed : " << "   "<< 3600 / tc_speed.get_timer_second() << "/hour  " << RESET;
            tc_speed.synchronization();
            std::cout  << MAGENTA << "  length:" << std::setw(4) << double(random_length) / double(1000000) << "*10**6"
                       << "  root size:" << std::setw(10) << conf.root_fan_out
                       <<"  inner size:"<<conf.fan_outs[0][0]<<" "<<conf.fan_outs[0][INNER_FANOUT_COLUMN/2]<< RESET << std::endl;
            std::cout <<RED<<"pid:"<<pid<<"   Cost:"<<exp_chosen.cost<<RESET<< std::endl;
            exps_local.push_back(exp_chosen);
        }
        for(auto i:exps_local){
            std::fwrite(&i, sizeof(experience_t), 1, file);
        }
        exps_local.clear();
    }
}

[[noreturn]] [[noreturn]] void train(){
    auto q_model = std::make_shared<Global_Q_network>(Global_Q_network());
    q_model->to(GPU_DEVICE);
    q_model->train();
    auto q_optimizer = torch::optim::Adam(q_model->parameters(),
                                          torch::optim::AdamOptions(train_lr).
                                                  weight_decay(train_wd));
    int break_times = 10;
    ExpGenerator exp_g;
    RewardScalar scalar(2);
    for(int steps = 0;;++steps){
        float loss_count = 0;
        for (int flush_count = 0;flush_count < break_times; flush_count++) {
            for (int i = 0; i < train_steps; ++i) {
                auto exp_tensor_batch = exp_g.exp_batch(BATCH_SIZE);
                auto pdf = std::get<0>(exp_tensor_batch).to(GPU_DEVICE);
                auto value = std::get<1>(exp_tensor_batch).to(GPU_DEVICE);
                auto root_fanout = std::get<2>(exp_tensor_batch).to(GPU_DEVICE);
                auto inner_fanout = std::get<3>(exp_tensor_batch).to(GPU_DEVICE);
                auto reward = std::get<4>(exp_tensor_batch).to(GPU_DEVICE);
                scalar.forward_and_fit(reward);
                auto pred = q_model->forward(pdf, value, root_fanout, inner_fanout);
                pred = scalar.inverse(pred);
                auto loss = torch::nn::L1Loss()->forward(pred,reward);
                q_optimizer.zero_grad();
                loss.backward();
                q_optimizer.step();
                fflush(stdout);
                tc.synchronization();
                loss_count += loss.to(CPU_DEVICE).item().toFloat();
                printf("loss:%f\r",loss.to(CPU_DEVICE).item().toFloat());
            }
        }
        std::cout << GREEN <<"avg loss:"<<float(loss_count / float(train_steps * break_times))<<RESET<<std::endl;
        q_model->to(CPU_DEVICE);
        torch::save(q_model, q_model_path);
        scalar.save(q_scalar_path);
        q_model->to(GPU_DEVICE);
        sleep(3);
    }
}
int main(int argc, char const *argv[]) {

    double random_rate_discount_rate = 0.9997;
//    double random_rate_discount_rate = 0.993;
    auto *rs = create_shared_memory<RunningStatus>();
    rs->random_rate = 1;
    std::cout << "rs->random_rate:" <<rs->random_rate << std::endl;
    const int process_count = 4;
    const int process_count2= 5;
//    clear_exp(experience_father_path,scanFiles(experience_father_path));
//    remove(q_model_path.c_str());
    rs->exp_num = max_exp_number(scanFiles(experience_father_path)) + 1;

    int pid = 0;
//    sample(pid);
    GPU_DEVICE = torch::Device(torch::DeviceType::CUDA,1);
    for ( auto i = 0; i < process_count; i++) {
        random_seed();
        if (fork() == 0) {
            random_seed();
            sample(pid);
        }
        ++pid;
        std::cout << "gen pid : " << pid << std::endl;
        usleep(100000);
        random_seed();
    }
    random_seed();
    GPU_DEVICE = torch::Device(torch::DeviceType::CUDA,0);
    for (auto i = 0; i < process_count2; i++) {
        random_seed();
        if (fork() == 0) {
            random_seed();
            sample(pid);
        }
        ++pid;
        std::cout << "gen pid : " << pid << std::endl;
        usleep(100000);
        random_seed();
    }
    for(auto sample_size=count_exp(scanFiles(experience_father_path)) ;sample_size< 100;sample_size = count_exp(scanFiles(experience_father_path))){
        std::cout <<"waiting for more samples !  --->"<<sample_size<<std::endl;
        sleep(1);
    }
    if(fork() == 0){
        puts("start training !");
        train();
    }

    int samples_num = 0;
    while(rs->random_rate > 3e-3){
        if(samples_num * sample_batch < count_exp(scanFiles(experience_father_path))){
            rs->random_rate *= random_rate_discount_rate;
            ++samples_num;
            std::cout <<"rs->random_rate:"<<rs->random_rate<< std::endl;
            continue;
        }
        std::cout << "  rr:" << rs->random_rate << "  samples:" << samples_num * sample_batch << std::endl;
        sleep(10);
    }
    return 0;
}