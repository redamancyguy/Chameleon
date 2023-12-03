//
// Created by wenli on 2023/1/25.
//
#include <iostream>
#include <random>
#include<functional>
#include <iomanip>
#include <torch/torch.h>
#include "../include/DEFINE.h"

//#define CB
#define using_small_network

#include "../index/include/Index.hpp"
#include "../include/DataSet.hpp"
#include "../index/include/experience.hpp"
#include "../index/include/Controller.hpp"
#include "../others/alex/core/alex.h"
#include "../others/lipp/lipp.h"
#include "../others/pgm/pgm_index_dynamic.hpp"
#include "../others/b+tree/btree.h"
#include "../others/hpli/PreciseIndex.hpp"
#include "../others/rmi/rmi.hpp"
#include "../others/rmi/models.hpp"


std::vector<int> query_dis;

std::vector<int> Zipf_GenData(int n) {
    std::vector<int> result;
    std::random_device rd;
    std::array<int, std::mt19937::state_size> seed_data{};
    std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
    std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
    auto engine = std::mt19937{seq};
    std::vector<double> probabilities;
    const double a = 0.1;

    for (int i = 1; i <= n; i++) {
        probabilities.push_back(1.0 / pow(i, a));
    }

    std::discrete_distribution<> di(probabilities.begin(), probabilities.end());

    for (int i = 0; i < n; i++) {
        result.push_back(di(engine));
    }
    return result;
}

std::vector<int> Uniform_GenData(int n) {
    std::vector<int> result;
    auto tmp_e = e;
    tmp_e.seed(1000);
    for (int i = 0; i < n; i++) {
        result.push_back(tmp_e() % n);
    }
    return result;
}

Cost rmi_basic_evaluation(const std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> &dataset) {
    experience_t exp_chosen;
    std::vector<KEY_TYPE> keys;
    keys.reserve(dataset.size());
    for (auto i: dataset) {
        keys.push_back(i.first);
    }
    std::sort(keys.begin(), keys.end());
    using layer1_type = rmi::LinearSpline;
    using layer2_type = rmi::LinearRegression;
    std::size_t layer2_size = 2UL << 16;
    auto rmi = new rmi::RmiLAbs<KEY_TYPE, layer1_type, layer2_type>(keys, layer2_size);
    exp_chosen.cost.memory = float(sizeof(rmi::RmiLAbs<KEY_TYPE, layer1_type, layer2_type>) + rmi->size_in_bytes() +
                                   sizeof(std::pair<KEY_TYPE, VALUE_TYPE>) * dataset.size()) / (1024 * 1024);
    tc.synchronization();
    for (auto id: query_dis) {
        auto range = rmi->search(dataset[id].first);
        auto pos = std::lower_bound(keys.begin() + range.lo, keys.begin() + range.hi, dataset[id].first);
        if (*pos != dataset[id].first) {
            std::cout << "rmi get error !" << std::endl;
        }
    }
    exp_chosen.cost.get = (float) ((double) tc.get_timer_nanoSec() / ((double) dataset.size()));
    return exp_chosen.cost;
}

GlobalController controller;

Cost hits_basic_evaluation(std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset, bool using_model = true) {
    experience_t exp_chosen;
    auto min_max = get_min_max<KEY_TYPE, VALUE_TYPE>(dataset.begin(), dataset.end());
    auto pdf = get_pdf<KEY_TYPE, VALUE_TYPE>(dataset.begin(), dataset.end(), min_max.first,
                                             min_max.second,  BUCKET_SIZE);
    std::copy(pdf.begin(), pdf.end(), exp_chosen.distribution);
    exp_chosen.data_size = float(dataset.size());
    CHA::Configuration conf;
    if (using_model) {
//        puts("getting action !");
        auto best_gen = controller.get_best_action_GA(exp_chosen);
//        puts("got action !");
        conf = best_gen.conf;
//        std::cout <<"root:"<<conf.root_fan_out<< std::endl;
//        std::cout <<"inner:"<<conf.fan_outs[0][2]<< std::endl;
    } else {
        conf = CHA::Configuration::default_configuration();
    }
    exp_chosen.conf = conf;
    auto index = new CHA::Index<KEY_TYPE, VALUE_TYPE>(conf, min_max.first, min_max.second);
    index->bulk_load(dataset.begin(), dataset.end());
    exp_chosen.cost.memory = float(index->memory_occupied() / (1024. * 1024.));
    std::cout <<"cha inner size:"<<exp_chosen.cost.memory - double((double(dataset.size())/ default_density) * (double(sizeof(std::pair<KEY_TYPE, VALUE_TYPE>)) + 1.0/32.0)/ (1024. * 1024.)) << std::endl;
    auto tmp_e = e;
    tmp_e.seed(1000);
    tc.synchronization();
    VALUE_TYPE value;
    CHA::inner_cost = 0;
    CHA::leaf_cost = 0;
    for (auto id: query_dis) {
//        if (!index->get(dataset[id].first, value) || dataset[id].second != value) {
        if (!index->get_with_cost(dataset[id].first, value) || dataset[id].second != value) {
            std::cout << "cha get error:" << dataset[id].first << std::endl;
        }
    }
    std::cout << "avg height:" << double(CHA::inner_cost) / double(query_dis.size()) << std::endl;
    std::cout << "avg error:" << double(CHA::leaf_cost) / double(query_dis.size()) << std::endl;
    std::cout <<"max height:"<<3 << std::endl;
    std::cout << "max error:" << double(CHA::leaf_max_cost) << std::endl;
    exp_chosen.cost.get = (float) ((double) tc.get_timer_nanoSec() / ((double) query_dis.size()));
    auto count_node_result = index->count_node_of_each_layer();
    std::cout <<count_node_result<< std::endl;
    delete index;
    return exp_chosen.cost;
}


Cost alex_basic_evaluation(std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset) {
    experience_t exp_chosen;
    auto index = new alex::Alex<KEY_TYPE, VALUE_TYPE>();
    index->bulk_load(dataset.data(), int(dataset.size()));
    exp_chosen.cost.memory =
            float(sizeof(alex::Alex<KEY_TYPE, VALUE_TYPE>) + index->data_size() + index->model_size()) / (1024 * 1024);
    std::cout <<"alex inner size:"<<double(index->model_size())/(1024.0 * 1024.0)<< std::endl;
    auto tmp_e = e;
    tmp_e.seed(1000);
    tc.synchronization();
    for (auto id: query_dis) {
        if (dataset[id].second != *index->get_payload(dataset[id].first)) {
            std::cout << "alex get error:" << dataset[id].first << std::endl;
        }
    }
    exp_chosen.cost.get = (float) ((double) tc.get_timer_nanoSec() / ((double) query_dis.size()));
    std::cout <<"avg height:"<<index->count_level() + 1 << std::endl;
    std::cout <<"avg error:"<<double(alex::leaf_cost) / double(query_dis.size()) << std::endl;
    std::cout <<"max height:"<<double(alex::inner_max_cost + 1) << std::endl;
    std::cout <<"max error:"<<double(alex::leaf_max_cost) << std::endl;
    auto layer_nodes = index->count_node_of_each_layer();
    std::cout <<layer_nodes<< std::endl;
    delete index;
    return exp_chosen.cost;
}

Cost lipp_basic_evaluation(std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset) {
    experience_t exp_chosen;
    auto index = new LIPP<KEY_TYPE, VALUE_TYPE>();
    index->bulk_load(dataset.data(), int(dataset.size()));
    exp_chosen.cost.memory = float(index->index_size()) / (1024 * 1024);
    auto tmp_e = e;
    tmp_e.seed(1000);
    tc.synchronization();
    for (auto id: query_dis) {
        if (dataset[id].second != index->at(dataset[id].first, false)) {
            std::cout << "pgm get error:" << dataset[id].first << std::endl;
        }
    }
    exp_chosen.cost.get = (float) ((double) tc.get_timer_nanoSec() / ((double) query_dis.size()));
    delete index;
    return exp_chosen.cost;
}


Cost pgm_basic_evaluation(std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset) {
    experience_t exp_chosen;
    auto index = new pgm::DynamicPGMIndex<KEY_TYPE, VALUE_TYPE>(dataset.begin(), dataset.end());
    exp_chosen.cost.memory = float(index->size_in_bytes()) / (1024 * 1024);
    auto tmp_e = e;
    tmp_e.seed(1000);
    tc.synchronization();
    for (auto id: query_dis) {
        if (dataset[id].second != index->find(dataset[id].first)->second) {
            std::cout << "pgm get error:" << dataset[id].first << std::endl;
        }
    }
    exp_chosen.cost.get = (float) ((double) tc.get_timer_nanoSec() / ((double) query_dis.size()));
    delete index;
    return exp_chosen.cost;
}

Cost b_tree_basic_evaluation(std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset) {
    experience_t exp_chosen;
    auto index = new stx::btree<KEY_TYPE, VALUE_TYPE>();
    index->bulk_load(dataset.begin(), dataset.end());
    exp_chosen.cost.memory =
            float(sizeof(stx::btree<KEY_TYPE, VALUE_TYPE>) +
                  index->m_stats.innernodes * sizeof(stx::btree<KEY_TYPE, VALUE_TYPE>::inner_node)
                  + index->m_stats.leaves *
                    sizeof(stx::btree<KEY_TYPE, VALUE_TYPE>::leaf_node)) / (1024 * 1024);
    auto tmp_e = e;
    tmp_e.seed(1000);
    tc.synchronization();
    for (auto id: query_dis) {
        if (dataset[id].second != index->find(dataset[id].first)->second) {
            std::cout << "b+tree get error:" << dataset[id].first << std::endl;
        }
    }
    exp_chosen.cost.get = (float) ((double) tc.get_timer_nanoSec() / ((double) query_dis.size()));
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


void exp5(){

    controller.load_in();
    std::vector<std::string> dataset_names = scanFiles(data_father_path);
    std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset;
    auto exp_chosen = experience_t();
    std::string dis_type;
    std::ofstream result("buffer.txt", std::ios::out | std::ios::binary);
    for (int length: std::vector<float>({20e6, 40e6})) {
//        for(int length:std::vector<int>({200000,400000})){
        for (const auto &dataset_name: std::vector<std::string>(
                {"osmc.data","uden.data", "local_skew.data","face.data",})) {//"uden.data",  ,"wiki.data",
            dataset = dataset_source::get_dataset<std::pair<KEY_TYPE, VALUE_TYPE>>(data_father_path + dataset_name);
            if(dataset.size() > length){
                dataset.resize(length);
            }
            std::cout << "length:" << length<< " dis:" << dis_type;
            std::cout << " dataset_name:" << dataset_name << std::endl;
            result << "length:" << length << " dis:" << dis_type;
            result << " dataset_name:" << dataset_name << std::endl;
            std::sort(dataset.begin(), dataset.end(),
                      [&](std::pair<KEY_TYPE, VALUE_TYPE> &a, std::pair<KEY_TYPE, VALUE_TYPE> &b) {
                          return a.first < b.first;});
//            exp_chosen.cost = b_tree_basic_evaluation(dataset);
//            std::cout << "b+tree:" << exp_chosen.cost << std::endl;
//            result << "b+tree:" << exp_chosen.cost << std::endl;
//            exp_chosen.cost = rmi_basic_evaluation(dataset);
//            std::cout << "rmi:" << exp_chosen.cost << std::endl;
//            result << "rmi:" << exp_chosen.cost << std::endl;
//            exp_chosen.cost = alex_basic_evaluation(dataset);
//            std::cout << "alex:" << exp_chosen.cost << std::endl;
//            result << "alex:" << exp_chosen.cost << std::endl;
//            exp_chosen.cost = pgm_basic_evaluation(dataset);
//            std::cout << "pgm:" << exp_chosen.cost << std::endl;
//            result << "pgm:" << exp_chosen.cost << std::endl;
//            if(dataset_name != "wiki.data"){// && dataset_name != "local_skew.data"
//                exp_chosen.cost = lipp_basic_evaluation(dataset);
//            }else{
//                exp_chosen.cost = Cost();
//            }
//            std::cout << "lipp:" << exp_chosen.cost << std::endl;
//            result << "lipp:" << exp_chosen.cost << std::endl;
//            exp_chosen.cost = hits_basic_evaluation(dataset, false);
//            std::cout << "cha-index:" << exp_chosen.cost << std::endl;
//            result << "cha-index:" << exp_chosen.cost << std::endl;
            exp_chosen.cost = hits_basic_evaluation(dataset, true);
            std::cout << "cha:" << exp_chosen.cost << std::endl;
            result << "cha:" << exp_chosen.cost << std::endl;
            puts("============================");
            result << "============================" << std::endl;
        }
    }
}

int main() {
    controller.load_in();
    std::vector<std::string> dataset_names = scanFiles(data_father_path);
    std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset;
    auto exp_chosen = experience_t();
    std::ifstream tsk_file(father_path + "tsk.txt");
    if (IsFileExist((father_path + "read_only/read_only_result.txt").c_str())) {
        std::time_t t = std::time(nullptr);
        std::stringstream string_s;
        string_s << std::put_time(std::localtime(&t), "%Y-%m-%d %H.%M.%S");
        rename((father_path + "read_only/read_only_result.txt").c_str(),
               (father_path + "read_only/read_only_result" + string_s.str() + ".txt").c_str());
    }
    std::ofstream result(father_path + "read_only/read_only_result.txt", std::ios::out | std::ios::binary);
    std::string dis_type;
//    for (int dis = 0; dis < 2; dis++) {
//        for (int length: std::vector<float>({100e6, 200e6})) {

    for (int dis = 0; dis < 1; dis++) {
        for (int length: std::vector<float>({200e6})) {
            for (const auto &dataset_name: std::vector<std::string>(
                    {"osmc.data","face.data","uden.data", "local_skew.data","logn.data"})) {//"uden.data",  ,"wiki.data",
//                    {"logn.data",})) {//"uden.data",  ,"wiki.data",
                dataset = dataset_source::get_dataset<std::pair<KEY_TYPE, VALUE_TYPE>>(data_father_path + dataset_name);
                std::sort(dataset.begin(), dataset.end(),
                          [&](std::pair<KEY_TYPE, VALUE_TYPE> &a, std::pair<KEY_TYPE, VALUE_TYPE> &b) {
                              return a.first < b.first;});
                if(dataset.size() > length){
                    dataset.resize(length);
                }
                query_dis.clear();
                if (dis == 0) {
                    query_dis = Zipf_GenData(dataset.size());
                    dis_type = "zipf";
                    std::cout << "generated zipf query distribution" << std::endl;
                } else if (dis == 1) {
                    query_dis = Uniform_GenData(dataset.size());
                    dis_type = "uniform";
                    std::cout << "generated uniform query distribution" << std::endl;
                }
                std::cout << "length:" << length<< " dis:" << dis_type;
                std::cout << " dataset_name:" << dataset_name << std::endl;
                result << "length:" << length << " dis:" << dis_type;
                result << " dataset_name:" << dataset_name << std::endl;
                std::sort(dataset.begin(), dataset.end(),
                          [&](std::pair<KEY_TYPE, VALUE_TYPE> &a, std::pair<KEY_TYPE, VALUE_TYPE> &b) {
                              return a.first < b.first;});
//                exp_chosen.cost = b_tree_basic_evaluation(dataset);
//                std::cout << "b+tree:" << exp_chosen.cost << std::endl;
//                result << "b+tree:" << exp_chosen.cost << std::endl;
//                exp_chosen.cost = rmi_basic_evaluation(dataset);
//                std::cout << "rmi:" << exp_chosen.cost << std::endl;
//                result << "rmi:" << exp_chosen.cost << std::endl;
                exp_chosen.cost = alex_basic_evaluation(dataset);
                std::cout << "alex:" << exp_chosen.cost << std::endl;
                result << "alex:" << exp_chosen.cost << std::endl;
//                exp_chosen.cost = pgm_basic_evaluation(dataset);
//                std::cout << "pgm:" << exp_chosen.cost << std::endl;
//                result << "pgm:" << exp_chosen.cost << std::endl;
//                if(dataset_name != "wiki.data"){
//                    exp_chosen.cost = lipp_basic_evaluation(dataset);
//                }else{
//                    exp_chosen.cost = Cost();
//                }
//                std::cout << "lipp:" << exp_chosen.cost << std::endl;
//                result << "lipp:" << exp_chosen.cost << std::endl;
//                exp_chosen.cost = hits_basic_evaluation(dataset, false);
//                std::cout << "cha-index:" << exp_chosen.cost << std::endl;
//                result << "cha-index:" << exp_chosen.cost << std::endl;
                exp_chosen.cost = hits_basic_evaluation(dataset, true);
                std::cout << "cha:" << exp_chosen.cost << std::endl;
                result << "cha:" << exp_chosen.cost << std::endl;
                puts("============================");
                result << "============================" << std::endl;
            }
        }
    }
}

