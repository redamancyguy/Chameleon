//
// Created by redamancyguy on 23-8-24.
//

#include <queue>
#include <string>
#include "../include/Parameter.h"
#include "../../include/DataSet.hpp"
#include "../include/Index.hpp"
#include "../include/RL_network.hpp"

class ExpSmall{
public:
    float pdf[SMALL_PDF_SIZE]{};
    int dataset_size = 0;
    int action = 0;
    float reward = 0;
};

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor> exp_small_to_tensor(const std::vector<ExpSmall>& exps){
    auto result = std::make_tuple(
            torch::rand({int(exps.size()), SMALL_PDF_SIZE}),
            torch::rand({int(exps.size()), VALUE_SIZE}),
            torch::rand({int(exps.size()), 1}),
            torch::rand({int(exps.size()), 1})
    );
    auto pdf_ptr = std::get<0>(result).data_ptr<float>();
    auto value_ptr = std::get<1>(result).data_ptr<float>();
    auto action_ptr = std::get<2>(result).data_ptr<float>();
    auto reward_ptr = std::get<3>(result).data_ptr<float>();
    for (auto &i:exps) {
//        for(auto j:i.pdf){
//            if(std::isnan(j)){
//                throw MyException("nan value exits !");
//            }
//        }
        std::copy(i.pdf, i.pdf + SMALL_PDF_SIZE, pdf_ptr);
        pdf_ptr += SMALL_PDF_SIZE;
        ///////////////////////////
        value_ptr[0] = float(i.dataset_size);
        value_ptr += VALUE_SIZE;
        ////////////////////////////
        action_ptr[0] = float(i.action);
        action_ptr += 1;
        reward_ptr[0] = i.reward;
        reward_ptr += 1;
    }
    return result;
}



std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor> exp_small_to_tensor(ExpSmall& exp){
    auto result = std::make_tuple(
            torch::rand({1, SMALL_PDF_SIZE}),
            torch::rand({1, VALUE_SIZE}),
            torch::rand({1, 1}),
            torch::rand({1, 1})
    );
    auto pdf_ptr = std::get<0>(result).data_ptr<float>();
    auto value_ptr = std::get<1>(result).data_ptr<float>();
    auto action_ptr = std::get<2>(result).data_ptr<float>();
    auto reward_ptr = std::get<3>(result).data_ptr<float>();
    auto &i = exp;
    std::copy(i.pdf, i.pdf + SMALL_PDF_SIZE, pdf_ptr);
    pdf_ptr += SMALL_PDF_SIZE;
    ///////////////////////////
    value_ptr[0] = float(i.dataset_size);
    value_ptr += VALUE_SIZE;
    ////////////////////////////
    action_ptr[0] = float(i.action);
    action_ptr += 1;
    reward_ptr[0] = i.reward;
    reward_ptr += REWARD_SIZE;
    return result;
}

#define dataset_size_for_tree 200000
//#define dataset_size_for_tree 5000000
#define memory_weight float(0.5)

float random_fanout(){
    return std::pow(2,random_u_0_1() * 8);
}

std::ofstream log_file("logfile.txt");
int main(){
    GPU_DEVICE = torch::Device(torch::DeviceType::CUDA,0);
    auto q_network = std::make_shared<Small_Q_network>();
    q_network->to(GPU_DEVICE);
    auto q_optimizer = torch::optim::Adam(q_network->parameters());
    q_network->train();
    auto q_target_network = std::make_shared<Small_Q_network>();
    q_target_network->to(GPU_DEVICE);
    q_target_network->eval();
    auto pai_network = std::make_shared<Small_PAI_network>();
    pai_network->to(GPU_DEVICE);
    pai_network->eval();
    float random_rate = 1;
    if(0){
        random_rate = 0.15;
        torch::load(q_network,model_father_path+"tmp.pt");
        torch::load(q_target_network,model_father_path+"tmp.pt");
        q_target_network->to(GPU_DEVICE);
        q_target_network->eval();
        q_target_network->to(GPU_DEVICE);
        q_target_network->train();
    }
    float discount_rate = 0.9;
    std::vector<std::string> dataset_names = scanFiles(train_dataset_path);
    std::vector<std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>> exp_tensors;
    for(int tree_id = 0;;++tree_id){

        std::cout <<"tree id:"<<tree_id<<"  random rate:"<<random_rate<< std::endl;
        auto tmp_exp_vector = std::vector<std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>>();
        tmp_exp_vector.reserve(exp_tensors.size());
        std::vector<int> shuffle_index;
        shuffle_index.reserve(exp_tensors.size());
        for(int i = 0;i < int(exp_tensors.size());++i){
            shuffle_index.push_back(i);
        }
        std::shuffle(shuffle_index.begin(), shuffle_index.end(),e);
        if(shuffle_index.size() > dataset_size_for_tree * 10){
            shuffle_index.resize(dataset_size_for_tree * 10);
        }
        for(auto i:shuffle_index){
            tmp_exp_vector.push_back(std::move(exp_tensors[i]));
        }
        exp_tensors = std::move(tmp_exp_vector);
        std::unordered_map<int ,int> id_to_father_id;
        std::unordered_map<int ,std::vector<int>> father_id_to_ids;
        std::unordered_map<int ,ExpSmall> id_to_exp;
        std::unordered_map<int ,void *> id_to_node;
        auto dataset = dataset_source::get_dataset<std::pair<KEY_TYPE, VALUE_TYPE>>(train_dataset_path + dataset_names[tree_id % dataset_names.size()]);
        auto random_length = std::min(int(dataset_size_for_tree * random_u_0_1_skew(0.2)), int(dataset.size() - 1));
        random_length = std::max(DATA_NODE_SIZE,random_length);
        std::cout <<"random_length:"<<random_length<< std::endl;
        auto random_start = (int) (e() % (dataset.size() - random_length));
        dataset.erase(dataset.begin(), dataset.begin() + random_start);
        dataset.erase(dataset.begin() + random_length, dataset.end());
        std::sort(dataset.begin(), dataset.end(),[=](std::pair<KEY_TYPE, VALUE_TYPE> &a,std::pair<KEY_TYPE, VALUE_TYPE> &b){return a.first < b.first;});
        struct Task{
            int node_id;
            int start;
            int stop;
            double lower;
            double upper;
        };
        std::queue<Task> tasks;
        CHA::Index<KEY_TYPE,VALUE_TYPE> *index;
        {
            auto min_max = get_min_max<KEY_TYPE, VALUE_TYPE>(
                    dataset.begin(),dataset.end());
            index = new CHA::Index<KEY_TYPE,VALUE_TYPE>(CHA::Configuration::default_configuration(), min_max.first, min_max.second);
            index->delete_tree(index->root);
            CHA::InnerNode<KEY_TYPE,VALUE_TYPE>::delete_segment(index->root);
            tasks.push({-1,0,int(dataset.size()),min_max.first,min_max.second});
        }
        int son_id = -1;
        while(!tasks.empty()){
            auto task = tasks.front();
            tasks.pop();
            ExpSmall &exp = id_to_exp[task.node_id];
            auto pdf = get_pdf<KEY_TYPE, VALUE_TYPE>(
                    dataset.begin() + task.start, dataset.begin() + task.stop,
                    task.lower, task.upper, SMALL_PDF_SIZE);
            std::copy(pdf.begin(),pdf.end(),exp.pdf);
            exp.dataset_size = task.stop - task.start;
            if(random_u_0_1() > random_rate){
                auto tensor_for_action = exp_small_to_tensor(exp);
                exp.action = 0;
                auto a = std::get<0>(tensor_for_action).mul(torch::ones({int(action_space.size()),std::get<0>(tensor_for_action).size(1)})).to(GPU_DEVICE);
                auto b = std::get<1>(tensor_for_action).mul(torch::ones({int(action_space.size()),std::get<1>(tensor_for_action).size(1)})).to(GPU_DEVICE);
                auto c =   torch::tensor(action_space).view({-1,1}).to(GPU_DEVICE);
                auto pred = q_target_network->forward(a,b,c);
                pred = torch::softmax(pred,0);
                pred = pred.to(CPU_DEVICE);
                std::vector<float> probability(pred.data_ptr<float>(),pred.data_ptr<float>() + pred.numel());
                auto p = random_u_0_1();
                for(int i = 0;i<int(probability.size());++i){
                    p -= probability[i];
                    if(p < 0){
                        exp.action = i;
                        break;
                    }
                }
            }else{
                exp.action = int(e() % action_space.size());
            }
            if(exp.dataset_size == 0){
                exp.action = 0;
            }
            if(exp.action > 0 && exp.dataset_size > DATA_NODE_SIZE){
                auto last_memory = CHA::node_memory_count;
                auto inner_node = CHA::InnerNode<KEY_TYPE,VALUE_TYPE>::new_segment(1 << int(exp.action), task.lower, task.upper);
                id_to_node[task.node_id] = inner_node;
                exp.reward = -QUERY_WEIGHT * inner_cost_weight - MEMORY_WEIGHT * float(CHA::node_memory_count - last_memory) / sizeof(CHA::DataNode<KEY_TYPE,VALUE_TYPE>::slot_type) / float(std::max(1, exp.dataset_size));
                auto result = CHA::split_dataset<KEY_TYPE,VALUE_TYPE>(dataset.begin() + task.start, dataset.begin() + task.stop, inner_node);
                for(auto const &i:result){
                    tasks.push({++son_id,i.first.first + task.start,i.first.second + task.start,i.second.first,i.second.second});//sub tasks son_id code
                    id_to_father_id[son_id] = task.node_id;
                }
            } else{
                auto last_memory = CHA::node_memory_count;
                auto data_node = CHA::DataNode<KEY_TYPE,VALUE_TYPE>::new_segment(std::max(DATA_NODE_SIZE, int(float(std::max(1, exp.dataset_size)) / default_density)), task.lower, task.upper);
                id_to_node[task.node_id] = data_node;
                for(auto i = task.start;i<task.stop;++i){
                    auto position = data_node->find_insert(dataset[i].first);
                    if(position < 0){throw MyException("bad position!");}
                    data_node->array[position].first = dataset[i].first;
                    data_node->array[position].second = dataset[i].second;
                    set_bitmap(data_node->bitmap_start(),position);
                    ++data_node->size;
                }
                CHA::leaf_cost = 0;
                for(auto i = task.start;i<task.stop;++i){
                    auto position = data_node->find_with_cost(dataset[i].first);
                    if(position < 0){ throw MyException("bad position!");}
                }
                exp.reward = -QUERY_WEIGHT * inner_cost_weight - float(double(CHA::leaf_cost) / double(std::max(exp.dataset_size, 1))) * leaf_cost_weight
                             - MEMORY_WEIGHT * float(CHA::node_memory_count - last_memory) / sizeof(CHA::DataNode<KEY_TYPE,VALUE_TYPE>::slot_type) / float(std::max(1, exp.dataset_size));
            }
        }
        for(auto const &i:id_to_father_id){
            father_id_to_ids[i.second].push_back(i.first);
        }
        for(auto &i:father_id_to_ids){
            std::sort(i.second.begin(), i.second.end());
        }
        index->root = (CHA::InnerNode<KEY_TYPE,VALUE_TYPE>*)id_to_node[-1];
        if(father_id_to_ids[-1].empty()){
            index->is_leaf = true;
        }
        for(auto const &i:father_id_to_ids){
            if(i.first != -1 && 1 << int(id_to_exp[i.first].action) != int(i.second.size())){
                throw MyException("bad sub ids size !");
            }
            if(i.first != -1 && ((CHA::InnerNode<KEY_TYPE,VALUE_TYPE>*)id_to_node[i.first])->capacity != int(i.second.size())){
                throw MyException("bad capacity !");
            }
            int size_count = 0;
            for(auto j:i.second){
                if(id_to_exp.find(j) == id_to_exp.end()){
                    throw MyException("bad finding !");
                }
                size_count += id_to_exp[j].dataset_size;
            }
            if(i.first != -1 && id_to_exp[i.first].dataset_size != size_count){
                throw MyException("un equal dataset size !");
            }
            auto inner_node = (CHA::InnerNode<KEY_TYPE,VALUE_TYPE>*)id_to_node[i.first];
            if(i.first != -1 && inner_node->capacity != int(i.second.size())){
                throw MyException("bad size");
            }
            for(int j = 0;j<int(i.second.size());++j) {
                if (father_id_to_ids.find(i.second[j]) != father_id_to_ids.end()) {
                    set_bitmap(inner_node->bitmap_start(),j);
                    auto sub_node = (CHA::InnerNode<KEY_TYPE, VALUE_TYPE> *) id_to_node[i.second[j]];
                    inner_node->array[j].inner_node = sub_node;
                } else {
                    if (id_to_node.find(i.second[j]) == id_to_node.end()) {
                        throw MyException("not find data node in dict !");
                    }
                    de_set_bitmap(inner_node->bitmap_start(),j);
                    auto data_node = (CHA::DataNode<KEY_TYPE, VALUE_TYPE> *) id_to_node[i.second[j]];
                    inner_node->array[j].data_node = data_node;
                    if (data_node->size != id_to_exp[i.second[j]].dataset_size) {
                        std::cout << data_node->size << ":" << id_to_exp[i.second[j]].dataset_size << std::endl;
                        throw MyException("bad data node size !");
                    }
                }
            }
        }
        CHA::inner_cost = 0;
        CHA::leaf_cost = 0;
        tc.synchronization();
        for(auto i:dataset){
            VALUE_TYPE  value;
            if(!index->get_with_root_leaf(i.first, value) || value != i.second){
                throw MyException("bad get result !");
            }
        }

        log_file << double(CHA::inner_cost) / double(dataset.size()) << "," << double(CHA::leaf_cost) / double(dataset.size()) << "," << tc.get_timer_nanoSec() / double(dataset.size()) << std::endl;
        std::cout <<"dataset:"<<dataset.size()<< std::endl;
        std::cout << BLUE << "average inner cost:" << double(CHA::inner_cost) / double(dataset.size()) << RESET << std::endl;
        std::cout << BLUE << "average leaf cost:" << double(CHA::leaf_cost) / double(dataset.size()) << RESET << std::endl;
        std::cout << BLUE << "average memory:" << double(CHA::node_memory_count / sizeof(CHA::DataNode<KEY_TYPE,VALUE_TYPE>::slot_type)) / double(dataset.size()) << RESET << std::endl;
        sleep(1);
        delete index;
        {
            for(auto i:id_to_node){
                auto node_id = i.first;
                auto u = torch::Tensor();
                if(int(father_id_to_ids[node_id].size()) != 0){
                    std::vector<ExpSmall> son_exps;
                    std::vector<float> exp_dataset_size;
                    for(auto j:father_id_to_ids[node_id]){
                        son_exps.push_back(id_to_exp[j]);
                        exp_dataset_size.push_back(float(son_exps.back().dataset_size));
                    }
                    auto son_tensors = exp_small_to_tensor(son_exps);
                    std::vector<torch::Tensor> tmp_son_result;
                    for(int action = 0;action < int(action_space.size());++action){
                        auto a = std::get<0>(son_tensors).to(GPU_DEVICE);
                        auto b = std::get<1>(son_tensors).to(GPU_DEVICE);
                        auto c = torch::tensor({action}).mul(torch::ones({int(son_exps.size()),1})).to(GPU_DEVICE);
                        auto son_u = q_target_network->forward(a,b,c);
                        tmp_son_result.push_back(son_u.view({son_u.size(0),son_u.size(1),1}));
                    }
                    auto all_son_result = torch::cat(tmp_son_result,2);//n 1 action
                    auto p_for_action = torch::softmax(all_son_result,2);
                    all_son_result = all_son_result.mul(p_for_action).sum(2);
                    all_son_result *= torch::tensor(exp_dataset_size).view({-1,1}).to(GPU_DEVICE);
                    all_son_result /= float(id_to_exp[node_id].dataset_size);
                    all_son_result = all_son_result.sum(0);
                    u = all_son_result.view({-1,1}).detach().to(CPU_DEVICE);
                }else{
                    u = torch::zeros({1,1});
                }
                auto tensor_exp = exp_small_to_tensor(id_to_exp[node_id]);
                exp_tensors.emplace_back(std::get<0>(tensor_exp),std::get<1>(tensor_exp),std::get<2>(tensor_exp),std::get<3>(tensor_exp),u);
            }
        }
        q_network->to(GPU_DEVICE);
        float loss_count = 0;

        for(int k = 0;k < 100 ;++k){
            std::vector<torch::Tensor> pdf_batch;
            std::vector<torch::Tensor> value_batch;
            std::vector<torch::Tensor> action_batch;
            std::vector<torch::Tensor> reward_batch;
            std::vector<torch::Tensor> u_batch;
            for(int i = 0;i<BATCH_SIZE;++i){
                auto &draw_sample = exp_tensors[e() % exp_tensors.size()];
                pdf_batch.push_back(std::get<0>(draw_sample));
                value_batch.push_back(std::get<1>(draw_sample));
                action_batch.push_back(std::get<2>(draw_sample));
                reward_batch.push_back(std::get<3>(draw_sample));
                u_batch.push_back(std::get<4>(draw_sample));
            }
            auto a_batch = torch::vstack(pdf_batch).detach().to(GPU_DEVICE);
            auto b_batch = torch::vstack(value_batch).detach().to(GPU_DEVICE);
            auto c_batch = torch::vstack(action_batch).detach().to(GPU_DEVICE);
            auto d_batch = torch::vstack(reward_batch).detach().to(GPU_DEVICE);
            auto e_batch = torch::vstack(u_batch).detach().to(GPU_DEVICE);
            auto pred = q_network->forward(a_batch,b_batch ,c_batch);
            auto target = d_batch +  e_batch * discount_rate;
            auto loss = torch::nn::L1Loss()->forward(pred,target);
            q_optimizer.zero_grad();
            loss.backward();
            q_optimizer.step();
            loss_count += loss.item().toFloat();
        }
        std::cout <<RED<<"loss_count:"<<loss_count / float(100)<<"  nodes:"<<id_to_node.size()<<RESET<< std::endl;
        if(tree_id > 3){
            int update_frequency = 10;
            if(tree_id % update_frequency == update_frequency-1){
                std::cout <<GREEN<<"update q network "<<RESET<< std::endl;
                q_network->to(CPU_DEVICE);
                torch::save(q_network,model_father_path+"tmp.pt");
                torch::load(q_target_network,model_father_path+"tmp.pt");
                q_target_network->to(GPU_DEVICE);
                q_network->to(GPU_DEVICE);
            }
            random_rate *= 0.9995;
        }
        if(random_rate < 3e-3){
            break;
        }
    }
}
