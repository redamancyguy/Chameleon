#include <iostream>
#include <thread>
#include <csignal>
#include "include/DEFINE.h"
#include "include/DataSet.hpp"
#include "index/include/Parameter.h"
#include "index/include/Index.hpp"

[[noreturn]] void output(){
    for(int i = 0;;++i){
        std::cout <<i<< std::endl;
        sleep(1);
    }
}
int main0() {
    for(int i = 0;i<100;++i){
        auto random_length = std::min(int(200000 * random_u_0_1_skew(0.2)),200000000);
        std::cout<<random_length<<std::endl;
    }
    return 0;
    while(true){

        std::thread th1(output);
        th1.detach();
        getchar();
//        pthread_cancel(th1.get_id());
    }
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
#include <stdexcept>
void fc_200M(){
    auto dataset = dataset_source::get_dataset<unsigned long long>("/media/redamancyguy/mobile_disk/new_dataset/fb_200M_uint64");
    std::cout <<dataset.size()<< std::endl;
    std::set<double> set(dataset.begin(),dataset.end());
    std::cout <<"set:"<<set.size()<< std::endl;
    return;
    std::vector<std::pair<double,double>> new_dataset;
    for(auto i :dataset){
        new_dataset.push_back({i,i});
    }
    dataset_source::set_dataset(data_father_path + "fb_200M_uint64.data",new_dataset);
}
void main1(){
    std::vector<std::pair<double,double>> dataset;
//    std::ifstream in_file("/media/redamancyguy/mobile_disk/new_dataset/NYCTaxi.txt");
    std::ifstream in_file("/media/redamancyguy/mobile_disk/new_dataset/TON_IoT.txt");
    std::set<double> sets;
    char buffer[128];
    while(!in_file.eof()){
        in_file.getline(buffer,128);
        auto str = std::string(buffer);
        auto offset = str.find(':');
        if(offset >=128){
            break;
        }
        auto str1 = str.substr(0,offset);
        auto str2 = str.substr(offset + 1,str.length());
//        std::cout <<str1<< std::endl;
//        std::cout <<str2<< std::endl;
        if(str1 == "nan"){
            continue;
        }
//        auto left = std::stod(str1);
//        auto right = std::stod(str2);
        try
        {

            auto left = std::stol(str1);
            auto right = std::stol(str2);
            sets.insert(left);
            dataset.emplace_back(left,right);
        }

        catch (std::invalid_argument& e)
        {
            std::cout <<str1<<" : "<<str1.length()<< std::endl;
            std::cout <<str2<< std::endl;
            std::cerr << e.what() << std::endl;
        }

    }
    std::cout <<"sets.size():"<<sets.size()<< std::endl;
}

std::string source_path = "/media/redamancyguy/storage/experiment_data/20230713/new_dataset/data_set/";
void generate_dataset(){
    std::vector<std::string> train_set_names = scanFiles(source_path);
    std::unordered_map<double,double> new_set;
    new_set.reserve(201 * 1e6);
    std::unordered_map<std::string,std::vector<std::pair<double,double>>> sets;
    while(new_set.size() < 200 * 1e6){
        auto ss = train_set_names[e() % train_set_names.size()];
        std::vector<std::pair<double,double>> *dataset;
        if(sets.find(ss) != sets.end()){
            dataset = &sets[ss];
        }
        else{
            if(sets.size() > 15){
                continue;
            }
            sets[ss] = dataset_source::get_dataset<std::pair<KEY_TYPE, VALUE_TYPE>>(
                    source_path + ss);
            continue;
        }
        auto random_length = e() % 333;
        auto random_position = e() % (dataset->size() - random_length);
        for(int j = 0;j < random_length;++j){
//            std::cout <<dataset->at(j+random_position).first<< std::endl;
            new_set[dataset->at(j+random_position).first] = dataset->at(j+random_position).second;
        }
        std::cout <<new_set.size()<< std::endl;
    }
    auto rr = std::vector<std::pair<double,double>>(new_set.begin(),new_set.end());
    dataset_source::set_dataset<std::pair<KEY_TYPE, VALUE_TYPE>>(
            data_father_path + "osmc.data",rr);
}


int main() {
//    generate_dataset();
//    return 0;
//    auto dataset_name = std::string("face.data");
//    for(int ii = 0;ii<20000000;ii+=33){
//        auto dataset = dataset_source::get_dataset<std::pair<KEY_TYPE, VALUE_TYPE>>(
//                data_father_path + dataset_name);
//        dataset.erase(dataset.begin(),dataset.begin() + ii);
//        dataset.erase(dataset.begin() + 33,dataset.end());
//        auto min_max = get_min_max<KEY_TYPE, VALUE_TYPE>(dataset.begin(), dataset.end());
//        auto pdf = get_pdf<KEY_TYPE, VALUE_TYPE>(dataset.begin(), dataset.end(), min_max.first, min_max.second, 16384,false);
//        double acc = 0;
//        std::vector<float> cdf;
//        for(int i = 0;i<pdf.size();++i){
//            acc += pdf[i];
//            cdf.push_back(acc);
//        }
//        for(auto &i:cdf){
//            i /= acc;
//        }
//        std::map<std::string,std::string>  info;
//        info["color"] = "blue";
//        matplotlibcpp::xlabel("None");
////        info["label"] = "pdf";
////        matplotlibcpp::plot(pdf,info);
////        info["label"] = "cdf";
//        matplotlibcpp::plot(cdf,info);
//        matplotlibcpp::title(dataset_name);
//        matplotlibcpp::show();
////        matplotlibcpp::save("picture/"+dataset_name+"-local.pdf");
////        matplotlibcpp::save("picture/"+dataset_name+".pdf");
//        matplotlibcpp::cla();
//        std::ofstream file("picture/"+dataset_name+"-local.txt");
////        std::ofstream file("picture/"+dataset_name+".txt");
//        for(auto i:cdf){
//            file <<i<<",";
//        }
//        std::cout <<ii<< std::endl;
//        file.close();
//    }

    std::vector<std::string> dataset_names = scanFiles(data_father_path);
    for (auto &dataset_name: dataset_names) {
        dataset_name  = "wiki.data";
        auto dataset = dataset_source::get_dataset<std::pair<KEY_TYPE, VALUE_TYPE>>(
                "/home/redamancyguy/Desktop/buffer/data/train_dataset/" + dataset_name);
        std::sort(dataset.begin(), dataset.end());
        {
            auto min_max = get_min_max<KEY_TYPE, VALUE_TYPE>(dataset.begin() , dataset.end() );
            auto pdf = get_pdf<KEY_TYPE, VALUE_TYPE>(dataset.begin(), dataset.end(), min_max.first, min_max.second, 16384,false);
            double acc = 0;
            std::vector<float> cdf;
            for(int i = 0;i<pdf.size();++i){
                acc += pdf[i];
                cdf.push_back(acc);
            }
            for(auto &i:cdf){
                i /= acc;
            }
            std::ofstream file("picture/"+dataset_name+".txt");
            for(auto i:cdf){
                file <<i<<",";
            }
            file.close();
        }
        for(int i = 0;i<100000;i++) {
            std::cout <<i<<std::endl;
            auto min_max = get_min_max<KEY_TYPE, VALUE_TYPE>(dataset.begin() + i * 300,
                                                             dataset.begin() + (i + 1) * 300);
            auto pdf = get_pdf<KEY_TYPE, VALUE_TYPE>(dataset.begin() + i * 300, dataset.begin() + (i + 1) * 300,
                                                     min_max.first, min_max.second, 16384, false);
            double acc = 0;
            std::vector<float> cdf;
            for (int i = 0; i < pdf.size(); ++i) {
                acc += pdf[i];
                cdf.push_back(acc);
            }
            for (auto &i: cdf) {
                i /= acc;
            }
            std::ofstream file("picture/" + dataset_name + "-local.txt" + std::to_string(i));
            for (auto i: cdf) {
                file << i << ",";
            }
            file.close();
        }

        return 0;
    }
    return 0;
//    std::vector<std::pair<double,double>> dataset;
////    std::ifstream in_file("/media/redamancyguy/mobile_disk/new_dataset/NYCTaxi.txt");
//    std::ifstream in_file("/media/redamancyguy/mobile_disk/new_dataset/NYCTaxi-timestamp.txt");
//    std::set<double> sets;
//    std::vector<double> vector;
//    char buffer[128];
//    while(!in_file.eof()){
//        in_file.getline(buffer,128);
//        try
//        {
//            auto left = std::stol(std::string(buffer));
//            sets.insert(left);
//            vector.push_back(left);
//            dataset.emplace_back(left,left);
//        }
//
//        catch (std::invalid_argument& e)
//        {
//            std::cout <<buffer<< std::endl;
//            std::cerr << e.what() << std::endl;
//        }
//
//    }
//    std::cout <<"sets.size():"<<sets.size()<< std::endl;
//    std::cout <<"vector.size():"<<vector.size()<< std::endl;
//
//

//    return 0;
//    typedef aidel::AIDEL<key_type, val_type> aidel_type;
//    aidel_type *ai;
//    ai = new aidel_type();
//    ai->train(exist_keys, exist_keys, 32);
//    ai->self_check();


}