//
// Created by wenli on 2023/1/10.
//

#ifndef DATASET_HPP
#define DATASET_HPP

#include <iostream>
#include <unordered_map>
#include <fstream>
#include <boost/math/distributions/normal.hpp>
#include "DEFINE.h"
#include "FIleLock.hpp"

class dataset_source {
public:
    template<class data_type>
    static void set_dataset(const std::string &filename, const std::vector<data_type> &dataset_input) {
        std::FILE *out_file = std::fopen(filename.c_str(), "wb");
        std::fwrite(dataset_input.data(), sizeof(data_type), dataset_input.size(), out_file);
        std::fclose(out_file);
    }

    template<class data_type>
    static std::vector<data_type> get_dataset(const std::string &filename) {
//        std::cout << "filename:" << filename << std::endl;
        std::FILE *in_file = std::fopen(filename.c_str(), "rb");
        std::fseek(in_file, 0, SEEK_END);
        std::size_t size = ::ftell(in_file) / sizeof(data_type);
        std::fseek(in_file, 0, SEEK_SET);
        std::vector<data_type> result(size);
        if (std::fread(result.data(), size, sizeof(data_type), in_file) == 0) {}
        std::fclose(in_file);
        return result;
    }

    template<class key_T, class value_T>
    static std::vector<std::pair<key_T, value_T>> random_dataset(double min, double max,
                                                                 int num) {
        auto tmp_e = e;
        tmp_e.seed(1000);
        std::unordered_map<key_T, value_T> result_map;
        std::vector<std::pair<key_T, value_T>> result;
        int i = 0;
        while (int(result_map.size()) < num) {
            result_map[(max - min) * ((double) (int(std::rand()) % std::numeric_limits<int>::max()) /
                                      (double) std::numeric_limits<int>::max()) + min] = i++;
        }
        for (auto &ii: result_map) {
            result.push_back(ii);
        }
        std::shuffle(result.begin(), result.end(), tmp_e);
        return result;
    }

private:
};

template<class key_T, class value_T>
std::pair<key_T, key_T> get_min_max(typename std::vector<std::pair<key_T, value_T>>::const_iterator begin,
                                    typename std::vector<std::pair<key_T, value_T>>::const_iterator end) {
    std::pair<key_T, key_T> result = {std::numeric_limits<key_T>::max(),
                                      -std::numeric_limits<key_T>::max()};
    for (auto i = begin; i != end; ++i) {
        if (i->first < result.first) {
            result.first = i->first;
        }
        if (i->first > result.second) {
            result.second = i->first;
        }
    }
    return result;
}

template<class key_T, class value_T>
std::vector<float> get_pdf(
        typename std::vector<std::pair<key_T, value_T>>::const_iterator begin,
        typename std::vector<std::pair<key_T, value_T>>::const_iterator end,
        double lower, double upper, int pdf_size, bool shrink = true) {
    std::vector<float> bucket(pdf_size);
    for (auto start = begin; start < end; ++start) {
        auto index = std::min(double(pdf_size - 1e-5),
                              std::max(0.0, double(pdf_size) * (start->first - lower) / (upper - lower)));
        ++bucket[index];
    }
    auto test_count = double(end - begin);
    if (test_count == 0) {
        return bucket;
    }
    for (float &i: bucket) {
        i = (float) (i * (double) (pdf_size) / test_count);
        if (shrink) {
            i = std::sqrt(i);
            i = std::sqrt(i);
        }
    }
    return bucket;
}

template<class T>
std::vector<T> load_osm_binary_data(std::string filename, int start, int length) {
    std::ifstream file(filename);
    std::vector<T> result;
    result.resize(length);
    file.seekg(start * sizeof(T));
    file.read((char *) result.data(), length * sizeof(T));
    return result;
}


std::vector<std::pair<double, double>> create_dataset(std::size_t length = 200e6, double skew = 1, int batch = 10) {
    std::unordered_map<double, double> map;
    double mean = 0;
    skew = 1 / skew;
    while (map.size() < length) {
        mean += 1;
        for (int i = 0; i < batch; i++) {
            auto random_key = boost::math::quantile(boost::math::normal_distribution<>(mean, skew),
                                                    0.001 + random_u_0_1() * 0.998);
            map[random_key] = random_key;
        }
    }
    std::vector<std::pair<double, double>> result;
    result.assign(map.begin(), map.end());
    return result;
}

template<class key_T, class value_T>
double local_skew(typename std::vector<std::pair<key_T, value_T>>::const_iterator begin,
                  typename std::vector<std::pair<key_T, value_T>>::const_iterator end,
                  double lower, double upper) {
    double rate = 1 / (upper - lower);
    double n_1 = (end - begin) - 1;
    std::priority_queue<double,std::vector<double>,std::greater<double>> pq;
    for (auto start = begin + 1; start < end; ++start) {
        auto key_left = ((start - 1)->first - lower) * rate;
        auto key_right = ((start)->first - lower) * rate;
        pq.push(1 / (key_right - key_left));
    }
    while(pq.size() > 1){
        auto a = pq.top();
        pq.pop();
        auto b = pq.top();
        pq.pop();
        pq.push(a + b);
    }
    auto end_value = pq.top();
    return end_value / (n_1 * n_1);
}

#endif //DATASET_HPP
