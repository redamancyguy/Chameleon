//
// Created by redamancyguy on 23-5-30.
//

#ifndef HITS_CONFIGURATION_H
#define HITS_CONFIGURATION_H

#include <complex>
#include "Parameter.h"
#include "../../include/DEFINE.h"
int default_root = 30000;
int default_inner = 100;
namespace CHA {
    class Configuration {
    public:
        float root_fan_out{};
        float fan_outs[INNER_FANOUT_ROW][INNER_FANOUT_COLUMN]{};

        Configuration() {
            root_fan_out = default_root_fan_out;
            for (auto &fan_out: fan_outs) {
                for (auto &i: fan_out) {
                    i = default_inner_fan_out;
                }
            }
        }

        static Configuration default_configuration() {
            Configuration conf;
            conf.root_fan_out = default_root;
            for(auto &i:conf.fan_outs){
                for(auto &j:i){
                    j = default_inner;
                }
            }
            return conf;
        }

        void shrink(){
            root_fan_out = shrink_root_fan_out(root_fan_out);
            for(auto &i:fan_outs){
                for(auto &j:i){
                    j = shrink_inner_fan_out(j);
                }
            }
        }

        static Configuration default_configuration_best(float dataset_size) {
            Configuration conf;
            conf.root_fan_out = dataset_size;
            for (auto &j: conf.fan_outs[0]) {
                j = 8;
            }
            return conf;
        }

        Configuration operator+(const Configuration &x) {
            auto conf = *this;
            conf.root_fan_out += x.root_fan_out;
            for (int i = 0; i < INNER_FANOUT_ROW; ++i) {
                for (int j = 0; j < INNER_FANOUT_COLUMN; ++j) {
                    conf.fan_outs[i][j] += x.fan_outs[i][j];
                }
            }
            return conf;
        }

        Configuration operator-(const Configuration &x) {
            auto conf = *this;
            conf.root_fan_out -= x.root_fan_out;
            for (int i = 0; i < INNER_FANOUT_ROW; ++i) {
                for (int j = 0; j < INNER_FANOUT_COLUMN; ++j) {
                    conf.fan_outs[i][j] -= x.fan_outs[i][j];
                }
            }
            return conf;
        }

        Configuration operator*(const Configuration &x) {
            auto conf = *this;
            conf.root_fan_out *= x.root_fan_out;
            for (int i = 0; i < INNER_FANOUT_ROW; ++i) {
                for (int j = 0; j < INNER_FANOUT_COLUMN; ++j) {
                    conf.fan_outs[i][j] *= x.fan_outs[i][j];
                }
            }
            return conf;
        }

        Configuration operator*(float x) {
            auto conf = *this;
            conf.root_fan_out *= x;
            for (auto &fan_out: conf.fan_outs) {
                for (float &j: fan_out) {
                    j *= x;
                }
            }
            return conf;
        }

//        Configuration operator/(const Configuration &x) {
//            auto conf = *this;
//            conf.root_fan_out /= x.root_fan_out;
//            for (int i = 0; i < INNER_FANOUT_ROW; ++i) {
//                for (int j = 0; j < INNER_FANOUT_COLUMN; ++j) {
//                    conf.fan_outs[i][j] /= x.fan_outs[i][j];
//                }
//            }
//            return conf;
//        }

        Configuration operator/(float x) {
            auto conf = *this;
            conf.root_fan_out /= x;
            for (auto &fan_out: conf.fan_outs) {
                for (float &j: fan_out) {
                    j /= x;
                }
            }
            return conf;
        }

        Configuration sqrt_invert() {
            auto conf = *this;
            conf.root_fan_out = std::sqrt(conf.root_fan_out);
            for (auto &fan_out: conf.fan_outs) {
                for (float &j: fan_out) {
                    j = std::sqrt(j);
                }
            }
            return conf;
        }

        static Configuration zeros() {
            Configuration conf;
            conf.root_fan_out = 1;
            for (auto &i: conf.fan_outs) {
                for (auto &j: i) {
                    j = 1e-6;
                }
            }
            return conf;
        }

        static Configuration random_configuration() {
            Configuration conf;
            conf.root_fan_out = float(std::pow(random_u_0_1(),2) * (max_root_fan_out - min_root_fan_out) + min_root_fan_out);
            for (auto &fan_out: conf.fan_outs) {
                for (auto &j: fan_out) {
                    j = float(std::pow(random_u_0_1(),2)) * (max_inner_fan_out - min_inner_fan_out) + min_inner_fan_out;
                }
            }
            return conf;
        }

        friend std::ostream &operator<<(std::ostream &out, Configuration &input)
        {
            out << "root:" << input.root_fan_out << std::endl;
            for (int i = 0; i < INNER_FANOUT_ROW; i++) {
                out << i << "{";
                for (int j = 0; j < INNER_FANOUT_COLUMN; j++) {
                    out << input.fan_outs[i][j] << " ";
                }
                out << "}" << std::endl;
            }
            return out;
        }

        bool operator<(const Configuration &another) const {
            if (this->root_fan_out < another.root_fan_out) {
                return true;
            } else if (this->root_fan_out > another.root_fan_out) {
                return false;
            }
            for (int i = 0; i < INNER_FANOUT_ROW; i++) {
                for (int j = 0; j < INNER_FANOUT_COLUMN; j++) {
                    if (this->fan_outs[i][j] < another.fan_outs[i][j]) {
                        return true;
                    } else if (this->fan_outs[i][j] > another.fan_outs[i][j]) {
                        return false;
                    }
                }
            }
            return false;
        }

        bool operator>(const Configuration &another) const {
            if (this->root_fan_out > another.root_fan_out) {
                return true;
            } else if (this->root_fan_out < another.root_fan_out) {
                return false;
            }
            for (int i = 0; i < INNER_FANOUT_ROW; i++) {
                for (int j = 0; j < INNER_FANOUT_COLUMN; j++) {
                    if (this->fan_outs[i][j] > another.fan_outs[i][j]) {
                        return true;
                    } else if (this->fan_outs[i][j] < another.fan_outs[i][j]) {
                        return false;
                    }
                }
            }
            return false;
        }

        bool operator==(const Configuration &another) const {
            if (this->root_fan_out != another.root_fan_out) {
                return false;
            }
            for (int i = 0; i < INNER_FANOUT_ROW; i++) {
                for (int j = 0; j < INNER_FANOUT_COLUMN; j++) {
                    if (this->fan_outs[i][j] != another.fan_outs[i][j]) {
                        return false;
                    }
                }
            }
            return true;
        }
    };

}
#endif //HITS_CONFIGURATION_H
