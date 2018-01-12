#include <string>
#include <vector>  
  
#include "boost/algorithm/string.hpp"  
#include "google/protobuf/text_format.h"  
#include "caffe/blob.hpp"  
#include "caffe/common.hpp"  
#include "caffe/net.hpp"  
#include "caffe/proto/caffe.pb.h"  
#include "caffe/util/db.hpp"  
#include "caffe/util/format.hpp"  
#include "caffe/util/io.hpp"  
#include <iostream>                                                            // 1. 用于将extract_features的结果保存在txt里  
#include <fstream>  
#include <sstream>  
  
using caffe::Blob;  
using caffe::Caffe;  
using caffe::Datum;  
using caffe::Net;  
using std::string;  
namespace db = caffe::db;  
  
template<typename Dtype>  
int feature_extraction_pipeline(int argc, char** argv);  
  
int main(int argc, char** argv) {  
    return feature_extraction_pipeline<float>(argc, argv);  
    //  return feature_extraction_pipeline<double>(argc, argv);  
}  
  
template<typename Dtype>  
int feature_extraction_pipeline(int argc, char** argv) {  
    ::google::InitGoogleLogging(argv[0]);  
    const int num_required_args = 7;  
    if (argc < num_required_args) {  
        LOG(ERROR) <<  
        "This program takes in a trained network and an input data layer, and then"
        " extract features of the input data produced by the net.\n"
        "Usage: extract_features  pretrained_net_param"
        "  feature_extraction_proto_file  extract_feature_blob_name1[,name2,...]"
        "  save_feature_dataset_name1[,name2,...]  num_mini_batches  db_type"
        "  [CPU/GPU] [DEVICE_ID=0]\n"
        "Note: you can extract multiple features in one pass by specifying"
        " multiple feature blob names and dataset names separated by ','."
        " The names cannot contain white space characters and the number of blobs"
        " and datasets must be equal.";
        return 1;  
    }  
    int arg_pos = num_required_args;  
  
    std::ofstream outfile("/root/caffe/examples/temp/allfeature.txt", std::ios::ate);        

    // 2. 输出的extract_featuresResult.txt ，其在和extract_features.cpp相同路径下  
  
    arg_pos = num_required_args;  
    if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {  
        LOG(ERROR) << "Using GPU";  
        int device_id = 0;  
        if (argc > arg_pos + 1) {  
            device_id = atoi(argv[arg_pos + 1]);  
            CHECK_GE(device_id, 0);  
        }  
        LOG(ERROR) << "Using Device_id=" << device_id;  
        Caffe::SetDevice(device_id);  
        Caffe::set_mode(Caffe::GPU);  
    }  
    else {  
        LOG(ERROR) << "Using CPU";  
        Caffe::set_mode(Caffe::CPU);  
    }  
  
    arg_pos = 0;  // the name of the executable  
    std::string pretrained_binary_proto(argv[++arg_pos]);  
  
    // Expected prototxt contains at least one data layer such as  
    //  the layer data_layer_name and one feature blob such as the  
    //  fc7 top blob to extract features.  
    /* 
    layers { 
    name: "data_layer_name" 
    type: DATA 
    data_param { 
    source: "/path/to/your/images/to/extract/feature/images_leveldb" 
    mean_file: "/path/to/your/image_mean.binaryproto" 
    batch_size: 128 
    crop_size: 227 
    mirror: false 
    } 
    top: "data_blob_name" 
    top: "label_blob_name" 
    } 
    layers { 
    name: "drop7" 
    type: DROPOUT 
    dropout_param { 
    dropout_ratio: 0.5 
    } 
    bottom: "fc7" 
    top: "fc7" 
    } 
    */  
    std::string feature_extraction_proto(argv[++arg_pos]);                              // 网络模型  
    boost::shared_ptr<Net<Dtype> > feature_extraction_net(  
        new Net<Dtype>(feature_extraction_proto, caffe::TEST));  
    feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);  
  
    std::string extract_feature_blob_names(argv[++arg_pos]);                            // 待提取的层  
    std::vector<std::string> blob_names;  
    boost::split(blob_names, extract_feature_blob_names, boost::is_any_of(","));  
  
    std::string save_feature_dataset_names(argv[++arg_pos]);                            // 将 待提取的层 放在 哪个文件夹  
    std::vector<std::string> dataset_names;  
    boost::split(dataset_names, save_feature_dataset_names,  
        boost::is_any_of(","));  
    CHECK_EQ(blob_names.size(), dataset_names.size()) <<  
        " the number of blob names and dataset names must be equal";  
    size_t num_features = blob_names.size();  
  
    for (size_t i = 0; i < num_features; i++) {                                          // 待提取的层  
        CHECK(feature_extraction_net->has_blob(blob_names[i]))  
            << "Unknown feature blob name " << blob_names[i]  
            << " in the network " << feature_extraction_proto;  
    }  
  
    int num_mini_batches = atoi(argv[++arg_pos]);  
  
  
    LOG(ERROR) << "Extracting Features";  
  
    Datum datum;  
    std::vector<int> image_indices(num_features, 0);  
    for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {  
        feature_extraction_net->Forward();  
        for (int i = 0; i < num_features; ++i) {  
            const boost::shared_ptr<Blob<Dtype> > feature_blob =  
                feature_extraction_net->blob_by_name(blob_names[i]);  
            int batch_size = feature_blob->num();  
            int dim_features = feature_blob->count() / batch_size;  
            const Dtype* feature_blob_data;  
            for (int n = 0; n < batch_size; ++n) {  
                datum.set_height(feature_blob->height());  
                datum.set_width(feature_blob->width());  
                datum.set_channels(feature_blob->channels());  
                datum.clear_data();  
                datum.clear_float_data();  
                feature_blob_data = feature_blob->cpu_data() +  
                    feature_blob->offset(n);  
                for (int d = 0; d < dim_features; ++d) {  
                    outfile << feature_blob_data[d] << " ";                         // 3. 把结果输出  
  
                }  
                outfile << "\n";  
                string key_str = caffe::format_int(image_indices[i], 10);  
  
            }  // for (int n = 0; n < batch_size; ++n)  
        }  // for (int i = 0; i < num_features; ++i)  
    }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)  
    outfile.close();                                                                // 4. 关闭txt文件  
    LOG(ERROR) << "Successfully extracted the all  features!";  
    return 0;  
}  
