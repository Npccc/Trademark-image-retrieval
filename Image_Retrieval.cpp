#include <string.h>
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
#include <iostream>                                                  // 1. 用于将extract_features的结果保存在txt里  
#include <fstream>  
#include <sstream>  
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include<math.h>
#include <algorithm>
  
using caffe::Blob;  
using caffe::Caffe;  
using caffe::Datum;  
using caffe::Net;  
using std::string;  
namespace db = caffe::db;
using namespace cv;  
using namespace std;  
template<typename Dtype>  
int feature_extraction_pipeline(int argc, char** argv);  
string ReadLine(char *filename,int line);
void imshowMany(const std::string& _winName, vector<Mat>& _imgs);
int main(int argc, char** argv) {  
    return feature_extraction_pipeline<float>(argc, argv);  
    //  return feature_extraction_pipeline<double>(argc, argv);  
}
  

string ReadLine(char *filename,int line)
{
	int i=0;  
	string temp;  
	fstream file;  
	file.open(filename,ios::in);
	while(getline(file,temp)&&i<line-1)  
	{  
		i++;  
	}  
	file.close();  
	return temp; 
																					
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
  
      
    ifstream inFile_all("/root/caffe/examples/temp/allfeature.txt",ios::in);
    
  
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
    vector<float> feature_single;  
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
//                    outfile << feature_blob_data[d] << " ";                         // 3. 把结果输出  
                     feature_single.push_back(feature_blob_data[d]);
  

                }  
//                outfile << "\n";  
                string key_str = caffe::format_int(image_indices[i], 10);  
  
            }  // for (int n = 0; n < batch_size; ++n)  
        }  // for (int i = 0; i < num_features; ++i)  
    }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)  
         
    
    vector<float> feature_all[200];
    vector<pair<float,int> > simRate;
	float data;
   
    for(int i = 0;i < 200; i++)
     {
		 int count = 0;                                                                                                                         
	     while(!inFile_all.eof()&&count != 4096)
	  	{
			inFile_all>>data;
	 		 feature_all[i].push_back(data);
			 count++;
		}
						
     }
						
	for(int j = 0;j < 200;j++)
	{
		float n = 0,Rate;
		for(int k = 0;k < 4096;k++)
		{
			 n += (feature_all[j][k] - feature_single[k])*(feature_all[j][k] - feature_single[k]);
				
		} 

		Rate = sqrt(n);
		simRate.push_back(pair<float,int>(Rate,j));	
	}			  

	sort(simRate.begin(),simRate.end());
	char filename[]="/root/caffe/examples/temp/file_list_train.txt";
	vector<string> adress;
	vector<int> num;
	string temp;
	ofstream outFile("/home/lee/classification_test/ture_pic.txt",ofstream::trunc);
	int n = 0,m = simRate.size() - 1;
	Mat src;
	string para1 = "/home/lee/important/resultimage/TURE_Image/";
	string para2 = "/home/lee/important/resultimage/FALSE_Image/";

	double sumSimRate = 0.0;
	
	for(int i = 0; i < simRate.size();i++){
		
		sumSimRate += 	simRate[i].first;	

	}


	int ranking = 0;
	int fill = 0;
	cout<<endl;
	while(n < simRate.size())
	{
		temp = ReadLine(filename,simRate[n].second+1);
		temp.erase(temp.end()-2,temp.end());
		adress.push_back(temp);
		if(n < 12)
		{
			src = imread(temp);
			temp.erase(temp.begin(),temp.begin()+40);
			para1 += temp;
			imwrite(para1,src);
			outFile<<para1;
			if(simRate[n].second < 100 && simRate[n].second >= 10)
			{
				outFile<<fill;
				outFile<<simRate[n].second;	
			}
			if(simRate[n].second < 10)
			{
				outFile<<fill<<fill;
				outFile<<simRate[n].second;
					
			}
			if(simRate[n].second >=100)
			{
				
				outFile<<simRate[n].second;	
			}
			outFile<<endl;
			para1 = "/home/lee/important/resultimage/TURE_Image/";
			ranking ++;
			cout<<"image: "<<temp<<"; rate: "<<(sumSimRate-simRate[n].first*10)*100/sumSimRate<<"%"<<"; ranking: "<<ranking<<endl;
			n ++;
			continue;
		}
		temp.erase(temp.begin(),temp.begin()+40);
		ranking ++;
		cout<<"image: "<<temp<<"; rate: "<<(sumSimRate-simRate[n].first*10)*100/sumSimRate<<"%"<<"; ranking:"<<ranking<<endl;
	    n ++;
	}

	while(simRate.size() - m < 51)
	{
		temp = ReadLine(filename,simRate[m].second+1);
		temp.erase(temp.end()-2,temp.end());
		src = imread(temp);
		temp.erase(temp.begin(),temp.begin()+40);
		para2 += temp;
		imwrite(para2,src);
		para2 = "/home/lee/important/resultimage/FALSE_Image/";
		m --;
	}



	vector<Mat> imgs(12);

	for(int i = 0; i < 12;i ++)
	{
		string str = adress[i];
		imgs[i] = imread(str);	
	}
	

        inFile_all.close();
	outFile.close();
	  

	imshowMany("Image Retrieval", imgs);
	LOG(ERROR) << "Retrieval Successfully ";
	waitKey();
    return 0;  
}  



void imshowMany(const std::string& _winName, vector<Mat>& _imgs)
{
	int nImg = (int)_imgs.size();
	Mat dispImg;
	int size;
	int x, y;
	// w - Maximum number of images in a row  
	// h - Maximum number of images in a column 
	int w, h;
	// scale - How much we have to resize the image
	float scale;		
	int max;
	if (nImg <= 0) 
	{
		printf("Number of arguments too small....\n");
		return;
	}
	else if (nImg > 12)
	{
		printf("Number of arguments too large....\n");
		return;
	}
	else if (nImg == 1)
	{
		w = h = 1;	
		size = 300;	
	}
	else if (nImg == 2)
	{
		w = 2; h = 1;
		size = 300;
	}
	else if (nImg == 3 || nImg == 4)
	{
		w = 2; h = 2;
		size = 300;
	}
	else if (nImg == 5 || nImg == 6)
    {
		w = 3; h = 2;
		size = 200;
	}
	else if (nImg == 7 || nImg == 8)
	{
		w = 4; h = 2;
		size = 200;
	}
	else
	{
		w = 4; h = 3;
		size = 150;
	}

	dispImg.create(Size(100 + size*w, 60 + size*h), CV_8UC3);
	for (int i= 0, m=20, n=20; i<nImg; i++, m+=(20+size))
	{
	
	x = _imgs[i].cols;
	y = _imgs[i].rows;

	max = (x > y)? x: y;
	scale = (float) ( (float) max / size );
	if (i%w==0 && m!=20)
	{
		m = 20;
		n += 20+size;
	}
	Mat imgROI = dispImg(Rect(m, n, (int)(x/scale), (int)(y/scale)));
	resize(_imgs[i], imgROI, Size((int)(x/scale), (int)(y/scale)));

	}

	namedWindow(_winName);
	imshow(_winName, dispImg);

}



