#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/string.hpp>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <opencv2/opencv.hpp>

#define NUM 25

using namespace std;
using namespace boost::program_options;
using namespace boost::filesystem;
using namespace boost::archive;
using namespace cv;
using namespace cv::ml;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N,BN,1,dlib::tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2,2,2,2,dlib::skip1<dlib::tag2<block<N,BN,2,dlib::tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<dlib::con<N,3,3,1,1,dlib::relu<BN<dlib::con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = dlib::relu<residual<block,N,dlib::affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block,N,dlib::affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = dlib::loss_metric<dlib::fc_no_bias<128,dlib::avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            dlib::max_pool<3,3,2,2,dlib::relu<dlib::affine<dlib::con<32,7,7,2,2,
                            dlib::input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

vector<dlib::matrix<dlib::rgb_pixel>> jitter_image(const dlib::matrix<dlib::rgb_pixel>& img);
void process(path imgpath,int label, vector<boost::tuple<dlib::matrix<float,0,1>,int> > & outputlist,dlib::shape_predictor & sp,anet_type & net,dlib::frontal_face_detector & frontaldetector, boost::mutex & mutex);
void convert(vector<boost::tuple<path,int> > & inputlist,vector<boost::tuple<dlib::matrix<float,0,1>,int> > & outputlist);

int main(int argc,char ** argv)
{
	options_description desc;
	string inputdir;
	desc.add_options()
		("help,h","打印当前使用方法")
		("input,i",value<string>(&inputdir),"WebFace文件夹路径");
	variables_map vm;
	store(parse_command_line(argc,argv,desc),vm);
	notify(vm);
	
	if(1 == argc || vm.count("help") || 1 != vm.count("input")) {
		cout<<desc;
		return EXIT_SUCCESS;
	}
	
	if(false == exists(inputdir) || false == is_directory(inputdir)) {
		cout<<"WebFace文件夹不存在！"<<endl;
		return EXIT_FAILURE;
	}
	
	//1)生成list
	int label = -1;
	map<int,string> label2name;
	vector<boost::tuple<path,int> > trainlist;
	vector<boost::tuple<path,int> > testlist;
	for(directory_iterator it(inputdir) ; it != directory_iterator() ; it++) {
		if(is_directory(it->path())) {
			label2name.insert(make_pair(label,it->path().filename().string()));
			//获得当前文件夹下面的文件列表
			vector<boost::tuple<path,int> > list;
			for(directory_iterator img_itr(it->path()) ; img_itr != directory_iterator() ; img_itr++)
				if(img_itr->path().extension().string() == ".jpg") list.push_back(boost::make_tuple(img_itr->path(),label));
			//对列表打乱顺序
			random_shuffle(list.begin(),list.end());
			//然后分别放到训练和测试集合列表
			testlist.insert(testlist.end(),list.begin(),list.begin() + 2);
			trainlist.insert(trainlist.end(),list.begin() + 2,list.end());
			label++;
		} // if 是文件夹
	}
	random_shuffle(trainlist.begin(),trainlist.end());
	random_shuffle(testlist.begin(),testlist.end());
	//2)生成训练集
#ifndef NDEBUG
	cout<<"生成训练集"<<endl;
#endif
	vector<boost::tuple<dlib::matrix<float,0,1>,int> > trainsamples;
	vector<boost::tuple<dlib::matrix<float,0,1>,int> > testsamples;
	convert(trainlist,trainsamples);
	convert(testlist,testsamples);
	//3)训练knn分类器
#ifndef NDEBUG
	cout<<"训练分类器"<<endl;
#endif
	Mat matTrainFeatures(trainsamples.size(),128,CV_32F);
	Mat matTrainLabels(trainsamples.size(),1,CV_32F);
	for(int i = 0 ; i < trainsamples.size() ; i++) {
		dlib::matrix<float,0,1> & fv = get<0>(trainsamples[i]);
		int & label = get<1>(trainsamples[i]);
#ifndef NDEBUG
		assert(128 == fv.size());
#endif
		copy(fv.begin(),fv.end(),matTrainFeatures.ptr<float>(i));
		matTrainLabels.at<float>(i,0) = label;
	}
	Mat matTestFeatures(testsamples.size(),128,CV_32F);
	Mat matTestLabels(testsamples.size(),1,CV_32F);
	for(int i = 0 ; i < testsamples.size() ; i++) {
		dlib::matrix<float,0,1> & fv = get<0>(testsamples[i]);
		int & label = get<1>(testsamples[i]);
#ifndef NDEBUG
		assert(128 == fv.size());
#endif
		copy(fv.begin(),fv.end(),matTestFeatures.ptr<float>(i));
		matTestLabels.at<float>(i,0) = label;
	}
	//序列化训练集和监督值
	std::ofstream out("训练参数.dat");
	text_oarchive oa(out);
	oa<<label2name<<matTrainFeatures<<matTrainLabels;
	Ptr<TrainData> trainingData = TrainData::create(matTrainFeatures,SampleTypes::ROW_SAMPLE,matTrainLabels);
	Ptr<KNearest> kclassifier = KNearest::create();
	kclassifier->setIsClassifier(true);
	kclassifier->setAlgorithmType(KNearest::Types::BRUTE_FORCE);
	kclassifier->setDefaultK(1); //只找到一个可能的身份
	kclassifier->train(trainingData);
	//4)测试训练结果
#ifndef NDEBUG
	cout<<"测试分类器结果"<<endl;
#endif
	Mat matResults(testsamples.size(),1,CV_32F);
	kclassifier->findNearest(matTestFeatures,kclassifier->getDefaultK(),matResults);
	int corrected = 0;
	for(int i = 0 ; i < matResults.rows ; i++)
		if(matResults.at<float>(i,0) == matTestLabels.at<float>(i,0)) corrected++;
	cout<<"accuracy = "<<static_cast<float>(corrected) / matResults.rows <<endl;
}

vector<dlib::matrix<dlib::rgb_pixel>> jitter_image(const dlib::matrix<dlib::rgb_pixel>& img)
{
    // All this function does is make 100 copies of img, all slightly jittered by being
    // zoomed, rotated, and translated a little bit differently.
    thread_local dlib::random_cropper cropper;
    cropper.set_chip_dims(150,150);
    cropper.set_randomly_flip(true);
    cropper.set_max_object_size(0.99999);
    cropper.set_background_crops_fraction(0);
    cropper.set_min_object_size(146,145);
    cropper.set_translate_amount(0.02);
    cropper.set_max_rotation_degrees(3);

    std::vector<dlib::mmod_rect> raw_boxes(1), ignored_crop_boxes;
    raw_boxes[0] = shrink_rect(get_rect(img),3);
    std::vector<dlib::matrix<dlib::rgb_pixel>> crops; 

    dlib::matrix<dlib::rgb_pixel> temp; 
    for (int i = 0; i < 100; ++i)
    {
        cropper(img, raw_boxes, temp, ignored_crop_boxes);
        crops.push_back(move(temp));
    }
    return crops;
}
							
void process(path imgpath,int label, vector<boost::tuple<dlib::matrix<float,0,1>,int> > & outputlist,dlib::shape_predictor & sp,anet_type & net,dlib::frontal_face_detector & frontaldetector, boost::mutex & mutex)
{
	dlib::matrix<dlib::rgb_pixel> img;
	load_image(img,imgpath.string());
	vector<dlib::rectangle> rects = frontaldetector(img);
	//如果没有检测到人脸就跳过当前图片
	if(0 == rects.size()) return;
	dlib::rectangle & rect = rects.front();
	dlib::full_object_detection shape = sp(img,rect);
	dlib::matrix<dlib::rgb_pixel> face_chip;
	dlib::extract_image_chip(img,dlib::get_face_chip_details(shape,150,0.25),face_chip);
	dlib::matrix<float,0,1> fv = mean(mat(net(jitter_image(face_chip))));
	{
		boost::mutex::scoped_lock scoped_lock(mutex);
		outputlist.push_back(boost::make_tuple(fv,label));
	}
}

void convert(vector<boost::tuple<path,int> > & inputlist,vector<boost::tuple<dlib::matrix<float,0,1>,int> > & outputlist)
{
#ifndef NDEBUG
	int count = 0;
#endif
	boost::mutex mutex;
	vector<boost::shared_ptr<boost::thread> > handlers;
	dlib::frontal_face_detector frontaldetector[NUM];
	dlib::shape_predictor sp[NUM];
	anet_type net[NUM];
	for(int i = 0 ; i < NUM ; i++) {
		frontaldetector[i] = dlib::get_frontal_face_detector();
		dlib::deserialize("models/dlib_face_recognition_resnet_model_v1.dat") >> net[i];
		dlib::deserialize("models/shape_predictor_68_face_landmarks.dat") >> sp[i];
	}
	for(vector<boost::tuple<path,int> >::iterator it = inputlist.begin() ; it != inputlist.end() ; it++) {
		handlers.push_back(boost::shared_ptr<boost::thread> (
			new boost::thread(boost::bind(
				::process,get<0>(*it),get<1>(*it),boost::ref(outputlist),boost::ref(sp[handlers.size()]),boost::ref(net[handlers.size()]),boost::ref(frontaldetector[handlers.size()]),boost::ref(mutex)
			))
		));
		if(NUM <= handlers.size()) {
			for(
				vector<boost::shared_ptr<boost::thread> >::iterator it = handlers.begin() ;
				it != handlers.end() ;
				it++
			) {
				(*it)->join();
#ifndef NDEBUG
				if(++count %1000 == 0) 	cout<<"已经处理了"<<count<<"个样本"<<endl;
#endif
			}
			handlers.clear();
		}
	}
	if(handlers.size()) {
		for(
			vector<boost::shared_ptr<boost::thread> >::iterator it = handlers.begin() ;
			it != handlers.end() ;
			it++
		) {
			(*it)->join();
#ifndef NDEBUG
			if(++count %1000 == 0) 	cout<<"已经处理了"<<count<<"个样本"<<endl;
#endif
		}
		handlers.clear();
	}
}

namespace boost{
	namespace serialization {
		template<class Archive> void serialize(Archive &ar, cv::Mat& mat, const unsigned int)
		{
			int cols, rows, type;
			bool continuous;

			if (Archive::is_saving::value) {
				cols = mat.cols; rows = mat.rows; type = mat.type();
				continuous = mat.isContinuous();
			}

			ar & cols & rows & type & continuous;

			if (Archive::is_loading::value)
				mat.create(rows, cols, type);

			if (continuous) {
				const unsigned int data_size = rows * cols * mat.elemSize();
				ar & boost::serialization::make_array(mat.ptr(), data_size);
			} else {
				const unsigned int row_size = cols*mat.elemSize();
				for (int i = 0; i < rows; i++) {
					ar & boost::serialization::make_array(mat.ptr(i), row_size);
				}
			}
		}
	}
}
