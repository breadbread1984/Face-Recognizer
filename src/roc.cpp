#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <set>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/shared_ptr.hpp>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

#define NUM 8

using namespace std;
using namespace boost::program_options;
using namespace boost::filesystem;

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

vector<dlib::matrix<dlib::rgb_pixel>> jitter_image(
    const dlib::matrix<dlib::rgb_pixel>& img
)
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

vector<boost::tuple<string,string,bool> > loadList(std::ifstream & list)
{
	vector<boost::tuple<string,string,bool> > retVal;
	string line;
	int n_set,n_num;
	getline(list,line);
	stringstream sstr;
	sstr<<line;
	sstr>>n_set>>n_num;
	for(int i = 0 ; i < n_set ; i++) {
		for(int j = 0 ; j < n_num ; j++) {
			getline(list,line);
			stringstream sstr;
			sstr << line;
			string name; int id1,id2;
			sstr>>name >>id1>>id2;
			ostringstream ss1,ss2;
			ss1<<setw(4)<<setfill('0')<<id1;
			ss2<<setw(4)<<setfill('0')<<id2;
			string file1 = name + "/" + name + "_" + ss1.str() + ".jpg";
			string file2 = name + "/" + name + "_" + ss2.str() + ".jpg";
			retVal.push_back(boost::make_tuple(file1,file2,true));
		}
		for(int j = 0 ; j < n_num ; j++) {
			getline(list,line);
			stringstream sstr;
			sstr << line;
			string name1,name2; int id1,id2;
			sstr >>name1 >> id1>>name2>>id2;
			ostringstream ss1,ss2;
			ss1<<setw(4)<<setfill('0')<<id1;
			ss2<<setw(4)<<setfill('0')<<id2;
			string file1 = name1 + "/" + name1 + "_" + ss1.str() + ".jpg";
			string file2 = name2 + "/" + name2 + "_" + ss2.str() + ".jpg";
			retVal.push_back(boost::make_tuple(file1,file2,false));
		}
	}
	return retVal;
}

void process(path file1,path file2,dlib::frontal_face_detector & frontaldetector, dlib::shape_predictor & sp, anet_type & net,vector<float> & dists,boost::mutex & mutex)
{
	dlib::matrix<dlib::rgb_pixel> img1,img2;
	load_image(img1,file1.string());
	load_image(img2,file2.string());
	vector<dlib::rectangle> rects1 = frontaldetector(img1);
	vector<dlib::rectangle> rects2 = frontaldetector(img2);
	if(0 == rects1.size()) {
		cout<<file1.string()<<endl;
		return;
	}
	if(0 == rects2.size()) {
		cout<<file2.string()<<endl;
		return;
	}
	dlib::rectangle & rect1 = rects1.front();
	dlib::rectangle & rect2 = rects2.front();
	dlib::full_object_detection shape1 = sp(img1,rect1);
	dlib::full_object_detection shape2 = sp(img2,rect2);
	dlib::matrix<dlib::rgb_pixel> face_chip1,face_chip2;
	dlib::extract_image_chip(img1,dlib::get_face_chip_details(shape1,150,0.25),face_chip1);
	dlib::extract_image_chip(img2,dlib::get_face_chip_details(shape2,150,0.25),face_chip2);
	dlib::matrix<float,0,1> fv1 = mean(mat(net(jitter_image(face_chip1))));
	dlib::matrix<float,0,1> fv2 = mean(mat(net(jitter_image(face_chip2))));
	dlib::matrix<float,0,1> diff = fv1 - fv2;
	dlib::matrix<float> res = trans(diff)*diff;
	{
		boost::mutex::scoped_lock scoped_lock(mutex);
		dists.push_back(sqrt(res(0,0)));
	}
}
							
int main(int argc,char ** argv)
{
	options_description desc;
	string inputdir;
	string listfile;
	desc.add_options()
		("help,h","打印当前使用方法")
		("input,i",value<string>(&inputdir),"LFW文件夹路径")
		("pair,p",value<string>(&listfile),"LFW验证列表文件路径");
	variables_map vm;
	store(parse_command_line(argc,argv,desc),vm);
	notify(vm);
	
	if(1 == argc || vm.count("help") || 1 != vm.count("input") || 1 != vm.count("pair")) {
		cout<<desc;
		return EXIT_SUCCESS;
	}
	
	path inputroot(inputdir);
	if(false == exists(inputroot) || false == is_directory(inputroot)) {
		cout<<"LFW文件夹不存在！"<<endl;
		return EXIT_FAILURE;
	}
	std::ifstream verifpair(listfile);
	if(false == verifpair.is_open()) {
		cout<<"LFW验证列表文件无法打开！"<<endl;
		return EXIT_FAILURE;
	}
	vector<boost::tuple<string,string,bool> > list = loadList(verifpair);
	//1)计算样本之间距离
	vector<boost::shared_ptr<boost::thread> > handlers;
	vector<float> pos_dists,neg_dists;
	boost::mutex mutex;
	dlib::frontal_face_detector frontaldetector[NUM];
	dlib::shape_predictor sp[NUM];
	anet_type net[NUM];
	for(int i = 0 ; i < NUM ; i++) {
		frontaldetector[i] = dlib::get_frontal_face_detector();
		dlib::deserialize("models/dlib_face_recognition_resnet_model_v1.dat") >> net[i];
		dlib::deserialize("models/shape_predictor_68_face_landmarks.dat") >> sp[i];
	}
	cout<<"计算样本之间距离"<<endl;
#ifndef NDEBUG
	int count = 0;
#endif
	for(vector<boost::tuple<string,string,bool> >::iterator it = list.begin() ; it != list.end() ; it++) {
		path file1 = inputroot / get<0>(*it);
		path file2 = inputroot / get<1>(*it);
		handlers.push_back(boost::shared_ptr<boost::thread> (
			new boost::thread(boost::bind(
				::process,file1,file2,
				boost::ref(frontaldetector[handlers.size()]),
				boost::ref(sp[handlers.size()]),
				boost::ref(net[handlers.size()]),
				boost::ref(get<2>(*it)?pos_dists:neg_dists),
				boost::ref(mutex)
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
				if(++count % 1000 == 0) cout<<"已经处理了"<<count<<"个图像对"<<endl;
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
			if(++count % 1000 == 0) cout<<"已经处理了"<<count<<"个图像对"<<endl;
#endif
		}
		handlers.clear();
	}
	//2)计算roc曲线
	cout<<"计算roc曲线"<<endl;
	set<float> thresholds;
	thresholds.insert(pos_dists.begin(),pos_dists.end());
	thresholds.insert(neg_dists.begin(),neg_dists.end());
	std::ofstream out("roc.txt");
	map<float,boost::tuple<float,float> > roc;
	float eer = 0;
	float prev_truepos_rate, prev_falsepos_rate;
	for(set<float>::iterator it = thresholds.begin() ; it != thresholds.end() ; it++) {
		float threshold = *it;
		int truepos = 0, falsepos = 0;
		int falseneg = 0, trueneg = 0;
		for(int i = 0 ; i < pos_dists.size() ; i++) if(pos_dists[i] < threshold) truepos++; else falseneg++;
		for(int i = 0 ; i < neg_dists.size() ; i++) if(neg_dists[i] > threshold) trueneg++; else falsepos++;
		float truepos_rate = static_cast<float>(truepos) / (truepos + falseneg);
		float falsepos_rate = static_cast<float>(falsepos) / (falsepos + trueneg);
		roc.insert(make_pair(*it,boost::make_tuple(falsepos_rate,truepos_rate)));
		if(it != thresholds.begin()) eer += 0.5 * (falsepos_rate - prev_falsepos_rate) * (truepos_rate + prev_truepos_rate);
		prev_truepos_rate = truepos_rate; prev_falsepos_rate = falsepos_rate;
		out<<falsepos_rate<<" "<<truepos_rate<<" "<<*it<<endl;
	}
	cout<<"eer = "<<eer<<endl;
		
	return EXIT_SUCCESS;
}
