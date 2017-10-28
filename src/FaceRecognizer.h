#ifndef FACERECOGNIZER_H
#define FACERECOGNIZER_H

#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/string.hpp>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>

//提取特征的时候多线程的数量，最好设置成cpu的核数
#define NUM 8

using namespace std;
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

class FaceRecognizer {
public:
	FaceRecognizer();
	virtual ~FaceRecognizer();
	bool loadFaces(string dir);
	vector<boost::tuple<Rect,string> > recognize(Mat);
	Mat visualize(vector<boost::tuple<Rect,string> > & faces,Mat img);
	bool save(string file);
	bool load(string file);
protected:
	vector<dlib::matrix<dlib::rgb_pixel>> jitter_image(const dlib::matrix<dlib::rgb_pixel>& img);
	void extractFeature(vector<boost::tuple<path,int> > & inputlist,vector<boost::tuple<dlib::matrix<float,0,1>,int> > & outputlist);
	void file2feature(path imgpath,int label, vector<boost::tuple<dlib::matrix<float,0,1>,int> > & outputlist,boost::mutex & mutex,int handler_num);
	void img2feature(int index,dlib::cv_image<dlib::bgr_pixel> & cimg,dlib::rectangle & rect, Mat & matFeatures, int handler_num);
	void loadSamples();
private:
	//要被序列化的属性
	map<int,string> label2name;
	Mat matTrainFeatures;
	Mat matTrainLabels;
	//不会被序列化的属性
	dlib::frontal_face_detector frontaldetector[NUM];
	anet_type net[NUM];
	dlib::shape_predictor sp[NUM];
	Ptr<TrainData> trainingData;
	Ptr<KNearest> kclassifier;
};

namespace boost{
	namespace serialization {
		template<class Archive> void serialize(Archive &ar, Mat& mat, const unsigned int)
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

#endif
