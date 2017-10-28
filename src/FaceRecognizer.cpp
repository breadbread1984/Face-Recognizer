#include "FaceRecognizer.h"

FaceRecognizer::FaceRecognizer()
{
	for(int i = 0 ; i < NUM ; i++) {
		frontaldetector[i] = dlib::get_frontal_face_detector();
		dlib::deserialize("models/dlib_face_recognition_resnet_model_v1.dat") >> net[i];
		dlib::deserialize("models/shape_predictor_68_face_landmarks.dat") >> sp[i];
	}
}

FaceRecognizer::~FaceRecognizer()
{
}

bool FaceRecognizer::loadFaces(string dir)
{
	if(false == exists(dir) || false == is_directory(dir)) return false;
	//1)生成list
	int label = -1;
	label2name.clear();
	vector<boost::tuple<path,int> > trainlist;
	for(directory_iterator it(dir) ; it != directory_iterator() ; it++) {
		if(is_directory(it->path())) {
			label2name.insert(make_pair(label,it->path().filename().string()));
			//获得当前文件夹下面的文件列表
			vector<boost::tuple<path,int> > list;
			for(directory_iterator img_itr(it->path()) ; img_itr != directory_iterator() ; img_itr++)
				if(img_itr->path().extension().string() == ".jpg") list.push_back(boost::make_tuple(img_itr->path(),label));
			//对列表打乱顺序
			random_shuffle(list.begin(),list.end());
			//然后分别放到训练和测试集合列表
			trainlist.insert(trainlist.end(),list.begin(),list.end());
			label++;
		} // if 是文件夹
	}
	random_shuffle(trainlist.begin(),trainlist.end());
	//2)提取人脸特征
	vector<boost::tuple<dlib::matrix<float,0,1>,int> > trainsamples;
	extractFeature(trainlist,trainsamples);
	//3)人脸特征放入Mat
	matTrainFeatures = Mat(trainsamples.size(),128,CV_32F);
	matTrainLabels = Mat(trainsamples.size(),1,CV_32F);
	for(int i = 0 ; i < trainsamples.size() ; i++) {
		dlib::matrix<float,0,1> & fv = get<0>(trainsamples[i]);
		int & label = get<1>(trainsamples[i]);
#ifndef NDEBUG
		assert(128 == fv.size());
#endif
		copy(fv.begin(),fv.end(),matTrainFeatures.ptr<float>(i));
		matTrainLabels.at<float>(i,0) = label;
	}
	//4)knn载入样本
	loadSamples();
	return true;
}

vector<boost::tuple<Rect,string> > FaceRecognizer::recognize(Mat img)
{
	if(kclassifier.empty()) throw logic_error("还没有载入任何样本");
	dlib::cv_image<dlib::bgr_pixel> cimg(img);
	vector<dlib::rectangle> frontalfaces = frontaldetector[0](cimg);
	//并行提取每张人脸的特征
	vector<boost::shared_ptr<boost::thread> > handlers;
	Mat matFeatures(frontalfaces.size(),128,CV_32F);
	for(int i = 0 ; i < frontalfaces.size() ; i++) {
		dlib::rectangle & rect = frontalfaces[i];
		handlers.push_back(boost::shared_ptr<boost::thread>(
			new boost::thread(boost::bind(
				&FaceRecognizer::img2feature,this,
				i,boost::ref(cimg),boost::ref(rect),boost::ref(matFeatures),handlers.size()
			))
		));
		if(NUM <= handlers.size()) {
			for(vector<boost::shared_ptr<boost::thread> >::iterator it = handlers.begin() ; it != handlers.end() ; it++) {
				(*it)->join();
			}
			handlers.clear();
		}
	}
	if(handlers.size()) {
		for(vector<boost::shared_ptr<boost::thread> >::iterator it = handlers.begin() ; it != handlers.end() ; it++) {
			(*it)->join();
		}
		handlers.clear();
	}
	//对所有人脸特征寻找最近邻
	Mat matResults(frontalfaces.size(),1,CV_32F);
	Mat matResponses(frontalfaces.size(),kclassifier->getDefaultK(),CV_32F);
	Mat dists(frontalfaces.size(),kclassifier->getDefaultK(),CV_32F);
	kclassifier->findNearest(matFeatures,kclassifier->getDefaultK(),matResults,matResponses,dists);
	//输出到返回变量
	vector<boost::tuple<Rect,string> > retVal;
	for(int i = 0 ; i < matResults.rows ; i++) {
		dlib::rectangle & rect = frontalfaces[i];
		Rect bounding(Point2i(rect.left(),rect.top()),Point2i(rect.right() + 1,rect.bottom() + 1));
		//检查当前人脸是否是已知的
		float nearestdist = numeric_limits<float>::max();
		for(int j = 0 ; j < kclassifier->getDefaultK() ; j++) {
			if(matResults.at<float>(i,0) == matResponses.at<float>(i,j))
				if(dists.at<float>(i,j) < nearestdist) nearestdist = dists.at<float>(i,j);
		}
		string id = (nearestdist <= 0.15)?label2name[static_cast<int>(matResults.at<float>(i,0))]:"";
		retVal.push_back(boost::make_tuple(bounding,id));
	}
	return retVal;
}

Mat FaceRecognizer::visualize(vector<boost::tuple<Rect,string> > & faces,Mat img)
{
	Mat retVal = img.clone();
	for(vector<boost::tuple<Rect,string> >::iterator it = faces.begin() ; it != faces.end() ; it++) {
		rectangle(retVal,get<0>(*it),Scalar(255,0,0),5);
		putText(
			retVal,
			(get<1>(*it) != "")?get<1>(*it):"???",
			Point(get<0>(*it).x,get<0>(*it).y),
			FONT_HERSHEY_SCRIPT_SIMPLEX,
			2,Scalar(0,255,0),3,8
		);
	}
	return retVal;
}

bool FaceRecognizer::save(string file)
{
	std::ofstream out(file);
	if(false == out.is_open()) return false;
	text_oarchive oa(out);
	oa<<label2name << matTrainFeatures << matTrainLabels;
	return true;
}

bool FaceRecognizer::load(string file)
{
	std::ifstream in(file);
	if(false == in.is_open()) return false;
	text_iarchive ia(in);
	ia >> label2name >> matTrainFeatures >> matTrainLabels;
	//knn载入样本
	loadSamples();
	return true;
}

vector<dlib::matrix<dlib::rgb_pixel>> FaceRecognizer::jitter_image(const dlib::matrix<dlib::rgb_pixel>& img)
{
	//提取人脸的特征
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

void FaceRecognizer::file2feature(path imgpath,int label, vector<boost::tuple<dlib::matrix<float,0,1>,int> > & outputlist,boost::mutex & mutex,int handler_num)
{
	//检测人脸位置和特征点，正面化，提取特征
	dlib::matrix<dlib::rgb_pixel> img;
	load_image(img,imgpath.string());
	vector<dlib::rectangle> rects = frontaldetector[handler_num](img);
	//如果没有检测到人脸就跳过当前图片
	if(0 == rects.size()) return;
	dlib::rectangle & rect = rects.front();
	dlib::full_object_detection shape = sp[handler_num](img,rect);
	dlib::matrix<dlib::rgb_pixel> face_chip;
	dlib::extract_image_chip(img,dlib::get_face_chip_details(shape,150,0.25),face_chip);
	dlib::matrix<float,0,1> fv = mean(mat(net[handler_num](jitter_image(face_chip))));
	{
		boost::mutex::scoped_lock scoped_lock(mutex);
		outputlist.push_back(boost::make_tuple(fv,label));
	}
}

void FaceRecognizer::img2feature(int index,dlib::cv_image<dlib::bgr_pixel> & cimg,dlib::rectangle & rect, Mat & matFeatures, int handler_num)
{
	dlib::full_object_detection shape = sp[handler_num](cimg,rect);
	dlib::matrix<dlib::rgb_pixel> face_chip;
	dlib::extract_image_chip(cimg,dlib::get_face_chip_details(shape,150,0.25),face_chip);
	dlib::matrix<float,0,1> fv = mean(mat(net[handler_num](jitter_image(face_chip))));
	copy(fv.begin(),fv.end(),matFeatures.ptr<float>(index));
}

void FaceRecognizer::extractFeature(vector<boost::tuple<path,int> > & inputlist,vector<boost::tuple<dlib::matrix<float,0,1>,int> > & outputlist)
{
#ifndef NDEBUG
	int count = 0;
#endif
	boost::mutex mutex;
	vector<boost::shared_ptr<boost::thread> > handlers;
	for(vector<boost::tuple<path,int> >::iterator it = inputlist.begin() ; it != inputlist.end() ; it++) {
		handlers.push_back(boost::shared_ptr<boost::thread> (
			new boost::thread(boost::bind(
				&FaceRecognizer::file2feature,this,
				get<0>(*it),get<1>(*it),boost::ref(outputlist),boost::ref(mutex),handlers.size()
			))
		));
		if(NUM <= handlers.size()) {
			for(vector<boost::shared_ptr<boost::thread> >::iterator it = handlers.begin() ; it != handlers.end() ; it++) {
				(*it)->join();
#ifndef NDEBUG
				if(++count %10 == 0) 	cout<<"已经处理了"<<count<<"个样本"<<endl;
#endif
			}
			handlers.clear();
		}
	}
	if(handlers.size()) {
		for(vector<boost::shared_ptr<boost::thread> >::iterator it = handlers.begin() ; it != handlers.end() ; it++) {
			(*it)->join();
#ifndef NDEBUG
			cout<<"已经处理了"<<++count<<"个样本"<<endl;
#endif
		}
		handlers.clear();
	}
}

void FaceRecognizer::loadSamples()
{
	trainingData = TrainData::create(matTrainFeatures,SampleTypes::ROW_SAMPLE,matTrainLabels);
	kclassifier = KNearest::create();
	kclassifier->setIsClassifier(true);
	kclassifier->setAlgorithmType(KNearest::Types::BRUTE_FORCE);
	kclassifier->setDefaultK(1); //只找到一个可能的身份
	kclassifier->train(trainingData);
}
