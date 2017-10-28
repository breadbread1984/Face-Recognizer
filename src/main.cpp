#include <cstdlib>
#include <cassert>
#include <iostream>
#include <boost/program_options.hpp>
#include "FaceRecognizer.h"

using namespace std;
using namespace boost;
using namespace boost::program_options;

int main(int argc,char ** argv)
{
	options_description desc;
	string facesdir;
	string paramfile;
	desc.add_options()
		("help,h","打印当前使用方法")
		("param,p",value<string>(&paramfile),"knn的参数文件")
		("dir,d",value<string>(&facesdir),"人脸样本文件夹");
	variables_map vm;
	store(parse_command_line(argc,argv,desc),vm);
	notify(vm);
	
	if(1 == argc || vm.count("help")) {
		cout<<desc;
		return EXIT_SUCCESS;
	}
	
	if(1 != vm.count("param") + vm.count("dir")) {
		cout<<"需要提供--param或者--dir中的一个"<<endl;
		return EXIT_FAILURE;
	}

	namedWindow("test",WINDOW_NORMAL);
	VideoCapture cap(CV_CAP_ANY);
	if(false == cap.isOpened()) {
		cout<<"无法打开摄像头"<<endl;
		return EXIT_FAILURE;
	}
	
	FaceRecognizer face_recognizer;
	if(vm.count("param")) {
		cout<<"从knn参数文件生成knn分类器"<<endl;
		assert(face_recognizer.load(paramfile));
	} else {
		cout<<"从样本图片生成knn分类器"<<endl;
		assert(face_recognizer.loadFaces(facesdir));
		assert(face_recognizer.save("knn参数文件.dat"));
	}
	
	Mat img;
	while(cap.read(img)) {
		vector<boost::tuple<Rect,string> > faces = face_recognizer.recognize(img);
		Mat labeledimg = face_recognizer.visualize(faces,img);
		imshow("test",labeledimg);
		char k = waitKey(1);
		if(k == 'q') break;
	}
	
	return EXIT_SUCCESS;
}
