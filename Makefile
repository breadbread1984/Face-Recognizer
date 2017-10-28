CXXFLAGS=-std=c++14 `pkg-config --cflags dlib-1 opencv` -O2 -msse4
LIBS=-std=c++14 `pkg-config --libs dlib-1 opencv` -lboost_filesystem -lboost_system -lboost_program_options -lboost_thread -lboost_serialization -lcblas -llapack
OBJS=$(patsubst %.cpp,%.o,$(wildcard src/*.cpp))

all: recognizer accuracy roc

recognizer: src/main.o src/FaceRecognizer.o
	$(CXX) $^ $(LIBS) -o ${@}

accuracy: src/accuracy.o
	$(CXX) $^ $(LIBS) -o ${@}
	
roc: src/roc.o
	$(CXX) $^ $(LIBS) -o ${@}
	
clean:
	$(RM) $(OBJS) recognizer accuracy roc 
