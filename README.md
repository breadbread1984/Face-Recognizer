# A Facial Recognizer implemented with Dlib and Opencv3 libraries
### Introduction

This project implements a facial recognizer, which load facial images or a knn parameter file and recognize faces from images captured from webcam.

### build everything
Build everything with following command
```Shell
make
```

### add facial images of new identities
Create a new folder with the name of the new identity in dataset directory, and put the facial images of the new identity into the new folder. Then, run the following command to create a knn as the facial recognizer and run the facial recognizer.
```Shell
./recognizer -d dataset
```
The parameter of the newly created knn is saved to file  knn参数文件.dat.

### load knn parameter file
The facial recognizer can be created from a knn parameter file through the following command.
```Shell
./recognizer -p knn参数文件.dat
```

### other evaluation tools
I create two evaluation tools to test the facial feature extractor of Dlib.
The accuracy of the facial recognizer on CASIA-WebFace can be evaluated through
```Shell
./accuracy  -i  /path/to/CASIA-WebFace
```

The ROC of facial verifying on LFW can be calculated through
```Shell
./roc -i  /path/to/LFW  -p  res/pairs.txt
```
The command will generate a file roc.txt. You can load the file in Matlab and draw the ROC through
```Shell
plot(roc(:,1),roc(:,2))
```
