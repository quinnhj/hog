hog
===

Instructions: 

Before all else, run make. 

To run the code, use the command
    ./hog -i inputfile -v version-number
That saves output to output/cpp_out




Files:

bigmastersword.jpg
    \- 2048x2048 image for testing
clhelp.cpp
    \- Helper file from homeworks 4-6 for common OpenCL actions
clhelp.h
    \- Header file for clhelp.cpp
compare.py
    \- Short script that compares two text files of numbers with a given error tolerance.
hog.cpp
    \- Main hog program that calls hog functions.
hog_parallel.cl_
    \- Kernel file containing our OpenCL kernels for hog
hog_parallel.cpp_
    \- C++ file containing functions that set up and execute openCL kernels.
hog_parallel.h_
    \- Header file
hog.py
    \- Reference hog implementation, taken from scikits-image and slightly modified.
hog_serial.cpp_
    \- C++ file containing functions to compute hog kernels serially.
hog_serial.h_
    \- Header file
makefile
    \- makefile. Only works on linux machines (only tested on hive)
mastersword.jpg
    \- 1024x1024 image for testing
readjpeg.cpp
    \- JPEG reading library provided in homework 2
readjpeg.h
    \- Header file
README.md
    \- This file!
readtest.sh
    \- Development tool that reads and processes logs.
test.sh
    \- File to run tests against a version of our code.
zelda.sh
    \- 256x256 image for testing


