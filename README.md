Instructions: 

Before all else, run make. 

To run the code, use the command
    ./hog -i inputfile -v version-number [From 1-4]
That saves output to output/cpp_out.txt (and the intermediate histogram).

hog.cpp sets up and tears down the environment, and uses switch statements
to do kernel calls via functions in either hog_serial.cpp or hog_parallel.cpp.
The functions in hog_parallel.cpp enqueue an OpenCL kernel held in
hog_parallel.cl. 

To validate, you may run our testing script with:
    mkdir input
    cp mastersword.jpg input/
    mkdir output
    ./test.sh version_number_to_test [From 1-4]

Note: In our interest of optimizing a specific case, we have only tested the
code on the three jpgs in this tarball. It should work on images that are
multiples of 8 (our cell sizes in both directions), but we haven't validated.
It will not work on non-multiples of 8 because we have not implemented padding
yet. 

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

hog_parallel.cl
    \- Kernel file containing our OpenCL kernels for hog

hog_parallel.cpp
    \- C++ file containing functions that set up and execute openCL kernels.

hog_parallel.h
    \- Header file

hog.py
    \- Reference hog implementation, taken from scikits-image and slightly modified.

hog_serial.cpp
    \- C++ file containing functions to compute hog kernels serially.

hog_serial.h
    \- Header file

makefile
    \- makefile. Only works on linux machines (only tested on hive machine)

mastersword.jpg
    \- 1024x1024 image for testing

readjpeg.cpp
    \- JPEG reading library provided in homework 2

readjpeg.h
    \- Header file

README
    \- This file!

test.sh
    \- File to run tests against a version of our code.

zelda.jpg
    \- 256x256 image for testing
