#!/bin/bash
# Histogram of Oriented Gradients Tester

if [ ! -d input ]; then
    echo "Create input directory of images"
    exit 1
fi
if [ ! -d output ]; then
    echo "Create output directory for images"
    exit 1
fi

mkdir -p log/failures

make

echo "Test,Compare,Total,Time #1,Percent #1,Time #2,Percent #2,Time #3,Percent #3,Time #4,Percent #4" > log/hog_out.log
echo "Test,Compare,Total,Time #1,Percent #1,Time #2,Percent #2,Time #3,Percent #3,Time #4,Percent #4" > log/hog_hist.log

chmod 766 log/*

DATA=""
HIST=""
FAILED=0
FAILED_TESTS=""
VERSION=1

if [ $1 != "" ]; then
    VERSION=$1
fi


echo "TESTING STARTED"
for FILE in $(ls input); do
    echo $FILE
    PATHNAME="input/"$FILE
    export PATHNAME
    python hog.py -i "input/"$FILE -o "output/"$FILE
    DATA=$(./hog -i "input/"$FILE -v $VERSION)
    HIST=$DATA
    DATA=$FILE","$(python compare.py output/py_out.txt output/cpp_out.txt)","${DATA%","}
    HIST=$FILE","$(python compare.py output/py_hist.txt output/cpp_hist.txt)","${HIST%","}

    FAIL=$(echo $DATA | cut -d"," -f2)

    PIXELS=$(python -c "if True:
    from PIL import Image
    import os
    def get_num_pixels(filepath):
        width, height = Image.open(filepath).size
        return width*height

    print str(get_num_pixels(os.environ['PATHNAME']))
    ")

    PERCENT=$(expr $FAIL \* 100)
    PERCENT=$(expr $PERCENT / $PIXELS)

    if [ $PERCENT != 0 ]; then
        FAILED=$(expr $FAILED + 1)
        FAILED_TESTS=$FAILED_TESTS" $FILE"
        FILE=$(echo $FILE | cut -d"." -f1)
        mv output/py_out.txt log/failures/$FILE"_py_out.txt"
        mv output/py_hist.txt log/failures/$FILE"_py_hist.txt"
        mv output/cpp_out.txt log/failures/$FILE"_cpp_out.txt"
        mv output/cpp_hist.txt log/failures/$FILE"_cpp_hist.txt"

    fi

    echo $DATA >> log/hog_out.log
    echo $HIST >> log/hog_hist.log
done

echo "TESTING COMPLETE"
if [ $FAILED == 0 ]; then
    echo "No Failures!"
else
    echo "Failed $FAILED tests"
    echo "Failed:$FAILED_TESTS"
fi
exit 0
