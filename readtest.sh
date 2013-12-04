#!/bin/bash

if [ ! $1 ]; then
    echo "Pass a test name"
    exit 1
fi

TEST=$(grep $1 log/hog_out.log)

if [ -z $TEST ]; then
    echo "Test not found"
    exit 0
fi

echo -n "Test: "
echo $TEST | cut -d"," -f1

TMP=$(echo $TEST | cut -d"," -f2)
echo "$TMP Pixels Failed"

echo "Total Time: "$(echo $TEST | cut -d"," -f3)

echo "#1"
PERCENT=$(echo $TEST | cut -d"," -f5)
echo -e "$(echo $TEST | cut -d"," -f4) seconds\tPercentage: $PERCENT"

echo "#2"
PERCENT=$(echo $TEST | cut -d"," -f7)
echo -e "$(echo $TEST | cut -d"," -f6) seconds\tPercentage: $PERCENT"

echo "#3"
PERCENT=$(echo $TEST | cut -d"," -f9)
echo -e "$(echo $TEST | cut -d"," -f8) seconds\tPercentage: $PERCENT"

echo "#4"
PERCENT=$(echo $TEST | cut -d"," -f11)
echo -e "$(echo $TEST | cut -d"," -f10) seconds\tPercentage: $PERCENT"

exit 0
