#!/bin/bash
make

echo "" > log_timer.txt
ITER=1
while [  $ITER -lt 5 ]; do
    ./hog -i bigmastersword.jpg -v $ITER >> log_timer.txt
    echo "" >> log_timer.txt
    let ITER=ITER+1
done
cat log_timer.txt


echo "" > log_timer.txt
ITER=1
while [  $ITER -lt 5 ]; do
    ./hog -i mastersword.jpg -v $ITER >> log_timer.txt
    echo "" >> log_timer.txt
    let ITER=ITER+1
done
cat log_timer.txt


echo "" > log_timer.txt
ITER=1
while [  $ITER -lt 5 ]; do
    ./hog -i zelda.jpg -v $ITER >> log_timer.txt
    echo "" >> log_timer.txt
    let ITER=ITER+1
done
cat log_timer.txt

