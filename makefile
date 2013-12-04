OBJS=hog.o readjpeg.o hog_serial.o hog_parallel.o clhelp.o
OCL_INC=/usr/local/cuda-4.2/include
OCL_LIB=/usr/local/cuda-4.2/lib64

%.o: %.cpp readjpeg.h clhelp.h
	g++ -fopenmp -O2 -c $< -I$(OCL_INC)

#%.o: %.cpp clhelp.h
	#g++ -O2 -c $< -I$(OCL_INC)

all: $(OBJS)
	g++ -fopenmp -O2 $(OBJS) -o hog -ljpeg -L$(OCL_LIB) -lOpenCL 


