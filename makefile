OBJS=hog.o readjpeg.o hog_serial.o

%.o: %.cpp readjpeg.h
	g++ -fopenmp -O2 -c $<

all: $(OBJS)
	g++ -fopenmp -O2 $(OBJS) -o hog -ljpeg 
