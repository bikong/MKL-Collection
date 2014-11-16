CXX ?= g++
CFLAGS = -Wall -Wconversion -O3 -fPIC -U_FORTIFY_SOURCE
LDFLAGS = -lrt
SHVER = 1

all: svm-train svm-predict svm-scale svm-predict-prod svm-train-prod

lib: svm.o
	$(CXX) -shared svm.o -o libsvm.so.$(SHVER)

svm-predict: svm-predict.c svm.o 
	$(CXX) $(CFLAGS) svm-predict.c svm.o timer.o -o svm-predict -lm $(LDFLAGS)
svm-train: svm-train.c svm.o
	$(CXX) $(CFLAGS) svm-train.c svm.o timer.o -o svm-train -lm $(LDFLAGS)
svm-predict-prod: svm-predict.c svm-prod.o 
	$(CXX) $(CFLAGS) svm-predict.c svm-prod.o timer.o -o svm-predict-prod -lm $(LDFLAGS)
svm-train-prod: svm-train.c svm-prod.o
	$(CXX) $(CFLAGS) svm-train.c svm-prod.o timer.o -o svm-train-prod -lm $(LDFLAGS)
svm-scale: svm-scale.c
	$(CXX) $(CFLAGS) svm-scale.c -o svm-scale $(LDFLAGS)
svm-prod.o: svm-prod.cpp svm.h timer.o
	$(CXX) $(CFLAGS) -c svm-prod.cpp
svm.o: svm.cpp svm.h timer.o
	$(CXX) $(CFLAGS) -c svm.cpp
timer.o: timer.cpp timer.hpp
	$(CXX) $(CFLAGS) -c timer.cpp
clean:
	rm -f *~ svm.o timer.o svm-prod.o svm-train svm-predict svm-scale svm-train-prod svm-predict-prod
