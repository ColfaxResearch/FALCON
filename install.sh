cmake -E make_directory build && cd build && cmake .. -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc && make && cp libfalcon.so ../lib && cp vgg_winograd ../example
