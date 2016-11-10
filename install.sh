cmake -E make_directory build && cd build && CC=icc cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_VERBOSE_MAKEFILE=on && make -j
