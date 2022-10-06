git submodule update --init
mkdir build
cd build
cmake ../
cmake --build ./ --config Release -j 2