git submodule update --init --recursive
mkdir build
cd build
cmake ..
cmake --build ./ --target ALL_BUILD --config Release