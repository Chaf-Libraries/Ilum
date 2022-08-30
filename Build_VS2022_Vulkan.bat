git submodule update --init
mkdir build
cd build
cmake --build ./ --config Release
cd ..
start ./bin/Engined.exe