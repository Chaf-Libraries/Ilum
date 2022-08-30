git submodule update --init
mkdir build
cd build
cmake ../
cmake --build ./ --config Release
cd ..
start ./bin/Engined.exe