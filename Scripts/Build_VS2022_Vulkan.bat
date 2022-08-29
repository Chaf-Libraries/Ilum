git submodule update --init
cd ..
mkdir build
cd build
cmake ../
cmake --build ./ --config Release
cd ..
start ./bin/Engined.exe