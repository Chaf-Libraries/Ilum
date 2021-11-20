# IlumEngine(WIP)
Ilum Graphics Engine, name after *Planet Ilum* from [Star Wars](https://starwars.fandom.com/es/wiki/Ilum)

## Build

* Windows 10
* Visual Studio 2019
* C++17
* CMake 3.14+

Run:

```shell
git clone https://github.com/Chaf-Libraries/IlumEngine --recursive
mkdir build
cd build
cmake ..
cmake --build ./ --target ALL_BUILD --config Release
```

## Feature

### Overview

![image-20211120113603895](README/image-20211120113603895.png)

### Render Graph

![image-20211120113259237](README/image-20211120113259237.png)

### Editor

![image-20211120113509528](README/image-20211120113509528.png)

## TODO

More features are on their way:

* Image based lighting

* Shadow
* RTX
* GI
* GPU driven rendering
* Screen space
  * SSR
  * SSGI
  * SSAO
  * ...
* Simulation
  * Ridge body
  * Fluid
  * Cloth
* ...
