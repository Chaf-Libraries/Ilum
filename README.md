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

* Architecture
  * Deferred Shading Pipeline
  * Render Graph
  * Entity Component System
  * Asynchronous Resource Loading
  * Scene Loading/Saving
* Rendering Optimization
  * Multi-Draw Indirect
  * Bindless Texture
  * Vertex/Index Buffer Packing
  * GPU Frustum Culling
* Rendering Feature
  * PBR Material
* Post Processing
  * Blooming

## Upcoming Feature

* Skybox
* Image Based Lighting


## Screenshot

![image-20211120113603895](README/image-20211120113603895.png)

![image-20211120113259237](README/image-20211120113259237.png)

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
