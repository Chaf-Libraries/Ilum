# Ilum(WIP)

[![Windows](https://github.com/Chaf-Libraries/Ilum/actions/workflows/windows.yml/badge.svg)](https://github.com/Chaf-Libraries/Ilum/actions/workflows/windows.yml) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/b0cb3a2729ee4be783dd5feb2cc67eb6)](https://www.codacy.com/gh/Chaf-Libraries/IlumEngine/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Chaf-Libraries/IlumEngine&amp;utm_campaign=Badge_Grade)

Ilum Graphics Playground, name after *Planet Ilum* from [Star Wars](https://starwars.fandom.com/es/wiki/Ilum)

![image-20211120113509528](README/image-20211120113509528.png)

## Build

* Windows 10
* Visual Studio 2019
* C++17
* CMake 3.14+

Run:

```shell
git clone https://github.com/Chaf-Libraries/Ilum --recursive
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
  * GPU Back-Face Cone Culling
  * GPU Hierarchy Z Buffer Occlusion Culling
* Curve Modeling
  * Bézier Curve
  * Cubic Spline Curve
  * B Spline Curve
  * Rational Bézier Curve
  * Rational B Spline Curve
* Tensor Product Surface Modeling
  * Bézier Surface
  * B Spline Surface
  * Rational Bézier Surface
  * Rational B Spline Surface
* Rasterization Shading
  * Cook-Torrance BRDF
  * Kulla-Conty Approximation
* Post Processing
  * Temporal Anti-Alias
  * Blooming


## Feature

### Deferred Physical Based Shading

![image-20211120113603895](README/image-20211120113603895.png)

### Render Passes Visualization

![image-20211120113259237](README/image-20211120113259237.png)

### Meshlet Culling

![image-20211130105935862](README/image-20211130105935862.png)

#### Kulla Conty Approximation

![image-20211231141039926](README/image-20211231141039926.png)

#### Curve Modeling

![image-20220108150839809](README/image-20220108150839809.png)

## Surface Modeling

![image-20220108151149909](README/image-20220108151149909.png)

#### Temporal Anti-Alias

![image-20220111121723309](README/image-20220111121723309.png)

## TODO

More features are on their way:

* Image based lighting
* Shadow Mapping
* RTX
* GI
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
