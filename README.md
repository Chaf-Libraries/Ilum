# Ilum(WIP)

[![Windows](https://github.com/Chaf-Libraries/Ilum/actions/workflows/windows.yml/badge.svg)](https://github.com/Chaf-Libraries/Ilum/actions/workflows/windows.yml) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/b0cb3a2729ee4be783dd5feb2cc67eb6)](https://www.codacy.com/gh/Chaf-Libraries/IlumEngine/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Chaf-Libraries/IlumEngine&amp;utm_campaign=Badge_Grade)

Ilum Graphics Playground, name after *Planet Ilum* from [Star Wars](https://starwars.fandom.com/es/wiki/Ilum)

![image-20220313211708122](README/image-20220313211708122.png)

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

## Vulkan Requirement

* Instance Extension
  * `VK_KHR_surface`
  * `VK_KHR_win32_surface`
  * `VK_EXT_debug_report`
  * `VK_EXT_debug_utils`
* Validation Layers
  * `VK_LAYER_KHRONOS_validation`
* Device Extension
  * `VK_KHR_swapchain`
  * `VK_KHR_acceleration_structure`
  * `VK_KHR_ray_tracing_pipeline`

## Feature

* Architecture
  * Render Graph
    * Customize Render Pass (Graphics, Compute, Ray Tracing)
    * Auto Resource Transition
    * Render Passes Visualization
    * Render Pass Setting
  * Runtime Shader Compilation
    * GLSL -> `glslang` -> SPIR-V
    * HLSL -> `DXC` -> SPIR-V
  * Entity Component System
  * Asynchronous Resource Loading
  * Scene Loading/Saving
* Rendering Feature For Performance
  * Multi-Draw Indirect
  * Bindless Texture
  * Vertex/Index Buffer Packing
  * GPU Frustum Culling
  * GPU Back-Face Cone Culling
  * GPU Hierarchy Z Buffer Occlusion Culling
* Geometry
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
  * Triangle Mesh Processing
    * Data Structure
      * Face Based Mesh
      * Edge Based Mesh
      * Half-Edge Mesh
    * Subdivision
      * Loop Subdivision
* Lighting Model
  * PBR
    * Cook-Torrance BRDF
    * Kulla-Conty Approximation
  * Shadow
    * Shadow Map -> Spot Light
    * Cascade Shadow Map -> Directional Light
    * Omnidirectional Shadow Map -> Point Light
* Post Processing
  * Temporal Anti-Alias
  * Blooming


## Demo

### Cook-Torrance BRDF

![image-20211120113603895](README/image-20211120113603895.png)

### Kulla-Conty Mutli-Bounce Approximation

| Multi-Bounce OFF                               | Multi-Bounce ON                              |
| ---------------------------------------------- | -------------------------------------------- |
| ![kulla_conty_off](README/kulla_conty_off.png) | ![kulla_conty_on](README/kulla_conty_on.png) |

### Render Passes Visualization

![image-20211120113259237](README/image-20211120113259237.png)

### Meshlet

![image-20211130105935862](README/image-20211130105935862.png)

### Hierarchy Z Buffer Generation

![image-20211210113933024](README/image-20211210113933024.png)

### Massive Scene Rendering

![image-20220302110444007](README/image-20220302110444007.png)

### Temporal Anti-Alias

|           TAA OFF            |         TAA ON         |
| :--------------------------: | :--------------------: |
| ![no_taa](README/no_taa.png) | ![taa](README/taa.png) |

### Blooming

| Blooming OFF                       | Blooming ON                      |
| ---------------------------------- | -------------------------------- |
| ![bloom_off](README/bloom_off.png) | ![bloom_on](README/bloom_on.png) |

### Shadow Map(Spot Light Shadow)

![spot_light_shadow](README/spot_light_shadow.png)

### Cascade Shadow Map(Directional Light Shadow)

#### Frustum Split

![frustum_split](README/frustum_split.png)

#### Directional Light Shadow

![directional_light_shadow](README/directional_light_shadow.png)

### Omnidirectional Shadow Map(Point Light Shadow)

![point_light_shadow](README/point_light_shadow.png)

### Soft Shadow Filter

#### PCF

|           PCF OFF            |            Uniform Sampling            |            Poisson Sampling            |
| :--------------------------: | :------------------------------------: | :------------------------------------: |
| ![no_pcf](README/no_pcf.png) | ![uniform_pcf](README/uniform_pcf.png) | ![poisson_pcf](README/poisson_pcf.png) |

#### PCSS

|           PCSS OFF           |             Uniform Sampling             |             Poisson Sampling             |
| :--------------------------: | :--------------------------------------: | :--------------------------------------: |
| ![no_pcf](README/no_pcf.png) | ![uniform_pcss](README/uniform_pcss.png) | ![poisson_pcss](README/poisson_pcss.png) |

### Curve Modeling

![image-20220108150839809](README/image-20220108150839809.png)

### Surface Modeling

![image-20220108151149909](README/image-20220108151149909.png)

### Loop Subdivision

|           Origin           |        Iteration #1        |        Iteration #2        |        Iteration #3        |        Iteration #4        |
| :------------------------: | :------------------------: | :------------------------: | :------------------------: | :------------------------: |
| ![loop0](README/loop0.png) | ![loop1](README/loop1.png) | ![loop2](README/loop2.png) | ![loop3](README/loop3.png) | ![loop4](README/loop4.png) |

### Minimum Surface

|          Origin          |             Minimum Surface              |
| :----------------------: | :--------------------------------------: |
| ![face](README/face.png) | ![mini_surface](README/mini_surface.png) |

### Tutte Parameterization

**Origin Mesh**

<img src="README/origin.png" alt="origin" style="zoom:50%;" />

|                                       |                       UV Visualization                       |              Parameterization Visualization              |
| :-----------------------------------: | :----------------------------------------------------------: | :------------------------------------------------------: |
|   Circle Boundary + Uniform Weight    |         ![uniform_circle](README/uniform_circle.png)         |   ![uniform_circle_vis](README/uniform_circle_vis.png)   |
|  Rectangle Boundary + Uniform Weight  |       ![uniform_rectange](README/uniform_rectange.png)       | ![uniform_rectange_vis](README/uniform_rectange_vis.png) |
|  Circle Boundary + Cotangent Weight   |       ![cotangent_circle](README/cotangent_circle.png)       | ![cotangent_circle_vis](README/cotangent_circle_vis.png) |
| Rectangle Boundary + Cotangent Weight | ![cotangent_rectange_vis](README/cotangent_rectange_vis.png) |   ![cotangent_rectange](README/cotangent_rectange.png)   |

