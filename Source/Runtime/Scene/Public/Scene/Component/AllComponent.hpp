#pragma once

#include "HierarchyComponent.hpp"
#include "LightComponent.hpp"
#include "MeshComponent.hpp"
#include "TagComponent.hpp"
#include "TransformComponent.hpp"

#define ALL_COMPONENTS Ilum::TagComponent, Ilum::HierarchyComponent, Ilum::TransformComponent, Ilum::StaticMeshComponent
#define FIXED_COMPONENTS Ilum::TagComponent, Ilum::HierarchyComponent, Ilum::TransformComponent
#define MESH_COMPONENTS Ilum::MeshComponent, Ilum::StaticMeshComponent
#define LIGHT_COMPONENTS Ilum::LightComponent, Ilum::PointLightComponent, Ilum::SpotLightComponent, Ilum::DirectionalLightComponent, Ilum::RectLightComponent