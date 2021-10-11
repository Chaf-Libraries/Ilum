#pragma once

#include "Utils/PCH.hpp"

#include "Math/Vector2.h"
#include "Math/Vector3.h"

#include "Graphics/Pipeline/Shader.hpp"

namespace Ilum
{
struct VertexInputState;

struct Vertex
{
	Math::Vector3 position;
	Math::Vector3 normal;
	Math::Vector2 uv;
	Math::Vector3 tengent;
};

struct InstanceData
{
	uint32_t index;
};

template<typename... T>
static VertexInputState getVertexInput()
{
	return {};
}
}

#include "Vertex.inl"