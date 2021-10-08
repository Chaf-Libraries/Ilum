#pragma once

#include "Core/Engine/PCH.hpp"

#include "Math/Vector2.h"
#include "Math/Vector3.h"

#include "Core/Graphics/Pipeline/Shader.hpp"

namespace Ilum
{
struct Vertex
{
	Math::Vector3 position;
	Math::Vector3 normal;
	Math::Vector2 uv;
	Math::Vector3 tengent;

	static Shader::VertexInput getVertexInput();
};

struct InstanceData
{
	uint32_t index;

	static Shader::VertexInput getVertexInput();
};
}