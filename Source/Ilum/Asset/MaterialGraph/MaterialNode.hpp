#pragma once

#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace Ilum
{
enum class PinType
{
	Float,
	Int,
	Uint,
	Float2,
	Int2,
	Uint2,
	Float3,
	Int3,
	Uint3,
	Float4,
	Int4,
	Uint4,

	Texture1D,
	Texture2D,
	Texture3D,
	RWTexture1D,
	RWTexture2D,
	RWTexture3D,

	SamplerState,

	BxDF,

	Invalid
};

namespace ShaderConstantPinType
{
using Float  = float;
using Int    = int32_t;
using Uint   = uint32_t;
using Float2 = glm::vec2;
using Int2   = glm::ivec2;
using Uint2  = glm::uvec2;
using Float3 = glm::vec3;
using Int3   = glm::ivec3;
using Uint3  = glm::uvec3;
using Float4 = glm::vec4;
using Int4   = glm::ivec4;
using Uint4  = glm::uvec4;
}        // namespace ShaderConstantPinType

#define ShaderConstant(Type) ShaderConstant##Type
#define PinType(Type) PinType::##Type

class MaterialGraphNode
{
  public:
	struct Pin
	{
		explicit Pin(PinType type_) :
		    pin_id(s_PinID++), type(type_)
		{
		}

		Pin(const Pin &) = delete;
		Pin(Pin &&)      = delete;
		Pin &operator=(const Pin &) = delete;
		Pin &operator=(Pin &&) = delete;

		PinType type = PinType::Invalid;

		size_t operator()()
		{
			return pin_id;
		}

	  private:
		inline static size_t s_PinID = 0;

		size_t pin_id = ~0;
	};

  public:
	MaterialGraphNode()  = default;
	~MaterialGraphNode() = default;

	inline const std::vector<Pin> &GetInputPins() const
	{
		return m_input_pins;
	}

	inline const std::vector<Pin> &GetOutputPins() const
	{
		return m_output_pins;
	}

	virtual void OnImGui()
	{
	}

	virtual void OnImnode()
	{
	}

	virtual std::string &GetResult() = 0;

  private:
	std::string m_name = "Untitled Node";

	std::vector<Pin> m_input_pins;
	std::vector<Pin> m_output_pins;
};

template <PinType T>
class ConstantMaterialGraphNode : public MaterialGraphNode
{
  public:
	ConstantMaterialGraphNode() :
	    m_output_pins({Pin(T)})
	{
	}

	virtual std::string GetResult() override
	{
		return std::to_string(m_data);
	}

  private:
	ShaderConstant(T) m_data;
};

using ConstantMaterialGraphNodeFloat = ConstantMaterialGraphNode<PinType::Float>;
using ConstantMaterialGraphNodeInt = ConstantMaterialGraphNode<PinType::Int>;
using ConstantMaterialGraphNodeUint = ConstantMaterialGraphNode<PinType::Uint>;
using ConstantMaterialGraphNodeFloat2 = ConstantMaterialGraphNode<PinType::Float2>;
using ConstantMaterialGraphNodeInt2 = ConstantMaterialGraphNode<PinType::Int2>;
using ConstantMaterialGraphNodeUint2 = ConstantMaterialGraphNode<PinType::Uint2>;
using ConstantMaterialGraphNodeFloat3 = ConstantMaterialGraphNode<PinType::Float3>;
using ConstantMaterialGraphNodeInt3 = ConstantMaterialGraphNode<PinType::Int3>;
using ConstantMaterialGraphNodeUint3 = ConstantMaterialGraphNode<PinType::Uint3>;
using ConstantMaterialGraphNodeFloat4 = ConstantMaterialGraphNode<PinType::Float4>;
using ConstantMaterialGraphNodeInt4 = ConstantMaterialGraphNode<PinType::Int4>;
using ConstantMaterialGraphNodeUint4 = ConstantMaterialGraphNode<PinType::Uint4>;
}        // namespace Ilum