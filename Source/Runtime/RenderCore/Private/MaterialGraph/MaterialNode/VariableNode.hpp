#pragma once

#include "RenderCore/MaterialGraph/MaterialNode.hpp"

namespace Ilum::MGNode
{
template <typename DerivedClass, typename T, glm::length_t N>
struct ToVariableNode : public MaterialNode
{
	virtual MaterialNodeDesc Create(size_t &handle) override
	{
		MaterialNodeDesc desc;

		const char *pin_name[] = {"X", "Y", "Z", "W"};

		const std::unordered_map<rttr::type, MaterialNodePin::Type> pin_type = {
		    {rttr::type::get<glm::vec<1, bool>>(), MaterialNodePin::Type::Bool},
		    {rttr::type::get<glm::vec<2, bool>>(), MaterialNodePin::Type::Bool2},
		    {rttr::type::get<glm::vec<3, bool>>(), MaterialNodePin::Type::Bool3},
		    {rttr::type::get<glm::vec<4, bool>>(), MaterialNodePin::Type::Bool4},

		    {rttr::type::get<glm::vec<1, uint32_t>>(), MaterialNodePin::Type::Uint},
		    {rttr::type::get<glm::vec<2, uint32_t>>(), MaterialNodePin::Type::Uint2},
		    {rttr::type::get<glm::vec<3, uint32_t>>(), MaterialNodePin::Type::Uint3},
		    {rttr::type::get<glm::vec<4, uint32_t>>(), MaterialNodePin::Type::Uint4},

		    {rttr::type::get<glm::vec<1, int32_t>>(), MaterialNodePin::Type::Int},
		    {rttr::type::get<glm::vec<2, int32_t>>(), MaterialNodePin::Type::Int2},
		    {rttr::type::get<glm::vec<3, int32_t>>(), MaterialNodePin::Type::Int3},
		    {rttr::type::get<glm::vec<4, int32_t>>(), MaterialNodePin::Type::Int4},

		    {rttr::type::get<glm::vec<1, float>>(), MaterialNodePin::Type::Float},
		    {rttr::type::get<glm::vec<2, float>>(), MaterialNodePin::Type::Float2},
		    {rttr::type::get<glm::vec<3, float>>(), MaterialNodePin::Type::Float3},
		    {rttr::type::get<glm::vec<4, float>>(), MaterialNodePin::Type::Float4},
		};

		for (int32_t i = 0; i < N; i++)
		{
			desc.AddPin(handle, pin_name[i], MaterialNodePin::Type::Bool | MaterialNodePin::Type::Uint | MaterialNodePin::Type::Int | MaterialNodePin::Type::Float, MaterialNodePin::Attribute::Input, T(0));
		}

		desc
		    .AddPin(handle, "Out", pin_type.at(rttr::type::get<glm::vec<N, T>>()), MaterialNodePin::Attribute::Output)
		    .SetName<DerivedClass>();

		return desc;
	}

	virtual void EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, MaterialEmitInfo &info) override
	{
	}
};

template <typename DerivedClass, glm::length_t N>
struct SplitVectorNode : public MaterialNode
{
	virtual MaterialNodeDesc Create(size_t &handle) override
	{
		MaterialNodeDesc desc;

		const char *pin_name[] = {"X", "Y", "Z", "W"};

		const std::unordered_map<glm::length_t, MaterialNodePin::Type> pin_type = {
		    {2, MaterialNodePin::Type::Bool2 | MaterialNodePin::Type::Float2 | MaterialNodePin::Type::Int2 | MaterialNodePin::Type::Uint2},
		    {3, MaterialNodePin::Type::Bool3 | MaterialNodePin::Type::Float3 | MaterialNodePin::Type::Int3 | MaterialNodePin::Type::Uint3},
		    {4, MaterialNodePin::Type::Bool4 | MaterialNodePin::Type::Float4 | MaterialNodePin::Type::Int4 | MaterialNodePin::Type::Uint4},
		};

		for (int32_t i = 0; i < N; i++)
		{
			desc.AddPin(handle, pin_name[i], MaterialNodePin::Type::Bool | MaterialNodePin::Type::Float | MaterialNodePin::Type::Int | MaterialNodePin::Type::Uint, MaterialNodePin::Attribute::Output);
		}

		desc.AddPin(handle, "In", pin_type.at(N), MaterialNodePin::Attribute::Input)
		    .SetName<DerivedClass>();

		return desc;
	}

	virtual void EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, MaterialEmitInfo &info) override
	{
		std::string output = fmt::format("v{}", desc.GetPin("Out").handle);
	}
};

#define DEINE_TO_VARIABLE_NODE(NAME, TYPE, LENGTH)                    \
	STRUCT(NAME, Enable, MaterialNode(#NAME), Category("Variable")) : \
	    public ToVariableNode<NAME, TYPE, LENGTH>                     \
	{                                                                 \
		RTTR_ENABLE(MaterialNode);                                    \
	};

#define DEINE_FROM_VARIABLE_NODE(NAME, LENGTH)                            \
	STRUCT(NAME, Enable, MaterialNode(#NAME), Category("Split Vector")) : \
	    public SplitVectorNode<NAME, LENGTH>                              \
	{                                                                     \
		RTTR_ENABLE(MaterialNode);                                        \
	};

DEINE_TO_VARIABLE_NODE(Float, float, 1);
DEINE_TO_VARIABLE_NODE(Uint, uint32_t, 1);
DEINE_TO_VARIABLE_NODE(Int, int32_t, 1);
DEINE_TO_VARIABLE_NODE(Bool, bool, 1);

DEINE_TO_VARIABLE_NODE(Float2, float, 2);
DEINE_TO_VARIABLE_NODE(Uint2, uint32_t, 2);
DEINE_TO_VARIABLE_NODE(Int2, int32_t, 2);
DEINE_TO_VARIABLE_NODE(Bool2, bool, 2);

DEINE_TO_VARIABLE_NODE(Float3, float, 3);
DEINE_TO_VARIABLE_NODE(Uint3, uint32_t, 3);
DEINE_TO_VARIABLE_NODE(Int3, int32_t, 3);
DEINE_TO_VARIABLE_NODE(Bool3, bool, 3);

DEINE_TO_VARIABLE_NODE(Float4, float, 4);
DEINE_TO_VARIABLE_NODE(Uint4, uint32_t, 4);
DEINE_TO_VARIABLE_NODE(Int4, int32_t, 4);
DEINE_TO_VARIABLE_NODE(Bool4, bool, 4);

DEINE_FROM_VARIABLE_NODE(SplitVec2, 2);
DEINE_FROM_VARIABLE_NODE(SplitVec3, 3);
DEINE_FROM_VARIABLE_NODE(SplitVec4, 4);

STRUCT(RGB, Enable, MaterialNode("RGB"), Category("Color")) :
    public MaterialNode
{
	STRUCT(RGBColor, Enable)
	{
		META(Editor("ColorEdit"), Name("Color"))
		glm::vec3 color = glm::vec3(0);
	};

	virtual MaterialNodeDesc Create(size_t & handle) override;

	virtual void EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, MaterialEmitInfo &info) override;
};

STRUCT(RGBA, Enable, MaterialNode("RGBA"), Category("Color")) :
    public MaterialNode
{
	STRUCT(RGBAColor, Enable)
	{
		META(Editor("ColorEdit"), Name(""))
		glm::vec4 color = glm::vec4(0);
	};

	virtual MaterialNodeDesc Create(size_t & handle) override;

	virtual void EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, MaterialEmitInfo &info) override;
};

}        // namespace Ilum::MGNode