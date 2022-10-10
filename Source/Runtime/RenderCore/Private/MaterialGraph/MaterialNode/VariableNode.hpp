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

		for (int32_t i = 0; i < N; i++)
		{
			desc.AddPin(handle, pin_name[i], fmt::format("{}", typeid(T).name()), MaterialNodePin::Attribute::Input, T(0));
		}

		desc
		    .AddPin(handle, "Out", fmt::format("{}{}", typeid(T).name(), N == 1 ? "" : std::to_string(N)), MaterialNodePin::Attribute::Output)
		    .SetName<DerivedClass>();

		return desc;
	}

	virtual std::string EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, std::string &source) override
	{
		std::string output      = fmt::format("v{}", desc.GetPin("Out").handle);
		std::string output_type = desc.GetPin("Out").type;
		const char *pin_name[]  = {"X", "Y", "Z", "W"};

		source += fmt::format("{} {} = {}(", output_type, output, output_type);
		for (auto i = 0; i < N; i++)
		{
			source += std::to_string(desc.GetPin(pin_name[i]).data.convert<T>());
			if (i != N - 1)
			{
				source += ", ";
			}
		}
		source += ");\n";
		return output;
	}
};

template <typename DerivedClass, typename T, glm::length_t N>
struct FromVariableNode : public MaterialNode
{
	virtual MaterialNodeDesc Create(size_t &handle) override
	{
		MaterialNodeDesc desc;

		const char *pin_name[] = {"X", "Y", "Z", "W"};

		for (int32_t i = 0; i < N; i++)
		{
			desc.AddPin(handle, pin_name[i], fmt::format("{}", typeid(T).name()), MaterialNodePin::Attribute::Output);
		}

		desc
		    .AddPin(handle, "In", fmt::format("{}{}", typeid(T).name(), N == 1 ? "" : std::to_string(N)), MaterialNodePin::Attribute::Input, glm::vec<N, T>(0))
		    .SetName<DerivedClass>();

		return desc;
	}

	virtual std::string EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, std::string &source) override
	{
		std::string output = fmt::format("v{}", desc.GetPin("Out").handle);
		return output;
	}
};

#define DEINE_TO_VARIABLE_NODE(NAME, TYPE, LENGTH)                    \
	STRUCT(NAME, Enable, MaterialNode(#NAME), Category("Variable")) : \
	    public ToVariableNode<NAME, TYPE, LENGTH>                     \
	{                                                                 \
		RTTR_ENABLE(MaterialNode);                                    \
	};

#define DEINE_FROM_VARIABLE_NODE(NAME, TYPE, LENGTH)                      \
	STRUCT(NAME, Enable, MaterialNode(#NAME), Category("Split Vector")) : \
	    public FromVariableNode<NAME, TYPE, LENGTH>                       \
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

DEINE_FROM_VARIABLE_NODE(SplitFloat2, float, 2);
DEINE_FROM_VARIABLE_NODE(SplitUint2, uint32_t, 2);
DEINE_FROM_VARIABLE_NODE(SplitInt2, int32_t, 2);
DEINE_FROM_VARIABLE_NODE(SplitBool2, bool, 2);

DEINE_FROM_VARIABLE_NODE(SplitFloat3, float, 3);
DEINE_FROM_VARIABLE_NODE(SplitUint3, uint32_t, 3);
DEINE_FROM_VARIABLE_NODE(SplitInt3, int32_t, 3);
DEINE_FROM_VARIABLE_NODE(SplitBool3, bool, 3);

DEINE_FROM_VARIABLE_NODE(SplitFloat4, float, 4);
DEINE_FROM_VARIABLE_NODE(SplitUint4, uint32_t, 4);
DEINE_FROM_VARIABLE_NODE(SplitInt4, int32_t, 4);
DEINE_FROM_VARIABLE_NODE(SplitBool4, bool, 4);

STRUCT(RGB, Enable, MaterialNode("RGB"), Category("Color")) :
    public MaterialNode
{
	STRUCT(RGBColor, Enable)
	{
		META(Editor("ColorEdit"))
		glm::vec3 color = glm::vec3(0);
	};

	virtual MaterialNodeDesc Create(size_t & handle) override;

	virtual std::string EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, std::string &source) override;
};

STRUCT(RGBA, Enable, MaterialNode("RGBA"), Category("Color")) :
    public MaterialNode
{
	STRUCT(RGBAColor, Enable)
	{
		META(Editor("ColorEdit"))
		glm::vec4 color = glm::vec4(0);
	};

	virtual MaterialNodeDesc Create(size_t & handle) override;

	virtual std::string EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, std::string &source) override;
};

}        // namespace Ilum::MGNode