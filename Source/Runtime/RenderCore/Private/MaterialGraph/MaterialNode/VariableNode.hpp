#pragma once

#include "RenderCore/MaterialGraph/MaterialGraph.hpp"
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
		const char                                       *pin_name[] = {"X", "Y", "Z", "W"};
		const std::unordered_map<rttr::type, std::string> type_name  = {
            {rttr::type::get<float>(), "float"},
            {rttr::type::get<bool>(), "bool"},
            {rttr::type::get<uint32_t>(), "int"},
            {rttr::type::get<int32_t>(), "uint"},
        };

		std::string components[N];

		for (int32_t i = 0; i < N; i++)
		{
			if (graph.HasLink(desc.GetPin(pin_name[i]).handle))
			{
				const auto &variable_desc = graph.GetNode(graph.LinkFrom(desc.GetPin(pin_name[i]).handle));
				auto        variable_node = rttr::type::get_by_name(variable_desc.name).create();
				variable_node.get_type().get_method("EmitHLSL").invoke(variable_node, variable_desc, graph, info);
				if (info.IsExpression(graph.LinkFrom(desc.GetPin(pin_name[i]).handle)))
				{
					components[i] = fmt::format("({})", info.expression.at(graph.LinkFrom(desc.GetPin(pin_name[i]).handle)));
				}
				else
				{
					components[i] = "S" + std::to_string(graph.LinkFrom(desc.GetPin(pin_name[i]).handle));
				}
			}
			else
			{
				components[i] = desc.GetPin(pin_name[i]).data.to_string();
			}
		}

		std::string shader_type_name = type_name.at(rttr::type::get<T>()) + (N == 1 ? std::string() : std::to_string(N));

		std::string defintion = shader_type_name + " S" + std::to_string(desc.GetPin("Out").handle) + " = " + shader_type_name + "(";

		for (int32_t i = 0; i < N - 1 && i >= 0; i++)
		{
			defintion += components[i] + ", ";
		}
		defintion += components[N - 1] + ");";

		info.definitions.push_back(defintion);
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
		if (graph.HasLink(desc.GetPin("In").handle))
		{
			const auto &variable_desc = graph.GetNode(graph.LinkFrom(desc.GetPin("In").handle));
			auto        variable_node = rttr::type::get_by_name(variable_desc.name).create();
			variable_node.get_type().get_method("EmitHLSL").invoke(variable_node, variable_desc, graph, info);

			MaterialNodePin::Type pin_type = variable_desc.GetPin(graph.LinkFrom(desc.GetPin("In").handle)).type;

			const char *pin_name[]  = {"X", "Y", "Z", "W"};
			const char *cmpt_name[] = {"x", "y", "z", "w"};

			const std::unordered_map<MaterialNodePin::Type, std::string> type_name = {
			    {MaterialNodePin::Type::Bool2, "bool"},
			    {MaterialNodePin::Type::Bool3, "bool"},
			    {MaterialNodePin::Type::Bool4, "bool"},
			    {MaterialNodePin::Type::Int2, "int"},
			    {MaterialNodePin::Type::Int3, "int"},
			    {MaterialNodePin::Type::Int4, "int"},
			    {MaterialNodePin::Type::Uint2, "uint"},
			    {MaterialNodePin::Type::Uint3, "uint"},
			    {MaterialNodePin::Type::Uint4, "uint"},
			    {MaterialNodePin::Type::Float2, "float"},
			    {MaterialNodePin::Type::Float3, "float"},
			    {MaterialNodePin::Type::Float4, "float"},
			};

			for (int32_t i = 0; i < N; i++)
			{
				info.expression.emplace(desc.GetPin(pin_name[i]).handle, fmt::format("S{}.{}", graph.LinkFrom(desc.GetPin("In").handle), cmpt_name[i]));
			}
		}
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