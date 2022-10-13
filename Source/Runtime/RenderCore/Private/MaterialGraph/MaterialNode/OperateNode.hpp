#pragma once

#include <RenderCore/MaterialGraph/MaterialGraph.hpp>
#include <RenderCore/MaterialGraph/MaterialNode.hpp>

namespace Ilum::MGNode
{
template <typename T>
struct BinaryOpNode : public MaterialNode
{
	std::string op = "";

	BinaryOpNode(const std::string &op) :
	    op(op)
	{
	}

	virtual MaterialNodeDesc Create(size_t &handle) override
	{
		MaterialNodeDesc desc;

		MaterialNodePin::Type variable_type =
		    MaterialNodePin::Type::Float | MaterialNodePin::Type::Int | MaterialNodePin::Type::Uint |
		    MaterialNodePin::Type::Float2 | MaterialNodePin::Type::Int2 | MaterialNodePin::Type::Uint2 |
		    MaterialNodePin::Type::Float3 | MaterialNodePin::Type::Int3 | MaterialNodePin::Type::Uint3 |
		    MaterialNodePin::Type::Float4 | MaterialNodePin::Type::Int4 | MaterialNodePin::Type::Uint4;

		desc
		    .AddPin(handle, "Out", variable_type, MaterialNodePin::Attribute::Output)
		    .AddPin(handle, "LHS", variable_type, MaterialNodePin::Attribute::Input)
		    .AddPin(handle, "RHS", variable_type, MaterialNodePin::Attribute::Input)
		    .SetName<T>();

		return desc;
	}

	inline std::string GetType(MaterialNodePin::Type type)
	{
		switch (type)
		{
			case MaterialNodePin::Type::Float:
			case MaterialNodePin::Type::Float2:
			case MaterialNodePin::Type::Float3:
			case MaterialNodePin::Type::Float4:
				return "float";
			case MaterialNodePin::Type::Int:
			case MaterialNodePin::Type::Int2:
			case MaterialNodePin::Type::Int3:
			case MaterialNodePin::Type::Int4:
				return "int";
			case MaterialNodePin::Type::Uint:
			case MaterialNodePin::Type::Uint2:
			case MaterialNodePin::Type::Uint3:
			case MaterialNodePin::Type::Uint4:
				return "uint";
			case MaterialNodePin::Type::Bool:
			case MaterialNodePin::Type::Bool2:
			case MaterialNodePin::Type::Bool3:
			case MaterialNodePin::Type::Bool4:
				return "bool";
			default:
				break;
		}
		return "";
	}

	inline int32_t GetLength(MaterialNodePin::Type type)
	{
		switch (type)
		{
			case MaterialNodePin::Type::Float:
			case MaterialNodePin::Type::Int:
			case MaterialNodePin::Type::Uint:
			case MaterialNodePin::Type::Bool:
				return 1;
			case MaterialNodePin::Type::Float2:
			case MaterialNodePin::Type::Int2:
			case MaterialNodePin::Type::Uint2:
			case MaterialNodePin::Type::Bool2:
				return 2;
			case MaterialNodePin::Type::Float3:
			case MaterialNodePin::Type::Int3:
			case MaterialNodePin::Type::Uint3:
			case MaterialNodePin::Type::Bool3:
				return 3;
			case MaterialNodePin::Type::Float4:
			case MaterialNodePin::Type::Int4:
			case MaterialNodePin::Type::Uint4:
			case MaterialNodePin::Type::Bool4:
				return 4;
			default:
				break;
		}
		return 0;
	}

	virtual void EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, MaterialEmitInfo &info) override
	{
		const std::unordered_map<MaterialNodePin::Type, std::string> type_name = {
		    {MaterialNodePin::Type::Bool, "Bool"},
		    {MaterialNodePin::Type::Bool2, "Bool2"},
		    {MaterialNodePin::Type::Bool3, "Bool3"},
		    {MaterialNodePin::Type::Bool4, "Bool4"},
		    {MaterialNodePin::Type::Int, "Int"},
		    {MaterialNodePin::Type::Int2, "Int2"},
		    {MaterialNodePin::Type::Int3, "Int3"},
		    {MaterialNodePin::Type::Int4, "Int4"},
		    {MaterialNodePin::Type::Uint, "Uint"},
		    {MaterialNodePin::Type::Uint2, "Uint2"},
		    {MaterialNodePin::Type::Uint3, "Uint3"},
		    {MaterialNodePin::Type::Uint4, "Uint4"},
		    {MaterialNodePin::Type::Float, "Float"},
		    {MaterialNodePin::Type::Float2, "Float2"},
		    {MaterialNodePin::Type::Float3, "Float3"},
		    {MaterialNodePin::Type::Float4, "Float4"},
		};

		MaterialNodePin::Type lhs_type = MaterialNodePin::Type::Float;
		MaterialNodePin::Type rhs_type = MaterialNodePin::Type::Float;

		std::string lhs = graph.GetEmitExpression(desc, "LHS", info);
		std::string rhs = graph.GetEmitExpression(desc, "RHS", info);

		size_t lhs_src = graph.LinkFrom(desc.GetPin("LHS").handle);
		size_t rhs_src = graph.LinkFrom(desc.GetPin("RHS").handle);

		lhs_type = graph.GetNode(graph.LinkFrom(lhs_src)).GetPin(lhs_src).type;
		rhs_type = graph.GetNode(graph.LinkFrom(rhs_src)).GetPin(rhs_src).type;

		if (lhs_type == rhs_type)
		{
			info.expression.emplace(desc.GetPin("Out").handle, fmt::format("({} {} {})", lhs, op, rhs));
		}
		else
		{
			info.expression.emplace(desc.GetPin("Out").handle, fmt::format("({} {} Cast{}({}))", lhs, op, type_name.at(lhs_type), rhs));
		}

		// if (graph.HasLink(desc.GetPin("LHS").handle))
		//{
		//	const auto &variable_desc = graph.GetNode(graph.LinkFrom(desc.GetPin("LHS").handle));
		//	auto        variable_node = rttr::type::get_by_name(variable_desc.name).create();
		//	variable_node.get_type().get_method("EmitHLSL").invoke(variable_node, variable_desc, graph, info);

		//	lhs_type = variable_desc.GetPin(graph.LinkFrom(desc.GetPin("LHS").handle)).type;

		//	lhs = "S" + std::to_string(graph.LinkFrom(desc.GetPin("LHS").handle));
		//}

		// if (graph.HasLink(desc.GetPin("RHS").handle))
		//{
		//	const auto &variable_desc = graph.GetNode(graph.LinkFrom(desc.GetPin("RHS").handle));
		//	auto        variable_node = rttr::type::get_by_name(variable_desc.name).create();
		//	variable_node.get_type().get_method("EmitHLSL").invoke(variable_node, variable_desc, graph, info);

		//	rhs_type = variable_desc.GetPin(graph.LinkFrom(desc.GetPin("RHS").handle)).type;

		//	rhs = "S" + std::to_string(graph.LinkFrom(desc.GetPin("RHS").handle));
		//}

		// if (lhs_type == rhs_type)
		//{
		//	info.expression.emplace(desc.GetPin("Out").handle, fmt::format("({} + {})", lhs, rhs));
		// }
		// else
		//{
		//	info.expression.emplace(desc.GetPin("Out").handle, fmt::format("({} {} Cast{}({}))", lhs, op, type_name.at(lhs_type), rhs));
		// }
	}
};

#define DEFINE_BINARY_OP_NODE(NAME, OP)                              \
	STRUCT(NAME, Enable, MaterialNode(#NAME), Category("Operate")) : \
	    BinaryOpNode<NAME>                                           \
	{                                                                \
		NAME() :                                                     \
		    BinaryOpNode<NAME>(#OP)                                  \
		{}                                                           \
		RTTR_ENABLE(MaterialNode);                                   \
	};

DEFINE_BINARY_OP_NODE(Add, +)
DEFINE_BINARY_OP_NODE(Sub, -)
DEFINE_BINARY_OP_NODE(Mul, *)
DEFINE_BINARY_OP_NODE(Div, /)

}        // namespace Ilum::MGNode