#pragma once

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

	virtual void EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, MaterialEmitInfo& info) override
	{

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