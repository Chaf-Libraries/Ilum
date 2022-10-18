#pragma once

#include "RenderCore/MaterialGraph/MaterialNode.hpp"

namespace Ilum::MGNode
{
STRUCT(ScalarCalculation, Enable, MaterialNode("Scalar Calculation"), Category("Convert")) :
    public MaterialNode
{
	virtual MaterialNodeDesc Create(size_t & handle) override;

	virtual void Validate(const MaterialNodeDesc &node, MaterialGraph *graph, ShaderValidateContext &context) override;

	virtual void Update(MaterialNodeDesc & node) override;

	virtual void EmitShader(const MaterialNodeDesc &desc, MaterialGraph *graph, ShaderEmitContext &context) override;

	ENUM(Type, Enable){
	    Addition,
	    Substrate,
	    Multiplication,
	    Division,

	    Maximum,
	    Minimum,

	    Greater,
	    Less,

	    Square,
	    Log,
	    Exp,
	    Sqrt,
	    Rcp,
	    Abs,
	    Sign,

	    Sin,
	    Cos,
	    Tan,

	    Asin,
	    Acos,
	    Atan,
	    Atan2,

	    Sinh,
	    Cosh,
	    Tanh,
	};

	STRUCT(ScalarCalculationType, Enable)
	{
		Type type;
	};
};

STRUCT(VectorCalculation, Enable, MaterialNode("Vector Calculation"), Category("Convert")) :
    public MaterialNode
{
	virtual MaterialNodeDesc Create(size_t & handle) override;

	virtual void Validate(const MaterialNodeDesc &node, MaterialGraph *graph, ShaderValidateContext &context) override;

	virtual void Update(MaterialNodeDesc & node) override;

	virtual void EmitShader(const MaterialNodeDesc &desc, MaterialGraph *graph, ShaderEmitContext &context) override;

	ENUM(Type, Enable){
	    Scale,
	    Length,
	    Distance,

	    Dot,
	    Cross,

	    Addition,
	    Substrate,
	    Multiplication,
	    Division,

	    Sin,
	    Cos,
	    Tan,

	    Maximum,
	    Minimum,

	    Abs,

	    Normalize,
	};

	STRUCT(VectorCalculationType, Enable)
	{
		Type type;
	};
};
}        // namespace Ilum::MGNode