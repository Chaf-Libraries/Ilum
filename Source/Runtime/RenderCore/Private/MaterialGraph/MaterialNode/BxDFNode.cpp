#include "BxDFNode.hpp"

namespace Ilum::MGNode
{
MaterialNodeDesc BlendBxDFNode::Create(size_t &handle)
{
	MaterialNodeDesc desc;

	desc.SetName<BlendBxDFNode>()
	    .AddPin(handle, "LHS", MaterialNodePin::Type::BSDF, MaterialNodePin::Attribute::Input)
	    .AddPin(handle, "Out", MaterialNodePin::Type::BSDF, MaterialNodePin::Attribute::Output)
	    .AddPin(handle, "RHS", MaterialNodePin::Type::BSDF, MaterialNodePin::Attribute::Input)
	    .AddPin(handle, "weight", MaterialNodePin::Type::Float, MaterialNodePin::Attribute::Input, BlendWeight{});

	return desc;
}

void BlendBxDFNode::EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, MaterialEmitInfo &info)
{

}
}        // namespace Ilum::MGNode