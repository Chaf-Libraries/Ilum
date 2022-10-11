#include "VariableNode.hpp"
#include "MaterialGraph/MaterialGraph.hpp"

namespace Ilum::MGNode
{
MaterialNodeDesc RGB::Create(size_t &handle)
{
	MaterialNodeDesc desc;

	desc
	    .AddPin(handle, "RGB", MaterialNodePin::Type::Float3, MaterialNodePin::Attribute::Input, RGBColor())
		.AddPin(handle, "Out", MaterialNodePin::Type::Float3, MaterialNodePin::Attribute::Output)
	    .SetName<RGB>();

	return desc;
}

void RGB::EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, MaterialEmitInfo &info)
{

}

MaterialNodeDesc RGBA::Create(size_t &handle)
{
	MaterialNodeDesc desc;

	desc
	    .AddPin(handle, "RGBA", MaterialNodePin::Type::Float4, MaterialNodePin::Attribute::Input, RGBAColor())
		.AddPin(handle, "Out", MaterialNodePin::Type::Float4, MaterialNodePin::Attribute::Output)
	    .SetName<RGBA>();

	return desc;
}

void RGBA::EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, MaterialEmitInfo &info)
{

}
}        // namespace Ilum::MGNode