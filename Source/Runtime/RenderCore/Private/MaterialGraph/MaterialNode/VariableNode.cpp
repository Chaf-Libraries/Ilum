#include "VariableNode.hpp"
#include "MaterialGraph/MaterialGraph.hpp"

namespace Ilum::MGNode
{
MaterialNodeDesc RGB::Create(size_t &handle)
{
	MaterialNodeDesc desc;

	desc.AddPin(handle, "Out", "float3", MaterialNodePin::Attribute::Output, RGBColor())
	    .SetName<RGB>();

	return desc;
}

std::string RGB::EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, std::string &source)
{
	std::string output      = fmt::format("v{}", desc.GetPin("Out").handle);
	std::string output_type = desc.GetPin("Out").type;
	RGBColor    data        = desc.GetPin("Out").data.convert<RGBColor>();
	source += fmt::format("float3 {} = float3({}, {}, {});\n", output, data.color.x, data.color.y, data.color.z);
	return output;
}

MaterialNodeDesc RGBA::Create(size_t &handle)
{
	MaterialNodeDesc desc;

	desc.AddPin(handle, "Out", "float4", MaterialNodePin::Attribute::Output, RGBAColor())
	    .SetName<RGBA>();

	return desc;
}

std::string RGBA::EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, std::string &source)
{
	std::string output      = fmt::format("v{}", desc.GetPin("Out").handle);
	std::string output_type = desc.GetPin("Out").type;
	RGBAColor   data        = desc.GetPin("Out").data.convert<RGBAColor>();
	source += fmt::format("float4 {} = float4({}, {}, {});\n", output, data.color.x, data.color.y, data.color.z, data.color.w);
	return output;
}
}        // namespace Ilum::MGNode