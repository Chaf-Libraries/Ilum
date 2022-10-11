#include "ResourceNode.hpp"

namespace Ilum::MGNode
{
MaterialNodeDesc ExternalTexture::Create(size_t &handle)
{
	MaterialNodeDesc desc;

	desc.AddPin(handle, "Out", MaterialNodePin::Type::Texture2D, MaterialNodePin::Attribute::Output)
	    .SetData(TextureID())
	    .SetName<ExternalTexture>();

	return desc;
}

void ExternalTexture::EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, MaterialEmitInfo &info)
{

}

MaterialNodeDesc SamplerState::Create(size_t &handle)
{
	MaterialNodeDesc desc;

	desc
	    .AddPin(handle, "In", MaterialNodePin::Type::Texture2D | MaterialNodePin::Type::Texture3D | MaterialNodePin::Type::TextureCube, MaterialNodePin::Attribute::Input)
	    .SetData(SamplerStateType{})
	    .AddPin(handle, "Out",
	            MaterialNodePin::Type::Float | MaterialNodePin::Type::Int | MaterialNodePin::Type::Uint |
	                MaterialNodePin::Type::Float2 | MaterialNodePin::Type::Int2 | MaterialNodePin::Type::Uint2 |
	                MaterialNodePin::Type::Float3 | MaterialNodePin::Type::Int3 | MaterialNodePin::Type::Uint3 |
	                MaterialNodePin::Type::Float4 | MaterialNodePin::Type::Int4 | MaterialNodePin::Type::Uint4,
	            MaterialNodePin::Attribute::Output)
	    .SetName<SamplerState>();

	return desc;
}

void SamplerState::EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, MaterialEmitInfo &info)
{

}
}        // namespace Ilum::MGNode