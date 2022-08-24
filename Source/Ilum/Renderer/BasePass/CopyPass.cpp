#include "CopyPass.hpp"

namespace Ilum::Pass
{
RenderPassDesc CopyPass::CreateDesc(size_t &handle)
{
	RenderPassDesc desc = {};

	desc.name = "CopyPass";
	desc
		.Read("Source", RenderPassDesc::ResourceInfo::Type::Texture, RHIResourceState::ShaderResource, handle)
		.Write("Target", RenderPassDesc::ResourceInfo::Type::Texture, RHIResourceState::RenderTarget, handle);

	return desc;
}
}        // namespace Ilum::Pass