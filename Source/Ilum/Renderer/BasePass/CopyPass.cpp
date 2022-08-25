#include "CopyPass.hpp"

namespace Ilum::Pass
{
RenderPassDesc CopyPass::CreateDesc()
{
	RenderPassDesc desc = {};

	desc.name = "CopyPass";
	desc
		.Read("Source", RenderPassDesc::ResourceInfo::Type::Texture, RHIResourceState::ShaderResource)
		.Write("Target", RenderPassDesc::ResourceInfo::Type::Texture, RHIResourceState::RenderTarget);

	return desc;
}
}        // namespace Ilum::Pass