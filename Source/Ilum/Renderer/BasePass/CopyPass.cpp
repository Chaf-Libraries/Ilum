#include "CopyPass.hpp"

namespace Ilum::Pass
{
RenderPassDesc CopyPass::CreateDesc()
{
	RenderPassDesc desc = {};

	desc.name = "CopyPass";
	desc
		.Read("Source", RenderResourceDesc::Type::Texture, RHIResourceState::ShaderResource)
		.Write("Target", RenderResourceDesc::Type::Texture, RHIResourceState::RenderTarget);

	return desc;
}
}        // namespace Ilum::Pass