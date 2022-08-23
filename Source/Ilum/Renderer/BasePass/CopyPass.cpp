#include "CopyPass.hpp"

namespace Ilum::Pass
{
RenderPassDesc CopyPass::CreateDesc(size_t &handle)
{
	RenderPassDesc desc = {};

	desc.name = "CopyPass";
	desc.reads.emplace("Source", std::make_pair(RenderPassDesc::ResourceType::Texture, RGHandle(handle++)));
	desc.writes.emplace("Target", std::make_pair(RenderPassDesc::ResourceType::Texture, RGHandle(handle++)));

	return desc;
}
}        // namespace Ilum::Pass