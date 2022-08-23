#include "VisibilityBufferPass.hpp"

namespace Ilum::Pass
{
RenderPassDesc VisibilityBufferPass::CreateDesc(size_t &handle)
{
	RenderPassDesc desc = {};

	desc.name = "VisibilityBufferPass";
	desc.writes.emplace("VisibilityBuffer", std::make_pair(RenderPassDesc::ResourceType::Texture, RGHandle(handle++)));
	desc.writes.emplace("DepthBuffer", std::make_pair(RenderPassDesc::ResourceType::Texture, RGHandle(handle++)));

	desc.variant = rttr::type::get_by_name("VisibilityBufferPass::Config").create();

	return desc;
}
}        // namespace Ilum::Pass