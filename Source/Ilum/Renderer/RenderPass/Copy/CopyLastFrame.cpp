#pragma once

#include "CopyLastFrame.hpp"

#include "Renderer/Renderer.hpp"
#include "Renderer/RenderGraph/RenderGraph.hpp"

namespace Ilum::pass
{
CopyLastFrame::CopyLastFrame(const std::string &last_frame_name):
    m_last_frame_name(last_frame_name)
{
}

void CopyLastFrame::setupPipeline(PipelineState &state)
{
	state.addDependency(m_last_frame_name, VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
	state.addDependency("LastFrame", VK_IMAGE_USAGE_TRANSFER_DST_BIT);
}

void CopyLastFrame::resolveResources(ResolveState &resolve)
{
}

void CopyLastFrame::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	cmd_buffer.copyImage(
	    ImageInfo{state.graph.getAttachment(m_last_frame_name), VK_IMAGE_USAGE_TRANSFER_SRC_BIT},
	    ImageInfo{state.graph.getAttachment("LastFrame"), VK_IMAGE_USAGE_TRANSFER_DST_BIT});
}
}        // namespace Ilum::pass