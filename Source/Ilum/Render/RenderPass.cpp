#include "RenderPass.hpp"

#include <imgui.h>
#include <imnodes.h>

namespace Ilum
{
RenderPass::RenderPass(const std::string &name) :
    m_name(name)
{
}

void RenderPass::Build(RGPass &pass)
{
	Prepare(pass.m_pso);
	pass.m_execute_callback = m_callback;
	pass.m_imgui_callback   = m_imgui_callback;
}

const std::string &RenderPass::GetName() const
{
	return m_name;
}

void RenderPass::AddResource(const RGHandle &handle)
{
	m_resources.push_back(handle);
}

const std::vector<RGHandle> &RenderPass::GetResources() const
{
	return m_resources;
}

void RenderPass::BindCallback(std::function<void(CommandBuffer &, PipelineState &, const RGResources &)> &&callback)
{
	m_callback = std::move(callback);
}

void RenderPass::BindImGui(std::function<void(ImGuiContext &, const RGResources &)> &&callback)
{
	m_imgui_callback = std::move(callback);
}
}        // namespace Ilum
