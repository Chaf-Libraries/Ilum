#pragma once

#include "RGHandle.hpp"
#include "RenderGraph.hpp"

#include "RHI/PipelineState.hpp"

namespace Ilum
{
class RGBuilder;
class RenderPass
{
  public:
	RenderPass(const std::string &name);
	~RenderPass() = default;

	virtual void Prepare(PipelineState &pso) = 0;
	virtual void Create(RGBuilder &builder)  = 0;
	virtual void Build(RGPass &pass);

	const std::string &GetName() const;

	void AddResource(const RGHandle &handl);

	const std::vector<RGHandle> &GetResources() const;

	void BindCallback(std::function<void(CommandBuffer &, PipelineState &, const RGResources &)> &&callback);

	void BindImGui(std::function<void(ImGuiContext &, const RGResources &)> &&callback);

  protected:
	std::string m_name;

	std::vector<RGHandle> m_resources;

	std::function<void(CommandBuffer &, PipelineState &, const RGResources &)> m_callback;

	std::function<void(ImGuiContext &, const RGResources &)> m_imgui_callback;
};
}        // namespace Ilum