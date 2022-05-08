#pragma once

#include "RGHandle.hpp"
#include "RenderGraph.hpp"

#include "RHI/PipelineState.hpp"

namespace Ilum
{
class RGBuilder;
class RenderPass
{
	friend class RGBuilder;

  public:
	RenderPass(const std::string &name);
	~RenderPass() = default;

	virtual void Create(RGBuilder &builder)  = 0;
	virtual void Build(RGPass &pass);

	const std::string &GetName() const;

	void AddResource(const RGHandle &handl);

	const std::vector<RGHandle> &GetResources() const;

	void BindCallback(std::function<void(CommandBuffer &, const RGResources &, Renderer &)> &&callback);

	void BindImGui(std::function<void(ImGuiContext &, const RGResources &)> &&callback);

	uint32_t GetHandle() const;

	void SetHandle(uint32_t handle);

  protected:
	std::string m_name;

	uint32_t m_handle;

	std::vector<RGHandle> m_resources;

	std::function<void(CommandBuffer &, const RGResources &, Renderer&)> m_callback;

	std::function<void(ImGuiContext &, const RGResources &)> m_imgui_callback;

  private:
	inline static uint32_t CURRENT_ID = 0;
};
}        // namespace Ilum