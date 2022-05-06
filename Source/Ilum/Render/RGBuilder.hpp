#pragma once

#include "RGHandle.hpp"

#include <RHI/Buffer.hpp>
#include <RHI/Texture.hpp>

#include <map>

namespace Ilum
{
class RenderPass;
class RenderGraph;

class RGBuilder
{
  public:
	RGBuilder(RHIDevice *device, RenderGraph &graph);
	~RGBuilder();

	RGHandle CreateTexture(const std::string &name, const TextureDesc &desc, const TextureState &state);
	RGHandle CreateBuffer(const std::string &name, const BufferDesc &desc, const BufferState &state);

	RGBuilder &AddPass(std::unique_ptr<RenderPass> &&pass);

	RGBuilder &Link(const RGHandle &from, const RGHandle &to);

	void Compile();

	void OnImGui();

  private:
	RHIDevice *p_device = nullptr;
	RenderGraph &m_graph;

	std::vector<std::unique_ptr<RenderPass>> m_render_passes;

	std::map<RGHandle, std::unique_ptr<ResourceDeclaration>> m_resources;

	std::map<uint32_t, std::pair<RGHandle, RGHandle>> m_edges;

  private:
	static std::vector<std::string> s_avaliable_passes;
};
}        // namespace Ilum