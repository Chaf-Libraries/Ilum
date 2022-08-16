#pragma once

#include <RHI/RHIContext.hpp>

#include <map>

namespace Ilum
{
class RenderPass;
class RenderResource;

struct RenderPassDesc
{
	std::string name;

	std::map<std::string, std::pair<uint32_t, RHIResourceState>> resources;
};

struct ResourceDesc
{
	std::string name;
	uint32_t    handle;
};

struct RGTextureDesc : ResourceDesc
{
	TextureDesc desc;
};

struct RGBufferDesc : ResourceDesc
{
	BufferDesc desc;
};

struct RenderGraphDesc
{
	std::vector<RenderPassDesc> passes;
	std::vector<RenderGraphDesc> subgraphs;
};

class RenderGraphBuilder
{
  public:
	RenderGraphBuilder(RHIContext *context);

	~RenderGraphBuilder();

	void Compile();

  private:
	RHIContext *p_context = nullptr;

	std::vector<std::unique_ptr<RenderPass>> m_render_passes;
	std::vector<RenderPass *>                m_executable_passes;
};
}        // namespace Ilum