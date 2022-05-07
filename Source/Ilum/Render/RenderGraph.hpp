#pragma once

#include "RGHandle.hpp"

#include <RHI/Command.hpp>
#include <RHI/PipelineState.hpp>

namespace Ilum
{
class RHIDevice;
class RenderGraph;
class RGPass;
class ImGuiContext;
class Renderer;

class RGResource
{
  public:
	RGResource(ResourceType type = ResourceType::None) :
	    m_type(type)
	{
	}

	virtual ~RGResource() = default;

	inline ResourceType GetType() const
	{
		return m_type;
	}

  private:
	ResourceType m_type;
};

class RGTexture : public RGResource
{
  public:
	RGTexture(RHIDevice *device, const TextureDesc &desc) :
	    RGResource(ResourceType::Texture)
	{
		m_handle = std::make_unique<Texture>(device, desc);
	}

	virtual ~RGTexture() = default;

	Texture *GetHandle() const
	{
		return m_handle.get();
	}

  private:
	std::unique_ptr<Texture> m_handle;
};

class RGBuffer : public RGResource
{
  public:
	RGBuffer(RHIDevice *device, const BufferDesc &desc) :
	    RGResource(ResourceType::Buffer)
	{
		m_handle = std::make_unique<Buffer>(device, desc);
	}

	virtual ~RGBuffer() = default;

	Buffer *GetHandle() const
	{
		return m_handle.get();
	}

  private:
	std::unique_ptr<Buffer> m_handle;
};

class RGNode
{
	friend class RGBuilder;

  public:
	RGNode(RenderGraph &graph, RGPass &pass, RGResource *resource);
	~RGNode() = default;

	RGResource *GetResource();

	const TextureState &GetCurrentState() const;
	const TextureState &GetLastState() const;

  private:
	RenderGraph &m_graph;
	RGPass      &m_pass;
	RGResource  *p_resource;
	TextureState m_current_state;
	TextureState m_last_state;
};

class RGResources
{
  public:
	RGResources(RenderGraph &graph, RGPass &pass);
	~RGResources() = default;

	Texture *GetTexture(const RGHandle &handle) const;
	Buffer  *GetBuffer(const RGHandle &handle) const;

  private:
	RenderGraph &m_graph;
	RGPass      &m_pass;
};

class RGPass
{
	friend class RGBuilder;
	friend class RenderPass;

  public:
	RGPass(RHIDevice *device, const std::string &name);
	~RGPass();

	void Execute(CommandBuffer &cmd_buffer, const RGResources &resources, Renderer &renderer);

	void OnImGui(ImGuiContext &context, const RGResources &resources);

	const std::string &GetName() const;

  private:
	RHIDevice *p_device = nullptr;

	std::string m_name;

	bool m_begin = false;

	PipelineState m_pso;

	std::function<void(CommandBuffer &, PipelineState &, const RGResources &, Renderer& renderer)> m_execute_callback;

	std::function<void(CommandBuffer &)> m_barrier_callback;
	std::function<void(CommandBuffer &)> m_barrier_initialize;

	std::function<void(ImGuiContext &, const RGResources &)> m_imgui_callback;
};

class RenderGraph
{
	friend class RGBuilder;
	friend class RGResources;

  public:
	RenderGraph(RHIDevice *device, Renderer &renderer);
	~RenderGraph();

	void Execute();

	void OnImGui(ImGuiContext &context);

	Texture *GetPresent() const;

  private:
	RHIDevice *p_device = nullptr;

	Renderer &m_renderer;

	std::vector<RGPass>                         m_passes;
	std::map<RGHandle, std::unique_ptr<RGNode>> m_nodes;
	std::vector<std::unique_ptr<RGResource>>    m_resources;
};
}        // namespace Ilum