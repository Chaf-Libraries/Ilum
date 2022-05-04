#pragma once

#include <RHI/Buffer.hpp>
#include <RHI/Command.hpp>
#include <RHI/PipelineState.hpp>
#include <RHI/Texture.hpp>

namespace Ilum
{
class RGPass;
class RenderGraph;

enum class RGResourceType
{
	None,
	Texture,
	Buffer,
};

class RGResourceHandle
{
  public:
	RGResourceHandle();
	~RGResourceHandle() = default;

	operator uint32_t() const;

	void Invalidate();
	bool IsInvalid() const;

  private:
	inline static uint32_t INVALID_ID = ~0U;
	inline static uint32_t CURRENT_ID = 0U;
	uint32_t               m_index    = INVALID_ID;
};

class RGResource
{
  public:
	RGResource(const std::string &name, RGResourceType type) :
	    m_name(name), m_type(type)
	{}

	~RGResource() = default;

	RGResourceType GetType() const
	{
		return m_type;
	}

	virtual void *GetResource() = 0;

  private:
	const std::string m_name;
	RGResourceType    m_type = RGResourceType::None;
};

class RGTexture : public RGResource
{
  public:
	RGTexture(const std::string &name, const TextureDesc &desc) :
	    RGResource(name, RGResourceType::Texture), m_desc(desc)
	{}

	~RGTexture() = default;

	const TextureDesc &GetDesc() const
	{
		return m_desc;
	}

	virtual void *GetResource() override
	{
		return m_resource.get();
	}

	void CreateResource(std::unique_ptr<Texture> &&resource)
	{
		m_resource = std::move(resource);
	}

  private:
	TextureDesc              m_desc;
	std::unique_ptr<Texture> m_resource = nullptr;
};

struct RGNode
{
	explicit RGNode(RGResource *resource) :
	    p_resource(resource)
	{}

	RGResource *p_resource = nullptr;
};

class RGPassResources
{
  public:
	RGPassResources(RGPass &pass, RenderGraph &graph);
	~RGPassResources();

	RGPassResources(const RGPassResources &) = delete;
	RGPassResources &operator=(const RGPassResources &) = delete;
	RGPassResources(RGPassResources &&)                 = delete;
	RGPassResources &operator=(RGPassResources &&) = delete;

	Texture &GetTexture(RGResourceHandle handle) const;
	Buffer  &GetBuffer(RGResourceHandle handle) const;

  private:
	RenderGraph &m_graph;
	RGPass      &m_pass;
};

class RGPassBuilder
{
  public:
	RGPassBuilder(RGPass &pass, RenderGraph &graph);
	~RGPassBuilder() = default;

	void Bind(std::function<void(CommandBuffer &, const RGPassResources &)> &&callback);

	RGResourceHandle Write(RGResourceHandle &resource);
	RGResourceHandle CreateTexture(const std::string &name, const TextureDesc &desc);
	RGResourceHandle CreateBuffer(const std::string &name, const BufferDesc &desc);

	PipelineState &GetPipelineState();


  private:
	RGPass      &m_pass;
	RenderGraph &m_graph;
};

class RGPass
{
  public:
	friend class RenderGraph;
	friend class RGPassBuilder;

	RGPass(RenderGraph &graph, const std::string &name);

	void Execute(CommandBuffer &cmd_buffer, const RGPassResources &resource);

	void SetCallback(std::function<void(CommandBuffer &, const RGPassResources &)> &&callback);

	bool ReadFrom(RGResourceHandle handle) const;

	bool WriteTo(RGResourceHandle handle) const;

	const std::string &GetName() const;

  private:
	std::function<void(CommandBuffer &, const RGPassResources &)> m_callback;

	std::string m_name;

	RenderGraph &m_graph;

	struct
	{
		VkPipeline          pipeline        = VK_NULL_HANDLE;
		VkRenderPass        pass            = VK_NULL_HANDLE;
		VkPipelineLayout    pipeline_layout = VK_NULL_HANDLE;
		VkPipelineBindPoint bind_point      = VK_PIPELINE_BIND_POINT_MAX_ENUM;
		std::vector<VkDescriptorSet> descriptor_sets;
	} m_pass_info;

	std::vector<RGResourceHandle> m_reads;
	std::vector<RGResourceHandle> m_writes;
};

class RenderGraph
{
  public:
	RenderGraph(RHIDevice *device);
	~RenderGraph();

	RenderGraph(const RenderGraph &) = delete;
	RenderGraph &operator=(const RenderGraph &) = delete;
	RenderGraph(RenderGraph &&)                 = delete;
	RenderGraph &operator=(RenderGraph &&) = delete;

	RGPassBuilder AddPass(const std::string &name);

	RGResourceHandle CreateTexture(const std::string &name, const TextureDesc &desc);
	RGResourceHandle CreateBuffer(const std::string &name, const BufferDesc &desc);

	void OnImGui();

  private:
	RGResourceHandle CreateResourceNode(RGResource *resource);

  private:
	RHIDevice *p_device = nullptr;

	std::vector<std::unique_ptr<RGPass>>     m_passes;
	std::vector<std::unique_ptr<RGResource>> m_resources;
	std::vector<RGNode>                      m_nodes;

	std::vector<std::pair<RGResourceHandle, RGResourceHandle>> m_edges;

  private:
	static std::vector<std::string> s_avaliable_passes;
};

}        // namespace Ilum