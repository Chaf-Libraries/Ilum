#pragma once

#include <RHI/RHIBuffer.hpp>
#include <RHI/RHIContext.hpp>
#include <RHI/RHITexture.hpp>

#include <rttr/registration>

namespace Ilum
{
STRUCT(RGHandle, Enable)
{
	size_t handle;

	RGHandle();

	RGHandle(size_t handle);

	~RGHandle() = default;

	bool IsValid();

	bool operator<(const RGHandle &rhs) const;

	bool operator==(const RGHandle &rhs) const;

	size_t GetHandle() const;
};

STRUCT(RenderResourceDesc, Enable)
{
	ENUM(Type, Enable){
	    Buffer,
	    Texture};

	ENUM(Attribute, Enable){
	    Read,
	    Write};

	Type type;

	Attribute attribute;

	RHIResourceState state;

	RGHandle handle;
};

ENUM(BindPoint, Enable){
    None,
    Rasterization,
    Compute,
    RayTracing,
    CUDA};

STRUCT(RenderPassDesc, Enable)
{
	std::string name;

	rttr::variant config;

	std::map<std::string, RenderResourceDesc> resources;

	RGHandle prev_pass;

	BindPoint bind_point;

	RenderPassDesc &Write(const std::string &name, RenderResourceDesc::Type type, RHIResourceState state)
	{
		resources.emplace(name, RenderResourceDesc{type, RenderResourceDesc::Attribute::Write, state});
		return *this;
	}

	RenderPassDesc &Read(const std::string &name, RenderResourceDesc::Type type, RHIResourceState state)
	{
		resources.emplace(name, RenderResourceDesc{type, RenderResourceDesc::Attribute::Read, state});
		return *this;
	}

	RenderPassDesc &SetName(const std::string &name)
	{
		this->name = name;
		return *this;
	}

	template <typename T>
	RenderPassDesc &SetConfig(T var)
	{
		config = var;
		return *this;
	}

	RenderPassDesc &SetBindPoint(BindPoint bind_point)
	{
		this->bind_point = bind_point;
		return *this;
	}
};

STRUCT(RenderGraphDesc, Enable)
{
	std::map<RGHandle, RenderPassDesc> passes;

	std::map<RGHandle, TextureDesc> textures;

	std::map<RGHandle, BufferDesc> buffers;
};

class RenderGraph
{
	friend class RenderGraphBuilder;

  public:
	using RenderTask  = std::function<void(RenderGraph &, RHICommand *, rttr::variant &)>;
	using BarrierTask = std::function<void(RenderGraph &, RHICommand *)>;

	struct RenderPassInfo
	{
		std::string name;

		BindPoint bind_point;

		rttr::variant config;

		RenderTask  execute;
		BarrierTask barrier;

		std::unique_ptr<RHIProfiler> profiler = nullptr;
	};

  public:
	RenderGraph(RHIContext *rhi_context);

	~RenderGraph();

	RHITexture *GetTexture(RGHandle handle);

	RHIBuffer *GetBuffer(RGHandle handle);

	RHITexture *GetCUDATexture(RGHandle handle);

	void Execute();

	const std::vector<RenderPassInfo> &GetRenderPasses() const;

  private:
	struct TextureCreateInfo
	{
		TextureDesc desc;
		RGHandle    handle;
	};

	struct BufferCreateInfo
	{
		BufferDesc desc;
		RGHandle   handle;
	};

	RenderGraph &AddPass(
	    const std::string            &name,
	    BindPoint                     bind_point,
	    const rttr::variant          &config,
	    RenderTask                  &&execute,
	    BarrierTask                 &&barrier);

	RenderGraph &AddInitializeBarrier(BarrierTask &&barrier);

	// Without memory alias
	RenderGraph &RegisterTexture(const TextureCreateInfo &create_infos);

	// With memory alias
	RenderGraph &RegisterTexture(const std::vector<TextureCreateInfo> &create_info);

	RenderGraph &RegisterBuffer(const BufferCreateInfo &create_info);

	RHISemaphore *MapToCUDASemaphore(RHISemaphore *semaphore);

  private:
	RHIContext *p_rhi_context = nullptr;

	BarrierTask m_initialize_barrier;

	std::vector<RenderPassInfo> m_render_passes;

	std::vector<std::unique_ptr<RHITexture>> m_textures;
	std::map<RGHandle, RHITexture *>         m_texture_lookup;
	std::map<RGHandle, RHITexture *>         m_cuda_textures;

	std::vector<std::unique_ptr<RHIBuffer>> m_buffers;
	std::map<RGHandle, RHIBuffer *>         m_buffer_lookup;

	std::map<RHISemaphore*, std::unique_ptr<RHISemaphore>> m_cuda_semaphore_map;

	bool m_init = false;
};
}        // namespace Ilum
