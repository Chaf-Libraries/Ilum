#pragma once

#include <RHI/RHIBuffer.hpp>
#include <RHI/RHIContext.hpp>
#include <RHI/RHITexture.hpp>

#include <rttr/registration>

namespace Ilum
{
struct RenderPassName
{
	const char           *name = nullptr;
	const RenderPassName *next = nullptr;

	RenderPassName(const char *name, const RenderPassName *&pass) :
	    name(name)
	{
		next = pass;
		pass = this;
	}
};

inline const static RenderPassName *RenderPassNameList = nullptr;

class RGHandle
{
  public:
	RGHandle() = default;

	RGHandle(size_t handle);

	~RGHandle() = default;

	size_t operator()();

	bool operator<(const RGHandle &rhs) const;

	bool operator==(const RGHandle &rhs) const;

	size_t GetHandle() const;

  private:
	size_t m_handle = ~0U;
};

struct RenderPassDesc
{
	struct ResourceInfo
	{
		enum class Type
		{
			Buffer,
			Texture
		} type;
		RHIResourceState state;
		RGHandle         handle;
	};

	std::string   name;
	rttr::variant variant;

	std::map<std::string, ResourceInfo> writes;
	std::map<std::string, ResourceInfo> reads;

	RenderPassDesc &Write(const std::string &name, ResourceInfo::Type type, RHIResourceState state, size_t &handle)
	{
		writes.emplace(name, ResourceInfo{type, state, handle++});
		return *this;
	}

	RenderPassDesc &Read(const std::string &name, ResourceInfo::Type type, RHIResourceState state, size_t &handle)
	{
		reads.emplace(name, ResourceInfo{type, state, handle++});
		return *this;
	}
};

struct RenderGraphDesc
{
	std::map<RGHandle, RenderPassDesc>  passes;
	std::map<RGHandle, TextureDesc>     textures;
	std::map<RGHandle, BufferDesc>      buffers;
	std::set<std::pair<size_t, size_t>> edges;

	static std::pair<RGHandle, RGHandle> DecodeEdge(size_t from, size_t to)
	{
		return std::make_pair(RGHandle(from / 2), RGHandle(to / 2));
	}
};

#define RENDER_PASS_REGISTERATION(Type)                                       \
	namespace RenderPass::Registeration::_##Type                              \
	{                                                                         \
		RTTR_REGISTRATION                                                     \
		{                                                                     \
			rttr::registration::method(#Type##"_Desc", &Type## ::CreateDesc); \
			rttr::registration::method(#Type##"_Creation", &Type## ::Create); \
		}                                                                     \
	}                                                                         \
	static RenderPassName RenderPass_##Type##_Name(#Type, RenderPassNameList);

#define RENDER_PASS_NAME_REGISTERATION(Type) \
	static RenderPassName RenderPass_##Type##_Name(#Type, RenderPassNameList);

#define RENDER_PASS_CONFIG_REGIST_BEGIN(Type)          \
	namespace RenderPassConfig::Registeration::_##Type \
	{                                                  \
		using Config = Type::Config;                   \
		RTTR_REGISTRATION                              \
		{                                              \
			rttr::registration::class_<Type::Config>(#Type "::Config").constructor<>()(rttr::policy::ctor::as_object)

#define RENDER_PASS_CONFIG_REGIST(Member) \
	.property(#Member, &Config::Member)

#define RENDER_PASS_CONFIG_REGIST_END() \
	;                                   \
	}                                   \
	}

class RenderGraph
{
	friend class RenderGraphBuilder;

  public:
	  using RenderTask =std::function<void(RenderGraph &, RHICommand *)>;

  public:
	RenderGraph(RHIContext *rhi_context);

	~RenderGraph();

	RHITexture *GetTexture(RGHandle handle);

	RHIBuffer *GetBuffer(RGHandle handle);

	void Execute();

  private:
	struct TextureCreateInfo
	{
		TextureDesc           desc;
		std::vector<RGHandle> handles;
	};

	struct BufferCreateInfo
	{
		BufferDesc            desc;
		std::vector<RGHandle> handles;
	};

	RenderGraph &AddPass(const std::string &name, RenderTask &&execute, RenderTask &&barrier);

	RenderGraph &AddInitializeBarrier(RenderTask &&barrier);

	// Without memory alias
	RenderGraph &RegisterTexture(const TextureCreateInfo &create_infos);

	// With memory alias
	RenderGraph &RegisterTexture(const std::vector<TextureCreateInfo> &create_info);

	RenderGraph &RegisterBuffer(const BufferCreateInfo &create_info);

  private:
	RHIContext *p_rhi_context = nullptr;

	struct RenderPassInfo
	{
		std::string name;

		RenderTask execute;
		RenderTask barrier;
	};

	RenderTask m_initialize_barrier;

	std::vector<RenderPassInfo> m_render_passes;

	std::vector<std::unique_ptr<RHITexture>> m_textures;
	std::map<RGHandle, RHITexture *>         m_texture_lookup;

	std::vector<std::unique_ptr<RHIBuffer>> m_buffers;
	std::map<RGHandle, RHIBuffer *>         m_buffer_lookup;

	bool m_init = false;
};
}        // namespace Ilum
