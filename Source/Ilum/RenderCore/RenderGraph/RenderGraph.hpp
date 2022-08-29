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

REFLECTION_STRUCT RGHandle
{
	REFLECTION_PROPERTY()
	size_t handle;

	RGHandle();

	RGHandle(size_t handle);

	~RGHandle() = default;

	bool IsValid();

	bool operator<(const RGHandle &rhs) const;

	bool operator==(const RGHandle &rhs) const;

	size_t GetHandle() const;
};

REFLECTION_STRUCT RenderResourceDesc
{
	enum class Type
	{
		Buffer,
		Texture
	};

	enum class Attribute
	{
		Read,
		Write
	};

	REFLECTION_PROPERTY()
	Type type;

	REFLECTION_PROPERTY()
	Attribute attribute;

	REFLECTION_PROPERTY()
	RHIResourceState state;

	REFLECTION_PROPERTY()
	RGHandle handle;
};

REFLECTION_STRUCT RenderPassDesc
{
	REFLECTION_PROPERTY()
	std::string name;

	REFLECTION_PROPERTY()
	rttr::variant config;

	REFLECTION_PROPERTY()
	std::map<std::string, RenderResourceDesc> resources;

	REFLECTION_PROPERTY()
	RGHandle prev_pass;

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
};

REFLECTION_STRUCT RenderGraphDesc
{
	REFLECTION_PROPERTY()
	std::map<RGHandle, RenderPassDesc> passes;

	REFLECTION_PROPERTY()
	std::map<RGHandle, TextureDesc> textures;

	REFLECTION_PROPERTY()
	std::map<RGHandle, BufferDesc> buffers;
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

class RenderGraph
{
	friend class RenderGraphBuilder;

  public:
	using RenderTask = std::function<void(RenderGraph &, RHICommand *)>;

  public:
	RenderGraph(RHIContext *rhi_context);

	~RenderGraph();

	RHITexture *GetTexture(RGHandle handle);

	RHIBuffer *GetBuffer(RGHandle handle);

	void Execute();

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
