#pragma once

#include <RHI/RHIBuffer.hpp>
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
	RGHandle(size_t handle);

	~RGHandle() = default;

	size_t operator()();

	bool operator<(const RGHandle &rhs) const;

	size_t GetHandle() const;

  private:
	size_t m_handle;
};

struct RenderPassDesc
{
	enum class ResourceType
	{
		Buffer,
		Texture
	};

	std::string   name;
	rttr::variant variant;

	std::map<std::string, std::pair<ResourceType, RGHandle>> reads;
	std::map<std::string, std::pair<ResourceType, RGHandle>> writes;
};

struct RenderGraphDesc
{
	std::map<RGHandle, RenderPassDesc>     passes;
	std::map<RGHandle, TextureDesc>        textures;
	std::map<RGHandle, BufferDesc>         buffers;
	std::vector<std::pair<size_t, size_t>> edges;

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
}        // namespace Ilum
