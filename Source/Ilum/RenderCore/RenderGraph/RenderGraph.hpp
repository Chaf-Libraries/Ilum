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

	std::string name;

	std::map<std::string, std::pair<ResourceType, RGHandle>> reads;
	std::map<std::string, std::pair<ResourceType, RGHandle>> writes;
};

struct RenderGraphDesc
{
	std::map<RGHandle, RenderPassDesc> passes;
	std::map<RGHandle, TextureDesc>    textures;
	std::map<RGHandle, BufferDesc>     buffers;
	std::map<RGHandle, RGHandle>       edges;
};

#define RENDER_PASS_DESC_REGISTERATION(Type)                              \
	RTTR_REGISTRATION                                                     \
	{                                                                     \
		rttr::registration::method(#Type##"_Desc", &Type## ::CreateDesc); \
	}                                                                     \
	static RenderPassName RenderPass_##Type##_Name(#Type, RenderPassNameList);

}        // namespace Ilum
