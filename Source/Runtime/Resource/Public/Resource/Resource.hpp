#pragma once

#include <Core/Macro.hpp>

namespace Ilum
{
ENUM(ResourceType, Enable){
    None,
    Model,
    Texture,
    Scene,
    RenderGraph};

class RHIContext;

template <ResourceType _Ty>
class TResource;

class Resource
{
  public:
	explicit Resource(size_t uuid);

	explicit Resource(size_t uuid, const std::string& meta, RHIContext *rhi_context);

	virtual ~Resource() = default;

	template <ResourceType _Ty>
	TResource<_Ty> *CastTo()
	{
		return static_cast<TResource<_Ty> *>(this);
	}

	size_t GetUUID() const;

	const std::string &GetMeta() const;

	bool IsValid() const;

	virtual void Load(RHIContext *rhi_context) = 0;

	virtual void Import(RHIContext *rhi_context, const std::string &path) = 0;

  protected:
	std::string m_meta;
	size_t      m_uuid;
	bool        m_valid = false;
};

template <ResourceType _Ty>
class TResource : public Resource
{
};
}        // namespace Ilum

#include "Resource/ModelResource.hpp"
#include "Resource/RenderGraphResource.hpp"
#include "Resource/SceneResource.hpp"
#include "Resource/TextureResource.hpp"