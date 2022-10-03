#pragma once

#include <Core/Macro.hpp>

namespace Ilum
{
ENUM(ResourceType, Enable){
	None,
    Model,
    Texture2D,
    Scene,
    RenderGraph};

template <ResourceType _Ty>
class TResource;

class Resource
{
  public:
	explicit Resource(size_t uuid, const std::string &meta) :
	    m_uuid(uuid), m_meta(meta)
	{
	}

	~Resource() = default;

	template <ResourceType _Ty>
	TResource<_Ty> *CastTo()
	{
		return static_cast<TResource<_Ty> *>(this);
	}

	size_t GetUUID() const
	{
		return m_uuid;
	}

	bool IsValid() const
	{
		return m_valid;
	}

	virtual void Load()
	{
	}

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

#include "Resource/Model.inl"
#include "Resource/RenderGraph.inl"
#include "Resource/Scene.inl"
#include "Resource/Texture2D.inl"