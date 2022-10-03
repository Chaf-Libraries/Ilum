#pragma once

#include "Resource.hpp"

#include <RHI/RHITexture.hpp>

namespace Ilum
{
template <>
class TResource<ResourceType::Texture2D> : public Resource
{
  public:
	TResource(size_t uuid, const std::string &meta) :
	    Resource(uuid, meta)
	{
	}

	virtual void Load() override
	{
		/*	SERIALIZE("Asset/Meta/" + std::to_string(uuid) + ".meta", ResourceType::Texture2D, uuid, meta_info, info);*/
		ResourceType type = ResourceType::None;
		TextureImportInfo info;
		DESERIALIZE("Asset/Meta/" + std::to_string(m_uuid) + ".meta", type, m_uuid, m_meta, info);
	}

	void SetTexture(std::unique_ptr<RHITexture> &&texture)
	{
		m_texture = std::move(texture);
	}

	void SetThumbnail(std::unique_ptr<RHITexture> &&thumbnail)
	{
		m_thumbnail = std::move(thumbnail);
	}

	RHITexture *GetTexture() const
	{
		return m_texture.get();
	}

	RHITexture *GetThumbnail() const
	{
		return m_thumbnail.get();
	}

  private:
	std::unique_ptr<RHITexture> m_texture   = nullptr;
	std::unique_ptr<RHITexture> m_thumbnail = nullptr;
};
}        // namespace Ilum