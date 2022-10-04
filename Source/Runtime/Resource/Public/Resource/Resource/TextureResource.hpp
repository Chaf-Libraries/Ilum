#pragma once

#include "Resource.hpp"

#include <RHI/RHITexture.hpp>

namespace Ilum
{
template <>
class TResource<ResourceType::Texture> : public Resource
{
  public:
	explicit TResource(size_t uuid);

	explicit TResource(size_t uuid, const std::string &meta, RHIContext *rhi_context);

	virtual ~TResource() override = default;

	virtual void Load(RHIContext *rhi_context) override;

	virtual void Import(RHIContext *rhi_context, const std::string &path) override;

	RHITexture *GetTexture() const;

  private:
	std::unique_ptr<RHITexture> m_texture   = nullptr;
};
}        // namespace Ilum