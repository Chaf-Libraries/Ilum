#pragma once

#include "Resource.hpp"

namespace Ilum
{
class RHITexture;
class RHIContext;

template <>
class EXPORT_API Resource<ResourceType::Texture> final : public IResource
{
  public:
	Resource(RHIContext *rhi_context, std::vector<uint8_t> &&data, const TextureDesc& desc);

	virtual ~Resource() override;

	RHITexture *GetTexture() const;

  private:
	struct Impl;
	Impl *m_impl = nullptr;
};
}        // namespace Ilum