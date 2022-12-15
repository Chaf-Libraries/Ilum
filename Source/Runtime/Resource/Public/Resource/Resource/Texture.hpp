#pragma once

#include "../Resource.hpp"

#include <RHI/RHIContext.hpp>

namespace Ilum
{
template <>
class EXPORT_API Resource<ResourceType::Texture2D> final : public IResource
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