#pragma once

#include "RHIDefinitions.hpp"
#include "RHITexture.hpp"

#include <array>
#include <optional>
#include <vector>

namespace Ilum
{
struct ColorAttachment
{
	RHILoadAction  load  = RHILoadAction::Clear;
	RHIStoreAction store = RHIStoreAction::Store;

	std::array<float, 4> clear_value = {0.f};
};

struct DepthStencilAttachment
{
	RHILoadAction  depth_load    = RHILoadAction::Clear;
	RHIStoreAction depth_store   = RHIStoreAction::Store;
	RHILoadAction  stencil_load  = RHILoadAction::DontCare;
	RHIStoreAction stencil_store = RHIStoreAction::DontCare;

	float    clear_depth   = 0.f;
	uint32_t clear_stencil = 0;
};

class RHIRenderTarget
{
  public:
	RHIRenderTarget(RHIDevice *device);

	virtual ~RHIRenderTarget() = default;

	static std::unique_ptr<RHIRenderTarget> Create(RHIDevice *device);

	virtual RHIRenderTarget &Set(uint32_t slot, RHITexture *texture, RHITextureDimension dimension, const ColorAttachment &attachment) = 0;
	virtual RHIRenderTarget &Set(uint32_t slot, RHITexture *texture, const TextureRange &range, const ColorAttachment &attachment)     = 0;
	virtual RHIRenderTarget &Set(RHITexture *texture, RHITextureDimension dimension, const DepthStencilAttachment &attachment)         = 0;
	virtual RHIRenderTarget &Set(RHITexture *texture, const TextureRange &range, const DepthStencilAttachment &attachment)             = 0;

	uint32_t GetWidth() const;

	uint32_t GetHeight() const;

	uint32_t GetLayers() const;

	virtual RHIRenderTarget &Clear() = 0;

  protected:
	RHIDevice *p_device = nullptr;

	uint32_t m_width  = 0;
	uint32_t m_height = 0;
	uint32_t m_layers = 0;
};
}        // namespace Ilum