#pragma once

#include <RHIDefinitions.hpp>

#include <cstdint>
#include <memory>

namespace Ilum
{
class RHIDevice;

REFLECTION_STRUCT TextureDesc
{
	REFLECTION_PROPERTY(display = "fuck", min = "what")
	std::string name;

	REFLECTION_PROPERTY()
	uint32_t width;

	REFLECTION_PROPERTY()
	uint32_t height;

	REFLECTION_PROPERTY()
	uint32_t depth;

	REFLECTION_PROPERTY()
	uint32_t mips;

	REFLECTION_PROPERTY()
	uint32_t layers;

	REFLECTION_PROPERTY()
	uint32_t samples;

	REFLECTION_PROPERTY()
	RHIFormat format;

	REFLECTION_PROPERTY()
	RHITextureUsage usage;
};

REFLECTION_STRUCT TextureRange
{
	REFLECTION_PROPERTY()
	RHITextureDimension dimension;

	REFLECTION_PROPERTY()
	uint32_t base_mip;

	REFLECTION_PROPERTY()
	uint32_t mip_count;

	REFLECTION_PROPERTY()
	uint32_t base_layer;

	REFLECTION_PROPERTY()
	uint32_t layer_count;

	REFLECTION_METHOD()
	size_t Hash() const
	{
		size_t hash = 0;
		HashCombine(hash, dimension, base_mip, mip_count, base_layer, layer_count);
		return hash;
	}
};

class RHITexture
{
  public:
	RHITexture(RHIDevice *device, const TextureDesc &desc);

	virtual ~RHITexture() = default;

	const TextureDesc &GetDesc() const;

	virtual std::unique_ptr<RHITexture> Alias(const TextureDesc &desc);

	static std::unique_ptr<RHITexture> Create(RHIDevice *device, const TextureDesc &desc);
	static std::unique_ptr<RHITexture> Create2D(RHIDevice *device, uint32_t width, uint32_t height, RHIFormat format, RHITextureUsage usage, bool mipmap, uint32_t samples = 1);
	static std::unique_ptr<RHITexture> Create3D(RHIDevice *device, uint32_t width, uint32_t height, uint32_t depth, RHIFormat format, RHITextureUsage usage);
	static std::unique_ptr<RHITexture> CreateCube(RHIDevice *device, uint32_t width, uint32_t height, RHIFormat format, RHITextureUsage usage, bool mipmap);
	static std::unique_ptr<RHITexture> Create2DArray(RHIDevice *device, uint32_t width, uint32_t height, uint32_t layers, RHIFormat format, RHITextureUsage usage, bool mipmap, uint32_t samples = 1);

  protected:
	RHIDevice  *p_device = nullptr;
	TextureDesc m_desc;
};

struct TextureStateTransition
{
	RHITexture      *texture;
	RHIResourceState src;
	RHIResourceState dst;
	TextureRange     range;
};
}        // namespace Ilum