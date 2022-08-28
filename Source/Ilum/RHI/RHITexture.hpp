#pragma once

#include <RHIDefinitions.hpp>

#include <cstdint>
#include <memory>

namespace Ilum
{
class RHIDevice;

struct TextureDesc
{
	std::string name;

	uint32_t width;
	uint32_t height;
	uint32_t depth;
	uint32_t mips;
	uint32_t layers;
	uint32_t samples;

	RHIFormat       format;
	RHITextureUsage usage;
};

REFLECTION_CLASS_BEGIN(TextureDesc)
REFLECTION_CLASS_PROPERTY(name)
REFLECTION_CLASS_PROPERTY(width)
REFLECTION_CLASS_PROPERTY(height)
REFLECTION_CLASS_PROPERTY(depth)
REFLECTION_CLASS_PROPERTY(mips)
REFLECTION_CLASS_PROPERTY(layers)
REFLECTION_CLASS_PROPERTY(samples)
REFLECTION_CLASS_PROPERTY(format)
REFLECTION_CLASS_PROPERTY(usage)
REFLECTION_CLASS_END()

struct TextureRange
{
	RHITextureDimension dimension;
	uint32_t            base_mip;
	uint32_t            mip_count;
	uint32_t            base_layer;
	uint32_t            layer_count;

	size_t Hash() const
	{
		size_t hash = 0;
		HashCombine(hash, dimension, base_mip, mip_count, base_layer, layer_count);
		return hash;
	}
};

REFLECTION_CLASS_BEGIN(TextureRange)
REFLECTION_CLASS_PROPERTY(dimension)
REFLECTION_CLASS_PROPERTY(base_mip)
REFLECTION_CLASS_PROPERTY(mip_count)
REFLECTION_CLASS_PROPERTY(base_layer)
REFLECTION_CLASS_PROPERTY(layer_count)
REFLECTION_CLASS_END()

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