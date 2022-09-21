#pragma once

#include <RHIDefinitions.hpp>

#include <cstdint>
#include <memory>

namespace Ilum
{
class RHIDevice;

struct TextureDesc
{
	std::string         name;
	[[min(1)]] uint32_t width;
	[[min(1)]] uint32_t height;
	[[min(1)]] uint32_t depth;
	[[min(1)]] uint32_t mips;
	[[min(1)]] uint32_t layers;
	[[min(1)]] uint32_t samples;
	RHIFormat           format;
	[[reflection(false)]] RHITextureUsage usage;
	[[reflection(false)]] bool external;
};

struct TextureRange
{
	RHITextureDimension dimension;
	[[min(0)]] uint32_t base_mip;
	[[min(1)]] uint32_t mip_count;
	[[min(0)]] uint32_t base_layer;
	[[min(1)]] uint32_t layer_count;

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

	RHIBackend GetBackend() const;

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

struct [[reflection(false), serialization(false)]] TextureStateTransition
{
	RHITexture      *texture;
	RHIResourceState src;
	RHIResourceState dst;
	TextureRange     range;
};
}        // namespace Ilum