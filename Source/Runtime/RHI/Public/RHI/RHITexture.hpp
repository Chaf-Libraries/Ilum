#pragma once

#include "RHIDefinitions.hpp"

#include <Core/Hash.hpp>

#include <cstdint>
#include <memory>

namespace Ilum
{
class RHIDevice;

STRUCT(TextureDesc, Enable)
{
	std::string name;

	uint32_t        width;
	uint32_t        height;
	uint32_t        depth;
	uint32_t        mips;
	uint32_t        layers;
	uint32_t        samples;
	RHIFormat       format;
	RHITextureUsage usage;
	bool            external;
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
	static std::unique_ptr<RHITexture> Create2D(RHIDevice *device, uint32_t width, uint32_t height, RHIFormat format, RHITextureUsage usage, bool mipmap, uint32_t samples = 1, bool external = false);
	static std::unique_ptr<RHITexture> Create3D(RHIDevice *device, uint32_t width, uint32_t height, uint32_t depth, RHIFormat format, RHITextureUsage usage, bool external = false);
	static std::unique_ptr<RHITexture> CreateCube(RHIDevice *device, uint32_t width, uint32_t height, RHIFormat format, RHITextureUsage usage, bool mipmap, bool external = false);
	static std::unique_ptr<RHITexture> Create2DArray(RHIDevice *device, uint32_t width, uint32_t height, uint32_t layers, RHIFormat format, RHITextureUsage usage, bool mipmap, uint32_t samples = 1, bool external = false);

  protected:
	RHIBackend  m_backend;
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
