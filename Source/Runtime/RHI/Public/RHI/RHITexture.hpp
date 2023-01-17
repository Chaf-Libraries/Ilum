#pragma once

#include "Fwd.hpp"

namespace Ilum
{
STRUCT(TextureDesc, Enable)
{
	std::string name;

	META(Min(1))
	uint32_t width = 1;

	META(Min(1))
	uint32_t height = 1;

	META(Min(1))
	uint32_t depth = 1;

	META(Min(1))
	uint32_t mips = 1;

	META(Min(1))
	uint32_t layers = 1;

	META(Min(1))
	uint32_t samples = 1;

	RHIFormat       format = RHIFormat::Undefined;
	RHITextureUsage usage  = RHITextureUsage::Undefined;

	template<typename Archive>
	void serialize(Archive & archive)
	{
		archive(name, width, height, depth, mips, layers, samples, format, usage);
	}
};

struct TextureRange
{
	RHITextureDimension dimension   = RHITextureDimension::Texture2D;
	uint32_t            base_mip    = 0;
	uint32_t            mip_count   = 1;
	uint32_t            base_layer  = 0;
	uint32_t            layer_count = 1;

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

	const std::string GetBackend() const;

	virtual std::unique_ptr<RHITexture> Alias(const TextureDesc &desc);

	virtual size_t GetMemorySize() const = 0;

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
	RHITexture *texture;

	RHIResourceState src;
	RHIResourceState dst;

	TextureRange range;

	RHIQueueFamily src_family = RHIQueueFamily::Graphics;
	RHIQueueFamily dst_family = RHIQueueFamily::Graphics;
};
}        // namespace Ilum
