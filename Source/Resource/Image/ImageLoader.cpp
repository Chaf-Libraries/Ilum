#include "ImageLoader.hpp"

#include <Graphics/Resource/Buffer.hpp>
#include <Graphics/Command/CommandBuffer.hpp>

#include <Core/Logger/Logger.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <filesystem>

namespace Ilum::Resource
{
inline Bitmap STBLoad(const std::string &filepath)
{
	int       width = 0, height = 0, channel = 0;
	const int req_channel = 4;

	uint8_t *data = stbi_load(filepath.c_str(), &width, &height, &channel, req_channel);

	std::vector<uint8_t> bitmap_data(static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(req_channel));
	std::memcpy(bitmap_data.data(), data, bitmap_data.size() * sizeof(uint8_t));
	stbi_image_free(data);

	return Bitmap{
	    std::move(bitmap_data),
	    VK_FORMAT_R8G8B8A8_UNORM,
	    static_cast<uint32_t>(width),
	    static_cast<uint32_t>(height)};
}

inline Bitmap STBLoad16Bit(const std::string &filepath)
{
	int       width = 0, height = 0, channel = 0;
	const int req_channel = 4;

	uint16_t *data = stbi_load_16(filepath.c_str(), &width, &height, &channel, req_channel);

	std::vector<uint8_t> bitmap_data(static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(req_channel) * 2);
	std::memcpy(bitmap_data.data(), data, bitmap_data.size());
	stbi_image_free(data);

	return Bitmap{
	    std::move(bitmap_data),
	    VK_FORMAT_R16G16B16A16_SFLOAT,
	    static_cast<uint32_t>(width),
	    static_cast<uint32_t>(height)};
}

inline Bitmap STBLoadHDR(const std::string &filepath)
{
	int       width = 0, height = 0, channel = 0;
	const int req_channel = 4;

	float *data = stbi_loadf(filepath.c_str(), &width, &height, &channel, req_channel);

	std::vector<uint8_t> bitmap_data(static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(req_channel) * 4);
	std::memcpy(bitmap_data.data(), data, bitmap_data.size());
	stbi_image_free(data);

	return Bitmap{
	    std::move(bitmap_data),
	    VK_FORMAT_R32G32B32A32_SFLOAT,
	    static_cast<uint32_t>(width),
	    static_cast<uint32_t>(height)};
}

inline Bitmap STBLoader(const std::string &filepath)
{
	if (stbi_is_hdr(filepath.c_str()))
	{
		return STBLoadHDR(filepath);
	}
	else if (stbi_is_16_bit(filepath.c_str()))
	{
		return STBLoad16Bit(filepath);
	}
	return STBLoad(filepath);
}

Bitmap ImageLoader::LoadTexture2D(const std::string &filepath)
{
	std::filesystem::path filename{filepath};

	auto extension = filename.extension().string();

	LOG_INFO("Load Image: {}", filepath);

	if (extension == ".png" || extension == ".jpg" || extension == ".hdr" || extension == ".jpeg")
	{
		return STBLoader(filepath);
	}

	ASSERT(false && "Unsupport image type!");

	return STBLoader(filepath);
}

std::unique_ptr<Graphics::Image> ImageLoader::LoadTexture2D(const Graphics::Device &device, Graphics::CommandBuffer &cmd_buffer, const Bitmap &bitmap, bool mipmaps)
{
	std::unique_ptr<Graphics::Image> image = std::make_unique<Graphics::Image>(device, bitmap.width, bitmap.height, bitmap.format, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY, true);

	Graphics::Buffer staging_buffer(device, bitmap.data.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

	uint32_t offset = 0;
	uint8_t *data   = staging_buffer.Map();
	std::memcpy(data, bitmap.data.data(), bitmap.data.size());
	staging_buffer.Unmap();

	cmd_buffer.Begin();
	cmd_buffer.CopyBufferToImage(Graphics::BufferInfo{staging_buffer, offset}, Graphics::ImageInfo{*image});

	offset += static_cast<uint32_t>(bitmap.data.size());

	if (mipmaps)
	{
		cmd_buffer.GenerateMipmap(*image, VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_FILTER_LINEAR);
	}

	cmd_buffer.TransferLayout(*image, VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_USAGE_SAMPLED_BIT);
	cmd_buffer.End();
	cmd_buffer.SubmitIdle();

	return image;
}

std::unique_ptr<Graphics::Image> ImageLoader::LoadTexture2DFromFile(const Graphics::Device &device, Graphics::CommandBuffer &cmd_buffer, const std::string &filepath, bool mipmaps)
{
	return LoadTexture2D(device, cmd_buffer, LoadTexture2D(filepath), mipmaps);
}
}        // namespace Ilum::Graphics