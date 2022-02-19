#include "ImageLoader.hpp"

//#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <Graphics/Device/Device.hpp>

#include <Graphics/Resource/Buffer.hpp>
#include "Graphics/Command/CommandBuffer.hpp"
#include "Graphics/GraphicsContext.hpp"
#include <Graphics/Resource/Image.hpp>



#include <Core/JobSystem/JobSystem.hpp>

#include <Graphics/Vulkan.hpp>
#include <Graphics/RenderContext.hpp>

namespace Ilum
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

Bitmap ImageLoader::loadImage(const std::string &filepath)
{
	std::filesystem::path filename{filepath};

	auto extension = filename.extension().string();

	if (extension == ".png" || extension == ".jpg" || extension == ".hdr" || extension == ".jpeg")
	{
		return STBLoader(filepath);
	}

	ASSERT(false && "Unsupport image type!");

	return STBLoader(filepath);
}

inline std::vector<uint8_t> extractCubmap(Bitmap &bitmap, size_t width, size_t height, size_t channel, size_t slice_y, size_t slice_x, VkFormat format)
{
	std::vector<uint8_t> result(bitmap.data.size() / 6);

	for (size_t i = 0; i < height; i++)
	{
		size_t y             = i + slice_y * height;
		size_t x             = slice_x * width;
		size_t bytes_per_row = width * channel;

		std::memcpy(result.data() + i * bytes_per_row, bitmap.data.data() + (y * bitmap.width + x) * channel, bytes_per_row);
	}

	return result;
}

Cubemap ImageLoader::loadCubemap(const std::string &filepath)
{
	auto bitmap = loadImage(filepath);

	Cubemap cubemap;

	cubemap.format = bitmap.format;
	cubemap.width  = bitmap.width / 4;
	cubemap.height = bitmap.height / 3;
	assert(cubemap.width == cubemap.height);

	//    0
	// 1 2 3 4
	//    5
	cubemap.data[0] = extractCubmap(bitmap, cubemap.width, cubemap.height, 4, 0, 1, VK_FORMAT_R8G8B8A8_UNORM);
	cubemap.data[1] = extractCubmap(bitmap, cubemap.width, cubemap.height, 4, 1, 0, VK_FORMAT_R8G8B8A8_UNORM);
	cubemap.data[2] = extractCubmap(bitmap, cubemap.width, cubemap.height, 4, 1, 1, VK_FORMAT_R8G8B8A8_UNORM);
	cubemap.data[3] = extractCubmap(bitmap, cubemap.width, cubemap.height, 4, 1, 2, VK_FORMAT_R8G8B8A8_UNORM);
	cubemap.data[4] = extractCubmap(bitmap, cubemap.width, cubemap.height, 4, 1, 3, VK_FORMAT_R8G8B8A8_UNORM);
	cubemap.data[5] = extractCubmap(bitmap, cubemap.width, cubemap.height, 4, 2, 1, VK_FORMAT_R8G8B8A8_UNORM);

	return cubemap;
}

Cubemap ImageLoader::loadCubemap(const std::array<std::string, 6> &filepaths)
{
	std::array<Bitmap, 6> bitmaps;

	for (uint32_t i = 0; i < 6; i++)
	{
		bitmaps[i] = loadImage(filepaths[i]);
	}

	Cubemap cubemap;

	cubemap.format = bitmaps[0].format;
	cubemap.width  = bitmaps[0].width;
	cubemap.height = bitmaps[0].height;

	for (uint32_t i = 0; i < 6; i++)
	{
		assert(cubemap.format == bitmaps[i].format);
		assert(cubemap.width == bitmaps[i].width);
		assert(cubemap.height == bitmaps[i].height);
	}

	//    0
	// 1 2 3 4
	//    5
	for (uint32_t i = 0; i < 6; i++)
	{
		cubemap.data[i] = std::move(bitmaps[i].data);
	}

	return cubemap;
}

void ImageLoader::loadImage(Graphics::Image &image, const Bitmap &bitmap, bool mipmaps)
{
	image = Graphics::Image(Graphics::RenderContext::GetDevice(), bitmap.width, bitmap.height, bitmap.format, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY, true);

	Graphics::Buffer staging_buffer(Graphics::RenderContext::GetDevice(), bitmap.data.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

	uint32_t offset = 0;
	uint8_t *data   = staging_buffer.Map();
	std::memcpy(data, bitmap.data.data(), bitmap.data.size());
	staging_buffer.Unmap();

	auto &            command_buffer = Graphics::RenderContext::CreateCommandBuffer();
	//CommandBuffer command_buffer;
	Graphics::VKDebugger::SetName(Graphics::RenderContext::GetDevice(), command_buffer, "transfer image data");
	command_buffer.Begin();
	command_buffer.CopyBufferToImage(Graphics::BufferInfo{staging_buffer, offset}, Graphics::ImageInfo{image});

	offset += static_cast<uint32_t>(bitmap.data.size());

	if (mipmaps)
	{
		if (bitmap.mip_levels.empty())
		{
			command_buffer.GenerateMipmap(image, VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_FILTER_LINEAR);
		}
		else
		{
			uint32_t mip_level = 1;
			for (const auto &mip_data : bitmap.mip_levels)
			{
				uint8_t *data = staging_buffer.Map();
				std::memcpy(data + offset, mip_data.data(), mip_data.size());
				staging_buffer.Unmap();
				command_buffer.CopyBufferToImage(Graphics::BufferInfo{staging_buffer, offset}, Graphics::ImageInfo{image, VK_IMAGE_USAGE_TRANSFER_DST_BIT, mip_level, 0});

				offset += static_cast<uint32_t>(mip_data.size());

				mip_level++;
			}
		}
	}

	command_buffer.TransferLayout(image, VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_USAGE_SAMPLED_BIT);
	command_buffer.End();
	command_buffer.SubmitIdle();

	Graphics::RenderContext::ResetCommandPool();
}

void ImageLoader::loadCubemap(Graphics::Image &image, const Cubemap &cubemap)
{
}

void ImageLoader::loadImageFromFile(Graphics::Image &image, const std::string &filepath, bool mipmaps)
{
	loadImage(image, loadImage(filepath), mipmaps);
}

void ImageLoader::loadCubemapFromFile(Graphics::Image &image, const std::string &filepath)
{
	loadCubemap(image, loadCubemap(filepath));
}

}        // namespace Ilum