#include "STBImporter.hpp"

#include <Core/Path.hpp>
#include <RHI/RHIContext.hpp>

#include <stb_image.h>

namespace Ilum
{
TextureImportInfo STBImporter::ImportImpl(const std::string &filename)
{
	TextureImportInfo info = {};

	info.desc.name    = Path::GetInstance().GetFileName(filename, false);
	info.desc.width   = 1;
	info.desc.height  = 1;
	info.desc.depth   = 1;
	info.desc.mips    = 1;
	info.desc.layers  = 1;
	info.desc.samples = 1;

	int32_t width = 0, height = 0, channel = 0;

	const int32_t req_channel = 4;

	void  *raw_data = nullptr;
	size_t size     = 0;

	if (stbi_is_hdr(filename.c_str()))
	{
		raw_data         = stbi_loadf(filename.c_str(), &width, &height, &channel, req_channel);
		size             = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(req_channel) * sizeof(float);
		info.desc.format = RHIFormat::R32G32B32A32_FLOAT;
	}
	else if (stbi_is_16_bit(filename.c_str()))
	{
		raw_data         = stbi_load_16(filename.c_str(), &width, &height, &channel, req_channel);
		size             = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(req_channel) * sizeof(uint16_t);
		info.desc.format = RHIFormat::R16G16B16A16_FLOAT;
	}
	else
	{
		raw_data         = stbi_load(filename.c_str(), &width, &height, &channel, req_channel);
		size             = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(req_channel) * sizeof(uint8_t);
		info.desc.format = RHIFormat::R8G8B8A8_UNORM;
	}

	info.desc.width  = static_cast<uint32_t>(width);
	info.desc.height = static_cast<uint32_t>(height);
	info.desc.mips   = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height))) + 1);
	info.desc.usage  = RHITextureUsage::ShaderResource | RHITextureUsage::Transfer;

	info.data.resize(size);
	std::memcpy(info.data.data(), raw_data, size);

	stbi_image_free(raw_data);

	return info;
}
}        // namespace Ilum