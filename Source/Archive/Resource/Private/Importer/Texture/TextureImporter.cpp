#pragma once

#include "TextureImporter.hpp"
#include "DDSImporter.hpp"
#include "STBImporter.hpp"

#include <Core/Path.hpp>

#include <stb_image.h>

namespace Ilum
{
TextureImportInfo TextureImporter::Import(const std::string &filename)
{
	std::string extension = Path::GetInstance().GetFileExtension(filename);

	if (extension == ".png" || extension == ".jpg" || extension == ".hdr" || extension == ".bmp")
	{
		return STBImporter::GetInstance().ImportImpl(filename);
	}
	else if (extension == ".dds")
	{
		return DDSImporter::GetInstance().ImportImpl(filename);
	}

	return {};
}

TextureImportInfo TextureImporter::ImportFromBuffer(const std::vector<uint8_t> &raw_data)
{
	TextureImportInfo info = {};

	// Load from buffer
	int32_t       width = 0, height = 0, channel = 0;
	const int32_t req_channel = 4;

	void  *data = nullptr;
	size_t size = 0;

	if (stbi_is_hdr_from_memory(static_cast<const stbi_uc *>(raw_data.data()), static_cast<uint32_t>(raw_data.size())))
	{
		data = stbi_loadf_from_memory(static_cast<const stbi_uc *>(raw_data.data()), static_cast<uint32_t>(raw_data.size()), &width, &height, &channel, req_channel);
		size = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(req_channel) * sizeof(float);

		info.desc.format = RHIFormat::R32G32B32A32_FLOAT;
	}
	else if (stbi_is_16_bit_from_memory(static_cast<const stbi_uc *>(raw_data.data()), static_cast<uint32_t>(raw_data.size())))
	{
		data = stbi_load_16_from_memory(static_cast<const stbi_uc *>(raw_data.data()), static_cast<uint32_t>(raw_data.size()), &width, &height, &channel, req_channel);
		size = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(req_channel) * sizeof(uint16_t);

		info.desc.format = RHIFormat::R16G16B16A16_FLOAT;
	}
	else
	{
		data = stbi_load_from_memory(static_cast<const stbi_uc *>(raw_data.data()), static_cast<uint32_t>(raw_data.size()), &width, &height, &channel, req_channel);
		size = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(req_channel) * sizeof(uint8_t);

		info.desc.format = RHIFormat::R8G8B8A8_UNORM;
	}

	info.desc.width  = static_cast<uint32_t>(width);
	info.desc.height = static_cast<uint32_t>(height);
	info.desc.mips   = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height))) + 1);
	info.desc.usage  = RHITextureUsage::ShaderResource | RHITextureUsage::Transfer;

	return info;
}
}        // namespace Ilum