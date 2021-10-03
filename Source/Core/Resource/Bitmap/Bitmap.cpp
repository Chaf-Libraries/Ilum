#include "Bitmap.hpp"

#include "Core/Engine/File/FileSystem.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>

namespace Ilum
{
inline bool generate_mipmaps(std::vector<uint8_t> &data, std::vector<Bitmap::Mipmap> &mipmaps, uint32_t bytes_per_pixel)
{
	if (mipmaps.size() > 1)
	{
		// Mipmaps already generated
		return true;
	}

	uint32_t width  = mipmaps[0].extent.width;
	uint32_t height = mipmaps[0].extent.height;

	uint32_t next_width  = std::max<uint32_t>(1u, width / 2);
	uint32_t next_heigth = std::max<uint32_t>(1u, height / 2);
	size_t   next_size   = static_cast<size_t>(next_width) * static_cast<size_t>(next_heigth) * static_cast<size_t>(bytes_per_pixel);

	while (true)
	{
		size_t old_size = static_cast<uint32_t>(data.size());
		data.resize(old_size + next_size);

		auto &prev_mipmap = mipmaps.back();

		Bitmap::Mipmap next_mipmap = {};
		next_mipmap.level          = prev_mipmap.level + 1;
		next_mipmap.offset         = static_cast<uint32_t>(old_size);
		next_mipmap.extent         = {next_width, next_heigth, 1u};

		if (bytes_per_pixel == 4)
		{
			// R8G8B8A8_Unorm
			stbir_resize_uint8(data.data() + prev_mipmap.offset, prev_mipmap.extent.width, prev_mipmap.extent.height, 0,
			                   data.data() + next_mipmap.offset, next_mipmap.extent.width, next_mipmap.extent.height, 0, bytes_per_pixel);
		}
		else if (bytes_per_pixel == 16)
		{
			// R32G32B32A32_Float
			stbir_resize_float(reinterpret_cast<float *>(data.data() + prev_mipmap.offset), prev_mipmap.extent.width, prev_mipmap.extent.height, 0,
			                   reinterpret_cast<float *>(data.data() + next_mipmap.offset), next_mipmap.extent.width, next_mipmap.extent.height, 0, bytes_per_pixel / 4);
		}
		else
		{
			LOG_ERROR("Unsupport image type!");
			return false;
		}

		mipmaps.emplace_back(std::move(next_mipmap));

		next_width       = std::max<uint32_t>(1u, next_width / 2);
		next_heigth      = std::max<uint32_t>(1u, next_heigth / 2);
		size_t next_size = static_cast<size_t>(next_width) * static_cast<size_t>(next_heigth) * static_cast<size_t>(bytes_per_pixel);

		if (next_width == 1 && next_heigth == 1)
		{
			break;
		}
	}

	return true;
}

inline bool load_stb(const std::vector<uint8_t> &mem_data, std::vector<uint8_t> &data, uint32_t &width, uint32_t &height, uint32_t &bytes_per_pixel)
{
	int comp, req_comp = 4, w = 0, h = 0;

	auto data_buffer = reinterpret_cast<const stbi_uc *>(mem_data.data());
	auto data_size   = static_cast<int>(mem_data.size());

	bytes_per_pixel = stbi_is_hdr_from_memory(data_buffer, data_size) ? 16 : 4;

	if (bytes_per_pixel == 1)
	{
		auto stb_data = stbi_load_from_memory(data_buffer, data_size, &w, &h, &comp, req_comp);
		if (!stb_data)
		{
			return false;
		}
		data.resize(data.size() + w * h * req_comp);
		std::memcpy(data.data(), stb_data, w * h * req_comp);
		stbi_image_free(stb_data);
	}
	else
	{
		auto stb_data = stbi_loadf_from_memory(data_buffer, data_size, &w, &h, &comp, req_comp);
		if (!stb_data)
		{
			return false;
		}
		data.resize(data.size() + w * h * bytes_per_pixel);
		std::memcpy(data.data(), stb_data, w * h * bytes_per_pixel);
		stbi_image_free(stb_data);
	}

	width  = static_cast<uint32_t>(w);
	height = static_cast<uint32_t>(h);
	return true;
}

Bitmap::Bitmap(std::vector<uint8_t> &&data, std::vector<Mipmap> &&mipmaps) :
    m_data(std::move(data)), m_mipmaps(std::move(mipmaps))
{
}

Bitmap::Bitmap(const std::string &path) :
    m_path(path)
{
	load(path);
}

bool Bitmap::load(const std::string &path)
{
	auto extension = FileSystem::getFileExtension(path);

	std::vector<uint8_t> mem_data;
	FileSystem::read(path, mem_data, true);

	if (extension == ".jpg" || extension == ".png" || extension == ".jpeg" || extension == ".bmp" || extension == ".hdr")
	{
		return load_stb(mem_data, m_data, m_mipmaps[0].extent.width, m_mipmaps[0].extent.height, m_bytes_per_pixel) && generate_mipmaps(m_data, m_mipmaps, m_bytes_per_pixel);
	}

	return false;
}

bool Bitmap::write(const std::string &path)
{
	return false;
}

Bitmap::operator bool() const noexcept
{
	return !m_data.empty();
}

bool Bitmap::hasMipmaps() const
{
	return m_mipmaps.size() > 1;
}

const std::string &Bitmap::getPath() const
{
	return m_path;
}

const std::vector<uint8_t> &Bitmap::getData() const
{
	return m_data;
}

const std::vector<Bitmap::Mipmap> &Bitmap::getMipmaps() const
{
	return m_mipmaps;
}

void Bitmap::setData(std::vector<uint8_t> &&data, std::vector<Mipmap> &&mipmaps)
{
	m_data    = std::move(data);
	m_mipmaps = std::move(mipmaps);
}

const uint32_t Bitmap::getWidth() const
{
	return m_mipmaps.at(0).extent.width;
}

const uint32_t Bitmap::getHeight() const
{
	return m_mipmaps.at(0).extent.height;
}

const uint32_t Bitmap::getChannel() const
{
	return m_mipmaps.at(0).extent.depth;
}

const uint32_t Bitmap::getBytesPerPixel() const
{
	return m_bytes_per_pixel;
}

scope<Bitmap> Bitmap::create(const std::string &path)
{
	return createScope<Bitmap>(path);
}
}        // namespace Ilum