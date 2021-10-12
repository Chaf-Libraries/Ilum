#include "Bitmap.hpp"

#include "File/FileSystem.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>

DISABLE_WARNINGS()
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
ENABLE_WARNINGS()

namespace Ilum
{
//
//inline bool generate_mipmaps(std::vector<uint8_t> &data, std::vector<Bitmap::Mipmap> &mipmaps, uint32_t bytes_per_pixel)
//{
//	if (mipmaps.size() > 1)
//	{
//		// Mipmaps already generated
//		return true;
//	}
//
//	uint32_t width  = mipmaps[0].extent.width;
//	uint32_t height = mipmaps[0].extent.height;
//
//	uint32_t next_width  = std::max<uint32_t>(1u, width / 2);
//	uint32_t next_heigth = std::max<uint32_t>(1u, height / 2);
//	size_t   next_size   = static_cast<size_t>(next_width) * static_cast<size_t>(next_heigth) * static_cast<size_t>(bytes_per_pixel);
//
//	while (true)
//	{
//		size_t old_size = static_cast<uint32_t>(data.size());
//		data.resize(old_size + next_size);
//
//		auto &prev_mipmap = mipmaps.back();
//
//		Bitmap::Mipmap next_mipmap = {};
//		next_mipmap.level          = prev_mipmap.level + 1;
//		next_mipmap.offset         = static_cast<uint32_t>(old_size);
//		next_mipmap.extent         = {next_width, next_heigth, 1u};
//
//		if (bytes_per_pixel == 4)
//		{
//			// R8G8B8A8_Unorm
//			stbir_resize_uint8(data.data() + prev_mipmap.offset, prev_mipmap.extent.width, prev_mipmap.extent.height, 0,
//			                   data.data() + next_mipmap.offset, next_mipmap.extent.width, next_mipmap.extent.height, 0, bytes_per_pixel);
//		}
//		else if (bytes_per_pixel == 16)
//		{
//			// R32G32B32A32_Float
//			stbir_resize_float(reinterpret_cast<float *>(data.data() + prev_mipmap.offset), prev_mipmap.extent.width, prev_mipmap.extent.height, 0,
//			                   reinterpret_cast<float *>(data.data() + next_mipmap.offset), next_mipmap.extent.width, next_mipmap.extent.height, 0, bytes_per_pixel / 4);
//		}
//		else
//		{
//			LOG_ERROR("Unsupport image type!");
//			return false;
//		}
//
//		mipmaps.emplace_back(std::move(next_mipmap));
//
//		next_width       = std::max<uint32_t>(1u, next_width / 2);
//		next_heigth      = std::max<uint32_t>(1u, next_heigth / 2);
//		size_t next_size = static_cast<size_t>(next_width) * static_cast<size_t>(next_heigth) * static_cast<size_t>(bytes_per_pixel);
//
//		if (next_width == 1 && next_heigth == 1)
//		{
//			break;
//		}
//	}
//
//	return true;
//}

//inline bool load_stb(const std::vector<uint8_t> &mem_data, std::vector<uint8_t> &data, uint32_t &width, uint32_t &height, uint32_t &bytes_per_pixel)
//{
//	int comp, req_comp = 4, w = 0, h = 0;
//
//	auto data_buffer = reinterpret_cast<const stbi_uc *>(mem_data.data());
//	auto data_size   = static_cast<int>(mem_data.size());
//
//	bytes_per_pixel = stbi_is_hdr_from_memory(data_buffer, data_size) ? 16 : 4;
//
//	if (bytes_per_pixel == 1)
//	{
//		auto stb_data = stbi_load_from_memory(data_buffer, data_size, &w, &h, &comp, req_comp);
//		if (!stb_data)
//		{
//			return false;
//		}
//		data.resize(data.size() + w * h * req_comp);
//		std::memcpy(data.data(), stb_data, w * h * req_comp);
//		stbi_image_free(stb_data);
//	}
//	else
//	{
//		auto stb_data = stbi_loadf_from_memory(data_buffer, data_size, &w, &h, &comp, req_comp);
//		if (!stb_data)
//		{
//			return false;
//		}
//		data.resize(data.size() + w * h * bytes_per_pixel);
//		std::memcpy(data.data(), stb_data, w * h * bytes_per_pixel);
//		stbi_image_free(stb_data);
//	}
//
//	width  = static_cast<uint32_t>(w);
//	height = static_cast<uint32_t>(h);
//	return true;
//}

Bitmap::Bitmap(const uint32_t width, const uint32_t height, const uint32_t bytes_per_pixel) :
    m_width(width),
    m_height(height),
    m_bytes_per_pixel(bytes_per_pixel),
    m_data(createScope<uint8_t[]>(getLength()))
{
}

Bitmap::Bitmap(scope<uint8_t[]> &&data, const uint32_t width, const uint32_t height, const uint32_t bytes_per_pixel) :
    m_width(width),
    m_height(height),
    m_bytes_per_pixel(bytes_per_pixel),
    m_data(std::move(data))
{
}

Bitmap::Bitmap(const std::string &path) :
    m_path(path)
{
	load(path);
}

void Bitmap::load(const std::string &path)
{
	auto extension = FileSystem::getFileExtension(path);

	std::vector<uint8_t> mem_data;
	std::vector<uint8_t> data;
	FileSystem::read(path, mem_data, true);

	if (extension == ".jpg" || extension == ".png" || extension == ".jpeg" || extension == ".bmp" || extension == ".hdr")
	{
		if (stbi_is_hdr_from_memory(mem_data.data(), static_cast<int32_t>(mem_data.size())))
		{
			// HDR
			m_data            = scope<uint8_t[]>(reinterpret_cast<uint8_t *>(stbi_loadf_from_memory(mem_data.data(), static_cast<int32_t>(mem_data.size()), reinterpret_cast<int32_t *>(&m_width), reinterpret_cast<int32_t *>(&m_height), reinterpret_cast<int32_t *>(&m_bytes_per_pixel), STBI_rgb_alpha)));
			m_bytes_per_pixel = 16;
		}
		else
		{
			m_data            = scope<uint8_t[]>(stbi_load_from_memory(mem_data.data(), static_cast<int32_t>(mem_data.size()), reinterpret_cast<int32_t *>(&m_width), reinterpret_cast<int32_t *>(&m_height), reinterpret_cast<int32_t *>(&m_bytes_per_pixel), STBI_rgb_alpha));
			m_bytes_per_pixel = 4;
		}
	}
}

bool Bitmap::write(const std::string &path)
{
	if (m_bytes_per_pixel == 4)
	{
		return stbi_write_png(path.c_str(), m_width, m_height, 4, m_data.get(), m_width * 4);
	}
	else if (m_bytes_per_pixel == 16)
	{
		return stbi_write_hdr(path.c_str(), m_width, m_height, 4, reinterpret_cast<float *>(m_data.get()));
	}

	return false;
}

Bitmap::operator bool() const noexcept
{
	return !m_data;
}

const std::string &Bitmap::getPath() const
{
	return m_path;
}

const scope<uint8_t[]> &Bitmap::getData() const
{
	return m_data;
}

const void Bitmap::setData(scope<uint8_t[]> &&data)
{
	m_data = std::move(data);
}

const uint32_t Bitmap::getWidth() const
{
	return m_width;
}

const uint32_t Bitmap::getHeight() const
{
	return m_height;
}

const uint32_t Bitmap::getBytesPerPixel() const
{
	return m_bytes_per_pixel;
}

const uint32_t Bitmap::getLength() const
{
	return m_width * m_height * m_bytes_per_pixel;
}

scope<Bitmap> Bitmap::create(const std::string &path)
{
	return createScope<Bitmap>(path);
}
}        // namespace Ilum