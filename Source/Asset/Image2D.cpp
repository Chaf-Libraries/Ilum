#include "Image2D.hpp"

#pragma warning(push, 0)
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#pragma warning(pop)

#include <filesystem>

namespace Ilum::Asset
{
Image2D::Image2D(uint32_t width, uint32_t height, uint32_t channel, uint32_t bytes_per_pixel, uint8_t *data) :
    m_width(width),
    m_height(height),
    m_channel(channel),
    m_bytes_per_pixel(bytes_per_pixel)
{
	uint32_t size = GetSize();
	if (size > 0)
	{
		m_raw_data = new uint8_t[size];
		std::memcpy(m_raw_data, data, size);
	}
}

Image2D::~Image2D()
{
	if (m_raw_data)
	{
		delete[] m_raw_data;
		m_raw_data = nullptr;
	}
}

Image2D::Image2D(Image2D &&other) noexcept :
    m_width(other.m_width),
    m_height(other.m_height),
    m_channel(other.m_channel),
    m_bytes_per_pixel(other.m_bytes_per_pixel),
    m_raw_data(other.m_raw_data)
{
	other.m_raw_data = nullptr;
}

Image2D &Image2D::operator=(Image2D &&other) noexcept
{
	m_width           = other.m_width;
	m_height          = other.m_height;
	m_channel         = other.m_channel;
	m_bytes_per_pixel = other.m_bytes_per_pixel;

	if (m_raw_data)
	{
		delete[] m_raw_data;
	}

	m_raw_data       = other.m_raw_data;
	other.m_raw_data = nullptr;
	return *this;
}

Image2D Image2D::Create(const std::string &filename)
{
	auto extension = std::filesystem::path(filename).extension().generic_string();

	Image2D image;

	if (extension == ".png" || extension == ".jpg" || extension == ".bmp" || extension == ".jpeg")
	{
		// Use stb lib
		int width = 0, height = 0, channel = 0;

		uint8_t *data = stbi_load(filename.c_str(), &width, &height, &channel, 0);

		if (data)
		{
			image.m_width           = width;
			image.m_height          = height;
			image.m_channel         = channel;
			image.m_bytes_per_pixel = sizeof(uint8_t);
			uint32_t size           = image.GetSize();
			image.m_raw_data        = new uint8_t[size];

			std::memcpy(image.m_raw_data, data, size);
		}

		stbi_image_free(data);
	}
	else if (extension == ".hdr")
	{
		// Use stb lib
		int width = 0, height = 0, channel = 0;

		float *data = stbi_loadf(filename.c_str(), &width, &height, &channel, 0);

		if (data)
		{
			image.m_width           = width;
			image.m_height          = height;
			image.m_channel         = channel;
			image.m_bytes_per_pixel = sizeof(float);
			uint32_t size           = image.GetSize();
			image.m_raw_data        = new uint8_t[size];

			std::memcpy(image.m_raw_data, data, size);
		}

		stbi_image_free(data);
	}

	return image;
}

Image2D Image2D::Create(uint32_t width, uint32_t height, uint32_t channel, uint32_t bytes_per_pixel, uint8_t *data)
{
	return Image2D(width, height, channel, bytes_per_pixel, data);
}

uint32_t Image2D::GetWidth() const
{
	return m_width;
}

uint32_t Image2D::GetHeight() const
{
	return m_height;
}

uint32_t Image2D::GetChannel() const
{
	return m_channel;
}

uint32_t Image2D::GetBytesPerPixel() const
{
	return m_bytes_per_pixel;
}

uint32_t Image2D::GetSize() const
{
	return m_width * m_height * m_channel * m_bytes_per_pixel;
}

bool Image2D::operator()() const
{
	return m_raw_data != nullptr;
}

const uint8_t *Image2D::GetRawData() const
{
	return m_raw_data;
}

void Image2D::Save(const std::string &filename)
{
	if (m_bytes_per_pixel == sizeof(uint8_t))
	{
		// Save as .png
		stbi_write_png((filename + ".png").c_str(), m_width, m_height, m_channel, m_raw_data, m_width * m_channel);
	}
	else if (m_bytes_per_pixel == sizeof(float))
	{
		// Save as .hdr
		stbi_write_hdr((filename + ".hdr").c_str(), m_width, m_height, m_channel, reinterpret_cast<float *>(m_raw_data));
	}
	else
	{
		assert(false && "Failed to save image!");
	}
}
}        // namespace Ilum::Asset