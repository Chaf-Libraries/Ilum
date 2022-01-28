#include "ImageCube.hpp"
#include "Image2D.hpp"

#include <filesystem>

namespace Ilum::Asset
{
ImageCube::ImageCube(uint32_t width, uint32_t height, uint32_t channel, uint32_t bytes_per_pixel, uint8_t *data) :
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

ImageCube::~ImageCube()
{
	if (m_raw_data)
	{
		delete[] m_raw_data;
		m_raw_data = nullptr;
	}
}

ImageCube::ImageCube(ImageCube &&other) noexcept :
    m_width(other.m_width),
    m_height(other.m_height),
    m_channel(other.m_channel),
    m_bytes_per_pixel(other.m_bytes_per_pixel),
    m_raw_data(other.m_raw_data)
{
	other.m_raw_data = nullptr;
}

ImageCube &ImageCube::operator=(ImageCube &&other) noexcept
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

ImageCube ImageCube::Create(const std::string &filename)
{
	// Load as 2D image
	Image2D image_2d = Image2D::Create(filename);

	ImageCube image_cube;

	image_cube.m_width           = image_2d.GetWidth() / 4;
	image_cube.m_height          = image_2d.GetHeight() / 3;
	image_cube.m_channel         = image_2d.GetChannel();
	image_cube.m_bytes_per_pixel = image_2d.GetBytesPerPixel();
	image_cube.m_raw_data        = new uint8_t[image_cube.GetSize()];

	uint32_t offset = 0;

	// +X
	for (uint32_t y = image_cube.m_height; y < 2 * image_cube.m_height; y++)
	{
		for (uint32_t x = 0; x < image_cube.m_width; x++)
		{
			std::memcpy(image_cube.m_raw_data + image_cube.m_channel * image_cube.m_bytes_per_pixel * (offset++),
			            image_2d.GetRawData() + image_2d.GetChannel() * image_2d.GetBytesPerPixel() * (x + y * image_2d.GetWidth()),
			            image_cube.m_channel * image_cube.m_bytes_per_pixel);
		}
	}

	// -X
	for (uint32_t y = image_cube.m_height; y < 2 * image_cube.m_height; y++)
	{
		for (uint32_t x = 2 * image_cube.m_width; x < 3 * image_cube.m_width; x++)
		{
			std::memcpy(image_cube.m_raw_data + image_cube.m_channel * image_cube.m_bytes_per_pixel * (offset++),
			            image_2d.GetRawData() + image_2d.GetChannel() * image_2d.GetBytesPerPixel() * (x + y * image_2d.GetWidth()),
			            image_cube.m_channel * image_cube.m_bytes_per_pixel);
		}
	}

	// +Y
	for (uint32_t y = image_cube.m_height; y < 2 * image_cube.m_height; y++)
	{
		for (uint32_t x = image_cube.m_width; x < 2 * image_cube.m_width; x++)
		{
			std::memcpy(image_cube.m_raw_data + image_cube.m_channel * image_cube.m_bytes_per_pixel * (offset++),
			            image_2d.GetRawData() + image_2d.GetChannel() * image_2d.GetBytesPerPixel() * (x + y * image_2d.GetWidth()),
			            image_cube.m_channel * image_cube.m_bytes_per_pixel);
		}
	}

	// -Y
	for (uint32_t y = image_cube.m_height; y < 2 * image_cube.m_height; y++)
	{
		for (uint32_t x = 3 * image_cube.m_width; x < 4 * image_cube.m_width; x++)
		{
			std::memcpy(image_cube.m_raw_data + image_cube.m_channel * image_cube.m_bytes_per_pixel * (offset++),
			            image_2d.GetRawData() + image_2d.GetChannel() * image_2d.GetBytesPerPixel() * (x + y * image_2d.GetWidth()),
			            image_cube.m_channel * image_cube.m_bytes_per_pixel);
		}
	}

	// +Z
	for (uint32_t y = 0; y < image_cube.m_height; y++)
	{
		for (uint32_t x = image_cube.m_width; x < 2 * image_cube.m_width; x++)
		{
			std::memcpy(image_cube.m_raw_data + image_cube.m_channel * image_cube.m_bytes_per_pixel * (offset++),
			            image_2d.GetRawData() + image_2d.GetChannel() * image_2d.GetBytesPerPixel() * (x + y * image_2d.GetWidth()),
			            image_cube.m_channel * image_cube.m_bytes_per_pixel);
		}
	}

	// -Z
	for (uint32_t y = 2 * image_cube.m_height; y < 3 * image_cube.m_height; y++)
	{
		for (uint32_t x = image_cube.m_width; x < 2 * image_cube.m_width; x++)
		{
			std::memcpy(image_cube.m_raw_data + image_cube.m_channel * image_cube.m_bytes_per_pixel * (offset++),
			            image_2d.GetRawData() + image_2d.GetChannel() * image_2d.GetBytesPerPixel() * (x + y * image_2d.GetWidth()),
			            image_cube.m_channel * image_cube.m_bytes_per_pixel);
		}
	}

	return image_cube;
}

ImageCube ImageCube::Create(const std::array<std::string, 6> &filename)
{
	ImageCube image_cube;

	for (uint32_t i = 0; i < 6; i++)
	{
		Image2D image_2d = Image2D::Create(filename.at(i));

		if (i == 0)
		{
			image_cube.m_width           = image_2d.GetWidth();
			image_cube.m_height          = image_2d.GetHeight();
			image_cube.m_channel         = image_2d.GetChannel();
			image_cube.m_bytes_per_pixel = image_2d.GetBytesPerPixel();
			image_cube.m_raw_data        = new uint8_t[image_cube.GetSize()];
		}

		uint32_t offset = image_2d.GetSize() * i;
		std::memcpy(image_cube.m_raw_data + offset, image_2d.GetRawData(), image_2d.GetSize());
	}

	return image_cube;
}

ImageCube ImageCube::Create(uint32_t width, uint32_t height, uint32_t channel, uint32_t bytes_per_pixel, uint8_t *data)
{
	return ImageCube(width, height, channel, bytes_per_pixel, data);
}

uint32_t ImageCube::GetWidth() const
{
	return m_width;
}

uint32_t ImageCube::GetHeight() const
{
	return m_height;
}

uint32_t ImageCube::GetChannel() const
{
	return m_channel;
}

uint32_t ImageCube::GetBytesPerPixel() const
{
	return m_bytes_per_pixel;
}

uint32_t ImageCube::GetSize() const
{
	return m_width * m_height * m_channel * m_bytes_per_pixel * 6;
}

bool ImageCube::operator()() const
{
	return m_raw_data != nullptr;
}

const uint8_t *ImageCube::GetRawData() const
{
	return m_raw_data;
}

void ImageCube::Save(const std::string &filename)
{
	uint8_t *data = new uint8_t[m_width * 4 * m_height * 3 * m_channel * m_bytes_per_pixel];
	std::memset(data, 0, m_width * 4 * m_height * 3 * m_channel * m_bytes_per_pixel);

	uint32_t offset = 0;

	// +X
	for (uint32_t y = m_height; y < 2 * m_height; y++)
	{
		for (uint32_t x = 0; x < m_width; x++)
		{
			std::memcpy(data + m_channel * m_bytes_per_pixel * (x + y * m_width * 4),
			            m_raw_data + m_channel * m_bytes_per_pixel * (offset++),
			            m_channel * m_bytes_per_pixel);
		}
	}

	// -X
	for (uint32_t y = m_height; y < 2 * m_height; y++)
	{
		for (uint32_t x = 2 * m_width; x < 3 * m_width; x++)
		{
			std::memcpy(data + m_channel * m_bytes_per_pixel * (x + y * m_width * 4),
			            m_raw_data + m_channel * m_bytes_per_pixel * (offset++),
			            m_channel * m_bytes_per_pixel);
		}
	}

	// +Y
	for (uint32_t y = m_height; y < 2 * m_height; y++)
	{
		for (uint32_t x = m_width; x < 2 * m_width; x++)
		{
			std::memcpy(data + m_channel * m_bytes_per_pixel * (x + y * m_width * 4),
			            m_raw_data + m_channel * m_bytes_per_pixel * (offset++),
			            m_channel * m_bytes_per_pixel);
		}
	}

	// -Y
	for (uint32_t y = m_height; y < 2 * m_height; y++)
	{
		for (uint32_t x = 3 * m_width; x < 4 * m_width; x++)
		{
			std::memcpy(data + m_channel * m_bytes_per_pixel * (x + y * m_width * 4),
			            m_raw_data + m_channel * m_bytes_per_pixel * (offset++),
			            m_channel * m_bytes_per_pixel);
		}
	}

	// +Z
	for (uint32_t y = 0; y < m_height; y++)
	{
		for (uint32_t x = m_width; x < 2 * m_width; x++)
		{
			std::memcpy(data + m_channel * m_bytes_per_pixel * (x + y * m_width * 4),
			            m_raw_data + m_channel * m_bytes_per_pixel * (offset++),
			            m_channel * m_bytes_per_pixel);
		}
	}

	// -Z
	for (uint32_t y = 2 * m_height; y < 3 * m_height; y++)
	{
		for (uint32_t x = m_width; x < 2 * m_width; x++)
		{
			std::memcpy(data + m_channel * m_bytes_per_pixel * (x + y * m_width * 4),
			            m_raw_data + m_channel * m_bytes_per_pixel * (offset++),
			            m_channel * m_bytes_per_pixel);
		}
	}

	Image2D image_2d(m_width * 4, m_height * 3, m_channel, m_bytes_per_pixel, data);
	delete[] data;

	image_2d.Save(filename);
}

void ImageCube::Save(const std::string &filename, Face face)
{
	uint32_t offset = m_width * m_height * m_bytes_per_pixel * m_channel * static_cast<uint32_t>(face);

	auto image_2d = Image2D::Create(m_width, m_height, m_channel, m_bytes_per_pixel, m_raw_data + offset);
	image_2d.Save(filename);
}
}        // namespace Ilum::Asset