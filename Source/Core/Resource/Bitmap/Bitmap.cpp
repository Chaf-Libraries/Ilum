#include "Bitmap.hpp"

namespace Ilum
{
Bitmap::Bitmap(const std::string &path):
    m_path(path)
{
}

Bitmap::Bitmap(const uint32_t width, const uint32_t height, const uint32_t bytes_per_pixel, std::unique_ptr<uint8_t[]> &&data)
{
}

void Bitmap::load(const std::string &path)
{
}

void Bitmap::write(const std::string &path)
{
}

Bitmap::operator bool() const noexcept
{
	return !m_data.empty();
}

const std::string &Bitmap::getPath() const
{
	return m_path;
}

const std::vector<uint8_t> &Bitmap::getData() const
{
	return m_data;
}

std::vector<uint8_t> &Bitmap::getData()
{
	return m_data;
}

void Bitmap::setData(std::vector<uint8_t> &&data)
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

const uint32_t Bitmap::getChannel() const
{
	return m_channel;
}

const uint32_t Bitmap::getBytesPerPixel() const
{
	return m_bytes_per_pixel;
}
}        // namespace Ilum