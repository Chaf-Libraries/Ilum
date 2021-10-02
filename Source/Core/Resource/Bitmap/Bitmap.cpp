#include "Bitmap.hpp"
#include "Stb.hpp"

#include "Core/Engine/File/FileSystem.hpp"

namespace Ilum
{
Bitmap::Bitmap(const std::string &path) :
    m_path(path)
{
	load(path);
}

//Bitmap::Bitmap(std::vector<uint8_t> &&data, std::vector<Bitmap::Mipmap> &&mipmaps) :
//    m_data(std::move(data)), m_mipmaps(std::move(mipmaps))
//{
//}

void Bitmap::load(const std::string &path)
{
	auto extension = FileSystem::getFileExtension(path);

	std::vector<uint8_t> mem_data;
	FileSystem::read(path, mem_data, true);

	if (extension == ".jpg" || extension == ".png" || extension == ".jpeg" || extension == ".bmp" || extension == ".hdr")
	{
		Stb::load(mem_data, m_data, m_mipmaps[0].extent.width, m_mipmaps[0].extent.height, m_mipmaps[0].extent.depth, m_bytes_per_pixel);
	}
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

const std::vector<Bitmap::Mipmap> &Bitmap::getMipmaps() const
{
	return m_mipmaps;
}

const std::vector<std::vector<VkDeviceSize>> &Bitmap::getOffsets() const
{
	return m_offsets;
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
}        // namespace Ilum