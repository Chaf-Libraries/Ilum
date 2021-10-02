#pragma once

#include "Core/Engine/PCH.hpp"

namespace Ilum
{
class Bitmap
{
  public:
	struct Mipmap
	{
		uint32_t   level  = 0;
		uint32_t   offset = 0;
		VkExtent3D extent = {0, 0, 0};
	};

  public:
	Bitmap() = default;

	Bitmap(const std::string &path);

	//Bitmap(std::vector<uint8_t> &&data = {}, std::vector<Mipmap> &&mipmaps = {{}});

	~Bitmap() = default;

	void load(const std::string &path);

	void write(const std::string &path);

	operator bool() const noexcept;

	const std::string &getPath() const;

	const std::vector<uint8_t> &getData() const;

	const std::vector<Mipmap> &getMipmaps() const;

	const std::vector<std::vector<VkDeviceSize>> &getOffsets() const;

	void setData(std::vector<uint8_t> &&data, std::vector<Mipmap> &&mipmaps);

	const uint32_t getWidth() const;

	const uint32_t getHeight() const;

	const uint32_t getChannel() const;

	const uint32_t getBytesPerPixel() const;

  private:
	const std::string                      m_path = "";
	std::vector<uint8_t>                   m_data;
	std::vector<Mipmap>                    m_mipmaps = {{}};
	std::vector<std::vector<VkDeviceSize>> m_offsets = {};

	uint32_t m_bytes_per_pixel = 1;
};
}        // namespace Ilum