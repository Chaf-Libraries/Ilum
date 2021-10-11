#pragma once

#include "Utils/PCH.hpp"

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
	Bitmap(const uint32_t width, const uint32_t height, const uint32_t bytes_per_pixel = 4);

	Bitmap(scope<uint8_t[]> &&data, const uint32_t width, const uint32_t height, const uint32_t bytes_per_pixel = 4);

	Bitmap(const std::string &path);

	~Bitmap() = default;

	void load(const std::string &path);

	bool write(const std::string &path);

	operator bool() const noexcept;

	const std::string &getPath() const;

	const scope<uint8_t[]> &getData() const;

	const void setData(scope<uint8_t[]> &&data);

	const uint32_t getWidth() const;

	const uint32_t getHeight() const;

	const uint32_t getBytesPerPixel() const;

	const uint32_t getLength() const;

  public:
	static scope<Bitmap> create(const std::string &path);

  private:
	const std::string m_path            = "";
	scope<uint8_t[]>    m_data            = nullptr;
	uint32_t          m_width           = 0;
	uint32_t          m_height          = 0;
	uint32_t          m_bytes_per_pixel = 1;
};
}        // namespace Ilum