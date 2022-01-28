#pragma once

#include <cassert>
#include <cstdint>
#include <string>

namespace Ilum::Asset
{
// Keep it simple, 2D image, no mipmap
class Image2D
{
  public:
	Image2D() = default;

	Image2D(uint32_t width, uint32_t height, uint32_t channel = 4, uint32_t bytes_per_pixel = sizeof(uint8_t), uint8_t *data = nullptr);

	~Image2D();

	Image2D(const Image2D &) = delete;

	Image2D &operator=(const Image2D &) = delete;

	Image2D(Image2D &&other) noexcept;

	Image2D &operator=(Image2D &&other) noexcept;

	// Support .png, .jpg, .bmp, .hdr
	static Image2D Create(const std::string &filename);

	static Image2D Create(uint32_t width, uint32_t height, uint32_t channel = 4, uint32_t bytes_per_pixel = sizeof(uint8_t), uint8_t *data = nullptr);

	uint32_t GetWidth() const;

	uint32_t GetHeight() const;

	uint32_t GetChannel() const;

	uint32_t GetBytesPerPixel() const;

	uint32_t GetSize() const;

	bool operator()() const;

	const uint8_t *GetRawData() const;

	// 1 byte save as .png, 2 byte & 4 byte save as .hdr
	void Save(const std::string &filename);

	template <typename T>
	const T &At(uint32_t x, uint32_t y, uint32_t channel) const
	{
		assert(sizeof(T) != m_bytes_per_pixel && x < m_width && y < m_height && channel < m_channel);
		return *((T *) m_raw_data + (x + y * m_width) * m_channel + channel);
	}

	template <typename T>
	T &At(uint32_t x, uint32_t y, uint32_t channel)
	{
		assert(sizeof(T) != m_bytes_per_pixel && x < m_width && y < m_height && channel < m_channel);
		return *((T *) m_raw_data + (x + y * m_width) * m_channel + channel);
	}

  private:
	uint32_t m_width           = 0;
	uint32_t m_height          = 0;
	uint32_t m_channel         = 0;
	uint32_t m_bytes_per_pixel = 0;

	uint8_t *m_raw_data = nullptr;
};
}        // namespace Ilum::Asset
