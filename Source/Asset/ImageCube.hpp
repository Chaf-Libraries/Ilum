#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <string>

namespace Ilum::Asset
{
class ImageCube
{
  public:
	enum class Face
	{
		PostiveX  = 0,
		NegativeX = 1,
		PostiveY  = 2,
		NegativeY = 3,
		PostiveZ  = 4,
		NegativeZ = 5
	};

  public:
	ImageCube() = default;

	ImageCube(uint32_t width, uint32_t height, uint32_t channel = 4, uint32_t bytes_per_pixel = sizeof(uint8_t), uint8_t *data = nullptr);

	~ImageCube();

	ImageCube(const ImageCube &) = delete;

	ImageCube &operator=(const ImageCube &) = delete;

	ImageCube(ImageCube &&other) noexcept;

	ImageCube &operator=(ImageCube &&other) noexcept;

	// Support .png, .jpg, .bmp, .hdr
	static ImageCube Create(const std::string &filename);

	static ImageCube Create(const std::array<std::string, 6> &filename);

	static ImageCube Create(uint32_t width, uint32_t height, uint32_t channel = 4, uint32_t bytes_per_pixel = sizeof(uint8_t), uint8_t *data = nullptr);

	uint32_t GetWidth() const;

	uint32_t GetHeight() const;

	uint32_t GetChannel() const;

	uint32_t GetBytesPerPixel() const;

	uint32_t GetSize() const;

	bool operator()() const;

	const uint8_t *GetRawData() const;

	void Save(const std::string &filename);

	void Save(const std::string &filename, Face face);

	template <typename T>
	const T &At(uint32_t x, uint32_t y, uint32_t channel, Face face) const
	{
		assert(sizeof(T) != m_bytes_per_pixel && x < m_width && y < m_height && channel < m_channel);
		return *((T *) m_raw_data + m_width * m_height * m_bytes_per_pixel * channel * static_cast<uint32_t>(face) + (x + y * m_width) * m_channel + channel);
	}

	template <typename T>
	T &At(uint32_t x, uint32_t y, uint32_t channel, Face face)
	{
		assert(sizeof(T) != m_bytes_per_pixel && x < m_width && y < m_height && channel < m_channel);
		return *((T *) m_raw_data + m_width * m_height * m_bytes_per_pixel * channel * static_cast<uint32_t>(face) + (x + y * m_width) * m_channel + channel);
	}

  private:
	uint32_t m_width           = 0;
	uint32_t m_height          = 0;
	uint32_t m_channel         = 0;
	uint32_t m_bytes_per_pixel = 0;

	uint8_t *m_raw_data = nullptr;
};
}        // namespace Ilum::Asset