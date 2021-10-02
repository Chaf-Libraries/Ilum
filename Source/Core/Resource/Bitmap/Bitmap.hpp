#pragma once

#include "Core/Engine/PCH.hpp"

namespace Ilum
{
class Bitmap
{
  public:
	Bitmap() = default;

	Bitmap(const std::string &path);

	Bitmap(const uint32_t width, const uint32_t height, const uint32_t bytes_per_pixel = 4, std::unique_ptr<uint8_t[]> &&data = nullptr);

	~Bitmap() = default;

	void load(const std::string &path);

	void write(const std::string &path);

	operator bool() const noexcept;

	const std::string &getPath() const;

	const std::vector<uint8_t> &getData() const;

	std::vector<uint8_t> &getData();

	void setData(std::vector<uint8_t> &&data);

	const uint32_t getWidth() const;

	const uint32_t getHeight() const;

	const uint32_t getChannel() const;

	const uint32_t getBytesPerPixel() const;

  private:
	const std::string    m_path = "";
	std::vector<uint8_t> m_data;
	uint32_t             m_width           = 0;
	uint32_t             m_height          = 0;
	uint32_t             m_channel         = 0;
	uint32_t             m_bytes_per_pixel = 0;
};
}        // namespace Ilum