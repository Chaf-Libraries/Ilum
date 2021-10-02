#include "Stb.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace Ilum
{
void Stb::load(const std::vector<uint8_t> &mem_data, std::vector<uint8_t> &data, uint32_t &width, uint32_t &height, uint32_t &depth, uint32_t &bytes_per_pixel)
{
	int comp, req_comp = 4, w = 0, h = 0;

	auto data_buffer = reinterpret_cast<const stbi_uc *>(mem_data.data());
	auto data_size   = static_cast<int>(mem_data.size());

	bytes_per_pixel = stbi_is_hdr_from_memory(data_buffer, data_size) ? 4 : 1;

	if (bytes_per_pixel == 1)
	{
		auto stb_data = stbi_load_from_memory(data_buffer, data_size, &w, &h, &comp, req_comp);
		data.resize(data.size() + w * h * req_comp);
		std::memcpy(data.data(), stb_data, w * h * req_comp);
		stbi_image_free(stb_data);
	}
	else
	{
		auto stb_data = stbi_loadf_from_memory(data_buffer, data_size, &w, &h, &comp, req_comp);
		data.resize(data.size() + w * h * req_comp * 4);
		std::memcpy(data.data(), stb_data, w * h * req_comp * 4);
		stbi_image_free(stb_data);
	}

	width  = static_cast<uint32_t>(w);
	height = static_cast<uint32_t>(h);
	depth  = static_cast<uint32_t>(comp);
}
}        // namespace Ilum