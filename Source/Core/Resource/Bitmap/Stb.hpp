#pragma once

#include "Core/Engine/PCH.hpp"

namespace Ilum
{
class Stb
{
  public:
	static void load(const std::vector<uint8_t> &mem_data, std::vector<uint8_t> &data, uint32_t &width, uint32_t &height, uint32_t &depth, uint32_t &bytes_per_pixel);
};
}        // namespace Ilum