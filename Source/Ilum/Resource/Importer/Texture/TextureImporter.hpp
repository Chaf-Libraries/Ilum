#pragma once

#include <string>
#include <vector>

namespace Ilum
{
class RHIContext;
struct TextureDesc;

class TextureImporter
{
  public:
	virtual void Import(RHIContext *rhi_context, const std::string &filename, std::vector<uint8_t> &data, TextureDesc &desc) = 0;
};
}        // namespace Ilum