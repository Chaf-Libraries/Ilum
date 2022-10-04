#pragma once

#include <RHI/RHITexture.hpp>

namespace Ilum
{
struct TextureImportInfo
{
	TextureDesc desc;

	std::vector<uint8_t> data;
};

class TextureImporter
{
  public:
	virtual TextureImportInfo ImportImpl(const std::string &filename) = 0;

	static TextureImportInfo Import(const std::string &filename);

	static TextureImportInfo ImportFromBuffer(const std::vector<uint8_t> &raw_data);
};
}