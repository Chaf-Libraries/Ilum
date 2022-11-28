#pragma once

#include "TextureImporter.hpp"

namespace Ilum
{
class RHIContext;

class STBImporter : public TextureImporter
{
  public:
	static STBImporter &GetInstance();

	virtual TextureImportInfo ImportImpl(const std::string &filename) override;
};
}        // namespace Ilum