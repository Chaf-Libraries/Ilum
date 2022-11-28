#pragma once

#include "TextureImporter.hpp"

namespace Ilum
{
class RHIContext;

class DDSImporter : public TextureImporter
{
  public:
	static DDSImporter &GetInstance();

	virtual TextureImportInfo ImportImpl(const std::string &filename) override;
};
}        // namespace Ilum