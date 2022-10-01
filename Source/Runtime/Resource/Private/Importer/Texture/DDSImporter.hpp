#pragma once

#include "TextureImporter.hpp"

#include <Core/Singleton.hpp>

namespace Ilum
{
class RHIContext;

class DDSImporter : public TextureImporter, public Singleton<DDSImporter>
{
  public:
	virtual TextureImportInfo ImportImpl(const std::string &filename) override;
};
}        // namespace Ilum