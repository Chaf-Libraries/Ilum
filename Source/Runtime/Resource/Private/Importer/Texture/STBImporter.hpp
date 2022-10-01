#pragma once

#include "TextureImporter.hpp"

#include <Core/Singleton.hpp>

namespace Ilum
{
class RHIContext;

class STBImporter : public TextureImporter, public Singleton<STBImporter>
{
  public:
	virtual TextureImportInfo ImportImpl(const std::string &filename) override;
};
}        // namespace Ilum