#pragma once

#include "ModelImporter.hpp"

namespace Ilum
{
class AssimpImporter : public ModelImporter, public Singleton<AssimpImporter>
{
  public:
	virtual ModelImportInfo ImportImpl(const std::string &filename) override;
};
}        // namespace Ilum