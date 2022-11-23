#pragma once

#include "ModelImporter.hpp"

namespace Ilum
{
class AssimpImporter : public ModelImporter
{
  public:
	static AssimpImporter &GetInstance();

	virtual ModelImportInfo ImportImpl(const std::string &filename) override;
};
}        // namespace Ilum