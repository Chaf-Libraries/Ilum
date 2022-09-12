#include "ModelImporter.hpp"
#include "AssimpImporter.hpp"

#include <Core/Path.hpp>

namespace Ilum
{
ModelImportInfo ModelImporter::Import(const std::string &filename)
{
	std::string extension = Path::GetInstance().GetFileExtension(filename);

	if (extension == ".gltf")
	{
		return AssimpImporter::GetInstance().ImportImpl(filename);
	}

	return {};
}
}        // namespace Ilum