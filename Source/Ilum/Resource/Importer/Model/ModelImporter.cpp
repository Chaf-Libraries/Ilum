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

uint32_t ModelImporter::PackTriangle(uint8_t v0, uint8_t v1, uint8_t v2)
{
	return (v0 & 0xff) + ((v1 & 0xff) << 8) + ((v2 & 0xff) << 16);
}
}        // namespace Ilum