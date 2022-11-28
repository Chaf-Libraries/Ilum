#include "ModelImporter.hpp"
#include "AssimpImporter.hpp"

#include <Core/Path.hpp>

namespace Ilum
{
ModelImportInfo ModelImporter::Import(const std::string &filename)
{
	std::string extension = Path::GetInstance().GetFileExtension(filename);

	if (extension == ".gltf" || extension == ".obj" || extension == ".fbx" || extension == "ply")
	{
		return AssimpImporter::GetInstance().ImportImpl(filename);
	}

	return {};
}

uint32_t ModelImporter::PackTriangle(uint8_t v0, uint8_t v1, uint8_t v2)
{
	return static_cast<uint32_t>(v0) + (static_cast<uint32_t>(v1) << 8) + (static_cast<uint32_t>(v2) << 16);
}
}        // namespace Ilum