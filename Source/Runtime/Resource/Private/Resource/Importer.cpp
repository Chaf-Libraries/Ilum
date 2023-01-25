#include "Importer.hpp"
#include "Resource/Prefab.hpp"
#include "Resource/Texture2D.hpp"
#include "Resource/Animation.hpp"

#include <Core/Plugin.hpp>

namespace Ilum
{
const std::map<ResourceType, std::map<std::string, std::string>> PluginMap = {
    {ResourceType::Texture2D,
     {
         {".png", "STB"},
         {".jpg", "STB"},
         {".jpeg", "STB"},
         {".bmp", "STB"},
         {".dds", "DDS"},
     }},
    {ResourceType::Prefab,
     {
         {".obj", "Assimp"},
         {".gltf", "Assimp"},
         {".glb", "Assimp"},
         {".dae", "Assimp"},
         {".fbx", "Assimp"},
         {".ply", "Assimp"},
         {".blend", "Assimp"},
     }},
};

template <ResourceType Type>
std::unique_ptr<Importer<Type>> &Importer<Type>::GetInstance(const std::string &plugin)
{
	static std::unique_ptr<Importer<Type>> importer = std::unique_ptr<Importer<Type>>(PluginManager::GetInstance().Call<Importer<Type> *>(fmt::format("shared/Importer/Importer.{}.dll", plugin), "Create"));
	return importer;
}

template <ResourceType Type>
void Importer<Type>::Import(ResourceManager *manager, const std::string &path, RHIContext *rhi_context)
{
	return GetInstance(PluginMap.at(Type).at(Path::GetInstance().GetFileExtension(path)))->Import_(manager, path, rhi_context);
}

template class Importer<ResourceType::Prefab>;
template class Importer<ResourceType::Texture2D>;
template class Importer<ResourceType::Mesh>;
template class Importer<ResourceType::SkinnedMesh>;
template class Importer<ResourceType::Material>;
template class Importer<ResourceType::Animation>;
template class Importer<ResourceType::RenderPipeline>;
template class Importer<ResourceType::Scene>;
}        // namespace Ilum