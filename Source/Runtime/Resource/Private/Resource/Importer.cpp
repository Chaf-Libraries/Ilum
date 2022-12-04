#include "Importer.hpp"
#include "Resource/Model.hpp"
#include "Resource/Texture.hpp"

#include <Core/Plugin.hpp>

namespace Ilum
{
const std::map<ResourceType, std::map<std::string, std::string>> PluginMap = {
    {ResourceType::Texture,
     {
         {".png", "STB"},
         {".jpg", "STB"},
         {".jpeg", "STB"},
         {".bmp", "STB"},
         {".dds", "DDS"},
     }},
    {ResourceType::Model,
     {
         {".obj", "Assimp.Model"},
         {".gltf", "Assimp.Model"},
         {".glb", "Assimp.Model"},
         {".dae", "Assimp.Model"},
         {".fbx", "Assimp.Model"},
         {".ply", "Assimp.Model"},
         {".blend", "Assimp.Model"},
     }},
    {ResourceType::Animation,
     {
         {".gltf", "Assimp.Animation"},
         {".glb", "Assimp.Animation"},
         {".dae", "Assimp.Animation"},
         {".fbx", "Assimp.Animation"},
         {".blend", "Assimp.Animation"},
     }},
};

template <ResourceType Type>
std::unique_ptr<Importer<Type>> &Importer<Type>::GetInstance(const std::string &plugin)
{
	static std::unique_ptr<Importer<Type>> importer = std::unique_ptr<Importer<Type>>(PluginManager::GetInstance().Call<Importer<Type> *>(fmt::format("Importer.{}.dll", plugin), "Create"));
	return importer;
}

template <ResourceType Type>
std::unique_ptr<Resource<Type>> Importer<Type>::Import(const std::string &path, RHIContext *rhi_context)
{
	return GetInstance(PluginMap.at(Type).at(Path::GetInstance().GetFileExtension(path)))->Import_(path, rhi_context);
}

template class EXPORT_API Importer<ResourceType::Model>;
template class EXPORT_API Importer<ResourceType::Texture>;
}        // namespace Ilum