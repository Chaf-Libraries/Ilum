#include "Importer.hpp"

#include <Core/Plugin.hpp>

namespace Ilum
{
template <ResourceType Type>
std::unique_ptr<Importer<Type>> &Importer<Type>::GetInstance(const std::string &plugin)
{
	static std::unique_ptr<Importer<Type>> importer = std::unique_ptr<Importer<Type>>(PluginManager::GetInstance().Call<Importer<Type>*>(fmt::format("Importer.{}.dll", plugin), "Create"));
	return importer;
}

template class EXPORT_API Importer<ResourceType::Model>;
template class EXPORT_API Importer<ResourceType::Texture>;
}        // namespace Ilum