#include "Resource.hpp"
#include "Resource/Texture.hpp"
#include "Resource/Model.hpp"

namespace Ilum
{
template class EXPORT_API Resource<ResourceType::Model>;
template class EXPORT_API Resource<ResourceType::Texture>;
}        // namespace Ilum