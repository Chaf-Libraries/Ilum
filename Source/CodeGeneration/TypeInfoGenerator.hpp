#pragma once

#include "Meta.hpp"

namespace Ilum
{
std::string GenerateTypeInfo(const std::vector<std::string>& headers, const std::vector<Meta::TypeMeta> &meta_types);
}