#pragma once

#include "Precompile.hpp"
#include "Class.hpp"
#include "Enum.hpp"

namespace Ilum
{
struct SchemaModule
{
	std::string name;
	std::vector<std::shared_ptr<Class>> classes;
	std::vector<std::shared_ptr<Enum>> enums;
};
}