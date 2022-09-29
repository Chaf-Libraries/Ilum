#pragma once

#include "Generator.hpp"

namespace Ilum
{
class ReflectionGenerator : public Generator
{
  public:
	ReflectionGenerator() = default;

	virtual bool Generate(const std::string &path, const SchemaModule &schema) override;

	virtual void Finish() override;
};
}        // namespace Ilum