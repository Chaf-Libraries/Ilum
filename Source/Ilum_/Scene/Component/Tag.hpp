#pragma once

#include "Component.hpp"

#include <string>

namespace Ilum::cmpt
{
class Tag : public Component
{
  public:
	Tag() = default;

	bool OnImGui(ImGuiContext &context) override;

	template <class Archive>
	void serialize(Archive &ar)
	{
		ar(m_name);
	}

	void               SetName(const std::string &name);
	const std::string &GetName() const;

  private:
	std::string m_name = "Untitled Entity";
};
}        // namespace Ilum::cmpt