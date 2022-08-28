#include "ImGuiHelper.hpp"

#include <glm/gtc/type_ptr.hpp>

namespace ImGui
{
template <typename T>
inline ImGuiDataType GetDataType()
{
	if (typeid(T) == typeid(float) ||
	    typeid(T) == typeid(glm::vec2) ||
	    typeid(T) == typeid(glm::vec3) ||
	    typeid(T) == typeid(glm::vec4))
	{
		return ImGuiDataType_Float;
	}

	if (typeid(T) == typeid(int32_t) ||
	    typeid(T) == typeid(glm::ivec2) ||
	    typeid(T) == typeid(glm::ivec3) ||
	    typeid(T) == typeid(glm::ivec4))
	{
		return ImGuiDataType_S32;
	}

	if (typeid(T) == typeid(uint32_t) ||
	    typeid(T) == typeid(glm::uvec2) ||
	    typeid(T) == typeid(glm::uvec3) ||
	    typeid(T) == typeid(glm::uvec4))
	{
		return ImGuiDataType_S32;
	}

	return ImGuiDataType_COUNT;
}

template <typename T>
inline int32_t GetComponentCount()
{
	if (typeid(T) == typeid(float) ||
	    typeid(T) == typeid(int32_t) ||
	    typeid(T) == typeid(uint32_t))
	{
		return 1;
	}

	if (typeid(T) == typeid(glm::vec2) ||
	    typeid(T) == typeid(glm::ivec2) ||
	    typeid(T) == typeid(glm::uvec2))
	{
		return 2;
	}

	if (typeid(T) == typeid(glm::vec3) ||
	    typeid(T) == typeid(glm::ivec3) ||
	    typeid(T) == typeid(glm::uvec3))
	{
		return 3;
	}

	if (typeid(T) == typeid(glm::vec4) ||
	    typeid(T) == typeid(glm::ivec4) ||
	    typeid(T) == typeid(glm::uvec4))
	{
		return 4;
	}

	return 0;
}

template <typename T>
bool EditScalar(const rttr::variant &var, const rttr::property &prop)
{
	T v = prop.get_value(var).convert<T>();
	if (ImGui::DragScalarN(prop.get_name().data(), GetDataType<T>(), &v, GetComponentCount<T>(), 0.01f, nullptr, nullptr, "%.2f"))
	{
		prop.set_value(var, v);
		return true;
	}
	return false;
}

inline static std::unordered_map<rttr::type, std::function<bool(const rttr::variant &, const rttr::property &)>> EditFunctions = {
    {rttr::type::get<float>(), EditScalar<float>},
    {rttr::type::get<glm::vec2>(), EditScalar<glm::vec2>},
    {rttr::type::get<glm::vec3>(), EditScalar<glm::vec3>},
    {rttr::type::get<glm::vec4>(), EditScalar<glm::vec4>},
    {rttr::type::get<int32_t>(), EditScalar<int32_t>},
    {rttr::type::get<glm::ivec2>(), EditScalar<glm::ivec2>},
    {rttr::type::get<glm::ivec3>(), EditScalar<glm::ivec3>},
    {rttr::type::get<glm::ivec4>(), EditScalar<glm::ivec4>},
    {rttr::type::get<uint32_t>(), EditScalar<uint32_t>},
    {rttr::type::get<glm::uvec2>(), EditScalar<glm::uvec2>},
    {rttr::type::get<glm::uvec3>(), EditScalar<glm::uvec3>},
    {rttr::type::get<glm::uvec4>(), EditScalar<glm::uvec4>},
};

bool EditVariant(const rttr::variant &var)
{
	bool update = false;

	for (auto &property_ : var.get_type().get_properties())
	{
		if (!property_.get_type().get_properties().empty())
		{
			update |= EditVariant(property_.get_value(var));
		}
		else
		{
			//update |= EditFunctions[property_.get_type()](var, property_);
		}
	}
	return update;
}
}        // namespace ImGui