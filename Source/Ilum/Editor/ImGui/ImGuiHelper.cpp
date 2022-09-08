#include "ImGuiHelper.hpp"

#include <glm/gtc/type_ptr.hpp>

#include <RHI/RHITexture.hpp>

namespace ImGui
{
template <typename T>
inline ImGuiDataType GetDataType()
{
	if (typeid(T) == typeid(float) ||
	    typeid(T) == typeid(glm::vec1) ||
	    typeid(T) == typeid(glm::vec2) ||
	    typeid(T) == typeid(glm::vec3) ||
	    typeid(T) == typeid(glm::vec4))
	{
		return ImGuiDataType_Float;
	}

	if (typeid(T) == typeid(int32_t) ||
	    typeid(T) == typeid(glm::ivec1) ||
	    typeid(T) == typeid(glm::ivec2) ||
	    typeid(T) == typeid(glm::ivec3) ||
	    typeid(T) == typeid(glm::ivec4))
	{
		return ImGuiDataType_S32;
	}

	if (typeid(T) == typeid(uint32_t) ||
	    typeid(T) == typeid(glm::uvec1) ||
	    typeid(T) == typeid(glm::uvec2) ||
	    typeid(T) == typeid(glm::uvec3) ||
	    typeid(T) == typeid(glm::uvec4))
	{
		return ImGuiDataType_U32;
	}

	if (typeid(T) == typeid(size_t) ||
	    typeid(T) == typeid(glm::u64vec1) ||
	    typeid(T) == typeid(glm::u64vec2) ||
	    typeid(T) == typeid(glm::u64vec3) ||
	    typeid(T) == typeid(glm::u64vec4))
	{
		return ImGuiDataType_U64;
	}

	return ImGuiDataType_COUNT;
}

template <typename T>
inline int32_t GetComponentCount()
{
	return T::length();
}

template <typename T>
bool DragScalar(const rttr::variant &var, const rttr::property &prop)
{
	using CmptType = decltype(T::x);

	T v = T{};

	if (GetComponentCount<T>() == 1)
	{
		v = T(prop.get_value(var).convert<CmptType>());
	}
	else
	{
		v = prop.get_value(var).convert<T>();
	}

	CmptType min_ = std::numeric_limits<CmptType>::lowest();
	CmptType max_ = std::numeric_limits<CmptType>::max(); 

	if (prop.get_metadata("min"))
	{
		min_ = prop.get_metadata("min").convert<CmptType>();
	}

	if (prop.get_metadata("max"))
	{
		max_ = prop.get_metadata("max").convert<CmptType>();
	}

	auto data_type = GetDataType<T>();
	bool update    = false;
	switch (data_type)
	{
		case ImGuiDataType_Float:
			if (ImGui::DragScalarN(prop.get_name().data(), data_type, &v.x, GetComponentCount<T>(), 0.01f, &min_, &max_, "%.2f"))
			{
				update = true;
			}
			break;
		case ImGuiDataType_S8:
		case ImGuiDataType_S16:
		case ImGuiDataType_S32:
		case ImGuiDataType_S64:
			if (ImGui::DragScalarN(prop.get_name().data(), data_type, &v.x, GetComponentCount<T>(), 1, &min_, &max_, "%d"))
			{
				update = true;
			}
			break;
		case ImGuiDataType_U8:
		case ImGuiDataType_U16:
		case ImGuiDataType_U32:
		case ImGuiDataType_U64:
			if (ImGui::DragScalarN(prop.get_name().data(), data_type, &v.x, GetComponentCount<T>(), 1, &min_, &max_, "%d"))
			{
				update = true;
			}
			break;
		default:
			break;
	}

	if (update)
	{
		if (GetComponentCount<T>() == 1)
		{
			prop.set_value(var, v.x);
		}
		else
		{
			prop.set_value(var, v);
		}
	}

	return update;
}

template <typename T>
bool SliderScalar(const rttr::variant &var, const rttr::property &prop)
{
	using CmptType = decltype(T::x);

	T v = T{};

	if (GetComponentCount<T>() == 1)
	{
		v = T(prop.get_value(var).convert<CmptType>());
	}
	else
	{
		v = prop.get_value(var).convert<T>();
	}

	CmptType min_ = (CmptType) 0;
	CmptType max_ = (CmptType) 0;

	if (prop.get_metadata("min"))
	{
		min_ = prop.get_metadata("min").convert<CmptType>();
	}

	if (prop.get_metadata("max"))
	{
		max_ = prop.get_metadata("max").convert<CmptType>();
	}

	auto data_type = GetDataType<T>();
	bool update    = false;
	switch (data_type)
	{
		case ImGuiDataType_Float:
			if (ImGui::SliderScalarN(prop.get_name().data(), data_type, &v, GetComponentCount<T>(), &min_, &max_, "%.2f"))
			{
				update = true;
			}
			break;
		case ImGuiDataType_S8:
		case ImGuiDataType_S16:
		case ImGuiDataType_S32:
		case ImGuiDataType_S64:
			if (ImGui::SliderScalarN(prop.get_name().data(), data_type, &v, GetComponentCount<T>(), &min_, &max_, "%d"))
			{
				update = true;
			}
			break;
		case ImGuiDataType_U8:
		case ImGuiDataType_U16:
		case ImGuiDataType_U32:
		case ImGuiDataType_U64:
			if (ImGui::SliderScalarN(prop.get_name().data(), data_type, &v, GetComponentCount<T>(), &min_, &max_, "%d"))
			{
				update = true;
			}
			break;
		default:
			break;
	}

	if (update)
	{
		if (GetComponentCount<T>() == 1)
		{
			prop.set_value(var, v.x);
		}
		else
		{
			prop.set_value(var, v);
		}
	}

	return update;
}

template <typename T>
bool EditScalar(const rttr::variant &var, const rttr::property &prop)
{
	T v = prop.get_value(var).convert<T>();

	if (prop.get_metadata("editor"))
	{
		if (prop.get_metadata("editor").convert<std::string>() == "edit color" &&
		    GetDataType<T>() == ImGuiDataType_Float)
		{
			if (GetComponentCount<T>() == 3)
			{
				if (ImGui::ColorEdit3(prop.get_name().data(), (float *) (&v.x)))
				{
					prop.set_value(var, v);
					return true;
				}
				else
				{
					return false;
				}
			}
			else if (GetComponentCount<T>() == 4)
			{
				if (ImGui::ColorEdit4(prop.get_name().data(), (float *) (&v.x)))
				{
					prop.set_value(var, v);
					return true;
				}
				else
				{
					return false;
				}
			}
		}
		else if (prop.get_metadata("editor").convert<std::string>() == "pick color" &&
		         GetDataType<T>() == ImGuiDataType_Float)
		{
			if (GetComponentCount<T>() == 3)
			{
				if (ImGui::ColorPicker3(prop.get_name().data(), (float *) (&v.x)))
				{
					prop.set_value(var, v);
					return true;
				}
				else
				{
					return false;
				}
			}
			else if (GetComponentCount<T>() == 4)
			{
				if (ImGui::ColorPicker4(prop.get_name().data(), (float *) (&v.x)))
				{
					prop.set_value(var, v);
					return true;
				}
				else
				{
					return false;
				}
			}
		}
		else if (prop.get_metadata("editor").convert<std::string>() == "slider" &&
		         prop.get_metadata("min") &&
		         prop.get_metadata("max"))
		{
			return SliderScalar<T>(var, prop);
		}
		else if (GetComponentCount<T>() == 4)
		{
			if (ImGui::ColorEdit4(prop.get_name().data(), (float *) (&v.x)))
			{
				prop.set_value(var, v);
				return true;
			}
		}
	}

	return DragScalar<T>(var, prop);
}

bool EditString(const rttr::variant &var, const rttr::property &prop)
{
	std::string str     = prop.get_value(var).convert<std::string>();
	char        buf[64] = {0};
	std::memcpy(buf, str.data(), sizeof(buf));
	if (ImGui::InputText(prop.get_name().data(), buf, sizeof(buf)))
	{
		str = buf;
		prop.set_value(var, str);
		return true;
	}
	return false;
}

bool EditEnumeration(const rttr::variant &var, const rttr::property &prop, const rttr::enumeration &enumeration)
{
	std::vector<const char *> enums;
	enums.reserve(enumeration.get_values().size());

	uint64_t prop_enum        = prop.get_value(var).convert<uint64_t>();
	int32_t  current_enum_idx = 0;

	for (auto &enum_ : enumeration.get_values())
	{
		if (prop_enum == enum_.convert<uint64_t>())
		{
			current_enum_idx = static_cast<int32_t>(enums.size());
		}
		enums.push_back(enumeration.value_to_name(enum_).data());
	}

	if (ImGui::Combo(prop.get_name().data(), reinterpret_cast<int32_t *>(&current_enum_idx), enums.data(), static_cast<int32_t>(enums.size())))
	{
		prop.set_value(var, enumeration.name_to_value(enums[current_enum_idx]));
		return true;
	}

	return false;
}

inline static std::unordered_map<rttr::type, std::function<bool(const rttr::variant &, const rttr::property &)>> EditFunctions = {
    {rttr::type::get<float>(), EditScalar<glm::vec1>},
    {rttr::type::get<glm::vec2>(), EditScalar<glm::vec2>},
    {rttr::type::get<glm::vec3>(), EditScalar<glm::vec3>},
    {rttr::type::get<glm::vec4>(), EditScalar<glm::vec4>},
    {rttr::type::get<int32_t>(), EditScalar<glm::ivec1>},
    {rttr::type::get<glm::ivec2>(), EditScalar<glm::ivec2>},
    {rttr::type::get<glm::ivec3>(), EditScalar<glm::ivec3>},
    {rttr::type::get<glm::ivec4>(), EditScalar<glm::ivec4>},
    {rttr::type::get<uint32_t>(), EditScalar<glm::uvec1>},
    {rttr::type::get<glm::uvec2>(), EditScalar<glm::uvec2>},
    {rttr::type::get<glm::uvec3>(), EditScalar<glm::uvec3>},
    {rttr::type::get<glm::uvec4>(), EditScalar<glm::uvec4>},
    {rttr::type::get<size_t>(), EditScalar<glm::u64vec1>},
    {rttr::type::get<glm::u64vec2>(), EditScalar<glm::u64vec2>},
    {rttr::type::get<glm::u64vec3>(), EditScalar<glm::u64vec3>},
    {rttr::type::get<glm::u64vec4>(), EditScalar<glm::u64vec4>},
    {rttr::type::get<glm::u64vec4>(), EditScalar<glm::u64vec4>},
    {rttr::type::get<std::string>(), EditString},
};

void DisplayImage(const rttr::variant& var)
{
	Ilum::RHITexture *texture = var.convert<Ilum::RHITexture*>();
	ImGui::Image(texture, ImVec2{100, 100});
}

bool EditVariant_(const rttr::variant &var)
{
	bool update = false;

	// Handle texture
	if (var.get_type() == rttr::type::get<Ilum::RHITexture*>())
	{
		DisplayImage(var);
	}

	for (auto &property_ : var.get_type().get_properties())
	{
		if (!property_.get_type().get_properties().empty())
		{
			auto prop = property_.get_value(var);
			update |= EditVariant(prop);
			property_.set_value(var, prop);
		}
		else
		{
			// Handle enum
			if (rttr::type::get_by_name(property_.get_type().get_name()).is_enumeration())
			{
				rttr::enumeration enum_ = rttr::type::get_by_name(property_.get_type().get_name()).get_enumeration();
				if (enum_)
				{
					update |= EditEnumeration(var, property_, enum_);
				}
			}
			else
			{
				update |= EditFunctions[property_.get_type()](var, property_);
			}
		}
	}
	return update;
}
}        // namespace ImGui