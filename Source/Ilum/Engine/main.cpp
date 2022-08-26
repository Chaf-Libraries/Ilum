#include "Engine.hpp"

#include <rttr/registration.h>

#include "Serialization.hpp"

#include <cereal/archives/json.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/vector.hpp>

#include <fstream>

// RTTR_REGISTERATION_BEGIN(float)
// RTTR_REGISTERATION_END()

//#define SERIALIZATION_TYPE(TYPE) \
//	Ilum::Serialization<cereal::JSONInputArchive>::GetInstance().SerialFunctions.emplace(rttr::type::get<TYPE>(), [](cereal::JSONInputArchive &ar, const rttr::variant &var, const rttr::property &prop) { Ilum::serialize<cereal::JSONInputArchive, TYPE>(ar, var, prop); });
//
// namespace NAMESPACE_RTTR_REGISTRATION_float
//{
// RTTR_REGISTRATION
//{
//	rttr::registration::class_<float>("float").constructor<>()(rttr::policy::ctor::as_object);
// }
// }        // namespace NAMESPACE_RTTR_REGISTRATION_float

struct Test
{
	float                      a = 1.f;
	int                        b = 100;
	std::string                c = "fuck";
	std::map<std::string, int> x = {
	    {"a", 11},
	    {"ab", 112},
	    {"abc", 11}};
};

namespace TTTTTTT
{
RTTR_REGISTRATION
{
	rttr::registration::class_<Test>("Test")
	    .constructor<>()(rttr::policy::ctor::as_object)
	    .property("a", &Test::a)
	    .property("b", &Test::b)
	    .property("x", &Test::x)
	    .property("c", &Test::c);



	InputSerializer::RegisterType<decltype(Test::a)>();
	InputSerializer::RegisterType<decltype(Test::b)>();
	InputSerializer::RegisterType<decltype(Test::c)>();
	InputSerializer::RegisterType<decltype(Test::x)>();
}

}

int main()
{


	//rttr::variant var = rttr::type::get<Test>().create();

	// std::ofstream os("test.json", std::ios::binary);

	// cereal::JSONOutputArchive archive(os);
	//  Ilum::serialize<cereal::JSONOutputArchive, float>(cereal::JSONOutputArchive & ar, const rttr::variant &var)



	std::ifstream            is("test.json", std::ios::binary);
	cereal::JSONInputArchive archive(is);
	// archive(var);

	//InputSerializer::GetInstance().serialize(archive, var);

	//auto v = var.convert<Test>();

	Ilum::Engine engine;
	engine.Tick();

	return 0;
}