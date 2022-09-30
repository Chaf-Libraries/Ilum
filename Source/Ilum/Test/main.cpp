
#include <iostream>
#include <fstream>

#include "SerializationTest.hpp"

#include <rttr/registration.h>

#	include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
using InputArchive  = cereal::BinaryInputArchive;
using OutputArchive = cereal::BinaryOutputArchive;

RTTR_REGISTRATION
{
	rttr::registration::class_<TestStruct>("TestStruct")
	    .constructor<>()(rttr::policy::ctor::as_object)
	    .property("a", &TestStruct::a)
	    .property("b", &TestStruct::b)

	    ;
}



int main()
{
	TestStruct t = {}; 
	t.a          = 1;
	t.b          = 3;

	{
		std::ofstream os("Test.json");
		OutputArchive archive(os);
		rttr::variant var = t;
		archive(var);
	}

	{
		std::ifstream is("Test.json");
		InputArchive  archive(is);
		rttr::variant var;
		archive(var);
		TestStruct tt = {};
		auto type = var.get_type();
		tt = var.convert<TestStruct>();
	}


	return 0;
}