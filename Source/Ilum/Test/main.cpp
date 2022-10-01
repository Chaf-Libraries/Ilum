
#include <iostream>
#include <fstream>

#include "SerializationTest.hpp"

#include <rttr/registration.h>

#	include <cereal/archives/binary.hpp>
#	include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>
using InputArchive  = cereal::BinaryInputArchive;
using OutputArchive = cereal::BinaryOutputArchive;

RTTR_REGISTRATION
{
	rttr::registration::class_<TestStruct>("TestStruct")
	    .constructor<>()(rttr::policy::ctor::as_object)
	    .property("a", &TestStruct::a)
	    .property("b", &TestStruct::b)
	    .property("v", &TestStruct::v)
	    .property("m", &TestStruct::m)

	    ;
}

int main()
{
	TestStruct t = {}; 
	t.a          = 1;
	t.b          = 3;
	t.v          = {"1", "2", "3"};
	t.m          = {{"a", 1}, {"b", 2}};

	{
		std::ofstream os("Test.json");
		OutputArchive archive(os);
		archive(t);
	}

	{
		std::ifstream is("Test.json");
		InputArchive  archive(is);
		TestStruct    tt = {};
		archive(tt);
	}


	return 0;
}