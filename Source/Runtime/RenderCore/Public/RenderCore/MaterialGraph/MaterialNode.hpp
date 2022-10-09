#pragma once

#include <Core/Macro.hpp>

namespace Ilum
{
STRUCT(MaterialNode, Enable){
    virtual void Emitter(){

    }};

STRUCT(TestNode, Enable):
	public MaterialNode
{
	STRUCT(Data, Enable)
	{
	} data;

	STRUCT(Input, Enable)
	{
	} input;

	STRUCT(Output, Enable)
	{
	} output;

	void Emitter()
	{
	}
};
}        // namespace Ilum