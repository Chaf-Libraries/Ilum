#pragma once

#include "MaterialNode.hpp"

#include "RHI/ImGuiContext.hpp"

namespace Ilum::MGNode
{
class Operator : public MaterialNode
{
  public:
	Operator(const std::string &name, MaterialGraph *material_graph);

	virtual ~Operator() override;

	virtual void OnImGui(ImGuiContext &context) override;

	virtual void OnImnode() override;

  protected:
	size_t  m_lhs_pin;
	size_t  m_rhs_pin;
	size_t  m_output_pin;
	PinType m_type = PinType::None;
};

class Addition : public Operator
{
  public:
	Addition(MaterialGraph *material_graph);
};

class Subtraction : public Operator
{
  public:
	Subtraction(MaterialGraph *material_graph);
};

class Multiplication : public Operator
{
  public:
	Multiplication(MaterialGraph *material_graph);
};

class Division : public Operator
{
  public:
	Division(MaterialGraph *material_graph);
};

inline static std::map<const char *, std::function<std::unique_ptr<MaterialNode>(MaterialGraph *)>> OperatorNodeCreation = {
    {"Addition", [](MaterialGraph *material_graph) { return std::make_unique<Addition>(material_graph); }},
    {"Subtraction", [](MaterialGraph *material_graph) { return std::make_unique<Subtraction>(material_graph); }},
    {"Multiplication", [](MaterialGraph *material_graph) { return std::make_unique<Multiplication>(material_graph); }},
    {"Division", [](MaterialGraph *material_graph) { return std::make_unique<Division>(material_graph); }}};
}        // namespace Ilum::MGNode