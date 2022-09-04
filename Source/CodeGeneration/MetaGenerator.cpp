#include "MetaGenerator.hpp"

namespace Ilum
{
struct Declarator
{
	Meta::Field::Mode            mode = Meta::Field::Mode::Variable;
	std::vector<Meta::Attribute> attributes;
	std::vector<std::string>     pointer_operators;        // *, &, &&
	bool                         is_packed = false;
	std::string                  name;
	std::string                  initializer;
	std::vector<Meta::Parameter> parameters;
	std::vector<std::string>     qualifiers;        // const, volatile, &, &&, exception

	inline Meta::Field GenerateField() &&
	{
		assert(!is_packed);

		Meta::Field field;

		field.mode              = mode;
		field.name              = std::move(name);
		field.attributes        = std::move(attributes);
		field.parameters        = std::move(parameters);
		field.qualifiers        = std::move(qualifiers);
		field.initializer       = std::move(initializer);
		field.pointer_operators = std::move(pointer_operators);

		return field;
	}
};

struct DeclSpecifierSeq
{
	std::vector<Meta::DeclSpecifier> decl_specifiers;
	std::vector<Meta::Attribute>     attributes;
};

const std::vector<Meta::TypeMeta> &TreeShapeVisitor::GetTypeMeta() const
{
	return m_type_metas;
}

std::string TreeShapeVisitor::GetTextPro(antlr4::ParserRuleContext *ctx) const
{
	return {
	    m_code.begin() + ctx->getStart()->getStartIndex(),
	    m_code.begin() + ctx->getStop()->getStopIndex() + 1};
}

antlrcpp::Any TreeShapeVisitor::visitChildren(antlr4::tree::ParseTree *node)
{
	antlrcpp::Any result = defaultResult();
	std::size_t   n      = node->children.size();
	for (std::size_t i = 0; i < n; i++)
	{
		if (!shouldVisitNextChild(node, result))
		{
			break;
		}

		result = std::move(node->children[i]->accept(this));
	}

	return result;
}

antlrcpp::Any TreeShapeVisitor::visitAttributeSpecifierSeq(CPP14Parser::AttributeSpecifierSeqContext *ctx)
{
	std::vector<Meta::Attribute> attributes;
	for (auto *ctx_attribute_specifier : ctx->attributeSpecifier())
	{
		std::vector<Meta::Attribute> partial_attributes = visit(ctx_attribute_specifier);
		for (auto &attribute : partial_attributes)
		{
			attributes.push_back(std::move(attribute));
		}
	}
	return attributes;
}

antlrcpp::Any TreeShapeVisitor::visitAttributeSpecifier(CPP14Parser::AttributeSpecifierContext *ctx)
{
	std::vector<Meta::Attribute> attributes;

	if (ctx->attributeList())
	{
		for (auto *ctx_attribute : ctx->attributeList()->attribute())
		{
			Meta::Attribute attribute = visit(ctx_attribute);
			attributes.push_back(std::move(attribute));
		}
		if (ctx->attributeNamespace())
		{
			const std::string ns = ctx->attributeNamespace()->getText();
			for (auto &attribute : attributes)
			{
				attribute._namespace = ns;
			}
		}
	}

	return attributes;
}

antlrcpp::Any TreeShapeVisitor::visitAttribute(CPP14Parser::AttributeContext *ctx)
{
	Meta::Attribute attribute;
	if (ctx->attributeNamespace())
	{
		attribute._namespace = ctx->attributeNamespace()->getText();
	}
	attribute.name = ctx->Identifier()->getText();
	if (ctx->attributeArgumentClause() && ctx->attributeArgumentClause()->balancedTokenSeq())
	{
		attribute.value = ctx->attributeArgumentClause()->balancedTokenSeq()->getText();
	}
	return attribute;
}

antlrcpp::Any TreeShapeVisitor::visitTypeParameter(CPP14Parser::TypeParameterContext *ctx)
{
	Meta::Parameter parameter;
	if (ctx->Typename_())
	{
		parameter.type = "typename";
	}
	else if (ctx->Class())
	{
		if (ctx->templateparameterList())
		{
			std::vector<Meta::Parameter> tparams = visit(ctx->templateparameterList());

			std::stringstream ss;
			ss << "template<";
			for (std::size_t i = 0; i < tparams.size(); i++)
			{
				ss << tparams[i].GenerateParameterName();
				if (i != tparams.size() - 1)
				{
					ss << ", ";
				}
			}
			ss << "> class";
			parameter.type = ss.str();
		}
		else
		{
			parameter.type = "class";
		}
	}
	else
	{
		assert(false);
	}

	if (ctx->Ellipsis())
	{
		parameter.is_packed = true;
	}
	if (ctx->Identifier())
	{
		parameter.name = ctx->Identifier()->getText();
	}
	return parameter;
}

antlrcpp::Any TreeShapeVisitor::visitPointerAbstractDeclarator(CPP14Parser::PointerAbstractDeclaratorContext *ctx)
{
	Declarator declarator;
	for (auto *ctx_pointer_operator : ctx->pointerOperator())
	{
		declarator.pointer_operators.push_back(ctx_pointer_operator->getText());
	}
	return declarator;
}

antlrcpp::Any TreeShapeVisitor::visitAbstractPackDeclarator(CPP14Parser::AbstractPackDeclaratorContext *ctx)
{
	Declarator declarator;
	declarator.is_packed = true;
	for (auto *ctx_pointer_operator : ctx->pointerOperator())
	{
		declarator.pointer_operators.push_back(ctx_pointer_operator->getText());
	}
	return declarator;
}

antlrcpp::Any TreeShapeVisitor::visitParameterDeclaration(CPP14Parser::ParameterDeclarationContext *ctx)
{
	Meta::Parameter parameter;

	std::size_t start = ctx->proDeclSpecifierSeq()->preDeclSpecifierSeq() ?
	                        ctx->proDeclSpecifierSeq()->preDeclSpecifierSeq()->getStart()->getStartIndex() :
	                        ctx->proDeclSpecifierSeq()->proSimpleTypeSpecifier()->getStart()->getStartIndex();

	std::size_t stop = ctx->proDeclSpecifierSeq()->postDeclSpecifierSeq() ?
	                       ctx->proDeclSpecifierSeq()->postDeclSpecifierSeq()->getStop()->getStopIndex() :
	                       ctx->proDeclSpecifierSeq()->proSimpleTypeSpecifier()->getStop()->getStopIndex();

	parameter.type = {m_code.begin() + start, m_code.begin() + stop + 1};

	if (ctx->declarator())
	{
		Declarator declarator = visit(ctx->declarator());
		parameter.is_packed   = declarator.is_packed;
		parameter.name        = declarator.name;
		for (const auto &pointer_operator : declarator.pointer_operators)
		{
			parameter.type += pointer_operator;
		}
	}
	else if (ctx->abstractDeclarator())
	{
		Declarator declarator = visit(ctx->abstractDeclarator());
		parameter.name        = declarator.name;
		for (const auto &pointer_operator : declarator.pointer_operators)
		{
			parameter.type += pointer_operator;
		}
		parameter.is_packed = declarator.is_packed;
	}

	if (ctx->initializerClause())
	{
		parameter.initializer = GetTextPro(ctx->initializerClause());
	}

	return parameter;
}

antlrcpp::Any TreeShapeVisitor::visitEnumSpecifier(CPP14Parser::EnumSpecifierContext *ctx)
{
	if (m_in_template_type)
	{
		return {};
	}

	if (!ctx->enumHead()->Identifier())
	{
		return {};
	}

	Meta::TypeMeta type_meta;

	type_meta.namespaces = m_namespaces;

	if (ctx->enumHead()->nestedNameSpecifier())
	{
		type_meta.name = GetTextPro(ctx->enumHead()->nestedNameSpecifier());
	}

	type_meta.name += ctx->enumHead()->Identifier()->getText();

	if (ctx->enumHead()->attributeSpecifierSeq())
	{
		type_meta.attributes = visit(ctx->enumHead()->attributeSpecifierSeq()).as<std::vector<Meta::Attribute>>();
	}

	if (ctx->enumeratorList())
	{
		type_meta.fields = visit(ctx->enumeratorList()).as<std::vector<Meta::Field>>();
	}

	m_type_metas.push_back(std::move(type_meta));

	return {};
}

antlrcpp::Any TreeShapeVisitor::visitEnumeratorList(CPP14Parser::EnumeratorListContext *ctx)
{
	std::vector<Meta::Field> fields;
	for (auto *ctx_enumerator_definition : ctx->enumeratorDefinition())
	{
		Meta::Field field = visit(ctx_enumerator_definition);
		fields.push_back(std::move(field));
	}
	return fields;
}

antlrcpp::Any TreeShapeVisitor::visitEnumeratorDefinition(CPP14Parser::EnumeratorDefinitionContext *ctx)
{
	Meta::Field field;
	field.mode            = Meta::Field::Mode::Value;
	field.access_specifier = Meta::AccessSpecifier::Public;
	field.name            = ctx->enumerator()->getText();
	if (ctx->attributeSpecifierSeq())
	{
		field.attributes = visit(ctx->attributeSpecifierSeq()).as<std::vector<Meta::Attribute>>();
	}
	return field;
}

antlrcpp::Any TreeShapeVisitor::visitTemplateDeclaration(CPP14Parser::TemplateDeclarationContext *ctx)
{
	m_template_parameters = visit(ctx->templateparameterList()).as<std::vector<Meta::Parameter>>();

	auto rst = visitDeclaration(ctx->declaration());

	m_template_parameters.clear();

	if (rst.is<std::vector<Meta::Field>>())
	{
		for (auto& field : rst.as<std::vector<Meta::Field>>())
		{
			field.is_template = true;
		}
	}
	else if (rst.is<Meta::Field>())
	{
		Meta::Field field  = rst;
		field.is_template = true;
		rst              = std::vector<Meta::Field>{field};
	}

	return rst;
}

antlrcpp::Any TreeShapeVisitor::visitTemplateparameterList(CPP14Parser::TemplateparameterListContext *ctx)
{}

antlrcpp::Any TreeShapeVisitor::visitClassSpecifier(CPP14Parser::ClassSpecifierContext *ctx)
{}

antlrcpp::Any TreeShapeVisitor::visitBaseClause(CPP14Parser::BaseClauseContext *ctx)
{}

antlrcpp::Any TreeShapeVisitor::visitAccessSpecifier(CPP14Parser::AccessSpecifierContext *ctx)
{}

antlrcpp::Any TreeShapeVisitor::visitNamespaceDefinition(CPP14Parser::NamespaceDefinitionContext *ctx)
{}

antlrcpp::Any TreeShapeVisitor::visitNestedNameSpecifier(CPP14Parser::NestedNameSpecifierContext *ctx)
{}

antlrcpp::Any TreeShapeVisitor::visitQualifiedNamespaceSpecifier(CPP14Parser::QualifiedNamespaceSpecifierContext *ctx)
{}

antlrcpp::Any TreeShapeVisitor::visitProDeclSpecifierSeq(CPP14Parser::ProDeclSpecifierSeqContext *ctx)
{}

antlrcpp::Any TreeShapeVisitor::visitMemberDeclaration(CPP14Parser::MemberDeclarationContext *ctx)
{}

antlrcpp::Any TreeShapeVisitor::visitMemberDeclaratorList(CPP14Parser::MemberDeclaratorListContext *ctx)
{}

antlrcpp::Any TreeShapeVisitor::visitMemberDeclarator(CPP14Parser::MemberDeclaratorContext *ctx)
{}

antlrcpp::Any TreeShapeVisitor::visitPointerDeclarator(CPP14Parser::PointerDeclaratorContext *ctx)
{}

antlrcpp::Any TreeShapeVisitor::visitNoPointerDeclarator(CPP14Parser::NoPointerDeclaratorContext *ctx)
{}

antlrcpp::Any TreeShapeVisitor::visitBraceOrEqualInitializer(CPP14Parser::BraceOrEqualInitializerContext *ctx)
{}

antlrcpp::Any TreeShapeVisitor::visitSimpleDeclaration(CPP14Parser::SimpleDeclarationContext *ctx)
{}

antlrcpp::Any TreeShapeVisitor::visitInitDeclaratorList(CPP14Parser::InitDeclaratorListContext *ctx)
{}

antlrcpp::Any TreeShapeVisitor::visitInitDeclarator(CPP14Parser::InitDeclaratorContext *ctx)
{}

antlrcpp::Any TreeShapeVisitor::visitFunctionDefinition(CPP14Parser::FunctionDefinitionContext *ctx)
{}
}        // namespace Ilum