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

TreeShapeVisitor::TreeShapeVisitor(const std::string_view &code) :
    m_code(code)
{
}

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

std::any TreeShapeVisitor::visitChildren(antlr4::tree::ParseTree *node)
{
	std::any    result = defaultResult();
	std::size_t n      = node->children.size();
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

std::any TreeShapeVisitor::visitAttributeSpecifierSeq(CPP14Parser::AttributeSpecifierSeqContext *ctx)
{
	std::vector<Meta::Attribute> attributes;
	for (auto *ctx_attribute_specifier : ctx->attributeSpecifier())
	{
		std::vector<Meta::Attribute> partial_attributes = std::any_cast<std::vector<Meta::Attribute>>(visit(ctx_attribute_specifier));
		for (auto &attribute : partial_attributes)
		{
			attributes.push_back(std::move(attribute));
		}
	}
	return attributes;
}

std::any TreeShapeVisitor::visitAttributeSpecifier(CPP14Parser::AttributeSpecifierContext *ctx)
{
	std::vector<Meta::Attribute> attributes;

	if (ctx->attributeList())
	{
		for (auto *ctx_attribute : ctx->attributeList()->attribute())
		{
			Meta::Attribute attribute = std::any_cast<Meta::Attribute>(visit(ctx_attribute));
			attributes.push_back(std::move(attribute));
		}
	}

	return attributes;
}

std::any TreeShapeVisitor::visitAttribute(CPP14Parser::AttributeContext *ctx)
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

std::any TreeShapeVisitor::visitTypeParameter(CPP14Parser::TypeParameterContext *ctx)
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
			std::vector<Meta::Parameter> tparams = std::any_cast<std::vector<Meta::Parameter>>(visit(ctx->templateparameterList()));

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

std::any TreeShapeVisitor::visitPointerAbstractDeclarator(CPP14Parser::PointerAbstractDeclaratorContext *ctx)
{
	Declarator declarator;
	for (auto *ctx_pointer_operator : ctx->pointerOperator())
	{
		declarator.pointer_operators.push_back(ctx_pointer_operator->getText());
	}
	return declarator;
}

std::any TreeShapeVisitor::visitAbstractPackDeclarator(CPP14Parser::AbstractPackDeclaratorContext *ctx)
{
	Declarator declarator;
	declarator.is_packed = true;
	for (auto *ctx_pointer_operator : ctx->pointerOperator())
	{
		declarator.pointer_operators.push_back(ctx_pointer_operator->getText());
	}
	return declarator;
}

std::any TreeShapeVisitor::visitParameterDeclaration(CPP14Parser::ParameterDeclarationContext *ctx)
{
	Meta::Parameter parameter;

	auto        s     = ctx->declSpecifierSeq()->getText();
	std::size_t start = ctx->declSpecifierSeq()->getStart()->getStartIndex();
	std::size_t stop  = ctx->declSpecifierSeq()->getStop()->getStopIndex();

	parameter.type = {m_code.begin() + start, m_code.begin() + stop + 1};

	if (ctx->declarator())
	{
		Declarator declarator = std::any_cast<Declarator>(visit(ctx->declarator()));
		parameter.is_packed   = declarator.is_packed;
		parameter.name        = declarator.name;
		for (const auto &pointer_operator : declarator.pointer_operators)
		{
			parameter.type += pointer_operator;
		}
	}
	else if (ctx->abstractDeclarator())
	{
		Declarator declarator = std::any_cast<Declarator>(visit(ctx->abstractDeclarator()));
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

std::any TreeShapeVisitor::visitEnumSpecifier(CPP14Parser::EnumSpecifierContext *ctx)
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
		type_meta.attributes = std::any_cast<std::vector<Meta::Attribute>>(visit(ctx->enumHead()->attributeSpecifierSeq()));
	}

	if (ctx->enumeratorList())
	{
		type_meta.fields = std::any_cast<std::vector<Meta::Field>>(visit(ctx->enumeratorList()));
	}

	type_meta.mode = Meta::TypeMeta::Mode::Enum;

	m_type_metas.push_back(std::move(type_meta));

	return {};
}

std::any TreeShapeVisitor::visitEnumeratorList(CPP14Parser::EnumeratorListContext *ctx)
{
	std::vector<Meta::Field> fields;
	for (auto *ctx_enumerator_definition : ctx->enumeratorDefinition())
	{
		Meta::Field field = std::any_cast<Meta::Field>(visit(ctx_enumerator_definition));
		fields.push_back(std::move(field));
	}
	return fields;
}

std::any TreeShapeVisitor::visitEnumeratorDefinition(CPP14Parser::EnumeratorDefinitionContext *ctx)
{
	Meta::Field field;
	field.mode             = Meta::Field::Mode::Value;
	field.access_specifier = Meta::AccessSpecifier::Public;
	field.name             = ctx->enumerator()->getText();
	return field;
}

std::any TreeShapeVisitor::visitTemplateDeclaration(CPP14Parser::TemplateDeclarationContext *ctx)
{
	m_template_parameters = std::any_cast<std::vector<Meta::Parameter>>(visit(ctx->templateparameterList()));

	auto rst = visitDeclaration(ctx->declaration());

	m_template_parameters.clear();

	if (rst.type() == typeid(std::vector<Meta::Field>))
	{
		std::vector<Meta::Field> fields = std::any_cast<std::vector<Meta::Field>>(rst);
		for (auto &field : fields)
		{
			field.is_template = true;
		}
		rst = fields;
	}
	else if (rst.type() == typeid(Meta::Field))
	{
		Meta::Field field = std::any_cast<Meta::Field>(rst);
		field.is_template = true;
		rst               = std::vector<Meta::Field>{field};
	}

	return rst;
}

std::any TreeShapeVisitor::visitTemplateparameterList(CPP14Parser::TemplateparameterListContext *ctx)
{
	std::vector<Meta::Parameter> rst;
	for (auto *ctxTemplateParameter : ctx->templateParameter())
	{
		Meta::Parameter ele = std::any_cast<Meta::Parameter>(visit(ctxTemplateParameter));
		rst.push_back(std::move(ele));
	}
	return rst;
}

std::any TreeShapeVisitor::visitClassSpecifier(CPP14Parser::ClassSpecifierContext *ctx)
{
	if (m_in_template_type)
	{
		return {};
	}

	if (!m_access_specifiers.empty() && m_access_specifiers.back() != Meta::AccessSpecifier::Public)
	{
		return {};
	}

	Meta::TypeMeta type_meta;

	type_meta.namespaces = m_namespaces;
	type_meta.name       = ctx->classHead()->classHeadName()->getText();
	m_namespaces.push_back(type_meta.name);

	if (ctx->classHead()->attributeSpecifierSeq())
	{
		type_meta.attributes = std::any_cast<std::vector<Meta::Attribute>>(visit(ctx->classHead()->attributeSpecifierSeq()));
	}

	type_meta.template_parameters = std::move(m_template_parameters);

	if (!type_meta.template_parameters.empty())
	{
		m_in_template_type = true;
	}

	if (ctx->classHead()->classKey()->getText() == "class")
	{
		m_access_specifiers.push_back(Meta::AccessSpecifier::Private);
		type_meta.mode = Meta::TypeMeta::Mode::Class;
	}
	else
	{
		m_access_specifiers.push_back(Meta::AccessSpecifier::Public);
		type_meta.mode = Meta::TypeMeta::Mode::Struct;
	}

	if (ctx->classHead()->baseClause())
	{
		type_meta.bases = std::any_cast<std::vector<Meta::Base>>(visit(ctx->classHead()->baseClause()));
	}

	if (ctx->memberSpecification())
	{
		for (auto *child : ctx->memberSpecification()->children)
		{
			auto rst = visit(child);
			if (rst.type() == typeid(std::vector<Meta::Field>))
			{
				for (auto &field : std::any_cast<std::vector<Meta::Field>>(rst))
				{
					type_meta.fields.push_back(std::move(field));
				}
			}
		}
	}

	m_type_metas.push_back(std::move(type_meta));

	m_access_specifiers.pop_back();
	m_namespaces.pop_back();
	m_in_template_type = false;

	return {};
}

std::any TreeShapeVisitor::visitBaseClause(CPP14Parser::BaseClauseContext *ctx)
{
	std::vector<Meta::Base> bases;
	for (auto *ctx_base_specifier : ctx->baseSpecifierList()->baseSpecifier())
	{
		Meta::Base base;
		base.name = ctx_base_specifier->baseTypeSpecifier()->getText();
		if (ctx_base_specifier->Virtual())
		{
			base.is_virtual = true;
		}
		if (auto ctx_access_specifier = ctx_base_specifier->accessSpecifier())
		{
			const auto access_specifier = ctx_access_specifier->getText();
			if (access_specifier == "public")
			{
				base.access_specifier = Meta::AccessSpecifier::Public;
			}
			else if (access_specifier == "protected")
			{
				base.access_specifier = Meta::AccessSpecifier::Protected;
			}
			else
			{
				base.access_specifier = Meta::AccessSpecifier::Private;
			}
		}
		bases.push_back(std::move(base));
	}
	return bases;
}

std::any TreeShapeVisitor::visitAccessSpecifier(CPP14Parser::AccessSpecifierContext *ctx)
{
	if (!m_access_specifiers.empty())
	{
		if (ctx->Public())
		{
			m_access_specifiers.back() = Meta::AccessSpecifier::Public;
		}
		else if (ctx->Protected())
		{
			m_access_specifiers.back() = Meta::AccessSpecifier::Protected;
		}
		else
		{
			m_access_specifiers.back() = Meta::AccessSpecifier::Private;
		}
	}

	return {};
}

std::any TreeShapeVisitor::visitNamespaceDefinition(CPP14Parser::NamespaceDefinitionContext *ctx)
{
	const auto origin_namespace_size = m_namespaces.size();

	if (ctx->Identifier())
	{
		m_namespaces.push_back(ctx->Identifier()->getText());
	}
	else if (ctx->originalNamespaceName())
	{
		std::vector<std::string> current_namespaces = std::any_cast<std::vector<std::string>>(visitOriginalNamespaceName(ctx->originalNamespaceName()));
		for (auto &ns : current_namespaces)
		{
			m_namespaces.push_back(std::move(ns));
		}
	}

	visit(ctx->declarationseq());

	while (m_namespaces.size() > origin_namespace_size)
	{
		m_namespaces.pop_back();
	}

	return {};
}

std::any TreeShapeVisitor::visitNestedNameSpecifier(CPP14Parser::NestedNameSpecifierContext *ctx)
{
	if (ctx->nestedNameSpecifier())
	{
		std::vector<std::string> current_namespaces = std::any_cast<std::vector<std::string>>(visitNestedNameSpecifier(ctx->nestedNameSpecifier()));
		current_namespaces.push_back(ctx->Identifier()->getText());
		return current_namespaces;
	}
	else
	{
		return std::vector<std::string>{ctx->theTypeName()->getText()};
	}
	return {};
}

// std::any TreeShapeVisitor::visitQualifiedNamespaceSpecifier(CPP14Parser::QualifiedNamespaceSpecifierContext *ctx)
//{
//	if (ctx->nestedNameSpecifier())
//	{
//		std::vector<std::string> current_namespaces = std::any_cast<std::vector<std::string>>(visitNestedNameSpecifier(ctx->nestedNameSpecifier()));
//		current_namespaces.push_back(ctx->namespaceName()->getText());
//		return current_namespaces;
//	}
//	else
//	{
//		return std::vector<std::string>{ctx->namespaceName()->getText()};
//	}
// }
//
// std::any TreeShapeVisitor::visitProDeclSpecifierSeq(CPP14Parser::ProDeclSpecifierSeqContext *ctx)
//{
//	DeclSpecifierSeq seq;
//	if (auto *pre = ctx->preDeclSpecifierSeq())
//	{
//		for (auto *ctx_decl_specifier : pre->nonSimpleTypeDeclSpecifier())
//		{
//			Meta::DeclSpecifier decl_specifier = GetTextPro(ctx_decl_specifier);
//			seq.decl_specifiers.push_back(decl_specifier);
//		}
//	}
//
//	Meta::DeclSpecifier decl_specifier = GetTextPro(ctx->proSimpleTypeSpecifier());
//	seq.decl_specifiers.push_back(decl_specifier);
//
//	if (auto *post = ctx->postDeclSpecifierSeq())
//	{
//		for (auto *ctx_decl_specifier : post->nonSimpleTypeDeclSpecifier())
//		{
//			Meta::DeclSpecifier decl_specifier = GetTextPro(ctx_decl_specifier);
//			seq.decl_specifiers.push_back(decl_specifier);
//		}
//	}
//	if (ctx->attributeSpecifierSeq())
//	{
//		std::vector<Meta::Attribute> attributes = std::any_cast<std::vector<Meta::Attribute>>(visit(ctx->attributeSpecifierSeq()));
//		for (auto &attribute : attributes)
//		{
//			seq.attributes.push_back(std::move(attribute));
//		}
//	}
//	return seq;
// }
//
// std::any TreeShapeVisitor::visitMemberDeclaration(CPP14Parser::MemberDeclarationContext *ctx)
//{
//	if (ctx->emptyDeclaration() || ctx->aliasDeclaration() || ctx->staticAssertDeclaration() || ctx->usingDeclaration())
//	{
//		return {};
//	}
//	if (ctx->declSpecifierSeq())
//	{
//		return visit(ctx->declSpecifierSeq());
//	}
//	if (ctx->templateDeclaration())
//	{
//		return visit(ctx->templateDeclaration());
//	}
//	if (ctx->functionDefinition())
//	{
//		Meta::Field field = std::any_cast<Meta::Field>(visit(ctx->functionDefinition()));
//		return std::vector<Meta::Field>{field};
//	}
//
//	std::vector<Meta::Attribute>     command_attributes;
//	std::vector<Meta::DeclSpecifier> decl_sepcifiers;
//
//	if (auto *pro = ctx->proDeclSpecifierSeq())
//	{
//		DeclSpecifierSeq seq = std::any_cast<DeclSpecifierSeq>(visit(pro));
//		decl_sepcifiers      = std::move(seq.decl_specifiers);
//		command_attributes   = std::move(seq.attributes);
//	}
//
//	if (ctx->attributeSpecifierSeq())
//	{
//		std::vector<Meta::Attribute> attributes = visit(ctx->attributeSpecifierSeq());
//		for (auto &attribute : attributes)
//		{
//			command_attributes.push_back(std::move(attribute));
//		}
//	}
//
//	std::vector<Meta::Field> fields;
//
//	if (ctx->memberDeclaratorList())
//	{
//		std::vector<Declarator> declarators = std::any_cast<std::vector<Declarator>>(visit(ctx->memberDeclaratorList()));
//		for (auto &declarator : declarators)
//		{
//			Meta::Field field = std::move(declarator).GenerateField();
//			if (!m_access_specifiers.empty())
//			{
//				field.access_specifier = m_access_specifiers.back();
//			}
//			field.decl_specifiers = decl_sepcifiers;
//			if (field.IsStaticConstexprVariable())
//			{
//				field.mode = Meta::Field::Mode::Value;
//			}
//			for (const auto &attribute : command_attributes)
//			{
//				field.attributes.push_back(attribute);
//			}
//			fields.push_back(std::move(field));
//		}
//	}
//
//	return fields;
// }

// std::any TreeShapeVisitor::visitDeclSpecifier(CPP14Parser::DeclSpecifierContext *ctx)
//{
//	return std::any_cast<std::string>(ctx->getText());
// }
//
// std::any TreeShapeVisitor::visitDeclSpecifierSeq(CPP14Parser::DeclSpecifierSeqContext *ctx)
//{
//	DeclSpecifierSeq seq;
//	for (auto decl_specifier : ctx->declSpecifier())
//	{
//		seq.decl_specifiers.push_back(std::any_cast<std::string>(visit(decl_specifier)));
//	}
//
//	if (ctx->attributeSpecifierSeq())
//	{
//		std::vector<Meta::Attribute> attributes = std::any_cast<std::vector<Meta::Attribute>>(visit(ctx->attributeSpecifierSeq()));
//		for (auto &attribute : attributes)
//		{
//			seq.attributes.push_back(std::move(attribute));
//		}
//	}
//	return seq;
// }

// std::any TreeShapeVisitor::visitMemberSpecification(CPP14Parser::MemberSpecificationContext *ctx)
//{
//	return visitChildren(ctx);
// }

// std::any TreeShapeVisitor::visitDeclSpecifier(CPP14Parser::DeclSpecifierContext *ctx)
//{
//	return visitChildren(ctx);
// }
//
// std::any TreeShapeVisitor::visitDeclSpecifierSeq(CPP14Parser::DeclSpecifierSeqContext *ctx)
//{
//	return visitChildren(ctx);
// }

std::any TreeShapeVisitor::visitMemberSpecification(CPP14Parser::MemberSpecificationContext *ctx)
{
	std::vector<Meta::Field> fields;
	for (auto &memberdeclaration : ctx->memberdeclaration())
	{
		auto rst = visit(memberdeclaration);
		if (rst.type() == typeid(std::vector<Meta::Field>))
		{
			for (auto &field : std::any_cast<std::vector<Meta::Field>>(rst))
			{
				fields.push_back(std::move(field));
			}
		}
	}
	return fields;
}

std::any TreeShapeVisitor::visitMemberdeclaration(CPP14Parser::MemberdeclarationContext *ctx)
{
	if (ctx->emptyDeclaration() || ctx->aliasDeclaration() || ctx->staticAssertDeclaration() || ctx->usingDeclaration())
	{
		return {};
	}
	if (ctx->declSpecifierSeq())
	{
		visit(ctx->declSpecifierSeq());
	}
	if (ctx->templateDeclaration())
	{
		return visit(ctx->templateDeclaration());
	}
	if (ctx->functionDefinition())
	{
		Meta::Field field = std::any_cast<Meta::Field>(visit(ctx->functionDefinition()));
		return std::vector<Meta::Field>{field};
	}

	std::vector<Meta::Attribute>     command_attributes;
	std::vector<Meta::DeclSpecifier> decl_sepcifiers;

	if (ctx->attributeSpecifierSeq())
	{
		std::vector<Meta::Attribute> attributes = std::any_cast<std::vector<Meta::Attribute>>(visit(ctx->attributeSpecifierSeq()));
		for (auto &attribute : attributes)
		{
			command_attributes.push_back(std::move(attribute));
		}
	}

	std::vector<Meta::Field> fields;

	if (auto *seq = ctx->declSpecifierSeq())
	{
		for (auto &decl_specifier : seq->declSpecifier())
		{
			std::size_t start = decl_specifier->getStart()->getStartIndex();
			std::size_t stop  = decl_specifier->getStop()->getStopIndex();

			decl_sepcifiers.push_back({m_code.begin() + start, m_code.begin() + stop + 1});
		}
	}

	if (ctx->memberDeclaratorList())
	{
		std::vector<Declarator> declarators = std::any_cast<std::vector<Declarator>>(visit(ctx->memberDeclaratorList()));
		for (auto &declarator : declarators)
		{
			Meta::Field field = std::move(declarator).GenerateField();
			if (!m_access_specifiers.empty())
			{
				field.access_specifier = m_access_specifiers.back();
			}
			field.decl_specifiers = decl_sepcifiers;
			if (field.IsStaticConstexprVariable())
			{
				field.mode = Meta::Field::Mode::Value;
			}
			for (const auto &attribute : command_attributes)
			{
				field.attributes.push_back(attribute);
			}
			fields.push_back(std::move(field));
		}
	}

	return fields;
}

std::any TreeShapeVisitor::visitMemberDeclaratorList(CPP14Parser::MemberDeclaratorListContext *ctx)
{
	std::vector<Declarator> declarators;
	for (auto *ctx_member_declarator : ctx->memberDeclarator())
	{
		Declarator declarator = std::any_cast<Declarator>(visitMemberDeclarator(ctx_member_declarator));
		declarators.push_back(std::move(declarator));
	}
	return declarators;
}

std::any TreeShapeVisitor::visitMemberDeclarator(CPP14Parser::MemberDeclaratorContext *ctx)
{
	Declarator declarator;
	if (ctx->declarator())
	{
		declarator = std::any_cast<Declarator>(visit(ctx->declarator()));
	}
	else if (ctx->Identifier())
	{
		declarator.name = ctx->Identifier()->getText();
		if (ctx->attributeSpecifierSeq())
		{
			declarator.attributes = std::any_cast<std::vector<Meta::Attribute>>(visit(ctx->attributeSpecifierSeq()));
		}
	}
	else
	{
		assert(false);
	}

	if (ctx->braceOrEqualInitializer())
	{
		declarator.initializer = std::any_cast<std::string>(visit(ctx->braceOrEqualInitializer()));
	}

	return declarator;
}

std::any TreeShapeVisitor::visitPointerDeclarator(CPP14Parser::PointerDeclaratorContext *ctx)
{
	Declarator declarator = std::any_cast<Declarator>(visit(ctx->noPointerDeclarator()));
	for (auto *ctx_pointer_opeartor : ctx->pointerOperator())
	{
		declarator.pointer_operators.push_back(ctx_pointer_opeartor->getText());
	}
	return declarator;
}

std::any TreeShapeVisitor::visitNoPointerDeclarator(CPP14Parser::NoPointerDeclaratorContext *ctx)
{
	Declarator declarator = ctx->noPointerDeclarator() ?
	                            std::any_cast<Declarator>(visit(ctx->noPointerDeclarator())) :
                                Declarator{};

	if (ctx->declaratorid())
	{
		declarator.name      = ctx->declaratorid()->idExpression()->getText();
		declarator.is_packed = ctx->declaratorid()->Ellipsis() != nullptr;
	}

	if (ctx->attributeSpecifierSeq())
	{
		std::vector<Meta::Attribute> attributes = std::any_cast<std::vector<Meta::Attribute>>(visit(ctx->attributeSpecifierSeq()));
		for (auto &attribute : attributes)
		{
			declarator.attributes.push_back(std::move(attribute));
		}
	}

	if (ctx->parametersAndQualifiers())
	{
		declarator.mode = Meta::Field::Mode::Function;
		if (ctx->parametersAndQualifiers()->attributeSpecifierSeq())
		{
			std::vector<Meta::Attribute> attributes = std::any_cast<std::vector<Meta::Attribute>>(visit(ctx->parametersAndQualifiers()->attributeSpecifierSeq()));
			for (auto &attribute : attributes)
			{
				declarator.attributes.push_back(std::move(attribute));
			}
		}
		if (auto *ctx_clause = ctx->parametersAndQualifiers()->parameterDeclarationClause())
		{
			for (auto *ctx_parameter_declaration : ctx_clause->parameterDeclarationList()->parameterDeclaration())
			{
				Meta::Parameter parameter = std::any_cast<Meta::Parameter>(visit(ctx_parameter_declaration));
				declarator.parameters.push_back(std::move(parameter));
			}
		}
		if (auto *ctx_cv_seq = ctx->parametersAndQualifiers()->cvqualifierseq())
		{
			for (auto *ctx_cv : ctx_cv_seq->cvQualifier())
			{
				declarator.qualifiers.push_back(ctx_cv->getText());
			}
		}
		if (auto *ctxRef = ctx->parametersAndQualifiers()->refqualifier())
		{
			declarator.qualifiers.push_back(ctxRef->getText());
		}

		if (ctx->parametersAndQualifiers()->exceptionSpecification())
		{
			declarator.qualifiers.push_back(ctx->parametersAndQualifiers()->exceptionSpecification()->getText());
		}
	}
	return declarator;
}

std::any TreeShapeVisitor::visitBraceOrEqualInitializer(CPP14Parser::BraceOrEqualInitializerContext *ctx)
{
	std::string rst;
	if (ctx->initializerClause())
	{
		rst = GetTextPro(ctx->initializerClause());
	}
	else
	{
		rst = GetTextPro(ctx->bracedInitList());
	}
	return rst;
}

std::any TreeShapeVisitor::visitSimpleDeclaration(CPP14Parser::SimpleDeclarationContext *ctx)
{
	if (!ctx->initDeclaratorList())
	{
		return visitChildren(ctx);
	}

	std::vector<Meta::Attribute>     common_attributes;
	std::vector<Meta::DeclSpecifier> decl_specifiers;

	if (ctx->attributeSpecifierSeq())
	{
		std::vector<Meta::Attribute> attributes = std::any_cast<std::vector<Meta::Attribute>>(visit(ctx->attributeSpecifierSeq()));
		for (auto &attribute : attributes)
		{
			common_attributes.push_back(std::move(attribute));
		}
	}

	std::vector<Meta::Field> fields;

	std::vector<Declarator> declarators = std::any_cast<std::vector<Declarator>>(visit(ctx->initDeclaratorList()));
	for (auto &declarator : declarators)
	{
		Meta::Field field     = std::move(declarator).GenerateField();
		field.decl_specifiers = decl_specifiers;
		if (!m_access_specifiers.empty())
		{
			field.access_specifier = m_access_specifiers.back();
		}
		if (field.IsStaticConstexprVariable())
		{
			field.mode = Meta::Field::Mode::Value;
		}
		for (const auto &attribute : common_attributes)
		{
			field.attributes.push_back(attribute);
		}
		fields.push_back(std::move(field));
	}

	return fields;
}

std::any TreeShapeVisitor::visitInitDeclaratorList(CPP14Parser::InitDeclaratorListContext *ctx)
{
	std::vector<Declarator> declarators;
	for (auto *ctx_init_declarator : ctx->initDeclarator())
	{
		Declarator declarator = std::any_cast<Declarator>(visit(ctx_init_declarator));
		declarators.push_back(std::move(declarator));
	}
	return declarators;
}

std::any TreeShapeVisitor::visitInitDeclarator(CPP14Parser::InitDeclaratorContext *ctx)
{
	Declarator declarator = std::any_cast<Declarator>(visit(ctx->declarator()));
	return declarator;
}

std::any TreeShapeVisitor::visitFunctionDefinition(CPP14Parser::FunctionDefinitionContext *ctx)
{
	Declarator  declarator = std::any_cast<Declarator>(visit(ctx->declarator()));
	Meta::Field field      = std::move(declarator).GenerateField();

	if (!m_access_specifiers.empty())
	{
		field.access_specifier = m_access_specifiers.back();
	}

	field.mode = Meta::Field::Mode::Function;

	if (ctx->attributeSpecifierSeq())
	{
		field.attributes = std::any_cast<std::vector<Meta::Attribute>>(visit(ctx->attributeSpecifierSeq()));
	}

	if (ctx->functionBody()->Delete())
	{
		field.initializer = "delete";
	}
	return field;
}
}        // namespace Ilum