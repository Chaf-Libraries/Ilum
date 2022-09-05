#pragma once

#pragma warning(push, 0)
#include <CPP14ParserBaseVisitor.h>
#pragma warning(pop)

#include "Meta.hpp"

#include <iostream>
#include <string>
#include <string_view>

namespace Ilum
{
class TreeShapeVisitor : public CPP14ParserBaseVisitor
{
  public:
	TreeShapeVisitor(const std::string_view &code);

	const std::vector<Meta::TypeMeta> &GetTypeMeta() const;

  public:
	std::any visitChildren(antlr4::tree::ParseTree *node) override;
	std::any visitAttributeSpecifierSeq(CPP14Parser::AttributeSpecifierSeqContext *ctx) override;
	std::any visitAttributeSpecifier(CPP14Parser::AttributeSpecifierContext *ctx) override;
	std::any visitAttribute(CPP14Parser::AttributeContext *ctx) override;
	std::any visitTypeParameter(CPP14Parser::TypeParameterContext *ctx) override;
	std::any visitPointerAbstractDeclarator(CPP14Parser::PointerAbstractDeclaratorContext *ctx) override;
	std::any visitAbstractPackDeclarator(CPP14Parser::AbstractPackDeclaratorContext *ctx) override;
	std::any visitParameterDeclaration(CPP14Parser::ParameterDeclarationContext *ctx) override;
	std::any visitEnumSpecifier(CPP14Parser::EnumSpecifierContext *ctx) override;
	std::any visitEnumeratorList(CPP14Parser::EnumeratorListContext *ctx) override;
	std::any visitEnumeratorDefinition(CPP14Parser::EnumeratorDefinitionContext *ctx) override;
	std::any visitTemplateDeclaration(CPP14Parser::TemplateDeclarationContext *ctx) override;
	std::any visitTemplateparameterList(CPP14Parser::TemplateparameterListContext *ctx) override;
	std::any visitClassSpecifier(CPP14Parser::ClassSpecifierContext *ctx) override;
	std::any visitBaseClause(CPP14Parser::BaseClauseContext *ctx) override;
	std::any visitAccessSpecifier(CPP14Parser::AccessSpecifierContext *ctx) override;
	std::any visitNamespaceDefinition(CPP14Parser::NamespaceDefinitionContext *ctx) override;
	std::any visitNestedNameSpecifier(CPP14Parser::NestedNameSpecifierContext *ctx) override;
	// std::any visitQualifiedNamespaceSpecifier(CPP14Parser::QualifiedNamespaceSpecifierContext *ctx) override;
	// std::any visitProDeclSpecifierSeq(CPP14Parser::ProDeclSpecifierSeqContext *ctx) override;
	// std::any visitDeclSpecifier(CPP14Parser::DeclSpecifierContext *ctx) override;
	// std::any visitDeclSpecifierSeq(CPP14Parser::DeclSpecifierSeqContext *ctx) override;
	std::any visitMemberSpecification(CPP14Parser::MemberSpecificationContext *ctx) override;
	std::any visitMemberdeclaration(CPP14Parser::MemberdeclarationContext *ctx) override;
	std::any visitMemberDeclaratorList(CPP14Parser::MemberDeclaratorListContext *ctx) override;
	std::any visitMemberDeclarator(CPP14Parser::MemberDeclaratorContext *ctx) override;
	std::any visitPointerDeclarator(CPP14Parser::PointerDeclaratorContext *ctx) override;
	std::any visitNoPointerDeclarator(CPP14Parser::NoPointerDeclaratorContext *ctx) override;
	std::any visitBraceOrEqualInitializer(CPP14Parser::BraceOrEqualInitializerContext *ctx) override;
	std::any visitSimpleDeclaration(CPP14Parser::SimpleDeclarationContext *ctx) override;
	std::any visitInitDeclaratorList(CPP14Parser::InitDeclaratorListContext *ctx) override;
	std::any visitInitDeclarator(CPP14Parser::InitDeclaratorContext *ctx) override;
	std::any visitFunctionDefinition(CPP14Parser::FunctionDefinitionContext *ctx) override;

  private:
	std::string GetTextPro(antlr4::ParserRuleContext *ctx) const;

  private:
	std::string_view                   m_code;
	std::vector<Meta::AccessSpecifier> m_access_specifiers;
	std::vector<Meta::TypeMeta>        m_type_metas;
	std::vector<std::string>           m_namespaces;
	std::vector<Meta::Parameter>       m_template_parameters;
	bool                               m_in_template_type = false;
};

}        // namespace Ilum