import unittest

from mcp_credit_optimizer.server import (
    analyze_prompt,
    get_golden_rules,
    get_strategy_for_task,
)


class BaselineBehaviorTests(unittest.TestCase):
    def test_simple_question_uses_chat_mode(self):
        result = analyze_prompt("O que é uma API?")
        self.assertEqual(result["recommendation"]["strategy"], "CHAT_MODE")
        self.assertFalse(result["analysis"]["needs_agent_action"])

    def test_current_fact_uses_research(self):
        result = analyze_prompt("Qual é o preço atual do serviço em 2026?")
        self.assertEqual(result["recommendation"]["strategy"], "BATCH_RESEARCH")
        self.assertTrue(result["analysis"]["needs_factual_data"])

    def test_code_task_uses_standard(self):
        result = analyze_prompt("Crie um script Python para ler um CSV")
        self.assertEqual(result["recommendation"]["strategy"], "DIRECT_STANDARD")
        self.assertEqual(result["analysis"]["intent"], "code_dev")

    def test_complex_project_decomposes(self):
        result = analyze_prompt("Desenvolva um sistema completo com autenticação e banco de dados")
        self.assertEqual(result["recommendation"]["strategy"], "DECOMPOSE_CASCADE")
        self.assertEqual(result["analysis"]["complexity"], "high")

    def test_unknown_task_type_is_explained(self):
        result = get_strategy_for_task("unknown")
        self.assertIn("error", result)
        self.assertIn("available_types", result)

    def test_empty_prompt_returns_structured_error(self):
        result = analyze_prompt("  \n")
        self.assertEqual(result["error"]["code"], "EMPTY_PROMPT")
        self.assertNotIn("analysis", result)

    def test_capital_does_not_match_api(self):
        result = analyze_prompt("Qual é a capital do Brasil?")
        self.assertEqual(result["analysis"]["intent"], "unknown")
        self.assertEqual(result["recommendation"]["strategy"], "CLARIFY_FIRST")

    def test_negated_action_is_not_execution(self):
        result = analyze_prompt("Não crie um site; explique apenas o que é HTML.")
        self.assertFalse(result["analysis"]["needs_agent_action"])

    def test_automation_is_decomposed(self):
        result = analyze_prompt("Automatize um relatório diário com notificação.")
        self.assertEqual(result["recommendation"]["strategy"], "DECOMPOSE_CASCADE")
        self.assertTrue(result["analysis"]["needs_agent_action"])

    def test_vague_media_uses_refine_first(self):
        result = analyze_prompt("Gere uma imagem moderna.")
        self.assertEqual(result["recommendation"]["strategy"], "REFINE_FIRST")

    def test_task_type_alias_is_normalized(self):
        result = get_strategy_for_task("Q&A")
        self.assertEqual(result["normalized_task_type"], "qa")

    def test_golden_rules_contract(self):
        result = get_golden_rules()
        self.assertEqual(len(result["rules"]), 10)
        self.assertIn("veto_rule", result)


if __name__ == "__main__":
    unittest.main()
