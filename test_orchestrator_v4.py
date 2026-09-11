import sys
from pathlib import Path

# UTF-8 terminal encoding fix for Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from crewai_aider_orchestrator.spec_expander import SpecificationExpander
from crewai_aider_orchestrator.critic_verifier import RuntimeVerifier
from crewai_aider_orchestrator.dynamic_factory import DynamicAgentFactory
from crewai_aider_orchestrator.tools.key_manager import SmartFallbackRouter


def test_v4_system():
    print("\n=======================================================")
    print("[TEST] [Orchestrator v4.0] Entegrasyon ve Mimari Testleri")
    print("=======================================================\n")

    workspace = str(Path(__file__).parent / "test_workspace")

    # 1. Key Router Test
    router = SmartFallbackRouter()
    arch_model, arch_key = router.get_best_model_and_key(tier="architect")
    worker_model, worker_key = router.get_best_model_and_key(tier="worker")
    assert "gemini" in arch_model
    assert "gemini" in worker_model
    print("1. SmartFallbackRouter Zirhi -> PASS [OK]")

    # 2. Spec Expander Test
    expander = SpecificationExpander(workspace)
    raw_prompt = "Haber gelince 10 pip alan MT5 scalper botu"
    spec = expander.expand_specification(raw_prompt)
    assert "expanded_technical_prompt" in spec
    assert len(spec["mandatory_edge_cases"]) > 0
    print(f"2. SpecificationExpander (Platform: {spec.get('target_platform')}) -> PASS [OK]")

    # 3. Dynamic Factory Context Injection Test
    factory = DynamicAgentFactory(workspace)
    blueprint = factory.parse_blueprint("{}", raw_prompt)
    parallel_agents, parallel_tasks, rev_agent, rev_task = factory.build_dynamic_crew_components(
        agents_blueprint=blueprint,
        user_prompt=raw_prompt,
        arch_spec=spec
    )
    assert len(parallel_agents) == 5
    assert "TEKNİK ŞARTNAME" in parallel_tasks[0].description
    print("3. DynamicAgentFactory (Tam Baglam Enjeksiyonu) -> PASS [OK]")

    # 4. Runtime Verifier Sandbox Test
    verifier = RuntimeVerifier(str(Path(__file__).parent))
    syntax_pass, syntax_msg = verifier.check_syntax()
    assert syntax_pass is True
    print(f"4. RuntimeVerifier (AST/Sozdizimi Taramasi: {syntax_msg}) -> PASS [OK]")

    print("\n=======================================================")
    print("ALL ORCHESTRATOR v4.0 CLOSED-LOOP TESTS PASSED 100%!")
    print("=======================================================\n")


if __name__ == "__main__":
    test_v4_system()
