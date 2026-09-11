import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from crewai_aider_orchestrator.tools.key_manager import SmartFallbackRouter, ARCHITECT_CASCADE, WORKER_CASCADE
from crewai_aider_orchestrator.tools.aider_tool import AiderExecutionTool
from crewai_aider_orchestrator.dynamic_factory import DynamicAgentFactory

def test_smart_router_and_cascades():
    router = SmartFallbackRouter()
    keys = router.get_all_keys()
    print(f"[TEST] Aktif Gemini API Anahtar Sayisi: {len(keys)}")
    assert len(keys) >= 4, f"En az 4 anahtar bekleniyordu, bulunan: {len(keys)}"

    # Architect tier testi
    arch_model, arch_key = router.get_best_model_and_key(tier="architect", preferred_agent_idx=0)
    print(f"[TEST] Architect Model Secimi: {arch_model} (Ilk Tercih: {ARCHITECT_CASCADE[0]})")
    assert arch_model in ARCHITECT_CASCADE

    # Worker tier testi
    worker_model, worker_key = router.get_best_model_and_key(tier="worker", preferred_agent_idx=1)
    print(f"[TEST] Worker Model Secimi: {worker_model} (Ilk Tercih: {WORKER_CASCADE[0]})")
    assert worker_model in WORKER_CASCADE

    print("SUCCESS: SmartFallbackRouter kademeli model ve anahtar testi basariyla gecti!")

def test_dynamic_factory():
    factory = DynamicAgentFactory(workspace_dir="./test_workspace")
    sample_prompt = "SMC Algoritmik Ticaret ve Fraktal Hesaplama Motoru"
    
    # Blueprint ayiklama testi
    sample_blueprint_json = """
    {
        "domain": "Algorithmic Finance",
        "project_name": "SMC Engine",
        "agents": [
            {"role": "Fractal Specialist", "goal": "Swing H/L math", "target_files": "fractals.py", "instruction": "Code fractals"},
            {"role": "FVG Engine Specialist", "goal": "Imbalance math", "target_files": "fvg.py", "instruction": "Code FVG"},
            {"role": "OrderBlock Validator", "goal": "OB detection", "target_files": "ob.py", "instruction": "Code OB"},
            {"role": "QA Math Validator", "goal": "Prevent zero division", "target_files": "tests/test_math.py", "instruction": "Code tests"},
            {"role": "Strategy Docs Specialist", "goal": "Documentation", "target_files": "README.md", "instruction": "Code README"}
        ]
    }
    """
    parsed = factory.parse_blueprint(sample_blueprint_json, sample_prompt)
    assert len(parsed) == 5
    assert parsed[0]["role"] == "Fractal Specialist"
    
    agents, tasks, reviewer, rev_task = factory.build_dynamic_crew_components(parsed, sample_prompt)
    assert len(agents) == 5
    assert len(tasks) == 5
    assert reviewer.role == "Lead Code Reviewer & Systems Integrator"
    
    print("SUCCESS: DynamicAgentFactory 5 uzmanli dinamik ajan uretim testi basariyla gecti!")

if __name__ == "__main__":
    test_smart_router_and_cascades()
    test_dynamic_factory()
