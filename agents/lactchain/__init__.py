from __future__ import annotations

'''Registration of all classes'''

from agents.registry.registry import Registry, CoupledRegistry, import_submodules

# registries required for agent factory 
lactchain_agent_registry = CoupledRegistry()
lactchain_prompt_registry = CoupledRegistry()
# lactchain_input_payload_registry = CoupledRegistry()
# lactchain_output_payload_registry = CoupledRegistry()