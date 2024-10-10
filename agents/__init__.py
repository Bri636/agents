from __future__ import annotations

'''Registration of all classes'''

from agents.registry.registry import Registry, CoupledRegistry, import_submodules

# registries required for agent factory 
action_agent_registry = CoupledRegistry()
prompt_registry = CoupledRegistry()
input_payload_registry = CoupledRegistry()
output_payload_registry = CoupledRegistry()
