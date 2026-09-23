"""Look up backends and agents by name, so neither side imports the other."""

import importlib

BACKENDS = {
    "modal": "hotloop.backends.modal_backend:ModalBackend",
    "local": "hotloop.backends.local_backend:LocalBackend",
}

AGENTS = {
    "openai": "hotloop.agents.openai_agent:OpenAIAgent",
}


def _load(spec: str):
    mod, attr = spec.split(":")
    return getattr(importlib.import_module(mod), attr)


def make_backend(name: str, **kw):
    return _load(BACKENDS[name])(**kw)


def make_agent(name: str, **kw):
    # Also accept a full "module:Class" path for agents that live outside hotloop.
    return _load(AGENTS.get(name, name))(**kw)
