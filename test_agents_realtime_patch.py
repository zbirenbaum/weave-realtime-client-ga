"""Test that patching openai_agents also patches realtime websockets."""

from unittest.mock import patch as mock_patch

from weave.integrations.openai_agents.openai_agents import OpenAIAgentsSettings
from weave.trace.autopatch import IntegrationSettings
from weave.integrations import patch as patch_module


def test_settings_defaults():
    s = OpenAIAgentsSettings()
    assert isinstance(s, IntegrationSettings)
    assert s.patch_realtime_websockets is True
    assert s.enabled is True
    print("PASS: default settings")


def test_settings_opt_out():
    s = OpenAIAgentsSettings(patch_realtime_websockets=False)
    assert s.patch_realtime_websockets is False
    assert s.enabled is True
    print("PASS: opt-out setting")


def test_isinstance_check():
    plain = IntegrationSettings()
    assert not isinstance(plain, OpenAIAgentsSettings)
    print("PASS: isinstance distinguishes types")


def test_patch_openai_agents_calls_realtime():
    """patch_openai_agents calls patch_openai_realtime by default."""
    patch_module._PATCHED_INTEGRATIONS.discard("openai_agents")
    patch_module._PATCHED_INTEGRATIONS.discard("openai_realtime")

    called = {"realtime": False, "agents": False}

    def fake_patch_integration(**kwargs):
        if "openai_agents" in kwargs.get("triggering_symbols", []):
            called["agents"] = True
            for name in kwargs["triggering_symbols"]:
                patch_module._PATCHED_INTEGRATIONS.add(name)
        elif "openai_realtime" in kwargs.get("triggering_symbols", []):
            called["realtime"] = True
            for name in kwargs["triggering_symbols"]:
                patch_module._PATCHED_INTEGRATIONS.add(name)

    with mock_patch.object(patch_module, "_patch_integration", side_effect=fake_patch_integration):
        patch_module.patch_openai_agents()

    assert called["agents"], "should call _patch_integration for agents"
    assert called["realtime"], "should also trigger realtime patching"
    print("PASS: patch_openai_agents triggers realtime patching")

    patch_module._PATCHED_INTEGRATIONS.discard("openai_agents")
    patch_module._PATCHED_INTEGRATIONS.discard("openai_realtime")


def test_patch_openai_agents_opt_out_realtime():
    """patch_openai_agents does NOT call realtime when opted out."""
    patch_module._PATCHED_INTEGRATIONS.discard("openai_agents")
    patch_module._PATCHED_INTEGRATIONS.discard("openai_realtime")

    called = {"realtime": False, "agents": False}

    def fake_patch_integration(**kwargs):
        if "openai_agents" in kwargs.get("triggering_symbols", []):
            called["agents"] = True
            for name in kwargs["triggering_symbols"]:
                patch_module._PATCHED_INTEGRATIONS.add(name)
        elif "openai_realtime" in kwargs.get("triggering_symbols", []):
            called["realtime"] = True
            for name in kwargs["triggering_symbols"]:
                patch_module._PATCHED_INTEGRATIONS.add(name)

    settings = OpenAIAgentsSettings(patch_realtime_websockets=False)
    with mock_patch.object(patch_module, "_patch_integration", side_effect=fake_patch_integration):
        patch_module.patch_openai_agents(settings)

    assert called["agents"], "should call _patch_integration for agents"
    assert not called["realtime"], "should NOT trigger realtime when opted out"
    print("PASS: opt-out prevents realtime patching")

    patch_module._PATCHED_INTEGRATIONS.discard("openai_agents")
    patch_module._PATCHED_INTEGRATIONS.discard("openai_realtime")


if __name__ == "__main__":
    test_settings_defaults()
    test_settings_opt_out()
    test_isinstance_check()
    test_patch_openai_agents_calls_realtime()
    test_patch_openai_agents_opt_out_realtime()
    print("\nAll tests passed!")
