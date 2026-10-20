"""Tests for orchestration guardrails and protocol enforcement.

No LLM. No autonomous loops.
"""

import pytest

from anomaly_lab.evidence.taxonomy import EvidenceClass, EvidenceRecord
from anomaly_lab.orchestration.actions import (
    ActionRegistry,
    ActionResult,
    act,
    register_action,
)
from anomaly_lab.orchestration.guardrails import (
    SKILL_OUTPUT_CANNOT_OVERRIDE_CORE_EVIDENCE,
    GuardrailResult,
    GuardrailViolation,
    check_skill_output,
    guardrail_summarize_no_mutation,
)
from anomaly_lab.orchestration.protocol import (
    PROTOCOL_COMPARE,
    PROTOCOL_DETECT,
    OrchestrationProtocol,
    StepType,
    WorkflowStep,
)

# -----------------------------------------------------------------------
# Protocol
# -----------------------------------------------------------------------


class TestWorkflowStep:
    def test_step_is_frozen(self):
        step = WorkflowStep(
            step_id="test",
            step_type=StepType.SKILL,
            skill_name="inspect_data",
            input_keys=("df",),
            output_key="inspection",
        )
        with pytest.raises(AttributeError):
            step.step_id = "changed"  # type: ignore[misc]

    def test_step_to_dict(self):
        step = WorkflowStep(
            step_id="inspect",
            step_type=StepType.SKILL,
            skill_name="inspect_data",
            input_keys=("df",),
            output_key="inspection",
            guardrails=(SKILL_OUTPUT_CANNOT_OVERRIDE_CORE_EVIDENCE,),
            description="Inspect data",
        )
        d = step.to_dict()
        assert d["step_id"] == "inspect"
        assert d["step_type"] == "SKILL"
        assert d["skill_name"] == "inspect_data"
        assert SKILL_OUTPUT_CANNOT_OVERRIDE_CORE_EVIDENCE in d["guardrails"]

    def test_step_continue_on_failure(self):
        step = WorkflowStep(
            step_id="fragile",
            step_type=StepType.SKILL,
            skill_name="fragile_skill",
            continue_on_failure=True,
        )
        assert step.continue_on_failure is True


class TestOrchestrationProtocol:
    def test_protocol_is_frozen(self):
        assert isinstance(PROTOCOL_DETECT, OrchestrationProtocol)
        # PROTOCOL_DETECT.steps is a tuple (immutable)
        assert isinstance(PROTOCOL_DETECT.steps, tuple)

    def test_protocol_step_ids(self):
        ids = PROTOCOL_DETECT.step_ids()
        assert "inspect" in ids
        assert "plan" in ids
        assert "run" in ids
        assert "summarise" in ids
        assert "export" in ids

    def test_protocol_step_lookup(self):
        step = PROTOCOL_DETECT.step("run")
        assert step is not None
        assert step.skill_name == "run_experiment"

    def test_protocol_unknown_step_returns_none(self):
        assert PROTOCOL_DETECT.step("nonexistent") is None

    def test_protocol_to_dict(self):
        d = PROTOCOL_DETECT.to_dict()
        assert d["name"] == "detect"
        assert d["version"] == "0.5"
        assert len(d["steps"]) == len(PROTOCOL_DETECT.steps)

    def test_compare_protocol_has_validate_comparison(self):
        ids = PROTOCOL_COMPARE.step_ids()
        assert "validate_comparison" in ids

    def test_guardrails_attached_to_run_step(self):
        run_step = PROTOCOL_DETECT.step("run")
        assert SKILL_OUTPUT_CANNOT_OVERRIDE_CORE_EVIDENCE in run_step.guardrails


# -----------------------------------------------------------------------
# Actions
# -----------------------------------------------------------------------


class TestActionResult:
    def test_result_is_frozen(self):
        result = ActionResult(action_name="test", success=True, output={"x": 1})
        with pytest.raises(AttributeError):
            result.success = False  # type: ignore[misc]

    def test_result_to_dict(self):
        result = ActionResult(action_name="test", success=True, output={"x": 1})
        d = result.to_dict()
        assert d["action_name"] == "test"
        assert d["success"] is True
        assert d["output"] == {"x": 1}


class TestActionRegistry:
    def test_register_and_get(self):
        registry = ActionRegistry()

        def my_action(context, **kwargs):
            return ActionResult(action_name="my_action", success=True, output="done")

        registry.register("my_action", my_action)
        assert registry.get("my_action") is my_action
        assert registry.names() == ("my_action",)

    def test_get_unknown_returns_none(self):
        registry = ActionRegistry()
        assert registry.get("does_not_exist") is None

    def test_register_overwrites(self):
        registry = ActionRegistry()

        def a1(context, **kwargs):
            return ActionResult(action_name="x", success=True, output=1)

        def a2(context, **kwargs):
            return ActionResult(action_name="x", success=True, output=2)

        registry.register("x", a1)
        registry.register("x", a2)
        assert registry.get("x") is a2


class TestAct:
    def test_act_unknown_action_returns_failure(self):
        result = act("does_not_exist", {})
        assert result.success is False
        assert "not found" in result.error

    def test_act_calls_registered_action(self):
        registry = ActionRegistry()

        def my_action(context, **kwargs):
            return ActionResult(action_name="mine", success=True, output=kwargs.get("val"))

        registry.register("mine", my_action)
        result = act("mine", {}, val=42, registry=registry)
        assert result.success is True
        assert result.output == 42

    def test_act_exception_returns_failure(self):
        registry = ActionRegistry()

        def bad_action(context, **kwargs):
            raise RuntimeError("oops")

        registry.register("bad", bad_action)
        result = act("bad", {}, registry=registry)
        assert result.success is False
        assert "RuntimeError" in result.error

    def test_act_non_action_result_wrapped(self):
        registry = ActionRegistry()

        def raw_action(context, **kwargs):
            return {"foo": "bar"}  # Returns plain dict, not ActionResult

        registry.register("raw", raw_action)
        result = act("raw", {}, registry=registry)
        assert result.success is True
        assert result.output == {"foo": "bar"}


class TestRegisterActionDecorator:
    def test_decorator_registers_in_global_registry(self):
        @register_action("decorated_action")
        def fn(context, **kwargs):
            return ActionResult(action_name="decorated_action", success=True, output="ok")

        # The function is registered in the global registry
        from anomaly_lab.orchestration.actions import _default_registry

        assert _default_registry.get("decorated_action") is fn

    def test_decorator_returns_original_function(self):
        @register_action("another")
        def my_fn(context, **kwargs):
            return ActionResult(action_name="another", success=True, output=None)

        assert my_fn.__name__ == "my_fn"


# -----------------------------------------------------------------------
# Guardrails
# -----------------------------------------------------------------------


class TestGuardrailResult:
    def test_frozen(self):
        result = GuardrailResult(
            guardrail_name="test",
            passed=True,
            violations=(),
        )
        with pytest.raises(AttributeError):
            result.passed = False  # type: ignore[misc]

    def test_to_dict(self):
        result = GuardrailResult(
            guardrail_name="test",
            passed=False,
            violations=(
                GuardrailViolation(
                    field="evidence_class",
                    original_value="POINT_ANOMALY",
                    attempted_value="FAKE",
                    skill_name="bad_skill",
                ),
            ),
        )
        d = result.to_dict()
        assert d["guardrail_name"] == "test"
        assert d["passed"] is False
        assert len(d["violations"]) == 1
        assert d["violations"][0]["field"] == "evidence_class"


class TestGuardrailViolation:
    def test_frozen(self):
        v = GuardrailViolation(
            field="evidence_class",
            original_value="A",
            attempted_value="B",
            skill_name="s",
        )
        with pytest.raises(AttributeError):
            v.original_value = "C"  # type: ignore[misc]


class TestCheckSkillOutput:
    def test_no_original_evidence_passes(self):
        result = check_skill_output("any", {"evidence_class": "ANY"}, original_evidence=None)
        assert result.passed is True

    def test_dict_with_tampered_evidence_class_fails(self):
        records = [
            EvidenceRecord(
                evidence_class=EvidenceClass.POINT_ANOMALY.value,
                confidence=0.9,
                detector="x",
                score=1.0,
                threshold=0.5,
                supporting_metrics={},
                assumptions=(),
                limitations=(),
                interpretation=".",
            )
        ]
        bad = {"evidence_class": "TAMPERED"}
        result = check_skill_output("bad_skill", bad, original_evidence=records)
        assert result.passed is False
        assert any(v.attempted_value == "TAMPERED" for v in result.violations)

    def test_list_of_dicts_tampering(self):
        records = [
            EvidenceRecord(
                evidence_class=EvidenceClass.POINT_ANOMALY.value,
                confidence=0.9,
                detector="x",
                score=1.0,
                threshold=0.5,
                supporting_metrics={},
                assumptions=(),
                limitations=(),
                interpretation=".",
            )
        ]
        bad_list = [{"evidence_class": "TAMPERED"}]
        result = check_skill_output("bad_skill", bad_list, original_evidence=records)
        assert result.passed is False

    def test_evidence_record_tampering(self):
        records = [
            EvidenceRecord(
                evidence_class=EvidenceClass.POINT_ANOMALY.value,
                confidence=0.9,
                detector="x",
                score=1.0,
                threshold=0.5,
                supporting_metrics={},
                assumptions=(),
                limitations=(),
                interpretation=".",
            )
        ]
        bad_record = EvidenceRecord(
            evidence_class="TAMPERED",
            confidence=0.9,
            detector="x",
            score=1.0,
            threshold=0.5,
            supporting_metrics={},
            assumptions=(),
            limitations=(),
            interpretation=".",
        )
        result = check_skill_output("bad_skill", bad_record, original_evidence=records)
        assert result.passed is False

    def test_none_skill_output_passes(self):
        records = [
            EvidenceRecord(
                evidence_class=EvidenceClass.POINT_ANOMALY.value,
                confidence=0.9,
                detector="x",
                score=1.0,
                threshold=0.5,
                supporting_metrics={},
                assumptions=(),
                limitations=(),
                interpretation=".",
            )
        ]
        result = check_skill_output("some_skill", None, original_evidence=records)
        assert result.passed is True

    def test_skill_output_same_class_passes(self):
        records = [
            EvidenceRecord(
                evidence_class=EvidenceClass.POINT_ANOMALY.value,
                confidence=0.9,
                detector="x",
                score=1.0,
                threshold=0.5,
                supporting_metrics={},
                assumptions=(),
                limitations=(),
                interpretation=".",
            )
        ]
        same_output = {"evidence_class": EvidenceClass.POINT_ANOMALY.value}
        result = check_skill_output("same_skill", same_output, original_evidence=records)
        assert result.passed is True


class TestGuardrailSummarizeNoMutation:
    def test_legitimate_summary_passes(self):
        records = [
            EvidenceRecord(
                evidence_class=EvidenceClass.POINT_ANOMALY.value,
                confidence=0.9,
                detector="x",
                score=1.0,
                threshold=0.5,
                supporting_metrics={},
                assumptions=(),
                limitations=(),
                interpretation=".",
            )
        ]
        legitimate = {
            "total_records": 1,
            "evidence_class_counts": {EvidenceClass.POINT_ANOMALY.value: 1},
            "aggregate_confidence": 0.9,
            "consensus_evidence_class": EvidenceClass.POINT_ANOMALY.value,
            "records_by_class": {
                EvidenceClass.POINT_ANOMALY.value: [
                    {"evidence_class": EvidenceClass.POINT_ANOMALY.value, "confidence": 0.9},
                ],
            },
            "warnings": (),
        }
        result = guardrail_summarize_no_mutation(legitimate, original_evidence=records)
        assert result.passed is True

    def test_wrong_class_in_group_fails(self):
        records = [
            EvidenceRecord(
                evidence_class=EvidenceClass.POINT_ANOMALY.value,
                confidence=0.9,
                detector="x",
                score=1.0,
                threshold=0.5,
                supporting_metrics={},
                assumptions=(),
                limitations=(),
                interpretation=".",
            )
        ]
        bad = {
            "records_by_class": {
                EvidenceClass.POINT_ANOMALY.value: [
                    {"evidence_class": "SOME_OTHER_CLASS", "confidence": 0.9},
                ],
            },
        }
        result = guardrail_summarize_no_mutation(bad, original_evidence=records)
        assert result.passed is False

    def test_non_dict_skill_output_passes(self):
        records = [
            EvidenceRecord(
                evidence_class=EvidenceClass.POINT_ANOMALY.value,
                confidence=0.9,
                detector="x",
                score=1.0,
                threshold=0.5,
                supporting_metrics={},
                assumptions=(),
                limitations=(),
                interpretation=".",
            )
        ]
        result = guardrail_summarize_no_mutation("not_a_dict", original_evidence=records)
        assert result.passed is True


# -----------------------------------------------------------------------
# Guardrail integration: no autonomous loops
# -----------------------------------------------------------------------


class TestNoAutonomousLoops:
    def test_protocol_steps_have_no_recursive_reference(self):
        """Verify that no WorkflowStep lists its own output_key as an input_key.

        A step consuming its own output is the definition of an autonomous loop.
        Merely naming output_key == step_id is fine (a step producing a result
        keyed by its own ID is normal).
        """
        for proto in [PROTOCOL_DETECT, PROTOCOL_COMPARE]:
            for step in proto.steps:
                # A step referencing its own output_key as an input is the forbidden pattern
                if step.output_key is not None and step.output_key in step.input_keys:
                    pytest.fail(
                        f"Step {step.step_id} lists its own output_key "
                        f"{step.output_key!r} in input_keys — would cause autonomous loop"
                    )

    def test_act_does_not_recursively_invoke_itself(self):
        """act() is a single-level dispatcher; it does not call itself."""
        registry = ActionRegistry()

        call_count = [0]

        def counting_action(context, **kwargs):
            call_count[0] += 1
            # This action does not call act() internally
            return ActionResult(action_name="counting", success=True, output=call_count[0])

        registry.register("counting", counting_action)
        result = act("counting", {}, registry=registry)
        assert result.output == 1
        assert call_count[0] == 1

    def test_guardrail_does_not_invoke_act(self):
        """Guardrails are pure check functions — they must not dispatch to act()."""
        import inspect

        guardrail_functions = [
            check_skill_output,
            guardrail_summarize_no_mutation,
        ]
        for fn in guardrail_functions:
            source = inspect.getsource(fn)
            assert "act(" not in source, f"{fn.__name__} must not call act()"


# -----------------------------------------------------------------------
# SKILL_OUTPUT_CANNOT_OVERRIDE_CORE_EVIDENCE — named constant
# -----------------------------------------------------------------------


class TestInvariantConstant:
    def test_invariant_name_matches(self):
        assert (
            SKILL_OUTPUT_CANNOT_OVERRIDE_CORE_EVIDENCE
            == "SKILL_OUTPUT_CANNOT_OVERRIDE_CORE_EVIDENCE"
        )

    def test_invariant_in_protocol_guardrails(self):
        for proto in [PROTOCOL_DETECT, PROTOCOL_COMPARE]:
            for step in proto.steps:
                if step.guardrails:
                    assert SKILL_OUTPUT_CANNOT_OVERRIDE_CORE_EVIDENCE in step.guardrails
