from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import importlib.util
from pathlib import Path
import shutil
import socket
from typing import Any, Callable
from urllib.parse import urlparse

from .backend import EmbodiedBackend
from .schemas import EpisodeTrace, Observation, PrimitiveCard, PrimitiveResult, TaskSpec, VerificationResult


JsonDict = dict[str, Any]
PolicySkill = Callable[[JsonDict], JsonDict | list[Any]]


@dataclass(frozen=True, slots=True)
class PolicySkillSpec:
    benchmark_id: str
    display_name: str
    primitive_prefix: str
    default_instruction: str
    action_chunk_native: bool
    policy_context: JsonDict = field(default_factory=dict)
    required_modules: tuple[str, ...] = ()
    native_modules: tuple[str, ...] = ()
    command_candidates: tuple[str, ...] = ()


@dataclass(slots=True)
class PolicySkillRuntimeConfig:
    benchmark_id: str = "openvla"
    policy_id: str = "fake-policy"
    model_id: str | None = None
    action_horizon: int = 4
    observation_contract: JsonDict = field(default_factory=dict)
    policy_context: JsonDict = field(default_factory=dict)
    model_path: str | None = None
    service_url: str | None = None
    live: bool = False

    def to_dict(self) -> JsonDict:
        return asdict(self)


POLICY_SKILL_SPECS: dict[str, PolicySkillSpec] = {
    "openvla": PolicySkillSpec(
        benchmark_id="openvla",
        display_name="OpenVLA",
        primitive_prefix="openvla",
        default_instruction="Call OpenVLA as a policy-as-skill over a provided observation/instruction contract.",
        action_chunk_native=False,
        policy_context={"native_runtime": "model_or_serving_wrapper", "output": "single_action_or_chunk"},
        required_modules=("torch", "transformers"),
        native_modules=("openvla", "prismatic"),
    ),
    "openpi": PolicySkillSpec(
        benchmark_id="openpi",
        display_name="OpenPI",
        primitive_prefix="openpi",
        default_instruction="Call OpenPI as a policy-as-skill over a policy server/client action-chunk contract.",
        action_chunk_native=True,
        policy_context={"native_runtime": "policy_server_or_client", "output": "action_chunk"},
        required_modules=("jax", "numpy"),
        native_modules=("openpi", "pi0"),
        command_candidates=("uv",),
    ),
    "lerobot": PolicySkillSpec(
        benchmark_id="lerobot",
        display_name="LeRobot",
        primitive_prefix="lerobot",
        default_instruction="Call a LeRobot dataset or policy bridge as a policy-as-skill over an episode/task contract.",
        action_chunk_native=False,
        policy_context={"native_runtime": "dataset_policy_env_bridge", "output": "policy_action_or_rollout_metrics"},
        required_modules=("torch", "gymnasium"),
        native_modules=("lerobot",),
        command_candidates=("lerobot-eval",),
    ),
    "octo": PolicySkillSpec(
        benchmark_id="octo",
        display_name="Octo",
        primitive_prefix="octo",
        default_instruction="Call Octo as a policy-as-skill over a provided observation/instruction contract.",
        action_chunk_native=False,
        policy_context={"native_runtime": "pretrained_octo_model", "output": "action_prediction"},
        required_modules=("jax", "flax"),
        native_modules=("octo",),
    ),
}


class PolicySkillAgentRuntimeBackend(EmbodiedBackend):
    """Contract-only W5 policy-as-skill runtime for coding agents.

    The backend exposes benchmark-specific primitives that let an agent inspect
    the policy context, probe the local/native runtime surface, validate
    observation fields, call an injected policy skill, submit the returned action
    chunk, and record rollout evidence. It is intentionally not an official
    benchmark success evaluator.
    """

    def __init__(
        self,
        config: PolicySkillRuntimeConfig | None = None,
        policy: PolicySkill | None = None,
        episode: JsonDict | None = None,
    ) -> None:
        self.config = config or PolicySkillRuntimeConfig()
        self._policy = policy or _default_fake_policy
        self._episode = deepcopy(episode or {})
        self._task_spec: TaskSpec | None = None
        self._trace: EpisodeTrace | None = None
        self._last_policy_call: JsonDict | None = None
        self._submitted_action_chunks: list[JsonDict] = []
        self._recorded_evidence: list[JsonDict] = []

    def reset(self, task_id: str, seed: int | None = None, config: JsonDict | None = None) -> TaskSpec:
        runtime_config = self._merged_config(config or {})
        spec = _get_spec(runtime_config.benchmark_id)
        self.config = runtime_config
        self._trace = EpisodeTrace(task_id=task_id)
        self._last_policy_call = None
        self._submitted_action_chunks = []
        self._recorded_evidence = []
        if "observation" not in self._episode:
            self._episode["observation"] = _default_observation()
        if "instruction" not in self._episode:
            self._episode["instruction"] = spec.default_instruction
        self._task_spec = TaskSpec(
            task_id=task_id,
            source=f"w5:policy-as-skill:{spec.benchmark_id}:contract_runtime",
            instruction=str(self._episode.get("instruction") or spec.default_instruction),
            goal={
                "policy_as_skill": True,
                "official_task_success_claimed": False,
                "stop_condition": "agent submits action chunks and records rollout evidence",
            },
            initial_state={"episode": _json_safe(self._episode)},
            budgets={"primitive_calls": 12, "policy_calls": 4, "verifier_calls": 2},
            tags=["w5", "policy-as-skill", spec.benchmark_id, "contract-only"],
            allowed_primitive_levels=["L1", "L2", "L3"],
            metadata={
                "benchmark_id": spec.benchmark_id,
                "display_name": spec.display_name,
                "runtime_config": runtime_config.to_dict(),
                "agent_runtime_contract": {
                    "policy_as_skill": True,
                    "coding_agent_callable": True,
                    "official_task_success_claimed": False,
                    "official_evaluator_called": False,
                    "runtime_probe_available": True,
                    "gpu_required_for_tests": False,
                    "qwen_called": False,
                },
            },
        )
        self.record_event("reset", {"task": self._task_spec.to_dict(), "seed": seed})
        return self._task_spec

    def observe(self) -> Observation:
        self._require_reset()
        spec = _get_spec(self.config.benchmark_id)
        observation = Observation(
            step=len(self.get_trace().events),
            data={
                "benchmark_id": spec.benchmark_id,
                "policy_as_skill": True,
                "instruction": self._episode.get("instruction"),
                "observation_contract": self._observation_contract(),
                "last_policy_call": deepcopy(self._last_policy_call),
                "submitted_action_chunks": deepcopy(self._submitted_action_chunks),
                "recorded_evidence_count": len(self._recorded_evidence),
                "official_task_success_claimed": False,
            },
            metadata={"benchmark_id": spec.benchmark_id},
        )
        self.record_event("observe", observation.to_dict())
        return observation

    def list_primitives(self, level: str | None = None) -> list[PrimitiveCard]:
        self._require_reset()
        prefix = _get_spec(self.config.benchmark_id).primitive_prefix
        cards = [
            self._primitive_card(
                f"get_{prefix}_policy_context",
                "L1",
                {"agent_context": "dict|None"},
                {"policy_context": "dict", "contract": "dict"},
                "Return the coding-agent-visible policy-as-skill context without loading a policy.",
            ),
            self._primitive_card(
                f"probe_{prefix}_policy_runtime",
                "L1",
                {"agent_context": "dict|None", "service_url": "str|None"},
                {"runtime_ready": "bool", "module_specs": "dict", "command_specs": "dict"},
                "Probe native package/command/model/server readiness without loading a heavy policy.",
            ),
            self._primitive_card(
                f"inspect_{prefix}_observation_contract",
                "L1",
                {"observation": "dict|None", "agent_context": "dict|None"},
                {"observation_contract": "dict", "missing_required_fields": "list[str]"},
                "Inspect observation fields, shapes, and required contract keys before policy invocation.",
            ),
            self._primitive_card(
                f"call_{prefix}_policy_skill",
                "L2",
                {"observation": "dict|None", "instruction": "str|None", "horizon": "int|None", "agent_context": "dict|None"},
                {"action_chunk": "list", "policy_call": "dict", "official_task_success_claimed": "bool"},
                "Call an injected policy-as-skill and normalize its output into an auditable action chunk.",
            ),
            self._primitive_card(
                f"score_or_record_{prefix}_rollout_evidence",
                "L1",
                {"evidence": "dict|None", "score": "float|None", "agent_context": "dict|None"},
                {"artifact_id": "str", "official_task_success_claimed": "bool"},
                "Record harness-side rollout evidence or non-official scores without claiming benchmark success.",
            ),
            self._primitive_card(
                f"submit_{prefix}_policy_action_chunk",
                "L3",
                {"action_chunk": "list|None", "target": "str|None", "agent_context": "dict|None"},
                {"submitted": "bool", "action_chunk_id": "str", "official_task_success_claimed": "bool"},
                "Submit a policy action chunk to the downstream bridge contract; no official evaluator is called.",
            ),
        ]
        if level is not None:
            cards = [card for card in cards if card.abstraction_level == level]
        self.record_event("list_primitives", {"level": level, "count": len(cards)})
        return cards

    def call_primitive(self, name: str, **kwargs: Any) -> PrimitiveResult:
        self._require_reset()
        prefix = _get_spec(self.config.benchmark_id).primitive_prefix
        handlers = {
            f"get_{prefix}_policy_context": self._primitive_get_policy_context,
            f"probe_{prefix}_policy_runtime": self._primitive_probe_policy_runtime,
            f"inspect_{prefix}_observation_contract": self._primitive_inspect_observation_contract,
            f"call_{prefix}_policy_skill": self._primitive_call_policy_skill,
            f"score_or_record_{prefix}_rollout_evidence": self._primitive_score_or_record_rollout_evidence,
            f"submit_{prefix}_policy_action_chunk": self._primitive_submit_policy_action_chunk,
        }
        handler = handlers.get(name)
        if handler is None:
            result = PrimitiveResult(name=name, ok=False, error=f"Primitive {name!r} is not exposed by {self.config.benchmark_id}")
        else:
            result = handler(name=name, **kwargs)
        self.record_event("primitive_call", {"name": name, "kwargs": kwargs, "result": result.to_dict()})
        return result

    def verify(self, scope: str = "audit", **kwargs: Any) -> VerificationResult:
        self._require_reset()
        if scope not in {"audit", "policy_skill_contract"}:
            result = VerificationResult(ok=False, scope=scope, message=f"W5 runtime does not expose official task success verification for scope {scope!r}")
        else:
            ok = self._last_policy_call is not None and bool(self._submitted_action_chunks or self._recorded_evidence)
            result = VerificationResult(
                ok=ok,
                scope=scope,
                message=(
                    "Policy-as-skill contract evidence is present; this is not official task success."
                    if ok
                    else "Policy-as-skill contract needs a policy call plus submitted action chunk or recorded evidence."
                ),
                metrics={
                    "policy_calls": 1.0 if self._last_policy_call is not None else 0.0,
                    "submitted_action_chunks": float(len(self._submitted_action_chunks)),
                    "recorded_evidence": float(len(self._recorded_evidence)),
                    "official_task_success": 0.0,
                },
                metadata={"official_task_success_claimed": False, "official_evaluator_called": False},
            )
        if result.ok:
            self.get_trace().final_status = "contract_evidence_recorded"
        self.record_event("verifier_call", result.to_dict())
        return result

    def get_trace(self) -> EpisodeTrace:
        if self._trace is None:
            raise RuntimeError("Call reset() before using the backend.")
        return self._trace

    def _primitive_get_policy_context(self, name: str, agent_context: JsonDict | None = None) -> PrimitiveResult:
        spec = _get_spec(self.config.benchmark_id)
        context = {
            **deepcopy(spec.policy_context),
            **deepcopy(self.config.policy_context),
            "benchmark_id": spec.benchmark_id,
            "policy_id": self.config.policy_id,
            "model_id": self.config.model_id,
            "action_horizon": self.config.action_horizon,
            "policy_as_skill": True,
            "official_task_success_claimed": False,
        }
        return PrimitiveResult(
            name=name,
            ok=True,
            output={
                "policy_context": context,
                "contract": self._contract_flags(),
                "agent_context": deepcopy(agent_context or {}),
            },
        )

    def _primitive_inspect_observation_contract(
        self,
        name: str,
        observation: JsonDict | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        contract = self._observation_contract(observation)
        required = list(self.config.observation_contract.get("required_fields", ["image", "state", "instruction"]))
        missing = [field for field in required if field not in contract["fields"] and field not in self._episode]
        return PrimitiveResult(
            name=name,
            ok=True,
            output={
                "observation_contract": contract,
                "required_fields": required,
                "missing_required_fields": missing,
                "agent_context": deepcopy(agent_context or {}),
                "official_task_success_claimed": False,
            },
        )

    def _primitive_probe_policy_runtime(
        self,
        name: str,
        agent_context: JsonDict | None = None,
        service_url: str | None = None,
    ) -> PrimitiveResult:
        spec = _get_spec(self.config.benchmark_id)
        required_modules = _as_tuple(self.config.policy_context.get("required_modules", spec.required_modules))
        native_modules = _as_tuple(self.config.policy_context.get("native_modules", spec.native_modules))
        command_candidates = _as_tuple(self.config.policy_context.get("command_candidates", spec.command_candidates))
        model_path = self.config.model_path or _first_present_string(
            self.config.policy_context,
            ("model_path", "checkpoint_path", "weights_path"),
        )
        resolved_service_url = service_url or self.config.service_url or _first_present_string(
            self.config.policy_context,
            ("service_url", "server_url", "websocket_url"),
        )
        required_specs = _module_specs(required_modules)
        native_specs = _module_specs(native_modules)
        command_specs = _command_specs(command_candidates)
        model_artifact = _path_status(model_path)
        service_probe = _service_probe(resolved_service_url)
        required_ready = all(item["available"] for item in required_specs.values())
        native_ready = any(item["available"] for item in native_specs.values()) if native_specs else True
        model_ready = True if model_artifact is None else bool(model_artifact["exists"])
        service_ready = True if service_probe is None else bool(service_probe["tcp_connectable"])
        runtime_ready = bool(required_ready and native_ready and model_ready and service_ready)
        output = {
            "benchmark_id": spec.benchmark_id,
            "display_name": spec.display_name,
            "runtime_ready": runtime_ready,
            "required_modules_ready": required_ready,
            "native_runtime_available": native_ready,
            "model_artifact_ready": model_ready,
            "service_ready": service_ready,
            "module_specs": {
                "required": required_specs,
                "native": native_specs,
            },
            "command_specs": command_specs,
            "model_artifact": model_artifact,
            "service_probe": service_probe,
            "agent_context": deepcopy(agent_context or {}),
            "contract": self._contract_flags(),
            "official_task_success_claimed": False,
            "official_evaluator_called": False,
        }
        return PrimitiveResult(name=name, ok=True, output=output)

    def _primitive_call_policy_skill(
        self,
        name: str,
        observation: JsonDict | None = None,
        instruction: str | None = None,
        horizon: int | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        payload = {
            "benchmark_id": self.config.benchmark_id,
            "policy_id": self.config.policy_id,
            "model_id": self.config.model_id,
            "observation": deepcopy(observation if observation is not None else self._episode.get("observation", {})),
            "instruction": instruction if instruction is not None else self._episode.get("instruction"),
            "horizon": int(horizon or self.config.action_horizon),
            "agent_context": deepcopy(agent_context or {}),
            "contract": self._contract_flags(),
        }
        raw = self._policy(deepcopy(payload))
        policy_output = raw if isinstance(raw, dict) else {"action_chunk": raw}
        action_chunk = _normalize_action_chunk(policy_output.get("action_chunk", policy_output.get("action", [])))
        self._last_policy_call = {
            "request": _json_safe(payload),
            "response": _json_safe(policy_output),
            "action_chunk": _json_safe(action_chunk),
            "official_task_success_claimed": False,
        }
        return PrimitiveResult(
            name=name,
            ok=True,
            output={
                "action_chunk": action_chunk,
                "policy_call": deepcopy(self._last_policy_call),
                "contract": self._contract_flags(),
                "official_task_success_claimed": False,
            },
        )

    def _primitive_score_or_record_rollout_evidence(
        self,
        name: str,
        evidence: JsonDict | None = None,
        score: float | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        artifact_id = f"{self.config.benchmark_id}_policy_evidence_{len(self._recorded_evidence)}"
        payload = {
            "evidence": deepcopy(evidence or {}),
            "score": score,
            "agent_context": deepcopy(agent_context or {}),
            "official_task_success_claimed": False,
        }
        self._recorded_evidence.append(payload)
        self.get_trace().add_artifact(artifact_id, _json_safe(payload))
        return PrimitiveResult(name=name, ok=True, output={"artifact_id": artifact_id, **payload})

    def _primitive_submit_policy_action_chunk(
        self,
        name: str,
        action_chunk: list[Any] | None = None,
        target: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        chunk = _normalize_action_chunk(action_chunk if action_chunk is not None else (self._last_policy_call or {}).get("action_chunk", []))
        action_chunk_id = f"{self.config.benchmark_id}_action_chunk_{len(self._submitted_action_chunks)}"
        payload = {
            "action_chunk_id": action_chunk_id,
            "action_chunk": chunk,
            "target": target or "downstream_env_bridge",
            "agent_context": deepcopy(agent_context or {}),
            "submitted": True,
            "official_task_success_claimed": False,
            "official_evaluator_called": False,
        }
        self._submitted_action_chunks.append(payload)
        self.get_trace().add_artifact(action_chunk_id, _json_safe(payload))
        return PrimitiveResult(name=name, ok=True, output=payload)

    def _merged_config(self, override: JsonDict) -> PolicySkillRuntimeConfig:
        data = self.config.to_dict()
        data.update(override)
        return PolicySkillRuntimeConfig(
            benchmark_id=str(data.get("benchmark_id", "openvla")).lower(),
            policy_id=str(data.get("policy_id", "fake-policy")),
            model_id=data.get("model_id"),
            action_horizon=int(data.get("action_horizon", 4)),
            observation_contract=dict(data.get("observation_contract", {})),
            policy_context=dict(data.get("policy_context", {})),
            model_path=data.get("model_path"),
            service_url=data.get("service_url"),
            live=bool(data.get("live", False)),
        )

    def _observation_contract(self, observation: JsonDict | None = None) -> JsonDict:
        obs = deepcopy(observation if observation is not None else self._episode.get("observation", {}))
        fields = {str(key): _field_summary(value) for key, value in obs.items()} if isinstance(obs, dict) else {}
        return {
            "benchmark_id": self.config.benchmark_id,
            "fields": fields,
            "episode_instruction_present": bool(self._episode.get("instruction")),
            "configured_contract": deepcopy(self.config.observation_contract),
            "policy_as_skill": True,
            "official_task_success_claimed": False,
        }

    def _contract_flags(self) -> JsonDict:
        spec = _get_spec(self.config.benchmark_id)
        return {
            "benchmark_id": spec.benchmark_id,
            "policy_as_skill": True,
            "coding_agent_callable": True,
            "action_chunk_native": spec.action_chunk_native,
            "official_task_success_claimed": False,
            "official_evaluator_called": False,
            "runtime_probe_available": True,
            "gpu_required_for_tests": False,
            "qwen_called": False,
        }

    @staticmethod
    def _primitive_card(name: str, level: str, input_schema: JsonDict, output_schema: JsonDict, description: str) -> PrimitiveCard:
        return PrimitiveCard(
            name=name,
            capability_tags=["w5", "policy-as-skill", "coding-agent-runtime"],
            input_schema=input_schema,
            output_schema=output_schema,
            preconditions=["reset_called"],
            side_effects=["trace_event_recorded"],
            failure_modes=["policy_backend_error", "missing_observation_field"],
            abstraction_level=level,
            leakage_risk="low",
            description=description,
        )

    def _require_reset(self) -> None:
        if self._task_spec is None:
            raise RuntimeError("Call reset() before using the W5 policy-as-skill runtime.")


def _get_spec(benchmark_id: str) -> PolicySkillSpec:
    try:
        return POLICY_SKILL_SPECS[benchmark_id]
    except KeyError as exc:
        known = ", ".join(sorted(POLICY_SKILL_SPECS))
        raise ValueError(f"Unknown W5 policy benchmark {benchmark_id!r}; expected one of: {known}") from exc


def _default_fake_policy(payload: JsonDict) -> JsonDict:
    horizon = int(payload.get("horizon", 1))
    return {
        "action_chunk": [[0.0, 0.0, 0.0, 1.0] for _ in range(max(horizon, 1))],
        "metadata": {"source": "default_fake_policy"},
    }


def _default_observation() -> JsonDict:
    return {
        "image": [[[0, 0, 0]]],
        "state": [0.0, 0.0, 0.0],
    }


def _normalize_action_chunk(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return [value]
    if value and not isinstance(value[0], (list, tuple, dict)):
        return [list(value)]
    return [list(item) if isinstance(item, tuple) else item for item in value]


def _field_summary(value: Any) -> JsonDict:
    return {
        "type": type(value).__name__,
        "shape": _shape(value),
        "sample": _json_safe(_sample(value)),
    }


def _module_specs(modules: tuple[str, ...]) -> dict[str, JsonDict]:
    specs: dict[str, JsonDict] = {}
    for module in modules:
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, AttributeError, ValueError):
            spec = None
        specs[module] = {
            "available": spec is not None,
            "origin": getattr(spec, "origin", None) if spec is not None else None,
            "loader": type(getattr(spec, "loader", None)).__name__ if spec is not None else None,
        }
    return specs


def _command_specs(commands: tuple[str, ...]) -> dict[str, JsonDict]:
    specs: dict[str, JsonDict] = {}
    for command in commands:
        resolved = shutil.which(command)
        specs[command] = {"available": resolved is not None, "path": resolved}
    return specs


def _path_status(path: str | None) -> JsonDict | None:
    if not path:
        return None
    candidate = Path(path).expanduser()
    exists = candidate.exists()
    return {
        "path": str(candidate),
        "exists": exists,
        "is_file": candidate.is_file() if exists else False,
        "is_dir": candidate.is_dir() if exists else False,
        "size_bytes": candidate.stat().st_size if exists and candidate.is_file() else None,
    }


def _service_probe(service_url: str | None) -> JsonDict | None:
    if not service_url:
        return None
    parsed = urlparse(service_url)
    host = parsed.hostname
    port = parsed.port or _default_port(parsed.scheme)
    result: JsonDict = {
        "url": service_url,
        "scheme": parsed.scheme,
        "host": host,
        "port": port,
        "tcp_connectable": False,
        "error": None,
    }
    if not host or port is None:
        result["error"] = "missing_host_or_port"
        return result
    try:
        with socket.create_connection((host, int(port)), timeout=0.5):
            result["tcp_connectable"] = True
    except OSError as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _default_port(scheme: str) -> int | None:
    if scheme in {"http", "ws"}:
        return 80
    if scheme in {"https", "wss"}:
        return 443
    return None


def _first_present_string(data: JsonDict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item))
    return ()


def _shape(value: Any) -> list[int]:
    if hasattr(value, "shape"):
        try:
            return [int(dim) for dim in value.shape]
        except TypeError:
            pass
    if isinstance(value, (list, tuple)):
        if not value:
            return [0]
        return [len(value), *_shape(value[0])]
    return []


def _sample(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): _sample(val) for key, val in list(value.items())[:3]}
    if isinstance(value, (list, tuple)):
        return [_sample(item) for item in list(value)[:2]]
    return value


def _json_safe(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
