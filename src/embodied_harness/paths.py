"""Central path contract for source, installed, host, and container execution.

The project root is discovered from this module's location in a source checkout,
from the current working directory when an installed package is used inside a
checkout, or from an explicit ``EMBODIED_ARENA_ROOT`` override.  Runtime code
must not assume a checkout is mounted at any legacy parent-workspace path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping


PROJECT_ROOT_ENV = "EMBODIED_ARENA_ROOT"
ARTIFACT_ROOT_ENV = "EMBODIED_ARENA_ARTIFACT_ROOT"
EXTERNAL_ROOT_ENV = "EMBODIED_ARENA_EXTERNAL_ROOT"
CONFIG_ROOT_ENV = "EMBODIED_ARENA_CONFIG_ROOT"
BENCHMARK_ROOT_ENV = "EMBODIED_ARENA_BENCHMARK_ROOT"
CONTAINER_ROOT_ENV = "EMBODIED_ARENA_CONTAINER_ROOT"

_PROJECT_MARKER = "pyproject.toml"
_DEFAULT_CONTAINER_ROOT = PurePosixPath("/workspace/agentic-embodied-arena")


class PathConfigurationError(ValueError):
    """Raised when a configured path is invalid or crosses a protected root."""


def _is_project_root(path: Path) -> bool:
    return (
        (path / _PROJECT_MARKER).is_file()
        and (path / "src" / "embodied_harness").is_dir()
    )


def _search_parents(start: Path) -> Path | None:
    candidate = start.expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if _is_project_root(directory):
            return directory
    return None


def resolve_project_root(
    explicit: str | os.PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    anchor: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the canonical project root without depending on the process cwd.

    ``explicit`` takes precedence over the environment.  An override must be an
    absolute path to a valid checkout, avoiding ambiguous cwd-relative behavior
    in Slurm jobs and installed packages.
    """

    environment = os.environ if environ is None else environ
    override = explicit if explicit is not None else environment.get(PROJECT_ROOT_ENV)
    if override:
        configured = Path(override).expanduser()
        if not configured.is_absolute():
            raise PathConfigurationError(
                f"{PROJECT_ROOT_ENV} must be absolute, got {override!r}"
            )
        configured = configured.resolve()
        if not _is_project_root(configured):
            raise PathConfigurationError(
                f"{PROJECT_ROOT_ENV} is not an Agentic Embodied Arena root: {configured}"
            )
        return configured

    package_anchor = Path(__file__) if anchor is None else Path(anchor)
    discovered = _search_parents(package_anchor)
    if discovered is not None:
        return discovered

    discovered = _search_parents(Path.cwd())
    if discovered is not None:
        return discovered

    raise PathConfigurationError(
        "cannot locate Agentic Embodied Arena; set "
        f"{PROJECT_ROOT_ENV} to an absolute checkout path"
    )


def resolve_under(root: Path, value: str | os.PathLike[str], *, label: str) -> Path:
    """Resolve ``value`` below ``root`` and reject absolute or ``..`` escapes."""

    relative = Path(value).expanduser()
    if relative.is_absolute():
        raise PathConfigurationError(f"{label} must be relative to {root}: {value!r}")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise PathConfigurationError(f"{label} escapes {resolved_root}: {value!r}")
    return resolved


def _host_root(
    environment: Mapping[str, str],
    variable: str,
    project_root: Path,
    default_relative: str,
    *,
    allow_external: bool,
) -> Path:
    value = environment.get(variable)
    if not value:
        return resolve_under(project_root, default_relative, label=variable)

    configured = Path(value).expanduser()
    if configured.is_absolute():
        resolved = configured.resolve()
        if not allow_external and not resolved.is_relative_to(project_root):
            raise PathConfigurationError(
                f"{variable} must remain below {project_root}: {resolved}"
            )
        return resolved
    return resolve_under(project_root, configured, label=variable)


def _container_root(environment: Mapping[str, str]) -> PurePosixPath:
    raw = environment.get(CONTAINER_ROOT_ENV, str(_DEFAULT_CONTAINER_ROOT))
    candidate = PurePosixPath(raw)
    if not candidate.is_absolute():
        raise PathConfigurationError(f"{CONTAINER_ROOT_ENV} must be absolute: {raw!r}")
    if ".." in candidate.parts:
        raise PathConfigurationError(f"{CONTAINER_ROOT_ENV} may not contain '..': {raw!r}")
    return candidate


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Resolved host and container roots used by all harness components."""

    project_root: Path
    artifact_root: Path
    external_root: Path
    config_root: Path
    benchmark_root: Path
    container_root: PurePosixPath

    def in_project(self, relative: str | os.PathLike[str]) -> Path:
        return resolve_under(self.project_root, relative, label="project path")

    def in_container(self, relative: str | os.PathLike[str]) -> PurePosixPath:
        candidate = PurePosixPath(str(relative))
        if candidate.is_absolute() or ".." in candidate.parts:
            raise PathConfigurationError(
                f"container path must remain below {self.container_root}: {relative!r}"
            )
        return self.container_root / candidate

    def external_upstream(self, family: str) -> Path:
        """Return the canonical checkout root for an upstream project."""

        return resolve_under(self.external_root, Path("upstreams") / family, label="upstream family")

    def external_environment(self, family: str) -> Path:
        """Return the canonical isolated runtime environment for a family."""

        return resolve_under(self.external_root, Path("environments") / family, label="environment family")

    def external_assets(self, family: str) -> Path:
        """Return the canonical downloaded asset root for a family."""

        return resolve_under(self.external_root, Path("assets") / family, label="asset family")


# Backward-friendly descriptive alias for callers that prefer the product name.
ArenaPaths = ProjectPaths


def get_project_paths(
    *,
    environ: Mapping[str, str] | None = None,
    anchor: str | os.PathLike[str] | None = None,
) -> ProjectPaths:
    """Resolve the complete path contract for the current process."""

    environment = os.environ if environ is None else environ
    project_root = resolve_project_root(environ=environment, anchor=anchor)
    return ProjectPaths(
        project_root=project_root,
        artifact_root=_host_root(
            environment,
            ARTIFACT_ROOT_ENV,
            project_root,
            "artifacts",
            allow_external=True,
        ),
        external_root=_host_root(
            environment,
            EXTERNAL_ROOT_ENV,
            project_root,
            "external",
            allow_external=True,
        ),
        config_root=_host_root(
            environment,
            CONFIG_ROOT_ENV,
            project_root,
            "configs",
            allow_external=False,
        ),
        benchmark_root=_host_root(
            environment,
            BENCHMARK_ROOT_ENV,
            project_root,
            "benchmarks",
            allow_external=False,
        ),
        container_root=_container_root(environment),
    )


def resolve_paths(
    *,
    environ: Mapping[str, str] | None = None,
    anchor: str | os.PathLike[str] | None = None,
) -> ProjectPaths:
    """Alias retained for concise runtime call sites."""

    return get_project_paths(environ=environ, anchor=anchor)
