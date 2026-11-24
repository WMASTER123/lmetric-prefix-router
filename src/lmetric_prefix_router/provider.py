"""Read-only Extension Manager checks for a process-bound native023 gateway."""

import json
from http.client import HTTPConnection, HTTPException
from importlib.metadata import version
from urllib.parse import urlsplit

BUNDLE_ID = "io.github.wmaster123.lmetric-prefix-router"
DIST = "lmetric-prefix-router"


def local_url(value):
    url = urlsplit(value)
    if (
        url.scheme != "http"
        or url.hostname != "127.0.0.1"
        or not url.port
        or url.username
        or url.password
        or url.path not in ("", "/")
        or url.query
        or url.fragment
    ):
        raise ValueError("expected a local IPv4 HTTP endpoint")
    return url


def get(url, path):
    conn = HTTPConnection(url.hostname, url.port, timeout=2)
    try:
        conn.request("GET", path)
        res = conn.getresponse()
        if res.status != 200:
            raise ValueError(f"{path} returned HTTP {res.status}")
        body = res.read(256 * 1024 + 1)
        if len(body) > 256 * 1024:
            raise ValueError("oversized health response")
        return body
    finally:
        conn.close()


class RouterProvider:
    name = DIST

    def supports(self, manifest):
        return manifest.bundle_id == BUNDLE_ID and manifest.host.provider == self.name

    def plan(self, manifest, configuration, *, enabled):
        from vllm_hust_ext.providers.base import PlanAction, ProviderPlan

        action = PlanAction(
            "check_external_router",
            BUNDLE_ID,
            manifest.lifecycle_owner,
            details={"enabled": enabled},
        )
        return ProviderPlan(
            BUNDLE_ID,
            self.name,
            (action,),
            {"router_url": configuration.get("router_url")},
            ("Operator owns start, traffic switch and drain.",),
        )

    def render(self, plan):
        from vllm_hust_ext.providers.base import RenderArtifact

        return (
            RenderArtifact(
                "router-intent.json",
                "application/json",
                json.dumps(plan.generated_config, indent=2),
            ),
        )

    def check(self, manifest, configuration):
        from vllm_hust_ext.providers.base import ProviderCheck, assess_compatibility

        installed = version(DIST)
        compatible, evidence = assess_compatibility(
            manifest, {}, detected_host_version=installed
        )
        if compatible is not True:
            return ProviderCheck(compatible, False, degraded=True, evidence=evidence)
        try:
            router = local_url(configuration["router_url"])
            backend = local_url(configuration["backend_url"])
            generation = configuration["engine_generation"]
            endpoints = configuration["replay_endpoints"]
            if (
                not isinstance(endpoints, dict)
                or set(endpoints) not in ({"0"}, {"0", "1"})
                or any(
                    not isinstance(url, str) or not url.startswith("tcp://127.0.0.1:")
                    for url in endpoints.values()
                )
            ):
                raise ValueError("expected one configured replay endpoint per rank")
            if (
                not isinstance(generation, str)
                or len(generation) != 64
                or any(c not in "0123456789abcdef" for c in generation)
            ):
                raise ValueError("expected an operator-pinned engine generation digest")
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            return ProviderCheck(True, False, degraded=True, evidence=(str(exc),))
        reachable = False
        try:
            status = json.loads(get(router, "/status"))
            reachable = True
            if (
                not isinstance(status, dict)
                or status.get("extension_id") != BUNDLE_ID
                or status.get("version") != installed
                or status.get("engine_generation") != generation
                or status.get("replay_endpoints") != endpoints
                or status.get("backend") != backend.geturl().rstrip("/")
                or status.get("policy") != "combined"
                or not status.get("ready")
            ):
                raise ValueError(
                    "gateway identity, binding, policy or readiness mismatch"
                )
            get(backend, "/health")
        except (OSError, ValueError, HTTPException) as exc:
            return ProviderCheck(
                True,
                True,
                reachable=reachable,
                healthy=False,
                degraded=True,
                evidence=(str(exc),),
            )
        return ProviderCheck(
            True,
            True,
            reachable=True,
            healthy=True,
            evidence=evidence
            + ("Package, engine generation, rank state and health verified.",),
        )
