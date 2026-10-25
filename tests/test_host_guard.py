# Copyright 2026 Marc Sullivan
# SPDX-License-Identifier: Apache-2.0
"""The Host allow-list at the door (dvara._host_refusal, called first in main.guest_gate).

The threat it answers: a web page in the owner's browser, with DNS rebinding, reaches 127.0.0.1 under the attacker's
own name and may send any header -- cf-access-authenticated-user-email: <owner> included. Every header the gate trusts
is forgeable from there; the Host is not, so it is checked first, for tunnel and local requests alike."""
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from smritilila import dvara
from smritilila.main import app

OWNER = "owner@example.com"
FORGED_OWNER = {"cf-ray": "abc-LHR", "cf-access-authenticated-user-email": OWNER}


def _request(host: str | None) -> Request:
    headers = [] if host is None else [(b"host", host.encode())]
    return Request({"type": "http", "method": "GET", "path": "/", "headers": headers, "query_string": b""})


@pytest.fixture(autouse=True)
def fresh_odd_hosts(monkeypatch):
    monkeypatch.setattr(dvara, "_odd_hosts", {})
    monkeypatch.setattr(dvara, "HOST_GUARD", "enforce")
    # the shipped default domain, plus the test client's own names (tests/conftest.py sets the env to those alone)
    monkeypatch.setattr(dvara, "ALLOWED_DOMAINS", dvara._domains("example.com,testserver,test"))


@pytest.mark.parametrize("raw,bare", [
    ("localhost:8198", "localhost"), ("127.0.0.1", "127.0.0.1"), ("127.0.0.1:8200", "127.0.0.1"),
    ("[::1]:8198", "::1"), ("[::1]", "::1"), ("LOCALHOST:8198", "localhost"), ("stonemarch.example.com.", "stonemarch.example.com"),
    ("evil.example:80", "evil.example"), ("  localhost  ", "localhost"), ("", ""), (None, ""), ("[::1", ""),
])
def test_host_of_reads_the_bare_name(raw, bare):
    assert dvara._host_of(_request(raw)) == bare


@pytest.mark.parametrize("host", [
    "localhost", "127.0.0.1", "::1", "example.com", "stonemarch.example.com", "crossing.example.com", "a.b.example.com",
    "testserver", "sub.testserver",  # tests/conftest.py's SMRITILILA_ALLOWED_HOSTS
])
def test_names_this_server_answers_to(host):
    assert dvara._host_allowed(host)


@pytest.mark.parametrize("host", [
    "", "evil.example", "127.0.0.1.evil.example", "localhost.evil.example", "example.com.evil.example", "evilexample.com",
    "example.com.", "example.comx", "notlocalhost", "x.localhost", "0.0.0.0", "192.168.12.123", "::", "[::1]",
])
def test_names_it_does_not(host):
    # "example.com." and "[::1]" are here because _host_of has already stripped the dot and the brackets by the time
    # a name reaches _host_allowed: an unstripped one is not a name this server answers to
    assert not dvara._host_allowed(host)


def test_the_domains_list_ignores_wildcard_dressing_and_blanks():
    assert dvara._domains("*.smritilila.lan, .other.example ,plain.example.,, ") == ("smritilila.lan", "other.example", "plain.example")
    assert dvara._domains("") == ()


def test_only_the_configured_domains_are_allowed(monkeypatch):
    monkeypatch.setattr(dvara, "ALLOWED_DOMAINS", dvara._domains("*.smritilila.lan"))
    assert dvara._host_allowed("smritilila.lan") and dvara._host_allowed("x.smritilila.lan") and dvara._host_allowed("localhost")
    assert not dvara._host_allowed("smritilila.lan.evil.example") and not dvara._host_allowed("testserver")


def test_no_domains_configured_means_this_machine_only(monkeypatch):
    monkeypatch.setattr(dvara, "ALLOWED_DOMAINS", ())
    assert dvara._host_allowed("127.0.0.1") and not dvara._host_allowed("stonemarch.example.com")


def test_the_audit_probe_is_refused():
    """2026-10-18: GET /players with 'Host: evil.example' to 127.0.0.1:8198 answered exactly as with the real host."""
    r = TestClient(app).get("/players", headers={"host": "evil.example"})
    assert r.status_code == 400 and r.json() == {"detail": "host not allowed"}


def test_forged_owner_headers_behind_a_foreign_host_are_refused():
    """The rebinding page's request: the owner's Cloudflare headers, forged, under the attacker's own name."""
    c = TestClient(app)
    for path in ("/health", "/me", "/admin", "/worlds", "/players"):
        r = c.get(path, headers={"host": "attacker.example:8198"} | FORGED_OWNER)
        assert r.status_code == 400, path


def test_local_and_tunnel_hosts_pass_the_door():
    c = TestClient(app)
    assert c.get("/health", headers={"host": "localhost:8198"}).status_code == 200
    assert c.get("/health", headers={"host": "127.0.0.1:8200"}).status_code == 200
    assert c.get("/health", headers={"host": "[::1]:8198"}).status_code == 200
    assert c.get("/health", headers={"host": "stonemarch.example.com"} | FORGED_OWNER).status_code == 200
    assert c.get("/health").status_code == 200  # the test client's own "testserver"


def test_a_missing_host_is_refused():
    # HTTP/1.0 without Host: nothing real sends this to a server that is only reached by name
    assert dvara._host_refusal(_request(None)).status_code == 400


def test_refusal_is_logged_once_per_host(monkeypatch):
    events = []
    monkeypatch.setattr(dvara, "log_event", lambda name, **kw: events.append((name, kw)))
    c = TestClient(app)
    for _ in range(3):
        c.get("/health", headers={"host": "evil.example"})
    c.get("/health", headers={"host": "other.example", "cf-ray": "x"})
    assert [(n, kw["host"], kw["via_cloudflare"]) for n, kw in events] == [
        ("host.refused", "evil.example", False), ("host.refused", "other.example", True)]
    assert dvara._odd_hosts == {"evil.example": 3, "other.example": 1}


def test_the_log_is_bounded_but_the_refusal_is_not(monkeypatch):
    events = []
    monkeypatch.setattr(dvara, "log_event", lambda name, **kw: events.append(name))
    monkeypatch.setattr(dvara, "_ODD_HOSTS_MAX", 2)
    c = TestClient(app)
    codes = [c.get("/health", headers={"host": f"h{i}.example"}).status_code for i in range(5)]
    assert codes == [400] * 5 and len(events) == 2 and len(dvara._odd_hosts) == 2


def test_report_mode_logs_and_lets_it_through(monkeypatch):
    events = []
    monkeypatch.setattr(dvara, "log_event", lambda name, **kw: events.append((name, kw["host"])))
    monkeypatch.setattr(dvara, "HOST_GUARD", "report")
    r = TestClient(app).get("/health", headers={"host": "evil.example"})
    assert r.status_code == 200 and events == [("host.would_refuse", "evil.example")]


def test_off_mode_is_the_old_behaviour(monkeypatch):
    events = []
    monkeypatch.setattr(dvara, "log_event", lambda name, **kw: events.append(name))
    monkeypatch.setattr(dvara, "HOST_GUARD", "off")
    assert TestClient(app).get("/health", headers={"host": "evil.example"}).status_code == 200 and events == []


def test_a_typo_in_the_mode_still_enforces(monkeypatch):
    # anything but "report" or "off" is enforcement: a misspelt setting must not open the door
    monkeypatch.setattr(dvara, "HOST_GUARD", "enfroce")
    assert TestClient(app).get("/health", headers={"host": "evil.example"}).status_code == 400
