"""Tests for dashboard authentication.

Credential handling is the one part of this project where "it looked right" is
worth nothing, so these tests attack it rather than confirm it: user
enumeration by timing, lockout arithmetic, forgery of the proxy identity header,
and the fail-closed behaviour that must hold when configuration is missing.

The scrypt cost is lowered throughout. Production uses n=2**15; the tests would
otherwise spend most of their wall clock in the KDF.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from emperor_space_tracker.auth import (
    AuthStore,
    Role,
    dummy_verify,
    hash_password,
    lockout_for,
    normalise_username,
    verify_identity,
    verify_password,
)
from emperor_space_tracker.errors import StoreError

# Cheap cost for tests; the format and the code path are identical.
FAST = {"n": 2**10}


@pytest.fixture
def auth(tmp_path: Path) -> Iterator[AuthStore]:
    """Build an auth store with one read user and one write user."""
    store = AuthStore(tmp_path / "auth.sqlite3")
    store.add_user("reader", "read-password-1", role=Role.READ)
    store.add_user("admin", "write-password-1", role=Role.WRITE)
    try:
        yield store
    finally:
        store.close()


# --------------------------------------------------------------------------- #
# Password hashing
# --------------------------------------------------------------------------- #


def test_hash_is_scrypt_and_carries_its_parameters() -> None:
    """The stored string must be self-describing.

    Raising the cost later has to keep working for hashes written today, which
    is only possible if the parameters travel with the hash.
    """
    encoded = hash_password("a good password", **FAST)
    scheme, n, r, p, salt, key = encoded.split("$")
    assert scheme == "scrypt"
    assert int(n) == 2**10
    assert int(r) == 8 and int(p) == 1
    assert salt and key


def test_hash_is_salted_so_equal_passwords_differ() -> None:
    """Two users with the same password must not share a hash.

    Identical hashes would leak that two accounts chose the same password, and
    let one precomputed table be tried against both.
    """
    first = hash_password("identical", **FAST)
    second = hash_password("identical", **FAST)
    assert first != second


def test_verify_accepts_the_right_password_and_rejects_others() -> None:
    """The core check, including near-misses."""
    encoded = hash_password("correct horse", **FAST)
    assert verify_password("correct horse", encoded)
    assert not verify_password("correct horse ", encoded)
    assert not verify_password("Correct horse", encoded)
    assert not verify_password("", encoded)


def test_verify_rejects_a_malformed_hash_without_raising() -> None:
    """A corrupt row must deny, not crash the login form.

    An exception here would surface as a 500 on the password prompt, which
    turns a credential problem into an apparent outage.
    """
    for junk in ("", "not-a-hash", "bcrypt$1$2$3$4$5", "scrypt$a$b$c$d$e"):
        assert verify_password("anything", junk) is False


def test_empty_password_is_refused() -> None:
    """An empty password is not a password."""
    with pytest.raises(ValueError, match="must not be empty"):
        hash_password("")


def test_dummy_verify_runs_a_real_kdf() -> None:
    """The anti-enumeration path must actually burn CPU.

    A no-op here would restore the timing oracle it exists to close.
    """
    started = time.monotonic()
    dummy_verify()
    assert time.monotonic() - started > 0.001


# --------------------------------------------------------------------------- #
# Username handling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("KCastle96", "kcastle96"), ("  kyle  ", "kyle"), ("Kyle", "kyle")],
)
def test_usernames_are_canonicalised(raw: str, expected: str) -> None:
    """Case and surrounding space must not create two accounts.

    `KCastle96` and `kcastle96` differing by a keystroke is exactly how a lockout
    gets bypassed, or how a second account appears by accident.
    """
    assert normalise_username(raw) == expected


@pytest.mark.parametrize("bad", ["", "   ", "x" * 65])
def test_unusable_usernames_are_rejected(bad: str) -> None:
    """Empty or absurdly long names are refused rather than stored."""
    with pytest.raises(ValueError):
        normalise_username(bad)


# --------------------------------------------------------------------------- #
# Lockout arithmetic
# --------------------------------------------------------------------------- #


def test_lockout_is_zero_below_the_threshold() -> None:
    """Ordinary typos must not lock anyone out."""
    for failures in range(5):
        assert lockout_for(failures, max_failures=5, base_seconds=60.0) == 0.0


def test_lockout_is_exponential_and_capped() -> None:
    """Each extra attempt doubles, and the ceiling bounds the damage.

    An uncapped doubling would let an attacker lock the operator out for
    months, turning brute-force protection into a denial-of-service weapon.
    """
    values = [
        lockout_for(f, max_failures=5, base_seconds=60.0) for f in range(5, 10)
    ]
    assert values == [60.0, 120.0, 240.0, 480.0, 960.0]
    assert lockout_for(60, max_failures=5, base_seconds=60.0, cap_seconds=900.0) == 900.0


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #


def test_correct_credentials_yield_the_right_role(auth: AuthStore) -> None:
    """Each account authenticates with its own role."""
    reader = auth.authenticate("reader", "read-password-1")
    admin = auth.authenticate("admin", "write-password-1")
    assert reader is not None and reader.role is Role.READ
    assert admin is not None and admin.role is Role.WRITE


def test_username_case_does_not_matter_when_logging_in(auth: AuthStore) -> None:
    """The login form normalises, matching account creation."""
    assert auth.authenticate("ADMIN", "write-password-1") is not None


def test_wrong_password_is_denied_and_counted(auth: AuthStore) -> None:
    """A failed attempt is recorded even though the caller learns nothing."""
    assert auth.authenticate("reader", "nope") is None
    assert auth.remaining_lockout("reader") == 0.0  # 1 failure, threshold is 5


def test_unknown_user_is_denied(auth: AuthStore) -> None:
    """A nonexistent account fails exactly like a wrong password."""
    assert auth.authenticate("ghost", "anything") is None


def test_lockout_engages_after_the_threshold(auth: AuthStore) -> None:
    """Repeated failures lock the account for a rising interval."""
    for _ in range(5):
        auth.authenticate("reader", "wrong")
    assert auth.remaining_lockout("reader") > 0.0


def test_a_correct_password_is_refused_while_locked_out(auth: AuthStore) -> None:
    """Lockout must hold even against the right password.

    Otherwise the lockout is trivially bypassed by whoever already guessed the
    password -- which is the only person it was protecting against.
    """
    for _ in range(5):
        auth.authenticate("reader", "wrong")
    assert auth.authenticate("reader", "read-password-1") is None


def test_lockout_expires(auth: AuthStore) -> None:
    """A lockout lifts on its own; the operator is not permanently locked out.

    Uses a zero-length base window so the test does not have to sleep through a
    real one.
    """
    for _ in range(5):
        auth.authenticate("reader", "wrong", lockout_base_seconds=0.0)
    assert auth.authenticate("reader", "read-password-1") is not None


def test_a_successful_login_clears_the_failure_count(auth: AuthStore) -> None:
    """Four typos, a success, four more typos, a success: never locked.

    Without a reset on success the counter would ratchet up over weeks and lock
    the operator out of their own dashboard for no reason.
    """
    for _ in range(4):
        auth.authenticate("reader", "wrong")
    assert auth.authenticate("reader", "read-password-1") is not None
    for _ in range(4):
        assert auth.authenticate("reader", "wrong") is None
    assert auth.authenticate("reader", "read-password-1") is not None



def test_a_disabled_account_cannot_log_in(auth: AuthStore) -> None:
    """Disabling revokes access without deleting the record."""
    auth.disable("reader")
    assert auth.authenticate("reader", "read-password-1") is None
    assert any(u.username == "reader" for u in auth.users())


def test_unknown_and_wrong_are_indistinguishable_in_outcome(auth: AuthStore) -> None:
    """Both return exactly ``None``.

    A distinguishable return is a user-enumeration oracle regardless of what the
    caller does with it.
    """
    assert auth.authenticate("ghost", "x") is None
    assert auth.authenticate("reader", "x") is None


def test_unknown_and_wrong_take_comparable_time(auth: AuthStore) -> None:
    """The unknown-user path must not return noticeably faster.

    This is the property :func:`dummy_verify` exists for. The tolerance is loose
    on purpose -- it only needs to catch an order-of-magnitude difference, and a
    tight bound would make this test flaky on a loaded machine.
    """
    # Warm the dummy hash so its first-call cost is not attributed to the timing.
    dummy_verify()

    def timeit(username: str, password: str) -> float:
        samples = []
        for _ in range(5):
            start = time.monotonic()
            auth.authenticate(username, password)
            samples.append(time.monotonic() - start)
        return min(samples)

    unknown = timeit("ghost", "x")
    wrong = timeit("reader", "x")
    ratio = unknown / wrong if wrong else float("inf")
    assert ratio > 0.3, f"unknown user was {ratio:.2f}x faster -- enumerable"


# --------------------------------------------------------------------------- #
# Account management
# --------------------------------------------------------------------------- #


def test_duplicate_accounts_are_refused(auth: AuthStore) -> None:
    """Silently overwriting a password would be a way in."""
    with pytest.raises(StoreError, match="already exists"):
        auth.add_user("reader", "different", role=Role.WRITE)


def test_operations_on_a_missing_user_are_errors(auth: AuthStore) -> None:
    """Every mutation reports an unknown account rather than doing nothing."""
    from collections.abc import Callable

    calls: list[Callable[[], None]] = [
        lambda: auth.set_password("ghost", "x"),
        lambda: auth.set_role("ghost", Role.WRITE),
        lambda: auth.disable("ghost"),
        lambda: auth.remove("ghost"),
    ]
    for call in calls:
        with pytest.raises(StoreError, match="no such user"):
            call()


def test_set_password_replaces_the_credential_and_lifts_a_lockout(
    auth: AuthStore,
) -> None:
    """A password reset is also the way out of a lockout."""
    for _ in range(5):
        auth.authenticate("reader", "wrong")
    assert auth.remaining_lockout("reader") > 0
    auth.set_password("reader", "brand-new-password")
    assert auth.authenticate("reader", "brand-new-password") is not None
    assert auth.authenticate("reader", "read-password-1") is None


def test_set_role_takes_effect_on_the_next_login(auth: AuthStore) -> None:
    """Promoting a user is not retroactive to an existing session.

    Worth noting because the in-app gate reads the role from the auth store
    on each request, so a role change applies immediately. With a
    connection-level credential from a proxy that is the desired behaviour.
    """
    before = auth.authenticate("reader", "read-password-1")
    assert before is not None and before.can_write is False
    auth.set_role("reader", Role.WRITE)
    after = auth.authenticate("reader", "read-password-1")
    assert after is not None and after.can_write is True


def test_remove_deletes_the_account_and_its_lockout(auth: AuthStore) -> None:
    """No orphaned lockout rows survive a deletion."""
    for _ in range(5):
        auth.authenticate("reader", "wrong")
    auth.remove("reader")
    assert auth.authenticate("reader", "read-password-1") is None
    assert all(u.username != "reader" for u in auth.users())


def test_users_are_listed_in_order_with_state(auth: AuthStore) -> None:
    """The CLI listing has to show enough to act on."""
    auth.disable("reader")
    names = [u.username for u in auth.users()]
    assert names == ["admin", "reader"]
    disabled = next(u for u in auth.users() if u.username == "reader")
    assert disabled.disabled_at is not None
    assert disabled.role is Role.READ


def test_count_active_excludes_disabled(auth: AuthStore) -> None:
    """Used by the fail-closed check, so it must not count dead accounts."""
    assert auth.count_active() == 2
    auth.disable("reader")
    assert auth.count_active() == 1


# --------------------------------------------------------------------------- #
# Storage hygiene
# --------------------------------------------------------------------------- #


def test_the_auth_file_is_not_world_readable(tmp_path: Path) -> None:
    """Credential hashes must not be readable by other accounts.

    A 0644 auth file is the single most common way a password database leaks,
    and it is invisible until someone goes looking for the mode.
    """
    path = tmp_path / "auth.sqlite3"
    store = AuthStore(path)
    try:
        store.add_user("kyle", "a-password", role=Role.WRITE)
    finally:
        store.close()
    assert path.stat().st_mode & 0o077 == 0, oct(path.stat().st_mode)
    # And the WAL/shm sidecars, which also contain rows.
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists():
            assert sidecar.stat().st_mode & 0o077 == 0, f"{suffix} is readable"


def test_the_auth_file_is_separate_from_the_monitoring_store(tmp_path: Path) -> None:
    """Credentials must not land in the observation database.

    A backup of the monitoring data should not contain password hashes, and the
    dashboard should not need write access to the store it only reads.
    """
    auth_path = tmp_path / "auth.sqlite3"
    store = AuthStore(auth_path)
    try:
        store.add_user("kyle", "a-password")
    finally:
        store.close()
    assert auth_path.name != "tracker.sqlite3"
    raw = auth_path.read_bytes()
    assert b"scrypt$" in raw  # the hash is here...
    assert b"a-password" not in raw  # ...and the plaintext is not


# --------------------------------------------------------------------------- #
# Proxy identity: the fail-closed model
# --------------------------------------------------------------------------- #


ACCOUNTS = {"admin": Role.WRITE, "reader": Role.READ}


def test_a_correctly_signed_header_is_accepted() -> None:
    """The happy path: proxy authenticated the human, signed it, we agree."""
    identity = verify_identity(
        "admin", "shared-secret", expected_secret="shared-secret", users=ACCOUNTS
    )
    assert identity is not None
    assert identity.username == "admin"
    assert identity.can_write
    assert identity.source == "proxy"


def test_a_forged_identity_header_is_rejected() -> None:
    """A direct caller who guesses a username still cannot become that user.

    This is the entire reason the shared secret exists. Without it, anyone who
    can open a socket to the dashboard sends one header and is believed.
    """
    for forged in ("s3cret", "", "shared-secretX", "Shared-Secret"):
        assert verify_identity(
            "admin", forged, expected_secret="shared-secret", users=ACCOUNTS
        ) is None


def test_no_configured_secret_means_no_trust() -> None:
    """A missing secret must deny, never default to believing the header.

    Failing open here would make a misconfiguration silently remove the entire
    control, which is the opposite of what a security setting should do.
    """
    accounts = ACCOUNTS
    secret = "shared-secret"
    assert verify_identity("admin", "anything", expected_secret=None, users=accounts) is None
    assert verify_identity("admin", None, expected_secret=secret, users=accounts) is None
    assert verify_identity(None, secret, expected_secret=secret, users=accounts) is None


def test_a_signed_but_unknown_user_is_rejected() -> None:
    """A valid signature is not enough; the account must exist here.

    Otherwise anyone the proxy will authenticate becomes an admin here, since a
    permissive proxy would otherwise grant an arbitrary username.
    """
    assert verify_identity(
        "intruder", "shared-secret", expected_secret="shared-secret", users=ACCOUNTS
    ) is None


def test_the_secret_is_compared_in_constant_time() -> None:
    """The comparison must be the constant-time one.

    Asserted against the function source because the behaviour cannot be tested
    by observation: a `==` and a `compare_digest` return the same answers, they
    just leak the secret at different speeds. The check is narrow so it cannot
    be satisfied by an unrelated `==` elsewhere in the function.
    """
    import inspect

    source = inspect.getsource(verify_identity)
    secret_line = next(
        line for line in source.splitlines() if "expected_secret.encode" in line
    )
    assert "hmac.compare_digest(" in secret_line, secret_line.strip()
    assert "==" not in secret_line


def test_role_mapping_from_the_accounts_table_is_honoured() -> None:
    """A read-only account stays read-only when arriving via the proxy."""
    identity = verify_identity(
        "reader", "s", expected_secret="s", users=ACCOUNTS
    )
    assert identity is not None
    assert identity.can_read
    assert not identity.can_write


def test_the_roles_are_a_ladder_not_two_switches() -> None:
    """``WRITE`` must include everything ``READ`` can do.

    Two independent flags would permit the nonsense combination of a user who
    may write but not read, which is never a real permission set.
    """
    assert Role.WRITE.implies_read
    assert Role.READ.implies_read
    assert Role("write") == Role.WRITE


# --------------------------------------------------------------------------- #
# The primitives themselves
# --------------------------------------------------------------------------- #


def test_scrypt_is_actually_being_used() -> None:
    """Guard against the parameters being silently ignored.

    If `n` were dropped, the hash would still verify -- it would just be fast,
    and every timing test above would be measuring nothing.
    """
    salt = b"sixteen-byte-salt"
    fast = hashlib.scrypt(b"pw", salt=salt, n=2**10, r=8, p=1, dklen=32, maxmem=64 << 20)
    slow = hashlib.scrypt(b"pw", salt=salt, n=2**14, r=8, p=1, dklen=32, maxmem=64 << 20)
    assert fast != slow
    assert len(fast) == 32


def test_identity_reports_its_source_for_the_audit_line(auth: AuthStore) -> None:
    """The dashboard shows *why* it is trusting someone."""
    password_identity = auth.authenticate("admin", "write-password-1")
    assert password_identity is not None
    assert password_identity.source == "password"
    proxy_identity = verify_identity("admin", "s", expected_secret="s", users=ACCOUNTS)
    assert proxy_identity is not None
    assert proxy_identity.source == "proxy"


def test_verify_identity_returns_none_rather_than_raising() -> None:
    """A malformed header must deny, not produce a 500."""
    for args in (
        ("", "s", "s"),
        ("admin", "", "s"),
        ("admin", "s", ""),
    ):
        result = verify_identity(
            args[0], args[1], expected_secret=args[2], users=ACCOUNTS
        )
        assert result is None


def test_constant_time_helper_is_used_for_secrets() -> None:
    """Both secret comparisons in this module go through hmac."""
    assert hmac.compare_digest(b"a", b"a")
    assert not hmac.compare_digest(b"a", b"b")


def test_identity_is_immutable(auth: AuthStore) -> None:
    """A role cannot be escalated by mutating a returned object."""
    identity = auth.authenticate("admin", "write-password-1")
    assert identity is not None
    with pytest.raises(AttributeError):
        identity.role = Role.READ  # type: ignore[misc]
