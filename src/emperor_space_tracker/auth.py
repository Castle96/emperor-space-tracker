"""Password hashing, users, and lockout for the dashboard.

Deliberately standard-library only, and deliberately separate from both the
monitoring store and the Streamlit layer. Two reasons.

First, the daemon's whole premise is that the core provisions with no
dependency tree, and auth is the one place where a supply-chain dependency is
least acceptable. :mod:`hashlib` gives scrypt, which is a memory-hard KDF
designed exactly for password storage, so there is no reason to reach for
bcrypt or argon2 from a wheel.

Second, this module must be testable without a browser or a running Streamlit
server. Credential handling is the one code path where "it looked right" is
worth nothing, so it lives somewhere the test suite can hammer it directly.

Why a separate database
-----------------------
Users and lockout state go in their own SQLite file, not the monitoring store.
Three reasons, in order of importance:

* the dashboard should not need write access to the observation database. A
  reader that also writes is a much bigger blast radius if the dashboard is
  compromised, and a WAL reader already needs write access to the ``-shm``
  file, so that is the only write it legitimately needs.
* ``est reset --data`` wipes observations. It should not silently revoke every
  login, and it certainly should not be able to grant one.
* the file can carry ``0600`` and hold nothing but credentials, so a backup of
  the monitoring data does not contain the password hashes.

What this module does not do
----------------------------
It does not manage sessions. Streamlit 1.64 removed ``[server] password`` and
exposes no cookie *setter* to scripts -- ``st.context.cookies`` is read-only --
so an app-side session cookie cannot be written through any supported API. Since
a session mechanism that cannot set its own cookie is a mechanism that fails
open the first time a browser drops the websocket, the login decision is made by
a reverse proxy that can hold a connection-level credential, and this module
supplies the part a proxy cannot: the role model, and the lockout that a static
basic-auth file has no way to express.

A proxy authenticating does not mean the app trusts the network. See
:func:`verify_identity` -- an identity header is only believed when it arrives
with a shared secret, and the check fails closed when the secret is absent.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

from .errors import StoreError

__all__ = [
    "AuthStore",
    "AuthUser",
    "Identity",
    "Role",
    "hash_password",
    "lockout_for",
    "normalise_username",
    "verify_identity",
    "verify_password",
]

_LOG = logging.getLogger("emperor.auth")

#: scrypt cost parameters.
#:
#: n=2**15 with r=8 needs ~32 MiB per hash and takes roughly 50-100 ms on the
#: class of hardware a field node has. That is the right order of magnitude:
#: slow enough to make an offline attack against a stolen hash file expensive,
#: fast enough that an operator can still log in. p=1 is deliberate -- scrypt's
#: parallelisation parameter buys nothing for a single-password-per-attempt
#: threat model, and costs memory linearly.
_SCRYPT_N: int = 2**15
_SCRYPT_R: int = 8
_SCRYPT_P: int = 1
_SCRYPT_DKLEN: int = 32
_SALT_BYTES: int = 16

#: A hash of a random password nobody can supply, used to equalise the time
#: taken when a username does not exist. Without it, "unknown user" returns
#: faster than "wrong password", which enumerates valid usernames.
_DUMMY_HASH: str = ""


class Role(StrEnum):
    """What a dashboard user is permitted to do.

    Split deliberately, and ``WRITE`` implies ``READ`` rather than being
    independent: the dashboard currently only reads, so the distinction is
    carried for the write paths that are coming (alert acknowledge, rule
    retuning) rather than invented as an authorisation system nobody uses.
    """

    READ = "read"
    WRITE = "write"

    @property
    def implies_read(self) -> bool:
        """Return whether this role can do everything ``READ`` can."""
        return self in {Role.READ, Role.WRITE}


@dataclass(frozen=True, slots=True)
class AuthUser:
    """A dashboard account.

    Attributes
    ----------
    username
        Normalised login name.
    role
        Permission level.
    created_at
        When the account was made.
    disabled_at
        When the account was disabled, or ``None`` if active.
    """

    username: str
    role: Role
    created_at: datetime
    disabled_at: datetime | None


@dataclass(frozen=True, slots=True)
class Identity:
    """A caller that has been authenticated and authorised.

    Attributes
    ----------
    username
        Who they are, for the audit line and for display.
    role
        What they may do.
    source
        How they were authenticated. Carried so the dashboard can show *why* it
        is trusting them, which is the difference between a security control
        and a mystery.
    """

    username: str
    role: Role
    source: str

    @property
    def can_read(self) -> bool:
        """Return whether this identity may view the dashboard."""
        return self.role.implies_read

    @property
    def can_write(self) -> bool:
        """Return whether this identity may perform write actions."""
        return self.role is Role.WRITE


def normalise_username(name: str) -> str:
    """Return a username in canonical form.

    Lowercased and stripped, so ``KCastle96`` and ``kcastle96`` are one account
    rather than two that differ by a keystroke.

    Parameters
    ----------
    name
        Raw input.

    Returns
    -------
    str
        The canonical username.

    Raises
    ------
    ValueError
        If the result is empty or implausibly long.
    """
    cleaned = name.strip().lower()
    if not cleaned:
        msg = "username must not be empty"
        raise ValueError(msg)
    if len(cleaned) > 64:
        msg = "username must be 64 characters or fewer"
        raise ValueError(msg)
    return cleaned


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def hash_password(password: str, *, n: int | None = None) -> str:
    """Hash a password with scrypt and a fresh random salt.

    The returned string carries its own parameters, so raising the cost later
    does not invalidate existing hashes -- :func:`verify_password` reads them
    back and uses what it finds.

    Parameters
    ----------
    password
        The plaintext. Never logged, never stored, never returned.
    n
        Override the CPU/memory cost, for tests that need to be fast. Never
        lower this in production.

    Returns
    -------
    str
        ``scrypt$n$r$p$salt$key``, base64 with standard padding.

    Raises
    ------
    ValueError
        If the password is empty.

    Examples
    --------
    >>> h = hash_password("correct horse battery staple")
    >>> h.startswith("scrypt$")
    True
    >>> verify_password("correct horse battery staple", h)
    True
    >>> verify_password("wrong", h)
    False
    """
    if not password:
        msg = "password must not be empty"
        raise ValueError(msg)
    cost = n or _SCRYPT_N
    salt = secrets.token_bytes(_SALT_BYTES)
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=cost,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
        maxmem=64 * 1024 * 1024,
    )
    return f"scrypt${cost}${_SCRYPT_R}${_SCRYPT_P}${_b64(salt)}${_b64(derived)}"


def _parse_hash(encoded: str) -> tuple[int, int, int, bytes, bytes]:
    """Split a stored hash into its parts.

    Raises
    ------
    ValueError
        If the string is not a well-formed hash.
    """
    parts = encoded.split("$")
    if len(parts) != 6 or parts[0] != "scrypt":
        msg = "unrecognised password hash format"
        raise ValueError(msg)
    n, r, p = int(parts[1]), int(parts[2]), int(parts[3])
    return n, r, p, base64.b64decode(parts[4]), base64.b64decode(parts[5])


def _derive(password: str, n: int, r: int, p: int, salt: bytes) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=n,
        r=r,
        p=p,
        dklen=_SCRYPT_DKLEN,
        maxmem=64 * 1024 * 1024,
    )


def verify_password(password: str, encoded: str) -> bool:
    """Check a password against a stored hash, in constant time.

    Parameters
    ----------
    password
        The candidate plaintext.
    encoded
        A hash as produced by :func:`hash_password`.

    Returns
    -------
    bool
        Whether they match. A malformed stored hash is ``False``, never an
        exception, so a corrupt row denies access rather than crashing the
        login form.
    """
    try:
        n, r, p, salt, expected = _parse_hash(encoded)
    except (ValueError, TypeError):
        _LOG.error("stored password hash is malformed; denying access")
        return False
    candidate = _derive(password, n, r, p, salt)
    return hmac.compare_digest(candidate, expected)


def dummy_verify() -> None:
    """Burn the same CPU a real verification would.

    Called when the supplied username does not exist, so that "no such user"
    and "wrong password" take indistinguishable time. Without it, response
    latency alone enumerates every valid account.

    Examples
    --------
    >>> dummy_verify()
    >>> verify_password("anything", _DUMMY_HASH or "scrypt$1024$8$1$AA==$AA==")
    False
    """
    try:
        n, r, p, salt, _ = _parse_hash(_DUMMY_HASH)
    except (ValueError, TypeError):
        # Lazily built so the module import does no KDF work.
        globals()["_DUMMY_HASH"] = hash_password(secrets.token_urlsafe(32))
        n, r, p, salt, _ = _parse_hash(_DUMMY_HASH)
    _derive("dummy", n, r, p, salt)


def lockout_for(
    failures: int,
    *,
    max_failures: int,
    base_seconds: float,
    cap_seconds: float = 3600.0,
) -> float:
    """Return how long an account should be locked after ``failures`` attempts.

    Exponential, because a flat window is either too short to matter or too long
    to be usable. The cap matters for a different reason: an attacker who can
    lock an account for a week has turned a brute-force defence into a
    denial-of-service tool against a monitoring node's dashboard.

    Parameters
    ----------
    failures
        Consecutive failed attempts so far.
    max_failures
        Attempts tolerated before the first lockout.
    base_seconds
        Length of the first lockout; each subsequent one doubles.
    cap_seconds
        Ceiling on the lockout.

    Returns
    -------
    float
        Seconds to lock, or ``0.0`` when under the threshold.

    Examples
    --------
    >>> lockout_for(0, max_failures=5, base_seconds=60.0)
    0.0
    >>> lockout_for(5, max_failures=5, base_seconds=60.0)
    60.0
    >>> lockout_for(6, max_failures=5, base_seconds=60.0)
    120.0
    >>> lockout_for(40, max_failures=5, base_seconds=60.0, cap_seconds=600.0)
    600.0
    """
    if failures < max_failures:
        return 0.0
    over = failures - max_failures
    return float(min(base_seconds * (2**over), cap_seconds))


class AuthStore:
    """Users, roles and lockout state in their own SQLite file.

    Parameters
    ----------
    path
        Location of the auth database. Created if absent with ``0600``.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> p = pathlib.Path(tempfile.mkdtemp()) / "auth.sqlite3"
    >>> store = AuthStore(p)
    >>> _ = store.add_user("kyle", "hunter2hunter2", role=Role.WRITE)
    >>> store.authenticate("kyle", "hunter2hunter2").role
    <Role.WRITE: 'write'>
    >>> store.authenticate("kyle", "wrong") is None
    True
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS users (
        username       TEXT PRIMARY KEY,
        password_hash  TEXT NOT NULL,
        role           TEXT NOT NULL,
        created_at     INTEGER NOT NULL,
        disabled_at    INTEGER
    );
    CREATE TABLE IF NOT EXISTS lockouts (
        username       TEXT PRIMARY KEY,
        failures       INTEGER NOT NULL,
        locked_until   INTEGER
    );
    """

    def __init__(self, path: Path) -> None:
        """Open (or create) the auth database.

        Parameters
        ----------
        path
            File location.

        Raises
        ------
        StoreError
            If the file cannot be created or opened.
        """
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(self._SCHEMA)
            self._conn.commit()
            # Credentials only. Written after creation so an existing file with
            # looser permissions is tightened too.
            self.path.chmod(0o600)
        except (OSError, sqlite3.Error) as exc:
            msg = f"cannot open auth store {self.path}: {exc}"
            raise StoreError(msg) from exc

    def close(self) -> None:
        """Close the connection."""
        self._conn.close()

    def __enter__(self) -> AuthStore:
        """Support ``with``."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close on exit."""
        self.close()

    # -- users ----------------------------------------------------------

    def add_user(self, username: str, password: str, *, role: Role = Role.READ) -> AuthUser:
        """Create an account.

        Parameters
        ----------
        username
            Login name; normalised.
        password
            Plaintext, hashed immediately and never stored or logged.
        role
            Permission level.

        Returns
        -------
        AuthUser
            The created account.

        Raises
        ------
        StoreError
            If the account already exists or the write fails.
        """
        name = normalise_username(username)
        now = utc_now()
        try:
            self._conn.execute(
                "INSERT INTO users(username, password_hash, role, created_at) "
                "VALUES(?,?,?,?)",
                (name, hash_password(password), str(role), to_micros(now)),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as exc:
            msg = f"user {name!r} already exists"
            raise StoreError(msg) from exc
        except sqlite3.Error as exc:
            msg = f"cannot add user {name!r}: {exc}"
            raise StoreError(msg) from exc
        _LOG.info("created dashboard user %s with role %s", name, role)
        return AuthUser(name, role, now, None)

    def set_password(self, username: str, password: str) -> None:
        """Replace a user's password and clear their lockout.

        Parameters
        ----------
        username
            Login name.
        password
            New plaintext.

        Raises
        ------
        StoreError
            If the user does not exist.
        """
        name = normalise_username(username)
        cursor = self._conn.execute(
            "UPDATE users SET password_hash=? WHERE username=?", (hash_password(password), name)
        )
        if cursor.rowcount == 0:
            msg = f"no such user: {name!r}"
            raise StoreError(msg)
        self._conn.execute("DELETE FROM lockouts WHERE username=?", (name,))
        self._conn.commit()

    def set_role(self, username: str, role: Role) -> None:
        """Change a user's role.

        Parameters
        ----------
        username
            Login name.
        role
            New permission level.

        Raises
        ------
        StoreError
            If the user does not exist.
        """
        name = normalise_username(username)
        cursor = self._conn.execute(
            "UPDATE users SET role=? WHERE username=?", (str(role), name)
        )
        if cursor.rowcount == 0:
            msg = f"no such user: {name!r}"
            raise StoreError(msg)
        self._conn.commit()
        _LOG.info("set dashboard user %s to role %s", name, role)

    def disable(self, username: str) -> None:
        """Disable an account without deleting its history.

        Parameters
        ----------
        username
            Login name.

        Raises
        ------
        StoreError
            If the user does not exist.
        """
        name = normalise_username(username)
        cursor = self._conn.execute(
            "UPDATE users SET disabled_at=? WHERE username=?", (to_micros(utc_now()), name)
        )
        if cursor.rowcount == 0:
            msg = f"no such user: {name!r}"
            raise StoreError(msg)
        self._conn.commit()

    def remove(self, username: str) -> None:
        """Delete an account and its lockout state.

        Parameters
        ----------
        username
            Login name.

        Raises
        ------
        StoreError
            If the user does not exist.
        """
        name = normalise_username(username)
        cursor = self._conn.execute("DELETE FROM users WHERE username=?", (name,))
        if cursor.rowcount == 0:
            msg = f"no such user: {name!r}"
            raise StoreError(msg)
        self._conn.execute("DELETE FROM lockouts WHERE username=?", (name,))
        self._conn.commit()

    def users(self) -> list[AuthUser]:
        """Return every account, active or not, alphabetically.

        Returns
        -------
        list[AuthUser]
        """
        rows = self._conn.execute(
            "SELECT username, role, created_at, disabled_at FROM users ORDER BY username"
        ).fetchall()
        return [
            AuthUser(
                username=row["username"],
                role=Role(row["role"]),
                created_at=from_micros(row["created_at"]),
                disabled_at=(
                    from_micros(row["disabled_at"]) if row["disabled_at"] else None
                ),
            )
            for row in rows
        ]

    def count_active(self) -> int:
        """Return how many accounts can actually log in.

        Returns
        -------
        int
        """
        return int(
            self._conn.execute(
                "SELECT count(*) FROM users WHERE disabled_at IS NULL"
            ).fetchone()[0]
        )

    # -- lockout --------------------------------------------------------

    def _lockout_row(self, username: str) -> sqlite3.Row | None:
        row: sqlite3.Row | None = self._conn.execute(
            "SELECT failures, locked_until FROM lockouts WHERE username=?", (username,)
        ).fetchone()
        return row

    def remaining_lockout(self, username: str) -> float:
        """Return seconds remaining on an active lockout.

        Parameters
        ----------
        username
            Login name.

        Returns
        -------
        float
            Seconds remaining, or ``0.0`` when not locked.
        """
        row = self._lockout_row(normalise_username(username))
        if row is None or not row["locked_until"]:
            return 0.0
        until: int = row["locked_until"]
        remaining = (until - to_micros(utc_now())) / 1_000_000
        return float(max(0.0, remaining))

    def _record_failure(self, username: str, *, max_failures: int, base: float) -> float:
        row = self._lockout_row(username)
        failures = (row["failures"] if row else 0) + 1
        seconds = lockout_for(failures, max_failures=max_failures, base_seconds=base)
        until = to_micros(utc_now() + timedelta(seconds=seconds)) if seconds else None
        self._conn.execute(
            "INSERT INTO lockouts(username, failures, locked_until) VALUES(?,?,?) "
            "ON CONFLICT(username) DO UPDATE SET failures=excluded.failures, "
            "locked_until=excluded.locked_until",
            (username, failures, until),
        )
        self._conn.commit()
        return seconds

    def _clear_failures(self, username: str) -> None:
        self._conn.execute("DELETE FROM lockouts WHERE username=?", (username,))
        self._conn.commit()

    # -- authentication -------------------------------------------------

    def authenticate(
        self,
        username: str,
        password: str,
        *,
        max_failures: int = 5,
        lockout_base_seconds: float = 60.0,
    ) -> Identity | None:
        """Verify a credential and return the caller's identity.

        The only path that decides whether a human gets in. Returns ``None`` for
        every failure mode -- unknown user, wrong password, disabled account,
        locked out -- because a caller that could tell them apart would leak
        which usernames exist.

        A locked-out account still has its password verified, so the timing does
        not reveal that it exists and is merely locked.

        Parameters
        ----------
        username
            Supplied login name.
        password
            Supplied plaintext.
        max_failures
            Attempts before the first lockout.
        lockout_base_seconds
            Length of the first lockout.

        Returns
        -------
        Identity | None
            The authenticated caller, or ``None`` on any failure.
        """
        try:
            name = normalise_username(username)
        except ValueError:
            dummy_verify()
            return None

        row = self._conn.execute(
            "SELECT username, password_hash, role, disabled_at FROM users WHERE username=?",
            (name,),
        ).fetchone()

        if row is None:
            # Equalise timing against the real path, then fail.
            dummy_verify()
            return None

        if not verify_password(password, row["password_hash"]):
            seconds = self._record_failure(
                name, max_failures=max_failures, base=lockout_base_seconds
            )
            if seconds:
                _LOG.warning(
                    "locked dashboard user %s for %.0fs after %d failed attempts",
                    name, seconds, max_failures,
                )
            return None

        # Correct password, but locked out: still deny, and say so in the log
        # without telling the caller.
        remaining = self.remaining_lockout(name)
        if remaining > 0:
            _LOG.warning("denied locked dashboard user %s (%.0fs remaining)", name, remaining)
            return None

        if row["disabled_at"]:
            _LOG.warning("denied disabled dashboard user %s", name)
            return None

        self._clear_failures(name)
        role = Role(row["role"])
        _LOG.info("dashboard login: user=%s role=%s", name, role)
        return Identity(username=name, role=role, source="password")


def verify_identity(
    username: str | None,
    secret: str | None,
    *,
    expected_secret: str | None,
    users: dict[str, Role],
) -> Identity | None:
    """Trust an identity header only when it is accompanied by a shared secret.

    This is the whole security model of the reverse-proxy arrangement, so it is
    worth being explicit about the threat.

    A proxy in front of the app authenticates the human and passes their name in
    a header. If the app believed that header on its own, then *any* host that
    could open a TCP connection to the dashboard -- not just the proxy -- could
    send ``X-EST-User: admin`` and be believed. The proxy's authentication would
    be decorative.

    The shared secret closes that. A direct caller does not have it, so a
    forged header is rejected. The secret is long and random, lives in the proxy
    config and the app config, and is compared in constant time.

    This fails **closed**: if no secret is configured, or the presented secret
    does not match, the result is ``None`` and the caller is refused. There is
    no configuration in which a missing secret means "trust the header".

    Parameters
    ----------
    username
        Identity as presented by the proxy, or ``None``.
    secret
        Shared secret as presented by the proxy, or ``None``.
    expected_secret
        The configured secret, or ``None`` if none is configured.
    users
        Known accounts and their roles.

    Returns
    -------
    Identity | None
        The verified identity, or ``None`` to refuse.

    Examples
    --------
    >>> accounts = {"kyle": Role.WRITE}
    >>> verify_identity("kyle", "s3cret", expected_secret="s3cret", users=accounts).role
    <Role.WRITE: 'write'>
    >>> verify_identity("kyle", "wrong", expected_secret="s3cret", users=accounts) is None
    True
    >>> verify_identity("kyle", None, expected_secret="s3cret", users=accounts) is None
    True
    >>> verify_identity("kyle", "s3cret", expected_secret=None, users=accounts) is None
    True
    >>> verify_identity("nobody", "s3cret", expected_secret="s3cret", users=accounts) is None
    True

    With an empty account table the proxy is the only authorisation layer:

    >>> proxy = verify_identity("anyone", "s3cret", expected_secret="s3cret", users={})
    >>> proxy.role
    <Role.WRITE: 'write'>
    """
    if not expected_secret or not secret or not username:
        return None
    if not hmac.compare_digest(secret.encode("utf-8"), expected_secret.encode("utf-8")):
        _LOG.warning(
            "rejected a dashboard identity header carrying the wrong shared secret; "
            "this is either a misconfigured proxy or a host forging an identity"
        )
        return None
    name = username.strip().lower()
    if not users:
        # No account store. The proxy authenticated the human and proved the
        # request came through it, which is the whole claim being made here; the
        # proxy is the authorisation layer. Everyone behind it is a writer,
        # which is correct while the dashboard only reads.
        return Identity(username=name, role=Role.WRITE, source="proxy")
    role = users.get(name)
    if role is None:
        _LOG.warning(
            "rejected dashboard identity for %r: authenticated by the proxy but no "
            "such account exists here. Add it with `est auth add-user`, or remove the "
            "account from the proxy.",
            name,
        )
        return None
    return Identity(username=name, role=role, source="proxy")


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(UTC)


def to_micros(moment: datetime) -> int:
    """Convert a datetime to integer microseconds since the epoch."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return int(moment.timestamp() * 1_000_000)


def from_micros(value: int) -> datetime:
    """Convert integer microseconds back to an aware datetime."""
    return datetime.fromtimestamp(value / 1_000_000, tz=UTC)
