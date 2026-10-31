"""Task-tree scoped policy stacks for the worker thread and blocking portal bridges.

This module provides a stack of :class:`ThreadBridgePolicy` frames that can be pushed
for the duration of a block of code inside an AnyIO task tree.  Each frame overrides
four independent dimensions of the thread bridging machinery:

* :attr:`ThreadBridgePolicy.cancel_handoff` – how
  :func:`~anyio.to_thread.run_sync` reacts to cancellation of the host task,
* :attr:`ThreadBridgePolicy.limiter_bind` – which capacity limiter (if any) a worker
  thread dispatch reserves a token in,
* :attr:`ThreadBridgePolicy.portal_admit` – whether a
  :class:`~anyio.from_thread.BlockingPortal` keeps accepting tasks from external
  threads,
* :attr:`ThreadBridgePolicy.token_gate` – whether
  :func:`~anyio.from_thread.run` and :func:`~anyio.from_thread.run_sync` may fall back
  to the implicit worker thread token.

The stack is deliberately separate from cancellation deadline scopes, memory object
stream bookkeeping, task group spawning and the test runner machinery.

Isolation model
---------------

The stack has two layers:

1. A :class:`~contextvars.ContextVar` chain holds the frames pushed by the current task
   context.  Like every context variable, the chain is copied when a new task is
   spawned, so two tasks running concurrently (even inside the same task tree) never
   see each other's pushes and pop can never restore a process-wide singleton.
2. Active frames are additionally registered against the *root task* of the current
   AnyIO task tree.  This registry is only used to answer cross-thread questions
   (whether a :class:`~anyio.from_thread.BlockingPortal` may admit a new task), because
   worker threads and external portal threads do not share the event loop thread's
   context.

This is **not** a :class:`~anyio.lowlevel.RunVar`: a ``RunVar`` buckets its values by
the native event loop, so two independent task trees sharing the same loop would see
the same value.  The root task registry keys by the task tree instead, and the
context variable chain additionally isolates concurrent tasks within one tree.

Dispatch snapshots
------------------

Every crossing reads the effective frame at dispatch time:

* :func:`~anyio.to_thread.run_sync` resolves the frame before the worker thread is
  handed the callable and the worker thread keeps that snapshot for its lifetime,
* a :class:`~anyio.from_thread.BlockingPortal` resolves the frame in the event loop
  thread when a task is spawned from an external thread.

Pushing another frame after the worker thread has started executing, or after the
portal task has been spawned, is therefore **void for the operation already in
flight**: it keeps the frame captured at dispatch time.  Later pushes only affect
later dispatches.

"""

from __future__ import annotations

__all__ = (
    "CancelHandoff",
    "LimiterBind",
    "PortalAdmit",
    "ThreadBridgePolicy",
    "ThreadBridgePolicyError",
    "TokenGate",
    "current_root_thread_bridge_policy",
    "current_thread_bridge_policy",
    "current_worker_thread_bridge_policy",
    "current_worker_token_gate",
    "enforce_token_gate",
    "resolve_cancel_handoff",
    "resolve_portal_admission",
    "resolve_worker_limiter",
    "thread_bridge_policy",
)

import enum
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from inspect import iscoroutinefunction
from typing import TYPE_CHECKING, Literal
from weakref import WeakKeyDictionary

from ._core._eventloop import get_async_backend, threadlocals

if TYPE_CHECKING:
    from ._core._synchronization import CapacityLimiter
    from .lowlevel import EventLoopToken

#: The name of the worker thread local that carries the dispatch-time frame.
WORKER_FRAME_ATTR = "thread_bridge_policy"


class CancelHandoff(enum.Enum):
    """
    Policy for how :func:`~anyio.to_thread.run_sync` hands off cancellation of the
    host task.
    """

    #: Follow the caller's ``abandon_on_cancel`` / deprecated ``cancellable`` argument.
    pin = "pin"
    #: Abandon the worker thread result on host cancellation even if the caller passed
    #: ``abandon_on_cancel=False``.
    force_abandon = "force_abandon"
    #: Wait for the worker thread to finish even if the caller passed
    #: ``abandon_on_cancel=True``.  The worker thread can still observe the host
    #: cancellation via :func:`~anyio.from_thread.check_cancelled`.
    force_wait = "force_wait"


class LimiterBind(enum.Enum):
    """Policy for which capacity limiter a worker thread dispatch binds to."""

    #: Use the caller's limiter, falling back to the default thread limiter.
    pin = "pin"
    #: Ignore the caller's limiter and always use the default thread limiter.
    force_default = "force_default"
    #: Dispatch without acquiring or waiting on any capacity limiter token.
    bypass_acquire = "bypass_acquire"


class PortalAdmit(enum.Enum):
    """Policy for whether a blocking portal admits tasks from external threads."""

    #: Keep accepting calls as usual.
    pin = "pin"
    #: Reject every ``call``/``start_task_soon``/``start_task``/wrapped async context
    #: manager entry before anything is spawned.
    reject_new = "reject_new"
    #: Reject entries that target a coroutine function or an awaitable; only plain
    #: synchronous callables are admitted.
    sync_callable_only = "sync_callable_only"


class TokenGate(enum.Enum):
    """Policy for resolving the event loop token in worker/external threads."""

    #: Fall back to the implicit worker thread token; require a token outside a worker.
    pin = "pin"
    #: Require an explicit token even inside an AnyIO worker thread.
    require_explicit = "require_explicit"
    #: Reject explicit tokens that do not belong to the current portal/backend loop.
    forbid_foreign_token = "forbid_foreign_token"


class ThreadBridgePolicyError(RuntimeError):
    """Raised when a thread bridge policy frame rejects a thread/portal operation."""


@dataclass(frozen=True)
class ThreadBridgePolicy:
    """
    A single frame of the thread bridge and portal scheduling policy stack.

    All dimensions default to ``pin`` so that an empty stack is a no-op.

    Frames compare by value (two all-pin frames are equal), but the stack registry
    always removes frames by identity, so nesting equal frames cannot pop the wrong
    entry.
    """

    cancel_handoff: CancelHandoff = CancelHandoff.pin
    limiter_bind: LimiterBind = LimiterBind.pin
    portal_admit: PortalAdmit = PortalAdmit.pin
    token_gate: TokenGate = TokenGate.pin

    @staticmethod
    def merged(frames: Sequence[ThreadBridgePolicy]) -> ThreadBridgePolicy:
        """
        Merge a chain of frames into an effective frame.

        :param frames: frames ordered from the innermost (top of stack) to the
            outermost
        :return: the first non-``pin`` value encountered for each dimension, or
            ``pin`` if every frame leaves that dimension pinned

        """
        cancel_handoff = CancelHandoff.pin
        limiter_bind = LimiterBind.pin
        portal_admit = PortalAdmit.pin
        token_gate = TokenGate.pin
        for frame in frames:
            if (
                cancel_handoff is CancelHandoff.pin
                and frame.cancel_handoff is not CancelHandoff.pin
            ):
                cancel_handoff = frame.cancel_handoff
            if (
                limiter_bind is LimiterBind.pin
                and frame.limiter_bind is not LimiterBind.pin
            ):
                limiter_bind = frame.limiter_bind
            if (
                portal_admit is PortalAdmit.pin
                and frame.portal_admit is not PortalAdmit.pin
            ):
                portal_admit = frame.portal_admit
            if token_gate is TokenGate.pin and frame.token_gate is not TokenGate.pin:
                token_gate = frame.token_gate

        return ThreadBridgePolicy(
            cancel_handoff, limiter_bind, portal_admit, token_gate
        )


#: Frames pushed in the current task context, innermost frame first.
_frame_chain: ContextVar[tuple[ThreadBridgePolicy, ...] | None] = ContextVar(
    "anyio_thread_bridge_policy_chain", default=None
)

#: Frames active in each task tree, in push (outermost first) order.  The registry is
#: keyed by the backend-specific root task object, never by the native event loop, so
#: two task trees sharing a loop do not share frames.
_active_frames: WeakKeyDictionary[object, list[ThreadBridgePolicy]] = (
    WeakKeyDictionary()
)


def _current_root_task() -> object:
    return get_async_backend().current_root_task()


@contextmanager
def thread_bridge_policy(
    *,
    cancel_handoff: CancelHandoff = CancelHandoff.pin,
    limiter_bind: LimiterBind = LimiterBind.pin,
    portal_admit: PortalAdmit = PortalAdmit.pin,
    token_gate: TokenGate = TokenGate.pin,
) -> Iterator[None]:
    """
    Push a thread bridge and portal scheduling policy frame for the current task tree.

    The frame is popped again when the context manager exits, including via an
    exception.  Each of the four dimensions can be overridden independently; the
    others remain ``pin`` (or inherit the value of an outer non-pin frame).

    :param cancel_handoff: override for :func:`~anyio.to_thread.run_sync` cancellation
    :param limiter_bind: override for worker thread capacity limiter binding
    :param portal_admit: override for blocking portal admissions
    :param token_gate: override for event loop token resolution in worker threads

    """
    frame = ThreadBridgePolicy(
        cancel_handoff=cancel_handoff,
        limiter_bind=limiter_bind,
        portal_admit=portal_admit,
        token_gate=token_gate,
    )
    root_task = _current_root_task()
    old_chain = _frame_chain.get()
    token = _frame_chain.set((frame, *old_chain) if old_chain else (frame,))
    active_frames = _active_frames.get(root_task)
    if active_frames is None:
        active_frames = []
        _active_frames[root_task] = active_frames

    active_frames.append(frame)
    try:
        yield
    finally:
        _frame_chain.reset(token)
        for index, active_frame in enumerate(active_frames):
            if active_frame is frame:
                del active_frames[index]
                break

        if not active_frames:
            _active_frames.pop(root_task, None)


def current_thread_bridge_policy() -> ThreadBridgePolicy:
    """
    Return the effective policy frame for the current task context.

    This is the view used by :func:`~anyio.to_thread.run_sync` at dispatch time.
    Dimensions inherit the topmost non-pin value of the stack; an empty stack yields
    an all-pin frame, which is indistinguishable from never having pushed a frame.

    """
    frames = _frame_chain.get()
    if not frames:
        return ThreadBridgePolicy()

    return ThreadBridgePolicy.merged(frames)


def current_root_thread_bridge_policy(root_task: object) -> ThreadBridgePolicy:
    """
    Return the effective policy frame active in the given task tree.

    This view only sees frames registered against the task tree root, and is meant
    for cross-thread checks that hop into the event loop thread (blocking portal
    admissions).  It never reads the calling thread's context variables.

    :param root_task: the backend-specific root task object, as returned by the
        backend's ``current_root_task()`` implementation

    """
    active_frames = _active_frames.get(root_task)
    if not active_frames:
        return ThreadBridgePolicy()

    # The registry is stored outermost-first, merging expects innermost-first.
    return ThreadBridgePolicy.merged(tuple(reversed(active_frames)))


def current_worker_thread_bridge_policy() -> ThreadBridgePolicy | None:
    """
    Return the policy frame captured when the current worker thread was dispatched.

    :return: the dispatch-time frame, or ``None`` if the current thread is not an
        AnyIO worker thread carrying a dispatched call

    """
    return getattr(threadlocals, WORKER_FRAME_ATTR, None)


def current_worker_token_gate() -> TokenGate:
    """
    Return the effective ``token_gate`` for the current worker/external thread.

    An unclaimed external thread has no dispatch-time frame and therefore sees
    :attr:`TokenGate.pin`.

    """
    frame = current_worker_thread_bridge_policy()
    return frame.token_gate if frame is not None else TokenGate.pin


def _bind_worker_frame(frame: ThreadBridgePolicy) -> None:
    """Attach the dispatch-time frame to the current worker thread."""
    setattr(threadlocals, WORKER_FRAME_ATTR, frame)


def _unbind_worker_frame() -> None:
    try:
        delattr(threadlocals, WORKER_FRAME_ATTR)
    except AttributeError:
        pass


#
# to_thread.run_sync exit
#


def resolve_cancel_handoff(abandon_on_cancel: bool) -> bool:
    """
    Merge the current frame's ``cancel_handoff`` with the caller's argument.

    :return: the effective ``abandon_on_cancel`` value for the worker thread dispatch

    """
    mode = current_thread_bridge_policy().cancel_handoff
    if mode is CancelHandoff.force_abandon:
        return True
    elif mode is CancelHandoff.force_wait:
        return False
    else:
        return abandon_on_cancel


def resolve_worker_limiter(
    limiter: CapacityLimiter | None,
) -> tuple[CapacityLimiter | None, bool]:
    """
    Merge the current frame's ``limiter_bind`` with the caller's limiter.

    :return: a tuple of the limiter to pass to the backend (``None`` means the
        backend's default limiter) and a flag indicating whether limiter acquisition
        must be bypassed entirely

    """
    mode = current_thread_bridge_policy().limiter_bind
    if mode is LimiterBind.bypass_acquire:
        return None, True
    elif mode is LimiterBind.force_default:
        return None, False
    else:
        return limiter, False


#
# from_thread.run / from_thread.run_sync / from_thread.check_cancelled exit
#


def enforce_token_gate(token: EventLoopToken | None) -> None:
    """
    Apply the current frame's ``token_gate`` to a worker/external thread call.

    :param token: the event loop token explicitly passed by the caller (``None`` if
        none was passed)
    :raises ThreadBridgePolicyError: if the frame requires an explicit token and none
        was passed, or if the explicit token does not belong to the current
        portal/backend event loop

    """
    frame = current_worker_thread_bridge_policy()
    if frame is None:
        # External (unclaimed) threads carry no dispatch-time frame, so the
        # pre-existing rule (explicit token required) applies unchanged.
        return

    gate = frame.token_gate
    if gate is TokenGate.require_explicit and token is None:
        raise ThreadBridgePolicyError(
            "The current thread bridge policy frame (token_gate=require_explicit) "
            "requires an explicit event loop token; pass the host's token= to "
            "from_thread.run()/from_thread.run_sync()"
        )

    if gate is TokenGate.forbid_foreign_token and token is not None:
        home = getattr(threadlocals, "current_token", None)
        if home is not None and (
            token.backend_class is not home.backend_class
            or token.native_token is not home.native_token
        ):
            raise ThreadBridgePolicyError(
                "The current thread bridge policy frame (token_gate=forbid_foreign_"
                "token) rejected an event loop token that does not belong to the "
                "current backend event loop"
            )


#
# BlockingPortal exit
#


def resolve_portal_admission(
    root_task: object, func: Callable[..., object]
) -> Literal["ok", "reject_new", "sync_callable_only"]:
    """
    Decide in the event loop thread whether a portal may spawn a new task.

    :param root_task: the portal task tree's root task
    :param func: the callable the external thread wants to run
    :return: ``"ok"`` to admit, ``"reject_new"`` if the frame blocks all new tasks,
        or ``"sync_callable_only"`` if ``func`` is a coroutine function while only
        synchronous callables are admitted

    """
    frame = current_root_thread_bridge_policy(root_task)
    if frame.portal_admit is PortalAdmit.reject_new:
        return "reject_new"
    elif frame.portal_admit is PortalAdmit.sync_callable_only and iscoroutinefunction(
        func
    ):
        return "sync_callable_only"
    else:
        return "ok"


def _portal_allows_awaitable(root_task: object) -> bool:
    """Return whether an awaitable result may be awaited in a portal task."""
    frame = current_root_thread_bridge_policy(root_task)
    return frame.portal_admit is not PortalAdmit.sync_callable_only
