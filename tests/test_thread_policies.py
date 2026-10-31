from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Coroutine, Iterator
from concurrent.futures import Future
from contextlib import asynccontextmanager, contextmanager, suppress
from functools import partial
from typing import Any

import pytest

from anyio import (
    CapacityLimiter,
    Event,
    create_task_group,
    from_thread,
    get_cancelled_exc_class,
    sleep_forever,
    to_thread,
    wait_all_tasks_blocked,
)
from anyio._core._eventloop import get_async_backend
from anyio.from_thread import start_blocking_portal
from anyio.lowlevel import EventLoopToken, current_token
from anyio.thread_policies import (
    CancelHandoff,
    LimiterBind,
    PortalAdmit,
    ThreadBridgePolicy,
    ThreadBridgePolicyError,
    TokenGate,
    current_thread_bridge_policy,
    current_worker_thread_bridge_policy,
    thread_bridge_policy,
)

#
# Stack mechanics
#


async def test_empty_stack_is_pin_everywhere() -> None:
    frame = current_thread_bridge_policy()
    assert frame == ThreadBridgePolicy()
    assert frame.cancel_handoff is CancelHandoff.pin
    assert frame.limiter_bind is LimiterBind.pin
    assert frame.portal_admit is PortalAdmit.pin
    assert frame.token_gate is TokenGate.pin
    # Baseline dispatch behavior is unchanged when no frame has ever been pushed.
    assert await to_thread.run_sync(lambda: 7) == 7


async def test_nested_frames_pop_back_to_outer_frame() -> None:
    with thread_bridge_policy(cancel_handoff=CancelHandoff.force_wait):
        assert current_thread_bridge_policy().cancel_handoff is CancelHandoff.force_wait
        with thread_bridge_policy(
            cancel_handoff=CancelHandoff.force_abandon,
            token_gate=TokenGate.require_explicit,
        ):
            frame = current_thread_bridge_policy()
            assert frame.cancel_handoff is CancelHandoff.force_abandon
            assert frame.token_gate is TokenGate.require_explicit
            # Dimensions not overridden by the inner frame inherit the outer frame.
            assert frame.limiter_bind is LimiterBind.pin

        frame = current_thread_bridge_policy()
        assert frame.cancel_handoff is CancelHandoff.force_wait
        assert frame.token_gate is TokenGate.pin

    assert current_thread_bridge_policy() == ThreadBridgePolicy()


async def test_frame_popped_on_exception() -> None:
    with pytest.raises(ValueError, match="boom"):
        with thread_bridge_policy(portal_admit=PortalAdmit.reject_new):
            raise ValueError("boom")

    assert current_thread_bridge_policy() == ThreadBridgePolicy()


async def test_concurrent_tasks_do_not_share_frames() -> None:
    """Two concurrent host tasks pushing different frames must not cross stacks."""
    results: dict[str, object] = {}

    def read_dispatched_handoff() -> CancelHandoff:
        frame = current_worker_thread_bridge_policy()
        assert frame is not None
        return frame.cancel_handoff

    async def child(
        mode: CancelHandoff,
        entered: Event,
        other_entered: Event,
        release: Event,
        key: str,
    ) -> None:
        with thread_bridge_policy(cancel_handoff=mode):
            entered.set()
            await other_entered.wait()
            # Read the effective frame from a freshly dispatched worker thread too,
            # so this exercises the dispatch snapshot rather than the host context.
            results[key] = await to_thread.run_sync(read_dispatched_handoff)
            await release.wait()

    a_entered = Event()
    b_entered = Event()
    release = Event()
    async with create_task_group() as tg:
        tg.start_soon(
            child,
            CancelHandoff.force_abandon,
            a_entered,
            b_entered,
            release,
            "a",
        )
        tg.start_soon(
            child,
            CancelHandoff.force_wait,
            b_entered,
            a_entered,
            release,
            "b",
        )
        await a_entered.wait()
        await b_entered.wait()
        # Both frames are active concurrently at this point.
        release.set()

    assert results == {
        "a": CancelHandoff.force_abandon,
        "b": CancelHandoff.force_wait,
    }


#
# to_thread.run_sync exit: cancel_handoff
#


class TestCancelHandoff:
    @pytest.mark.parametrize("abandon_on_cancel", [False, True])
    async def test_force_abandon_overrides_caller_argument(
        self, abandon_on_cancel: bool
    ) -> None:
        last_active: str | None = None

        def thread_worker() -> None:
            nonlocal last_active
            from_thread.run_sync(sleep_event.set)
            time.sleep(0.2)
            last_active = "thread"
            from_thread.run_sync(finish_event.set)

        async def task_worker() -> None:
            nonlocal last_active
            try:
                with thread_bridge_policy(cancel_handoff=CancelHandoff.force_abandon):
                    await to_thread.run_sync(
                        thread_worker, abandon_on_cancel=abandon_on_cancel
                    )
            finally:
                last_active = "task"

        sleep_event = Event()
        finish_event = Event()
        async with create_task_group() as tg:
            tg.start_soon(task_worker)
            await sleep_event.wait()
            tg.cancel_scope.cancel()

        await finish_event.wait()
        # The frame forced abandonment even though the caller passed False.
        assert last_active == "thread"

    @pytest.mark.parametrize("abandon_on_cancel", [False, True])
    async def test_force_wait_overrides_caller_argument(
        self, abandon_on_cancel: bool
    ) -> None:
        last_active: str | None = None

        def thread_worker() -> None:
            nonlocal last_active
            from_thread.run_sync(sleep_event.set)
            time.sleep(0.2)
            last_active = "thread"
            from_thread.run_sync(finish_event.set)

        async def task_worker() -> None:
            nonlocal last_active
            try:
                with thread_bridge_policy(cancel_handoff=CancelHandoff.force_wait):
                    await to_thread.run_sync(
                        thread_worker, abandon_on_cancel=abandon_on_cancel
                    )
            finally:
                last_active = "task"

        sleep_event = Event()
        finish_event = Event()
        async with create_task_group() as tg:
            tg.start_soon(task_worker)
            await sleep_event.wait()
            tg.cancel_scope.cancel()

        await finish_event.wait()
        # The frame forced waiting even though the caller passed True.
        assert last_active == "task"

    async def test_force_abandon_not_faked_via_deprecated_alias(self) -> None:
        """force_abandon must win over an explicit deprecated cancellable=False."""
        released = threading.Event()

        def thread_worker() -> None:
            from_thread.run_sync(started.set)
            released.wait(5)

        started = Event()

        async def task_worker() -> None:
            with thread_bridge_policy(cancel_handoff=CancelHandoff.force_abandon):
                with pytest.warns(DeprecationWarning, match="`cancellable=`"):
                    await to_thread.run_sync(thread_worker, cancellable=False)

        async with create_task_group() as tg:
            tg.start_soon(task_worker)
            await started.wait()
            tg.cancel_scope.cancel()
            released.set()

    async def test_force_wait_keeps_cancellation_observable_in_worker(self) -> None:
        """check_cancelled() still raises in the worker under a force_wait frame."""
        check_future: Future[None] = Future()
        returned: dict[str, str] = {}

        def thread_worker() -> str:
            from_thread.run_sync(started.set)
            release.wait(5)
            try:
                from_thread.check_cancelled()
            except BaseException as exc:
                check_future.set_exception(exc)
            else:
                check_future.set_result(None)

            return "finished"

        async def task_worker() -> None:
            with thread_bridge_policy(cancel_handoff=CancelHandoff.force_wait):
                # The caller asks for abandonment, but the frame forces waiting.
                returned["value"] = await to_thread.run_sync(
                    thread_worker, abandon_on_cancel=True
                )

        started = Event()
        release = threading.Event()
        async with create_task_group() as tg:
            tg.start_soon(task_worker)
            await started.wait()
            tg.cancel_scope.cancel()
            release.set()

        assert returned["value"] == "finished"
        with pytest.raises(get_cancelled_exc_class()):
            check_future.result(1)

    async def test_late_push_is_void_for_inflight_dispatch(self) -> None:
        """
        A frame pushed after the worker thread has started executing must not change
        the dispatch-time snapshot of the worker already in flight.

        """
        inside = Event()
        pushed = Event()
        release = threading.Event()

        def worker() -> str:
            from_thread.run_sync(inside.set)
            release.wait(5)
            # The sibling's late require_explicit frame must not reach this worker;
            # its implicit-token hop must still work against the pinned snapshot.
            frame = current_worker_thread_bridge_policy()
            assert frame is not None
            assert frame.token_gate is TokenGate.pin
            assert from_thread.run_sync(lambda: "probed") == "probed"
            return "done"

        async def late_pusher() -> None:
            await inside.wait()
            with thread_bridge_policy(token_gate=TokenGate.require_explicit):
                pushed.set()
                await Event().wait()

        async def driver() -> None:
            assert await to_thread.run_sync(worker) == "done"

        async with create_task_group() as tg:
            tg.start_soon(late_pusher)
            tg.start_soon(driver)
            await pushed.wait()
            release.set()
            await wait_all_tasks_blocked()
            tg.cancel_scope.cancel()


#
# to_thread.run_sync exit: limiter_bind
#


class TestLimiterBind:
    @asynccontextmanager
    async def one_token_default_limiter(self) -> AsyncIterator[CapacityLimiter]:
        limiter = to_thread.current_default_thread_limiter()
        old_total = limiter.total_tokens
        limiter.total_tokens = 1
        try:
            yield limiter
        finally:
            limiter.total_tokens = old_total

    async def test_force_default_ignores_caller_limiter(self) -> None:
        async with self.one_token_default_limiter() as one_token_default_limiter:
            release = threading.Event()

            def holder() -> None:
                from_thread.run_sync(holder_started.set)
                release.wait(5)

            holder_started = Event()
            custom_limiter = CapacityLimiter(1)
            blocked_done = Event()

            async def blocked_dispatch() -> None:
                with thread_bridge_policy(limiter_bind=LimiterBind.force_default):
                    # The free custom limiter must be ignored in favor of the
                    # exhausted default limiter.
                    await to_thread.run_sync(lambda: None, limiter=custom_limiter)

                blocked_done.set()

            async with create_task_group() as tg:
                tg.start_soon(to_thread.run_sync, holder)
                await holder_started.wait()
                assert one_token_default_limiter.borrowed_tokens == 1
                assert custom_limiter.borrowed_tokens == 0
                tg.start_soon(blocked_dispatch)
                await wait_all_tasks_blocked()
                assert not blocked_done.is_set()
                assert custom_limiter.borrowed_tokens == 0
                release.set()
                await blocked_done.wait()

            assert custom_limiter.borrowed_tokens == 0

    async def test_bypass_acquire_skips_limiter_tokens(self) -> None:
        async with self.one_token_default_limiter() as one_token_default_limiter:
            release = threading.Event()

            def holder() -> None:
                from_thread.run_sync(holder_started.set)
                release.wait(5)

            holder_started = Event()
            custom_limiter = CapacityLimiter(1)
            collected: dict[str, str] = {}
            bypass_done = Event()

            async def bypass_dispatch() -> None:
                with thread_bridge_policy(limiter_bind=LimiterBind.bypass_acquire):
                    collected["value"] = await to_thread.run_sync(
                        lambda: "bypassed", limiter=custom_limiter
                    )

                bypass_done.set()

            async with create_task_group() as tg:
                tg.start_soon(to_thread.run_sync, holder)
                await holder_started.wait()
                assert one_token_default_limiter.borrowed_tokens == 1
                # The bypassing dispatch runs to completion while the default
                # limiter is exhausted and borrows neither limiter's tokens.
                tg.start_soon(bypass_dispatch)
                await bypass_done.wait()
                assert collected["value"] == "bypassed"
                assert one_token_default_limiter.borrowed_tokens == 1
                assert custom_limiter.borrowed_tokens == 0
                release.set()

    async def test_pin_respects_caller_limiter(self) -> None:
        async with self.one_token_default_limiter() as one_token_default_limiter:
            release = threading.Event()

            def holder() -> None:
                from_thread.run_sync(holder_started.set)
                release.wait(5)

            holder_started = Event()
            custom_limiter = CapacityLimiter(1)
            done = Event()

            def set_done() -> None:
                from_thread.run_sync(done.set)

            async with create_task_group() as tg:
                tg.start_soon(to_thread.run_sync, holder)
                await holder_started.wait()
                assert one_token_default_limiter.borrowed_tokens == 1
                tg.start_soon(
                    partial(to_thread.run_sync, set_done, limiter=custom_limiter)
                )
                await done.wait()
                release.set()

            assert custom_limiter.borrowed_tokens == 0


#
# from_thread.run / run_sync / check_cancelled exit: token_gate
#


class TestTokenGate:
    async def test_require_explicit_rejects_implicit_worker_token(self) -> None:
        def worker() -> str:
            with pytest.raises(ThreadBridgePolicyError, match="require_explicit"):
                from_thread.run_sync(lambda: None)

            with pytest.raises(ThreadBridgePolicyError, match="require_explicit"):
                from_thread.run(sleep_forever)

            return "ok"

        with thread_bridge_policy(token_gate=TokenGate.require_explicit):
            assert await to_thread.run_sync(worker) == "ok"

    async def test_require_explicit_accepts_explicit_token(self) -> None:
        host_token = current_token()

        def worker() -> int:
            return from_thread.run_sync(lambda: 11, token=host_token)

        with thread_bridge_policy(token_gate=TokenGate.require_explicit):
            assert await to_thread.run_sync(worker) == 11

    async def test_check_cancelled_keeps_working_under_require_explicit(self) -> None:
        def worker() -> str:
            from_thread.check_cancelled()
            return "ok"

        with thread_bridge_policy(token_gate=TokenGate.require_explicit):
            assert await to_thread.run_sync(worker) == "ok"

    async def test_forbid_foreign_token_rejects_foreign_loop(self) -> None:
        host_token = current_token()
        with _foreign_event_loop_token() as foreign_token:

            def worker() -> None:
                # The token for a different backend event loop is rejected.
                with pytest.raises(ThreadBridgePolicyError, match="forbid_foreign"):
                    from_thread.run_sync(lambda: None, token=foreign_token)

                # The token for the home loop is accepted.
                assert from_thread.run_sync(lambda: 5, token=host_token) == 5

            with thread_bridge_policy(token_gate=TokenGate.forbid_foreign_token):
                await to_thread.run_sync(worker)

    async def test_pin_accepts_implicit_worker_token(self) -> None:
        def worker() -> int:
            return from_thread.run_sync(lambda: 3)

        assert await to_thread.run_sync(worker) == 3


@contextmanager
def _foreign_event_loop_token() -> Iterator[EventLoopToken]:
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    assert ready.wait(5)
    backend_class = get_async_backend("asyncio")
    try:
        yield EventLoopToken(backend_class, loop)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()


#
# BlockingPortal exit: portal_admit
#


class TestPortalAdmit:
    def test_reject_new_blocks_every_entry(
        self, anyio_backend_name: str, anyio_backend_options: dict[str, Any]
    ) -> None:
        with start_blocking_portal(anyio_backend_name, anyio_backend_options) as portal:
            ready = threading.Event()
            popped = threading.Event()
            controller_future = portal.start_task_soon(
                _reject_controller, ready, popped
            )
            assert ready.wait(5)

            try:
                with pytest.raises(ThreadBridgePolicyError, match="reject_new"):
                    portal.call(lambda: 1)

                with pytest.raises(ThreadBridgePolicyError, match="reject_new"):
                    portal.start_task_soon(lambda: None)

                async def async_worker() -> None:
                    return None

                with pytest.raises(ThreadBridgePolicyError, match="reject_new"):
                    portal.start_task(async_worker)

                class AsyncCM:
                    async def __aenter__(self) -> int:
                        return 1

                    async def __aexit__(self, *exc_info: object) -> None:
                        return None

                with pytest.raises(ThreadBridgePolicyError, match="reject_new"):
                    with portal.wrap_async_context_manager(AsyncCM()):
                        pass
            finally:
                _cancel_controller(controller_future, popped)

            # After the frame is popped, the portal admits again.
            assert portal.call(lambda: 9) == 9

    def test_reject_new_does_not_touch_already_spawned_task(
        self, anyio_backend_name: str, anyio_backend_options: dict[str, Any]
    ) -> None:
        with start_blocking_portal(anyio_backend_name, anyio_backend_options) as portal:
            release_event = portal.call(Event)
            entered = threading.Event()

            async def long_running() -> str:
                entered.set()
                await release_event.wait()
                return "done"

            future = portal.start_task_soon(long_running)
            assert entered.wait(5)

            ready = threading.Event()
            popped = threading.Event()
            controller_future = portal.start_task_soon(
                _reject_controller, ready, popped
            )
            assert ready.wait(5)
            try:
                with pytest.raises(ThreadBridgePolicyError):
                    portal.call(lambda: None)

                # The task spawned before the frame was pushed is unaffected.
                assert not future.done()
                # Release it through a direct token hop rather than a portal
                # admission, since the frame currently rejects new portal calls.
                from_thread.run_sync(release_event.set, token=portal._token)
                assert future.result(5) == "done"
            finally:
                _cancel_controller(controller_future, popped)

            assert portal.call(lambda: 1) == 1

    def test_sync_callable_only(
        self, anyio_backend_name: str, anyio_backend_options: dict[str, Any]
    ) -> None:
        with start_blocking_portal(anyio_backend_name, anyio_backend_options) as portal:
            ready = threading.Event()
            popped = threading.Event()
            controller_future = portal.start_task_soon(
                _sync_only_controller, ready, popped
            )
            assert ready.wait(5)
            try:
                # Plain synchronous callables are admitted.
                assert portal.call(lambda: 4) == 4

                async def async_worker() -> int:
                    return 4

                with pytest.raises(ThreadBridgePolicyError, match="sync_callable_only"):
                    portal.call(async_worker)

                with pytest.raises(ThreadBridgePolicyError, match="sync_callable_only"):
                    portal.start_task_soon(async_worker)

                with pytest.raises(ThreadBridgePolicyError, match="sync_callable_only"):
                    portal.start_task(async_worker)

                # A synchronous callable that returns an awaitable is rejected at
                # execution time, before the awaitable is awaited.
                def returns_awaitable() -> Coroutine[Any, Any, int]:
                    async def coro() -> int:
                        return 1

                    return coro()

                with pytest.raises(ThreadBridgePolicyError, match="sync_callable_only"):
                    portal.call(returns_awaitable)
            finally:
                _cancel_controller(controller_future, popped)

            assert portal.call(async_worker) == 4

    def test_frames_do_not_cross_between_portals(
        self, anyio_backend_name: str, anyio_backend_options: dict[str, Any]
    ) -> None:
        with start_blocking_portal(
            anyio_backend_name, anyio_backend_options
        ) as portal_a:
            with start_blocking_portal(
                anyio_backend_name, anyio_backend_options
            ) as portal_b:
                ready = threading.Event()
                popped = threading.Event()
                controller_future = portal_a.start_task_soon(
                    _reject_controller, ready, popped
                )
                assert ready.wait(5)
                try:
                    with pytest.raises(ThreadBridgePolicyError):
                        portal_a.call(lambda: None)

                    # The other portal's task tree is a different root and keeps
                    # admitting calls.
                    assert portal_b.call(lambda: "b") == "b"
                finally:
                    _cancel_controller(controller_future, popped)

                assert portal_a.call(lambda: "a") == "a"


def _cancel_controller(controller_future: Future[Any], popped: threading.Event) -> None:
    """Cancel a controller task and wait until its policy frame has been popped."""
    controller_future.cancel()
    assert popped.wait(5)
    # Drain the future so its outcome cannot surprise portal shutdown.
    with suppress(Exception):
        controller_future.result(1)


async def _reject_controller(ready: threading.Event, popped: threading.Event) -> None:
    try:
        with thread_bridge_policy(portal_admit=PortalAdmit.reject_new):
            ready.set()
            await sleep_forever()
    finally:
        # Signal only after the policy frame has actually been popped.
        popped.set()


async def _sync_only_controller(
    ready: threading.Event, popped: threading.Event
) -> None:
    try:
        with thread_bridge_policy(portal_admit=PortalAdmit.sync_callable_only):
            ready.set()
            await sleep_forever()
    finally:
        popped.set()
