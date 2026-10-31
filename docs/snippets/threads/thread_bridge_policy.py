import time

from anyio import run, to_thread
from anyio.thread_policies import CancelHandoff, thread_bridge_policy


async def main() -> None:
    # Every to_thread.run_sync() call in this block abandons the worker thread when
    # the host task is cancelled, regardless of the arguments passed by callers.
    with thread_bridge_policy(cancel_handoff=CancelHandoff.force_abandon):
        await to_thread.run_sync(time.sleep, 5, abandon_on_cancel=False)


run(main)
