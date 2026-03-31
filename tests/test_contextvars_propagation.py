"""Diagnostic: does ContextVar survive LangGraph's tool dispatch?"""
import asyncio
from contextvars import ContextVar

_cv: ContextVar[str | None] = ContextVar("_cv", default=None)

async def test_contextvar_survives_direct_await():
    """If tool is directly awaited (not create_task), ContextVar propagates."""
    _cv.set("active")

    async def tool_like():
        return _cv.get()

    result = await tool_like()
    assert result == "active", f"Direct await: got {result!r}"


async def test_contextvar_lost_in_create_task():
    """If tool is spawned as a new Task, ContextVar is *copied* at creation time."""
    _cv.set("active")

    async def tool_like():
        return _cv.get()

    # create_task copies context at creation time — so "active" IS visible
    result = await asyncio.create_task(tool_like())
    assert result == "active", f"create_task (created after set): got {result!r}"


async def test_contextvar_lost_when_set_after_task_created():
    """If the task is already running when we set the ContextVar, it's lost."""
    _cv.set(None)

    results = {}
    ready = asyncio.Event()
    go = asyncio.Event()

    async def tool_like():
        ready.set()
        await go.wait()
        results["val"] = _cv.get()

    task = asyncio.create_task(tool_like())
    await ready.wait()
    # Set AFTER task already started
    _cv.set("too_late")
    go.set()
    await task
    # The task sees the value at creation time (None), not the updated one
    assert results["val"] is None, f"Set after task start: got {results['val']!r}"


if __name__ == "__main__":
    asyncio.run(test_contextvar_survives_direct_await())
    print("✓ Direct await: ContextVar propagates")
    asyncio.run(test_contextvar_lost_in_create_task())
    print("✓ create_task (set before): ContextVar propagates")
    asyncio.run(test_contextvar_lost_when_set_after_task_created())
    print("✓ create_task (set after): ContextVar lost")
