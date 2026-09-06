"""Tests for bartorch.core.context (BartContext, bart_context)."""

from __future__ import annotations

import pytest
import torch

from bartorch.core.context import BartContext, bart_context

__all__: list[str] = []


def test_not_active_by_default():
    assert not BartContext.is_active()
    assert BartContext.current() is None


def test_active_inside_with():
    ctx = BartContext()
    with ctx:
        assert BartContext.is_active()
        assert BartContext.current() is ctx


def test_not_active_after_with():
    ctx = BartContext()
    with ctx:
        pass
    assert not BartContext.is_active()


def test_fresh_name_unique():
    ctx = BartContext()
    names = {ctx.fresh_name() for _ in range(100)}
    assert len(names) == 100


def test_fresh_name_mem_suffix():
    ctx = BartContext()
    assert ctx.fresh_name().endswith(".mem")


def test_register_unregister():
    ctx = BartContext()
    t = torch.zeros(1)
    ctx.register("foo.mem", t)
    assert "foo.mem" in ctx._registered
    ctx.unregister("foo.mem")
    assert "foo.mem" not in ctx._registered


def test_bart_context_yields_context():
    with bart_context() as ctx:
        assert isinstance(ctx, BartContext)
        assert BartContext.is_active()


def test_bart_context_cleanup_on_exit():
    with bart_context():
        pass
    assert not BartContext.is_active()
