"""Wave planning for the M-step of iterative estimation pipelines.

Blocks of one wave are computed from a single snapshot of the mixture, so the
planner derives read/write dependencies between blocks automatically instead of
requiring them to be declared by the user. The result reproduces the sequential
semantics of running the blocks one by one in list order.

The planning relies on the following assumptions:

- Declared reads and writes of every strategy are truthful.
- Strategies do not modify ``state.H``, ``state.X``, or mixture weights.
- A block writes only the component it targets, ``block.component_id``.
- Strategies are deterministic.
- The set of strategies is closed: ``MaximizationStrategy`` is an enum.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Mapping, Sequence
from enum import Enum, auto
from types import MappingProxyType

from .steps.block import MaximizationStrategy, OptimizationBlock


class _Access(Enum):
    """What a maximization strategy reads from the mixture."""

    OWN_COMPONENT = auto()
    """Reads only the component it updates."""
    ALL_COMPONENTS = auto()
    """Reads every component of the mixture."""


STRATEGY_ACCESS: Mapping[MaximizationStrategy, _Access] = MappingProxyType(
    {
        MaximizationStrategy.QFUNCTION: _Access.OWN_COMPONENT,
        MaximizationStrategy.MOMENTS: _Access.OWN_COMPONENT,
        MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD: _Access.ALL_COMPONENTS,
    }
)
"""Read access of each maximization strategy. Every block writes only its own component."""


def plan_waves(blocks: Sequence[OptimizationBlock]) -> list[list[int]]:
    """Group optimization blocks into waves of mutually independent blocks.

    A block writes only its own component. A strict dependency ``a -> i`` exists
    when block ``a`` writes a component that block ``i`` reads or writes, so
    ``wave(i) > wave(a)``. A non-strict dependency exists when block ``i`` writes
    a component that block ``a`` reads, so ``wave(i) >= wave(a)``. The wave of a
    block is the maximum over these constraints, or 0 if there are none. Blocks
    of one wave see the same mixture snapshot, which reproduces the result of
    running the blocks sequentially in list order.

    Parameters
    ----------
    blocks : Sequence[OptimizationBlock]
        Optimization tasks in sequential execution order.

    Returns
    -------
    list[list[int]]
        Block indices grouped by wave in increasing wave order; indices inside
        a wave follow the order of ``blocks``. An empty input gives ``[]``.
    """

    wave_of = [0] * len(blocks)
    for i, block_i in enumerate(blocks):
        access_i = STRATEGY_ACCESS[block_i.maximization_strategy]
        for a in range(i):
            block_a = blocks[a]
            access_a = STRATEGY_ACCESS[block_a.maximization_strategy]
            same_component = block_a.component_id == block_i.component_id
            # Strict edge: block i reads or writes what block a writes.
            # Block a always writes block_a.component_id; block i reads all
            # components or only block_i.component_id, and always writes
            # block_i.component_id.
            if access_i is _Access.ALL_COMPONENTS or same_component:
                wave_of[i] = max(wave_of[i], wave_of[a] + 1)
            # Non-strict edge: block i writes what block a reads. Block a reads
            # all components or only block_a.component_id; same_component is
            # impossible here because it already implied the strict edge.
            elif access_a is _Access.ALL_COMPONENTS:
                wave_of[i] = max(wave_of[i], wave_of[a])

    waves: list[list[int]] = []
    for index, wave in enumerate(wave_of):
        if wave == len(waves):
            waves.append([index])
        else:
            waves[wave].append(index)
    return waves
