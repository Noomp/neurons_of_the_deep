"""Write-sparse slot memory: the "GRU that decides what to keep".

A standard GRU rewrites its whole state at every step,
``h_t = (1 - z_t) * h_{t-1} + z_t * h~_t``, with ``z_t`` a vector gate. Here the
state *is* the memory and should rarely change, so the update gate is
factorised into

* a scalar write gate ``g_t`` ("is this view worth saving?"), biased towards 0;
* a softmax address ``w_t`` over K slots ("which slot does it replace?");

and the candidate ``h~_t`` is simply the encoded view ``F_t``:

    e_t[k]  = g_t * w_t[k]
    M_t[k]  = (1 - e_t[k]) * M_{t-1}[k] + e_t[k] * F_t

When ``g_t`` is 0 the memory is copied unchanged and its gradient passes
through untouched (dM_t / dM_{t-1} = I), so nothing decays between writes.
There is no reset gate: the current view is always available to the reader
next to the memory, so the memory only needs what is no longer visible.

Each slot also keeps two soft counters used by the reader and the address:
``age`` (steps since the slot was written) and ``fill`` (how much real content
it holds; 0 for a slot never written).

See report section 4 for the comparison with a GRU and the alternatives.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MemoryState:
    """Memory of B sequences.

    Attributes:
        slots: (B, K, P, d) stored view features.
        age: (B, K) steps since each slot was written, as seen by the next read.
        fill: (B, K) in [0, 1], how much of each slot holds written content.
    """

    slots: torch.Tensor
    age: torch.Tensor
    fill: torch.Tensor

    @classmethod
    def empty(cls, batch_size: int, num_slots: int, num_cells: int, dim: int, device=None) -> MemoryState:
        return cls(
            slots=torch.zeros(batch_size, num_slots, num_cells, dim, device=device),
            age=torch.zeros(batch_size, num_slots, device=device),
            fill=torch.zeros(batch_size, num_slots, device=device),
        )

    def reset(self, where: torch.Tensor) -> MemoryState:
        """Clear the sequences where ``where`` (B,) bool is True (new episodes)."""
        keep = (~where).float()
        return MemoryState(self.slots * keep[:, None, None, None], self.age * keep[:, None], self.fill * keep[:, None])

    def detach(self) -> MemoryState:
        """Cut the gradient (truncated back-propagation through time)."""
        return MemoryState(self.slots.detach(), self.age.detach(), self.fill.detach())

    def select(self, rows: torch.Tensor | list[int]) -> MemoryState:
        rows = torch.as_tensor(rows, dtype=torch.long, device=self.slots.device)
        return MemoryState(self.slots[rows], self.age[rows], self.fill[rows])

    def to(self, device) -> MemoryState:
        return MemoryState(self.slots.to(device), self.age.to(device), self.fill.to(device))


@dataclass
class WriteInfo:
    gate: torch.Tensor  # (B,) write gate g_t
    address: torch.Tensor  # (B, K) slot address w_t


class SlotMemory(nn.Module):
    """Decides whether and where to write the current view into memory.

    Args:
        dim: feature width d of each cell.
        mission_dim: width of the mission vector (what is worth saving depends on the goal).
        num_slots: K. With K = 0 the memory is disabled and ``write`` is a no-op.
        gate_bias: initial bias of the write gate; negative means "rarely write".
        write_mode: "soft" (g in (0, 1)) or "hard" (g in {0, 1} in the forward
            pass, straight-through gradient in the backward pass).
    """

    def __init__(self, dim: int, mission_dim: int, num_slots: int, gate_bias: float = -2.0, write_mode: str = "soft"):
        super().__init__()
        if write_mode not in ("soft", "hard"):
            raise ValueError(f"write_mode must be 'soft' or 'hard', got {write_mode!r}")
        self.num_slots = num_slots
        self.write_mode = write_mode
        # Gate input: [view summary, memory summary, mission, time encoding of t].
        self.gate = nn.Linear(3 * dim + mission_dim, 1)
        nn.init.constant_(self.gate.bias, gate_bias)
        # Address input per slot: [view summary, slot summary, log(1 + age), fill].
        self.address = nn.Sequential(nn.Linear(2 * dim + 2, dim), nn.ReLU(), nn.Linear(dim, 1))

    def write(
        self, memory: MemoryState, view: torch.Tensor, mission: torch.Tensor, t_enc: torch.Tensor
    ) -> tuple[MemoryState, WriteInfo]:
        """Write the current view into memory.

        Args:
            memory: the memory before this step (the one the reader just used).
            view: (B, P, d) features of the current view.
            mission: (B, mission_dim).
            t_enc: (B, d) time encoding of the current step.

        Returns:
            The new memory and the gate and address used.
        """
        b = view.shape[0]
        if self.num_slots == 0:
            return memory, WriteInfo(view.new_zeros(b), view.new_zeros(b, 0))

        view_summary = view.mean(dim=1)  # (B, d)
        slot_summary = memory.slots.mean(dim=2)  # (B, K, d)
        weights = memory.fill / memory.fill.sum(dim=1, keepdim=True).clamp_min(1e-6)
        memory_summary = (weights[..., None] * slot_summary).sum(dim=1)

        gate = torch.sigmoid(self.gate(torch.cat([view_summary, memory_summary, mission, t_enc], dim=-1))).squeeze(-1)
        if self.write_mode == "hard":
            gate = (gate > 0.5).float() + gate - gate.detach()  # straight-through estimator

        per_slot = torch.cat(
            [
                view_summary[:, None].expand(-1, self.num_slots, -1),
                slot_summary,
                torch.log1p(memory.age)[..., None],
                memory.fill[..., None],
            ],
            dim=-1,
        )
        address = torch.softmax(self.address(per_slot).squeeze(-1), dim=-1)  # (B, K)

        erase = gate[:, None] * address  # (B, K)
        e = erase[..., None, None]
        new = MemoryState(
            slots=(1 - e) * memory.slots + e * view[:, None],
            age=1 + (1 - erase) * memory.age,  # a fresh write is 1 step old at the next read
            fill=memory.fill + erase * (1 - memory.fill),
        )
        return new, WriteInfo(gate=gate, address=address)
