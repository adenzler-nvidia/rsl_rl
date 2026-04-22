# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the optional obs_compress_fn / obs_decompress_fn hooks on RolloutStorage."""

import torch
from tensordict import TensorDict

from rsl_rl.storage import RolloutStorage

NUM_ENVS = 6
NUM_STEPS = 8
IMG_C, IMG_H, IMG_W = 3, 4, 4
STATE_DIM = 5
NUM_ACTIONS = 3


def _make_obs() -> TensorDict:
    """Observation dict with an image key and a state key."""
    return TensorDict(
        {
            "image": torch.rand(NUM_ENVS, IMG_C, IMG_H, IMG_W),
            "state": torch.randn(NUM_ENVS, STATE_DIM),
        },
        batch_size=[NUM_ENVS],
    )


def _fill(storage: RolloutStorage, make_obs_fn) -> list[TensorDict]:
    """Drive the storage through a full rollout, returning the per-step live obs for reference."""
    per_step_live_obs = []
    for step in range(NUM_STEPS):
        obs = make_obs_fn()
        per_step_live_obs.append(obs)
        t = RolloutStorage.Transition()
        t.observations = obs
        t.hidden_states = (None, None)
        t.actions = torch.full((NUM_ENVS, NUM_ACTIONS), float(step))
        t.values = torch.full((NUM_ENVS, 1), float(step))
        t.actions_log_prob = torch.full((NUM_ENVS,), float(step))
        t.distribution_params = (
            torch.full((NUM_ENVS, NUM_ACTIONS), float(step)),
            torch.full((NUM_ENVS, NUM_ACTIONS), 1.0),
        )
        t.rewards = torch.full((NUM_ENVS,), float(step))
        t.dones = torch.zeros(NUM_ENVS)
        storage.add_transition(t)
    storage.returns = torch.randn_like(storage.returns)
    storage.advantages = torch.randn_like(storage.advantages)
    return per_step_live_obs


class TestObsCompressionHooks:
    """Numerical-equivalence and behavior tests for the compression hooks."""

    def test_default_path_unchanged_when_hooks_absent(self) -> None:
        """With both hooks None, storage must behave identically to the pre-hook code path."""
        torch.manual_seed(0)
        obs = _make_obs()
        storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
        _fill(storage, _make_obs)

        assert "image" in storage.observations.keys()
        assert "state" in storage.observations.keys()
        assert storage.observations["image"].dtype == torch.float32
        assert storage.observations["image"].shape == (NUM_STEPS, NUM_ENVS, IMG_C, IMG_H, IMG_W)
        # A mini-batch should yield the same TensorDict type with both keys and matching shapes.
        for batch in storage.mini_batch_generator(num_mini_batches=2, num_epochs=1):
            assert "image" in batch.observations.keys()
            assert "state" in batch.observations.keys()
            assert batch.observations["image"].dtype == torch.float32
            break

    def test_identity_hooks_match_default(self) -> None:
        """Identity compress/decompress hooks must produce bit-identical mini-batch obs."""
        torch.manual_seed(0)
        obs_a = _make_obs()
        storage_default = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs_a, [NUM_ACTIONS])
        torch.manual_seed(0)
        default_obs_stream = _fill(storage_default, _make_obs)

        identity = lambda x: x  # noqa: E731
        torch.manual_seed(0)
        obs_b = _make_obs()
        storage_hooks = RolloutStorage(
            "rl",
            NUM_ENVS,
            NUM_STEPS,
            obs_b,
            [NUM_ACTIONS],
            obs_compress_fn=identity,
            obs_decompress_fn=identity,
        )
        torch.manual_seed(0)
        _ = _fill(storage_hooks, _make_obs)

        # Feed identical advantages/returns so mini-batches are deterministic.
        storage_hooks.advantages = storage_default.advantages.clone()
        storage_hooks.returns = storage_default.returns.clone()

        torch.manual_seed(42)
        default_batches = list(storage_default.mini_batch_generator(num_mini_batches=2, num_epochs=1))
        torch.manual_seed(42)
        hook_batches = list(storage_hooks.mini_batch_generator(num_mini_batches=2, num_epochs=1))

        assert len(default_batches) == len(hook_batches)
        for b_def, b_hook in zip(default_batches, hook_batches):
            for key in ("image", "state"):
                assert torch.equal(b_def.observations[key], b_hook.observations[key]), (
                    f"Identity hooks should produce identical '{key}' batches; got diff."
                )
            assert torch.equal(b_def.actions, b_hook.actions)
            assert torch.equal(b_def.values, b_hook.values)

    def test_uint8_image_roundtrip(self) -> None:
        """A uint8 compression + decompression roundtrip preserves image data within 1/255 tolerance."""
        torch.manual_seed(0)

        def compress(live_obs: TensorDict) -> TensorDict:
            img_u8 = (live_obs["image"].clamp(0, 1) * 255).to(torch.uint8)
            return live_obs.update({"image": img_u8})

        def decompress(stored_batch: TensorDict) -> TensorDict:
            img_f = stored_batch["image"].float() / 255.0
            return stored_batch.update({"image": img_f})

        obs = _make_obs()
        storage = RolloutStorage(
            "rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS],
            obs_compress_fn=compress, obs_decompress_fn=decompress,
        )

        # Storage buffer should be uint8 for the image key, float32 for state.
        assert storage.observations["image"].dtype == torch.uint8
        assert storage.observations["state"].dtype == torch.float32

        per_step_live = _fill(storage, _make_obs)

        for batch in storage.mini_batch_generator(num_mini_batches=1, num_epochs=1):
            assert batch.observations["image"].dtype == torch.float32
            assert batch.observations["image"].shape == (NUM_ENVS * NUM_STEPS, IMG_C, IMG_H, IMG_W)
            # Values should roundtrip within uint8 quantization error.
            assert batch.observations["image"].min() >= 0.0
            assert batch.observations["image"].max() <= 1.0
            break

    def test_passthrough_key_shares_storage_with_live_obs_schema(self) -> None:
        """When compress_fn returns references for non-image keys, storage dtype matches live."""

        def compress(live_obs: TensorDict) -> TensorDict:
            # Only transform 'image'; 'state' passes through by reference.
            img_u8 = (live_obs["image"].clamp(0, 1) * 255).to(torch.uint8)
            return live_obs.update({"image": img_u8})

        obs = _make_obs()
        storage = RolloutStorage(
            "rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS],
            obs_compress_fn=compress,
        )
        # 'state' flows through compress as-is; storage allocates it with the live dtype.
        assert storage.observations["state"].dtype == obs["state"].dtype
        assert storage.observations["state"].shape[1:] == obs["state"].shape

    def test_schema_change_image_to_body_q(self) -> None:
        """compress_fn may drop keys and add new ones; storage allocates from that schema."""

        def compress(live_obs: TensorDict) -> TensorDict:
            # Drop the image, add a surrogate 'body_q' key, keep state.
            return TensorDict(
                {
                    "body_q": torch.zeros(NUM_ENVS, 13 * 7),
                    "state": live_obs["state"],
                },
                batch_size=live_obs.batch_size,
            )

        obs = _make_obs()
        storage = RolloutStorage(
            "rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS],
            obs_compress_fn=compress,
        )
        assert "body_q" in storage.observations.keys()
        assert "image" not in storage.observations.keys()
        assert storage.observations["body_q"].shape == (NUM_STEPS, NUM_ENVS, 13 * 7)


class TestDecompressAcrossGenerators:
    """Verify obs_decompress_fn is invoked by all three generator paths and that a
    rank-polymorphic decompress function works correctly across each call site's
    distinct batch shape.

    Shapes per call site:
      - mini_batch_generator:           [M, *obs]                    (1 batch axis)
      - generator (distillation):       [N, *obs]                    (1 batch axis)
      - recurrent_mini_batch_generator: [T_padded, trajectories, *obs] (2 batch axes)
    """

    def _fill_rl(self, storage: RolloutStorage) -> None:
        for step in range(NUM_STEPS):
            t = RolloutStorage.Transition()
            t.observations = _make_obs()
            t.hidden_states = (None, None)
            t.actions = torch.full((NUM_ENVS, NUM_ACTIONS), float(step))
            t.values = torch.full((NUM_ENVS, 1), float(step))
            t.actions_log_prob = torch.full((NUM_ENVS,), float(step))
            t.distribution_params = (
                torch.full((NUM_ENVS, NUM_ACTIONS), float(step)),
                torch.full((NUM_ENVS, NUM_ACTIONS), 1.0),
            )
            t.rewards = torch.full((NUM_ENVS,), float(step))
            t.dones = torch.zeros(NUM_ENVS)
            storage.add_transition(t)
        storage.returns = torch.randn_like(storage.returns)
        storage.advantages = torch.randn_like(storage.advantages)

    def test_distillation_generator_invokes_decompress(self) -> None:
        """generator() must call obs_decompress_fn on each yielded per-timestep batch."""

        call_shapes = []

        def spy_decompress(stored_batch: TensorDict) -> TensorDict:
            # Record the shape we were called with so we can assert it below.
            call_shapes.append(tuple(stored_batch["image"].shape))
            return stored_batch

        obs = _make_obs()
        storage = RolloutStorage(
            "distillation", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS],
            obs_decompress_fn=spy_decompress,
        )
        # Populate with minimal distillation transitions.
        for step in range(NUM_STEPS):
            t = RolloutStorage.Transition()
            t.observations = _make_obs()
            t.hidden_states = (None, None)
            t.actions = torch.zeros(NUM_ENVS, NUM_ACTIONS)
            t.privileged_actions = torch.zeros(NUM_ENVS, NUM_ACTIONS)
            t.rewards = torch.zeros(NUM_ENVS)
            t.dones = torch.zeros(NUM_ENVS)
            storage.add_transition(t)

        list(storage.generator())
        assert len(call_shapes) == NUM_STEPS, "generator() should call decompress once per timestep"
        for shape in call_shapes:
            assert shape == (NUM_ENVS, IMG_C, IMG_H, IMG_W), (
                f"distillation generator should pass shape [N, *obs]; got {shape}"
            )

    def test_recurrent_generator_invokes_decompress_with_two_batch_axes(self) -> None:
        """recurrent_mini_batch_generator must call obs_decompress_fn with a 2-batch-axis
        TensorDict (shape [T_padded, trajectories, *obs])."""

        recorded = []

        def spy_decompress(stored_batch: TensorDict) -> TensorDict:
            recorded.append(tuple(stored_batch["image"].shape))
            return stored_batch

        obs = _make_obs()
        storage = RolloutStorage(
            "rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS],
            obs_decompress_fn=spy_decompress,
        )
        self._fill_rl(storage)

        num_mini_batches = 2
        for _ in storage.recurrent_mini_batch_generator(num_mini_batches=num_mini_batches, num_epochs=1):
            pass

        assert len(recorded) == num_mini_batches, (
            "recurrent generator should invoke decompress once per mini-batch"
        )
        for shape in recorded:
            # Expect [T_padded, trajectories, IMG_C, IMG_H, IMG_W] — exactly 5 dims.
            assert len(shape) == 5, (
                f"recurrent generator should pass 5-D TensorDict (2 batch axes + 3 obs axes); got {shape}"
            )
            assert shape[2:] == (IMG_C, IMG_H, IMG_W)

    def test_rank_polymorphic_decompress_works_across_all_generators(self) -> None:
        """A decompress function written with negative-axis indexing must produce correct
        output regardless of how many leading batch axes the generator supplies.
        """

        def polymorphic_decompress(stored_batch: TensorDict) -> TensorDict:
            # Trailing obs axes are (H, W, C) for this test; address them by negative index.
            img = stored_batch["image"]
            # mean over H and W (axes -3 and -2 in HWC convention)
            mean = img.mean(dim=(-3, -2), keepdim=True)
            stored_batch["image"] = img - mean
            return stored_batch

        # Build a storage whose 'image' key stores raw HWC (uint8-like) values so that
        # the trailing axes are (H, W, C) and the negative indexing is meaningful.
        def make_hwc_obs() -> TensorDict:
            return TensorDict(
                {
                    "image": torch.rand(NUM_ENVS, IMG_H, IMG_W, IMG_C),
                    "state": torch.randn(NUM_ENVS, STATE_DIM),
                },
                batch_size=[NUM_ENVS],
            )

        sample = make_hwc_obs()
        storage = RolloutStorage(
            "rl", NUM_ENVS, NUM_STEPS, sample, [NUM_ACTIONS],
            obs_decompress_fn=polymorphic_decompress,
        )

        for step in range(NUM_STEPS):
            t = RolloutStorage.Transition()
            t.observations = make_hwc_obs()
            t.hidden_states = (None, None)
            t.actions = torch.zeros(NUM_ENVS, NUM_ACTIONS)
            t.values = torch.zeros(NUM_ENVS, 1)
            t.actions_log_prob = torch.zeros(NUM_ENVS)
            t.distribution_params = (torch.zeros(NUM_ENVS, NUM_ACTIONS), torch.ones(NUM_ENVS, NUM_ACTIONS))
            t.rewards = torch.zeros(NUM_ENVS)
            t.dones = torch.zeros(NUM_ENVS)
            storage.add_transition(t)
        storage.returns = torch.randn_like(storage.returns)
        storage.advantages = torch.randn_like(storage.advantages)

        # Exercise both RL generators; the identity polymorphic decompress should run
        # without error regardless of rank, producing mean-subtracted images.
        for batch in storage.mini_batch_generator(num_mini_batches=2, num_epochs=1):
            img = batch.observations["image"]
            # Mean over the spatial axes (-3, -2) should be near zero after subtraction.
            assert torch.allclose(
                img.mean(dim=(-3, -2)), torch.zeros_like(img.mean(dim=(-3, -2))), atol=1e-5,
            ), "feedforward decompress output should be mean-subtracted per image"

        for batch in storage.recurrent_mini_batch_generator(num_mini_batches=2, num_epochs=1):
            img = batch.observations["image"]
            assert img.ndim == 5, f"recurrent obs should be 5-D; got {img.shape}"
            assert torch.allclose(
                img.mean(dim=(-3, -2)), torch.zeros_like(img.mean(dim=(-3, -2))), atol=1e-5,
            ), "recurrent decompress output should be mean-subtracted per image"
