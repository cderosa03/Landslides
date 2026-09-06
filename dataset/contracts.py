"""Tensor contracts shared by multimodal datasets and DataLoaders."""

import torch


SAMPLE_KEYS = {
    "planet_pre", "planet_post", "aux", "s2_pre", "s2_post",
    "s2_valid_pre", "s2_valid_post", "mask", "event", "patch_id",
}


def _require_keys(data, required, context):
    missing = sorted(required.difference(data))
    if missing:
        raise KeyError(f"{context}: modalità mancanti: {missing}")


def validate_multimodal_sample(planet, sentinel, alignment, n_temporal):
    """Raise a clear error when an unbatched multimodal sample is invalid."""
    sample_id = f"{alignment['event']}/{alignment['patch_id']}"
    _require_keys(
        planet, {"pre", "post", "aux", "mask", "event", "patch_id"},
        f"Campione {sample_id} Planet",
    )
    _require_keys(
        sentinel, {"pre", "post", "valid_pre", "valid_post", "event", "patch_id"},
        f"Campione {sample_id} Sentinel-2",
    )

    expected_id = (alignment["event"], alignment["patch_id"])
    if (planet["event"], planet["patch_id"]) != expected_id:
        raise ValueError(f"Campione {sample_id}: identità Planet incoerente")
    if (sentinel["event"], sentinel["patch_id"]) != expected_id:
        raise ValueError(f"Campione {sample_id}: identità Sentinel-2 incoerente")

    tensors = {
        "planet_pre": planet["pre"],
        "planet_post": planet["post"],
        "aux": planet["aux"],
        "mask": planet["mask"],
        "s2_pre": sentinel["pre"],
        "s2_post": sentinel["post"],
        "s2_valid_pre": sentinel["valid_pre"],
        "s2_valid_post": sentinel["valid_post"],
    }
    invalid = [name for name, value in tensors.items() if not isinstance(value, torch.Tensor)]
    if invalid:
        raise TypeError(f"Campione {sample_id}: non tensor: {invalid}")

    planet_pre = tensors["planet_pre"]
    if planet_pre.ndim != 3 or planet_pre.shape[0] != 3:
        raise ValueError(
            f"Campione {sample_id}: planet_pre atteso (3,H,W), "
            f"ricevuto {tuple(planet_pre.shape)}"
        )
    _, height, width = planet_pre.shape
    expected_shapes = {
        "planet_post": (3, height, width),
        "aux": (4, height, width),
        "mask": (1, height, width),
        "s2_pre": (n_temporal, 10, height, width),
        "s2_post": (n_temporal, 10, height, width),
        "s2_valid_pre": (n_temporal,),
        "s2_valid_post": (n_temporal,),
    }
    wrong_shapes = {
        name: tuple(tensors[name].shape)
        for name, expected in expected_shapes.items()
        if tensors[name].shape != expected
    }
    if wrong_shapes:
        raise ValueError(
            f"Campione {sample_id}: forme errate {wrong_shapes}; "
            f"attese {expected_shapes}"
        )
    if (
        tensors["s2_valid_pre"].dtype != torch.bool
        or tensors["s2_valid_post"].dtype != torch.bool
    ):
        raise TypeError(f"Campione {sample_id}: valid_pre/valid_post devono essere bool")
    if tensors["mask"].dtype not in (torch.bool, torch.uint8):
        raise TypeError(f"Campione {sample_id}: mask deve essere bool o uint8")

    for name in ("planet_pre", "planet_post", "aux", "s2_pre", "s2_post"):
        tensor = tensors[name]
        if not torch.is_floating_point(tensor):
            raise TypeError(f"Campione {sample_id}: {name} deve essere floating point")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"Campione {sample_id}: {name} contiene NaN o Inf")


def validate_multimodal_batch(batch, n_temporal, loader_name):
    """Verify that default DataLoader collation preserved every modality."""
    _require_keys(batch, SAMPLE_KEYS, f"Batch {loader_name}")
    tensors = {name: batch[name] for name in SAMPLE_KEYS - {"event", "patch_id"}}
    invalid = [name for name, value in tensors.items() if not isinstance(value, torch.Tensor)]
    if invalid:
        raise TypeError(f"Batch {loader_name}: non tensor: {invalid}")

    planet_pre = batch["planet_pre"]
    if planet_pre.ndim != 4 or planet_pre.shape[1] != 3:
        raise ValueError(
            f"Batch {loader_name}: planet_pre atteso (B,3,H,W), "
            f"ricevuto {tuple(planet_pre.shape)}"
        )
    batch_size, _, height, width = planet_pre.shape
    expected_shapes = {
        "planet_post": (batch_size, 3, height, width),
        "aux": (batch_size, 4, height, width),
        "mask": (batch_size, 1, height, width),
        "s2_pre": (batch_size, n_temporal, 10, height, width),
        "s2_post": (batch_size, n_temporal, 10, height, width),
        "s2_valid_pre": (batch_size, n_temporal),
        "s2_valid_post": (batch_size, n_temporal),
    }
    wrong_shapes = {
        name: tuple(batch[name].shape)
        for name, expected in expected_shapes.items()
        if batch[name].shape != expected
    }
    if wrong_shapes:
        raise ValueError(f"Batch {loader_name}: forme errate {wrong_shapes}")
