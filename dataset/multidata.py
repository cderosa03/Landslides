import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class MultiModalLandslideDataset(Dataset):
    """
    Dataset multimodale che allinea PlanetScope e Sentinel-2
    e restituisce input separati per ciascun encoder.

    Input PlanetScope : pre/post (3, H, W)
    Input Sentinel-2  : pre/post (N_TEMPORAL, 10, H, W) + valid_pre/post (N_TEMPORAL,)
    Ground truth      : mask (1, H, W) da PlanetScope
    """

    def __init__(self, planet_ds, s2_ds, apply_transform: bool = False):
        self.planet_ds = planet_ds
        self.s2_ds = s2_ds
        self.apply_transform = apply_transform

        self.aligned_indices = []

        # Costruisce mappa (event, patch_id) → indice PlanetScope
        # Legge direttamente dall'indice interno (niente I/O su disco)
        planet_map = {}
        for i, entry in enumerate(tqdm(self.planet_ds.folders,
                                       desc="Indicizzazione PlanetScope")):
            key = (entry["event"], entry["patch_id"])
            planet_map[key] = i

        # Allinea gli indici S2 con quelli PlanetScope
        for j, entry in enumerate(tqdm(self.s2_ds.samples,
                                       desc="Allineamento S2 ↔ PlanetScope")):
            key = (entry["event"], entry["patch_id"])
            if key in planet_map:
                self.aligned_indices.append({
                    "planet_idx": planet_map[key],
                    "s2_idx":     j,
                    "event":      entry["event"],
                    "patch_id":   entry["patch_id"],
                })

        print(f"MultiModal dataset pronto: {len(self.aligned_indices)} patch allineate")

    def __len__(self):
        return len(self.aligned_indices)

    @staticmethod
    def validate_batch(batch, expected_temporal, loader_name):
        """Validate the tensors produced by a collated DataLoader batch."""
        required = {
            "planet_pre", "planet_post", "aux", "s2_pre", "s2_post",
            "s2_valid_pre", "s2_valid_post", "mask", "event", "patch_id",
        }
        missing = sorted(required.difference(batch))
        if missing:
            raise KeyError(
                f"Batch {loader_name}: modalità mancanti dopo il collate: {missing}"
            )

        tensors = {
            name: value for name, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        tensor_names = {
            "planet_pre", "planet_post", "aux", "s2_pre", "s2_post",
            "s2_valid_pre", "s2_valid_post", "mask",
        }
        missing_tensors = sorted(tensor_names.difference(tensors))
        if missing_tensors:
            raise TypeError(
                f"Batch {loader_name}: modalità non tensor dopo il collate: "
                f"{missing_tensors}"
            )

        planet_pre = batch["planet_pre"]
        planet_post = batch["planet_post"]
        aux = batch["aux"]
        mask = batch["mask"]
        s2_pre = batch["s2_pre"]
        s2_post = batch["s2_post"]
        valid_pre = batch["s2_valid_pre"]
        valid_post = batch["s2_valid_post"]

        if planet_pre.ndim != 4 or planet_pre.shape[1] != 3:
            raise ValueError(
                f"Batch {loader_name}: planet_pre deve avere forma "
                f"(B,3,H,W), ricevuta {tuple(planet_pre.shape)}"
            )
        batch_size, _, height, width = planet_pre.shape
        if planet_post.shape != (batch_size, 3, height, width):
            raise ValueError(
                f"Batch {loader_name}: Planet pre/post non congruenti: "
                f"{tuple(planet_pre.shape)} vs {tuple(planet_post.shape)}"
            )
        if aux.shape != (batch_size, 3, height, width):
            raise ValueError(
                f"Batch {loader_name}: AUX deve avere forma "
                f"(B,3,{height},{width}), ricevuta {tuple(aux.shape)}"
            )
        if mask.shape != (batch_size, 1, height, width):
            raise ValueError(
                f"Batch {loader_name}: mask deve avere forma "
                f"(B,1,{height},{width}), ricevuta {tuple(mask.shape)}"
            )

        expected_s2 = (batch_size, expected_temporal, 10, height, width)
        if s2_pre.shape != expected_s2 or s2_post.shape != expected_s2:
            raise ValueError(
                f"Batch {loader_name}: Sentinel-2 pre/post devono avere forma "
                f"{expected_s2}, ricevute {tuple(s2_pre.shape)} e "
                f"{tuple(s2_post.shape)}"
            )
        expected_valid = (batch_size, expected_temporal)
        if valid_pre.shape != expected_valid or valid_post.shape != expected_valid:
            raise ValueError(
                f"Batch {loader_name}: maschere temporali devono avere forma "
                f"{expected_valid}, ricevute {tuple(valid_pre.shape)} e "
                f"{tuple(valid_post.shape)}"
            )
        if valid_pre.dtype != torch.bool or valid_post.dtype != torch.bool:
            raise TypeError(
                f"Batch {loader_name}: maschere temporali non booleane: "
                f"{valid_pre.dtype}, {valid_post.dtype}"
            )

        for name in ("planet_pre", "planet_post", "aux", "s2_pre", "s2_post"):
            tensor = batch[name]
            if not torch.is_floating_point(tensor):
                raise TypeError(
                    f"Batch {loader_name}: {name} deve essere floating point, "
                    f"ricevuto {tensor.dtype}"
                )
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"Batch {loader_name}: {name} contiene NaN o Inf")

        if mask.dtype not in (torch.bool, torch.uint8):
            raise TypeError(
                f"Batch {loader_name}: mask deve essere bool o uint8, "
                f"ricevuto {mask.dtype}"
            )

    def _validate_sample(self, planet, s2, alignment):
        """Validate the complete multimodal tensor contract for one sample."""
        sample_id = f"{alignment['event']}/{alignment['patch_id']}"
        planet_keys = {"pre", "post", "aux", "mask", "event", "patch_id"}
        s2_keys = {
            "pre", "post", "valid_pre", "valid_post", "event", "patch_id"
        }

        missing_planet = sorted(planet_keys.difference(planet))
        missing_s2 = sorted(s2_keys.difference(s2))
        if missing_planet or missing_s2:
            raise KeyError(
                f"Campione {sample_id}: modalità mancanti; "
                f"Planet={missing_planet or 'nessuna'}, "
                f"Sentinel-2={missing_s2 or 'nessuna'}"
            )

        expected_identity = (alignment["event"], alignment["patch_id"])
        planet_identity = (planet["event"], planet["patch_id"])
        s2_identity = (s2["event"], s2["patch_id"])
        if planet_identity != expected_identity or s2_identity != expected_identity:
            raise ValueError(
                f"Campione {sample_id}: identità incoerenti; "
                f"Planet={planet_identity}, Sentinel-2={s2_identity}"
            )

        tensors = {
            "planet_pre": planet["pre"],
            "planet_post": planet["post"],
            "aux": planet["aux"],
            "mask": planet["mask"],
            "s2_pre": s2["pre"],
            "s2_post": s2["post"],
            "s2_valid_pre": s2["valid_pre"],
            "s2_valid_post": s2["valid_post"],
        }
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"Campione {sample_id}: {name} non è un torch.Tensor "
                    f"({type(tensor).__name__})"
                )

        planet_pre = tensors["planet_pre"]
        planet_post = tensors["planet_post"]
        aux = tensors["aux"]
        mask = tensors["mask"]
        s2_pre = tensors["s2_pre"]
        s2_post = tensors["s2_post"]
        valid_pre = tensors["s2_valid_pre"]
        valid_post = tensors["s2_valid_post"]

        if planet_pre.ndim != 3 or planet_pre.shape[0] != 3:
            raise ValueError(
                f"Campione {sample_id}: planet_pre deve avere forma (3,H,W), "
                f"ricevuta {tuple(planet_pre.shape)}"
            )
        if planet_post.shape != planet_pre.shape:
            raise ValueError(
                f"Campione {sample_id}: Planet pre/post non congruenti: "
                f"{tuple(planet_pre.shape)} vs {tuple(planet_post.shape)}"
            )

        _, height, width = planet_pre.shape
        if aux.shape != (3, height, width):
            raise ValueError(
                f"Campione {sample_id}: AUX deve avere forma "
                f"(3,{height},{width}), ricevuta {tuple(aux.shape)}"
            )
        if mask.shape != (1, height, width):
            raise ValueError(
                f"Campione {sample_id}: mask deve avere forma "
                f"(1,{height},{width}), ricevuta {tuple(mask.shape)}"
            )

        expected_temporal = self.s2_ds.n_temporal
        expected_s2_shape = (expected_temporal, 10, height, width)
        if s2_pre.shape != expected_s2_shape or s2_post.shape != expected_s2_shape:
            raise ValueError(
                f"Campione {sample_id}: Sentinel-2 pre/post devono avere forma "
                f"{expected_s2_shape}, ricevute {tuple(s2_pre.shape)} e "
                f"{tuple(s2_post.shape)}"
            )

        expected_valid_shape = (expected_temporal,)
        if valid_pre.shape != expected_valid_shape or valid_post.shape != expected_valid_shape:
            raise ValueError(
                f"Campione {sample_id}: le maschere temporali devono avere forma "
                f"{expected_valid_shape}, ricevute {tuple(valid_pre.shape)} e "
                f"{tuple(valid_post.shape)}"
            )
        if valid_pre.dtype != torch.bool or valid_post.dtype != torch.bool:
            raise TypeError(
                f"Campione {sample_id}: le maschere temporali devono essere bool, "
                f"ricevuti {valid_pre.dtype} e {valid_post.dtype}"
            )

        float_tensors = {
            "planet_pre": planet_pre,
            "planet_post": planet_post,
            "aux": aux,
            "s2_pre": s2_pre,
            "s2_post": s2_post,
        }
        for name, tensor in float_tensors.items():
            if not torch.is_floating_point(tensor):
                raise TypeError(
                    f"Campione {sample_id}: {name} deve essere floating point, "
                    f"ricevuto {tensor.dtype}"
                )
            if not torch.isfinite(tensor).all().item():
                raise ValueError(
                    f"Campione {sample_id}: {name} contiene NaN o Inf"
                )

        if mask.dtype not in (torch.bool, torch.uint8):
            raise TypeError(
                f"Campione {sample_id}: mask deve essere bool o uint8, "
                f"ricevuto {mask.dtype}"
            )

    def __getitem__(self, idx):
        alignment = self.aligned_indices[idx]

        planet = self.planet_ds[alignment["planet_idx"]]
        s2     = self.s2_ds[alignment["s2_idx"]]
        self._validate_sample(planet, s2, alignment)

        # ── PlanetScope ───────────────────────────────────────────────────
        planet_pre  = planet["pre"]    # (C_p, H, W)
        planet_post = planet["post"]   # (C_p, H, W)
        mask        = planet["mask"]   # (1, H, W)  — ground truth

        # ── Sentinel-2 (serie temporale) ──────────────────────────────────
        s2_pre        = s2["pre"]         # (N_TEMPORAL, 10, H, W)
        s2_post       = s2["post"]        # (N_TEMPORAL, 10, H, W)
        s2_valid_pre  = s2["valid_pre"]   # (N_TEMPORAL,) bool
        s2_valid_post = s2["valid_post"]  # (N_TEMPORAL,) bool

        return {
            "planet_pre":    planet_pre,
            "planet_post":   planet_post,
            "aux":            planet["aux"],
            "s2_pre":        s2_pre,
            "s2_post":       s2_post,
            "s2_valid_pre":  s2_valid_pre,
            "s2_valid_post": s2_valid_post,
            "mask":          mask,
            "event":         alignment["event"],
            "patch_id":      alignment["patch_id"],
        }
