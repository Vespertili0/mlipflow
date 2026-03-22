from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
from wfl.configset import ConfigSet, OutputSpec

from mlipflow.data import setup_logging

if TYPE_CHECKING:
    from ase import Atoms

setup_logging()
logger = logging.getLogger(__name__)


class RefData(TypedDict):
    isomer_block: np.ndarray
    reaction_block: np.ndarray
    topology: tuple


class GaussianCMLabeller:
    """
    Matches a configuration to a reference structure using partitioned
    Coulomb Matrices and distance-based topologies.
    """

    def __init__(
        self,
        references: dict[str, Atoms],
        idx_org: list | np.ndarray | slice,
        idx_h: list | np.ndarray | slice,
        sigma_isomer: float = 50.0,
        sigma_reaction: float = 10.0,
        threshold_isomer: float = 0.90,
        threshold_reaction: float = 0.85,
        bond_cutoff: float = 1.45,
        h2_cutoff: float = 1.0,
        label_key: str = "label",
    ):
        self.bond_cutoff = bond_cutoff
        self.h2_cutoff = h2_cutoff
        self.threshold_isomer = threshold_isomer
        self.threshold_reaction = threshold_reaction
        self.label_key = label_key

        # Precompute math denominators for the Gaussian kernels
        self._denom_isomer = 2.0 * (sigma_isomer**2)
        self._denom_reaction = 2.0 * (sigma_reaction**2)

        if not references:
            raise ValueError("References dictionary cannot be empty.")

        # Precompute indices as NumPy arrays once
        n_atoms = len(next(iter(references.values())))
        self.idx_org_arr = self._get_indices(idx_org, n_atoms)
        self.idx_h_arr = self._get_indices(idx_h, n_atoms)

        # Precompute reference matrix blocks and topologies
        self.ref_blocks: dict[str, RefData] = {}
        for label, atoms in references.items():
            cm = self._compute_full_cm(atoms)
            self.ref_blocks[label] = {
                "isomer_block": self._get_isomer_block(cm),
                "reaction_block": self._get_reaction_block(cm),
                "topology": self._get_topology(atoms),
            }

    def _get_indices(self, idx, size: int) -> np.ndarray:
        if isinstance(idx, slice):
            return np.arange(*idx.indices(size))
        return np.array(idx)

    def _get_topology(self, atoms: Atoms) -> tuple:
        dist_mat = atoms.get_all_distances(mic=True)

        np.fill_diagonal(dist_mat, np.inf)

        # Check for H2 formation
        if len(self.idx_h_arr) >= 2:
            h_h_dists = dist_mat[np.ix_(self.idx_h_arr, self.idx_h_arr)]
            if np.any(h_h_dists < self.h2_cutoff):
                return ("H2_gas",)

        # Organic Topology Logic
        h_org_dists = dist_mat[np.ix_(self.idx_h_arr, self.idx_org_arr)]
        _, bonded_org_local_indices = np.where(h_org_dists < self.bond_cutoff)
        bonded_org_global_indices = self.idx_org_arr[bonded_org_local_indices]

        return tuple(sorted(set(bonded_org_global_indices)))

    def _compute_full_cm(self, atoms: Atoms) -> np.ndarray:
        numbers = atoms.get_atomic_numbers()
        dist_mat = atoms.get_all_distances(mic=True)

        np.fill_diagonal(dist_mat, np.inf)
        z_grid = numbers[:, np.newaxis] * numbers[np.newaxis, :]
        cm = z_grid / dist_mat

        diag_vals = 0.5 * (numbers**2.4)
        np.fill_diagonal(cm, diag_vals)

        return cm

    def _get_isomer_block(self, cm: np.ndarray) -> np.ndarray:
        return cm[np.ix_(self.idx_org_arr, self.idx_org_arr)]

    def _get_reaction_block(self, cm: np.ndarray) -> np.ndarray:
        return cm[np.ix_(self.idx_org_arr, self.idx_h_arr)]

    def _gaussian_similarity(
        self, block1: np.ndarray, block2: np.ndarray, denom: float
    ) -> float:
        dist_sq = np.sum((block1 - block2) ** 2)
        return float(np.exp(-dist_sq / denom))

    def _get_similarity_to_ref(
        self, isomer_block: np.ndarray, reaction_block: np.ndarray, ref_label: str
    ) -> tuple[float, float]:
        ref = self.ref_blocks[ref_label]
        sim_iso = self._gaussian_similarity(
            isomer_block, ref["isomer_block"], self._denom_isomer
        )
        sim_rxn = self._gaussian_similarity(
            reaction_block, ref["reaction_block"], self._denom_reaction
        )
        return sim_iso, sim_rxn

    def match_config(self, atoms: Atoms) -> str:
        cm_frame = self._compute_full_cm(atoms)
        isomer_block = self._get_isomer_block(cm_frame)
        reaction_block = self._get_reaction_block(cm_frame)

        current_label = atoms.info.get(self.label_key)

        # Fast-path: Check if the current label is still valid
        if current_label in self.ref_blocks:
            sim_iso, sim_rxn = self._get_similarity_to_ref(
                isomer_block, reaction_block, current_label
            )
            if sim_iso >= self.threshold_isomer and sim_rxn >= self.threshold_reaction:
                return current_label

        best_label = "unknown"
        max_combined_sim = -1.0

        for label in self.ref_blocks:
            if label == current_label:
                continue

            sim_iso, sim_rxn = self._get_similarity_to_ref(
                isomer_block, reaction_block, label
            )

            # Only calculate combined sim if both thresholds are met
            if sim_iso >= self.threshold_isomer and sim_rxn >= self.threshold_reaction:
                combined_sim = sim_iso * sim_rxn
                if combined_sim > max_combined_sim:
                    max_combined_sim = combined_sim
                    best_label = label

        return best_label

    def match_by_topology(self, atoms: Atoms) -> str:
        frame_topology = self._get_topology(atoms)

        if frame_topology == ("H2_gas",):
            return "unknown"

        valid_refs = {
            label: data
            for label, data in self.ref_blocks.items()
            if data["topology"] == frame_topology
        }

        if not valid_refs:
            return "unknown"

        cm_frame = self._compute_full_cm(atoms)
        isomer_block = self._get_isomer_block(cm_frame)

        current_label = atoms.info.get(self.label_key)

        # Fast-path
        if current_label in valid_refs:
            sim_iso = self._gaussian_similarity(
                isomer_block,
                valid_refs[current_label]["isomer_block"],
                self._denom_isomer,
            )
            if sim_iso >= self.threshold_isomer:
                return current_label

        best_label = "unknown"
        max_sim = -1.0

        for label, ref_data in valid_refs.items():
            if label == current_label:
                continue

            sim_iso = self._gaussian_similarity(
                isomer_block, ref_data["isomer_block"], self._denom_isomer
            )
            if sim_iso >= self.threshold_isomer and sim_iso > max_sim:
                max_sim = sim_iso
                best_label = label

        return best_label

    def evaluate_frame(
        self, atoms: Atoms, use_topology: bool = False
    ) -> dict[str, Any]:
        cm_frame = self._compute_full_cm(atoms)
        isomer_block = self._get_isomer_block(cm_frame)

        results = {"similarities": {}, "probabilities": {}}
        total_sim = 0.0

        if use_topology:
            frame_topology = self._get_topology(atoms)
            results["topology"] = frame_topology

            for label, ref_data in self.ref_blocks.items():
                if (ref_data["topology"] != frame_topology) or (
                    frame_topology == ("H2_gas",)
                ):
                    results["similarities"][label] = 0.0
                    continue

                sim_iso = self._gaussian_similarity(
                    isomer_block, ref_data["isomer_block"], self._denom_isomer
                )
                results["similarities"][label] = sim_iso
                total_sim += sim_iso
        else:
            reaction_block = self._get_reaction_block(cm_frame)
            for label in self.ref_blocks:
                sim_iso, sim_rxn = self._get_similarity_to_ref(
                    isomer_block, reaction_block, label
                )
                combined_sim = sim_iso * sim_rxn
                results["similarities"][label] = combined_sim
                total_sim += combined_sim

        for label, sim in results["similarities"].items():
            results["probabilities"][label] = sim / (total_sim + 1e-12)

        return results


def relabel_configs(
    in_file: str,
    reference_configs: str,
    label_key: str = "species",
    use_topology: bool = True,
    **kwargs: Any,
) -> OutputSpec:
    """
    Relabels the configurations in a ConfigSet using GaussianCMLabeller.

    Parameters
    ----------
    in_file : str
        Path to the input file containing configurations.
    references : str
        Path to the reference configurations file.
    label_key : str, default 'species'
        The key in atoms.info to store the label.
    use_topology : bool, default False
        Whether to use topology-based matching instead of Gaussian-based matching.
    **kwargs : dict
        Additional arguments for GaussianCMLabeller.
        - 'idx_org' : Union[list, np.ndarray, slice]
            Indices of the organic molecule atoms.
        - 'idx_h' : Union[list, np.ndarray, slice]
            Indices of the reacting H-atoms.
        - 'threshold_isomer' : float
            Threshold for isomer matching.
        - 'threshold_reaction' : float
            Threshold for reaction matching.

    Returns
    -------
    Tuple[list[Atoms], list[Atoms]]
        A tuple containing the known and unknown configurations.
    """
    logger.info(f"Starting {label_key} label check...")

    references = {at.info[label_key]: at for at in list(ConfigSet(reference_configs))}

    known_configs = []
    unknown_configs = []

    labeller = GaussianCMLabeller(references, **kwargs)

    for at in list(ConfigSet(in_file)):
        if use_topology:
            new_label = labeller.match_by_topology(at)
        else:
            new_label = labeller.match_config(at)

        at.info[label_key] = new_label

        if new_label == "unknown":
            unknown_configs.append(at)
        else:
            known_configs.append(at)

    logger.info(
        f"{label_key} label check complete: {len(known_configs)} known + "
        f"{len(unknown_configs)} unknown configurations."
    )

    return known_configs, unknown_configs
