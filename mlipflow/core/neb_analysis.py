"""NEB trajectory analysis and standardised visualisation module.

Provides :class:`NEBAnalysis` for parsing multi-pathway NEB XYZ trajectories,
comparing results across computational methods (e.g., MACE vs DFT), and
generating standardised matplotlib figures compatible with ``mlipflow``'s
path-factory routing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from ase.geometry import find_mic
from ase.io import read
from ase.utils.forcecurve import fit_raw
from wfl.configset import ConfigSet

from mlipflow.data import setup_logging

if TYPE_CHECKING:
    from ase import Atoms

setup_logging()
logger = logging.getLogger(__name__)


class NEBAnalysis:
    """
    A class to analyze and plot Nudged Elastic Band (NEB) trajectories.

    Allows for comparison between different computational methods (e.g., MACE, DFT)
    and overlaying multiple pathways extracted from a single XYZ trajectory,
    automatically grouped by their reaction species.
    """

    def __init__(self, images: list[Atoms], n_frames: int | None = None) -> None:
        """
        Initializes the NEBAnalysis class.

        Args:
            images (list): A list of ASE Atoms objects read from an xyz file.
            n_frames (int, optional): The number of frames per NEB pathway.
                If provided, the 'images' list will be split into multiple pathways.
                If None, treats the entire 'images' list as a single pathway.
        """
        self.images = images
        if n_frames is not None:
            if len(images) % n_frames != 0:
                raise ValueError(
                    f"Total images ({len(images)}) is not a multiple of n_frames ({n_frames})."
                )
            # Chunk the continuous trajectory into individual paths
            self.paths = [
                images[i : i + n_frames] for i in range(0, len(images), n_frames)
            ]
        else:
            self.paths = [images]

        # Group paths by their reaction (Educt -> Product)
        self.reaction_groups = {}
        for i, path in enumerate(self.paths):
            # Extract species from the first (educt) and last (product) frames
            educt = path[0].info.get("species", "Unknown Educt")
            product = path[-1].info.get("species", "Unknown Product")
            reaction_key = f"{educt} -> {product}"
            if reaction_key not in self.reaction_groups:
                self.reaction_groups[reaction_key] = []
            self.reaction_groups[reaction_key].append(i)

    @classmethod
    def from_file(
        cls, source: str | Path | ConfigSet, n_frames: int | None = None
    ) -> NEBAnalysis:
        """Construct a :class:`NEBAnalysis` instance from a file path or ConfigSet.

        Parameters
        ----------
        source : str, Path, or ConfigSet
            - ``str`` or ``Path``: an XYZ trajectory file path, passed directly
              to ``ase.io.read`` with ``index=":"`` to load all frames.
            - ``ConfigSet``: a ``wfl`` configset, converted to a list via
              ``list(source)``.
        n_frames : int, optional
            Number of frames per NEB pathway. See :meth:`__init__` for details.

        Returns
        -------
        NEBAnalysis
            A fully initialised instance.

        Raises
        ------
        TypeError
            If ``source`` is not a ``str``, ``Path``, or ``ConfigSet`` instance.
        FileNotFoundError
            If ``source`` is a file path that does not exist on disk.
        """
        if isinstance(source, ConfigSet):
            images = list(source)
        elif isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(
                    f"NEBAnalysis.from_file: file not found: {path}"
                )
            logger.info("Loading NEB trajectory from: %s", path)
            images = read(str(path), index=":")
        else:
            raise TypeError(
                f"NEBAnalysis.from_file: expected str, Path, or ConfigSet, "
                f"got {type(source).__name__!r}"
            )
        return cls(images=images, n_frames=n_frames)

    def get_force_components(self, images, prefix="MACE"):
        """
        Executes force calculations separated from the plotting logic.
        Allows for quick analysis of force distributions in pathway-frames and endpoints
        without needing to satisfy all conditions for reliable pathway plotting.
        """
        nim = len(images)
        if nim < 3:
            raise ValueError("NEB trajectory must contain at least 3 images.")

        # 1. Collect Positions and Forces
        R = [atoms.get_positions() for atoms in images]
        F_raw = [atoms.arrays[f"{prefix}_forces"] for atoms in images]
        F_total = []
        fmax_list = []
        # --- Constraint Handling & fmax Calculation ---
        for i, atoms in enumerate(images):
            # Build the boolean mask manually for ASE constraints
            fixed_mask = np.zeros(len(atoms), dtype=bool)
            for constraint in atoms.constraints:
                if hasattr(constraint, "index"):
                    fixed_mask[constraint.index] = True
                elif hasattr(constraint, "get_indices"):
                    fixed_mask[constraint.get_indices()] = True
            # Clean the forces: zero out forces on constrained atoms
            current_f = F_raw[i].copy()
            current_f[fixed_mask] = 0.0
            F_total.append(current_f)
            # Calculate fmax only for the MOVING atoms
            moving_forces = F_raw[i][~fixed_mask]
            if moving_forces.size > 0:
                # Max of the magnitudes of individual active atom force vectors
                f_atom_magnitudes = np.sqrt((moving_forces**2).sum(axis=1))
                fmax = f_atom_magnitudes.max()
            else:
                fmax = 0.0
            fmax_list.append(np.round(fmax, 5))

        # 2. Force Decomposition (Iterating through ALL images)
        f_parallel_norms = []
        f_perp_norms = []
        for i in range(nim):
            # --- Tangent Calculation ---
            if i == 0:
                tangent, _ = find_mic(R[i + 1] - R[i], images[i].cell, images[i].pbc)
            elif i == nim - 1:
                tangent, _ = find_mic(R[i] - R[i - 1], images[i].cell, images[i].pbc)
            else:
                tangent, _ = find_mic(
                    R[i + 1] - R[i - 1], images[i].cell, images[i].pbc
                )
            # Normalize the tangent vector
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm < 1e-8:
                tau = np.zeros_like(
                    tangent
                )  # Fallback if images are perfectly identical
            else:
                tau = tangent / tangent_norm

            # --- Decomposition ---
            fi = F_total[i].ravel()  # Masked forces (fixed atoms = 0)
            tau_flat = tau.ravel()
            # F_parallel
            f_parallel_mag = np.dot(fi, tau_flat)
            f_parallel_norms.append(abs(f_parallel_mag))
            # F_perp
            f_parallel_vec = f_parallel_mag * tau_flat
            f_perp_vec = fi - f_parallel_vec
            f_perp_norm = np.linalg.norm(f_perp_vec)
            f_perp_norms.append(f_perp_norm)

        return fmax_list, f_parallel_norms, f_perp_norms

    def analyse_neb_force_decomposition(
        self, images, prefix="MACE", ax=None, x_coords=None, color=None
    ):
        """
        Analyzes an ASE NEB trajectory to decompose forces into components
        parallel and perpendicular to the reaction path.
        """
        # Execute the separate mathematical logic
        fmax_list, f_parallel_norms, f_perp_norms = self.get_force_components(
            images, prefix
        )
        pathway_fmax = max(fmax_list)
        logger.info("[%s] Pathway global active fmax: %.4f eV/Å", prefix, pathway_fmax)

        # 3. Plotting
        show_plot = ax is None
        if show_plot:
            fig, ax = plt.subplots(figsize=(8, 5))

        if x_coords is None:
            x_coords = list(range(len(images)))
            ax.set_xlabel("Image Index", fontsize=12)

        # Plot matching prefix color if provided (for shared axis multi-plots)
        if color:
            ax.plot(
                x_coords,
                f_parallel_norms,
                "o-",
                color=color,
                label=f"{prefix} " + r"$|F_{\parallel}|$",
            )
            ax.plot(
                x_coords,
                f_perp_norms,
                "s--",
                color=color,
                alpha=0.7,
                label=f"{prefix} " + r"$|F_{\perp}|$",
            )
            ax.plot(
                x_coords,
                fmax_list,
                ":",
                color=color,
                linewidth=1.5,
                alpha=0.8,
                label=f"{prefix} " + r"$f_{max}$",
            )
        else:
            # Standalone default plot
            ax.plot(
                x_coords,
                f_parallel_norms,
                "o-",
                color="tab:red",
                label=r"$|F_{\parallel}|$ (Parallel)",
            )
            ax.plot(
                x_coords,
                f_perp_norms,
                "s--",
                color="tab:blue",
                label=r"$|F_{\perp}|$ (Perpendicular)",
            )
            ax.plot(
                x_coords, fmax_list, "k--", linewidth=1.0, alpha=0.8, label=r"$f_{max}$"
            )

        ax.set_ylabel("Force Component [eV/Å]", fontsize=10)
        ax.legend(fontsize=8, loc="upper right", ncol=2 if color else 1)
        ax.grid(True, linestyle=":")

        # Highlight images with high F_perp relative to F_para
        max_f = max(*f_parallel_norms, *f_perp_norms)
        if max_f > 0:
            threshold = 0.5  # Highlight if F_perp is > 50% of the maximum force
            for i, f_perp in enumerate(f_perp_norms):
                if f_perp > (max_f * threshold):
                    ax.axvline(x_coords[i], color="k", alpha=0.1, linewidth=5)

        if show_plot:
            plt.tight_layout()
        return ax

    def _prepare_force_fit(self, path, prefix):
        """
        Extracts positions, relative energies, and forces for a specific method prefix,
        and prepares the force fit.
        """
        R = [at.positions for at in path]
        # Extract energies and make them relative to the first image (E[0] = 0)
        E = np.array([at.info[f"{prefix}_energy"] for at in path])
        E -= E[0]
        F = [at.arrays[f"{prefix}_forces"] for at in path]
        A = path[0].cell
        pbc = path[0].pbc
        return fit_raw(E, F, R, A, pbc)

    def get_reaction_string(self, path_index):
        """Helper function to get the reaction string for a specific path."""
        path = self.paths[path_index]
        educt = path[0].info.get("species", "Unknown")
        product = path[-1].info.get("species", "Unknown")
        return f"{educt} -> {product}"

    def get_barrier(self, path_index=0, prefix="DFT", fit=True):
        """
        Calculates the forward barrier, reverse barrier, and the reaction energy (dE).
        """
        forcefit = self._prepare_force_fit(self.paths[path_index], prefix)
        energies = forcefit.energies
        fit_energies = forcefit.fit_energies
        dE = energies[-1] - energies[0]

        if fit:
            barrier_f = max(fit_energies) - fit_energies[0]
            barrier_r = max(fit_energies) - fit_energies[-1]
        else:
            barrier_f = max(energies) - energies[0]
            barrier_r = max(energies) - energies[-1]

        return barrier_f, barrier_r, dE

    def plot_band(self, path_index=0, prefix="DFT", ax=None, label=None, color=None):
        """
        Plots a single NEB band on a matplotlib axes object.
        """
        if ax is None:
            fig, ax = plt.subplots()

        forcefit = self._prepare_force_fit(self.paths[path_index], prefix)

        lbl_data = f"{label} (Data)" if label else f"{prefix} (Data)"
        lbl_fit = f"{label} (Fit)" if label else f"{prefix} (Fit)"

        # Plot discrete image data points
        (line,) = ax.plot(
            forcefit.path, forcefit.energies, "o", color=color, label=lbl_data
        )

        # Plot the interpolated cubic spline fit
        ax.plot(
            forcefit.fit_path,
            forcefit.fit_energies,
            "-",
            color=line.get_color(),
            label=lbl_fit,
        )

        ax.set_xlabel("Path [Å]")
        ax.set_ylabel("Relative Energy [eV]")

        return ax

    def plot_comparison(self, path_index=0, prefixes=("MACE", "DFT"), ax=None):
        """
        Plots the NEB path computed with multiple methods (e.g., MACE and DFT)
        in the SAME diagram for direct comparison.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        for prefix in prefixes:
            self.plot_band(path_index=path_index, prefix=prefix, ax=ax, label=prefix)

        reaction = self.get_reaction_string(path_index)
        ax.legend()
        ax.set_title(
            f"NEB Pathway Comparison (Path Index {path_index})\nReaction: {reaction}"
        )
        ax.grid(True, linestyle="--", alpha=0.6)

        if ax.figure is not None:
            return ax.figure
        return ax

    def plot_multiple_pathways(self, prefixes=("MACE", "DFT")):
        """
        Overlays multiple pathway calculations, grouped dynamically by reaction.
        Generates a separate figure for each reaction (Educt -> Product).
        Sorts pathways within each reaction by barrier, rendering higher-barrier paths
        as lighter gradients and highlighting the Minimum Energy Pathway (MEP).

        Returns:
            dict: A dictionary mapping reaction strings to (Figure, Axes) tuples.
        """
        n_prefixes = len(prefixes)
        cmap_names = ["Blues", "Reds", "Greens", "Purples", "Oranges"]
        reaction_figures = {}
        self.mep_indices = {}  # Track the MEP path index for each reaction

        for reaction_key, path_indices in self.reaction_groups.items():
            fig, axes = plt.subplots(
                1, n_prefixes, figsize=(6 * n_prefixes, 6), sharey=True
            )

            # Handle the case where the user only passed one prefix
            if n_prefixes == 1:
                axes = [axes]

            for i, prefix in enumerate(prefixes):
                ax = axes[i]
                cmap = plt.get_cmap(cmap_names[i % len(cmap_names)])

                # 1. Collect data and calculate barriers for pathways IN THIS REACTION
                path_data = []
                for p_idx in path_indices:
                    forcefit = self._prepare_force_fit(self.paths[p_idx], prefix)
                    Ef = max(forcefit.fit_energies) - forcefit.fit_energies[0]
                    Er = max(forcefit.fit_energies) - forcefit.fit_energies[-1]
                    path_data.append((p_idx, forcefit, Ef, Er))

                # 2. Sort pathways by forward barrier (descending)
                path_data.sort(key=lambda x: x[2], reverse=True)

                N = len(path_data)
                for rank, (p_idx, forcefit, Ef, Er) in enumerate(path_data):
                    is_mep = rank == N - 1
                    if is_mep:
                        # We use the first prefix as the reference for identifying the MEP path index
                        if i == 0:
                            self.mep_indices[reaction_key] = p_idx
                        color = cmap(0.9)
                        alpha = 1.0
                        lw = 2.5
                        label = f"MEP ($E_f$={Ef:.2f}, $E_r$={Er:.2f} eV)"
                        zorder = 5
                    else:
                        intensity = 0.3 + 0.4 * (rank / max(1, N - 2)) if N > 2 else 0.5
                        color = cmap(intensity)
                        alpha = 0.6
                        lw = 1.0
                        label = None
                        zorder = 3

                    ax.plot(
                        forcefit.path,
                        forcefit.energies,
                        "o",
                        color=color,
                        alpha=alpha,
                        zorder=zorder,
                    )
                    ax.plot(
                        forcefit.fit_path,
                        forcefit.fit_energies,
                        "-",
                        color=color,
                        alpha=alpha,
                        linewidth=lw,
                        label=label,
                        zorder=zorder,
                    )

                ax.set_title(f"{prefix} Pathways\n{reaction_key}")
                ax.set_xlabel("Path [Å]")
                ax.legend(loc="upper right")
                ax.grid(True, linestyle="--", alpha=0.6)
                if i == 0:
                    ax.set_ylabel("Relative Energy [eV]")

            plt.tight_layout()
            reaction_figures[reaction_key] = (fig, axes)

        return reaction_figures

    def plot_mep_pathways(self, prefixes=("MACE", "DFT")):
        """
        Plots a comprehensive two-panel comparison for the Minimum Energy Pathway (MEP) of each reaction.
        Top panel covers the Energy Pathway. Bottom panel covers the shared-axis force decomposition.
        Note: plot_multiple_pathways() must be called first to populate the MEP indices.

        Returns:
            dict: A dictionary mapping reaction strings to Figure objects.
        """
        if not hasattr(self, "mep_indices") or not self.mep_indices:
            raise ValueError(
                "MEP indices not found. Please run plot_multiple_pathways() first."
            )

        mep_figures = {}
        # Fetching colors from matplotlib's current runtime color cycle
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for reaction_key, p_idx in self.mep_indices.items():
            # Setup shared-x 2-panel figure
            fig, (ax_e, ax_f) = plt.subplots(
                2, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
            )
            for i, prefix in enumerate(prefixes):
                # Lock in one color per method/prefix for continuity
                color = color_cycle[i % len(color_cycle)]

                # 1. Plot energy band top
                self.plot_band(
                    path_index=p_idx, prefix=prefix, ax=ax_e, label=prefix, color=color
                )

                # 2. Extract shared x-coordinates (accumulated path) and plot forces bottom
                forcefit = self._prepare_force_fit(self.paths[p_idx], prefix)
                self.analyse_neb_force_decomposition(
                    self.paths[p_idx],
                    prefix=prefix,
                    ax=ax_f,
                    x_coords=forcefit.path,
                    color=color,
                )

            # Clean up axes & labeling
            ax_e.set_title(
                f"MEP Comparison & Force Breakdown\nReaction: {reaction_key}"
            )
            ax_e.set_ylabel("Relative Energy [eV]")
            ax_e.grid(True, linestyle="--", alpha=0.6)
            ax_e.legend(loc="upper right")
            # Ensure the only path label sits on the lower bottom plot
            ax_e.set_xlabel("")
            ax_f.set_xlabel("Path [Å]")

            plt.tight_layout()
            mep_figures[reaction_key] = fig

        return mep_figures

    def plot_fmax_distribution(self, prefixes=("MACE", "DFT")):
        """
        Analyzes and plots the fmax distribution of all frames across all pathways for each prefix.
        Generates a two-panel figure showing the fmax distribution for path endpoints
        and pathway-frames (intermediates) separately.

        Returns:
            tuple: (Figure, Axes) of the generated plot.
        """
        fmax_endpoints = {prefix: [] for prefix in prefixes}
        fmax_pathway = {prefix: [] for prefix in prefixes}

        for path in self.paths:
            for prefix in prefixes:
                try:
                    # Leverage the separate force component mathematical logic
                    fmax_list, _, _ = self.get_force_components(path, prefix)
                    if len(fmax_list) >= 2:
                        fmax_endpoints[prefix].append(fmax_list[0])
                        fmax_endpoints[prefix].append(fmax_list[-1])
                    if len(fmax_list) > 2:
                        fmax_pathway[prefix].extend(fmax_list[1:-1])
                except Exception as e:
                    logger.warning(
                        "Skipped a path for prefix %r during fmax distribution analysis: %s",
                        prefix,
                        e,
                    )

        fig, (ax_end, ax_mid) = plt.subplots(1, 2, figsize=(12, 5))
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for i, prefix in enumerate(prefixes):
            color = color_cycle[i % len(color_cycle)]
            # Endpoints histogram
            if fmax_endpoints[prefix]:
                ax_end.hist(
                    fmax_endpoints[prefix],
                    bins=20,
                    alpha=0.6,
                    color=color,
                    label=prefix,
                    edgecolor="black",
                    linewidth=0.5,
                )
            # Pathway-frames histogram
            if fmax_pathway[prefix]:
                ax_mid.hist(
                    fmax_pathway[prefix],
                    bins=20,
                    alpha=0.6,
                    color=color,
                    label=prefix,
                    edgecolor="black",
                    linewidth=0.5,
                )

        ax_end.set_title("Endpoints $f_{max}$ Distribution\n(Images 0 and N)")
        ax_end.set_xlabel("$f_{max}$ [eV/Å]")
        ax_end.set_ylabel("Frequency")
        ax_end.legend()
        ax_end.grid(True, linestyle="--", alpha=0.6)

        ax_mid.set_title("Pathway-Frames $f_{max}$ Distribution\n(Intermediate Images)")
        ax_mid.set_xlabel("$f_{max}$ [eV/Å]")
        ax_mid.set_ylabel("Frequency")
        ax_mid.legend()
        ax_mid.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()

        return fig, (ax_end, ax_mid)

    def save_plots(
        self,
        output_dir: str | Path,
        prefixes: tuple[str, ...] = ("MACE", "DFT"),
        dpi: int = 150,
        fmt: str = "png",
    ) -> dict[str, list[str]]:
        """Run all plot methods and save outputs to ``output_dir``.

        Calls :meth:`plot_multiple_pathways`, :meth:`plot_mep_pathways`, and
        :meth:`plot_fmax_distribution` in sequence, saving each figure to disk
        using deterministic filenames derived from the reaction string.

        Parameters
        ----------
        output_dir : str or Path
            Target directory. Created (including parents) if it does not exist.
        prefixes : tuple of str
            Method prefixes forwarded to all plot methods.
        dpi : int
            Resolution for raster output formats. Default 150.
        fmt : str
            File format extension (``"png"``, ``"pdf"``, ``"svg"``). Default ``"png"``.

        Returns
        -------
        dict[str, list[str]]
            Mapping of plot-type keys to lists of absolute file paths written:
            ``{"multiple_pathways": [...], "mep_pathways": [...], "fmax_distribution": [...]}``.

        Notes
        -----
        Reaction-string keys are sanitised for use as filenames by replacing
        spaces, ``>``, and ``->`` with underscores and stripping non-alphanumeric
        characters (except ``_`` and ``-``).
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved: dict[str, list[str]] = {
            "multiple_pathways": [],
            "mep_pathways": [],
            "fmax_distribution": [],
        }

        def _sanitise(reaction_key: str) -> str:
            """Convert a reaction string into a safe filename stem."""
            safe = (
                reaction_key.replace(" -> ", "_to_")
                .replace("->", "_to_")
                .replace(" ", "_")
            )
            return "".join(c for c in safe if c.isalnum() or c in ("_", "-"))

        # 1. plot_multiple_pathways
        reaction_figures = self.plot_multiple_pathways(prefixes=prefixes)
        for reaction_key, (fig, _axes) in reaction_figures.items():
            stem = _sanitise(reaction_key)
            fpath = str(out / f"multiple_pathways_{stem}.{fmt}")
            fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved["multiple_pathways"].append(fpath)
            logger.info("Saved multiple_pathways figure: %s", fpath)

        # 2. plot_mep_pathways (requires mep_indices populated by plot_multiple_pathways)
        mep_figures = self.plot_mep_pathways(prefixes=prefixes)
        for reaction_key, fig in mep_figures.items():
            stem = _sanitise(reaction_key)
            fpath = str(out / f"mep_pathways_{stem}.{fmt}")
            fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved["mep_pathways"].append(fpath)
            logger.info("Saved mep_pathways figure: %s", fpath)

        # 3. plot_fmax_distribution (single figure across all pathways)
        fig_fmax, _axes = self.plot_fmax_distribution(prefixes=prefixes)
        fpath = str(out / f"fmax_distribution.{fmt}")
        fig_fmax.savefig(fpath, dpi=dpi, bbox_inches="tight")
        plt.close(fig_fmax)
        saved["fmax_distribution"].append(fpath)
        logger.info("Saved fmax_distribution figure: %s", fpath)

        return saved
