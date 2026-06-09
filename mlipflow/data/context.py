"""Legacy data management context.

.. deprecated::
    The ``DataManager`` class is deprecated. Use
    :func:`mlipflow.utils.path_factory.create_iteration_directory` and
    :func:`mlipflow.utils.path_factory.resolve_step_path` for path
    management, and the standalone functions in
    :mod:`mlipflow.data.processing` for data utilities.
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

from ase.io import read, write
from wfl.configset import ConfigSet, OutputSpec

from mlipflow.data.processing import (
    check_maxforce_and_cleanarrays,
    filter_info_dict,
    merge_clean_chunks,
    split_success_failed_configs,
    update_training_data,
)
from mlipflow.utils.path_factory import create_iteration_directory

_DEPRECATION_MSG = (
    "DataManager is deprecated and will be removed in a future release. "
    "Use mlipflow.utils.path_factory for path management and "
    "mlipflow.data.processing for data utilities."
)


# manage incoming/outgoing data
class DataManager:
    """Manages data and file operations for MLIP training.

    .. deprecated::
        This class is deprecated. Use the centralised path factory
        (``mlipflow.utils.path_factory``) for path management and
        standalone functions in ``mlipflow.data.processing`` for data
        processing utilities.

    Attributes
    ----------
    workdir : str
        Working directory for all operations.
    files : dict
        Dictionary of file paths.
    mlip_file_fmt : dict
        File format mappings for different MLIP types.
    """

    def __init__(self, workdir: str) -> None:
        warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        self.workdir = workdir
        self.files = {}
        self.mlip_file_fmt = {"MACE": "model", "GAP": "xml"}

    def setup_iteration(self, fit_idx: int) -> None:
        """Set up the folder structure for a new training iteration.

        .. deprecated::
            Use :func:`mlipflow.utils.path_factory.create_iteration_directory`
            instead.
        """
        warnings.warn(
            "setup_iteration is deprecated. Use "
            "mlipflow.utils.path_factory.create_iteration_directory instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        dirs = create_iteration_directory(fit_idx, workdir=self.workdir)
        self.iter_dir = dirs["iter_dir"]
        self.MLIP_dir = dirs["mlip_dir"]
        self.SGen_dir = dirs["sgen_dir"]
        self._ensemble_dir = dirs["ensemble_dir"]

        # general structure-files
        def f1(x, y):
            return f"{self.SGen_dir}/{x}{y}_{fit_idx}.xyz"

        self.files["dft_0th"] = f1("dft", "")
        self.files["mlip_output"] = f1("sgen", "_mlip")
        self.files["dft_output"] = f1("sgen", "_dft")
        self.files["eval"] = f1("eval", "")
        self.files["eval_train"] = f1("eval", "_train")
        self.files["eval_test"] = f1("eval", "_test")

        self.files["training"] = str(Path(self.MLIP_dir) / f"training_{fit_idx}.xyz")
        self.files["ensemble_traj"] = str(Path(self._ensemble_dir) / "ensemble.traj")
        self.files["ensemble_xyz"] = str(Path(self._ensemble_dir) / "ensemble.xyz")

        # GAP-specific files
        def f2(x):
            return f"{self.MLIP_dir}/GAP_{fit_idx}.xml{x}"

        self.files["desc"] = f1("desc", "")
        self.files["fps"] = f1("sgen", "_fps")
        self.files["training_desc"] = f1("training_desc", "")
        self.files["gap_params"] = f2(".descriptor_dicts.yaml")

    def get_model_name(self, fit_idx: int, mlip_prefix: str) -> None:
        """Write the model name to the files dictionary.

        Parameters
        ----------
        fit_idx : int
            The index of the current fit iteration.
        mlip_prefix : str
            The prefix for the MLIP model
            (e.g., ``'.model'`` for ``'MACE'`` or ``'.xml'`` for ``'GAP'``).
        """
        model_fmt = self.mlip_file_fmt.get(mlip_prefix)
        self.files["mlip_model"] = str(
            Path(self.MLIP_dir) / f"{mlip_prefix}_{fit_idx}.{model_fmt}"
        )

    def update_training_data(self, training_xyz, add_xyz, out_file) -> None:
        """Update training data by merging two XYZ files.

        .. deprecated::
            Delegates to :func:`mlipflow.data.processing.update_training_data`.
        """
        warnings.warn(
            "DataManager.update_training_data is deprecated. Use "
            "mlipflow.data.processing.update_training_data instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        update_training_data(training_xyz, add_xyz, out_file)

    def initialise_ensembles(self, ensemble_traj: str) -> None:
        """Initialise ensemble from trajectory file."""
        if not Path(ensemble_traj).exists():
            raise FileNotFoundError(
                f"Ensemble trajectory file {ensemble_traj} not found"
            )
        try:
            shutil.copy2(ensemble_traj, self.files.get("ensemble_traj"))
            configs = read(self.files.get("ensemble_traj"), ":")
            write(self.files.get("ensemble_xyz"), configs)
        except Exception as e:
            raise RuntimeError(f"Failed to initialise ensembles: {e!s}") from e

    def move_mace_model_file(self, file_prefix: str) -> None:
        """Move the compiled MACE model file to the MLIP directory."""
        shutil.copy2(
            Path(self.MLIP_dir) / "MACE_model" / f"{file_prefix}_compiled.model",
            Path(self.MLIP_dir) / f"{file_prefix}.model",
        )

    def check_maxforce_and_cleanarrays(
        self, in_file, out_file, mlip_prefix, calc, max_force=15
    ):
        """Remove structures with forces exceeding threshold.

        .. deprecated::
            Delegates to :func:`mlipflow.data.processing.check_maxforce_and_cleanarrays`.
        """
        warnings.warn(
            "DataManager.check_maxforce_and_cleanarrays is deprecated. Use "
            "mlipflow.data.processing.check_maxforce_and_cleanarrays instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        check_maxforce_and_cleanarrays(in_file, out_file, mlip_prefix, calc, max_force)

    def update_configset_tag(self, in_config, out_file, tag_dict):
        """Update a wfl.configset with a new tag.

        Parameters
        ----------
        in_config : items
            Configurations to store.
        out_file : str or Path
            Output file path.
        tag_dict : dict
            Extra ``Atoms.info`` keys to set in written configs.
        """
        configs = ConfigSet(in_config)
        OutputSpec(out_file, tags=tag_dict, overwrite=True).write(configs)

    def filter_info_dict(self, info_dict: dict, keep_info: list) -> dict:
        """Filter the info dictionary to keep only specified keys.

        .. deprecated::
            Delegates to :func:`mlipflow.data.processing.filter_info_dict`.
        """
        return filter_info_dict(info_dict, keep_info)

    def split_success_failed_configs(
        self, configs: list, key: str = "DFT_energy"
    ) -> tuple[list, list]:
        """Split configurations into successful and failed.

        .. deprecated::
            Delegates to :func:`mlipflow.data.processing.split_success_failed_configs`.
        """
        return split_success_failed_configs(configs, key)

    def merge_clean_chunks(self, in_files, out_file) -> None:
        """Merge all chunks into one file and clean up.

        .. deprecated::
            Delegates to :func:`mlipflow.data.processing.merge_clean_chunks`.
        """
        warnings.warn(
            "DataManager.merge_clean_chunks is deprecated. Use "
            "mlipflow.data.processing.merge_clean_chunks instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        merge_clean_chunks(in_files, out_file)

    def clean_up(self, key="_chunk_") -> None:
        """Remove temporary chunk directories and files."""
        try:
            run_dirs = [rd for rd in Path().iterdir() if key in rd.name]
            for rd in run_dirs:
                if rd.is_dir():
                    shutil.rmtree(rd)
                elif rd.is_file():
                    rd.unlink()
        except Exception as e:
            raise RuntimeError(f"Error removing {rd}: {e}") from e

    def _create_folder_structure(self, fit_idx):
        """Create the iteration folder structure.

        .. deprecated::
            Use :func:`mlipflow.utils.path_factory.create_iteration_directory`
            instead.
        """
        warnings.warn(
            "_create_folder_structure is deprecated. Use "
            "mlipflow.utils.path_factory.create_iteration_directory instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        dirs = create_iteration_directory(fit_idx, workdir=self.workdir)
        self._ensemble_dir = dirs["ensemble_dir"]
        self.iter_dir = dirs["iter_dir"]
        self.MLIP_dir = dirs["mlip_dir"]
        self.SGen_dir = dirs["sgen_dir"]

    @property
    def ensemble_dir(self) -> str:
        """Path to ensemble directory."""
        return str(Path(self.workdir) / "ENSEMBLE")

    @property
    def mlip_dir(self) -> str:
        """Path to MLIP directory."""
        return str(Path(self.iter_dir) / "MLIP")
