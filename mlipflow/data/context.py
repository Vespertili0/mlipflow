from __future__ import annotations

import os
import shutil

import numpy as np
from ase.io import read, write
from wfl.configset import ConfigSet, OutputSpec


# manage incoming/outgoing data
class DataManager:
    """
    Manages data and file operations for ML Interatomic Potential (MLIP) training.

    Handles file organization, data preprocessing, and structure management for
    iterative MLIP training workflows.

    Attributes:
        workdir (str): Working directory for all operations
        files (dict): Dictionary of file paths
        mlip_file_fmt (dict): File format mappings for different MLIP types
    """
    def __init__(self, workdir: str) -> None:
        self.workdir = workdir
        self.files = {}
        self.mlip_file_fmt = {
            "MACE": "model",
            "GAP": "xml"
        }


    def setup_iteration(self, fit_idx: int) -> None:
        """
        Function to set up the folder structure for a new iteration of the MLIP training.
        This includes creating the necessary directories and defining the file paths for various outputs.
        """
        self._create_folder_structure(fit_idx=fit_idx)

        # general structure-files
        f1 = lambda x, y: f"{self.SGen_dir}/{x}{y}_{fit_idx}.xyz"
        self.files["dft_0th"] = f1("dft", "")
        self.files["mlip_output"] = f1("sgen", "_mlip")
        self.files["dft_output"] = f1("sgen", "_dft")
        self.files["eval"] = f1("eval", "")
        self.files["eval_train"] = f1("eval", "_train")
        self.files["eval_test"] = f1("eval", "_test")

        self.files["training"] = os.path.join(self.MLIP_dir, f"training_{fit_idx}.xyz")
        self.files["ensemble_traj"] = os.path.join(self.ensemble_dir, "ensemble.traj")
        self.files["ensemble_xyz"] = os.path.join(self.ensemble_dir, "ensemble.xyz")

        # GAP-specific files
        f2 = lambda x: f"{self.MLIP_dir}/GAP_{fit_idx}.xml{x}"
        self.files["desc"] = f1("desc", "")
        self.files["fps"]  = f1("sgen", "_fps")
        self.files["training_desc"] = f1("training_desc", "")
        self.files["gap_params"] = f2(".descriptor_dicts.yaml")

    def get_model_name(self, fit_idx: int, mlip_prefix: str) -> None:
        """
        Function to write the model name to the files dictionary.
        The model name is based on the MLIP prefix and the fit index.

        fit_idx: int
            The index of the current fit iteration.
        mlip_prefix: str
            The prefix for the MLIP model
            (e.g., '.model' for 'MACE' or '.xml' for 'GAP').
        """
        model_fmt = self.mlip_file_fmt.get(mlip_prefix)
        self.files["mlip_model"] = os.path.join(
            self.MLIP_dir,
            f"{mlip_prefix}_{fit_idx}.{model_fmt}"
        )

    def update_training_data(self, training_xyz, add_xyz, out_file)->None:
        """
        """
        new_training = read(training_xyz, ":")
        new_training += read(add_xyz, ":")
        self.update_configset_tag(
            in_config=new_training,
            out_file=out_file,
            tag_dict={"data_type": "train"}
        )

    def initialise_ensembles(self, ensemble_traj: str) -> None:
        """Initialise ensemble from trajectory file."""
        if not os.path.exists(ensemble_traj):
            raise FileNotFoundError(f"Ensemble trajectory file {ensemble_traj} not found")
        try:
            shutil.copy2(ensemble_traj, self.files.get("ensemble_traj"))
            configs = read(self.files.get("ensemble_traj"), ":")
            write(self.files.get("ensemble_xyz"), configs)
        except Exception as e:
            raise RuntimeError(f"Failed to Initialise ensembles: {e!s}")

    def move_mace_model_file(self, file_prefix: str) -> None:
        """
        Function to move the compiled MACE model file to the MLIP directory.
        """
        shutil.copy2(
            os.path.join(self.MLIP_dir, "MACE_model", f"{file_prefix}_compiled.model"),
            os.path.join(self.MLIP_dir, f"{file_prefix}.model")
        )

    def check_maxforce_and_cleanarrays(self, in_file, out_file, mlip_prefix, calc, max_force=15):
        """
        Function to remove structures from ase-file with forces exceeding threshold.
        Cleans up the at.arrays and at.info, only keeping relevant data.

        in_file:    path/str
            ase-readable file-format; will be overwritten with cleaned
        max_force:  float/int
            threshold for DFT-forces in eV/Å
        """
        keys = {"md": "md", "opt": "optimize"}
        array_keys = ["numbers", "positions", "tags", "DFT_forces", f"last_op__{keys.get(calc)}_forces"] #, 'GAP_uncertainty_meV'
        info_keys = ["MD_step", "DFT_energy", f"last_op__{keys.get(calc)}_energy", "config_type", "data_type"]

        all_configs = [a for a in read(in_file, ":") if "DFT_energy" in a.info]

        assert all_configs, "no valid structures from DFT! check data"

        selected_configs = [a for a in all_configs if np.max(np.abs(a.arrays["DFT_forces"])) <= max_force]
        for config in selected_configs:
            config.arrays = {
                (f"{mlip_prefix}_forces" if key == f"last_op__{keys.get(calc)}_forces" else key): config.arrays.get(key) for key in array_keys
            }
            config.info = {
                (f"{mlip_prefix}_energy" if key == f"last_op__{keys.get(calc)}_energy" else key): config.info.get(key) for key in info_keys
            }

        write(out_file, selected_configs)

    def update_configset_tag(self, in_config, out_file, tag_dict):
        """
        Function to update a wfl.configset with a new tag

        Parameters
        ----------
        in_config:  items (Atoms / list(Atoms) / list(list...(Atoms)) / str / Path / list(str) / list(Path))
            configurations to store, or list of file name globs for the configurations.
        out_file:   str / Path / iterable(str / Path
            list of files to store configs in
        tag_dict:   dict
            dict of extra Atoms.info keys to set in written configs
        """
        configs = ConfigSet(in_config)
        OutputSpec(out_file, tags=tag_dict, overwrite=True).write(configs)

        return

    def filter_info_dict(self, info_dict: dict, keep_info: list) -> dict:
        return {key: info_dict[key] for key in keep_info if key in info_dict}


    def split_success_failed_configs(self, configs:list, key:str ="DFT_energy") -> tuple[list, list]:
        """
        Split the configurations into successful and failed ones based on the presence of a key in the info dictionary.
        Parameters
        ----------
        configs:    list
            list of configurations to split
        key:        str
            key to check for in the info dictionary
            failed configurations DO NOT have this key.

        Returns
        -------
        tuple (success_configs, failed_configs)
        """
        success_configs = []
        failed_configs = []
        for config in configs:
            if key not in config.info:
                failed_configs.append(config)
            else:
                success_configs.append(config)

        return success_configs, failed_configs


    def merge_clean_chunks(self, in_files, out_file)-> None:
        """
        Merge all chunks into one file and clean up the data.info
        write the failed and successful configurations into separate files.
        Parameters
        ----------
        in_files:   list
            list of input files to merge
        out_file:   str
            name of the output file
        """
        success_configs = []
        failed_configs = []
        # loop over all chunks
        for file in in_files:
            success, failed = self.split_success_failed_configs(
                configs=read(file, ":"),
                key="DFT_energy"
            )
            for config in success:
                config.info = self.filter_info_dict(
                    info_dict=config.info,
                    keep_info=["DFT_energy"]
                )
            success_configs += success
            failed_configs += failed
        # write new files
        OutputSpec(out_file, tags={"data_type": "train"}).write(ConfigSet(success_configs))
        name, ext = os.path.splitext(out_file)
        OutputSpec(f"{name}_failed{ext}").write(ConfigSet(failed_configs))


    def clean_up(self, key="_chunk_") -> None:
        try:
            run_dirs = [rd for rd in os.listdir() if key in rd]
            for rd in run_dirs:
                if os.path.isdir(rd):
                    shutil.rmtree(rd)
                elif os.path.isfile(rd):
                    os.remove(rd)
        except Exception as e:
            raise RuntimeError(f"Error removing {rd}: {e}")


    def _create_folder_structure(self, fit_idx):
        self.ensemble_dir = os.path.join(self.workdir, "ENSEMBLE")
        self.iter_dir = os.path.join(self.workdir, f"{fit_idx}_iteration")
        self.MLIP_dir = os.path.join(self.iter_dir, "MLIP")
        self.SGen_dir = os.path.join(self.iter_dir, "SGEN")

        for folder in [self.ensemble_dir, self.SGen_dir, self.MLIP_dir]:
            os.makedirs(folder, exist_ok=True)

    @property
    def ensemble_dir(self) -> str:
        """Path to ensemble directory."""
        return os.path.join(self.workdir, "ENSEMBLE")

    @property
    def mlip_dir(self) -> str:
        """Path to MLIP directory."""
        return os.path.join(self.iter_dir, "MLIP")
