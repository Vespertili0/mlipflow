import os, shutil
import numpy as np
from ase.io import read, write
from wfl.configset import ConfigSet, OutputSpec

# manage incoming/outgoing data
class DataManager:
    def __init__(self, workdir) -> None:
        self.workdir = workdir
        self.files = {}
        self.mlip_file_fmt = {
            'MACE': 'model', 
            'GAP': 'xml'
        }
        

    def setup_iteration(self, fit_idx: int) -> None:
        """
        Function to set up the folder structure for a new iteration of the MLIP training.
        This includes creating the necessary directories and defining the file paths for various outputs.
        """
        self._create_folder_structure(fit_idx=fit_idx)
        
        # general structure-files
        f1 = lambda x, y: f"{self.SGen_dir}/{x}{y}_{fit_idx}.xyz"
        self.files["dft_0th"] = f1("dft", '')
        self.files["mlip_output"] = f1("sgen", '_mlip')
        self.files["dft_output"] = f1("sgen", '_dft')
        self.files["eval"] = f1("eval", '')
        self.files["eval_train"] = f1("eval", '_train')
        self.files["eval_test"] = f1("eval", '_test')

        self.files["training"] = os.path.join(self.MLIP_dir, f'training_{fit_idx}.xyz')
        self.files["ensemble_traj"] = os.path.join(self.ensemble_dir, 'ensemble.traj')
        self.files["ensemble_xyz"] = os.path.join(self.ensemble_dir, 'ensemble.xyz')
        
        # GAP-specific files
        f2 = lambda x: f"{self.MLIP_dir}/GAP_{fit_idx}.xml{x}"
        self.files["desc"] = f1("desc", '')
        self.files["fps"]  = f1("sgen", '_fps')
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
        self.files['mlip_model'] = os.path.join(
            self.MLIP_dir,
            f'{mlip_prefix}_{fit_idx}.{model_fmt}'
        )

    def update_training_data(self, training_xyz, add_xyz, out_file)->None:
        """
        """
        new_training = read(training_xyz, ':') 
        new_training += read(add_xyz, ':')
        self._update_configset_tag(
            in_config=new_training,
            out_file=out_file,
            tag_dict={'data_type': 'train'}
        )

    def initialise_ensembles(self, ensemble_traj: str) -> None:
        """
        Function copying provided ase-traj file of ensemble to xyz-format.
        The xyz-file is used for the training of the MLIP, while the traj-file is used for the exploration.
        """
        shutil.copy2(
            ensemble_traj, 
            self.files.get("ensemble_traj")
        )
        write(
            self.files.get("ensemble_xyz"),
            read(self.files.get("ensemble_traj"), ':')
        )
    
    def move_mace_model_file(self, mlip_prefix: str) -> None:
        """
        Function to move the compiled MACE model file to the MLIP directory.
        """
        shutil.copy2(
            os.path.join(self.MLIP_dir, 'MACE_model', f'{mlip_prefix}_compiled.model'),
            os.path.join(self.MLIP_dir, f'{mlip_prefix}.model')
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
        keys = {'md': 'md', 'opt': 'optimize'}
        array_keys = ['numbers', 'positions', 'tags', 'DFT_forces', f'last_op__{keys.get(calc)}_forces'] #, 'GAP_uncertainty_meV'
        info_keys = ['MD_step', 'DFT_energy', f'last_op__{keys.get(calc)}_energy', 'config_type', 'data_type']

        all_configs = [a for a in read(in_file, ':') if 'DFT_energy' in a.info.keys()]
        
        assert all_configs, 'no valid structures from DFT! check data'
        
        selected_configs = [a for a in all_configs if np.max(np.abs(a.arrays['DFT_forces'])) <= max_force] 
        for config in selected_configs: 
            config.arrays = {
                (f'{mlip_prefix}_forces' if key == f'last_op__{keys.get(calc)}_forces' else key): config.arrays.get(key) for key in array_keys
            }
            config.info = {
                (f'{mlip_prefix}_energy' if key == f'last_op__{keys.get(calc)}_energy' else key): config.info.get(key) for key in info_keys
            }

        write(out_file, selected_configs)

    def _update_configset_tag(self, in_config, out_file, tag_dict):
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

        return None

    def _create_folder_structure(self, fit_idx):
        self.ensemble_dir = os.path.join(self.workdir, 'ENSEMBLE')
        self.iter_dir = os.path.join(self.workdir, f'{fit_idx}_iteration')
        self.MLIP_dir = os.path.join(self.iter_dir, 'MLIP')
        self.SGen_dir = os.path.join(self.iter_dir, 'SGEN')

        for folder in [self.ensemble_dir, self.SGen_dir, self.MLIP_dir]:
            os.makedirs(folder, exist_ok=True)
    