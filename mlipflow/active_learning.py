import os, shutil, logging
import numpy as np
from ase.io import read
from wfl.calculators.generic import calculate as generic_calc
from wfl.configset import ConfigSet, OutputSpec
from wfl.autoparallelize import AutoparaInfo

from wfl.fit import error
from wfl.utils.configs import atomization_energy
from wfl.utils import logging
from mlipflow.data_manager import DataManager
from mlipflow.structure_generator import StructureGenStrategy, MDGen
from mlipflow.mlip_strategy import MLIPStrategy
from mlipflow.qe_calculator import QChemStrategy
#####################################################################

# contains all critical steps of the active-learning loop
class ActiveLearner:
    def __init__(
            self, 
            data_manager: DataManager = None,
            structure_generation_strategy: StructureGenStrategy = None,
            mlip_strategy: MLIPStrategy = None,
            qchem_strategy: QChemStrategy = None
            ) -> None:
        
        self.data_manager = data_manager
        self.structure_generation_strategy = structure_generation_strategy
        self.mlip_strategy = mlip_strategy
        self.qchem_strategy = qchem_strategy

    def initialise_learning(self, ensemble_traj, initial_xyz, qchem=False):
        """
        Run 0th-iteration of active-learning
        """
        self.data_manager.setup_iteration(fit_idx=0)
        self._write_log('initialise')        
        self.data_manager.initialise_ensembles(ensemble_traj=ensemble_traj)

        # run DFT-reference calculations
        if qchem == True:
            self._write_log('dft')
            self.run_single_point(
                in_file=self.data_manager.files["ensemble_xyz"], # !!!!
                out_file=self.data_manager.files['dft_0th'],
                output_prefix=self.qchem_strategy.qe_prefix,
                calculator=self.qchem_strategy.get_calculator(
                    job_name = f'QE_0'
                    ),
                remote_info=self.qchem_strategy.remote_info,
            )
            add_training = self.data_manager.files['dft_0th']
        else:
            self._write_log('custom', '...DFT provided')  
            add_training = self.data_manager.files["ensemble_xyz"]

        # prepare training-data and fit 0th-MLIP
        self.data_manager.update_training_data(
            training_xyz=initial_xyz,
            add_xyz=add_training,
            out_file=self.data_manager.files["training"]
        )
        self._write_log('fit')
        self.mlip_strategy.fit_new_model(
            in_file=self.data_manager.files["training"],
            model_name=f'{self.mlip_strategy.mlip_prefix}_0',
            run_dir=self.data_manager.MLIP_dir
        )

    def run_iteration(self, n_iter):
        """
        Set up active learning iteration

        Parameters
        ----------
        """
        self.data_manager.setup_iteration(fit_idx=n_iter)
        self.data_manager.get_model_name(
            fit_idx=n_iter,
            mlip_prefix=self.mlip_strategy.mlip_prefix
        )
        self.mlip_strategy.mlip_file = self.data_manager.files['mlip_model']
        
        # run MLIP single-point calculation
        self._write_log('sp')
        self.run_single_point(
            in_file=self.data_manager.files["training"], # !!!!
            out_file=self.data_manager.files['eval_train'],
            output_prefix=f'{self.mlip_strategy.mlip_prefix}_',
            calculator=self.mlip_strategy.get_calculator(
                job_name = f'mSP_{n_iter}'
                ),
            remote_info=self.mlip_strategy.remote_info
        )
        # generate new structures via MLIP-MD/OPT/NEB
        self._write_log('MD')
        if isinstance(self.structure_generation_strategy, MDGen):
            in_file = self.data_manager.files["ensemble_xyz"]
        else:
            in_file = self.data_manager.files["ensemble_traj"]    

        self.structure_generation_strategy.generate_new_structures(
            in_file=in_file,
            out_file=self.data_manager.files['mlip_output'],
            calculator=self.mlip_strategy.get_calculator(
                job_name = f'mMP_{n_iter}'
                ),
            remote_info=self.mlip_strategy.remote_info
        )
        # run DFT single-point calculation and clean data
        self._write_log('dft')
        self.run_single_point(
            in_file=self.data_manager.files["mlip_output"], # !!!!
            out_file=self.data_manager.files['dft_output'],
            output_prefix=self.qchem_strategy.qe_prefix,
            calculator=self.qchem_strategy.get_calculator(
                job_name = f'QE_{n_iter}'
                ),
            remote_info=self.qchem_strategy.remote_info,
        )
        self.data_manager.check_maxforce_and_cleanarrays(
            in_file=self.data_manager.files['dft_output'], 
            out_file=self.data_manager.files['eval_test'], 
            calc='md',        # should interface with StructureGenStrategy !!!
            mlip_prefix=self.mlip_strategy.mlip_prefix,
            max_force=15
        )
        # calculate MLIP-error
        self._write_log('custom', '...Assessing errors')
        self.calculate_mlip_error(
            in_configs=[
                self.data_manager.files['eval_train'],
                self.data_manager.files['eval_test']
                ],
            out_file=self.data_manager.files['eval'],
            fit_idx=n_iter,
            iter_dir=self.data_manager.iter_dir
        )      
        # prepare training-data and fit new MLIP
        self._write_log('fit')
        old_training = self.data_manager.files["training"]
        old_eval = self.data_manager.files['eval_test']
        
        self.data_manager.setup_iteration(fit_idx=n_iter+1)
        
        self.data_manager.update_training_data(
            training_xyz=old_training,
            add_xyz=old_eval,
            out_file=self.data_manager.files["training"]
        )
        self.mlip_strategy.fit_new_model(
            in_file=self.data_manager.files["training"],
            model_name=f'{self.mlip_strategy.mlip_prefix}_{n_iter+1}',
            run_dir=self.data_manager.MLIP_dir
        )

        # clean up temp-files
        self._write_log('custom', '...Cleaning up tmp-dirs')
        self.data_manager.clean_up()


    def run_single_point(self, in_file, out_file, output_prefix,
                         calculator, remote_info=None) -> None:
        """
        Runs single-point calculation on ase-configs using GAP-file provided

        Parameters
        ----------

        """
        in_config = ConfigSet(in_file)
        out_config = OutputSpec(out_file)

        # calculate GAP-energies locally   
        if remote_info is None:
            
            generic_calc(
                inputs=in_config,
                outputs=out_config,
                calculator=calculator,
                output_prefix=output_prefix,
                properties=["energy", "forces"]
                )

        # calculate GAP-energy remotely
        elif remote_info:

            generic_calc(
                inputs=in_config,
                outputs=out_config,
                calculator=calculator,
                output_prefix=output_prefix,
                properties=["energy", "forces"],
                autopara_info=AutoparaInfo(
                    remote_info=remote_info,
                    num_inputs_per_python_subprocess=1
                    )
                )


#    def calculate_mlip_error(self, in_configs, out_file,
#                             fit_idx, iter_dir) -> dict:
#        """
#        """
#        OutputSpec(out_file).write(ConfigSet(in_configs))
#        
#        # calculate atomisation-energy for DFT- & GAP-energy
#        for prop in [f'{self.mlip_strategy.mlip_prefix}_', 'DFT_']:
#            atomization_energy(
#                inputs=ConfigSet(out_file), 
#                outputs=OutputSpec(out_file, overwrite=True), 
#                prop_prefix=prop
#                )
#
#        # calculate errors
#        errors, diffs, parity = error.calc(
#            inputs=ConfigSet(out_file),
#            calc_property_prefix=f'{self.mlip_strategy.mlip_prefix}_',
#            ref_property_prefix='DFT_',
#            category_keys='data_type', 
#            config_properties=["atomization_energy/atom"], #"energy/atom"
#            atom_properties=["forces/comp"]
#            )
#        
#        # plot errors
#        for error_type in ['RMSE', 'MAE']:
#            error.value_error_scatter(
#                all_errors = errors,
#                all_diffs=diffs, 
#                all_parity=parity,
#                output=os.path.join(iter_dir, f"{self.mlip_strategy.mlip_prefix}_{error_type}.png"),
#                ref_property_prefix='DFT_',
#                calc_property_prefix=f'{self.mlip_strategy.mlip_prefix}_',
#                error_type=error_type
#                )
#        
#        return errors
#
#
#    def run_chunked_dftsp(self, in_file, out_file, chunk_size=150)-> None:
#        """
#        Run DFT single-point calculation in chunks of input file
#        """
#        chunk_list = self._chunk_indices(in_file, chunk_size=chunk_size)
#        chunk_files = [f'tmp_{n}.xyz' for n in range(len(chunk_list))]
#        for n, chunk in enumerate(chunk_list):
#            # run SP-calculation
#            atoms = read(in_file, index=chunk)
#            self.run_single_point(
#                in_file=atoms, 
#                out_file=f'tmp_{n}.xyz', 
#                output_prefix=self.qchem_strategy.qe_prefix,
#                calculator=self.qchem_strategy.get_calculator(job_name='QE_'), 
#                remote_info=self.qchem_strategy.remote_info
#            )
#            # clean up chunk directories
#            self.data_manager.clean_up()
#        
#        # combine all chunks
#        self.data_manager.merge_clean_chunks(
#            in_files=chunk_files,
#            out_file=out_file
#        )
#        # clean up tmp-files
#        self.data_manager.clean_up(key='tmp_')
#
#
#    def _chunk_indices(self, in_file, chunk_size=150)-> list:
#        """
#        Split input file into chunks of size chunk_size
#        """
#        n_configs = len(read(in_file, index=':'))
#        return [f'{i}:{min(i+chunk_size, n_configs)}' for i in range(0, n_configs, chunk_size)]


    def _write_log(self, step, comment=None):
        """
        Parameters
        ----------
        step:       str 
            option from 'md', 'dft', 'fit', 'desc', 
            'criteria' or 'custom' 
        comment:    str
            message printed if 'custom' selected
        """
        if step == 'dft':
            msg = '...Running DFT'
        elif step == 'fit':
            msg = '...Fitting new MLIP'
        elif step == 'criteria':
            msg = '...Checking termination criteria'
        elif step == 'error':
            msg = '...Assessing errors'
        elif step == 'custom':
            msg = comment
        elif step == 'initialise':
            msg = '\n' + '-'*50 + '\n' + ' '*22 + 'mlipflow (v0.1)\n' + '-'*50 + '\n'
            msg += f'\nInitialising 0th-{self.mlip_strategy.mlip_prefix}'
        else:
            msg = f'...Running {self.mlip_strategy.mlip_prefix}-{step.upper()}'

        n_space = 59 - len(msg.split('\n')[-1])
        msg += ' ' * n_space

        with open(f'{self.data_manager.iter_dir}/../MLIPlog.log', 'a') as logfile:
            logging.print_log(msg=msg, logfile=logfile)

        return None