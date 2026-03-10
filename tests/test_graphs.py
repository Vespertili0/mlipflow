import os
import shutil
import pytest
from unittest.mock import MagicMock
from ase.io import read
from ase.constraints import FixAtoms
from mlipflow.graphflow.nodes import EnsembleState
from mlipflow.graphflow.graphs import execute_initial_basin_pathsampling_md_block
from mlipflow.strategies.mlip import MACEModel
from mlipflow.strategies.structure_generators import MDGen
from mlipflow.strategies.dft import EMTCalc

def test_execute_initial_basin_pathsampling_md_block(tmp_path):
    """
    Test execute_initial_basin_pathsampling_md_block with MACE and MDGen.
    We mock MACE calculator with EMTCalc to ensure compatibility.
    """
    # Setup paths
    test_dir = os.path.dirname(os.path.abspath(__file__))
    src_xyz = os.path.join(test_dir, 'data', 'test_data.xyz')
    # src_model = os.path.join(test_dir, 'data', 'mace_test.model')

    config_file = tmp_path / "test_data.xyz"
    # model_file = tmp_path / "mace_test.model"

    shutil.copy2(src_xyz, config_file)
    # shutil.copy2(src_model, model_file)

    # Change directory to tmp_path to capture outputs
    cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        # Instantiate strategies
        # mlip_name should be 'mace_test' so it looks for 'mace_test.model'
        # We don't need the actual model file if we mock get_calculator
        mlip_strategy = MACEModel(mlip_name="mace_test", run_mode="local")

        # Mock get_calculator to return EMT calculator
        # The signature of get_calculator is (job_name, dispersion=True, ...)
        # We return (EMT, [], kwargs) so wfl uses EMT(**kwargs)
        mlip_strategy.remote_info = None
        # Use generic parameters for EMT
        # Use EMTCalc().get_calculator for the mock side_effect to match the return signature
        mlip_strategy.get_calculator = MagicMock(side_effect=EMTCalc().get_calculator)

        # MDGen with minimal steps
        structure_gen_strategy = MDGen(
            uncertainty_thrs=float('inf'), # Infinite threshold to ensure MD runs even with bad random structures
            n_failed_steps=2,
            params={'steps': 5, 'dt': 1.0, 'temperature': 300.0, 'traj_step_interval': 1}
        )

        # QChem strategy (dummy)
        qchem_strategy = EMTCalc()

        # Constraints
        # Fix first atom (index 0)
        constraints = [FixAtoms(indices=[0])]

        # Reaction constraints for NEB
        # tFUR+2H -> tFURao+H
        rxn_constraints_dict = {
            'tFUR+2H -> tFURao+H': constraints
        }

        calculation_kwargs = {
            'initial_sampling': {
                'basin_constraints': constraints,
                'neb_config': {
                    'rxn_constraints_dict': rxn_constraints_dict,
                    'method': 'random',
                    'n_pathways': 1,
                    'n_images': 3
                }
            },
            'mlip_gen': {'dispersion': False}, # Disable dispersion to avoid dftd3 dependency issues if any
            'mlip_sp': {'dispersion': False},
            'fps_selection': {
                'descriptor_string': 'soap n_species=4 species_Z={1 6 8 29} l_max=6 n_max=8 cutoff=3.5 atom_sigma=0.5 zeta=6',
                'info_field': 'MACE_energy',
                'n_optimal': 5
            }
        }

        state = EnsembleState(
            configs=['test_data.xyz'],
            qchem_strategy=qchem_strategy,
            mlip_strategy=mlip_strategy,
            structure_gen_strategy=structure_gen_strategy,
            calculation_kwargs=calculation_kwargs
        )

        # Compile and run
        app = execute_initial_basin_pathsampling_md_block()
        result = app.invoke(state)

        # Verify
        assert result['outfile'] is None
        assert result['configs'] is not None
        assert len(result['configs']) > 0

        # Check files
        for f in result['configs']:
            assert os.path.exists(f)
            assert f.endswith('final_selection.xyz')

            atoms = read(f, ':')
            assert len(atoms) > 0

            # Check for MACE results
            # run_mlip_sp uses output_prefix=mlip.mlip_prefix which is 'MACE_'
            for at in atoms:
                assert 'MACE_energy' in at.info
                # For EMT, forces are computed.
                # Note: 'MACE_forces' might be missing if single_point didn't run properly?
                # But run_mlip_sp calls run_single_point which requests properties=["energy", "forces"]
                assert 'MACE_forces' in at.arrays

    finally:
        os.chdir(cwd)
