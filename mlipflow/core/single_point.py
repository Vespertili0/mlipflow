from wfl.configset import ConfigSet, OutputSpec
from wfl.calculators.generic import calculate as generic_calc
from wfl.autoparallelize import AutoparaInfo
from ase.io import read


def run_single_point(in_file, out_file, output_prefix, calculator, remote_info=None):
    """
    Run a single point calculation using the provided calculator.   
    """
    
    in_config = ConfigSet(in_file)
    out_config = OutputSpec(out_file)
    if remote_info is None:
        generic_calc(
            inputs=in_config,
            outputs=out_config,
            calculator=calculator,
            output_prefix=output_prefix,
            properties=["energy", "forces"]
        )
    else:
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


def run_chunked_sp(in_file, out_file, chunk_size, qchem_strategy, data_manager,
                   ecut_eV: int = 450, kpts: tuple = (3,3,1),
                   dipole: bool = True, dftd3: bool = True)->None:
    """
    Run a chunked single point calculation using the provided calculator.
    This function splits the input file into chunks of a specified size,
    runs single point calculations on each chunk, and merges the results
    into a single output file.
    """
    chunk_list = _chunk_indices(in_file, chunk_size=chunk_size)
    chunk_files = [f'tmp_{n}.xyz' for n in range(len(chunk_list))]
    
    # run single point calculations on each chunk
    for n, chunk in enumerate(chunk_list):
        atoms = read(in_file, index=chunk)
        run_single_point(
            in_file=atoms,
            out_file=f'tmp_{n}.xyz',
            output_prefix=qchem_strategy.qe_prefix,
            calculator=qchem_strategy.get_calculator(
                job_name='QE_', 
                ecut_eV=ecut_eV, 
                kpts=kpts, 
                dipole=dipole, 
                dftd3=dftd3
            ),
            remote_info=qchem_strategy.remote_info
        )
        data_manager.clean_up()
    
    # merge all chunks into one file
    data_manager.merge_clean_chunks(
        in_files=chunk_files,
        out_file=out_file
    )
    
    # remove the temporary files
    data_manager.clean_up(key='tmp_')


def _chunk_indices(in_file: str, chunk_size:int = 150)->list:
    n_configs = len(read(in_file, index=':'))
    return [f'{i}:{min(i+chunk_size, n_configs)}' for i in range(0, n_configs, chunk_size)]