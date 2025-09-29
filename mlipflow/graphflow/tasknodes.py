from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from mlipflow.core.single_point import run_single_point
from mlipflow.structure_generator import StructureGenStrategy, MDGen
from mlipflow.mlip_strategy import MLIPStrategy
from mlipflow.qe_calculator import QChemStrategy

#####################################################


class EnsembleState(TypedDict):
    configs: list[str]
    outfile: list[str] = None


def run_dft_sp(state, qchem_strategy: QChemStrategy):
    
    run_single_point(
        in_file=state['configs'],
        out_file=state['outfile'],
        output_prefix=qchem_strategy.qe_prefix,
        calculator=qchem_strategy.get_calculator(
            job_name='QE_',
            ecut_eV=450,
            kpts=(3,3,1),
            calc_type='scf'
        ),
        remote_info=qchem_strategy.remote_info
    )

def prepare_train_test_sets(state, split_ratio=0.8):
    pass

def assess_n_select(state, n_select=100):
    pass

def run_mace_fit(state, mlip_strategy: MLIPStrategy):
    mlip_strategy.fit_new_model(
        in_file=state['configs'],
        out_file='mace_model.out',
        train_fraction=0.8
    )
    
