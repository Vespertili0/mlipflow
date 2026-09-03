## DEPRECIATED ##

# from __future__ import annotations
#
# from mlipflow.core.single_point import run_single_point
# from mlipflow.qe_calculator import QECalculator
#
#
# def run_qe_relaxation(
#    in_file: str,
#    out_file: str,
#    basic_params: str,
#    pseudo_dir: str,
#    ecut_eV: int = 450,
#    kpts: tuple = (3, 3, 1),
#    dipole: bool = False,
#    dftd3: bool = True,
# ):
#    """
#    Run relaxation using QE directly.
#    """
#    dft = QECalculator(basic_params, pseudo_dir)
#    calculator = dft.get_calculator(
#        job_name="QE_",
#        ecut_eV=ecut_eV,
#        kpts=kpts,
#        calc_type="relax",
#        dipole=dipole,
#        dftd3=dftd3,
#    )
#    run_single_point(
#        in_file=in_file,
#        out_file=out_file,
#        output_prefix="DFT_",
#        calculator=calculator,
#        remote_info=dft.remote_info,
#    )
#
