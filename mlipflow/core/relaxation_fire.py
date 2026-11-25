from __future__ import annotations

import sys

from ase.optimize import FIRE
from wfl.autoparallelize import autoparallelize, autoparallelize_docstring
from wfl.generate.utils import save_config_type
from wfl.utils.misc import atoms_to_list
from wfl.utils.parallel import construct_calculator_picklesafe
from wfl.utils.save_calc_results import at_copy_save_calc_results


def _run_autopara_wrappable(
    atoms,
    calculator,
    fmax=1.0e-3,
    _smax=None,
    steps=1000,
    traj_step_interval=1,
    traj_subselect=None,
    skip_failures=True,
    results_prefix="last_op__optimize_",
    verbose=False,
    update_config_type="append",
    _rng=None,
    _autopara_per_item_info=None,
    **_opt_kwargs,
):
    """runs a structure optimization. By default calculator properties will be stored in keys
    prefixed with "last_op__optimize_", which may be overwritten by next operation.

    Parameters
    ----------
    atoms: list(Atoms)
        input configs
    calculator: Calculator / (Initialiser, args, kwargs)
        ASE calculator or routine to call to create calculator
    fmax: float, default 1e-3
        force convergence tolerance
    smax: float, default None
        stress convergence tolerance, default from fmax
    steps: int, default 1000
        max number of steps
    pressure: None / float / tuple
        applied pressure distribution (GPa), as parsed by wfl.utils.pressure.sample_pressure()
    stress_mask: None / list(bool)
        mask for stress components to pass to variable-cell filter
    keep_symmetry: bool, default True
        constrain symmetry to maintain initial
    traj_step_interval: int, default 1
        if present, interval between trajectory snapshots
    traj_subselect: "last_converged", default None
        rule for sub-selecting configs from the full trajectory.
        Currently implemented: "last_converged", which takes the last config, if converged.
    skip_failures: bool, default True
        just skip optimizations that raise an exception
    results_prefix: str, default "last_op__optimize_"
        prefix to info/arrays keys where calculator properties will be stored.
        Will overwrite any other properties that start with same "<str>__", so that by
        default only last op's properties will be stored.
    verbose: bool, default False
        verbose output
        optimisation logs are not printed unless this is True
    update_config_type: ["append" | "overwrite" | False], default "append"
        whether/how to add at.info['optimize_config_type'] to at.info['config_type']
    opt_kwargs
        keyword arguments for PreconLBFGS
    rng: numpy.random.Generator, default None
        random number generator to use (needed for pressure sampling, initial temperature, or Langevin dynamics)
    _autopara_per_item_info: dict
        INTERNALLY used by autoparallelization framework to make runs reproducible (see
        wfl.autoparallelize.autoparallelize() docs)

    Returns
    -------
        list(Atoms) trajectories
    """
    calculator = construct_calculator_picklesafe(calculator)

    all_trajs = []

    for _at_i, at in enumerate(atoms_to_list(atoms)):
        # original constraints
        org_constraints = at.constraints
        at.calc = calculator
        opt = FIRE(at, logfile=sys.stdout if verbose else None)

        # default status, will be overwritten for first and last configs in traj
        at.info["optimize_config_type"] = "optimize_mid"
        traj = []

        def process_step(traj=traj, at=at, org_constraints=org_constraints):
            if len(traj) > 0 and traj[-1] == at:
                # Some optimization algorithms sometimes seem to repeat, perhaps
                # only in weird circumstances, e.g. bad gradients near breakdown.
                # Do not store those duplicate configs.
                return

            new_config = at_copy_save_calc_results(at, prefix=results_prefix)
            new_config.set_constraint(org_constraints)
            traj.append(new_config)

        opt.attach(process_step, interval=traj_step_interval)

        # preliminary value
        final_status = "unconverged"

        try:
            if opt.run(fmax=fmax, steps=steps):  # smax=smax,
                final_status = "converged"
        except Exception as exc:
            # label actual failed optimizations
            # when this happens, the atomic config somehow ends up with a 6-vector stress, which can't be
            # read by xyz reader.
            # that should probably never happen
            final_status = "exception"
            if skip_failures:
                sys.stderr.write(
                    f"Structure optimization failed with exception '{exc}'\n"
                )
                sys.stderr.flush()
            else:
                raise

        if len(traj) == 0 or traj[-1] != at:
            new_config = at_copy_save_calc_results(at, prefix=results_prefix)
            new_config.set_constraint(org_constraints)
            traj.append(new_config)

        # set for first config, to be overwritten if it's also last config
        traj[0].info["optimize_config_type"] = "optimize_initial"

        traj[-1].info["optimize_last_status"] = final_status
        traj[-1].info["optimize_n_steps"] = opt.get_number_of_steps()

        for _at in traj:
            save_config_type(_at, update_config_type, _at.info["optimize_config_type"])

        # Note that if resampling doesn't include original last config, later
        # steps won't be able to identify those configs as the (perhaps unconverged) minima.
        # Perhaps status should be set after resampling?
        traj = subselect_from_traj(traj, subselect=traj_subselect)

        all_trajs.append(traj)

    return all_trajs


def optimise(*args, **kwargs):
    default_autopara_info = {"num_inputs_per_python_subprocess": 10}

    return autoparallelize(
        _run_autopara_wrappable,
        *args,
        default_autopara_info=default_autopara_info,
        **kwargs,
    )


autoparallelize_docstring(optimise, _run_autopara_wrappable, "Atoms")


# Just a placeholder for now. Could perhaps include:
#    equispaced in energy
#    equispaced in Cartesian path length
#    equispaced in some other kind of distance (e.g. SOAP)
# also, should it also have max distance instead of number of samples?
def subselect_from_traj(traj, subselect=None):
    """Sub-selects configurations from trajectory.

    Parameters
    ----------
    subselect: int or string, default None

        - None: full trajectory is returned
        - int: (not implemented) how many samples to take from the trajectory.
        - str: specific method

          - "last": returns [last_config]
          - "last_converged": returns [last_config] if converged, or None if not.

    """
    if subselect is None:
        return traj
    elif subselect == "last":
        return [traj[-1]]
    elif subselect == "last_converged":
        return (
            [traj[-1]]
            if (traj[-1].info.get("optimize_last_status") == "converged")
            else []
        )

    raise RuntimeError(
        f"Subselecting confgs from trajectory with rule "
        f'"subselect={subselect}" is not yet implemented'
    )
