import sys
import json
import numpy as np

from ase.md.langevin import Langevin
try:
    from ase.md.bussi import Bussi
except ImportError:
    Bussi = None

from ase.md.logger import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.units import fs

from wfl.autoparallelize import autoparallelize, autoparallelize_docstring
from wfl.utils.save_calc_results import at_copy_save_calc_results
from wfl.utils.misc import atoms_to_list
from wfl.utils.parallel import construct_calculator_picklesafe
from wfl.generate.utils import save_config_type
#from wfl.generate.md.utils import _get_temperature


def _sample_autopara_wrappable(atoms, calculator, steps, dt, integrator="Langevin", 
              temperature=None, temperature_tau=100.0,
              traj_step_interval=1, skip_failures=True, results_prefix='last_op__md_', verbose=False, 
              update_config_type="append", traj_select_during_func=lambda at: True, 
              traj_select_after_func=None, abort_check=None, logger_interval=0, logger_kwargs=None, 
              rng=None, _autopara_per_item_info=None):
    """
    Runs an NVT MD trajectory.
    
    Parameters
    ----------
    integrator: "Langevin" or "Bussi"
        - Langevin: Good for sampling diversity (stochastic friction).
        - Bussi: Good for dynamical properties (velocity rescaling).
    """

    return _sample_autopara_wrappable_kwargs(
        atoms, calculator, steps, dt,
        integrator=integrator,
        temperature=temperature,
        temperature_tau=temperature_tau,
        traj_step_interval=traj_step_interval,
        skip_failures=skip_failures,
        results_prefix=results_prefix,
        verbose=verbose,
        update_config_type=update_config_type,
        traj_select_during_func=traj_select_during_func,
        traj_select_after_func=traj_select_after_func,
        abort_check=abort_check,
        logger_interval=logger_interval,
        logger_kwargs=logger_kwargs,
        rng=rng,
        _autopara_per_item_info=_autopara_per_item_info
    )


def _sample_autopara_wrappable_single(at, at_i, calculator, steps, dt, logger_interval, 
                                      logger_constructor, logger_logfile, logger_kwargs, 
                                      integrator="Langevin", temperature=None, temperature_tau=100.0,
                                      traj_step_interval=1, skip_failures=True, 
                                      results_prefix='last_op__md_', verbose=False, 
                                      update_config_type="append", 
                                      traj_select_during_func=lambda at: True, 
                                      traj_select_after_func=None, abort_check=None, 
                                      rng=None, _autopara_per_item_info=None):
    
    # 1. validate integrator is present
    if integrator == "Bussi" and Bussi is None:
        raise ImportError("Integrator 'Bussi' selected but ase.md.bussi not found. Upgrade ASE.")
    if integrator not in ["Langevin", "Bussi"]:
        raise ValueError(f"Integrator must be 'Langevin' or 'Bussi', got {integrator}")

    rng = _autopara_per_item_info[at_i].get("rng")
    item_i = _autopara_per_item_info[at_i].get("item_i")

    at.calc = calculator
    
    # 2. set up temperature schedule
    temperature_use = _get_temperature(temperature, temperature_tau, steps)
    if temperature_use is None:
        raise ValueError(f"{integrator} integrator requires a valid temperature.")

    if temperature_use is not None:
        assert rng is not None
        MaxwellBoltzmannDistribution(at, temperature_K=temperature_use[0]['T_i'], force_temp=True, communicator=None, rng=rng)
        Stationary(at, preserve_temperature=True)

    # 3. set up integrator parameters
    # Both need atomic units for time parameters
    # Langevin expects "friction" (1/time), Bussi expects "taut" (time)
    tau_au = temperature_tau * fs
    
    all_stage_kwargs = []
    all_run_kwargs = []

    for t_stage in temperature_use:
        stage_steps = t_stage['traj_frac'] * steps
        
        # Base arguments (Shared)
        stage_base = {
            'timestep': dt * fs,
            'rng': rng
        }
        
        # Specific arguments
        if integrator == "Langevin":
            stage_base['friction'] = 1.0 / tau_au
        elif integrator == "Bussi":
            stage_base['taut'] = tau_au

        if t_stage['T_f'] == t_stage['T_i']:
            stage_args = stage_base.copy()
            stage_args['temperature_K'] = t_stage['T_i']
            all_stage_kwargs.append(stage_args)
            all_run_kwargs.append({'steps': int(np.round(stage_steps))})
        else:
            substage_steps = int(np.round(stage_steps / t_stage['n_stages']))
            for T in np.linspace(t_stage['T_i'], t_stage['T_f'], t_stage['n_stages']):
                stage_args = stage_base.copy()
                stage_args['temperature_K'] = T
                all_stage_kwargs.append(stage_args)
            
            all_run_kwargs.extend([{'steps': substage_steps}] * t_stage['n_stages'])

    # 4. execute main loop of MD
    traj = []
    cur_step = 1
    first_step_of_later_stage = False

    def process_step(interval):
        nonlocal cur_step, first_step_of_later_stage

        if not first_step_of_later_stage and cur_step % interval == 0:
            at.info['MD_time_fs'] = cur_step * dt
            at.info['MD_step'] = cur_step
            at.info["MD_current_temperature"] = at.get_temperature()
            
            at_save = at_copy_save_calc_results(at, prefix=results_prefix)
            if traj_select_during_func(at):
                traj.append(at_save)

            if abort_check is not None and abort_check.stop(at):
                raise RuntimeError(f"MD stopped by {abort_check.__class__.__name__}")

        first_step_of_later_stage = False
        cur_step += 1

    for stage_i, (stage_kwargs, run_kwargs) in enumerate(zip(all_stage_kwargs, all_run_kwargs)):
        if verbose:
            print(f'Running stage: T={stage_kwargs["temperature_K"]}K, steps={run_kwargs["steps"]}')
        
        # avoid double counting of steps at end of each stage and beginning of next
        cur_step -= 1 

        at.info['MD_temperature_K'] = stage_kwargs['temperature_K']

        # Factory Logic for integrator to attach process_step
        if integrator == "Langevin":
            md = Langevin(at, **stage_kwargs)
        elif integrator == "Bussi":
            md = Bussi(at, **stage_kwargs)

        md.attach(process_step, 1, traj_step_interval)
        
        # Logging
        if logger_interval > 0:
            if logger_logfile == "-":
                logger_kwargs["logfile"] = "-"
                hdr_prefix = f"config {item_i} "
            else:
                logger_kwargs["logfile"] = f"{logger_logfile}.config_{item_i}"
                hdr_prefix = ""

            logger_kwargs["dyn"] = md
            logger_kwargs["atoms"] = at
            logger = logger_constructor(**logger_kwargs)
            
            if hdr_prefix:
                logger.hdr = hdr_prefix + logger.hdr
                logger.fmt = hdr_prefix + logger.fmt
            
            md.attach(logger, logger_interval)

        if stage_i > 0:
            first_step_of_later_stage = True

        try:
            md.run(**run_kwargs)
        except Exception as exc:
            if skip_failures:
                sys.stderr.write(f'MD failed with exception \'{exc}\'\n')
                sys.stderr.flush()
                break
            else:
                raise

    if len(traj) == 0 or traj[-1] != at:
        if traj_select_during_func(at):
            at.info['MD_time_fs'] = cur_step * dt
            traj.append(at_copy_save_calc_results(at, prefix=results_prefix))

    if traj_select_after_func is not None:
        traj = traj_select_after_func(traj)

    for at in traj:
        save_config_type(at, update_config_type, 'MD')

    return traj


def _sample_autopara_wrappable_kwargs(atoms, calculator, steps, dt, **kwargs):
    calculator = construct_calculator_picklesafe(calculator)

    logger_interval = kwargs.pop("logger_interval", 0)
    logger_kwargs = kwargs.pop("logger_kwargs", None) or {}
    
    logger_constructor = None
    logger_logfile = None
    if logger_interval > 0:
        logger_constructor = logger_kwargs.pop("logger", MDLogger)
        logger_logfile = logger_kwargs.pop("logfile", "-")

    kwargs_at = kwargs.copy()
    kwargs_orig = {}

    all_trajs = []
    for at_i, at in enumerate(atoms_to_list(atoms)):
        kwargs_at.update(kwargs_orig)
        
        # Allow JSON overrides including 'integrator'
        for k, v in json.loads(at.info.get("WFL_MD_KWARGS", "{}")).items():
            if k not in kwargs_orig:
                kwargs_orig[k] = kwargs[k]
            kwargs_at[k] = v

        traj = _sample_autopara_wrappable_single(
            at, at_i, calculator, steps, dt,
            logger_interval, logger_constructor, logger_logfile, logger_kwargs, 
            **kwargs_at
        )

        all_trajs.append(traj)

    return all_trajs


def md(*args, **kwargs):
    default_autopara_info = {"num_inputs_per_python_subprocess": 10}
    return autoparallelize(_sample_autopara_wrappable, *args,
                           default_autopara_info=default_autopara_info, **kwargs)

autoparallelize_docstring(md, _sample_autopara_wrappable, "Atoms")



def _get_temperature(temperature_use, temperature_tau, steps):
    if temperature_tau is None and (temperature_use is not None and not isinstance(temperature_use, (float, int, np.floating, np.integer))):
        raise RuntimeError(f'NVE (temperature_tau is None) can only accept temperature=float for initial T, got {type(temperature_use)}')

    if temperature_use is not None:
        # assume that dicts are already in temperature profile format
        if not isinstance(temperature_use, dict):
            try:
                # check if it's a list, tuple, etc
                len(temperature_use)
            except TypeError:
                # number into a list
                temperature_use = [temperature_use]
        if not isinstance(temperature_use[0], dict):
            # create a stage dict from a constant or ramp
            t_stage_data = temperature_use
            # start with constant
            t_stage = {'T_i': t_stage_data[0], 'T_f': t_stage_data[0], 'traj_frac': 1.0, 'n_stages': 10, 'steps': steps}
            if len(t_stage_data) >= 2:
                # set different final T for ramp
                t_stage['T_f'] = t_stage_data[1]
            if len(t_stage_data) >= 3:
                # set number of stages
                t_stage['n_stages'] = t_stage_data[2]
            temperature_use = [t_stage]
        else:
            for t_stage in temperature_use:
                if 'n_stages' not in t_stage:
                    t_stage['n_stages'] = 10

    return temperature_use