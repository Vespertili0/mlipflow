import os
from wfl.configset import ConfigSet, OutputSpec
from wfl.fit import error
from wfl.utils.configs import atomization_energy


def calculate_mlip_error(in_configs, out_file, calc_property_prefix, ref_property_prefix='DFT_', fig_dir='.') -> dict:
    """
    """
    # write joint configs to file
    OutputSpec(out_file).write(ConfigSet(in_configs))
    
    # calculate atomisation-energy for DFT- & MLIP-energy
    for prop in [calc_property_prefix, ref_property_prefix]:
        atomization_energy(
            inputs=ConfigSet(out_file), 
            outputs=OutputSpec(out_file, overwrite=True), 
            prop_prefix=prop
        )

    # calculate errors
    errors, diffs, parity = error.calc(
        inputs=ConfigSet(out_file),
        calc_property_prefix=calc_property_prefix,
        ref_property_prefix=ref_property_prefix,
        category_keys='data_type', 
        config_properties=["atomization_energy/atom"], #"energy/atom"
        atom_properties=["forces/comp"]
    )
    
    # plot errors
    for error_type in ['RMSE', 'MAE']:
        error.value_error_scatter(
            all_errors = errors,
            all_diffs=diffs, 
            all_parity=parity,
            output=os.path.join(
                fig_dir, 
                f"{calc_property_prefix}_{error_type}.png"
            ),
            calc_property_prefix=calc_property_prefix,
            ref_property_prefix=ref_property_prefix,
            error_type=error_type
        )
    
    return errors