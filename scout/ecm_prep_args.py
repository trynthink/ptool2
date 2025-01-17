from __future__ import annotations
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
import re
from scout.config import Config, FilePaths as fp


def ecm_args(args: list = None) -> argparse.NameSpace:  # noqa: F821
    """Parse arguments for ecm_prep.py

    Args:
        args (list, optional): ecm_prep.py input arguments, if not provided, command line
            arguments will be used in Config. Defaults to None.
    """

    # Retrieve  config file and CLI arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            epilog="For more detail please read: "
                            "https://scout-bto.readthedocs.io/en/latest/tutorials.html"
                            "#tutorial-3-preparing-ecms-for-analysis")
    config = Config(parser, "ecm_prep", args)
    opts = config.parse_args()
    opts = translate_inputs(opts)  # Translate for use downstream

    return opts


def translate_inputs(opts: argparse.NameSpace) -> argparse.NameSpace:  # noqa: F821
    """Process args and translate them for use in ecm_prep.py

    Args:
        opts (argparse.NameSpace): object in which to store user responses

    Returns:
        argparse.NameSpace: opts variable with user inputs added
    """

    # Set ECMs if subset is provided
    ecm_dir_files = [
        file.stem for file in fp.ECM_DEF.iterdir() if file.is_file() and
        file.suffix == '.json' and file.stem != 'package_ecms']
    missing_ecms = []
    if opts.ecm_files is not None:
        missing_ecms = [ecm for ecm in opts.ecm_files if ecm not in ecm_dir_files]
    if missing_ecms:
        msg = ("WARNING: The following ECMs specified with the `ecm_files` argument are not"
               f" present in {fp.ECM_DEF} and will not be prepared: {missing_ecms}")
        warnings.warn(msg)
        opts.ecm_files = list(set(opts.ecm_files) - set(missing_ecms))

    # Find matches to `ecm_files_regex`, warn if none found
    ecm_file_matches = [file for file in ecm_dir_files if
                        any(re.match(pattern, file) for pattern in opts.ecm_files_regex)]
    if opts.ecm_files_regex and not ecm_file_matches:
        msg = ("WARNING: A regular expression was specified with `ecm_files_regex`, but it does"
               f"not match any ECMs in {fp.ECM_DEF}")
        warnings.warn(msg)

    # Store ECMs set by the user
    opts.ecm_files_user = []
    if opts.ecm_files is not None:
        opts.ecm_files_user.extend(opts.ecm_files)
        ecm_file_matches = [ecm for ecm in ecm_file_matches if ecm not in opts.ecm_files]
    opts.ecm_files_user.extend(ecm_file_matches)

    # Use all ecms in the ecm directory if `ecm_files` is None and `ecm_files_regex` is empty
    if opts.ecm_files is None and not opts.ecm_files_regex:
        opts.ecm_files = ecm_dir_files
    else:
        opts.ecm_files = opts.ecm_files_user.copy()

    # Set adoption scenario restrictions
    if opts.adopt_scn_restrict:
        opts.adopt_scn_restrict = [opts.adopt_scn_restrict]
    else:
        opts.adopt_scn_restrict = ['Technical potential', 'Max adoption potential']

    # Screen for cases where user desires time-sensitive valuation metrics
    # or hourly sector-level load shapes but EMM or state regions are not used
    # (such options require baseline data to be resolved by EMM region)
    opts.tsv_metrics = False
    if opts.tsv_type:
        opts.tsv_metrics = True

    # Set fugitive emissions to include, if any, and whether to use typical refrigerants
    # (including or excluding representation of expected phase-out years) or user-defined
    # low-GWP refrigerants
    if opts.fugitive_emissions:
        input_var = ['0', None, None]
        refrigerants = ["refrigerant" in elem for elem in opts.fugitive_emissions]
        methane = ["methane" in elem for elem in opts.fugitive_emissions]
        if all(methane):
            input_var[0] = '1'
        elif all(refrigerants):
            input_var[0] = '2'
        elif any(methane) and any(refrigerants):
            input_var[0] = '3'

        for emission in opts.fugitive_emissions:
            if emission == "typical refrigerant":  # Includes representation of phase-out years
                input_var[1] = '2'
            elif emission == "low-gwp refrigerant":
                input_var[1] = '3'
            elif emission == "typical refrigerant no phaseout":
                input_var[1] = '1'

        for emission in opts.fugitive_emissions:
            if emission == "methane-low":  # Lower bound methane leakage
                input_var[2] = '1'
            elif emission == "methane-mid":  # Mid-range methane leakage
                input_var[2] = '2'
            elif emission == "methane-high":  # Upper bound methane leakage
                input_var[2] = '3'

        opts.fugitive_emissions = input_var
    else:
        opts.fugitive_emissions = False

    # Modify early retrofit settings from the default
    input_var = ['1', None, None]
    if opts.retrofit_type == "constant":
        input_var[0] = '2'
    elif opts.retrofit_type == "increasing":
        input_var[0] = '3'
        input_var[1] = opts.retrofit_multiplier
        input_var[2] = opts.retrofit_mult_year
    opts.retro_set = input_var

    # Set exogenous HP rates scenario and retrofit HP behavior
    input_var = [0, 0]
    if opts.exog_hp_rate_scenario:
        # Scenario name maps to Guidehouse E3HP conversion scenario in the input file
        input_var[0] = opts.exog_hp_rate_scenario

        # Retrofit HP switching
        if opts.retro_set[0] != '1':
            if opts.switch_all_retrofit_hp:
                input_var[1] = "1"
            else:
                input_var[1] = "2"
        opts.exog_hp_rates = input_var

        # Ensure that if HP conversion data are to be applied, EMM regional
        # breakouts are set (HP conversion data use this resolution)
        if (opts.alt_regions not in ["EMM", "State"]):
            opts.alt_regions = "EMM"
            warnings.warn(
                "WARNING: Analysis regions were set to EMM to allow HP conversion rates; ensure"
                " that ECM data reflect these EMM regions or states")
    else:
        opts.exog_hp_rates = False

    # Set grid decarbonization scenario and how electricity emissions and
    # cost factors should be handled
    if opts.grid_decarb_level and opts.grid_assessment_timing:
        opts.grid_decarb = True
        # Ensure that if alternate grid decarbonization scenario to be used,
        # fossil fuel equivalency method not used for site-source conversion
        if opts.captured_energy is True:
            opts.captured_energy = False
            warnings.warn(
                "WARNING: Site-source conversion method was set to use the "
                "fossil fuel equivalency method because captured energy "
                "site-source conversion factors not compatible with alternate "
                "grid decarbonization scenario")
    else:
        if opts.grid_decarb_level or opts.grid_assessment_timing:
            warnings.warn(
                    "WARNING: Both `grid_decarb_level` and `grid_assessment_timing` arguments "
                    "must be set run alternate grid decarbonization scenarios.")
        opts.grid_decarb = False

    if (opts.alt_regions not in ["EMM", "State"]) and any([
            x is not False for x in [
            opts.tsv_metrics, opts.sect_shapes, opts.grid_decarb]]):
        opts.alt_regions = "EMM"
        # Craft custom warning message based on the option provided
        warn_text = (
            "tsv metrics, sector-level 8760 savings shapes, and/or "
            "high grid decarbonization user inputs")
        warnings.warn(
            f"WARNING: Analysis regions were set to EMM to allow {warn_text}; "  # noqa: E702
            "ensure that ECM data reflect these EMM regions")
    # Set time-sensitive efficiency values
    if opts.tsv_average_days and (opts.tsv_power_agg != "average" and opts.tsv_type == "power"):
        opts.tsv_average_days = None
        warnings.warn(
            "WARNING: argument tsv_average_days was provided but is not applicable,"
            " argument will be ignored")
    output_type = {"energy": "1", "power": "2"}.get(opts.tsv_type, None)
    hours = {"all": "1", "peak": "2", "low": "3"}.get(opts.tsv_daily_hr_restrict)
    sys_shape = {"total reference": "1",
                 "total high renewables": "2",
                 "net renewable reference": "3",
                 "net renewable high renewables": "4"}.get(opts.tsv_sys_shape_case, "0")
    season = {"summer": "1", "winter": "2", "intermediate": "3"}.get(opts.tsv_season)
    if opts.tsv_type == "energy":
        calc_type = {"sum": "1", "average": "2"}[opts.tsv_energy_agg]
    elif opts.tsv_type == "power":
        calc_type = {"peak": "1", "average": "2"}[opts.tsv_power_agg]
    day_type = {"all": "1", "weekdays": "2", "weekends": "3"}.get(opts.tsv_average_days, "0")
    if output_type is not None:
        opts.tsv_metrics = [output_type, hours, season, calc_type, sys_shape, day_type]
    else:
        opts.tsv_metrics = False

    # Set detailed breakout options
    if opts.detail_brkout:
        if "all" in opts.detail_brkout:
            opts.detail_brkout = ["buildings", "fuel types", "regions"]
        input_var_dict = {
            ("buildings", "fuel types", "regions"): "1",
            ("regions",): "2",
            ("buildings",): "3",
            ("fuel types",): "4",
            ("buildings", "regions"): "5",
            ("fuel types", "regions"): "6",
            ("buildings", "fuel types"): "7",
        }
        opts.detail_brkout.sort()
        opts.detail_brkout = tuple(opts.detail_brkout)
        opts.detail_brkout = input_var_dict.get(opts.detail_brkout)
    else:
        opts.detail_brkout = False

    # Ensure that if public cost health data are to be applied, EMM regional
    # breakouts are set (health data use this resolution)
    if opts.health_costs is True and opts.alt_regions != "EMM":
        opts.alt_regions = "EMM"
        warnings.warn(
            "WARNING: Analysis regions were set to EMM to allow public health "
            "cost adders; ensure that ECM data reflect these EMM regions")

    # Set inclusion of HVAC-only measures in measure competition
    opts.pkg_env_costs = {"include HVAC": "1", "exclude HVAC": "2"}.get(opts.pkg_env_costs, False)

    return opts
