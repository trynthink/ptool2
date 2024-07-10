#!/usr/bin/env python3
from __future__ import annotations
import numpy
import re
import itertools
import json
from collections import OrderedDict
from os import listdir, getcwd, stat
from os.path import isfile, join
import copy
import warnings
from urllib.parse import urlparse
import gzip
import pickle
from functools import reduce  # forward compatibility for Python 3
import operator
from ast import literal_eval
import math
import pandas as pd
from datetime import datetime
from pathlib import PurePath, Path
import argparse
from scout.ecm_prep_args import ecm_args
from scout.config import FilePaths as fp


class MyEncoder(json.JSONEncoder):
    """Convert numpy arrays to list for JSON serializing."""

    def default(self, obj):
        """Modify 'default' method from JSONEncoder."""
        # Case where object to be serialized is numpy array
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        if isinstance(obj, PurePath):
            return str(obj)
        # All other cases
        else:
            return super(MyEncoder, self).default(obj)


class UsefulInputFiles(object):
    """Class of input file paths to be used by this routine.

    Attributes:
        msegs_in (tuple): Database of baseline microsegment stock/energy.
        msegs_cpl_in (tuple): Database of baseline technology characteristics.
        iecc_reg_map (tuple): Maps IECC climates to AIA or EMM regions/states.
        ash_emm_map (tuple): Maps ASHRAE climates to EMM regions.
        aia_altreg_map (tuple): Maps AIA climates to EMM regions or states.
        state_emm_map (tuple): Maps states to EMM regions.
        state_aia_map (tuple): Maps states to AIA regions.
        metadata (str) = Baseline metadata inc. min/max for year range.
        glob_vars (str) = Global settings from ecm_prep to use later in run
        cost_convert_in (tuple): Database of measure cost unit conversions.
        cap_facts (tuple): Database of commercial equip. capacity factors.
        cbecs_sf_byvint (tuple): Commercial sq.ft. by vintage data.
        indiv_ecms (tuple): Individual ECM JSON definitions folder.
        ecm_packages (tuple): Measure package data.
        ecm_prep (tuple): Prepared measure attributes data for use in the analysis engine.
        ecm_prep_env_cf (tuple): Prepared envelope/HVAC package measure
            attributes data with effects of HVAC removed (isolate envelope).
        ecm_prep_shapes (tuple): Prepared measure sector shapes data.
        ecm_prep_env_cf_shapes (tuple): Prepared envelope/HVAC package measure
            sector shapes data with effects of HVAC removed (isolate envelope).
        ecm_compete_data (tuple): Folder with contributing microsegment data
            needed to run measure competition in the analysis engine.
        ecm_eff_fs_splt_data (tuple): Folder with data needed to determine the
            fuel splits of efficient case results for fuel switching measures.
        run_setup (str): Names of active measures that should be run in
            the analysis engine.
        cpi_data (tuple): Historical Consumer Price Index data.
        ss_data (tuple): Site-source, emissions, and price data, national.
        ss_data_nonfs (tuple): Site-source, emissions, and price data,
            national, to assign in certain cases to non-fuel switching
            microsegments under high grid decarb case.
        ss_data_altreg (tuple): Emissions/price data, EMM- or state-resolved.
        ss_data_altreg_nonfs (tuple): Base-case emissions/price data, EMM– or
            state-resolved, to assign in certain cases to non-fuel switching
            microsegments under high grid decarb case.
        tsv_load_data (tuple): Time sensitive energy demand data.
        tsv_cost_data (tuple): Time sensitive electricity price data.
        tsv_carbon_data (tuple): Time sensitive average CO2 emissions data.
        tsv_cost_data_nonfs (tuple): Time sensitive electricity price data to
            assign in certain cases to non-fuel switching microsegments under
            high grid decarb case.
        tsv_carbon_data_nonfs (tuple): Time sensitive average CO2 emissions
            data to assign in certain cases to non-fuel switching microsegments
            under high grid decarb case.
        tsv_shape_data (tuple): Custom hourly savings shape data.
        tsv_metrics_data_tot (tuple): Total system load data by EMM region.
        tsv_metrics_data_net (tuple): Net system load shape data by EMM region.
        health_data (tuple): EPA public health benefits data by EMM region.
        hp_convert_rates (tuple): Fuel switching conversion rates.
        fug_emissions_dat (tuple): Refrigerant and supply chain methane leakage
            data to asses fugitive emissions sources.
    """

    def __init__(self, opts):
        if opts.alt_regions == 'AIA':
            # UNCOMMENT WITH ISSUE 188
            # self.msegs_in = fp.STOCK_ENERGY / "mseg_res_com_cz_2017.json"
            # UNCOMMENT WITH ISSUE 188
            # self.msegs_cpl_in = fp.STOCK_ENERGY / "cpl_res_com_cz_2017.json"
            self.msegs_in = fp.STOCK_ENERGY / "mseg_res_com_cz.json"
            self.msegs_cpl_in = fp.STOCK_ENERGY / "cpl_res_com_cz.gz"
            self.iecc_reg_map = (
                fp.CONVERT_DATA / "geo_map" / "IECC_AIA_ColSums.txt")
            self.state_aia_map = (
                fp.CONVERT_DATA / "geo_map" / "AIA_State_RowSums.txt")
        elif opts.alt_regions == 'EMM':
            self.msegs_in = fp.STOCK_ENERGY / "mseg_res_com_emm.gz"
            self.msegs_cpl_in = fp.STOCK_ENERGY / "cpl_res_com_emm.gz"
            self.ash_emm_map = fp.CONVERT_DATA / "geo_map" / "ASH_EMM_ColSums.txt"
            self.aia_altreg_map = fp.CONVERT_DATA / "geo_map" / "AIA_EMM_ColSums.txt"
            self.iecc_reg_map = fp.CONVERT_DATA / "geo_map" / "IECC_EMM_ColSums.txt"
            self.state_emm_map = fp.CONVERT_DATA / "geo_map" / "EMM_State_RowSums.txt"
        elif opts.alt_regions == 'State':
            self.msegs_in = fp.STOCK_ENERGY / "mseg_res_com_state.gz"
            self.msegs_cpl_in = fp.STOCK_ENERGY / "cpl_res_com_cdiv.gz"
            self.aia_altreg_map = fp.CONVERT_DATA / "geo_map" / "AIA_State_ColSums.txt"
            self.iecc_reg_map = fp.CONVERT_DATA / "geo_map" / "IECC_State_ColSums.txt"
        else:
            raise ValueError(
                "Unsupported regional breakout (" + opts.alt_regions + ")")

        self.set_decarb_grid_vars(opts)
        self.metadata = fp.METADATA_PATH
        self.glob_vars = fp.GENERATED / "glob_run_vars.json"
        # UNCOMMENT WITH ISSUE 188
        # self.metadata = "metadata_2017.json"
        self.cost_convert_in = fp.CONVERT_DATA / "ecm_cost_convert.json"
        self.cap_facts = fp.CONVERT_DATA / "cap_facts.json"
        self.cbecs_sf_byvint = fp.CONVERT_DATA / "cbecs_sf_byvintage.json"
        self.indiv_ecms = fp.ECM_DEF
        self.ecm_packages = fp.ECM_DEF / "package_ecms.json"
        self.ecm_prep = fp.GENERATED / "ecm_prep.json"
        self.ecm_prep_env_cf = fp.GENERATED / "ecm_prep_env_cf.json"
        self.ecm_prep_shapes = fp.GENERATED / "ecm_prep_shapes.json"
        self.ecm_prep_env_cf_shapes = fp.GENERATED / "ecm_prep_env_cf_shapes.json"
        self.ecm_compete_data = fp.ECM_COMP
        self.ecm_eff_fs_splt_data = fp.EFF_FS_SPLIT
        self.run_setup = fp.GENERATED / "run_setup.json"
        self.cpi_data = fp.CONVERT_DATA / "cpi.csv"
        self.tsv_load_data = fp.TSV_DATA / "tsv_load.json"
        self.tsv_shape_data = (
            fp.ECM_DEF / "energyplus_data" / "savings_shapes")
        self.tsv_metrics_data_tot_ref = fp.TSV_DATA / "tsv_hrs_tot_ref.csv"
        self.tsv_metrics_data_net_ref = fp.TSV_DATA / "tsv_hrs_net_ref.csv"
        self.tsv_metrics_data_tot_hr = fp.TSV_DATA / "tsv_hrs_tot_hr.csv"
        self.tsv_metrics_data_net_hr = fp.TSV_DATA / "tsv_hrs_net_hr.csv"
        self.health_data = fp.CONVERT_DATA / "epa_costs.csv"
        self.hp_convert_rates = fp.CONVERT_DATA / "hp_convert_rates.json"
        self.fug_emissions_dat = fp.CONVERT_DATA / "fugitive_emissions_convert.json"

    def set_decarb_grid_vars(self, opts: argparse.NameSpace):  # noqa: F821
        """Assign instance variables related to grid decarbonization which are dependent on the
            alt_regions, alt_ref_carb, grid_decarb_level, and grid_assessment_timing arguments.

        Args:
            opts (argparse.NameSpace): argparse object containing the argument attributes
        """

        def get_suffix(arg):
            """Return a suffix derived from a user-supplied argument string to append to filepath
                variables; if argument is None, return an empty string.
            """
            if arg is None:
                return ''
            else:
                return f"-{arg}"
        alt_ref_carb_suffix = get_suffix(opts.alt_ref_carb)
        grid_decarb_level_suffix = get_suffix(opts.grid_decarb_level)
        # Toggle EMM emissions and price data based on whether or not a grid decarbonization
        # scenario is used
        if opts.alt_regions == 'EMM':
            emission_var_map = {}  # Map UsefulInputFiles instance vars to filenames suffixes
            if opts.grid_decarb_level:
                # Set grid decarbonization case
                emission_var_map["ss_data_altreg"] = grid_decarb_level_suffix
            else:
                # Set baseline emissions factors
                emission_var_map["ss_data_altreg"] = alt_ref_carb_suffix
                self.ss_data_altreg_nonfs = None
            if opts.grid_assessment_timing and opts.grid_assessment_timing == "before":
                # Set emissions/cost reductions for non-fuel switching measures before grid
                # decarbonization
                emission_var_map["ss_data_altreg_nonfs"] = alt_ref_carb_suffix
            elif (not opts.grid_decarb or
                    (opts.grid_assessment_timing and opts.grid_assessment_timing == "after")):
                # Set emissions/cost reductions for non-fuel switching measures after grid
                # decarbonization
                self.ss_data_altreg_nonfs = None
            # Set filepaths for EMM region emissions and prices
            for var, filesuffix in emission_var_map.items():
                filepath = fp.CONVERT_DATA / f"emm_region_emissions_prices{filesuffix}.json"
                setattr(self, var, filepath)
        # Ensure that state-level regions are not being used alongside a high grid
        # decarbonization scenario (incompatible currently)
        elif opts.alt_regions == 'State':
            if opts.grid_decarb:
                raise ValueError("Unsupported regional breakout for use with alternate grid"
                                 f" decarbonization scenario ({opts.alt_regions})")
            else:
                self.ss_data_altreg = fp.CONVERT_DATA / "state_emissions_prices.json"
                self.ss_data_altreg_nonfs = None

        # Set site-source conversions files for grid decarbonization case
        if opts.grid_decarb:
            self.ss_data = (fp.CONVERT_DATA /
                            f"site_source_co2_conversions{grid_decarb_level_suffix}.json")
            # Update tsv data file suffixes for DECARB levels
            if "DECARB" in grid_decarb_level_suffix:
                grid_decarb_level_suffix = {"-DECARB-mid": "-95by2050",
                                            "-DECARB-high": "-100by2035"}[grid_decarb_level_suffix]
            self.tsv_cost_data = fp.TSV_DATA / f"tsv_cost{grid_decarb_level_suffix}.json"
            self.tsv_carbon_data = fp.TSV_DATA / f"tsv_carbon{grid_decarb_level_suffix}.json"

        # Set site-source conversions for non-fuel switching measures before grid decarbonization
        if opts.grid_assessment_timing and opts.grid_assessment_timing == "before":
            self.ss_data_nonfs = (fp.CONVERT_DATA /
                                  f"site_source_co2_conversions{alt_ref_carb_suffix}.json")
            self.tsv_cost_data_nonfs = fp.TSV_DATA / "tsv_cost-MidCaseTCExpire.json"
            self.tsv_carbon_data_nonfs = fp.TSV_DATA / "tsv_carbon-MidCaseTCExpire.json"
        # Set site-source conversions for non-fuel switching measures after grid decarbonization
        elif (not opts.grid_decarb or
                (opts.grid_assessment_timing and opts.grid_assessment_timing == "after")):
            self.ss_data_nonfs, self.tsv_cost_data_nonfs, \
                self.tsv_carbon_data_nonfs = (None for n in range(3))

        # Set site-source conversions for captured energy method
        if opts.captured_energy:
            self.ss_data = fp.CONVERT_DATA / "site_source_co2_conversions-ce.json"
        elif not opts.grid_decarb:
            self.ss_data = (fp.CONVERT_DATA /
                            f"site_source_co2_conversions{alt_ref_carb_suffix}.json")
            self.tsv_cost_data = fp.TSV_DATA / "tsv_cost-MidCaseTCExpire.json"
            self.tsv_carbon_data = fp.TSV_DATA / "tsv_carbon-MidCaseTCExpire.json"
            self.ss_data_nonfs, self.tsv_cost_data_nonfs, \
                self.tsv_carbon_data_nonfs = (None for n in range(3))

class ImmutableDict:
    def __init__(self, d):
        self._dict = d

    def __getitem__(self, key):
        return self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __contains__(self, key):
        return key in self._dict

    def items(self):
        return self._dict.items()

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def __repr__(self):
        return repr(self._dict)

    def __str__(self):
        return str(self._dict)

    def __eq__(self, other):
        if isinstance(other, ImmutableDict):
            return self._dict == other._dict
        return self._dict == other

    def __setitem__(self, key, value):
        raise ValueError(f"Cannot modify ImmutableDict: {key}")

    def __delitem__(self, key):
        raise ValueError("Cannot modify ImmutableDict")

    def clear(self):
        raise ValueError("Cannot modify ImmutableDict")

    def pop(self, key, default=None):
        raise ValueError("Cannot modify ImmutableDict")

    def popitem(self):
        raise ValueError("Cannot modify ImmutableDict")

    def setdefault(self, key, default=None):
        raise ValueError("Cannot modify ImmutableDict")

    def update(self, *args, **kwargs):
        raise ValueError("Cannot modify ImmutableDict")

    def to_dict(self):
        return dict(self._dict)


class UsefulVars(object):
    """Class of variables that are used globally across functions.

    Attributes:
        adopt_schemes (list): Possible consumer adoption scenarios.
        discount_rate (float): Rate to use in discounting costs/savings.
        nsamples (int): Number of samples to draw from probability distribution
            on measure inputs.
        regions (string): User region settings.
        aeo_years (list): Modeling time horizon.
        aeo_years_summary (list): Reduced set of snapshot years in the horizon.
        retro_rate (dict): Annual rate of deep retrofitting existing stock.
        demand_tech (list): All demand-side heating/cooling technologies.
        zero_cost_tech (list): All baseline technologies with cost of zero.
        inverted_relperf_list (list) = Performance units that require
            an inverted relative performance calculation (e.g., an air change
            rate where lower numbers indicate higher performance).
        valid_submkt_urls (list) = Valid URLs for sub-market scaling fractions.
        consumer_price_ind (numpy.ndarray) = Historical Consumer Price Index.
        ss_conv (dict): Site-source conversion factors by fuel type.
        fuel_switch_conv (dict): Performance unit conversions for expected
            fuel switching cases.
        carb_int (dict): Carbon intensities by fuel type (MMTon/quad).
        ecosts (dict): Energy costs by building and fuel type ($/MMBtu).
        ccosts (dict): Carbon costs ($/MTon).
        com_timeprefs (dict): Commercial adoption time preference premiums.
        hp_rates (dict): Exogenous rates of conversions from baseline
            equipment to heat pumps, if applicable.
        fug_emissions (dict): Refrigerant leakage data and supply chain
            methane data to support assessments of fugitive emissions.
        in_all_map (dict): Maps any user-defined measure inputs marked 'all' to
            list of climates, buildings, fuels, end uses, or technologies.
        valid_mktnames (list): List of all valid applicable baseline market
            input names for a measure.
        out_break_czones (OrderedDict): Maps measure climate zone names to
            the climate zone categories used in summarizing measure outputs.
        out_break_bldgtypes (OrderedDict): Maps measure building type names to
            the building sector categories used in summarizing measure outputs.
        out_break_enduses (OrderedDict): Maps measure end use names to
            the end use categories used in summarizing measure outputs.
        out_break_eus_w_fsplits (List): List of end use categories that
            would potentially apply across multiple fuels.
        out_break_fuels (OrderedDict): Maps measure fuel types to electric vs.
            non-electric fuels (for heating, cooling, WH, and cooking).
        out_break_in (OrderedDict): Breaks out key measure results by
            climate zone, building sector, and end use.
        cconv_topkeys_map (dict): Maps measure cost units to top-level keys in
            an input cost conversion data dict.
        tech_units_rmv (list): Flags baseline performance units that cannot
            currently be handled, thus the associated segment must be removed.
        tech_units_map (dict): Maps baseline performance units to measure units
            in cases where the conversion is expected (e.g., EER to COP).
        sf_to_house (dict): Stores information for mapping stock units in
            sf to number of households, as applicable.
        com_eqp_eus_nostk (list): Flags commercial equipment end uses for
            which no service demand data (which are used to represent com.
            "stock") are available and square footage should be used for stock.
        res_lts_per_home (list): RECS 2015 Table HC5.1 number of lights per
            household, by building type, used to get from $/home to $/bulb
        cconv_tech_mltstage_map (dict): Maps measure cost units to cost
            conversion dict keys for demand-side heating/cooling
            technologies and controls technologies requiring multiple
            conversion steps (e.g., $/ft^2 glazing -> $/ft^2 wall ->
            $/ft^2 floor).
        cconv_bybldg_units (list): Flags cost unit conversions that must
            be re-initiated for each new microsegment building type.
        deflt_choice (list): Residential technology choice capital/operating
            cost parameters to use when choice data are missing.
        regions (str): Regions to use in geographically breaking out the data.
        region_cpl_mapping (str or dict): Maps states to census divisions for
            the case where states are used; otherwise empty string.
        self.com_RTU_fs_tech (list): Flag heating tech. that pairs with RTU.
        self.com_nRTU_fs_tech (list): Flag heating tech. that pairs with
            larger commercial cooling equipment (not RTU).
        resist_ht_wh_tech (list): Flag for resistance-based heat/WH technology.
        minor_hvac_tech (list): Minor/secondary HVAC tech. to remove stock/
            stock/cost data for when major tech. is also in measure definition.
        alt_perfcost_brk_map (dict): Mapping factors used to handle alternate
            regional breakouts in measure performance or cost units.
        months (str): Month sequence for accessing time-sensitive data.
        tsv_feature_types (list): Possible types of TSV features.
        tsv_climate_regions (list): Possible ASHRAE/IECC climate regions for
            time-sensitive analysis and metrics.
        tsv_nerc_regions (list): Possible NERC regions for time-sensitive data.
        tsv_metrics_data (str): Includes information on max/min net system load
            hours, peak/take net system load windows, and peak days by EMM
            region/season, as well as days of year to attribute to each season.
        tsv_hourly_price (dict): Dict for storing hourly price factors.
        tsv_hourly_emissions (dict): Dict for storing hourly emissions factors.
        tsv_hourly_lafs (dict): Dict for storing annual energy, cost, and
            carbon adjustment factors by region, building type, and end use.
        emm_name_num_map (dict): Maps EMM region names to EIA region numbers.
        cz_emm_map (dict): Maps climate zones to EMM region net system load
            shape data.
        health_scn_names (list): List of public health data scenario names.
        health_scn_data (numpy.ndarray): Public health cost data.
        env_heat_ls_scrn (tuple): Envelope heat gains to screen out of time-
            sensitive valuation for heating (no load shapes for these gains).
    """

    def __init__(self, base_dir, handyfiles, opts, allow_overwrite=False):
        self.adopt_schemes = opts.adopt_scn_restrict
        self.discount_rate = 0.07
        self.nsamples = 100
        self.regions = opts.alt_regions
        # Load metadata including AEO year range
        with open(handyfiles.metadata, 'r') as aeo_yrs:
            try:
                aeo_yrs = json.load(aeo_yrs)
            except ValueError as e:
                raise ValueError(f"Error reading in '{handyfiles.metadata}': {str(e)}") from None
        # Set minimum modeling year to current year
        aeo_min = datetime.today().year
        # Set maximum modeling year
        aeo_max = aeo_yrs["max year"]
        # Derive time horizon from min/max years
        self.aeo_years = [
            str(i) for i in range(aeo_min, aeo_max + 1)]
        self.aeo_years_summary = ["2030", "2050"]
        # Set early retrofit rate assumptions

        # Default case (zero early retrofits) or user has set early retrofits
        # to zero
        if opts.retro_set is False or opts.retro_set[0] == "1":
            self.retro_rate = {yr: 0 for yr in self.aeo_years}
        # User has set early retrofits to non-zero
        else:
            # Set default assumptions about starting values for early
            # retrofits at the technology component-level. Values are based
            # on survey questions about renovations in CBECS and the American
            # Housing Survey, which cover lighting, HVAC, and envelope for
            # commercial and HVAC and envelope for residential, respectively.
            # Water heating values are assumed to be identical to HVAC
            # values for the given building type, and residential lighting
            # values are assumed to be identical to commercial values. Values
            # for all other components are set to zero.
            start_vals = {
                "commercial": {
                    "lighting": 0.015, "HVAC": 0.009, "roof": 0.006,
                    "windows": 0.003, "wall": 0.003,
                    "water heating": 0.009, "other": 0
                },
                "residential": {
                    "lighting": 0.015, "HVAC": 0.005, "roof": 0.0027,
                    "windows": 0.0023, "wall": 0.0006,
                    "water heating": 0.005, "other": 0
                }
            }

            # Set multipliers that progressively scale up the early retrofit
            # values over time

            # User desires no change in starting values for early retrofits
            # across the modeled time horizon; set multipliers to 1 across yrs.
            if opts.retro_set[0] == "2":
                multipliers = {yr: 1 for yr in self.aeo_years}
            # User specified a rate multiplier and year by which it is
            # achieved; assume linear increase in early retrofit rates from
            # starting values to the increased values by the indicated year,
            # and maintain increased value for all years thereafter
            else:
                # Pull in user-defined rate multiplier and year by which it
                # is achieved
                rate_inc, yr_inc = opts.retro_set[1:3]
                # Calculate progressively increasing multipliers to the early
                # retrofit rate based on user settings
                multipliers = {yr: 1 + ((rate_inc - 1) / (yr_inc - aeo_min)) *
                               (int(yr) - aeo_min) if int(yr) < yr_inc else
                               rate_inc for yr in self.aeo_years}
            # For each year, multiply starting early retrofit rate values by
            # rate multipliers to obtain final early retrofit rates by year;
            # nest by building type and technology component, consistent with
            # the structure of the starting values above
            self.retro_rate = {bldg: {cmpo: {
                yr: start_vals[bldg][cmpo] * multipliers[yr]
                for yr in self.aeo_years} for cmpo in start_vals[bldg].keys()}
                for bldg in start_vals.keys()}

        self.demand_tech = [
            'roof', 'ground', 'lighting gain', 'windows conduction',
            'equipment gain', 'floor', 'infiltration', 'people gain',
            'windows solar', 'ventilation', 'other heat gain', 'wall']
        self.zero_cost_tech = ['infiltration']
        self.inverted_relperf_list = ["ACH", "CFM/ft^2 @ 0.3 in. w.c.",
                                      "kWh/yr", "kWh/day", "SHGC", "HP/CFM",
                                      "kWh/cycle"]
        self.valid_submkt_urls = [
            '.eia.gov', '.doe.gov', '.energy.gov', '.data.gov',
            '.energystar.gov', '.epa.gov', '.census.gov', '.pnnl.gov',
            '.lbl.gov', '.nrel.gov', 'www.sciencedirect.com', 'www.costar.com',
            'www.navigantresearch.com']
        try:
            self.consumer_price_ind = numpy.genfromtxt(
                handyfiles.cpi_data,
                names=True, delimiter=',',
                dtype=[('DATE', 'U10'), ('VALUE', '<f8')])
            # Ensure that consumer price date is in expected format
            if len(self.consumer_price_ind['DATE'][0]) != 10:
                raise ValueError("CPI date format should be YYYY-MM-DD")
        except ValueError as e:
            raise ValueError(
                f"Error reading in '{handyfiles.cpi_data}': {str(e)}") from None
        # Read in commercial equipment capacity factors
        with open(handyfiles.cap_facts, 'r') as cpfc:
            try:
                self.cap_facts = json.load(cpfc)
            except ValueError:
                raise ValueError(f"Error reading in '{handyfiles.cap_facts}'")
        # Read in national-level site-source, emissions, and costs data
        with open(handyfiles.ss_data, 'r') as ss:
            try:
                cost_ss_carb = json.load(ss)
            except ValueError as e:
                raise ValueError(
                    "Error reading in '" +
                    handyfiles.ss_data + "': " + str(e)) from None
        # Set base-case emissions/cost data to use in assessing reductions for
        # non-fuel switching microsegments under a high grid decarbonization
        # case, if desired by the user
        if handyfiles.ss_data_nonfs is not None:
            # Read in national-level site-source, emissions, and costs data
            with open(handyfiles.ss_data_nonfs, 'r') as ss:
                try:
                    cost_ss_carb_nonfs = json.load(ss)
                except ValueError as e:
                    raise ValueError(
                        "Error reading in '" +
                        handyfiles.ss_data + "': " + str(e)) from None
        else:
            cost_ss_carb_nonfs = None
        # Set national site to source conversion factors
        self.ss_conv = {
            "electricity": cost_ss_carb[
                "electricity"]["site to source conversion"]["data"],
            "natural gas": {yr: 1 for yr in self.aeo_years},
            "distillate": {yr: 1 for yr in self.aeo_years},
            "other fuel": {yr: 1 for yr in self.aeo_years}}
        # Set electric emissions intensities and prices differently
        # depending on whether EMM regions are specified (use EMM-specific
        # data) or not (use national data)
        if self.regions in ["EMM", "State"]:
            # Read in EMM- or state-specific emissions factors and price data
            with open(handyfiles.ss_data_altreg, 'r') as ss:
                try:
                    cost_ss_carb_altreg = json.load(ss)
                except ValueError as e:
                    raise ValueError(
                        "Error reading in '" +
                        handyfiles.ss_data_altreg + "': " + str(e)) from None
            # Set base-case emissions/cost data to use in assessing reductions
            # for non-fuel switching microsegments under a high grid
            # decarbonization case, if desired by the user
            if handyfiles.ss_data_altreg_nonfs is not None:
                # Read in EMM- or state-specific emissions factors and price
                # data
                with open(handyfiles.ss_data_altreg_nonfs, 'r') as ss:
                    try:
                        cost_ss_carb_altreg_nonfs = json.load(ss)
                    except ValueError:
                        raise ValueError(
                            f"Error reading in '{handyfiles.ss_data_altreg_nonfs}'")
            else:
                cost_ss_carb_altreg_nonfs = None
            # Initialize CO2 intensities based on electricity intensities by
            # EMM region or state; convert CO2 intensities from Mt/TWh site to
            # MMTon/MMBTu site to match expected multiplication by site energy
            self.carb_int = {bldg: {"electricity": {reg: {
                yr: round((cost_ss_carb_altreg["CO2 intensity of electricity"][
                    "data"][reg][yr] / 3412141.6331), 10) for
                yr in self.aeo_years} for reg in cost_ss_carb_altreg[
                    "CO2 intensity of electricity"]["data"].keys()}} for
                bldg in ["residential", "commercial"]}
            # Initialize energy costs based on electricity prices by EMM region
            # or state; convert prices from $/kWh site to $/MMBTu site to match
            # expected multiplication by site energy units
            self.ecosts = {bldg: {"electricity": {reg: {
                yr: round((cost_ss_carb_altreg["End-use electricity price"][
                    "data"][bldg][reg][yr] / 0.003412), 6) for
                yr in self.aeo_years} for reg in cost_ss_carb_altreg[
                    "End-use electricity price"]["data"][bldg].keys()}} for
                bldg in ["residential", "commercial"]}
            # Finalize base-case emissions/cost data to use in assessing
            # reductions for non-fuel switching microsegments under a high grid
            # decarbonization case, if desired by the user
            if cost_ss_carb_altreg_nonfs is not None:
                self.carb_int_nonfs = {bldg: {"electricity": {reg: {
                    yr: round((cost_ss_carb_altreg_nonfs[
                        "CO2 intensity of electricity"][
                        "data"][reg][yr] / 3412141.6331), 10) for
                    yr in self.aeo_years} for reg in cost_ss_carb_altreg_nonfs[
                        "CO2 intensity of electricity"]["data"].keys()}} for
                    bldg in ["residential", "commercial"]}
                self.ecosts_nonfs = {bldg: {"electricity": {reg: {
                    yr: round((cost_ss_carb_altreg_nonfs[
                        "End-use electricity price"][
                        "data"][bldg][reg][yr] / 0.003412), 6) for
                    yr in self.aeo_years} for reg in cost_ss_carb_altreg_nonfs[
                        "End-use electricity price"]["data"][bldg].keys()}} for
                    bldg in ["residential", "commercial"]}
            else:
                self.carb_int_nonfs, self.ecosts_nonfs = (
                    None for n in range(2))
        else:
            # Initialize CO2 intensities based on national CO2 intensities
            # for electricity; convert CO2 intensities from Mt/quad source to
            # Mt/MMBTu source to match expected multiplication by source energy
            self.carb_int = {bldg: {"electricity": {yr: cost_ss_carb[
                "electricity"]["CO2 intensity"]["data"][bldg][yr] /
                1000000000 for yr in self.aeo_years}} for bldg in [
                "residential", "commercial"]}
            # Initialize energy costs based on national electricity prices; no
            # conversion needed as the prices will be multiplied by MMBtu
            # source energy units and are already in units of $/MMBtu source
            self.ecosts = {bldg: {"electricity": {yr: cost_ss_carb[
                "electricity"]["price"]["data"][bldg][yr] for
                yr in self.aeo_years}} for bldg in [
                "residential", "commercial"]}
            # Finalize base-case emissions/cost data to use in assessing
            # reductions for non-fuel switching microsegments under a high grid
            # decarbonization case, if desired by the user
            if cost_ss_carb_nonfs is not None:
                self.carb_int_nonfs = {
                    bldg: {"electricity": {yr: cost_ss_carb_nonfs[
                        "electricity"]["CO2 intensity"]["data"][bldg][yr] /
                        1000000000 for yr in self.aeo_years}} for bldg in [
                        "residential", "commercial"]}
                self.ecosts_nonfs = {
                    bldg: {"electricity": {yr: cost_ss_carb_nonfs[
                        "electricity"]["price"]["data"][bldg][yr] for
                        yr in self.aeo_years}} for bldg in [
                        "residential", "commercial"]}
            else:
                self.carb_int_nonfs, self.ecosts_nonfs = (
                    None for n in range(2))
        # Pull non-electric CO2 intensities and energy prices and update
        # the CO2 intensity and energy cost dicts initialized above
        # accordingly; convert CO2 intensities from Mt/quad source to
        # Mt/MMBTu source to match expected multiplication by source energy;
        # price data are already in units of $/MMBtu source and do not require
        # further conversion
        carb_int_nonelec = {bldg: {fuel: {yr: (
            cost_ss_carb[fuel_map]["CO2 intensity"]["data"][
                bldg][yr] / 1000000000) for yr in self.aeo_years}
                for fuel, fuel_map in zip(
                ["natural gas", "distillate", "other fuel"],
                ["natural gas", "distillate", "propane"])
            } for bldg in ["residential", "commercial"]}
        ecosts_nonelec = {bldg: {fuel: {yr: cost_ss_carb[
            fuel_map]["price"]["data"][bldg][yr] for yr in
            self.aeo_years} for fuel, fuel_map in zip([
                "natural gas", "distillate", "other fuel"], [
                "natural gas", "distillate", "propane"])} for bldg in [
            "residential", "commercial"]}
        for bldg in ["residential", "commercial"]:
            self.carb_int[bldg].update(carb_int_nonelec[bldg])
            self.ecosts[bldg].update(ecosts_nonelec[bldg])
            # Update base-case emissions/cost data to use in
            # assessing reductions for non-fuel switching microsegments
            # under a high grid decarbonization case to reflect non-electric
            # emissions intensities/energy costs
            if self.carb_int_nonfs is not None:
                self.carb_int_nonfs[bldg].update(carb_int_nonelec[bldg])
            if self.ecosts_nonfs is not None:
                self.ecosts_nonfs[bldg].update(ecosts_nonelec[bldg])
        # Set carbon costs
        ccosts_init = cost_ss_carb["CO2 price"]["data"]
        # Multiply carbon costs by 1000000 to reflect
        # conversion from import units of $/MTon to $/MMTon
        self.ccosts = {
            yr_key: (ccosts_init[yr_key] * 1000000) for
            yr_key in self.aeo_years}
        self.com_timeprefs = {
            "rates": [10.0, 1.0, 0.45, 0.25, 0.15, 0.065, 0.0],
            "distributions": {
                "heating": {
                    key: [0.265, 0.226, 0.196, 0.192, 0.105, 0.013, 0.003]
                    for key in self.aeo_years},
                "cooling": {
                    key: [0.264, 0.225, 0.193, 0.192, 0.106, 0.016, 0.004]
                    for key in self.aeo_years},
                "water heating": {
                    key: [0.263, 0.249, 0.212, 0.169, 0.097, 0.006, 0.004]
                    for key in self.aeo_years},
                "ventilation": {
                    key: [0.265, 0.226, 0.196, 0.192, 0.105, 0.013, 0.003]
                    for key in self.aeo_years},
                "cooking": {
                    key: [0.261, 0.248, 0.214, 0.171, 0.097, 0.005, 0.004]
                    for key in self.aeo_years},
                "lighting": {
                    key: [0.264, 0.225, 0.193, 0.193, 0.085, 0.013, 0.027]
                    for key in self.aeo_years},
                "refrigeration": {
                    key: [0.262, 0.248, 0.213, 0.170, 0.097, 0.006, 0.004]
                    for key in self.aeo_years}}}
        # Load external data on conversion rates for HP measures
        if opts.exog_hp_rates is not False:
            with open(handyfiles.hp_convert_rates, 'r') as fs_r:
                try:
                    self.hp_rates = json.load(fs_r)
                except ValueError:
                    print(f"Error reading in 'f{handyfiles.hp_convert_rates}'")
            # Set a priori assumptions about which non-elec-HP heating/cooling
            # technologies in commercial buildings are part of an RTU config.
            # vs. not; this is necessary to choose the appropriate exogenous
            # fuel switching rates for such technologies, if applicable

            # Use RTU HP fuel switching rates for furnace and/or small electric
            # resistance + AC tech.
            self.com_RTU_fs_tech = [
                "gas_furnace", "oil_furnace", "electric_res-heat",
                "rooftop_AC", "wall-window_room_AC", "res_type_central_AC"]
            # Use non-RTU HP fuel switching rates for boiler/chiller tech.
            # and/or gas chillers/HPs
            self.com_nRTU_fs_tech = [
                "elec_boiler", "gas_eng-driven_RTHP-heat",
                "res_type_gasHP-heat", "gas_boiler", "oil_boiler",
                "scroll_chiller", "reciprocating_chiller",
                "centrifugal_chiller", "screw_chiller",
                "gas_eng-driven_RTAC", "gas_chiller", "res_type_gasHP-cool",
                "gas_eng-driven_RTHP-cool"]
        # Fugitive refrigerant emissions calculations also require
        # understanding of which commercial heating/cooling technologies fall
        # into the RTU/small commercial category vs. large commercial category
        elif opts.fugitive_emissions is not False:
            self.hp_rates = None
            self.com_RTU_fs_tech = [
                "gas_furnace", "oil_furnace", "electric_res-heat",
                "rooftop_AC", "wall-window_room_AC", "res_type_central_AC"]
            self.com_nRTU_fs_tech = [
                "elec_boiler", "gas_eng-driven_RTHP-heat",
                "res_type_gasHP-heat", "gas_boiler", "oil_boiler",
                "scroll_chiller", "reciprocating_chiller",
                "centrifugal_chiller", "screw_chiller",
                "gas_eng-driven_RTAC", "gas_chiller", "res_type_gasHP-cool",
                "gas_eng-driven_RTHP-cool"]
        else:
            self.hp_rates, self.com_RTU_fs_tech, self.com_nRTU_fs_tech = (
                None for n in range(3))
        self.resist_ht_wh_tech = [
                "elec_boiler", "electric_res-heat", "resistance heat",
                "electric WH", "elec_booster_water_heater",
                "elec_water_heater", "Solar water heater", "solar WH"]
        self.minor_hvac_tech = [
                "room AC", "wall-window_room_AC", "secondary heater",
                "secondary heater (wood)", "secondary heater (coal)",
                "secondary heater (kerosene)", "secondary heater (LPG)"]

        # Load external refrigerant and supply chain methane leakage data
        # to assess fugitive emissions sources
        if opts.fugitive_emissions is not False:
            with open(handyfiles.fug_emissions_dat, 'r') as fs_r:
                try:
                    self.fug_emissions = json.load(fs_r)
                except ValueError:
                    print(f"Error reading in '{handyfiles.fug_emissions_dat}'")
        else:
            self.fug_emissions = None

        # Set valid region names and regional output categories
        if opts.alt_regions == "AIA":
            valid_regions = [
             "AIA_CZ1", "AIA_CZ2", "AIA_CZ3", "AIA_CZ4", "AIA_CZ5"]
            regions_out = [
                ('AIA CZ1', 'AIA_CZ1'), ('AIA CZ2', 'AIA_CZ2'),
                ('AIA CZ3', 'AIA_CZ3'), ('AIA CZ4', 'AIA_CZ4'),
                ('AIA CZ5', 'AIA_CZ5')]
            self.region_cpl_mapping = ''
            # Read in mapping for alternate performance/cost unit breakouts
            # IECC -> AIA mapping
            try:
                iecc_reg_map = numpy.genfromtxt(
                    handyfiles.iecc_reg_map,
                    names=True, delimiter='\t', dtype=(
                        ['<U25'] * 1 + ['<f8'] * len(valid_regions)))
            except ValueError as e:
                raise ValueError(
                    f"Error reading in '{handyfiles.iecc_reg_map}': {str(e)}") from None
            # Store alternate breakout mapping in dict for later use
            self.alt_perfcost_brk_map = {
                "IECC": iecc_reg_map, "levels": str([
                    "IECC_CZ" + str(n + 1) for n in range(8)])}
            # Read in state -> AIA mapping data only when methane leakage is
            # assessed
            if opts.fugitive_emissions is not False and \
                    opts.fugitive_emissions[0] in ['1', '3']:
                try:
                    self.fugitive_emissions_map = numpy.genfromtxt(
                        handyfiles.state_aia_map, names=True,
                        delimiter='\t', dtype=(['<U25'] * 1 + ['<f8'] * 51))
                except ValueError as e:
                    raise ValueError(
                        f"Error reading in '{handyfiles.state_aia_map}': {str(e)}") from None
            else:
                self.fugitive_emissions_map = None
            # HP conversion rates unsupported for AIA regional breakouts
            self.hp_rates_reg_map = None
        elif opts.alt_regions in ["EMM", "State"]:
            if opts.alt_regions == "EMM":
                valid_regions = [
                    'TRE', 'FRCC', 'MISW', 'MISC', 'MISE', 'MISS',
                    'ISNE', 'NYCW', 'NYUP', 'PJME', 'PJMW', 'PJMC',
                    'PJMD', 'SRCA', 'SRSE', 'SRCE', 'SPPS', 'SPPC',
                    'SPPN', 'SRSG', 'CANO', 'CASO', 'NWPP', 'RMRG', 'BASN']
                self.region_cpl_mapping = ''
                try:
                    self.ash_emm_map = numpy.genfromtxt(
                        handyfiles.ash_emm_map, names=True, delimiter='\t',
                        dtype=(['<U25'] * 1 + ['<f8'] * len(valid_regions)))
                except ValueError as e:
                    raise ValueError(
                        f"Error reading in '{handyfiles.ash_emm_map}': {str(e)}") from None
                # If applicable, pull regional mapping needed to read in
                # HP conversion rate data for certain measures/microsegments
                if self.hp_rates:
                    self.hp_rates_reg_map = {
                        "midwest": [
                            "SPPN", "MISW", "SPPC", "MISC",
                            "PJMW", "PJMC", "MISE"],
                        "northeast": [
                            "PJME", "NYCW", "NYUP", "ISNE"],
                        "south": [
                            "SPPS", "TRE", "MISS", "SRCE", "PJMD",
                            "SRCA", "SRSE", "FRCC"],
                        "west": [
                            "NWPP", "BASN", "RMRG", "SRSG", "CASO", "CANO"]
                    }
                else:
                    self.hp_rates_reg_map = None
                # Read in state -> EMM mapping data only when methane leakage
                # is assessed
                if opts.fugitive_emissions is not False and \
                        opts.fugitive_emissions[0] in ['1', '3']:
                    try:
                        self.fugitive_emissions_map = numpy.genfromtxt(
                            handyfiles.state_emm_map, names=True,
                            delimiter='\t', dtype=(['<U25'] * 1 + ['<f8'] * 51))
                    except ValueError as e:
                        raise ValueError(
                            f"Error reading in '{handyfiles.state_emm_map}': {str(e)}") from None
                else:
                    self.fugitive_emissions_map = None
            else:
                # Note: for now, exclude AK and HI
                valid_regions = [
                    'AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL',
                    'GA', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME',
                    'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH',
                    'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI',
                    'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI',
                    'WY']
                self.region_cpl_mapping = {
                    "new england": ['CT', 'MA', 'ME', 'NH', 'RI', 'VT'],
                    "mid atlantic": ['NJ', 'NY', 'PA'],
                    "east north central": ['IL', 'IN', 'MI', 'OH', 'WI'],
                    "west north central": [
                        'IA', 'KS', 'MN', 'MO', 'ND', 'NE', 'SD'],
                    "south atlantic": [
                        'DC', 'DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'WV'],
                    "east south central": ['AL', 'KY', 'MS', 'TN'],
                    "west south central": ['AR', 'LA', 'OK', 'TX'],
                    "mountain": [
                        'AZ', 'CO', 'ID', 'MT', 'NM', 'NV', 'UT', 'WY'],
                    "pacific": ['AK', 'CA', 'HI', 'OR', 'WA']}
                # If applicable, pull regional mapping needed to read in
                # HP conversion rate data for certain measures/microsegments
                if self.hp_rates:
                    self.hp_rates_reg_map = {
                        "midwest": [
                            "ND", "SD", "NE", "KS", "MO", "IA", "MN", "WI",
                            "IL", "IN", "MI", "OH"],
                        "northeast": [
                            "PA", "NY", "NJ", "CT", "RI", "MA", "VT", "NH",
                            "ME"],
                        "south": [
                            "TX", "OK", "AR", "LA", "MS", "AL", "GA", "FL",
                            "SC", "NC", "TN", "KY", "WV", "VA", "DC", "MD",
                            "DE"],
                        "west": [
                            "WA", "OR", "ID", "MT", "WY", "CA", "NV", "UT",
                            "AZ", "NM", "CO", "AK", "HI"]
                    }
                else:
                    self.hp_rates_reg_map = None
            regions_out = [(x, x) for x in valid_regions]

            # Read in mapping for alternate performance/cost unit breakouts
            # AIA -> EMM or State mapping
            try:
                # Hard code number of valid states at 51 (includes DC) to avoid
                # potential issues later when indexing numpy columns by state
                if opts.alt_regions == "State":
                    len_reg = 51
                else:
                    len_reg = len(valid_regions)
                # Read in the data
                aia_altreg_map = numpy.genfromtxt(
                    handyfiles.aia_altreg_map, names=True, delimiter='\t',
                    dtype=(['<U25'] * 1 + ['<f8'] * len_reg))
            except ValueError:
                raise ValueError(
                    f"Error reading in '{str(handyfiles.aia_altreg_map)}'")
            # IECC -> EMM or State mapping
            try:
                iecc_altreg_map = numpy.genfromtxt(
                    handyfiles.iecc_reg_map, names=True, delimiter='\t',
                    dtype=(['<U25'] * 1 + ['<f8'] * len(valid_regions)))
            except ValueError as e:
                raise ValueError(
                    f"Error reading in '{handyfiles.iecc_reg_map}': {str(e)}") from None
            # Store alternate breakout mapping in dict for later use
            self.alt_perfcost_brk_map = {
                "IECC": iecc_altreg_map, "AIA": aia_altreg_map,
                "levels": str([
                    "IECC_CZ" + str(n + 1) for n in range(8)]) + " 0R " + str([
                        "AIA_CZ" + str(n + 1) for n in range(5)])}
        self.months = ["january", "february", "march", "april", "may", "june",
                       "july", "august", "september", "october", "november",
                       "december"]
        self.in_all_map = {
            "climate_zone": valid_regions,
            "bldg_type": {
                "residential": [
                    "single family home", "multi family home", "mobile home"],
                "commercial": [
                    "assembly", "education", "food sales", "food service",
                    "health care", "lodging", "large office", "small office",
                    "mercantile/service", "warehouse", "other",
                    "unspecified"]},
            "structure_type": ["new", "existing"],
            "fuel_type": {
                "residential": [
                    "electricity", "natural gas", "distillate", "other fuel"],
                "commercial": [
                    "electricity", "natural gas", "distillate"]},
            "end_use": {
                "residential": {
                    "electricity": [
                        'drying', 'other', 'water heating',
                        'cooling', 'cooking', 'computers', 'lighting',
                        'secondary heating', 'TVs', 'heating', 'refrigeration',
                        'fans and pumps', 'ceiling fan'],
                    "natural gas": [
                        'drying', 'water heating', 'cooling', 'heating',
                        'cooking', 'secondary heating', 'other'],
                    "distillate": [
                        'water heating', 'heating', 'secondary heating',
                        'other'],
                    "other fuel": [
                        'water heating', 'cooking', 'heating',
                        'secondary heating', 'other']},
                "commercial": {
                    "electricity": [
                        'ventilation', 'water heating', 'cooling',
                        'heating', 'refrigeration', 'MELs',
                        'non-PC office equipment', 'PCs', 'lighting',
                        'cooking', "unspecified"],
                    "natural gas": [
                        'cooling', 'water heating', 'cooking', 'heating',
                        'other', 'unspecified'],
                    "distillate": [
                        'water heating', 'heating', 'other', 'unspecified']}},
            "technology": {
                "residential": {
                    "supply": {
                        "electricity": {
                            'other': [
                                'dishwasher', 'clothes washing', 'freezers',
                                'rechargeables', 'coffee maker',
                                'dehumidifier', 'electric other',
                                'small kitchen appliances', 'microwave',
                                'smartphones', 'pool heaters', 'pool pumps',
                                'security system', 'portable electric spas',
                                'smart speakers', 'tablets', 'wine coolers'],
                            'water heating': ['solar WH', 'electric WH'],
                            'cooling': [
                                'room AC', 'ASHP', 'GSHP', 'central AC'],
                            'computers': [
                                'desktop PC', 'laptop PC', 'network equipment',
                                'monitors'],
                            'lighting': [
                                'linear fluorescent (T-8)',
                                'linear fluorescent (T-12)',
                                'reflector (LED)', 'general service (CFL)',
                                'external (high pressure sodium)',
                                'general service (incandescent)',
                                'external (CFL)',
                                'external (LED)', 'reflector (CFL)',
                                'reflector (incandescent)',
                                'general service (LED)',
                                'external (incandescent)',
                                'linear fluorescent (LED)',
                                'reflector (halogen)'],
                            'secondary heating': ['secondary heater'],
                            'TVs': [
                                'home theater and audio', 'set top box',
                                'video game consoles', 'TV',
                                'OTT streaming devices'],
                            'heating': ['GSHP', 'resistance heat', 'ASHP'],
                            'ceiling fan': [None],
                            'fans and pumps': [None],
                            'refrigeration': [None],
                            'drying': [None],
                            'cooking': [None]},
                        "natural gas": {
                            'cooling': ['NGHP'],
                            'heating': ['furnace (NG)', 'NGHP', 'boiler (NG)'],
                            'secondary heating': ['secondary heater'],
                            'drying': [None],
                            'water heating': [None],
                            'cooking': [None],
                            'other': ["other appliances"]},
                        "distillate": {
                            'heating': [
                                'boiler (distillate)', 'furnace (distillate)'],
                            'secondary heating': ['secondary heater'],
                            'water heating': [None],
                            'other': ["other appliances"]},
                        "other fuel": {
                            'heating': [
                                'furnace (kerosene)',
                                'stove (wood)', 'furnace (LPG)'],
                            'secondary heating': [
                                'secondary heater (wood)',
                                'secondary heater (coal)',
                                'secondary heater (kerosene)',
                                'secondary heater (LPG)'],
                            'cooking': [None],
                            'water heating': [None],
                            'other': ["other appliances"]}},
                    "demand": [
                        'roof', 'ground', 'windows solar',
                        'windows conduction', 'equipment gain',
                        'people gain', 'wall', 'infiltration']},
                "commercial": {
                    "supply": {
                        "electricity": {
                            'ventilation': ['VAV_Vent', 'CAV_Vent'],
                            'water heating': [
                                'Solar water heater', 'HP water heater',
                                'elec_booster_water_heater',
                                'elec_water_heater'],
                            'cooling': [
                                'rooftop_AC', 'scroll_chiller',
                                'res_type_central_AC', 'reciprocating_chiller',
                                'comm_GSHP-cool', 'centrifugal_chiller',
                                'rooftop_ASHP-cool', 'wall-window_room_AC',
                                'screw_chiller'],
                            'heating': [
                                'electric_res-heat', 'comm_GSHP-heat',
                                'rooftop_ASHP-heat', 'elec_boiler'],
                            'refrigeration': [
                                'Commercial Beverage Merchandisers',
                                'Commercial Compressor Rack Systems',
                                'Commercial Condensers',
                                'Commercial Ice Machines',
                                'Commercial Reach-In Freezers',
                                'Commercial Reach-In Refrigerators',
                                'Commercial Refrigerated Vending Machines',
                                'Commercial Supermarket Display Cases',
                                'Commercial Walk-In Freezers',
                                'Commercial Walk-In Refrigerators'],
                            'MELs': [
                                'distribution transformers',
                                'kitchen ventilation', 'security systems',
                                'lab fridges and freezers',
                                'medical imaging', 'large video boards',
                                'coffee brewers', 'non-road electric vehicles',
                                'fume hoods', 'laundry', 'elevators',
                                'escalators', 'IT equipment', 'office UPS',
                                'data center UPS', 'shredders',
                                'private branch exchanges',
                                'voice-over-IP telecom',
                                'point-of-sale systems', 'warehouse robots',
                                'televisions',  'water services',
                                'telecom systems', 'other'
                            ],
                            'lighting': [
                                '100W A19 Incandescent',
                                '100W Equivalent A19 Halogen',
                                '100W Equivalent CFL Bare Spiral',
                                '100W Equivalent LED A Lamp',
                                'Halogen Infrared Reflector (HIR) PAR38',
                                'Halogen PAR38',
                                'LED Integrated Luminaire',
                                'LED PAR38',
                                'Mercury Vapor',
                                'Metal Halide',
                                'Sodium Vapor',
                                'T5 4xF54 HO High Bay',
                                'T5 F28',
                                'T8 F28',
                                'T8 F32',
                                'T8 F59',
                                'T8 F96'
                            ],
                            'cooking': [
                                'electric_range_oven_24x24_griddle'],
                            'PCs': [None],
                            'non-PC office equipment': [None],
                            'unspecified': [None]},
                        "natural gas": {
                            'cooling': [
                                'gas_eng-driven_RTAC', 'gas_chiller',
                                'res_type_gasHP-cool',
                                'gas_eng-driven_RTHP-cool'],
                            'water heating': [
                                'gas_water_heater', 'gas_instantaneous_WH',
                                'gas_booster_WH'],
                            'cooking': [
                                'gas_range_oven_24x24_griddle'],
                            'heating': [
                                'gas_eng-driven_RTHP-heat',
                                'res_type_gasHP-heat', 'gas_boiler',
                                'gas_furnace'],
                            'other': [None],
                            'unspecified': [None]},
                        "distillate": {
                            'water heating': ['oil_water_heater'],
                            'heating': ['oil_boiler', 'oil_furnace'],
                            'other': [None],
                            'unspecified': [None]}},
                    "demand": [
                        'roof', 'ground', 'lighting gain',
                        'windows conduction', 'equipment gain',
                        'floor', 'infiltration', 'people gain',
                        'windows solar', 'ventilation',
                        'other heat gain', 'wall']}}}
        # Find the full set of valid names for describing a measure's
        # applicable baseline that do not begin with 'all'
        mktnames_non_all = self.append_keyvals(
            self.in_all_map, keyval_list=[]) + ['supply', 'demand']
        # Find the full set of valid names for describing a measure's
        # applicable baseline that do begin with 'all'
        mktnames_all_init = ["all", "all residential", "all commercial"] + \
            self.append_keyvals(self.in_all_map["end_use"], keyval_list=[])
        mktnames_all = ['all ' + x if 'all' not in x else x for
                        x in mktnames_all_init]
        self.valid_mktnames = mktnames_non_all + mktnames_all
        if opts.detail_brkout in ['1', '2', '5', '6']:
            self.out_break_czones = OrderedDict(regions_out)
        else:
            if opts.alt_regions == "EMM":
                # Map to modified version of AVERT regions
                self.out_break_czones = OrderedDict([
                    ("Northwest", ["NWPP"]),
                    ("Great Basin", ["BASN"]),
                    ("California", ["CASO", "CANO"]),
                    ("Rocky Mountains", ["RMRG"]),
                    ("Upper Midwest", ["SPPN", "MISW", "MISC"]),
                    ("Lower Midwest", ["SPPC", "SPPS"]),
                    ("Lakes/Mid-Atl.", [
                        "MISE", "PJMW", "PJMC", "PJME"]),
                    ("Texas", ["TRE"]),
                    ("Southwest", ["SRSG"]),
                    ("Southeast", ["PJMD", "SRCA", "SRSE", "FRCC",
                                   "MISS", "SRCE"]),
                    ("Northeast", ["NYCW", "NYUP", "ISNE"])])
            elif opts.alt_regions == "State":
                # Map to Census subregions
                self.out_break_czones = OrderedDict([
                    ("New England", ['CT', 'MA', 'ME', 'NH', 'RI', 'VT']),
                    ("Mid Atlantic", ['NJ', 'NY', 'PA']),
                    ("East North Central", ['IL', 'IN', 'MI', 'OH', 'WI']),
                    ("West North Central", [
                        'IA', 'KS', 'MN', 'MO', 'ND', 'NE', 'SD']),
                    ("South Atlantic", [
                        'DC', 'DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'WV']),
                    ("East South Central", ['AL', 'KY', 'MS', 'TN']),
                    ("West South Central", ['AR', 'LA', 'OK', 'TX']),
                    ("Mountain", [
                        'AZ', 'CO', 'ID', 'MT', 'NM', 'NV', 'UT', 'WY']),
                    ("Pacific", ['AK', 'CA', 'HI', 'OR', 'WA'])])
            else:
                self.out_break_czones = OrderedDict(regions_out)

        if opts.detail_brkout in ['1', '3', '5', '7']:
            # Map to more granular building type definition
            self.out_break_bldgtypes = OrderedDict([
                ('Single Family Homes', [
                    'new', 'existing', 'single family home', 'mobile home']),
                ('Multi Family Homes', [
                    'new', 'existing', 'multi family home']),
                ('Hospitals', ['new', 'existing', 'health care']),
                ('Large Offices', ['new', 'existing', 'large office']),
                ('Small/Medium Offices', ['new', 'existing', 'small office']),
                ('Retail', ['new', 'existing', 'food sales',
                            'mercantile/service']),
                ('Hospitality', [
                    'new', 'existing', 'lodging', 'food service']),
                ('Education', ['new', 'existing', 'education']),
                ('Assembly/Other', [
                    'new', 'existing', 'assembly', 'other', 'unspecified']),
                ('Warehouses', ['new', 'existing', 'warehouse'])])
        else:
            self.out_break_bldgtypes = OrderedDict([
                ('Residential (New)', [
                    'new', 'single family home', 'multi family home',
                    'mobile home']),
                ('Residential (Existing)', [
                    'existing', 'single family home', 'multi family home',
                    'mobile home'],),
                ('Commercial (New)', [
                    'new', 'assembly', 'education', 'food sales',
                    'food service', 'health care', 'mercantile/service',
                    'lodging', 'large office', 'small office', 'warehouse',
                    'other', 'unspecified']),
                ('Commercial (Existing)', [
                    'existing', 'assembly', 'education', 'food sales',
                    'food service', 'health care', 'mercantile/service',
                    'lodging', 'large office', 'small office', 'warehouse',
                    'other', 'unspecified'])])
        self.out_break_enduses = OrderedDict([
            ('Heating (Equip.)', ["heating", "secondary heating"]),
            ('Cooling (Equip.)', ["cooling"]),
            ('Heating (Env.)', ["heating", "secondary heating"]),
            ('Cooling (Env.)', ["cooling"]),
            ('Ventilation', ["ventilation"]),
            ('Lighting', ["lighting"]),
            ('Water Heating', ["water heating"]),
            ('Refrigeration', ["refrigeration", "other"]),
            ('Cooking', ["cooking"]),
            ('Computers and Electronics', [
                "PCs", "non-PC office equipment", "TVs", "computers"]),
            ('Other', [
                "drying", "ceiling fan", "fans and pumps",
                "MELs", "other", "unspecified"])])
        self.out_break_eus_w_fsplits = [
            "Heating (Equip.)", "Cooling (Equip.)", "Heating (Env.)",
            "Cooling (Env.)", "Water Heating", "Cooking", "Other"]
        # Configure output breakouts for fuel type if user has set this option
        if opts.split_fuel is True:
            if opts.detail_brkout in ['1', '4', '6', '7']:
                # Map to more granular fuel type breakout
                self.out_break_fuels = OrderedDict([
                    ('Electric', ["electricity"]),
                    ('Natural Gas', ["natural gas"]),
                    ('Propane', ["other fuel"]),
                    ('Distillate/Other', ['distillate', 'other fuel']),
                    ('Biomass', ["other fuel"])])
            else:
                self.out_break_fuels = OrderedDict([
                    ('Electric', ["electricity"]),
                    ('Non-Electric', [
                        "natural gas", "distillate", "other fuel"])])
        else:
            self.out_break_fuels = {}
        # Use the above output categories to establish a dictionary with blank
        # values at terminal leaf nodes; this dict will eventually store
        # partitioning fractions needed to breakout the measure results
        # Determine all possible outcome category combinations
        out_levels = [
            self.out_break_czones.keys(), self.out_break_bldgtypes.keys(),
            self.out_break_enduses.keys()]
        out_levels_keys = list(itertools.product(*out_levels))
        # Create dictionary using outcome category combinations as key chains
        self.out_break_in = OrderedDict()
        for kc in out_levels_keys:
            current_level = self.out_break_in
            for ind, elem in enumerate(kc):
                # If fuel splits are desired and applicable for the current
                # end use breakout, add the fuel splits to the dict vals
                if len(self.out_break_fuels.keys()) != 0 and (
                        elem in self.out_break_eus_w_fsplits) and \
                        elem not in current_level:
                    current_level[elem] = OrderedDict(
                        [(x, OrderedDict()) for x in
                         self.out_break_fuels.keys()])
                # Otherwise, set dict vals to another empty dict
                elif elem not in current_level:
                    current_level[elem] = OrderedDict()
                current_level = current_level[elem]
        self.cconv_bybldg_units = [
            "$/ft^2 glazing", "$/ft^2 roof", "$/ft^2 wall",
            "$/ft^2 footprint", "$/ft^2 floor", "$/occupant", "$/node"]
        self.cconv_tech_mltstage_map = {
            "windows": {
                "key": ["$/ft^2 glazing"],
                "conversion stages": ["windows", "walls"]},
            "roof": {
                "key": ["$/ft^2 roof"],
                "conversion stages": ["roof", "footprint"]},
            "walls": {
                "key": ["$/ft^2 wall"],
                "conversion stages": ["walls"]},
            "footprint": {
                "key": ["$/ft^2 footprint"],
                "conversion stages": ["footprint"]}}
        self.tech_units_rmv = ["HHV"]
        # Note: EF handling for ECMs written before scout v0.5 (AEO 2019)
        self.tech_units_map = {
            "COP": {"AFUE": 1, "EER": 0.2930712},
            "EER": {"COP": 3.412},
            "AFUE": {"COP": 1}, "UEF": {"SEF": 1},
            "EF": {"UEF": 1, "SEF": 1, "CEF": 1},
            "SEF": {"UEF": 1}}
        self.com_eqp_eus_nostk = [
            "PCs", "non-PC office equipment", "MELs", "other"]
        self.res_lts_per_home = {
            "single family home": 36,
            "multi family home": 15,
            "mobile home": 19
        }
        # Assume that missing technology choice parameters come from the
        # appliances/MELs areas; default is thus the EIA choice parameters
        # for refrigerator technologies
        self.deflt_choice = [-0.01, -0.12]

        # Set valid types of TSV feature types
        self.tsv_feature_types = ["shed", "shift", "shape"]

        # Use EMM region setting as a proxy for desired time-sensitive
        # valuation (TSV) and associated need to initialize handy TSV variables
        if opts.alt_regions == "EMM":
            self.tsv_climate_regions = [
                "2A", "2B", "3A", "3B", "3C", "4A", "4B",
                "4C", "5A", "5B", "5C", "6A", "6B", "7"]
            self.tsv_nerc_regions = [
                "FRCC", "MRO", "NPCC", "RFC", "SERC", "SPP", "TRE", "WECC"]
            # Set a dict that maps each ASH climate zone to an EMM region
            # in the climate zone with the most representative set of
            # min/max system load hour and peak/take system load hour
            # windows to use for that climate zone. For most climates, two
            # of these representative regions is assumed to account for
            # varying types of renewable mixes (e.g., high solar vs. low
            # solar, which yield differences in net load shapes and net
            # peak/take periods). In these cases, the terminal value is
            # formatted as a list with the EMM region number with the
            # representative load hour data stored in the first element,
            # and all other EMM regions in the climate that are covered
            # by that representative load hour data stored in the second
            # element. NOTE: the selection of representative EMM regions
            # for each ASH region is based on the plots found here:
            # https://drive.google.com/drive/folders/
            # 1JSoQb78LgooUD_uXqBOzAC7Nl7eLJZnc?usp=sharing
            self.cz_emm_map = {
                "2A": {
                    "set 1": [2, (1, 2, 17)],
                    "set 2": [6, (6, 15)]},
                "2B": {
                    "set 1": [20, (1, 20)]},
                "3A": {
                    "set 1": [15, (6, 13, 14, 15, 16)],
                    "set 2": [1, (1, 17)]},
                "3B": {
                    "set 1": [22, (21, 22)],
                    "set 2": [25, (1, 17, 20, 25)]},
                "3C": {
                    "set 1": [21, (21, 22)]},
                "4A": {
                    "set 1": [10, (4, 8, 10, 11, 17, 18)],
                    "set 2": [16, (6, 13, 14, 15, 16)]},
                "4B": {
                    "set 1": [20, (1, 17, 20, 24)],
                    "set 2": [21, (21, 22)]},
                "4C": {
                    "set 1": [23, (23,)],
                    "set 2": [21, (21,)]},
                "5A": {
                    "set 1": [11, (3, 4, 7, 9, 10, 11, 18, 19, 24)],
                    "set 2": [5, (5, 12, 14)]},
                "5B": {
                    "set 1": [24, (20, 23, 24, 25)],
                    "set 2": [21, (21,)]},
                "5C": {
                    "set 1": [23, (23,)]},
                "6A": {
                    "set 1": [3, (3, 5, 19)],
                    "set 2": [7, (7, 9, 10, 24)]},
                "6B": {
                    "set 1": [23, (3, 19, 23, 24, 25)],
                    "set 2": [22, (21, 22)]},
                "7": {
                    "set 1": [3, (3, 19)],
                    "set 2": [24, (7, 24, 25)]}}
            if opts.tsv_metrics is not False:
                # Develop weekend day flags
                wknd_day_flags = [0 for n in range(365)]
                current_wkdy = 1
                for d in range(365):
                    # Flag weekend day
                    if current_wkdy in [1, 7]:
                        wknd_day_flags[d] = 1
                    # Advance day of week by one unless Saturday (7), in which
                    # case day switches back to 1 (Sunday)
                    if current_wkdy <= 6:
                        current_wkdy += 1
                    else:
                        current_wkdy = 1

                # Develop lists with seasonal day of year ranges, both with and
                # without weekends

                # Summer days of year
                sum_days = list(range(152, 274))
                sum_days_wkdy = [
                    x for x in sum_days if wknd_day_flags[(x - 1)] != 1]
                sum_days_wknd = [
                    x for x in sum_days if wknd_day_flags[(x - 1)] == 1]
                # Winter days of year
                wint_days = (list(
                            range(1, 91)) + list(range(335, 366)))
                wint_days_wkdy = [
                    x for x in wint_days if wknd_day_flags[(x - 1)] != 1]
                wint_days_wknd = [
                    x for x in wint_days if wknd_day_flags[(x - 1)] == 1]
                # Intermediate days of year
                inter_days = (list(
                            range(91, 152)) + list(range(274, 335)))
                inter_days_wkdy = [
                    x for x in inter_days if wknd_day_flags[(x - 1)] != 1]
                inter_days_wknd = [
                    x for x in inter_days if wknd_day_flags[(x - 1)] == 1]

                # Set column names for a dataset that includes information on
                # max/min net system load hours and peak/take net system load
                # hour windows by season and EMM region
                peak_take_names = (
                    "Region", "Year", "SummerMaxHr", "SummerMinHr",
                    "SummerPeakStartHr", "SummerPeakEndHr",
                    "SummerTakeStartHr1", "SummerTakeEndHr1",
                    "SummerTakeStartHr2", "SummerTakeEndHr2",
                    "SummerTakeStartHr3", "SummerTakeEndHr3",
                    "WinterMaxHr", "WinterMinHr",
                    "WinterPeakStartHr", "WinterPeakEndHr",
                    "WinterTakeStartHr1", "WinterTakeEndHr1",
                    "WinterTakeStartHr2", "WinterTakeEndHr2",
                    "WinterTakeStartHr3", "WinterTakeEndHr3",
                    "InterMaxHr", "InterMinHr",
                    "InterPeakStartHr", "InterPeakEndHr",
                    "InterTakeStartHr1", "InterTakeEndHr1",
                    "InterTakeStartHr2", "InterTakeEndHr2",
                    "InterTakeStartHr3", "InterTakeEndHr3")

                # Choose the appropriate data to use in determining peak/take
                # windows (total vs. net system load under reference vs. "Low
                # Renewable Cost" supply-side AEO case)
                if opts.tsv_metrics[-2] == "1":
                    metrics_data = handyfiles.tsv_metrics_data_tot_ref
                elif opts.tsv_metrics[-2] == "2":
                    metrics_data = handyfiles.tsv_metrics_data_tot_hr
                elif opts.tsv_metrics[-2] == "3":
                    metrics_data = handyfiles.tsv_metrics_data_net_ref
                else:
                    metrics_data = handyfiles.tsv_metrics_data_net_hr

                # Import system max/min and peak/take hour load by EMM region
                sysload_dat = numpy.genfromtxt(
                    metrics_data, names=peak_take_names, delimiter=',',
                    dtype="<i4", encoding="latin1", skip_header=1)
                # Find unique set of projection years in system peak/take data
                sysload_dat_yrs = numpy.unique(sysload_dat["Year"])
                # Set a dict that maps EMM region names to region
                # numbers as defined by EIA
                # (https://www.eia.gov/outlooks/aeo/pdf/f2.pdf)
                self.emm_name_num_map = {
                    name: (ind + 1) for ind, name in enumerate(valid_regions)}
                # Initialize a set of dicts that will store representative
                # system load data for the summer, winter, and intermediate
                # seasons by projection year
                sysld_sum, sysld_wint, sysld_int = ({
                    str(yr): {key: {key_sub: None for
                              key_sub in self.cz_emm_map[key].keys()} if
                              type(self.cz_emm_map[key]) is dict else None
                              for key in self.cz_emm_map.keys()} for yr in
                    sysload_dat_yrs} for n in range(3))
                # Fill in the dicts with seasonal system load data by year
                # Loop through all projection years available in the system
                # peak/take period data
                for sys_yr in sysload_dat_yrs:
                    # Convert projection year to string for dict keys
                    sys_yr_str = str(sys_yr)
                    sysload_dat_yr = sysload_dat[
                        numpy.where((sysload_dat["Year"] == sys_yr))]
                    # Loop through all climate zones
                    for cz in self.cz_emm_map.keys():
                        # Handle climate zones with one representative system
                        # load shape differently than those with more than one
                        # such shape
                        if type(self.cz_emm_map[cz]) == int:
                            # Fill in seasonal system load data
                            sysld_sum[sys_yr_str][cz], \
                                sysld_wint[sys_yr_str][cz], \
                                sysld_int[sys_yr_str][cz] = self.set_peak_take(
                                    sysload_dat_yr, self.cz_emm_map[cz])
                        else:
                            # Loop through the multiple EMM regions with
                            # representative system load data for the current
                            # climate zone
                            for set_v in self.cz_emm_map[cz].keys():
                                # Fill in seasonal system load data
                                sysld_sum[sys_yr_str][cz][set_v], \
                                    sysld_wint[sys_yr_str][cz][set_v], \
                                    sysld_int[sys_yr_str][cz][set_v] = \
                                    self.set_peak_take(
                                        sysload_dat_yr,
                                        self.cz_emm_map[cz][set_v][0])
                self.tsv_metrics_data = {
                    "season days": {
                        "all": {
                            "summer": sum_days,
                            "winter": wint_days,
                            "intermediate": inter_days
                        },
                        "weekdays": {
                            "summer": sum_days_wkdy,
                            "winter": wint_days_wkdy,
                            "intermediate": inter_days_wkdy

                        },
                        "weekends": {
                            "summer": sum_days_wknd,
                            "winter": wint_days_wknd,
                            "intermediate": inter_days_wknd

                        }
                    },
                    "system load hours": {
                        "summer": sysld_sum,
                        "winter": sysld_wint,
                        "intermediate": sysld_int
                    },
                    "peak days": {
                        "summer": {
                            "2A": 199,
                            "2B": 186,
                            "3A": 192,
                            "3B": 171,
                            "3C": 220,
                            "4A": 192,
                            "4B": 206,
                            "4C": 241,
                            "5A": 199,
                            "5B": 178,
                            "5C": 206,
                            "6A": 186,
                            "6B": 220,
                            "7": 206},
                        "winter": {
                            "2A": 24,
                            "2B": 17,
                            "3A": 31,
                            "3B": 10,
                            "3C": 10,
                            "4A": 31,
                            "4B": 339,
                            "4C": 38,
                            "5A": 26,
                            "5B": 10,
                            "5C": 12,
                            "6A": 10,
                            "6B": 17,
                            "7": 31}
                    },
                    "hourly index": list(enumerate(
                        itertools.product(range(365), range(24))))
                }
            else:
                self.tsv_metrics_data = None
                self.emm_name_num_map = {
                    name: (ind + 1) for ind, name in enumerate(valid_regions)}
            self.tsv_hourly_price, self.tsv_hourly_emissions = ({
                reg: None for reg in valid_regions
            } for n in range(2))

            self.tsv_hourly_lafs = {
                reg: {
                    "residential": {
                        bldg: {
                            eu: None for eu in self.in_all_map[
                                "end_use"]["residential"]["electricity"]
                        } for bldg in self.in_all_map[
                            "bldg_type"]["residential"]
                    },
                    "commercial": {
                        bldg: {
                            eu: None for eu in self.in_all_map[
                                "end_use"]["commercial"]["electricity"]
                        } for bldg in self.in_all_map[
                            "bldg_type"]["commercial"]
                    }
                } for reg in valid_regions
            }
        else:
            self.tsv_hourly_lafs = None

        # Condition health data scenario initialization on whether user
        # has requested that public health costs be accounted for
        if opts.health_costs is True:
            # For each health data scenario, set the intended measure name
            # appendage (tuple element 1), type of efficiency to attach health
            # benefits to (element 2), and column in the data file from which
            # to retrieve these benefit values (element 3)
            self.health_scn_names = [
                ("PHC-EE (low)", "Uniform EE", "2017cents_kWh_7pct_low"),
                ("PHC-EE (high)", "Uniform EE", "2017cents_kWh_3pct_high")]
            # Set data file with public health benefits information
            self.health_scn_data = numpy.genfromtxt(
                handyfiles.health_data,
                names=("AVERT_Region", "EMM_Region", "Category",
                       "2017cents_kWh_3pct_low", "2017cents_kWh_3pct_high",
                       "2017cents_kWh_7pct_low",
                       "2017cents_kWh_7pct_high"),
                delimiter=',', dtype=(['<U25'] * 3 + ['<f8'] * 4))
        self.env_heat_ls_scrn = (
            "windows solar", "equipment gain", "people gain",
            "other heat gain")
        self._allow_overwrite = allow_overwrite
        # self._wrap_dicts()
        self._initialized = True

    def _wrap_dicts(self):
        for name, value in self.__dict__.items():
            if isinstance(value, (dict, OrderedDict)):
                super().__setattr__(name, ImmutableDict(value))

    def __setattr__(self, name, value):
        if not hasattr(self, "_initialized"):
            super().__setattr__(name, value)
        elif hasattr(self, "_allow_overwrite") and self._allow_overwrite:
            super().__setattr__(name, value)
        else:
            raise ValueError(f"Cannot modify instance variables after initialization: {name}")

    def set_peak_take(self, sysload_dat, restrict_key):
        """Fill in dicts with seasonal system load shape data.

            Args:
                sysload_dat (numpy.ndarray): System load shape data.
                restrict_key (int): EMM region to restrict net load data to.

            Returns:
                Appropriate min/max net system load hour and peak/take net
                system load hour window data for the EMM region of interest,
                stored in dicts that are distinguished by season.
        """

        # Restrict net system load data to the representative EMM region for
        # the current climate zone
        peak_take_cz = sysload_dat[numpy.where(
            (sysload_dat["Region"] == restrict_key))]
        # Set summer max load hour, min load hour, and peak/take windows
        sum_peak_take = {
            "max": peak_take_cz["SummerMaxHr"][0],
            "min": peak_take_cz["SummerMinHr"][0],
            "peak range": list(range(peak_take_cz["SummerPeakStartHr"][0],
                                     peak_take_cz["SummerPeakEndHr"][0] + 1)),
            "take range": list(range(peak_take_cz["SummerTakeStartHr1"][0],
                                     peak_take_cz["SummerTakeEndHr1"][0] + 1))}
        # Set winter max load hour, min load hour, and peak/take windows
        wint_peak_take = {
            "max": peak_take_cz["WinterMaxHr"][0],
            "min": peak_take_cz["WinterMinHr"][0],
            "peak range": list(range(peak_take_cz["WinterPeakStartHr"][0],
                                     peak_take_cz["WinterPeakEndHr"][0] + 1)),
            "take range": list(range(peak_take_cz["WinterTakeStartHr1"][0],
                                     peak_take_cz["WinterTakeEndHr1"][0] + 1))}
        # Set intermediate max load hour, min load hour, and peak/take windows
        inter_peak_take = {
            "max": peak_take_cz["InterMaxHr"][0],
            "min": peak_take_cz["InterMinHr"][0],
            "peak range": list(range(peak_take_cz["InterPeakStartHr"][0],
                                     peak_take_cz["InterPeakEndHr"][0] + 1)),
            "take range": list(range(peak_take_cz["InterTakeStartHr1"][0],
                                     peak_take_cz["InterTakeEndHr1"][0] + 1))}
        # Handle cases where seasonal low demand periods cover two or three
        # non-contiguous time segments (e.g., 2-6AM, 10AM-2PM)

        # Loop through seasonal take variable names
        for seas in ["SummerTake", "WinterTake", "InterTake"]:
            # Loop through segment number in the variable name
            for seg in ["2", "3"]:
                # Sandwich start/end hour information between season and
                # segment information in the variable name
                st_key = seas + "StartHr" + seg
                end_key = seas + "EndHr" + seg
                # Check to see whether data are present for the given season
                # and segment (use segment starting hour variable as indicator)
                if numpy.isfinite(peak_take_cz[st_key][0]):
                    # Append additional low demand periods as appropriate for
                    # the given season
                    if "Summer" in seas:
                        sum_peak_take["take range"].extend(list(
                            range(peak_take_cz[st_key][0],
                                  peak_take_cz[end_key][0])))
                    elif "Winter" in seas:
                        wint_peak_take["take range"].extend(list(
                            range(peak_take_cz[st_key][0],
                                  peak_take_cz[end_key][0])))
                    else:
                        inter_peak_take["take range"].extend(list(
                            range(peak_take_cz[st_key][0],
                                  peak_take_cz[end_key][0])))

        return sum_peak_take, wint_peak_take, inter_peak_take

    def append_keyvals(self, dict1, keyval_list):
        """Append all terminal key values in a dict to a list.

        Note:
            Values already in the list should not be appended.

        Args:
            dict1 (dict): Dictionary with terminal key values
                to append.

        Returns:
            List including all terminal key values from dict.

        Raises:
            ValueError: If terminal key values are not formatted as
                either lists or strings.
        """
        for (k, i) in dict1.items():
            if isinstance(i, dict):
                self.append_keyvals(i, keyval_list)
            elif isinstance(i, list):
                keyval_list.extend([
                    x for x in i if x not in keyval_list])
            elif isinstance(i, str) and i not in keyval_list:
                keyval_list.append(i)
            else:
                raise ValueError(
                    "Input dict terminal key values expected to be "
                    "lists or strings in the 'append_keyvals' function"
                    "for ECM '" + self.name + "'")

        return keyval_list


class EPlusMapDicts(object):
    """Class of dicts used to map Scout measure definitions to EnergyPlus.

    Attributes:
        czone (dict): Scout-EnergyPlus climate zone mapping.
        bldgtype (dict): Scout-EnergyPlus building type mapping. Shown are
            the EnergyPlus commercial reference building names that correspond
            to each AEO commercial building type, and the weights needed in
            some cases to map multiple EnergyPlus reference building types to
            a single AEO type. See 'convert_data' JSON for more details.
        fuel (dict): Scout-EnergyPlus fuel type mapping.
        enduse (dict): Scout-EnergyPlus end use mapping.
        structure_type (dict): Scout-EnergyPlus structure type mapping.
    """

    def __init__(self):
        self.czone = {
            "sub arctic": "BA-SubArctic",
            "very cold": "BA-VeryCold",
            "cold": "BA-Cold",
            "marine": "BA-Marine",
            "mixed humid": "BA-MixedHumid",
            "mixed dry": "BA-MixedDry",
            "hot dry": "BA-HotDry",
            "hot humid": "BA-HotHumid"}
        self.bldgtype = {
            "assembly": {
                "Hospital": 1},
            "education": {
                "PrimarySchool": 0.26,
                "SecondarySchool": 0.74},
            "food sales": {
                "Supermarket": 1},
            "food service": {
                "QuickServiceRestaurant": 0.31,
                "FullServiceRestaurant": 0.69},
            "health care": None,
            "lodging": {
                "SmallHotel": 0.26,
                "LargeHotel": 0.74},
            "large office": {
                "LargeOfficeDetailed": 0.9,
                "MediumOfficeDetailed": 0.1},
            "small office": {
                "SmallOffice": 0.12,
                "OutpatientHealthcare": 0.88},
            "mercantile/service": {
                "RetailStandalone": 0.53,
                "RetailStripmall": 0.47},
            "warehouse": {
                "Warehouse": 1},
            "other": None,
            "unspecified": None}
        self.fuel = {
            'electricity': 'electricity',
            'natural gas': 'gas',
            'distillate': 'other_fuel'}
        self.enduse = {
            'heating': [
                'heating_electricity', 'heat_recovery_electricity',
                'humidification_electricity', 'pump_electricity',
                'heating_gas', 'heating_other_fuel'],
            'cooling': [
                'cooling_electricity', 'pump_electricity',
                'heat_rejection_electricity'],
            'water heating': [
                'service_water_heating_electricity',
                'service_water_heating_gas',
                'service_water_heating_other_fuel'],
            'ventilation': ['fan_electricity'],
            'cooking': [
                'interior_equipment_gas', 'interior_equipment_other_fuel'],
            'lighting': ['interior_lighting_electricity'],
            'refrigeration': ['refrigeration_electricity'],
            'PCs': ['interior_equipment_electricity'],
            'non-PC office equipment': ['interior_equipment_electricity'],
            'MELs': ['interior_equipment_electricity']}
        # Note: assumed year range for each structure vintage shown in lists
        self.structure_type = {
            "new": '90.1-2013',
            "retrofit": {
                '90.1-2004': [2004, 2009],
                '90.1-2010': [2010, 2012],
                'DOE Ref 1980-2004': [1980, 2003],
                'DOE Ref Pre-1980': [0, 1979]}}


class EPlusGlobals(object):
    """Class of global variables used in parsing EnergyPlus results file.

    Attributes:
        cbecs_sh (xlrd sheet object): CBECs square footages Excel sheet.
        vintage_sf (dict): Summary of CBECs square footages by vintage.
        eplus_coltypes (list): Expected EnergyPlus variable data types.
        eplus_basecols (list): Variable columns that should never be removed.
        eplus_perf_files (list): EnergyPlus simulation output file names.
        eplus_vintages (list): EnergyPlus building vintage types.
        eplus_vintage_weights (dicts): Square-footage-based weighting factors
            for EnergyPlus vintages.
    """

    def __init__(self, eplus_dir, cbecs_sf_byvint):
        # Set building vintage square footage data from CBECS
        self.vintage_sf = cbecs_sf_byvint
        self.eplus_coltypes = [
            ('building_type', '<U50'), ('climate_zone', '<U50'),
            ('template', '<U50'), ('measure', '<U50'), ('status', '<U50'),
            ('ep_version', '<U50'), ('os_version', '<U50'),
            ('timestamp', '<U50'), ('cooling_electricity', '<f8'),
            ('cooling_water', '<f8'), ('district_chilled_water', '<f8'),
            ('district_hot_water_heating', '<f8'),
            ('district_hot_water_service_hot_water', '<f8'),
            ('exterior_equipment_electricity', '<f8'),
            ('exterior_equipment_gas', '<f8'),
            ('exterior_equipment_other_fuel', '<f8'),
            ('exterior_equipment_water', '<f8'),
            ('exterior_lighting_electricity', '<f8'),
            ('fan_electricity', '<f8'),
            ('floor_area', '<f8'), ('generated_electricity', '<f8'),
            ('heat_recovery_electricity', '<f8'),
            ('heat_rejection_electricity', '<f8'),
            ('heating_electricity', '<f8'), ('heating_gas', '<f8'),
            ('heating_other_fuel', '<f8'), ('heating_water', '<f8'),
            ('humidification_electricity', '<f8'),
            ('humidification_water', '<f8'),
            ('interior_equipment_electricity', '<f8'),
            ('interior_equipment_gas', '<f8'),
            ('interior_equipment_other_fuel', '<f8'),
            ('interior_equipment_water', '<f8'),
            ('interior_lighting_electricity', '<f8'),
            ('net_site_electricity', '<f8'), ('net_water', '<f8'),
            ('pump_electricity', '<f8'),
            ('refrigeration_electricity', '<f8'),
            ('service_water', '<f8'),
            ('service_water_heating_electricity', '<f8'),
            ('service_water_heating_gas', '<f8'),
            ('service_water_heating_other_fuel', '<f8'), ('total_gas', '<f8'),
            ('total_other_fuel', '<f8'), ('total_site_electricity', '<f8'),
            ('total_water', '<f8')]
        self.eplus_basecols = [
            'building_type', 'climate_zone', 'template', 'measure']
        # Set EnergyPlus data file name list, given local directory
        self.eplus_perf_files = [
            f for f in listdir(eplus_dir) if
            isfile(join(eplus_dir, f)) and '_scout_' in f]
        # Import the first of the EnergyPlus measure performance files and use
        # it to establish EnergyPlus vintage categories
        eplus_file = numpy.genfromtxt(
            (eplus_dir + '/' + self.eplus_perf_files[0]), names=True,
            dtype=self.eplus_coltypes, delimiter=",", missing_values='')
        self.eplus_vintages = numpy.unique(eplus_file['template'])
        # Determine appropriate weights for mapping EnergyPlus vintages to the
        # 'new' and 'retrofit' building structure types of Scout
        self.eplus_vintage_weights = self.find_vintage_weights()

    def find_vintage_weights(self):
        """Find square-footage-based weighting factors for building vintages.

        Note:
            Use CBECs building vintage square footage data to derive weighting
            factors that will map the EnergyPlus building vintages to the 'new'
            and 'retrofit' building structure types of Scout.

        Returns:
            Weights needed to map each EnergyPlus vintage category to the 'new'
            and 'retrofit' structure types defined in Scout.

        Raises:
            ValueError: If vintage weights do not sum to 1.
            KeyError: If unexpected vintage names are discovered in the
                EnergyPlus file.
        """
        handydicts = EPlusMapDicts()

        # Set the expected names of the EnergyPlus building vintages and the
        # low and high year limits of each building vintage category
        expected_eplus_vintage_yr_bins = [
            handydicts.structure_type['new']] + \
            list(handydicts.structure_type['retrofit'].keys())
        # Initialize a variable meant to translate the summed square footages
        # of multiple 'retrofit' building vintages into weights that sum to 1;
        # also initialize a variable used to check that these weights indeed
        # sum to 1
        total_retro_sf, retro_weight_sum = (0 for n in range(2))

        # Check for expected EnergyPlus vintage names
        if sorted(self.eplus_vintages) == sorted(
                expected_eplus_vintage_yr_bins):
            # Initialize a dictionary with the EnergyPlus vintages as keys and
            # associated square footage values starting at zero
            eplus_vintage_weights = dict.fromkeys(self.eplus_vintages, 0)

            # Loop through the EnergyPlus vintages and assign associated
            # weights by mapping to cbecs square footage data
            for k in eplus_vintage_weights.keys():
                # If looping through the EnergyPlus vintage associated with the
                # 'new' Scout structure type, set vintage weight to 1 (only one
                # vintage category will be associated with this structure type)
                if k == handydicts.structure_type['new']:
                    eplus_vintage_weights[k] = 1
                # Otherwise, set EnergyPlus vintage weight initially to the
                # square footage that corresponds to that vintage in cbecs
                else:
                    # Loop through all cbecs vintage bins
                    for k2 in self.vintage_sf.keys():
                        # Find the limits of the cbecs vintage bin
                        cbecs_match = re.search(
                            r'(\D*)(\d*)(\s*)(\D*)(\s*)(\d*)', k2)
                        cbecs_t1 = cbecs_match.group(2)
                        cbecs_t2 = cbecs_match.group(6)
                        # Handle a 'Before Year X' case in cbecs (e.g., 'Before
                        # 1920'),  setting the lower year limit to zero
                        if cbecs_t2 == '':
                            cbecs_t2 = 0
                        # Determine a single average year that represents the
                        # current cbecs vintage bin
                        cbecs_yr = (int(cbecs_t1) + int(cbecs_t2)) / 2
                        # If the cbecs bin year falls within the year limits of
                        # the current EnergyPlus vintage bin, add the
                        # associated cbecs ft^2 data to the EnergyPlus
                        # vintage weight value
                        if cbecs_yr >= handydicts.structure_type[
                            'retrofit'][k][0] and \
                           cbecs_yr < handydicts.structure_type[
                           'retrofit'][k][1]:
                            eplus_vintage_weights[k] += self.vintage_sf[k2]
                            total_retro_sf += self.vintage_sf[k2]

            # Run through all EnergyPlus vintage weights, normalizing the
            # square footage-based weights for each 'retrofit' vintage to the
            # total square footage across all 'retrofit' vintage categories
            for k in eplus_vintage_weights.keys():
                # If running through the 'new' EnergyPlus vintage bin, register
                # the value of its weight (should be 1)
                if k == handydicts.structure_type['new']:
                    new_weight_sum = eplus_vintage_weights[k]
                # If running through a 'retrofit' EnergyPlus vintage bin,
                # normalize the square footage for that vintage by total
                # square footages across 'retrofit' vintages to arrive at the
                # final weight for that EnergyPlus vintage
                else:
                    eplus_vintage_weights[k] /= total_retro_sf
                    retro_weight_sum += eplus_vintage_weights[k]

            # Check that the 'new' EnergyPlus vintage weight equals 1 and that
            # all 'retrofit' EnergyPlus vintage weights sum to 1
            if new_weight_sum != 1:
                raise ValueError("Incorrect new vintage weight total when "
                                 "instantiating 'EPlusGlobals' object")
            elif retro_weight_sum != 1:
                raise ValueError("Incorrect retrofit vintage weight total when"
                                 "instantiating 'EPlusGlobals' object")

        else:
            raise KeyError(
                "Unexpected EnergyPlus vintage(s) when instantiating "
                "'EPlusGlobals' object; "
                "check EnergyPlus vintage assumptions in structure_type "
                "attribute of 'EPlusMapDict' object")

        return eplus_vintage_weights


class Measure(object):
    """Set up a class representing efficiency measures as objects.

    Attributes:
        **kwargs: Arbitrary keyword arguments used to fill measure attributes
            from an input dictionary.
        remove (boolean): Determines whether measure should be removed from
            analysis engine due to insufficient market source data.
        usr_opts (dict): Records several user command line input
            selections that affect measure outputs.
        eff_fs_splt (dict): Data needed to determine the fuel splits of
            efficient case results for fuel switching measures.
        handyvars (object): Global variables useful across class methods.
        sf_to_house (dict): Stores information for mapping stock units in
            sf to number of households, as applicable.
        retro_rate (float or list): Stock retrofit rate specific to the ECM.
        tech_switch_to (str, None): Technology switch to flag.
        technology_type (string): Flag for supply- or demand-side technology.
        yrs_on_mkt (list): List of years that the measure is active on market.
        markets (dict): Data grouped by adoption scheme on:
            a) 'master_mseg': a measure's master market microsegments (stock,
               energy, carbon, cost),
            b) 'mseg_adjust': all microsegments that contribute to each master
               microsegment (required later for measure competition).
            c) 'mseg_out_break': master microsegment breakdowns by key
               variables (climate zone, building class, end use)
        sector_shapes (dict): Sector-level hourly baseline and efficient load
            shapes by adopt scheme, EMM region, and year
    """

    def __init__(
            self, base_dir, handyvars, handyfiles, opts_dict, **kwargs):
        # Read Measure object attributes from measures input JSON.
        for key, value in kwargs.items():
            setattr(self, key, value)
        # Check to ensure that measure name is proper length for plotting;
        # for now, exempt measures with public health cost adders
        # *** COMMENT FOR NOW ***
        # if len(self.name) > 45 and "PHC" not in self.name:
        #     raise ValueError(
        #         "ECM '" + self.name + "' name must be <= 45 characters")
        self.remove = False
        # Set user options to the command line settings
        self.usr_opts = opts_dict
        # Check to ensure that proper settings are used for tsv metrics calcs
        if self.usr_opts["tsv_metrics"] is not False and (
            self.fuel_type not in ["electricity", ["electricity"]]) and \
                self.fuel_switch_to != "electricity":
            raise ValueError(
                "Non-electric fuel found for measure '" + self.name +
                " alongside '--tsv_metrics' option. Such metrics cannot "
                "be calculated for non-electric baseline segments of "
                "energy use. To address this issue, restrict the "
                "measure's fuel type to electricity.")
        # Reset health costs option to be more informative for the specific
        # measure, if available
        if self.usr_opts["health_costs"] is True:
            # Look for pre-determined health cost scenario names in the
            # UsefulVars class, "health_scn_names" attribute
            if "PHC-EE (low)" in self.name:
                self.usr_opts["health_costs"] = "Uniform EE-low"
            elif "PHC-EE (high)" in self.name:
                self.usr_opts["health_costs"] = "Uniform EE-high"
        self.eff_fs_splt = {a_s: {} for a_s in handyvars.adopt_schemes}
        self.sector_shapes = None
        # Deep copy handy vars to avoid any dependence of changes to these vars
        # across other measures that use them
        # self.handyvars = copy.deepcopy(handyvars)
        self.handyvars = handyvars
        self.sf_to_house = {}
        self.tsv_hourly_price = copy.deepcopy(handyvars.tsv_hourly_price)
        self.tsv_hourly_emissions = copy.deepcopy(handyvars.tsv_hourly_emissions)
        # Set the rate of baseline retrofitting for ECM stock-and-flow calcs
        try:
            # Check first to see whether pulling up retrofit rate errors
            self.retro_rate
            # If retrofit rate is set to None, use global retrofit rate value
            if self.retro_rate is None:
                self.retro_rate = self.handyvars.retro_rate
            # Extend retrofit rate point value across all years
            elif type(self.retro_rate) is float:
                self.retro_rate = {
                    yr: self.retro_rate for yr in self.handyvars.aeo_years}
            # Accommodate retrofit rate input as a probability distribution
            elif type(self.retro_rate) is list and isinstance(
                    self.retro_rate[0], str):
                # Sample measure retrofit rate values
                self.retro_rate = {
                    yr: self.rand_list_gen(
                        self.retro_rate, self.handyvars.nsamples) for yr in
                    self.handyvars.aeo_years}
            # Raise error in case where input is incorrectly specified
            else:
                raise ValueError(
                    "ECM " + self.name + " 'retro_rate' must be float or "
                    "if specified as a distribution must " +
                    "be formatted as [<distribution name> (string), " +
                    "<distribution parameters> (floats)]")
        except AttributeError:
            # If no 'retro_rate' attribute was given for the ECM, use default
            # retrofit rate value
            self.retro_rate = self.handyvars.retro_rate
        # Check for technology switching attribute, and if not there set None
        try:
            self.tech_switch_to
        except AttributeError:
            self.tech_switch_to = None
        # Determine whether the measure replaces technologies pertaining to
        # the supply or the demand of energy services
        self.technology_type = None
        # Measures replacing technologies in a pre-specified
        # 'demand_tech' list are of the 'demand' side technology type
        if (isinstance(self.technology, list) and all([
            x in self.handyvars.demand_tech for x in self.technology])) or \
                self.technology in self.handyvars.demand_tech:
            self.technology_type = "demand"
        # Measures replacing technologies not in a pre-specified
        # 'demand_tech' list are of the 'supply' side technology type
        else:
            self.technology_type = "supply"
        # Reset market entry year if None or earlier than min. year
        if self.market_entry_year is None or (int(
                self.market_entry_year) < int(self.handyvars.aeo_years[0])):
            self.market_entry_year = int(self.handyvars.aeo_years[0])
        # If a global year by which an elevated performance floor is
        # implemented has been imposed by the user and no measures with
        # typical/BAU efficiency are represented on the market, assume that
        # all measures begin showing market impacts in that year
        if self.usr_opts["floor_start"] is not None and \
            self.usr_opts["add_typ_eff"] is False and \
                self.market_entry_year < self.usr_opts["floor_start"]:
            self.market_entry_year = self.usr_opts["floor_start"]
        # Reset measure market exit year if None or later than max. year
        if self.market_exit_year is None or (int(
                self.market_exit_year) > (int(
                    self.handyvars.aeo_years[-1]) + 1)):
            self.market_exit_year = int(self.handyvars.aeo_years[-1]) + 1
        # If a global year by which an elevated performance floor is
        # implemented has been imposed by the user and the measure represents
        # a typical/BAU efficiency level, remove the measure from the market
        # once the elevated floor goes into effect
        if self.usr_opts["floor_start"] is not None and (
                self.usr_opts["add_typ_eff"] is not False and
                any([x in self.name for x in ["Ref. Case", "Min. Eff."]])):
            self.market_exit_year = self.usr_opts["floor_start"]
        self.yrs_on_mkt = [str(i) for i in range(
            self.market_entry_year, self.market_exit_year)]
        # Test for whether a user has set time sensitive valuation features
        # for the given measure. If no "tsv_features" parameter was
        # specified for the ECM, set this parameter to None
        try:
            # Try to access the ECM's TSV feature dict keys
            self.tsv_features.keys()
            # If TSV features are present, ensure that EMM regions are selected
            # and that the measure only applies to electricity (and no fuel
            # switching is selected)
            if self.usr_opts["alt_regions"] != "EMM":
                raise ValueError(
                    "Measure '" + self.name + "' has time sensitive "
                    "assessment features (see 'tsv_features' attribute) but "
                    "regions are not set to EMM; try running 'ecm_prep.py' "
                    "again with the --alt_regions option included and select "
                    "EMM regions when prompted.")
            if (self.fuel_type not in ["electricity", ["electricity"]]) and \
                    self.fuel_switch_to != "electricity":
                raise ValueError(
                    "Non-electric fuel found for measure '" + self.name +
                    " alongside time sensitive valuation features. Such "
                    "features cannot be implemented for non-electric "
                    "baseline segments of energy use. To address this issue, "
                    "restrict the measure's fuel type to electricity.")
            # If the ECM is assigned a custom savings shape, load the
            # associated custom savings shape data from a CSV file
            if "shape" in self.tsv_features.keys() and \
                "custom_annual_savings" in \
                    self.tsv_features["shape"].keys():
                # Retrieve custom savings shapes for all applicable
                # end use, building type, and climate zone combinations
                # and store within a dict for use in 'apply_tsv' function

                print("Retrieving custom savings shape data for measure "
                      + self.name + "...", end="", flush=True)
                # Determine the CSV file name
                csv_shape_file_name = \
                    self.tsv_features["shape"]["custom_annual_savings"]
                # Assuming the standard location for ECM savings shape CSV
                # files, import custom savings shape data as numpy array and
                # store it in the ECM's custom savings shape attribute for
                # subsequent use in the 'apply_tsv' function
                try:
                    self.tsv_features["shape"]["custom_annual_savings"] = \
                        numpy.genfromtxt(
                            handyfiles.tsv_shape_data / csv_shape_file_name,
                            names=True, delimiter=',', dtype=[
                                ('Hour_of_Year', '<i4'),
                                ('Climate_Zone', '<U25'),
                                ('Net_Load_Version', '<i4'),
                                ('Building_Type', '<U25'),
                                ('End_Use', '<U25'),
                                ('Baseline_Load', '<f8'),
                                ('Measure_Load', '<f8'),
                                ('Relative_Savings', '<f8')],
                            encoding="latin1")
                except OSError:
                    raise OSError(
                        "Savings shape data file indicated in 'tsv_features' "
                        "attribute of measure '" + self.name + "' not found; "
                        "looking for file " +
                        str(handyfiles.tsv_shape_data / csv_shape_file_name) + ". "
                        "Find the latest measure savings shape data here: "
                        "https://doi.org/10.5281/zenodo.4602369, files "
                        "'Latest_Com_Shapes.zip' and 'Latest_Res_Shapes.zip'")

                # Set shorthand for custom savings shape data
                css_dat = self.tsv_features["shape"][
                    "custom_annual_savings"]
                # Initialize dict to use in storing shape data
                css_dict = {}
                # Find all unique end uses in the shape data
                euses = numpy.unique(css_dat["End_Use"])
                # Loop through all end uses in the data
                for eu in euses:
                    # Handle case where end use names in the data are
                    # read in with added quotes (e.g., 'heating' comes in
                    # as '"heating"'), or are not strings. In the first
                    # instance, use eval() to strip the added quotes from the
                    # end use name and key in the savings shape information
                    # by the result
                    try:
                        eu_key = eval(eu)
                    except (NameError, SyntaxError):
                        eu_key = eu
                    if type(eu_key) != str:
                        eu_key = str(eu_key)
                    # Restrict shape data to that of the current end use
                    css_dat_eu = css_dat[
                        numpy.in1d(css_dat["End_Use"], eu)]
                    # Translate "drying" key to "clothes drying" to be
                    # consistent with what's in tsv_load
                    if eu_key == "drying":
                        eu_key = "clothes drying"
                    # Initialize dict under the current end use key
                    css_dict[eu_key] = {}
                    # Find all unique building types and climate zones in
                    # the end-use-restricted shape data
                    bldg_types = numpy.unique(
                        css_dat_eu["Building_Type"])
                    czones = numpy.unique(
                        css_dat_eu["Climate_Zone"])
                    # Loop through all building types under the current
                    # end use
                    for bd in bldg_types:
                        # Handle case where building type names in the data
                        # are read in with added quotes, or are not strings
                        try:
                            bd_key = eval(bd)
                        except (NameError, SyntaxError):
                            bd_key = bd
                        if type(bd_key) != str:
                            bd_key = str(bd_key)
                        # Account for possible use of "StandAlone" naming in
                        # savings shape CSV, vs. expected "Standalone"
                        if bd_key == "RetailStandAlone":
                            bd_key = "RetailStandalone"
                        # Account for possible use of "MediumOffice" naming
                        # in savings shape CSV, vs. expected
                        # "MediumOfficeDetailed"
                        elif bd_key == "MediumOffice":
                            bd_key = "MediumOfficeDetailed"
                        # Account for possible use of "LargeOffice" naming
                        # in savings shape CSV, vs. expected
                        # "LargeOfficeDetailed""
                        elif bd_key == "LargeOffice":
                            bd_key = "LargeOfficeDetailed"
                        # Initialize dict under the current end use and
                        # building type keys
                        css_dict[eu_key][bd_key] = {}
                        # Loop through all climate zones under the current
                        # end use
                        for cz in czones:
                            # Handle case where climate zone names in the
                            # data are read in with added quotes, or are not
                            # strings
                            try:
                                cz_key = eval(cz)
                            except (NameError, SyntaxError):
                                cz_key = cz
                            if type(cz_key) != str:
                                cz_key = str(cz_key)
                            # Account for possible use of climate 7A naming
                            # in savings shape CSV, vs. 7 naming in Scout's
                            # baseline load shapes file
                            if cz_key == "7A":
                                cz_key = "7"
                            # Restrict shape data to that of the current
                            # end use, building type, and climate zone
                            # combination
                            css_dat_eu_bldg_cz = css_dat_eu[
                                numpy.in1d(css_dat_eu["Building_Type"], bd) &
                                numpy.in1d(css_dat_eu["Climate_Zone"], cz)]
                            # Initialize dict under the current end use and
                            # building type keys
                            css_dict[eu_key][bd_key][cz_key] = {}
                            # Find all unique representative system load
                            # shapes for the current climate zone
                            sys_v = numpy.unique(
                                css_dat_eu_bldg_cz["Net_Load_Version"])
                            # If "Net_Load_Version" column is blank, set unique
                            # net load versions to 1
                            if len(sys_v) == 0 or (
                                    len(sys_v) == 1 and sys_v[0] == -1):
                                sys_v = [1]
                            for sv in sys_v:
                                # Restrict data further to current net load
                                # shape version
                                css_dat_eu_bldg_cz_nlv = \
                                    css_dat_eu_bldg_cz[numpy.in1d(
                                            css_dat_eu_bldg_cz[
                                                "Net_Load_Version"], sv)]
                                # Set measure and baseline load 8760s
                                eff_l, base_l = [
                                    css_dat_eu_bldg_cz_nlv["Measure_Load"],
                                    css_dat_eu_bldg_cz_nlv["Baseline_Load"]]
                                # Check to ensure that the resultant load
                                # data are expected 8760 elements long; if
                                # not, throw error
                                if not all([len(x) == 8760 for x in [
                                        eff_l, base_l]]):
                                    raise ValueError(
                                        "Measure '" + self.name +
                                        "', requires "
                                        "custom savings shape data, but the "
                                        "custom shape given for climate "
                                        "zone " + cz_key +
                                        ", building type "
                                        + bd_key + ", and end use " + eu_key +
                                        " has more or less than 8760 values. "
                                        "Check that 8760 hourly savings " +
                                        "fractions are available for all " +
                                        "baseline market segments the " +
                                        "measure applies to in "
                                        f"{fp.ECM_DEF / 'energy_plus_data' / 'savings_shapes'}.")
                                # Calculate baseline hourly load fractions of
                                # annual load in CSV (to be used later to scale
                                # hourly fractions of annual load in tsv_load
                                # to ensure consistency with baseline from CSV)
                                # ; also calculate relative hourly load
                                # scaling fractions vs. baseline (efficient
                                # over baseline hourly loads, to be applied
                                # later to baseline hourly load fractions of
                                # annual load to derive the same for the
                                # efficient case)
                                else:
                                    # Calculate baseline hourly load fractions
                                    # of annual load

                                    # Annual sum across all hourly loads
                                    ann_load = sum(base_l)
                                    # Find hourly fractions of annual load
                                    if ann_load != 0:
                                        base_l_frac = base_l / ann_load
                                        # Ensure no NaNs in result
                                        base_l_frac[
                                            numpy.isnan(base_l_frac)] = 0
                                    else:
                                        base_l_frac = numpy.zeros_like(base_l)

                                    # Calculate relative hourly load fractions
                                    # vs. baseline

                                    # Divide efficient by baseline hourly
                                    # loads to derive hourly change in load
                                    rel_chg = numpy.divide(
                                        eff_l, base_l,
                                        out=numpy.ones_like(base_l),
                                        where=base_l != 0)
                                    # Ensure no NaNs in result
                                    rel_chg[numpy.isnan(rel_chg)] = 1

                                    # Record above values in dict for later use

                                    # Net load version key to use in dict
                                    v_key = "set " + str(sv)
                                    # Store CSV hourly baseline fractions and
                                    # relative change values for later use
                                    css_dict[eu_key][bd_key][cz_key][v_key] = {
                                        "CSV base frac. annual": base_l_frac,
                                        "CSV relative change": rel_chg}

                # Set custom savings shape information to populated dict
                self.tsv_features["shape"]["custom_annual_savings"] = \
                    css_dict
                print("Data import complete")
        except AttributeError:
            self.tsv_features = None
        # Check to ensure that the proper EMM regions are defined in the
        # measure 'climate_zone' attribute if time sensitive ECM features
        # and/or output metrics are desired
        valid_tsv_regions = [
            'TRE', 'FRCC', 'MISW', 'MISC', 'MISE', 'MISS',
            'ISNE', 'NYCW', 'NYUP', 'PJME', 'PJMW', 'PJMC',
            'PJMD', 'SRCA', 'SRSE', 'SRCE', 'SPPS', 'SPPC',
            'SPPN', 'SRSG', 'CANO', 'CASO', 'NWPP', 'RMRG', 'BASN']
        if ((self.tsv_features is not None or
             self.usr_opts["tsv_metrics"] is not False) and ((
                type(self.climate_zone) == list and any([
                    x not in valid_tsv_regions for x in self.climate_zone])) or
            (type(self.climate_zone) != list and self.climate_zone != "all"
                and (self.climate_zone not in valid_tsv_regions)))):
            raise ValueError(
                "Invalid 'climate_zone' attribute value(s) for ECM '" +
                self.name + "' given desired time sensitive valuation "
                "operations/outputs. Currently, only EMM regions are "
                "supported for time sensitive valuation. This issue can "
                "be addressed by ensuring all ECM 'climate_zone' values "
                "reflect one of the EMM regions.")
        self.markets = {}

        for adopt_scheme in handyvars.adopt_schemes:
            self.markets[adopt_scheme] = OrderedDict([(
                "master_mseg", OrderedDict([(
                    "stock", {
                        "total": {
                            "all": None, "measure": None},
                        "competed": {
                            "all": None, "measure": None}}),
                    (
                    "energy", {
                        "total": {
                            "baseline": None, "efficient": None},
                        "competed": {
                            "baseline": None, "efficient": None}}),
                    (
                    "carbon", {
                        "total": {
                            "baseline": None, "efficient": None},
                        "competed": {
                            "baseline": None, "efficient": None}}),
                    (
                    "cost", {
                        "stock": {
                            "total": {
                                "baseline": None, "efficient": None},
                            "competed": {
                                "baseline": None, "efficient": None}},
                        "energy": {
                            "total": {
                                "baseline": None, "efficient": None},
                            "competed": {
                                "baseline": None, "efficient": None}},
                        "carbon": {
                            "total": {
                                "baseline": None, "efficient": None},
                            "competed": {
                                "baseline": None, "efficient": None}}}),
                    (
                    "lifetime", {"baseline": None, "measure": None})])),
                (
                "mseg_adjust", {
                    "contributing mseg keys and values": {},
                    "competed choice parameters": {},
                    "secondary mseg adjustments": {
                        "sub-market": {
                            "original energy (total)": {},
                            "adjusted energy (sub-market)": {}},
                        "stock-and-flow": {
                            "original energy (total)": {},
                            "adjusted energy (previously captured)": {},
                            "adjusted energy (competed)": {},
                            "adjusted energy (competed and captured)": {}},
                        "market share": {
                            "original energy (total captured)": {},
                            "original energy (competed and captured)": {},
                            "adjusted energy (total captured)": {},
                            "adjusted energy (competed and captured)": {}}}}),
                (
                "mseg_out_break", {key: {
                    "baseline": copy.deepcopy(self.handyvars.out_break_in),
                    "efficient": copy.deepcopy(self.handyvars.out_break_in),
                    "savings": copy.deepcopy(self.handyvars.out_break_in)} for
                    key in ["energy", "carbon", "cost"]})])
            # Add fugitive emissions key to output dict if fugitive
            # emissions option is set
            if self.usr_opts["fugitive_emissions"] is not False:
                # Initialize methane/refrigerant values as zero when that
                # fugitive option is being assessed (in which case new data
                # will be summed into the zero dicts), and as None otherwise

                # Check for methane assessment/initialization of zero dict
                if self.usr_opts["fugitive_emissions"][0] in ['1', '3']:
                    init_meth = {yr: 0 for yr in self.handyvars.aeo_years}
                else:
                    init_meth = None
                # Check for refrigerants assessment/initialization of zero dict
                if self.usr_opts["fugitive_emissions"][0] in ['2', '3']:
                    init_refr = {yr: 0 for yr in self.handyvars.aeo_years}
                else:
                    init_refr = None
                # Organize methane and refrigerants dict under broader key
                self.markets[adopt_scheme]["master_mseg"][
                    "fugitive emissions"] = {
                        "methane": {
                            "total": {
                                "baseline": copy.deepcopy(init_meth),
                                "efficient": copy.deepcopy(init_meth)},
                            "competed": {
                                "baseline": copy.deepcopy(init_meth),
                                "efficient": copy.deepcopy(init_meth)}},
                        "refrigerants": {
                            "total": {
                                "baseline": copy.deepcopy(init_refr),
                                "efficient": copy.deepcopy(init_refr)},
                            "competed": {
                                "baseline": copy.deepcopy(init_refr),
                                "efficient": copy.deepcopy(init_refr)}}}

    def fill_eplus(self, msegs, eplus_dir, eplus_coltypes,
                   eplus_files, vintage_weights, base_cols):
        """Fill in measure performance with EnergyPlus simulation results.

        Note:
            Find the appropriate set of EnergyPlus simulation results for
            the current measure, and use the relative savings percentages
            in these results to determine the measure performance attribute.

        Args:
            msegs (dict): Baseline microsegment stock/energy data to use in
                validating categorization of measure performance information.
            eplus_dir (string): Directory of EnergyPlus performance files.
            eplus_coltypes (list): Expected EnergyPlus variable data types.
            eplus_files (list): EnergyPlus performance file names.
            vintage_weights (dict): Square-footage-derived weighting factors
                for each EnergyPlus building vintage type.

        Returns:
            Updated Measure energy_efficiency, energy_efficiency_source, and
            energy_efficiency_source attribute values.

        Raises:
            ValueError: If EnergyPlus file is not matched to Measure
                definition or more than one EnergyPlus file matches the
                Measure definition.
        """
        # Instantiate useful EnergyPlus-Scout mapping dicts
        handydicts = EPlusMapDicts()
        # Determine the relevant EnergyPlus building type name(s)
        bldg_type_names = []
        for x in self.bldg_type:
            bldg_type_names.extend(handydicts.bldgtype[x].keys())
        # Find all EnergyPlus files including the relevant building type
        # name(s)
        eplus_perf_in = [(eplus_dir + '/' + x) for x in eplus_files if any([
            y.lower() in x for y in bldg_type_names])]

        # Import EnergyPlus input file as array and use it to fill a dict
        # of measure performance data
        if len(eplus_perf_in) > 0:
            # Assemble the EnergyPlus data into a record array
            eplus_perf_array = self.build_array(eplus_coltypes, eplus_perf_in)
            # Create a measure performance dictionary, zeroed out, to
            # be updated with data from EnergyPlus array
            perf_dict_empty = self.create_perf_dict(msegs)

            # Update measure performance based on EnergyPlus data
            # (* Note: only update efficiency information for
            # secondary microsegments if applicable)
            if perf_dict_empty['secondary'] is not None:
                self.energy_efficiency = self.fill_perf_dict(
                    perf_dict_empty, eplus_perf_array,
                    vintage_weights, base_cols, eplus_bldg_types={})
            else:
                self.energy_efficiency = self.fill_perf_dict(
                    perf_dict_empty['primary'], eplus_perf_array,
                    vintage_weights, base_cols, eplus_bldg_types={})
            # Set the energy efficiency data source for the measure to
            # EnergyPlus and set to highest data quality rating
            self.energy_efficiency_source = 'EnergyPlus/OpenStudio'
        else:
            raise ValueError(
                "Failure to find relevant EPlus files for " +
                "Scout building type(s) " + str(self.bldg_type) +
                "in ECM '" + self.name + "'")

    def fill_mkts(self, msegs, msegs_cpl, convert_data, tsv_data_init, opts,
                  ctrb_ms_pkg_prep, tsv_data_nonfs):
        """Fill in a measure's market microsegments using EIA baseline data.

        Args:
            msegs (dict): Baseline microsegment stock and energy use.
            msegs_cpl (dict): Baseline technology cost, performance, and
                lifetime.
            convert_data (dict): Measure -> baseline cost unit conversions.
            tsv_data_init (dict): Data for time sensitive valuation of
                efficiency.
            opts (object): Stores user-specified execution options.
            ctrb_ms_pkg_prep (list): List of measure names that contribute
                to active packages in the preparation run.
            tsv_data_nonfs (dict): If applicable, base-case TSV data to apply
                to non-fuel switching measures under a high decarb. scenario.

        Returns:
            Updated measure stock, energy/carbon, and cost market microsegment
            information, as stored in the 'markets' attribute.

        Raises:
            KeyError: If measure and baseline performance or cost units are
                inconsistent, or no valid baseline market microsegments can
                be found for the given measure definition.
            ValueError: If an input value from the measure definition is
                invalid, or if baseline market microsegment information cannot
                be mapped to a valid breakout category for measure outputs.
        """
        # Check that the measure's applicable baseline market input definitions
        # are valid before attempting to retrieve data on this baseline market
        self.check_mkt_inputs()

        # Notify user that ECM is being updated; suppress new line
        # if not in verbose mode ('Success' is appended to this message on
        # the same line of the console upon completion of ECM update)
        if opts is not None and opts.verbose is True:
            print("Updating ECM '" + self.name + "'...")
        else:
            print("Updating ECM '" + self.name + "'...", end="", flush=True)

        # If multiple runs are required to handle probability distributions on
        # measure inputs, set a number to seed each random draw of cost,
        # performance, and or lifetime with for consistency across all
        # microsegments that contribute to the measure's master microsegment
        if self.handyvars.nsamples is not None:
            rnd_sd = numpy.random.randint(10000)

        # Initialize a counter of key chains that yield "stock" and "energy"
        # keys in the baseline data dict; that have valid stock/energy data;
        # that have valid cost/performance/lifetime data; and that have valid
        # consumer choice data. Also initialize a cost conversion flag
        valid_keys, valid_keys_stk_energy, valid_keys_cpl, \
            valid_keys_consume, cost_converts = (0 for n in range(5))
        # Initialize lists of technology names that have yielded warnings
        # for invalid stock/energy data, cost/performance/lifetime data
        # and consumer data, EIA baseline cost adjustments (in the case
        # of heat pump HVAC) and a list of the climate zones, building types,
        # and structure type of removed primary microsegments (used to remove
        # associated secondary microsegments)
        stk_energy_warn, cpl_warn, consume_warn, hp_warn, removed_primary = (
            [] for n in range(5))

        # Initialize flags for invalid information about sub-market fraction
        # source, URL, and derivation
        sbmkt_source_invalid, sbmkt_url_invalid, sbmkt_derive_invalid = (
            0 for n in range(3))

        # Establish a flag for a commercial lighting case where the user has
        # not specified secondary end use effects on heating and cooling.  In
        # this case, secondary effects are added automatically by adjusting
        # the "lighting gain" thermal load component in accordance with the
        # lighting efficiency change (e.g., a 40% relative savings from
        # efficient lighting equipment translates to a 40% increase in heating
        # loads and 40% decrease in cooling load)
        light_scnd_autoperf = False

        # Fill out any "secondary" end use impact information and any climate
        # zone, building type, fuel type, end use, and/or technology attributes
        # marked 'all' by users
        self.fill_attr()

        # Flag the auto-generation of reference case technology analogues for
        # all of the current measure's applicable markets, if applicable –
        # exclude 'Ref. Case' switching of fossil-based heat or resistance heat
        # to HPs under exogenous switching rates, since the competing Ref. Case
        # analogue will be a min. efficiency HP and this is manually defined
        if (opts.add_typ_eff is True and "Ref. Case" in self.name):
            agen_ref = True
        else:
            agen_ref = ""

        # Flag for whether or not the measure requires calculation of sector-
        # level electricity savings shapes. Exclude envelope measures that are
        # part of packages w/HVAC equipment; these measures simply scale HVAC
        # equipment shapes based on annual results, and thus do not require
        # shapes of their own. Also exclude any measures that track the
        # reference case; these measures have zero effects on baseline loads
        # and will therefore not generate meaningful sector-level savings
        # shapes. Finally, exclude any measures that do not apply to electric
        # baselines or else fuel switch to electricity (shapes inapplicable)
        if opts.sect_shapes is True:
            calc_sect_shapes = (
                (self.technology_type["primary"] != "demand" or
                 self.name not in ctrb_ms_pkg_prep) and (not agen_ref) and
                ("electricity" in self.fuel_type["primary"] or
                 self.fuel_switch_to == "electricity") and (all([
                    x not in self.name for x in [
                        "Gas", "Oil", "N-Elec", "Fossil"]])))
        else:
            calc_sect_shapes = False

        # Fill in sector baseline/efficient 8760 shapes attribute across all
        # applicable regions for the measure with a list of 8760 zeros (if
        # necessary).
        if calc_sect_shapes is True:
            # Initialize sector shape dicts by adoption scheme
            self.sector_shapes = {
                a_s: {} for a_s in self.handyvars.adopt_schemes}
            # Find applicable region list (ensure it is in list format)s
            if type(self.climate_zone) is str:
                grid_regions = copy.deepcopy([self.climate_zone])
            else:
                grid_regions = copy.deepcopy(self.climate_zone)
            for a_s in self.handyvars.adopt_schemes:
                self.sector_shapes[a_s] = {reg: {yr: {
                    "baseline": [0 for x in range(8760)],
                    "efficient": [0 for x in range(8760)]} for yr in
                    self.handyvars.aeo_years_summary} for reg in grid_regions}

        # Find all possible microsegment key chains.  First, determine all
        # "primary" microsegment key chains, where "primary" refers to the
        # baseline microsegment(s) directly affected by a measure (e.g.,
        # incandescent bulb lights for an LED replacement measure).  Second,
        # if needed, determine all "secondary" microsegment key chains, where
        # "secondary" refers to baseline microsegments that are indirectly
        # affected by the measure (e.g., heating and cooling for the above
        # LED replacement).  Secondary microsegments are only relevant for
        # energy/carbon and associated energy/carbon cost calculations, as
        # they do not indicate additional equipment purchases (and thus do not
        # affect stock, stock costs, or equipment lifetime calculations)

        # Determine "primary" microsegment key chains
        ms_iterable, ms_lists = self.create_keychain("primary")

        # Insert a flag for linking heating and cooling microsegments, if
        # applicable; when linked, heat pump conversion rates for heating
        # microsegments will also be applied to the cooling microsegments
        # affected by the measure in a given building type/region
        if self.handyvars.hp_rates and all([
                x in ms_lists[3] for x in ["heating", "cooling"]]):
            link_htcl_fs_rates = True
        else:
            link_htcl_fs_rates = ""

        # If needed, fill out any secondary microsegment fuel type, end use,
        # and/or technology input attributes marked 'all' by users. Determine
        # secondary microsegment key chains and add to the primary
        # microsegment key chain list. In a commercial lighting measure case
        # where no heating/cooling effects from lighting are directly
        # specified, use the "lighting gain" thermal load component
        # microsegments to represent secondary effects of the lighting measure
        if self.end_use["secondary"] is not None:
            ms_iterable_second, ms_lists_second = self.create_keychain(
                "secondary")
            ms_iterable.extend(ms_iterable_second)
        elif "lighting" in self.end_use["primary"] and (
            not opts or opts.no_scnd_lgt is not True) and any([
                x not in self.end_use["primary"] for x in [
                "heating", "cooling"]]) and \
            any([x not in ["single family home", "multi family home",
                           "mobile home"] for x in self.bldg_type]):
            # Set secondary lighting mseg performance flag to True
            light_scnd_autoperf = True
            # Set secondary energy efficiency value to "Missing"
            # (used below as a flag)
            self.energy_efficiency["secondary"] = \
                "Missing (secondary lighting)"
            # Set secondary energy efficiency units to "relative
            # savings"
            self.energy_efficiency_units["secondary"] = \
                "relative savings (constant)"
            # Set secondary fuel type to include all heating/cooling
            # fuels
            self.fuel_type["secondary"] = [
                "electricity", "natural gas", "distillate"]
            # Set relevant secondary end uses
            self.end_use["secondary"] = ["heating", "cooling"]
            # Set secondary technology type ("demand" as the lighting
            # measure affects heating/cooling loads)
            self.technology_type["secondary"] = "demand"
            # Set secondary technology class to "lighting gain", which
            # will access the portion of commercial heating/cooling
            # demand that is attributable to waste heat from lights
            self.technology["secondary"] = "lighting gain"

            # Determine secondary microsegment key chains and add to
            # the primary microsegment key chain list
            ms_iterable_second, ms_lists_second = self.create_keychain(
                "secondary")
            ms_iterable.extend(ms_iterable_second)

        # Set flag for use of ft^2 floor area as microsegment stock. This is
        # used for envelope microsegments ("demand" technology type) or for
        # certain commercial equipment end uses for which service demand data
        # (which are used as "stock") do not exist. Note that unspecified
        # commercial building type has no square footage data
        if "demand" in self.technology_type["primary"] or (all(
            [x not in [
                "single family home", "mobile home", "multi family home"] for
                x in self.bldg_type]) and all([
                x in self.handyvars.com_eqp_eus_nostk for x in
                self.end_use["primary"]])):
            sqft_subst_flag = True
        else:
            sqft_subst_flag = ""

        # Initialize lists of warnings and list of suppressed incentives data
        warn_list, suppress_incent = ([] for n in range(2))

        # Initialize flag for whether loop through previous microsegment
        # has modified original measure costs/units when pulling incentives
        # information and these need to be reset for future microsegment
        # measure cost updates
        meas_incent_flag = ""

        # Loop through discovered key chains to find needed performance/cost
        # and stock/energy information for measure
        for ind, mskeys in enumerate(ms_iterable):
            # Move to next key chain for 'unspecified' building type and 'new'
            # vintage; there are no new/existing floor space data for this
            # building type and all data are pulled into 'existing' key chains.
            if mskeys[2] == "unspecified" and "new" in mskeys:
                continue
            # Set indicator for use of ft^2 floor area as stock for the
            # currently looped through microsegment; suppress use of square
            # footage as stock for 'unspecified' commercial building type,
            # which lacks square footage data
            if sqft_subst_flag and "unspecified" not in mskeys:
                sqft_subst = 1
            else:
                sqft_subst = 0

            # Set building sector for the current microsegment
            if mskeys[2] in [
                    "single family home", "mobile home", "multi family home"]:
                bldg_sect = "residential"
            else:
                bldg_sect = "commercial"

            # Develop "switched to" microsegment information for measures
            # that change to a different technology from the baseline;
            # screen out ventilation segments, which are sometimes attached
            # to commercial measures that fuel switch or switch from resistance
            # heating to HPs but are not assumed to switch ventilation tech.
            if self.tech_switch_to not in [None, "NA"] and (
                    "secondary" not in mskeys and mskeys[4] != "ventilation"):
                # Handle tech. switch from same fuel (e.g., resistance to HP);
                # in this case, fuel remains same as baseline mseg info.
                if self.fuel_switch_to is None:
                    mskeys_swtch_fuel = mskeys[3]
                else:
                    mskeys_swtch_fuel = self.fuel_switch_to
                # Handle switching of secondary heating to ASHP heating
                # (switched from info. has secondary heating as end use, but
                # switched to info. needs heating end use to pull ASHP data)
                if mskeys[4] == "secondary heating":
                    mskeys_swtch_eu = "heating"
                else:
                    mskeys_swtch_eu = mskeys[4]

                # Handle potentially erroneous/inconsistent user tech
                # switching inputs to the degree possible
                if bldg_sect == "residential":
                    # HVAC
                    if mskeys[4] in ["heating", "cooling"]:
                        if "ASHP" in self.tech_switch_to:
                            mskeys_swtch_tech = "ASHP"
                        elif "GSHP" in self.tech_switch_to:
                            mskeys_swtch_tech = "GSHP"
                        else:
                            mskeys_swtch_tech = self.tech_switch_to
                    # Water heating
                    elif mskeys[4] == "water heating":
                        mskeys_swtch_tech = "electric WH"
                    # Cooking and drying
                    elif mskeys[4] in ["cooking", "drying"]:
                        mskeys_swtch_tech = None
                    # Lighting - switch to comparable LED product based
                    # on lighting class in baseline
                    elif mskeys[4] == "lighting":
                        if "general" in mskeys[-2]:
                            mskeys_swtch_tech = "general service (LED)"
                        elif "reflector" in mskeys[-2]:
                            mskeys_swtch_tech = "reflector (LED)"
                        elif "linear" in mskeys[-2]:
                            mskeys_swtch_tech = "linear fluorescent (LED)"
                        else:
                            mskeys_swtch_tech = "external (LED)"
                    else:
                        mskeys_swtch_tech = self.tech_switch_to
                elif bldg_sect == "commercial":
                    # HVAC
                    if mskeys[4] in ["heating", "cooling"]:
                        if "ASHP" in self.tech_switch_to and \
                                mskeys[4] == "heating":
                            mskeys_swtch_tech = "rooftop_ASHP-heat"
                        elif "ASHP" in self.tech_switch_to and \
                                mskeys[4] == "cooling":
                            mskeys_swtch_tech = "rooftop_ASHP-cool"
                        elif "GSHP" in self.tech_switch_to and \
                                mskeys[4] == "heating":
                            mskeys_swtch_tech = "comm_GSHP-heat"
                        elif "GSHP" in self.tech_switch_to and \
                                mskeys[4] == "cooling":
                            mskeys_swtch_tech = "comm_GSHP-cool"
                        else:
                            mskeys_swtch_tech = self.tech_switch_to
                    # Water heating
                    elif mskeys[4] == "water heating":
                        mskeys_swtch_tech = "HP water heater"
                    # Cooking
                    elif mskeys[4] == "cooking":
                        mskeys_swtch_tech = \
                            "electric_range_oven_24x24_griddle"
                    # Lighting - switch to comparable LED product based
                    # on lighting class in baseline
                    elif mskeys[4] == "lighting":
                        if "100W" in mskeys[-2]:
                            mskeys_swtch_tech = "100W Equivalent LED A Lamp"
                        elif "PAR38" in mskeys[-2]:
                            mskeys_swtch_tech = "LED PAR38"
                        else:
                            mskeys_swtch_tech = "LED Integrated Luminaire"
                    else:
                        mskeys_swtch_tech = self.tech_switch_to
                else:
                    mskeys_swtch_tech = self.tech_switch_to
                # Produce "switched to" microsegment information
                if "supply" in mskeys:
                    # primary/secondary, region, bldg, fuel, end use, tech
                    # type (supply/demand), tech, new/existing
                    mskeys_swtch = [
                        mskeys[0], mskeys[1], mskeys[2], mskeys_swtch_fuel,
                        mskeys_swtch_eu, mskeys[5], mskeys_swtch_tech,
                        mskeys[7]]
                else:
                    # primary/secondary, region, bldg, fuel, end use, tech,
                    # new/existing
                    mskeys_swtch = [
                        mskeys[0], mskeys[1], mskeys[2], mskeys_swtch_fuel,
                        mskeys_swtch_eu, mskeys_swtch_tech, mskeys[6]]
            else:
                mskeys_swtch = ""

            # Check whether early retrofit rates are specified at the
            # component (microsegment) level; if so, restrict early retrofit
            # information to that of the current microsegment
            if bldg_sect in self.retro_rate.keys():
                # Lighting, water heating microsegments
                if mskeys[4] in self.retro_rate[bldg_sect].keys():
                    retro_rate_mseg = self.retro_rate[bldg_sect][mskeys[4]]
                # HVAC or envelope microsegments
                elif mskeys[4] in ["heating", "cooling"]:
                    # HVAC equipment microsegment ("supply" tech. type)
                    if "supply" in mskeys:
                        retro_rate_mseg = self.retro_rate[bldg_sect]["HVAC"]
                    # Envelope microsegment ("demand" tech. type)
                    else:
                        # All envelope tech. except windows
                        try:
                            retro_rate_mseg = self.retro_rate[
                                bldg_sect][mskeys[-2]]
                        # Windows require special handling b/c windows
                        # microsegment tech. is broken into conduction/solar
                        except KeyError:
                            retro_rate_mseg = self.retro_rate[
                                bldg_sect]["windows"]
                # All other microsegments – set early retrofit rate to zero
                else:
                    retro_rate_mseg = {
                        yr: 0 for yr in self.handyvars.aeo_years}
            # If early retrofit rates are not specified at the component
            # (microsegment) level, no further operations are needed
            else:
                retro_rate_mseg = self.retro_rate

            # Adjust the key chain to be used in registering contributing
            # microsegment information for cases where 'windows solar'
            # or 'windows conduction' are in the key chain. Change
            # such entries to just 'windows' to ensure the competition
            # of 'windows conduction' and 'windows solar' contributing
            # microsegments in the 'adjust_savings' function below
            contrib_mseg_key = mskeys
            if any([x is not None and "windows" in x for x in
                    contrib_mseg_key]):
                contrib_mseg_key = list(contrib_mseg_key)
                contrib_mseg_key[numpy.where([x is not None and "windows" in x
                                 for x in contrib_mseg_key])[0][0]] = "windows"
                contrib_mseg_key = tuple(contrib_mseg_key)

            # Initialize measure performance/cost/lifetime, associated units,
            # and sub-market scaling fractions/sources if: a) For loop through
            # all measure mseg key chains is in first iteration, b) A switch
            # has been made from updating "primary" microsegment info. to
            # updating "secondary" microsegment, and c) Any of performance/
            # cost/lifetime units is a dict which must be parsed further to
            # reach the final value. Additionally, for cost/cost units only,
            # initialize if a) Incentives information has been applied, b) $/sf
            # is used as cost units, c) Switching from one end use to another,
            # or from one building vintage to another * Note: cost/lifetime/
            # sub-market info. is not updated for "secondary" microsegments,
            # which do not pertain to these variables; lifetime units in years
            if ind == 0 or (ms_iterable[ind][0] != ms_iterable[ind - 1][0]) \
               or isinstance(self.energy_efficiency, dict):
                perf_meas = self.energy_efficiency
            if ind == 0 or (ms_iterable[ind][0] != ms_iterable[ind - 1][0]) \
               or isinstance(self.energy_efficiency_units, dict):
                perf_units = self.energy_efficiency_units
            if mskeys[0] == "secondary":
                cost_meas, life_meas = (0 for n in range(2))
                cost_units = "NA"
                # * Note: no unique sub-market scaling fractions for secondary
                # microsegments; secondary microsegments are only scaled down
                # by the sub-market fraction for their associated primary
                # microsegments
                mkt_scale_frac, mkt_scale_frac_source = (
                    None for n in range(2))
            else:
                # Case where incentives were added in previous mseg update;
                # reset costs/units to those of original measure
                if meas_incent_flag:
                    cost_units, cost_meas = [
                        self.cost_units, self.installed_cost]
                    # Reset flag for original measure cost reset due to its
                    # modification in incentives calculations for previous
                    # microsegments
                    meas_incent_flag = ""
                # Case where square footage cost units are used or there is
                # a switch from one end use, building type, or building vintage
                # to another, or original cost/cost units are in dict format;
                # reset cost/cost units to those of original measure
                elif ((sqft_subst == 1 or
                      "$/ft^2 floor" in self.cost_units) or ind == 0 or (
                        ms_iterable[ind][0] != ms_iterable[ind - 1][0]) or (
                        ms_iterable[ind][4] != ms_iterable[ind - 1][4]) or (
                        ms_iterable[ind][-1] != ms_iterable[ind - 1][-1]) or
                      any([isinstance(x, dict) for x in [
                        self.installed_cost, self.cost_units]])):
                    cost_units, cost_meas = [
                        self.cost_units, self.installed_cost]
                # Set lifetime attribute to initial value
                if ind == 0 or isinstance(
                        self.product_lifetime, dict):
                    life_meas = self.product_lifetime
                # Set market scaling attributes to initial values
                if ind == 0 or isinstance(
                        self.market_scaling_fractions, dict):
                    mkt_scale_frac = self.market_scaling_fractions
                if ind == 0 or isinstance(
                        self.market_scaling_fractions_source, dict):
                    mkt_scale_frac_source = \
                        self.market_scaling_fractions_source

            # Set the appropriate carbon intensity, and energy cost data dicts
            # to use for the current microsegment; when assuming a high grid
            # decarbonization case, the user can choose to assess emissions
            # and cost reductions in non-fuel switching microsegments using
            # base-case emissions intensities and energy costs (e.g., before
            # additional grid decarbonization)
            if opts.grid_decarb is not False and all([x is not None for x in [
                    self.handyvars.carb_int_nonfs,
                    self.handyvars.ecosts_nonfs]]) and (
                self.fuel_switch_to is None or (
                    self.fuel_switch_to == "electricity" and
                    "electricity" in mskeys)):
                carb_int_dat = self.handyvars.carb_int_nonfs
                cost_dat = self.handyvars.ecosts_nonfs
            else:
                carb_int_dat = self.handyvars.carb_int
                cost_dat = self.handyvars.ecosts

            # Set baseline and measure site-source conversion factors,
            # accounting for any fuel switching from baseline to measure tech.
            if self.fuel_switch_to is None:
                # Set site-source conversions to 1 if user flagged
                # site energy use outputs, to appropriate input data otherwise
                if opts is not None and opts.site_energy is True:
                    site_source_conv_base, site_source_conv_meas = [{
                        yr: 1 for yr in self.handyvars.aeo_years}
                        for n in range(2)]
                else:
                    site_source_conv_base, site_source_conv_meas = (
                        self.handyvars.ss_conv[mskeys[3]] for
                        n in range(2))
            else:
                # Set site-source conversions to 1 if user flagged
                # site energy use outputs, to appropriate input data otherwise
                if opts is not None and opts.site_energy is True:
                    site_source_conv_base, site_source_conv_meas = [{
                        yr: 1 for yr in self.handyvars.aeo_years}
                        for n in range(2)]
                else:
                    site_source_conv_base = self.handyvars.ss_conv[
                        mskeys[3]]
                    site_source_conv_meas = self.handyvars.ss_conv[
                        self.fuel_switch_to]

            # Set fuel type string for selection of baseline and measure
            # carbon intensities and fuel prices to handle special
            # technology cases
            if mskeys[6] == 'furnace (kerosene)':
                ftkey = 'distillate'
            else:
                ftkey = mskeys[3]

            # Set baseline and measure carbon intensities, accounting for any
            # fuel switching from baseline technology to measure technology
            if self.fuel_switch_to is None:
                # Case where use has flagged site energy outputs
                if opts is not None and opts.site_energy is True:
                    # Intensities are specified by EMM region or state based on
                    # site energy and require no further conversion to match
                    # the user's site energy setting
                    try:
                        intensity_carb_base, intensity_carb_meas = [{
                            yr: carb_int_dat[bldg_sect][ftkey][
                                mskeys[1]][yr] for
                            yr in self.handyvars.aeo_years} for n in range(2)]
                    # Intensities are specified nationally based on source
                    # energy and require multiplication by site-source factor
                    # to match the user's site energy setting
                    except KeyError:
                        intensity_carb_base, intensity_carb_meas = [{
                            yr: carb_int_dat[bldg_sect][
                                ftkey][yr] *
                            self.handyvars.ss_conv[ftkey][yr] for
                            yr in self.handyvars.aeo_years} for n in range(2)]
                # Case where user has not flagged site energy outputs
                else:
                    # Intensities are specified by EMM region or state based on
                    # site energy and require division by site-source factor to
                    # match the user's source energy setting
                    try:
                        intensity_carb_base, intensity_carb_meas = [{
                            yr: carb_int_dat[bldg_sect][ftkey][
                                mskeys[1]][yr] /
                            self.handyvars.ss_conv[ftkey][yr] for
                            yr in self.handyvars.aeo_years} for n in range(2)]
                    # Intensities are specified nationally based on source
                    # energy and require no further conversion to match the
                    # user's source energy setting
                    except KeyError:
                        intensity_carb_base, intensity_carb_meas = (
                            carb_int_dat[bldg_sect][
                                ftkey] for n in range(2))
            else:
                # Interpretation of the calculations below is the same as for
                # the case above without fuel switching; the only difference
                # here is that baseline vs. measure settings use different
                # fuels and must therefore be calculated separately

                # Case where use has flagged site energy outputs
                if opts is not None and opts.site_energy is True:
                    # Intensities broken out by EMM region or state
                    try:
                        # Base fuel intensity broken by region
                        intensity_carb_base = carb_int_dat[
                            bldg_sect][ftkey][mskeys[1]]
                    except KeyError:
                        # Base fuel intensity not broken by region
                        intensity_carb_base = {yr: carb_int_dat[
                            bldg_sect][ftkey][yr] *
                            self.handyvars.ss_conv[ftkey][yr]
                            for yr in self.handyvars.aeo_years}
                    try:
                        # Measure fuel intensity broken by region
                        intensity_carb_meas = carb_int_dat[
                            bldg_sect][self.fuel_switch_to][mskeys[1]]
                    except KeyError:
                        # Measure fuel intensity not broken by region
                        intensity_carb_meas = {yr: carb_int_dat[
                            bldg_sect][self.fuel_switch_to][yr] *
                            self.handyvars.ss_conv[self.fuel_switch_to][yr]
                            for yr in self.handyvars.aeo_years}
                # Case where user has not flagged site energy outputs
                else:
                    try:
                        # Base fuel intensity broken by region
                        intensity_carb_base = {yr: carb_int_dat[
                            bldg_sect][ftkey][mskeys[1]][yr] /
                            self.handyvars.ss_conv[ftkey][yr]
                            for yr in self.handyvars.aeo_years}
                    except KeyError:
                        # Base fuel intensity not broken by region
                        intensity_carb_base = carb_int_dat[
                            bldg_sect][ftkey]
                    try:
                        # Measure fuel intensity broken by region
                        intensity_carb_meas = {yr: carb_int_dat[
                            bldg_sect][self.fuel_switch_to][mskeys[1]][yr] /
                            self.handyvars.ss_conv[self.fuel_switch_to][yr]
                            for yr in self.handyvars.aeo_years}
                    except KeyError:
                        # Measure fuel intensity not broken by region
                        intensity_carb_meas = carb_int_dat[
                            bldg_sect][self.fuel_switch_to]

            # Set baseline and measure fuel costs, accounting for any
            # fuel switching from baseline technology to measure technology;
            # interpretation of the calculations is the same as for the carbon
            # intensity calculations above
            if self.fuel_switch_to is None:
                # Case where use has flagged site energy outputs
                if opts is not None and opts.site_energy is True:
                    # Costs broken out by EMM region or state
                    try:
                        cost_energy_base, cost_energy_meas = (
                            cost_dat[bldg_sect][ftkey][
                                mskeys[1]] for n in range(2))
                    # National fuel costs
                    except KeyError:
                        cost_energy_base, cost_energy_meas = [{
                            yr: cost_dat[bldg_sect][
                                ftkey][yr] * self.handyvars.ss_conv[
                                ftkey][yr] for yr in
                            self.handyvars.aeo_years} for n in range(2)]
                # Case where user has not flagged site energy outputs
                else:
                    # Costs broken out by EMM region or state
                    try:
                        cost_energy_base, cost_energy_meas = [{
                            yr: cost_dat[
                                bldg_sect][ftkey][mskeys[1]][yr] /
                            self.handyvars.ss_conv[ftkey][yr] for
                            yr in self.handyvars.aeo_years} for
                            n in range(2)]
                    # National fuel costs
                    except KeyError:
                        cost_energy_base, cost_energy_meas = (
                            cost_dat[bldg_sect][ftkey] for
                            n in range(2))
            else:
                # Case where use has flagged site energy outputs
                if opts is not None and opts.site_energy is True:
                    try:
                        # Base fuel cost broken out by region
                        cost_energy_base = cost_dat[bldg_sect][
                            ftkey][mskeys[1]]
                    except KeyError:
                        # Base fuel cost not broken out by region
                        cost_energy_base = {
                            yr: cost_dat[bldg_sect][
                                ftkey][yr] * self.handyvars.ss_conv[
                                ftkey][yr] for yr in
                            self.handyvars.aeo_years}
                    try:
                        # Measure fuel cost broken out by region
                        cost_energy_meas = cost_dat[bldg_sect][
                            self.fuel_switch_to][mskeys[1]]
                    except KeyError:
                        # Measure fuel cost not broken out by region
                        cost_energy_meas = {
                            yr: cost_dat[bldg_sect][
                                self.fuel_switch_to][yr] *
                            self.handyvars.ss_conv[self.fuel_switch_to][yr] for
                            yr in self.handyvars.aeo_years}
                # Case where user has not flagged site energy outputs
                else:
                    try:
                        # Base fuel cost broken out by region
                        cost_energy_base = {
                            yr: cost_dat[bldg_sect][ftkey][
                                mskeys[1]][yr] / self.handyvars.ss_conv[
                                ftkey][yr] for yr in
                            self.handyvars.aeo_years}
                    except KeyError:
                        # Base fuel cost not broken out by region
                        cost_energy_base = cost_dat[bldg_sect][
                            ftkey]
                    try:
                        # Measure fuel cost broken out by region
                        cost_energy_meas = {
                            yr: cost_dat[bldg_sect][
                                self.fuel_switch_to][mskeys[1]][yr] /
                            self.handyvars.ss_conv[self.fuel_switch_to][yr] for
                            yr in self.handyvars.aeo_years}
                    except KeyError:
                        # Measure fuel cost not broken out by region
                        cost_energy_meas = cost_dat[bldg_sect][
                            self.fuel_switch_to]

            # For the case where the baseline technology is a wood
            # stove, set the energy cost and carbon intensity to zero
            if mskeys[6] == 'stove (wood)':
                intensity_carb_base = dict.fromkeys(intensity_carb_base, 0)
                cost_energy_base = dict.fromkeys(cost_energy_base, 0)
                if self.fuel_switch_to is None:
                    intensity_carb_meas = dict.fromkeys(intensity_carb_meas, 0)
                    cost_energy_meas = dict.fromkeys(cost_energy_meas, 0)

            # For electricity microsegments in measure scenarios that
            # require the addition of public health cost data, retrieve
            # the appropriate cost data for the given EMM region and add
            if opts.health_costs is not False and ("PHC" in self.name and (
                    "electricity" in mskeys or
                    self.fuel_switch_to == "electricity")):
                # Set row/column key information for the public health
                # cost scenario suggested by the measure name
                row_key = [x[1] for x in self.handyvars.health_scn_names if
                           x[0] in self.name][0]
                col_key = [x[2] for x in self.handyvars.health_scn_names if
                           x[0] in self.name][0]
                # Public health costs are specified in units of $/MMBtu source;
                # add a multiplier to account for the case where the energy
                # outputs that these data will be multiplied by are specified
                # in site units; otherwise, set this multiplier to 1
                if opts is not None and opts.site_energy is True:
                    phc_site_mult = self.handyvars.ss_conv["electricity"]
                else:
                    phc_site_mult = {yr: 1 for yr in self.handyvars.aeo_years}
                # Pull the appropriate public health cost information for
                # the current health cost scenario and EMM region; convert
                # from units of cents/primary kWh to $/MMBtu source and add
                # source-site multiplier, if necessary
                phc_dat = {yr: ((self.handyvars.health_scn_data[
                    numpy.in1d(
                        self.handyvars.health_scn_data["Category"], row_key) &
                    numpy.in1d(
                        self.handyvars.health_scn_data["EMM_Region"],
                        mskeys[1])][col_key])[0] / 100) * 293.07107 *
                    phc_site_mult[yr] for yr in self.handyvars.aeo_years}
                # Update energy costs with public health data; in fuel switch
                # case, do not add to baseline as baseline was non-electric
                if self.fuel_switch_to == "electricity":
                    # Update measure
                    cost_energy_meas = {yr: (val + phc_dat[yr]) for yr, val in
                                        cost_energy_meas.items()}
                else:
                    # Update baseline
                    cost_energy_base = {yr: (val + phc_dat[yr]) for yr, val in
                                        cost_energy_base.items()}
                    # Update measure
                    cost_energy_meas = {yr: (val + phc_dat[yr]) for yr, val in
                                        cost_energy_meas.items()}

            # Initialize cost/performance/lifetime, stock/energy, square
            # footage, and new building fraction variables for the baseline
            # microsegment associated with the current key chain
            base_cpl = msegs_cpl
            # Initialize same data for "switched to" mseg, if applicable
            if mskeys_swtch:
                base_cpl_swtch = base_cpl
            else:
                base_cpl_swtch = ""
            mseg = msegs
            mseg_sqft_stock = msegs
            new_constr = {"annual new": {}, "total new": {},
                          "total": {}, "new fraction": {}}

            # Initialize a variable for measure relative performance (broken
            # out by year in modeling time horizon)
            rel_perf = {}

            # In cases where measure and baseline cost/performance/lifetime
            # data and/or baseline stock/energy market size data are formatted
            # as nested dicts, loop recursively through dict levels until
            # appropriate terminal value is reached
            for i in range(0, len(mskeys)):
                # For use of state regions, cost/performance/lifetime data
                # are broken out by census division; map the state of the
                # current microsegment to the census division it belongs to,
                # to enable retrieval of the cost/performance/lifetime data
                if (i == 1) and self.handyvars.region_cpl_mapping:
                    reg_cpl_map = [
                        x[0] for x in
                        self.handyvars.region_cpl_mapping.items() if
                        mskeys[1] in x[1]][0]
                    # Mapping should yield single string for census division
                    if not isinstance(reg_cpl_map, str):
                        raise ValueError("State " + mskeys[1] +
                                         " could not be mapped to a census "
                                         "division for the purpose of "
                                         "retrieving baseline cost, "
                                         "performance, and lifetime data")
                else:
                    reg_cpl_map = ''

                # Check whether baseline microsegment cost/performance/lifetime
                # data are in dict format and current key is in dict keys; if
                # so, proceed further with the recursive loop. * Note: dict key
                # hierarchies and syntax are assumed to be consistent across
                # all measure and baseline cost/performance/lifetime and
                # stock/energy market data, with the exception of state data,
                # where cost/performance/lifetime data are broken out by
                # census divisions and must be mapped to the state breakouts
                # used in the stock_energy market data
                if (isinstance(base_cpl, dict) and (
                    (mskeys[i] in base_cpl.keys()) or (
                     reg_cpl_map and reg_cpl_map in base_cpl.keys())) or
                    mskeys[i] in [
                        "primary", "secondary", "new", "existing", None]):
                    # Skip over "primary", "secondary", "new", and "existing"
                    # keys in updating baseline stock/energy, cost and lifetime
                    # information (this information is not broken out by these
                    # categories)
                    if mskeys[i] not in [
                            "primary", "secondary", "new", "existing"]:

                        # Restrict base cost/performance/lifetime dict to key
                        # chain info.
                        if reg_cpl_map:  # Handle region mapping
                            base_cpl = base_cpl[reg_cpl_map]
                            # Use same mapping for "switched to" data if
                            # applicable
                            if base_cpl_swtch:
                                base_cpl_swtch = \
                                    base_cpl_swtch[reg_cpl_map]
                        else:
                            if mskeys[i] is not None:  # Handle 'None' tech.
                                base_cpl = base_cpl[mskeys[i]]
                            # Do the same for "switched to" data if applicable
                            if base_cpl_swtch and mskeys_swtch[i] is not None:
                                try:
                                    base_cpl_swtch = \
                                        base_cpl_swtch[mskeys_swtch[i]]
                                except KeyError:
                                    raise KeyError(
                                        "Provided microsegment keys for "
                                        "measure '" + self.name + "'' of " +
                                        str(mskeys_swtch) +
                                        " invalid for pulling switched to "
                                        "tech data. Check 'fuel_switch_to' "
                                        "and 'tech_switch_to' fields in "
                                        "measure definition to ensure names "
                                        "are consistent with those here: "
                                        "https://scout-bto.readthedocs.io/"
                                        "en/latest/ecm_reference."
                                        "html#technology. Currently these "
                                        "fields are set to " +
                                        str(self.fuel_switch_to) + " and " +
                                        str(self.tech_switch_to) +
                                        ", respectively")

                        if mskeys[i] is not None:
                            # Restrict stock/energy dict to key chain info.
                            mseg = mseg[mskeys[i]]

                            # Restrict ft^2 floor area dict to key chain info.
                            if i < 3:  # Note: ft^2 fl. broken out 2 levels
                                mseg_sqft_stock = mseg_sqft_stock[mskeys[i]]

                    # Handle a superfluous 'undefined' key in the ECM
                    # cost, performance, and lifetime fields that is generated
                    # by the 'Add ECM' web form in certain cases *** NOTE: WILL
                    # FIX IN FUTURE UI VERSION ***
                    if (any([type(x) is dict and "undefined" in x.keys()
                             for x in [perf_meas, perf_units, cost_meas,
                                       cost_units, life_meas]])):
                        if isinstance(perf_meas, dict) and "undefined" in \
                           perf_meas.keys():
                            perf_meas = perf_meas["undefined"]
                        if isinstance(perf_units, dict) and "undefined" in \
                           perf_units.keys():
                            perf_units = perf_units["undefined"]
                        if isinstance(cost_meas, dict) and "undefined" in \
                           cost_meas.keys():
                            cost_meas = cost_meas["undefined"]
                        if isinstance(cost_units, dict) and "undefined" in \
                           cost_units.keys():
                            cost_units = cost_units["undefined"]
                        if isinstance(life_meas, dict) and "undefined" in \
                           life_meas.keys():
                            life_meas = life_meas["undefined"]

                    # Check for/handle breakouts of performance, cost,
                    # lifetime, or market scaling information

                    # Determine the full set of potential breakout keys
                    # that should be represented in the given ECM attribute for
                    # the current microsegment level (used to check for
                    # missing information below); for region (level 2), provide
                    # a set of alternate breakout keys that may be used
                    if (any([(type(x) is dict or type(x) is list) for x in [
                        perf_meas, perf_units, cost_meas, cost_units,
                        life_meas, mkt_scale_frac,
                            mkt_scale_frac_source]])):
                        # primary/secondary level
                        if (i == 0):
                            break_keys = ["primary", "secondary"]
                            alt_break_keys = ''
                            err_message = ''
                        # full set of climate breakouts
                        elif (i == 1):
                            break_keys = self.climate_zone
                            # set of alternate regional breakout possibilities
                            alt_break_keys = \
                                self.handyvars.alt_perfcost_brk_map["levels"]
                            err_message = "regions the measure applies to: "
                        # full set of building breakouts
                        elif (i == 2):
                            break_keys = self.bldg_type
                            alt_break_keys = ''
                            err_message = "buildings the measure applies to: "
                        # full set of fuel breakouts
                        elif (i == 3):
                            break_keys = self.fuel_type[mskeys[0]]
                            alt_break_keys = ''
                            err_message = "fuel types the measure applies to: "
                        # full set of end use breakouts
                        elif (i == 4):
                            break_keys = self.end_use[mskeys[0]]
                            alt_break_keys = ''
                            err_message = "end uses the measure applies to: "
                        # full set of technology breakouts
                        elif (i == (len(mskeys) - 2)):
                            break_keys = self.technology[mskeys[0]]
                            alt_break_keys = ''
                            err_message = \
                                "technologies the measure applies to: "
                        # full set of vintage breakouts
                        elif (i == (len(mskeys) - 1)):
                            break_keys = self.structure_type
                            alt_break_keys = ''
                            err_message = \
                                "building vintages the measure applies to: "
                        else:
                            break_keys = ''
                            alt_break_keys = ''
                            err_message = ''

                        # Restrict any measure cost/performance/lifetime/market
                        # scaling info. that is a dict type to key chain info.
                        # - in the process, check to ensure that if there is
                        # breakout information provided for the given level in
                        # the microsegment, this breakout information is
                        # complete (for example, if performance is broken out
                        # by climate region, ALL climate regions should be
                        # present in the breakout keys)

                        # Performance data

                        # Case where data are broken out directly by mseg info.
                        if isinstance(perf_meas, dict) and break_keys and all([
                                x in perf_meas.keys() for x in break_keys]):
                            perf_meas = perf_meas[mskeys[i]]
                        # Case where region is being looped through in the mseg
                        # and performance data use alternate regional breakout
                        elif isinstance(perf_meas, dict) and alt_break_keys:
                            # Determine the alternate regions by which the
                            # performance data are broken out (e.g., IECC, or
                            # - if the analysis uses EMM regions or states -
                            # AIA)
                            alt_key_reg_typ = [
                                x for x in
                                self.handyvars.alt_perfcost_brk_map.keys()
                                if any([
                                    x in y for y in perf_meas.keys()])]
                            # If the alternate regional breakout is supported,
                            # reformat the performance data for subsequent
                            # calculations
                            if len(alt_key_reg_typ) > 0:
                                alt_key_reg_typ = alt_key_reg_typ[0]
                                # Check to ensure the expected alternate
                                # breakout keys are provided
                                if sorted(perf_meas.keys()) == sorted(
                                    self.handyvars.alt_perfcost_brk_map[
                                        alt_key_reg_typ][alt_key_reg_typ]):
                                    # Store data in a list, where the first
                                    # element is a dict of performance data
                                    # broken out by each alternate region and
                                    # the second element is the portion of
                                    # each alternate region that falls in the
                                    # current mseg region
                                    perf_meas = copy.deepcopy([
                                        perf_meas,
                                        self.handyvars.alt_perfcost_brk_map[
                                            alt_key_reg_typ][mskeys[1]]])
                                # If unexpected keys are present, yield error
                                else:
                                    raise KeyError(
                                        self.name +
                                        ' energy performance (energy_'
                                        'efficiency) must be broken out '
                                        'by ALL ' + err_message +
                                        str(break_keys) + ' OR alternate '
                                        'regions ' + alt_break_keys)
                        # Case where performance data broken out by alternate
                        # regions were reformatted as a list and require
                        # further work to finalize as a single value for the
                        # current mseg
                        elif isinstance(perf_meas, list) and \
                                isinstance(perf_meas[0], dict):
                            # Check the first element of the list for
                            # performance data in each of the alternate regions
                            # that is still in dict format and must be keyed
                            # in further by info. for the current mseg.
                            for k in perf_meas[0].keys():
                                if isinstance(perf_meas[0][k], dict) and \
                                    break_keys and all([
                                        x in perf_meas[0][k].keys()
                                        for x in break_keys]):
                                    perf_meas[0][k] = perf_meas[0][k][
                                        mskeys[i]]
                            # If none of the performance data in the first
                            # element of the list needs to be keyed in further,
                            # perform a weighted sum of the data across the
                            # alternate regions into the current mseg region,
                            # to arrive at a final performance value for that
                            # region
                            if all([type(x) != dict for
                                    x in perf_meas[0].values()]):
                                perf_meas = sum([x * y for x, y in zip(
                                    perf_meas[0].values(), perf_meas[1])])
                        # If none of the above cases holds, yield error
                        elif isinstance(perf_meas, dict) and any(
                                [x in perf_meas.keys() for x in break_keys]):
                            raise KeyError(
                                    self.name +
                                    ' energy performance (energy_efficiency) '
                                    'must be broken out '
                                    'by ALL ' + err_message + str(break_keys))
                        # Cost data - same approach as performance data

                        # Case where data are broken out directly by mseg info.
                        if isinstance(cost_meas, dict) and break_keys and \
                            all([x in cost_meas.keys() for
                                 x in break_keys]):
                            cost_meas = cost_meas[mskeys[i]]
                        # Case where region is being looped through in the mseg
                        # and cost data use alternate regional breakout
                        elif isinstance(cost_meas, dict) and alt_break_keys:
                            # Determine the alternate regions by which the
                            # cost data are broken out (e.g., IECC, or
                            # - if the analysis uses EMM regions or states -
                            # AIA)
                            alt_key_reg_typ = [
                                x for x in
                                self.handyvars.alt_perfcost_brk_map.keys()
                                if any([
                                    x in y for y in cost_meas.keys()])]
                            # If the alternate regional breakout is supported,
                            # reformat the cost data for subsequent
                            # calculations
                            if len(alt_key_reg_typ) > 0:
                                alt_key_reg_typ = alt_key_reg_typ[0]
                                # Check to ensure the expected alternate
                                # breakout keys are provided
                                if sorted(cost_meas.keys()) == sorted(
                                    self.handyvars.alt_perfcost_brk_map[
                                        alt_key_reg_typ][alt_key_reg_typ]):
                                    # Store data in a list, where the first
                                    # element is a dict of cost data
                                    # broken out by each alternate region and
                                    # the second element is the portion of
                                    # each alternate region that falls in the
                                    # current mseg region
                                    cost_meas = copy.deepcopy([
                                        cost_meas,
                                        self.handyvars.alt_perfcost_brk_map[
                                            alt_key_reg_typ][mskeys[1]]])
                                # If unexpected keys are present, yield error
                                else:
                                    raise KeyError(
                                        self.name +
                                        ' installed cost (installed_'
                                        'cost) must be broken out '
                                        'by ALL ' + err_message +
                                        str(break_keys) + ' OR alternate '
                                        'regions ' + alt_break_keys)
                        # Case where cost data broken out by alternate
                        # regions were reformatted as a list and require
                        # further work to finalize as a single value for the
                        # current mseg
                        elif isinstance(cost_meas, list) and \
                                isinstance(cost_meas[0], dict):
                            # Check the first element of the list for
                            # cost data in each of the alternate regions
                            # that is still in dict format and must be keyed
                            # in further by info. for the current mseg.
                            for k in cost_meas[0].keys():
                                if isinstance(cost_meas[0][k], dict) and \
                                    break_keys and all([
                                        x in cost_meas[0][k].keys()
                                        for x in break_keys]):
                                    cost_meas[0][k] = cost_meas[0][k][
                                        mskeys[i]]
                            # If none of the cost data in the first element of
                            # the list needs to be keyed in further, perform a
                            # weighted sum of the data across the alternate
                            # regions into the current mseg region, to arrive
                            # at a final cost value for that region
                            if all([type(x) != dict for
                                    x in cost_meas[0].values()]):
                                cost_meas = sum([x * y for x, y in zip(
                                    cost_meas[0].values(), cost_meas[1])])
                        elif isinstance(cost_meas, dict) and any(
                                [x in cost_meas.keys() for x in break_keys]):
                            if alt_break_keys:
                                pass
                            else:
                                raise KeyError(
                                    self.name +
                                    ' installed cost (installed_cost) must '
                                    'be broken out '
                                    'by ALL ' + err_message + str(break_keys))

                        # Performance units data
                        if isinstance(perf_units, dict) and break_keys and \
                                all([x in perf_units.keys() for
                                     x in break_keys]):
                            perf_units = perf_units[mskeys[i]]
                        elif isinstance(perf_units, dict) and any(
                                [x in perf_units.keys() for x in break_keys]):
                            raise KeyError(
                                self.name +
                                ' energy performance units ('
                                'energy_efficiency_units) must be broken '
                                'out by ALL ' + err_message +
                                str(break_keys))
                        # Cost units data
                        if isinstance(cost_units, dict) and break_keys and \
                            all([x in cost_units.keys() for
                                 x in break_keys]):
                            cost_units = cost_units[mskeys[i]]
                        elif isinstance(cost_units, dict) and any(
                                [x in cost_units.keys() for x in break_keys]):
                            raise KeyError(
                                self.name +
                                ' installed cost units (installed_cost_'
                                'units) must be broken out '
                                'by ALL ' + err_message + str(break_keys))
                        # Lifetime data
                        if isinstance(life_meas, dict) and break_keys and all([
                                x in life_meas.keys() for x in break_keys]):
                            life_meas = life_meas[mskeys[i]]
                        elif isinstance(life_meas, dict) and any(
                                [x in life_meas.keys() for x in break_keys]):
                            raise KeyError(
                                self.name +
                                ' lifetime (product_lifetime) must be '
                                'broken out '
                                'by ALL ' + err_message + str(break_keys))
                        # Market scaling fractions
                        if isinstance(mkt_scale_frac, dict) and break_keys \
                            and all([x in mkt_scale_frac.keys() for
                                     x in break_keys]):
                            mkt_scale_frac = mkt_scale_frac[mskeys[i]]
                        elif isinstance(mkt_scale_frac, dict) and any(
                                [x in mkt_scale_frac.keys() for
                                 x in break_keys]):
                            raise KeyError(
                                self.name +
                                ' market scaling fractions (market_'
                                'scaling_fractions) must be '
                                'broken out by ALL ' + err_message +
                                str(break_keys))
                        # Market scaling fraction source
                        if isinstance(mkt_scale_frac_source, dict) and \
                            break_keys and all([
                                x in mkt_scale_frac_source.keys() for
                                x in break_keys]):
                            mkt_scale_frac_source = \
                                mkt_scale_frac_source[mskeys[i]]
                        elif isinstance(mkt_scale_frac_source, dict) and any(
                                [x in mkt_scale_frac_source.keys() for
                                 x in break_keys]):
                            raise KeyError(
                                self.name +
                                ' market scaling fraction source (market_'
                                'scaling_fraction_source) must be '
                                'broken out by ALL ' + err_message +
                                str(break_keys))

                # If no key match, break the loop
                else:
                    if mskeys[i] is not None:
                        mseg = {}
                    break

            # Continue loop if key chain doesn't yield "stock"/"energy" keys
            if any([x not in list(mseg.keys()) for x in ["stock", "energy"]]):
                continue
            # Continue loop if time-sensitive valuation is required and the
            # current microsegment technology does not have the necessary
            # load shape information (pertinent to internal heat gains for
            # envelope heating microsegments, when such measures are being
            # assessed in isolation and not as part of a package (in package
            # case no envelope load shape information is generated/needed))
            elif ("demand" in mskeys and
                  self.name not in ctrb_ms_pkg_prep) and ((
                    (opts.tsv_metrics is not False or
                     opts.sect_shapes is True) or
                    self.tsv_features is not None) and (
                     mskeys[4] in ["heating", "secondary heating"] and
                     mskeys[-2] in self.handyvars.env_heat_ls_scrn)):
                continue
            # Continue loop if key chain yields "stock"/"energy" keys but
            # the stock or energy data are missing
            elif any([x == {} for x in [mseg["stock"], mseg["energy"]]]):
                if mskeys[-2] not in stk_energy_warn:
                    stk_energy_warn.append(mskeys[-2])
                    verboseprint(
                        opts.verbose,
                        "WARNING: ECM '" + self.name +
                        "' missing valid baseline "
                        "stock/energy data " +
                        "for technology '" + str(mskeys[-2]) +
                        "'; removing technology from analysis")
                # Add to the overall number of key chains that yield "stock"/
                # "energy" keys (but in this case, are missing data)
                valid_keys += 1
                continue
            # Otherwise update all stock/energy/cost information for each year
            else:
                # Restrict count of key chains with valid stock/energy data to
                # "primary" microsegment key chains only (the key chain
                # count is used later in stock and stock cost calculations,
                # which secondary microsegments do not contribute to)
                if mskeys[0] == "primary":
                    valid_keys += 1
                    valid_keys_stk_energy += 1
                    # Flag missing stock data, as applicable
                    if mseg["stock"] == "NA" and sqft_subst != 1:
                        no_stk_mseg = True
                    else:
                        no_stk_mseg = ""

                # If applicable, determine the rate of conversion from baseline
                # equipment to heat pumps (including fuel switching cases and
                # same fuel replacements of e.g., resistance heating/WH).
                # Assume only heating, secondary heating, water heating, and
                # cooking end uses are covered by these exogenous rates, and
                # ensure that these rates are only assessed for equipment
                # microsegments (e.g., they do not apply to envelope component
                # heating energy msegs); equipment cooling microsegments that
                # are linked with the heating microsegments are subject to the
                # rates; set to None otherwise
                if self.handyvars.hp_rates and "demand" not in mskeys and (any(
                    [x in mskeys for x in ["secondary heating", "heating",
                                           "water heating", "cooking"]]) or (
                        link_htcl_fs_rates and "cooling" in mskeys)):
                    # Map secondary heating end use to heating HP switch rates
                    if mskeys[4] == "secondary heating":
                        hp_eu_key = "heating"
                    else:
                        hp_eu_key = mskeys[4]
                    # Map the current mseg region to the regionality of the
                    # HP conversion rate data
                    reg = [r[0] for r in
                           self.handyvars.hp_rates_reg_map.items() if
                           mskeys[1] in r[1]][0]
                    # Pull in HP conversion rate data for the user-selected
                    # fuel switching scenario and the region and building type
                    # of the current microsegment
                    hp_rate_dat = self.handyvars.hp_rates[
                        "data (by scenario)"][opts.exog_hp_rates[0]][reg][
                        bldg_sect]
                    # Attempt to further restrict HP conversion data by
                    # fuel type, end use, technology, and building vintage;
                    # handle cases where data are applicable to "all"
                    # technologies within a given combination of fuel, end use,
                    # and vintage, or otherwise set the HP conversion rate to
                    # None if no data are available for the current mseg
                    try:
                        hp_rate = hp_rate_dat[
                            mskeys[3]][hp_eu_key][mskeys[-2]][mskeys[-1]]
                    except KeyError:
                        try:
                            hp_rate = hp_rate_dat[
                                mskeys[3]][hp_eu_key]["all"][mskeys[-1]]
                        except KeyError:
                            if hp_eu_key == "cooling":
                                # HP conversion rates for NGHP cooling msegs
                                # are not directly addressed in the exogenous
                                # file structure but should be set to the same
                                # as NGHP heating
                                if "NGHP" in mskeys:
                                    try:
                                        hp_rate = hp_rate_dat[mskeys[3]][
                                            "heating"][mskeys[-2]][mskeys[-1]]
                                    except KeyError:
                                        hp_rate = None
                                # HP conversion rates for electric cooling
                                # msegs attached to heating msegs that are fuel
                                # switching from fossil to electric and subject
                                # to the HP rates should be subject to the same
                                # rates; attach cooling scaling to NG rates (
                                # NG most prevalent fossil-based heating
                                # technology)
                                elif self.fuel_switch_to == "electricity":
                                    # Separately handle different bldg. types
                                    if bldg_sect == "residential":
                                        try:
                                            hp_rate = hp_rate_dat[
                                                "natural gas"]["heating"][
                                                "furnace (NG)"][mskeys[-1]]
                                        except KeyError:
                                            try:
                                                hp_rate = hp_rate_dat[
                                                    "natural gas"]["heating"][
                                                    "all"][mskeys[-1]]
                                            except KeyError:
                                                hp_rate = None
                                    elif any([mskeys[-2] in x for x in [
                                        self.handyvars.com_RTU_fs_tech,
                                            self.handyvars.com_nRTU_fs_tech]]):
                                        # Determine whether the current cooling
                                        # tech. falls into switch from an RTU
                                        # or other tech.
                                        if mskeys[-2] in \
                                                self.handyvars.com_RTU_fs_tech:
                                            tech_key = "RTUs"
                                        else:
                                            tech_key = "all other"
                                        # Try resultant tech. key
                                        try:
                                            hp_rate = hp_rate_dat[
                                                "natural gas"]["heating"][
                                                tech_key][mskeys[-1]]
                                        except KeyError:
                                            hp_rate = None
                                    else:
                                        hp_rate = None
                                # HP conversion rates for electric cooling
                                # msegs attached to electric resistance heating
                                # msegs that are subject to the HP rates should
                                # be subject to the same rates
                                elif self.fuel_switch_to is None and (
                                    mskeys[-2] is not None and
                                        "HP" not in mskeys[-2]):
                                    # Separately handle different bldg. types
                                    if bldg_sect == "residential":
                                        try:
                                            hp_rate = hp_rate_dat[
                                                "electricity"]["heating"][
                                                "resistance heat"][mskeys[-1]]
                                        except KeyError:
                                            try:
                                                hp_rate = hp_rate_dat[
                                                    "electricity"]["heating"][
                                                    "all"][mskeys[-1]]
                                            except KeyError:
                                                hp_rate = None
                                    elif any([mskeys[-2] in x for x in [
                                        self.handyvars.com_RTU_fs_tech,
                                            self.handyvars.com_nRTU_fs_tech]]):
                                        # Determine whether the current cooling
                                        # tech. falls into switch from an RTU
                                        # or other tech.
                                        if mskeys[-2] in \
                                                self.handyvars.com_RTU_fs_tech:
                                            tech_key = "RTUs"
                                        else:
                                            tech_key = "all other"
                                        # Try resultant tech. key
                                        try:
                                            hp_rate = hp_rate_dat[
                                                "electricity"]["heating"][
                                                tech_key][mskeys[-1]]
                                        except KeyError:
                                            hp_rate = None
                                    else:
                                        hp_rate = None
                                else:
                                    hp_rate = None
                            # Handle switch from commercial heating in RTUs vs.
                            # other technologies
                            elif hp_eu_key == "heating" and \
                                bldg_sect == "commercial" and any([
                                    mskeys[-2] in x for x in [
                                        self.handyvars.com_RTU_fs_tech,
                                        self.handyvars.com_nRTU_fs_tech]]):
                                # Determine whether the current heating tech.
                                # falls into switch from an RTU or other tech.
                                if mskeys[-2] in \
                                        self.handyvars.com_RTU_fs_tech:
                                    tech_key = "RTUs"
                                else:
                                    tech_key = "all other"
                                # Try resultant tech. key
                                try:
                                    hp_rate = hp_rate_dat[mskeys[3]][
                                        hp_eu_key][tech_key][mskeys[-1]]
                                except KeyError:
                                    hp_rate = None
                            # Residential secondary heating
                            elif mskeys[4] == "secondary heating" and \
                                    bldg_sect == "residential":
                                # For the tech key, use resistance and/or
                                # furnace tech. depending on the fuel
                                if mskeys[3] == "electricity":
                                    tech_key = "resistance heat"
                                elif mskeys[3] == "distillate":
                                    tech_key = "furnace (distillate)"
                                elif mskeys[3] == "natural gas":
                                    tech_key = "furnace (NG)"
                                else:
                                    tech_key = "furnace (LPG)"

                                # Try resultant tech. key
                                try:
                                    hp_rate = hp_rate_dat[mskeys[3]][
                                        hp_eu_key][tech_key][mskeys[-1]]
                                except KeyError:
                                    hp_rate = None
                            else:
                                hp_rate = None
                else:
                    hp_rate = None

                # For cases where the measure is switching fossil-based
                # or resistance heating and associated cooling to a HP or
                # resistance-based WH to a HP and an external HP conversion
                # rate has been imposed, append an '-FS' or '-RST' to the
                # contributing microsegment tech. information needed for ECM
                # competition; this will ensure that the mseg is only competed
                # with other measures of the same type

                # Cases where measure is converting fossil-based heating or
                # water heating and associated cooling to HP
                if hp_rate and self.fuel_switch_to == "electricity":
                    contrib_mseg_key = list(contrib_mseg_key)
                    # Tech info. is second to last mseg list element
                    try:
                        contrib_mseg_key[-2] += "-FS"
                    # Handle Nonetype on technology
                    except TypeError:
                        contrib_mseg_key[-2] = "-FS"
                    contrib_mseg_key = tuple(contrib_mseg_key)
                # Cases where measure is converting resistance-based heating
                # and associated cooling or WH to HP
                elif hp_rate and any(
                        [x in self.technology["primary"] for x in
                         self.handyvars.resist_ht_wh_tech]) and \
                        "electricity" in mskeys:
                    contrib_mseg_key = list(contrib_mseg_key)
                    # Tech info. is second to last mseg list element
                    try:
                        contrib_mseg_key[-2] += "-RST"
                    # Handle Nonetype on technology
                    except TypeError:
                        contrib_mseg_key[-2] = "-RST"
                    contrib_mseg_key = tuple(contrib_mseg_key)

                # If sub-market scaling fraction is non-numeric (indicating
                # it is not applicable to current microsegment), set to 1
                if mkt_scale_frac is None or isinstance(mkt_scale_frac, dict):
                    mkt_scale_frac = 1

                # If a sub-market scaling fraction is to be applied to the
                # current baseline microsegment, check that the source
                # information for the fraction is sufficient; if not, remove
                # the measure from further analysis
                if isinstance(mkt_scale_frac_source, dict) and \
                        "title" in mkt_scale_frac_source.keys():
                    # Establish sub-market fraction general source, URL, and
                    # derivation information

                    # Set general source info. for the sub-market fraction
                    source_info = [
                        mkt_scale_frac_source['title'],
                        mkt_scale_frac_source['organization']]
                    # Set URL for the sub-market fraction
                    url = mkt_scale_frac_source['URL']
                    # Set information about how sub-market fraction was derived
                    frac_derive = mkt_scale_frac_source['fraction_derivation']

                    # Check the validity of sub-market fraction source, URL,
                    # and derivation information

                    # Check sub-market fraction general source information,
                    # yield warning if source information is invalid and
                    # invalid source information flag hasn't already been
                    # raised for this measure
                    if sbmkt_source_invalid != 1 and (any([
                        not isinstance(x, str) or
                       len(x) < 2 for x in source_info]) is True):
                        # Print invalid source information warning
                        warnings.warn(
                            "WARNING: '" + self.name + "' has invalid "
                            "sub-market scaling fraction source title, author,"
                            " organization, and/or year information")
                        # Set invalid source information flag to 1
                        sbmkt_source_invalid = 1
                    # Check sub-market fraction URL, yield warning if URL is
                    # invalid and invalid URL flag hasn't already been raised
                    # for this measure
                    if sbmkt_url_invalid != 1:
                        # Parse the URL into components (addressing scheme,
                        # network location, etc.)
                        url_check = urlparse(url)
                        # Check for valid URL address scheme and network
                        # location components
                        if (any([len(url_check.scheme),
                                 len(url_check.netloc)]) == 0 or
                            all([x not in url_check.netloc for x in
                                 self.handyvars.valid_submkt_urls])):
                            # Print invalid URL warning
                            warnings.warn(
                                "WARNING: '" + self.name + "' has invalid "
                                "sub-market scaling fraction source URL "
                                "information")
                            # Set invalid URL flag to 1
                            sbmkt_url_invalid = 1
                    # Check sub-market fraction derivation information, yield
                    # warning if invalid
                    if not isinstance(frac_derive, str):
                        # Print invalid derivation warning
                        warnings.warn(
                            "WARNING: '" + self.name + "' has invalid "
                            "sub-market scaling fraction derivation "
                            "information")
                        # Set invalid derivation flag to 1
                        sbmkt_derive_invalid = 1

                    # If the derivation information or the general source
                    # and URL information for the sub-market fraction are
                    # invalid, yield warning that measure will be removed from
                    # analysis, reset the current valid contributing key chain
                    # count to a 999 flag, and flag the measure as inactive
                    # such that it will be removed from all further routines
                    if sbmkt_derive_invalid == 1 or (
                            sbmkt_source_invalid == 1 and
                            sbmkt_url_invalid == 1):
                        # Print measure removal warning
                        warnings.warn(
                            "WARNING (CRITICAL): '" + self.name + "' has "
                            "insufficient sub-market source information and "
                            "will be removed from analysis")
                        # Reset measure 'active' attribute to zero
                        self.remove = True
                        # Break from all further baseline stock/energy/carbon
                        # and cost information updates for the measure
                        break

                # Seed the random number generator such that performance, cost,
                # and lifetime draws are consistent across all microsegments
                # that contribute to a measure's master microsegment (e.g, if
                # measure performance, cost, and/or lifetime distributions
                # are identical relative to two contributing baseline
                # microsegments, the numpy arrays yielded by the random number
                # generator for these measure parameters and microsegments
                # will also be identical)
                numpy.random.seed(rnd_sd)

                # If the measure performance/cost/lifetime variable is list
                # with distribution information, sample values accordingly
                if isinstance(perf_meas, list) and isinstance(perf_meas[0],
                                                              str):
                    # Sample measure performance values
                    perf_meas = self.rand_list_gen(
                        perf_meas, self.handyvars.nsamples)
                    # Set any measure performance values less than zero to
                    # zero, for cases where performance isn't relative
                    if perf_units != 'relative savings (constant)' and \
                        type(perf_units) is not list and any(
                            perf_meas < 0) is True:
                        perf_meas[numpy.where(perf_meas < 0)] = 0
                # Ensure that the code errors if a user has requested
                # data from EnergyPlus simulations anywhere in the ECM's
                # energy performance input value (currently unsupported)
                elif perf_meas == "From EnergyPlus":
                    raise ValueError(
                        "ECM '" + self.name + "' requires EnergyPlus data for "
                        "its performance input; EnergyPlus-based ECM "
                        "performance data are currently unsupported.")

                if isinstance(cost_meas, list) and isinstance(cost_meas[0],
                                                              str):
                    # Sample measure cost values
                    cost_meas = self.rand_list_gen(
                        cost_meas, self.handyvars.nsamples)
                    # Set any measure cost values less than zero to zero
                    if any(cost_meas < 0) is True:
                        cost_meas[numpy.where(cost_meas < 0)] = 0
                if isinstance(life_meas, list) and isinstance(life_meas[0],
                                                              str):
                    # Sample measure lifetime values
                    life_meas = self.rand_list_gen(
                        life_meas, self.handyvars.nsamples)
                    # Set any measure lifetime values in list less than zero
                    # to 1
                    if any(life_meas < 0) is True:
                        life_meas[numpy.where(life_meas < 0)] = 1
                elif isinstance(life_meas, float) or \
                        isinstance(life_meas, int) and mskeys[0] == "primary":
                    # Set measure lifetime point values less than zero to 1
                    # (minimum lifetime)
                    if life_meas < 1:
                        life_meas = 1

                # Initialize base cost, performance, and lifetime data as None
                cost_base, perf_base, life_base,  = (
                    {yr: None for yr in self.handyvars.aeo_years} for n in
                    range(3))
                cost_base_units, perf_base_units = (None for n in range(2))

                # For primary microsegments, set baseline technology cost,
                # cost units, performance, performance units, and lifetime, if
                # data are available on these parameters; if data are not
                # available, remove primary microsegment from further analysis,
                # and later remove any associated secondary microsegments
                if mskeys[0] == "primary":
                    try:
                        # Set baseline performance; try for case where baseline
                        # performance is broken out by new and existing
                        # vintage; given an exception, expect a single set of
                        # values across both vintages
                        try:
                            perf_base = base_cpl["performance"]["typical"][
                                mskeys[-1]]
                        except KeyError:
                            perf_base = base_cpl["performance"]["typical"]
                        # Set baseline performance units
                        perf_base_units = base_cpl["performance"]["units"]

                        # Check for incentives information in costs; in this
                        # case, costs before incentives are under a sub-key
                        # and incentives under another sub-key

                        # Pull out typical cost data before incentives
                        if "incentives" in base_cpl["installed cost"].keys():
                            # Set baseline costs before incentives dict
                            cost_base_init = base_cpl["installed cost"][
                                "before incentives"]
                            # Ensure that add-on measures (e.g., controls)
                            # are not assigned incentives that are meant
                            # for equipment replacements to which the add-ons
                            # are attached. Also do not proceed w/ incentives
                            # data for measures with relative performance
                            # units as these cannot be linked back to EIA
                            # incentives levels
                            if (self.measure_type != "add-on" and
                                    "relative savings" not in perf_units):
                                # Pull out baseline incentives data
                                cost_incentives = base_cpl["installed cost"][
                                    "incentives"]["by performance tier"]
                                # Further restrict incentives by new or
                                # existing, if applicable
                                if mskeys[-1] in cost_incentives.keys():
                                    cost_incentives = \
                                        cost_incentives[mskeys[-1]]
                                # Set perf. units for base mseg incentives
                                i_units_base = base_cpl[
                                    "installed cost"][
                                        "incentives"]["performance units"]
                                # Set incentives for "switched to" mseg if
                                # needed
                                if base_cpl_swtch:
                                    # Pull out measure incentives data
                                    cost_incentives_meas = base_cpl_swtch[
                                        "installed cost"]["incentives"][
                                        "by performance tier"]
                                    # Further restrict incentives by new or
                                    # existing, if applicable
                                    if mskeys_swtch[-1] in \
                                            cost_incentives_meas.keys():
                                        cost_incentives_meas = \
                                            cost_incentives_meas[mskeys[-1]]
                                    # Set performance units for measure mseg
                                    # incentives
                                    i_units_meas = base_cpl_swtch[
                                        "installed cost"]["incentives"][
                                        "performance units"]
                                else:
                                    cost_incentives_meas, \
                                        i_units_meas = (
                                            "" for n in range(2))
                            else:
                                # No incentives assigned to add-on measures
                                # in either the baseline or the measure
                                cost_incentives, cost_incentives_meas, \
                                    i_units_base, \
                                    i_units_meas = (
                                        "" for n in range(4))
                        else:
                            # Set baseline costs
                            cost_base_init = base_cpl["installed cost"]
                            # No incentives data
                            cost_incentives, cost_incentives_meas, \
                                i_units_base, \
                                i_units_meas = (
                                    "" for n in range(4))

                        # In some cases, typical cost data will be split
                        # further by new vs. existing keys; handle accordingly
                        # and finalize costs (before incentives)
                        if mskeys[-1] in cost_base_init["typical"].keys():
                            cost_base = cost_base_init["typical"][mskeys[-1]]
                        else:
                            cost_base = cost_base_init["typical"]
                        # Set baseline cost units
                        cost_base_units = cost_base_init["units"]

                        # Set baseline lifetime
                        life_base = base_cpl["lifetime"]["average"]
                        # Extend baseline lifetime to dict broken out by
                        # year if necessary
                        if type(base_cpl["lifetime"]["average"]) in [
                                float, int]:
                            life_base = {yr: life_base for
                                         yr in self.handyvars.aeo_years}

                        # Check for cases where baseline data are available but
                        # set to zero, "NA", or 999 values (excepting cases
                        # where baseline cost is expected to be zero). In such
                        # cases, raise a ValueError after checking to ensure
                        # that all baseline data are invalid

                        # Installed cost
                        if any([((("lighting" in mskeys and (isinstance(
                            x[1], float) and round(x[1]) in [0, 999])) or
                            x[1] in [0, "NA", 999]) and mskeys[-2] not in
                            self.handyvars.zero_cost_tech) for x in
                                cost_base.items()]):
                            # If some years have valid cost data, take the max
                            # from those years and extend across the full
                            # time horizon (cases like commercial lighting
                            # sometimes have typical CPL data that declines to
                            # zero with declining stock)
                            mx_cb = round(max([
                                x[1] for x in cost_base.items() if
                                x[1] != "NA"]))
                            if mx_cb not in [0, 999]:
                                cost_base = {yr: mx_cb for yr
                                             in self.handyvars.aeo_years}
                            else:
                                raise ValueError
                        # Performance
                        if any([(("lighting" in mskeys and (isinstance(
                            y[1], float) and round(y[1]) in [0, 999])) or
                            y[1] in [0, "NA", 999]) for y in
                                perf_base.items()]):
                            # If some years have valid performance data, take
                            # the max from those years and extend across the
                            # full time horizon
                            mx_pb = round(max([
                                x[1] for x in perf_base.items() if
                                x[1] != "NA"]))
                            if mx_pb not in [0, 999]:
                                perf_base = {yr: mx_pb for yr
                                             in self.handyvars.aeo_years}
                            else:
                                raise ValueError
                        # Lifetime
                        if any([z[1] in [0, "NA"] for z in life_base.items()]):
                            # If some years have valid lifetime data, take
                            # the max from those years and extend across the
                            # full time horizon
                            mx_lf = round(max([
                                x[1] for x in life_base.items() if
                                x[1] != "NA"]))
                            if mx_lf not in [0, 999]:
                                life_base = {yr: mx_lf for yr
                                             in self.handyvars.aeo_years}
                            else:
                                raise ValueError

                        # Handle case where measure units do not equal
                        # baseline units and baseline units cannot be
                        # converted to measure units; the baseline segment
                        # must be removed from the analysis
                        if (not (
                            perf_units == 'relative savings (constant)' or
                           (isinstance(perf_units, list) and perf_units[0] ==
                            'relative savings (dynamic)')) and
                                (perf_units != perf_base_units)) and \
                                (perf_base_units in
                                 self.handyvars.tech_units_rmv):
                            # Warn user of the removal of the baseline segment
                            if mskeys[-2] is not None and \
                                    mskeys[-2] not in cpl_warn:
                                cpl_warn.append(mskeys[-2])
                                verboseprint(
                                    opts.verbose,
                                    "WARNING: ECM '" + self.name +
                                    "' uses invalid performance units for "
                                    "technology '" + str(mskeys[-2]) +
                                    "' (requires " + str(perf_base_units) +
                                    "); removing technology from analysis")
                            # Continue to the next microsegment
                            continue
                        # Handle case where measure units do not equal baseline
                        # units but baseline units can be converted to measure
                        # units (for example, AFUE to COP)
                        elif (not (
                            perf_units == 'relative savings (constant)' or
                              (isinstance(perf_units, list) and
                               perf_units[0] == 'relative savings (dynamic)'))
                                and (perf_units != perf_base_units)) and \
                            (perf_units in
                                self.handyvars.tech_units_map.keys()):
                            convert_fact = self.handyvars.tech_units_map[
                                perf_units][perf_base_units]
                            # Convert base performance values to values in
                            # measure performance units
                            perf_base = {yr: (perf_base[yr] * convert_fact) for
                                         yr in self.handyvars.aeo_years}
                            # Set baseline performance units to measure units
                            perf_base_units = perf_units
                            # Set baseline mseg incentive performance units
                            # to measure units, if applicable
                            if i_units_base:
                                i_units_base = perf_units
                            # Warn the user of the value/units conversions
                            if mskeys[-2] is not None and \
                                    mskeys[-2] not in cpl_warn:
                                verboseprint(
                                    opts.verbose, "WARNING: ECM '" +
                                    self.name + "' uses units of COP for "
                                    "technology '" + str(mskeys[-2]) +
                                    "' (requires " + str(perf_base_units) +
                                    "); base units changed to " +
                                    str(perf_base_units) + " and base values"
                                    " multiplied by " + str(convert_fact))

                        # Handle case where user has defined a 'windows
                        # conduction' technology without 'windows solar',
                        # or vice versa, and the additional market must be
                        # added at the baseline performance level (relative
                        # savings percentage of zero)
                        if not isinstance(perf_units, list) and \
                            perf_units != 'relative savings (constant)' and (
                            perf_base_units != perf_units and any([
                                x is not None and "windows" in x for
                                x in mskeys])):
                            perf_meas, perf_units = [
                                0, "relative savings (constant)"]

                        # If the baseline technology is a heat pump in the
                        # residential sector, account for the fact that EIA
                        # divides all existing heat pump costs by 2 when
                        # separately considered across the heating and cooling
                        # services. For new construction, EIA puts the full
                        # cost in heating and zeroes out cooling (as we also
                        # do above)
                        if bldg_sect == "residential" and (
                                mskeys[-1] == "existing" and
                                (mskeys[-2] is not None and
                                 "HP" in mskeys[-2])):
                            # Multiply heat pump baseline cost units by 2
                            # to account for EIA heat pump cost handling
                            cost_base = {yr: cost_base[yr] * 2 for yr in
                                         self.handyvars.aeo_years}
                            # Warn the user about the modification to EIA's
                            # baseline cost data (and stock data, which is
                            # also multiplied by 2 below) for this segment
                            if mskeys[-2] not in hp_warn:
                                hp_warn.append(mskeys[-2])
                                verboseprint(
                                    opts.verbose,
                                    "WARNING: Stock/stock cost data for "
                                    "comparable residential baseline "
                                    "technology '" + str(mskeys[-2]) + "' "
                                    "multiplied by two to account for EIA "
                                    "handling of heat pump stock/stock costs "
                                    "for the residential heating and cooling "
                                    "end uses (both are divided by 2 when "
                                    "separately considered across heating and "
                                    "cooling in the raw EIA data)")
                        # Adjust residential baseline lighting lifetimes to
                        # reflect the fact that input data assume 24 h/day of
                        # lighting use, rather than 3 h/day as assumed for
                        # measure lifetime definitions
                        if bldg_sect == "residential" and \
                                mskeys[4] == "lighting":
                            life_base = {yr: life_base[yr] * (24 / 3) for
                                         yr in self.handyvars.aeo_years}
                        # Add to count of primary microsegment key chains with
                        # valid cost/performance/lifetime data
                        valid_keys_cpl += 1
                    except (TypeError, ValueError):
                        # No incentives data
                        cost_incentives, cost_incentives_meas, \
                            i_units_base, i_units_meas = (
                                "" for n in range(4))
                        # In cases with missing baseline technology cost,
                        # performance, or lifetime data where the user
                        # specifies the measure as an 'add-on' type or
                        # applies to a secondary heating technology AND
                        # specifies relative savings for energy performance
                        # units, set the baseline cost to zero and - if no
                        # baseline lifetime data are available - set baseline
                        # lifetime to 10 to ensure that all subsequent stock
                        # and energy impact calculations will continue for that
                        # baseline segment. Note: this marks a special
                        # exception to the general rule that baseline
                        # market segments without complete unit-level cost,
                        # performance, and/or lifetime data will be removed
                        # from further analysis. The exception is needed to
                        # handle ECMs that apply to baseline market segments
                        # with poor technology-level data - for example,
                        # residential vacancy sensors that reduce MELs
                        # energy use by turning off power draws to circuits
                        # when an occupant isn't home. The user will now be
                        # able to evaluate such measures given measure relative
                        # performance, incremental cost, and lifetime data
                        if (self.measure_type == "add-on" or
                            "secondary heating" in mskeys) and \
                                perf_units == "relative savings (constant)":
                            # Set baseline cost to zero with appropriate units
                            # for the building sector of interest
                            cost_base = {yr: 0 for
                                         yr in self.handyvars.aeo_years}
                            if sqft_subst == 1:
                                cost_base_units = "$/ft^2 floor"
                            elif bldg_sect == "residential":
                                cost_base_units = "$/unit"
                            elif bldg_sect == "commercial":
                                # Units per end use service demand
                                if mskeys[4] == "lighting":
                                    cost_base_units = '$/1000 lm'
                                elif mskeys[4] == "ventilation":
                                    cost_base_units = '$/1000 CFM'
                                else:
                                    cost_base_units = '$/kBtu/h ' + mskeys[4]
                            # Attempt to retrieve baseline lifetime data, if
                            # that fails set baseline lifetime to 10 years
                            try:
                                if any([x[1] in [0, "NA"] for x in base_cpl[
                                        "lifetime"]["average"].items()]):
                                    raise ValueError
                                life_base = base_cpl["lifetime"]["average"]
                            except (TypeError, ValueError):
                                life_base = {yr: 10 for
                                             yr in self.handyvars.aeo_years}
                            # Add to count of primary microsegment key chains
                            # with valid cost/performance/lifetime data,
                            # given special exception
                            valid_keys_cpl += 1
                            # Nevertheless, warn the user about the
                            # special exception and how it's handled
                            if mskeys[-2] not in cpl_warn:
                                cpl_warn.append(mskeys[-2])
                                verboseprint(
                                    opts.verbose,
                                    "WARNING: ECM '" + self.name +
                                    "' missing valid baseline "
                                    "cost/performance/lifetime data " +
                                    "for technology '" + str(mskeys[-2]) +
                                    "'; technology will " +
                                    "remain in analysis with cost of zero; " +
                                    "; if lifetime data are missing, " +
                                    "lifetime is set to 10 years")

                        # Additionally, include an exception for lighting
                        # cases, where some segments of lighting energy use
                        # at or near zero contribution lack any cost,
                        # performance, and lifetime data. In such cases, set
                        # the baseline cost and performance to the measure cost
                        # and performance; if lifetime data are not available,
                        # set the baseline lifetime to 10 years.
                        elif "lighting" in mskeys:
                            # Set baseline performance/units to measure
                            # performance/units
                            perf_base = {yr: perf_meas for
                                         yr in self.handyvars.aeo_years}
                            perf_base_units = perf_units
                            # Set baseline cost/units to measure cost/units;
                            # account for possible formatting of measure costs
                            # as a dict broken out by year
                            cost_base = {yr: cost_meas if not isinstance(
                                cost_meas, dict) else cost_meas[yr] for
                                yr in self.handyvars.aeo_years}
                            cost_base_units = cost_units
                            # Attempt to retrieve baseline lifetime data, if
                            # that fails set baseline lifetime to 10 years
                            try:
                                if any([x[1] in [0, "NA"] for x in base_cpl[
                                        "lifetime"]["average"].items()]):
                                    raise ValueError
                                life_base = base_cpl["lifetime"]["average"]
                            except (TypeError, ValueError):
                                life_base = {yr: 10 for
                                             yr in self.handyvars.aeo_years}
                            # Add to count of primary microsegment key chains
                            # with valid cost/performance/lifetime data,
                            # given special exception
                            valid_keys_cpl += 1
                            # Nevertheless, warn the user about the special
                            # exception and how it's handled
                            if mskeys[-2] not in cpl_warn:
                                cpl_warn.append(mskeys[-2])
                                verboseprint(
                                    opts.verbose,
                                    "WARNING: ECM '" + self.name +
                                    "' missing valid baseline "
                                    "cost/performance/lifetime data " +
                                    "for technology '" + str(mskeys[-2]) +
                                    "'; technology applies to special " +
                                    "lighting case and will " +
                                    "remain in analysis at same cost/" +
                                    "performance as ECM; if lifetime data " +
                                    "are missing, lifetime is set to 10 years")
                        # For all other cases, record missing baseline data; if
                        # in verbose mode and the user has not already been
                        # warned about missing data for the given technology,
                        # print warning; exclude technologies without data from
                        # further analysis
                        else:
                            if mskeys[-2] is not None and \
                                    mskeys[-2] not in cpl_warn:
                                cpl_warn.append(mskeys[-2])
                                verboseprint(
                                    opts.verbose,
                                    "WARNING: ECM '" + self.name +
                                    "' missing valid baseline "
                                    "cost/performance/lifetime data " +
                                    "for technology '" + str(mskeys[-2]) +
                                    "'; removing technology from analysis")
                            continue
                else:
                    # Set baseline cost and performance characteristics for any
                    # remaining secondary microsegments to that of the measure
                    # and baseline lifetime to ten years (typical commercial
                    # lighting lifetime)
                    cost_base, perf_base, life_base = [
                        {yr: x for yr in self.handyvars.aeo_years} for x in [
                            cost_meas, perf_meas, 10]]
                    cost_base_units, perf_base_units = [cost_units, perf_units]
                    cost_incentives, cost_incentives_meas, \
                        i_units_base, i_units_meas = ("" for n in range(4))

                # Handle special case of commercial heat pumps, where costs
                # may be specified in $/kBtu/h heating or cooling but the
                # measure will apply to both heating and cooling
                # microsegments and possibly also ventilation
                # microsegments; in this case, set the measure cost units
                # to the baseline cost units (while preserving year for
                # each set of units), under the assumption that costs are
                # identical per unit heating vs. per unit cooling, and that
                # costs will be anchored on the measure's heating
                # microsegment(s) (see comment beginning 'Remove double
                # counted stock and stock cost...')
                if any([x in cost_units for x in [
                       '$/kBtu/h heating', '$/kBtu/h cooling']]):
                    if mskeys[4] in ["heating", "cooling"] or (
                        mskeys[4] == "ventilation" and any([
                            x in ms_lists[3] for x in [
                            "heating", "cooling"]])):
                        # Separate measure and baseline cost units into
                        # the year vs. everything else
                        cost_meas_units_unpack, cost_base_units_unpack = [
                            re.search(r'(\d*)(.*)', x) for x in [
                                cost_units, cost_base_units]]
                        # Measure units preserve measure cost year but
                        # otherwise switch to baseline cost units
                        cost_units = cost_meas_units_unpack.group(1) + \
                            cost_base_units_unpack.group(2)
                    else:
                        raise ValueError(
                            "Cost units of '" + cost_units +
                            "' are incompatible with definition of "
                            "measure '" + self.name +
                            "'; check definition")

                # Convert user-defined measure cost units to align with
                # baseline units, given input stock and cost conversion data
                if mskeys[0] == "primary" and not no_stk_mseg and \
                        cost_base_units != cost_units:
                    # Case where measure cost has not yet been recast
                    # across AEO years
                    if not isinstance(cost_meas, dict):
                        cost_meas, cost_units, cost_base_units = \
                            self.convert_costs(
                                convert_data, bldg_sect, mskeys, cost_meas,
                                cost_units, cost_base_units, opts.verbose)
                    # Case where measure cost has been recast across AEO
                    # years
                    else:
                        # Loop through all AEO years by which measure cost
                        # data are broken out and make the conversion
                        for yr in self.handyvars.aeo_years:
                            cost_meas[yr], cost_units, cost_base_units = \
                                self.convert_costs(
                                    convert_data, bldg_sect, mskeys,
                                    cost_meas[yr], cost_units, cost_base_units,
                                    opts.verbose)
                    cost_converts += 1

                # Handle special case where residential cost units in $/ft^2
                # floor must be converted to a per household basis for
                # compatibility with other res. equipment measure types
                if bldg_sect == "residential" and (sqft_subst == 1 or (
                        sqft_subst != 1 and "$/ft^2 floor" in cost_units)):
                    # Key for pulling sf to household (assumed synonymous with
                    # "unit" for envelope cases) conversion data
                    sf_to_house_key = str((mskeys[1], mskeys[2]))
                    # Check if data were already pulled for current
                    # combination of climate/building type; if not,
                    # pull and store the data
                    if sf_to_house_key not in self.sf_to_house.keys():
                        # Conversion from $/sf to $/home divides number of
                        # homes in a given climate/res. building type by the
                        # total square footage of those homes; multiply by 1M
                        # given EIA convention of reporting in Msf
                        self.sf_to_house[sf_to_house_key] = {
                            yr: mseg_sqft_stock["total homes"][yr] / (
                                mseg_sqft_stock["total square footage"][yr] *
                                1000000) for yr in self.handyvars.aeo_years}
                    # Convert measure costs to $/home (assumed synonymous with
                    # $/unit for envelope cases); handle cases where measure
                    # cost is separated out by year or not
                    try:
                        cost_meas = {
                            yr: cost_meas[yr] / self.sf_to_house[
                                sf_to_house_key][yr]
                            for yr in self.handyvars.aeo_years}
                    except (IndexError, TypeError):
                        cost_meas = {
                            yr: cost_meas / self.sf_to_house[
                                sf_to_house_key][yr]
                            for yr in self.handyvars.aeo_years}
                    # Handle special case of a residential lighting controls
                    # measure where the user has specified the measure costs
                    # in $/ft^2 floor and these costs must be converted to
                    # $/unit; at this point, costs are in $/household and
                    # must be converted to $/unit via units/household RECS data
                    if sqft_subst != 1 and "lighting" in mskeys:
                        cost_meas = {yr: cost_meas[yr] /
                                     self.handyvars.res_lts_per_home[mskeys[2]]
                                     for yr in self.handyvars.aeo_years}
                    # Convert residential envelope baseline costs, which will
                    # be in $/ft^2 floor, to $/home (again, household is
                    # assumed synonymous with "unit" here for envelope
                    # measures)
                    if sqft_subst == 1:
                        cost_base = {
                            yr: cost_base[yr] / self.sf_to_house[
                                sf_to_house_key][yr]
                            for yr in self.handyvars.aeo_years}
                    # Set measure and baseline cost units to $/unit
                    cost_units, cost_base_units = ("$/unit" for n in range(2))
                else:
                    sf_to_house_key = None

                # HVAC equipment measure case where measure contributes to
                # an HVAC/envelope package and is flagged as counterfactual
                # that is used to isolate the envelope portion of the package
                # impacts; set the performance and cost impacts of the measure
                # to zero to facilitate isolation of the envelope impacts
                if opts.pkg_env_sep is True and "(CF)" in self.name:
                    rel_perf = {yr: 1 for yr in self.handyvars.aeo_years}
                    cost_meas = {
                        yr: cost_base[yr] for yr in self.handyvars.aeo_years}
                # For cases where a typical/BAU efficiency
                # measure is being assessed, set the measure performance,
                # cost, and lifetime characteristics to the baseline values
                # (note, this excludes typical/BAU fuel switching measures,
                # which are defined and assessed like normal measures)
                elif agen_ref:
                    # Reset measure performance and cost to track baseline
                    perf_meas = {
                        yr: perf_base[yr] for yr in self.handyvars.aeo_years}
                    # For cost, ensure that measure is not listed as add-on;
                    # if so, added cost is zero
                    if self.measure_type == "add-on":
                        cost_meas = {
                            yr: 0 for
                            yr in self.handyvars.aeo_years}
                    else:
                        cost_meas = {
                            yr: cost_base[yr] for
                            yr in self.handyvars.aeo_years}

                    # Set measure lifetime to track baseline (to remain
                    # consistent with measure lifetime formatting as a
                    # point value, pull data for the first year in the
                    # modeling time horizon under the assumption that
                    # baseline lifetime projections don't change over time)
                    life_meas = life_base[self.handyvars.aeo_years[0]]
                    # Set relative performance to track baseline
                    rel_perf = {yr: 1 for yr in self.handyvars.aeo_years}
                # For all other measures, determine relative performance after
                # checking for consistent baseline/measure performance and cost
                # units, as applicable; make an exception for cases where
                # performance is specified in 'relative savings' units (no
                # explicit check of baseline units needed in this case)
                elif (perf_units == 'relative savings (constant)' or (
                    isinstance(perf_units, list) and perf_units[0] ==
                    'relative savings (dynamic)') or
                    perf_base_units == perf_units) and (
                    mskeys[0] == "secondary" or no_stk_mseg or
                        cost_base_units == cost_units):

                    # Relative performance calculation depends on whether the
                    # performance units are already specified as 'relative
                    # savings' over the baseline technology; if not, the
                    # calculation depends on the technology case (i.e. COP  of
                    # 4 is higher relative performance than a baseline COP 3,
                    # but 1 ACH50 is higher rel. performance than 13 ACH50).
                    # Note that relative performance values are stored in a
                    # dict with keys for each year in the modeling time horizon
                    if perf_units == 'relative savings (constant)' or \
                       (isinstance(perf_units, list) and perf_units[0] ==
                            'relative savings (dynamic)'):
                        # In a commercial lighting case where the relative
                        # savings impact of the lighting change on a secondary
                        # end use (heating/cooling) has not been user-
                        # specified, draw from the "light_scnd_autoperf"
                        # variable to determine relative performance for this
                        # secondary microsegment; in all other cases where
                        # relative savings are directly user-specified in the
                        # measure definition, calculate relative performance
                        # based on the relative savings value
                        if type(perf_meas) != numpy.ndarray and \
                           perf_meas == "Missing (secondary lighting)":
                            rel_perf = light_scnd_autoperf
                        else:
                            # Set the original measure relative savings value
                            # (potentially adjusted via re-baselining)
                            perf_meas_orig = perf_meas
                            # Loop through all years in modeling time horizon
                            # and calculate relative measure performance
                            for yr in self.handyvars.aeo_years:
                                # If relative savings must be adjusted to
                                # account for changes in baseline performance,
                                # scale the relative savings value by the
                                # ratio of current year baseline to that of
                                # an anchor year specified with the measure
                                # performance units
                                if isinstance(perf_units, list):
                                    # Ensure that anchor year is not set to
                                    # a year that is before the beginning of
                                    # the modeling time horizon
                                    if perf_units[1] < int(
                                            self.handyvars.aeo_years[0]):
                                        perf_units[1] = int(
                                            self.handyvars.aeo_years[0])
                                    try:
                                        if perf_base_units not in \
                                            self.handyvars.\
                                                inverted_relperf_list:
                                            perf_meas = 1 - (perf_base[yr] / (
                                                perf_base[str(perf_units[1])] /
                                                (1 - perf_meas_orig)))
                                        else:
                                            perf_meas = 1 - ((
                                                perf_base[str(perf_units[1])] *
                                                (1 - perf_meas_orig)) /
                                                perf_base[yr])
                                    except ZeroDivisionError:
                                        verboseprint(
                                            opts.verbose,
                                            "WARNING: Measure '" + self.name +
                                            "' has baseline " +
                                            "or measure performance of zero;" +
                                            " baseline and measure " +
                                            "performance set equal")
                                    # Ensure that the adjusted relative savings
                                    # fraction is not greater than 1 or less
                                    # than 0 if not originally specified as
                                    # less than 0. * Note: savings will
                                    # initially be specified as less than zero
                                    # in lighting efficiency cases, which
                                    # secondarily increase heating energy use
                                    if type(perf_meas) == numpy.array:
                                        if any(perf_meas > 1):
                                            perf_meas[
                                                numpy.where(perf_meas > 1)] = 1
                                        elif any(perf_meas < 0) and \
                                                all(perf_meas_orig) > 0:
                                            perf_meas[
                                                numpy.where(perf_meas < 0)] = 0
                                    elif type(perf_meas) != numpy.array and \
                                            perf_meas > 1:
                                        perf_meas = 1
                                    elif type(perf_meas) != numpy.array and \
                                            perf_meas < 0 and \
                                            perf_meas_orig > 0:
                                        perf_meas = 0
                                # Calculate relative performance
                                rel_perf[yr] = 1 - perf_meas
                    elif perf_units not in \
                            self.handyvars.inverted_relperf_list:
                        for yr in self.handyvars.aeo_years:
                            # Suppress numpy warnings about ZeroDivision
                            # errors, which are handled explicitly
                            with numpy.errstate(all='ignore'):
                                try:
                                    # In cases where an ECM enhances the
                                    # performance of a baseline technology,
                                    # add the ECM and baseline performance
                                    # values and compare to the baseline
                                    # performance value; otherwise, compare
                                    # the ECM performance
                                    # value to the baseline performance value
                                    if self.measure_type == "add-on":
                                        rel_perf[yr] = (
                                            perf_base[yr] /
                                            (perf_meas + perf_base[yr]))
                                    else:
                                        rel_perf[yr] = (
                                            perf_base[yr] / perf_meas)
                                except ZeroDivisionError:
                                    verboseprint(
                                        opts.verbose,
                                        "WARNING: Measure '" + self.name +
                                        "' has measure performance of zero; " +
                                        "baseline and measure performance set "
                                        + "equal")
                                    rel_perf[yr] = 1
                                # Ensure that relative performance is a finite
                                # number; if not, set to 1
                                if (type(rel_perf[yr]) != numpy.ndarray and (
                                    not numpy.isfinite(rel_perf[yr]))) or (
                                   type(rel_perf[yr]) == numpy.ndarray and
                                   any([not numpy.isfinite(x) for
                                        x in rel_perf[yr]])):
                                    rel_perf[yr] = 1
                    else:
                        for yr in self.handyvars.aeo_years:
                            try:
                                rel_perf[yr] = (perf_meas / perf_base[yr])
                            except ZeroDivisionError:
                                verboseprint(
                                    opts.verbose,
                                    "WARNING: Measure '" + self.name +
                                    "' has baseline performance of zero; " +
                                    "baseline and measure performance set " +
                                    "equal")
                                rel_perf[yr] = 1

                    # If looping through a commercial lighting microsegment
                    # where secondary end use effects (heating/cooling) are not
                    # specified by the user and must be added, store the
                    # relative performance of the efficient lighting equipment
                    # for later use in updating these secondary microsegments
                    if mskeys[4] == "lighting" and mskeys[0] == "primary" and \
                            light_scnd_autoperf is True:
                        light_scnd_autoperf = rel_perf

                    # Check whether the user has optionally locked measure
                    # relative performance across the model time horizon to
                    # that of the measure's market entry year (for example,
                    # a user may wish to assume that the 'Best Available'
                    # measure on the market maintains a consistent degree of
                    # improvement over the comparable baseline technology
                    # across its lifetime). In this case, set the measure's
                    # relative performance value for all years to that
                    # calculated for its market entry year. NOTE: do not
                    # perform calculations on typical/BAU efficiency measures,
                    # as their performance and cost already tracks baseline
                    if opts.rp_persist is True and (
                            opts.add_typ_eff is not True or not agen_ref):
                        # Preclude consistent performance improvements for
                        # prospective measures, which tend to already be at
                        # performance limits
                        if all([x not in self.name for x in [
                                "Prospective", "Emerging", "Target"]]):
                            rel_perf = {
                                yr: rel_perf[str(self.market_entry_year)] for
                                yr in self.handyvars.aeo_years}
                        # If performance is escalated at the baseline rate,
                        # also increase/decrease measure cost at the baseline
                        # rate, anchored on the measure market entry year

                        # Determine baseline cost change rate for the
                        # current microsegment; set to 1 if baseline cost
                        # at market entry is zero
                        cost_chg = {yr: cost_base[yr] / cost_base[
                            str(self.market_entry_year)] if (numpy.isfinite(
                                cost_base[str(self.market_entry_year)]) and
                                cost_base[str(self.market_entry_year)] != 0)
                            else 1 for yr in self.handyvars.aeo_years}
                        # Multiple measure cost by rate of change in baseline
                        # cost for each year in the AEO horizon; handle case
                        # where measure cost has not yet been broken out by
                        # AEO year
                        try:
                            cost_meas = {yr: cost_meas * cost_chg[yr] for yr in
                                         self.handyvars.aeo_years}
                        except TypeError:
                            cost_meas = {yr: cost_meas[str(
                                self.market_entry_year)] * cost_chg[yr] for
                                yr in self.handyvars.aeo_years}
                    else:
                        # If measure performance is not escalated at the
                        # baseline rate, assume no mapping between changes in
                        # baseline cost and measure cost and set measure cost
                        # to the same value across all AEO years
                        if not isinstance(cost_meas, dict):
                            cost_meas = {yr: cost_meas for yr in
                                         self.handyvars.aeo_years}
                    # If incentives data were found, apply them to
                    # baseline and measures tech. costs on the basis of their
                    # performance levels
                    if cost_incentives:
                        # Check for and attempt to address inconsistencies
                        # between the performance units attached to baseline
                        # and measure cost incentives and the performance units
                        # attached to the baseline/measure equipment data

                        # Set the baseline/measure performance units to
                        # check incentives performance units against
                        i_unit_chk = [perf_base_units, perf_units]
                        # Initialize list of unit conversion factors to
                        # be applied below to address inconsistencies
                        i_unit_cnv = [None for n in range(2)]
                        # Loop through baseline/measure units and attempt
                        # to pull factors to address inconsistencies
                        for ind_u, i_unit in enumerate([
                                i_units_base, i_units_meas]):
                            # No inconsistency; set conversion factor to 1
                            if not i_unit or (i_unit == i_unit_chk[ind_u]):
                                i_unit_cnv[ind_u] = 1
                            # Inconsistency that can be address via units
                            # conversion
                            elif i_unit != i_unit_chk[ind_u] and i_unit in \
                                    self.handyvars.tech_units_map.keys():
                                # Find units conversion factor
                                i_unit_cnv[ind_u] = \
                                    self.handyvars.tech_units_map[
                                    i_unit][i_unit_chk[ind_u]]
                                # Set incentive performance units to those
                                # of the baseline/measure performance units
                                # being checked against
                                i_unit = i_unit_chk
                        # Set final baseline and measure incentives units
                        # conversions to be applied to incentives data below
                        cnv_b, cnv_m = [i_unit_cnv[0], i_unit_cnv[1]]

                        # Check to ensure that performance units for
                        # incentives are harmonized, and if not, do not
                        # apply incentives/issue warning
                        if all([x is not None for x in [cnv_b, cnv_m]]):
                            # Initialize incentives multipliers as 1
                            mlt_b, mlt_m = (1 for n in range(2))
                            # Multiply any existing residential HP heating
                            # incentives by 2 while zeroing out existing
                            # residential HP cooling incentives to correct for
                            # EIA convention of splitting incentives between
                            # the two (Scout puts all costs in heating)
                            if bldg_sect == "residential" and \
                                    mskeys[-1] == "existing":
                                # Baseline mseg incentives multiplier
                                if mskeys[-2] is not None and \
                                        "HP" in mskeys[-2]:
                                    if mskeys[4] == "heating":
                                        mlt_b = 2
                                    else:
                                        mlt_b = 0
                                # Measure "switch to" mseg incentives
                                # multiplier (if applicable)
                                if mskeys_swtch and \
                                        mskeys_swtch[-2] is not None and \
                                        "HP" in mskeys_swtch[-2]:
                                    if mskeys_swtch[4] == "heating":
                                        mlt_m = 2
                                    else:
                                        mlt_m = 0

                            # Loop through all years and find/apply incentives
                            for yr in self.handyvars.aeo_years:
                                # Handle inverted performance units (lower
                                # is better performance/more incentives)
                                if perf_base_units not in \
                                        self.handyvars.inverted_relperf_list:
                                    # Performance levels greater/equal to
                                    # minimum (e.g., COP) get the incentive

                                    # Handle case where measure switches from
                                    # baseline segment (and thus uses different
                                    # incentives information)
                                    if not cost_incentives_meas:
                                        base_val_yr, meas_val_yr = ([
                                            x[1] * mlt_b * cnv_b for x in
                                            cost_incentives[yr] if y >= x[0]]
                                            for y in [
                                                perf_base[yr], perf_meas])
                                    else:
                                        base_val_yr = [
                                            x[1] * mlt_b * cnv_b for x in
                                            cost_incentives[yr] if
                                            perf_base[yr] >= x[0]]
                                        meas_val_yr = [
                                            x[1] * mlt_m * cnv_m for x in
                                            cost_incentives_meas[yr]
                                            if perf_meas >= x[0]]
                                else:
                                    # Performance levels less than/equal to
                                    # maximum (e.g. kwh/yr) get the incentive

                                    # Handle case where measure switches from
                                    # baseline segment (and thus uses different
                                    # incentives information)
                                    if not cost_incentives_meas:
                                        base_val_yr, meas_val_yr = (
                                            [x[1] * mlt_b * cnv_b for x in
                                             cost_incentives[yr] if y <= x[0]]
                                            for y in [
                                                perf_base[yr], perf_meas])
                                    else:
                                        base_val_yr = [
                                            x[1] * mlt_b * cnv_b for x in
                                            cost_incentives[yr]
                                            if perf_base[yr] <= x[0]]
                                        meas_val_yr = [
                                            x[1] * mlt_m * cnv_m for x in
                                            cost_incentives_meas[yr]
                                            if perf_meas <= x[0]]
                                # Apply the largest retrieved incentive level
                                # to baseline and measure installed costs
                                if len(base_val_yr) > 0:
                                    cost_base[yr] -= max(base_val_yr)
                                    # Ensure that baseline costs after
                                    # incentives are never negative
                                    if cost_base[yr] <= 0:
                                        cost_base[yr] = 0
                                if len(meas_val_yr) > 0:
                                    cost_meas[yr] -= max(meas_val_yr)
                                    # Ensure that measure costs after
                                    # incentives are never negative
                                    if cost_meas[yr] <= 0:
                                        cost_meas[yr] = 0
                                    meas_incent_flag = True
                        else:
                            # Record suppression of incentives application
                            suppress_incent.append(mskeys)
                            pass
                else:
                    raise KeyError(
                        "Invalid performance or cost units for ECM '" +
                        self.name + "'")

                # Set stock turnover info. and consumer choice info. to
                # appropriate building type
                if bldg_sect == "residential":
                    # Update new building construction information
                    for yr in self.handyvars.aeo_years:
                        # Find new and total buildings for current year
                        new_constr["annual new"][yr] = \
                            mseg_sqft_stock["new homes"][yr]
                        new_constr["total"][yr] = \
                            mseg_sqft_stock["total homes"][yr]

                    # Update technology choice parameters needed to choose
                    # between multiple efficient technology options that
                    # access this baseline microsegment. For the residential
                    # sector, these parameters are found in the baseline
                    # technology cost, performance, and lifetime JSON
                    if mskeys[0] == "secondary":
                        choice_params = {}  # No choice params for 2nd msegs
                    else:
                        # Use try/except to handle cases with missing
                        # or invalid consumer choice data (where choice
                        # parameter values of 0 or "NA" are invalid)
                        try:
                            if any([x[1] in [0, "NA"] for x in base_cpl[
                                "consumer choice"]["competed market share"][
                                    "parameters"]["b1"].items()]):
                                raise ValueError
                            choice_params = {
                                "b1": {
                                    key: base_cpl["consumer choice"][
                                        "competed market share"]["parameters"][
                                        "b1"][yr] for key in
                                    self.handyvars.aeo_years},
                                "b2": {
                                    key: base_cpl["consumer choice"][
                                        "competed market share"]["parameters"][
                                        "b2"][yr] for key in
                                    self.handyvars.aeo_years}}
                            # Add to count of primary microsegment key chains
                            # with valid consumer choice data
                            valid_keys_consume += 1
                        # Update invalid consumer choice parameters
                        except (ValueError, TypeError, KeyError):
                            # Record missing consumer data for primary
                            # technologies; if in verbose mode and the user
                            # has not already been warned about missing data
                            # for the given technology, print warning message
                            if mskeys[0] == "primary":
                                if mskeys[4] not in consume_warn:
                                    consume_warn.append(mskeys[4])
                                    verboseprint(
                                        opts.verbose,
                                        "WARNING: ECM '" + self.name +
                                        "' missing valid consumer choice "
                                        "data for end use '" + str(mskeys[4]) +
                                        "'; using default choice data for " +
                                        "refrigeration end use")
                            choice_params = {
                                "b1": {
                                    key: self.handyvars.deflt_choice[0] for
                                    key in self.handyvars.aeo_years},
                                "b2": {
                                    key: self.handyvars.deflt_choice[1] for
                                    key in self.handyvars.aeo_years}}
                else:
                    # Note: unspecified building type does not have any
                    # square footage data for new/existing splits
                    if mskeys[2] != "unspecified":
                        # Update new building construction information
                        for yr in self.handyvars.aeo_years:
                            # Find new and total square footage for current yr.
                            new_constr["annual new"][yr] = \
                                mseg_sqft_stock["new square footage"][yr]
                            new_constr["total"][yr] = \
                                mseg_sqft_stock["total square footage"][yr]

                    # Update technology choice parameters needed to choose
                    # between multiple efficient technology options that
                    # access this baseline microsegment. For the commercial
                    # sector, these parameters are specified at the
                    # beginning of run.py in com_timeprefs (* Note:
                    # com_timeprefs info. may eventually be integrated into
                    # the  baseline technology cost, performance, and
                    # lifetime JSON as for residential)
                    if mskeys[0] == "secondary":
                        choice_params = {}  # No choice params for 2nd msegs
                    else:
                        # Use try/except to handle cases with missing
                        # consumer choice data
                        try:
                            choice_params = {"rate distribution":
                                             self.handyvars.com_timeprefs[
                                                 "distributions"][mskeys[4]]}
                            # Add to count of primary microsegment key chains
                            # with valid consumer choice data
                            valid_keys_consume += 1
                        except KeyError:
                            # Record missing consumer data for primary
                            # technologies; if in verbose mode and the user
                            # has not already been warned about missing data
                            # for the given technology, print warning message
                            if mskeys[4] not in consume_warn:
                                consume_warn.append(mskeys[4])
                                verboseprint(
                                    opts.verbose,
                                    "WARNING: ECM '" + self.name +
                                    "' missing valid consumer choice data " +
                                    "for end use '" + str(mskeys[4]) +
                                    "'; using default choice data for " +
                                    "refrigeration end use")
                            choice_params = {"rate distribution":
                                             self.handyvars.com_timeprefs[
                                                 "distributions"][
                                                 "refrigeration"]}

                # Find fraction of total new buildings in each year.
                # Note: in each year, this fraction is calculated by summing
                # the annual new building/floor space figures for all
                # preceding years. Handle unspecified case, where no such
                # floor space data exist (put all data under existing msegs)
                if mskeys[2] != "unspecified":
                    for yr in self.handyvars.aeo_years:
                        # Find cumulative total of new building/floor space
                        if yr == self.handyvars.aeo_years[0]:
                            new_constr["total new"][yr] = \
                                new_constr["annual new"][yr]
                        else:
                            # Handle case where data for previous year are
                            # unavailable; set to current year's data
                            try:
                                new_constr["total new"][yr] = \
                                    new_constr["annual new"][yr] + \
                                    new_constr["total new"][str(int(yr) - 1)]
                            except KeyError:
                                new_constr["total new"][yr] = \
                                    new_constr["annual new"][yr]
                        # Calculate new vs. existing fraction of stock
                        if new_constr["total new"][yr] <= new_constr[
                                "total"][yr]:
                            new_constr["new fraction"][yr] = \
                                new_constr["total new"][yr] / \
                                new_constr["total"][yr]
                        else:
                            new_constr["new fraction"][yr] = 1
                else:
                    new_constr["new fraction"] = {
                        yr: 0 for yr in self.handyvars.aeo_years}

                # Determine the fraction to use in scaling down the stock,
                # energy, and carbon microsegments to the applicable structure
                # type indicated in the microsegment key chain (e.g., new
                # structures or existing structures)
                if mskeys[-1] == "new":
                    new_existing_frac = {key: val for key, val in
                                         new_constr["new fraction"].items()}
                else:
                    new_existing_frac = {key: (1 - val) for key, val in
                                         new_constr["new fraction"].items()}

                # Update bass diffusion parameters needed to determine the
                # fraction of the baseline microsegment that will be captured
                # by efficient alternatives to the baseline technology
                # (* BLANK FOR NOW, WILL CHANGE IN FUTURE *)
                diffuse_params = None
                # diffuse_params = base_cpl["consumer choice"][
                #    "competed market"]["parameters"]

                # For primary lighting microsegments with secondary effects,
                # initialize a variable used to track the total lighting
                # energy for the microsegment's climate zone, building
                # type, and structure type
                if mskeys[4] == "lighting" and (
                        not opts or opts.no_scnd_lgt is not True) and \
                        self.end_use["secondary"] is not None:
                    # Initialize total lighting energy
                    energy_total_scnd = True
                else:
                    energy_total_scnd = False

                # Update total stock, energy use, and carbon emissions for the
                # current contributing microsegment. Note that secondary
                # microsegments make no contribution to the stock calculation,
                # as they only affect energy/carbon and associated costs, and
                # that in instances where no stock data have been pulled,
                # stocks should be set to zero.

                # Total stock
                if mskeys[0] == 'secondary':
                    add_stock = dict.fromkeys(self.handyvars.aeo_years, 0)
                elif sqft_subst == 1:  # Use ft^2 floor area in lieu of # units
                    # For residential envelope microsegments, stock is
                    # converted to a per house (per "unit") basis to facilitate
                    # comparison and packaging with res. equipment measures;
                    # the required conversion factor was already calculated
                    # above and applied to the stock costs for these
                    # microsegments, and is reused here for the stock
                    if sf_to_house_key and sf_to_house_key in \
                            self.sf_to_house.keys():
                        add_stock = {
                            key: val * new_existing_frac[key] *
                            self.sf_to_house[sf_to_house_key][key] *
                            1000000 for key, val in mseg_sqft_stock[
                                "total square footage"].items()
                            if key in self.handyvars.aeo_years}
                    else:
                        add_stock = {
                            key: val * new_existing_frac[key] * 1000000 for
                            key, val in mseg_sqft_stock[
                                "total square footage"].items()
                            if key in self.handyvars.aeo_years}
                elif not no_stk_mseg:  # Check stock (units/service) data exist
                    add_stock = {
                        key: val * new_existing_frac[key] for key, val in
                        mseg["stock"].items() if key in
                        self.handyvars.aeo_years}
                else:  # If no stock data exist, set stock to zero
                    add_stock = {
                        key: 0 for key in self.handyvars.aeo_years}

                # If the baseline technology is a heat pump in the residential
                # sector, account for the fact that EIA divides all heat pump
                # stocks by 2 when separately considered across the heating
                # and cooling services; note that per unit stock costs for
                # these technologies are treated in the same way by EIA and
                # were also multiplied by 2 above
                if bldg_sect == "residential" and (
                        mskeys[-2] is not None and "HP" in mskeys[-2]):
                    add_stock = {yr: add_stock[yr] * 2 for
                                 yr in self.handyvars.aeo_years}

                # Total energy use
                add_energy = {
                    key: val * site_source_conv_base[key] *
                    new_existing_frac[key] for key, val in mseg[
                        "energy"].items() if key in
                    self.handyvars.aeo_years}
                # Total lighting energy use for climate zone, building type,
                # and structure type of current primary lighting
                # microsegment (used to adjust secondary effects)
                if energy_total_scnd is True:
                    energy_total_scnd = self.find_scnd_overlp(
                        new_existing_frac, site_source_conv_base, reduce(
                            operator.getitem, mskeys[1:5], msegs),
                        energy_tot=dict.fromkeys(
                            self.handyvars.aeo_years, 0))
                # Total carbon emissions
                add_carb = {key: val * intensity_carb_base[key]
                            for key, val in add_energy.items()
                            if key in self.handyvars.aeo_years}

                # If applicable, determine fugitive emissions from
                # supply chain methane leakage and refrigerants

                # Calculate fugitive emissions from methane leakage
                # for cases where baseline fuel is natural gas. Map
                # the current mseg region to the regionality of the
                # fugitive emissions methane leakage data (state breakouts)

                # Non-state region setting must be mapped to state
                if opts.fugitive_emissions is not False and \
                    opts.fugitive_emissions[0] in ['1', '3'] and (
                        opts.alt_regions != "State" and
                        mskeys[3] == "natural gas"):
                    # Prepare fractions needed to map state-resolved
                    # fugitive methane data to current region
                    try:
                        reg_weight_row = [r_i for r_i, r in enumerate(
                            self.handyvars.fugitive_emissions_map) if
                            r[0] == mskeys[1]][0]
                    except (IndexError):
                        raise ValueError("No fugitive emissions state-level \
                            mapping data found for region '" + mskeys[1] + "'")
                    reg_weight = self.handyvars.fugitive_emissions_map[
                        reg_weight_row]
                    # Apply mapping fractions to develop methane leakage rate
                    try:
                        lkg_rate = sum([self.handyvars.fug_emissions[
                            "methane"]["total_leakage_rate"][state] *
                            reg_weight[state] for state in
                            self.handyvars.fug_emissions[
                            "methane"]["total_leakage_rate"].keys()])
                    except (KeyError):
                        raise ValueError(
                            "Inconsistent state keys "
                            "between fugitive emissions leakage rate data "
                            "and geographical mapping data")
                    # Handle case where measure is switching away from
                    # a baseline case with methane leakage to a non-gas tech.
                    # without such leakage
                    if self.fuel_switch_to is not None:
                        lkg_fmeth_base = copy.deepcopy(lkg_rate)
                        lkg_fmeth_meas = 0
                    else:
                        lkg_fmeth_base, lkg_fmeth_meas = (
                            copy.deepcopy(lkg_rate) for n in range(2))
                # State region setting requires no further mapping
                elif opts.fugitive_emissions is not False and \
                    opts.fugitive_emissions[0] in ['1', '3'] and \
                        mskeys[3] == "natural gas":
                    # Directly pull methane leakage rate for current state
                    lkg_rate = self.handyvars.fug_emissions[
                        "methane"]["total_leakage_rate"][mskeys[1]]
                    # Handle case where measure is switching away from
                    # a baseline case with methane leakage to a non-gas tech.
                    # without such leakage
                    if self.fuel_switch_to is not None:
                        lkg_fmeth_base = copy.deepcopy(lkg_rate)
                        lkg_fmeth_meas = 0
                    else:
                        lkg_fmeth_base, lkg_fmeth_meas = (
                            copy.deepcopy(lkg_rate) for n in range(2))
                else:
                    lkg_rate, lkg_fmeth_base, lkg_fmeth_meas = (
                        None for n in range(3))

                # Calculate fugitive emissions from refrigerants
                # for cases where supporting refrigerant leakage data are
                # available for the current mseg building/technology type
                if opts.fugitive_emissions is not False and \
                        opts.fugitive_emissions[0] in ['2', '3']:
                    # Set building type name (residential/commercial) to key
                    # refrigerant data
                    bldg_name_chk = copy.deepcopy(bldg_sect)

                    # Set technology names to key baseline and efficient case
                    # refrigerant data for the current mseg; set name
                    # separately for measure and comparable baseline
                    # technology

                    # Thermal end use technology cases
                    if mskeys[4] in ["heating", "cooling", "secondary heating",
                                     "water heating"]:
                        # Flag for HP measure (requires special handling)
                        hp_flag = ((
                            self.tech_switch_to is not None and
                            "HP" in self.tech_switch_to) or any([
                                x in self.name for x in [
                                    "HP", "heat pump", "Heat Pump"]]))

                        # Measure is HP for heating/cooling
                        if hp_flag and mskeys[4] in [
                                "heating", "cooling", "secondary heating"]:
                            # Set measure refrigerant data to that of
                            # a GSHP or ASHP based on measure name
                            if any([x in self.name for x in [
                                    "GSHP", "Ground", "ground"]]):
                                if bldg_sect == "residential":
                                    tech_name_chk_e = "GSHP"
                                else:
                                    tech_name_chk_e = "comm_GSHP-cool"
                            else:
                                # Residential case; assume switching to
                                # air source heat pump
                                if bldg_sect == "residential":
                                    tech_name_chk_e = "ASHP"
                                # Commercial case; assume switching to
                                # air source heat pump for small
                                # commercial HVAC, and water/ground
                                # source HP for large
                                else:
                                    # Small commercial
                                    if mskeys[-2] in \
                                            self.handyvars.com_RTU_fs_tech:
                                        tech_name_chk_e = \
                                            "rooftop_ASHP-cool"
                                    # Large commercial
                                    else:
                                        tech_name_chk_e = "comm_GSHP-cool"
                            # Set baseline refrigerant data to that of the
                            # measure in the case of a like-for-like heat
                            # pump replacement or the case of switching to
                            # a HP from another baseline heating technology
                            if (mskeys[-2] is not None and
                                "HP" in mskeys[-2]) \
                                    or "cooling" not in mskeys:
                                tech_name_chk_b = copy.deepcopy(
                                    tech_name_chk_e)
                                # Given like-for-like HP replacement, anchor
                                # baseline and measure refrigerant emissions
                                # on the cooling end use and set flag to zero
                                # out baseline and measure refrigerant
                                # emissions for the heating end use
                                # (to avoid double counting of refrigerants
                                # since HP stock units are processed for each
                                # of the heating and cooling end uses)
                                if (mskeys[-2] is not None and
                                        "HP" in mskeys[-2]):
                                    if mskeys[4] == "cooling":
                                        zero_b_r_flag, zero_m_r_flag = (
                                            False for n in range(2))
                                    else:
                                        zero_b_r_flag, zero_m_r_flag = (
                                            True for n in range(2))

                                # Given switch to HP from another baseline
                                # heating technology, set flag to zero out
                                # baseline refrigerant emissions (since they
                                # do not occur in baseline)
                                else:
                                    zero_b_r_flag = True
                                    zero_m_r_flag = False
                            # Set baseline refrigerant data to that of the
                            # baseline technology name in the case of a
                            # switch to HP from another baseline cooling tech.
                            else:
                                tech_name_chk_b = mskeys[-2]
                                # Given switch to HP from another baseline
                                # cooling technology, set flag to zero out
                                # measure refrigerant emissions (since they
                                # are assessed for the heating stock portion of
                                # the HP technology and thus would otherwise
                                # be double counted)
                                zero_b_r_flag = False
                                zero_m_r_flag = True
                        # Measure is HP for water heating
                        elif hp_flag and mskeys[4] in "water heating":
                            # Set measure refrigerant data to that of a HPWH,
                            # names differ by bldg. typ.
                            if bldg_sect == "residential":
                                tech_name_chk_e = "HPWH"
                            else:
                                tech_name_chk_e = "HP water heater"
                            # Set baseline refrigerant data to that of the
                            # measure in the case of a like-for-like heat
                            # pump WH replacement or the case of switching to
                            # a HPWH from another baseline WH technology (e.g.,
                            # in all possible cases)
                            tech_name_chk_b = copy.deepcopy(tech_name_chk_e)
                            # Given like-for-like HPWH replacement, do not
                            # set flag to zero out baseline refrigerant
                            # emissions (since they occur in baseline too)
                            if (mskeys[-2] is not None and
                                    "HP" in mskeys[-2]):
                                tech_name_chk_b = copy.deepcopy(
                                    tech_name_chk_e)
                                zero_b_r_flag, zero_m_r_flag = (
                                        False for n in range(2))
                            # Given switch to HPWH from another baseline
                            # heating technology, set flag to zero out
                            # baseline refrigerant emissions (since they
                            # do not occur in baseline)
                            else:
                                zero_b_r_flag = True
                                zero_m_r_flag = False
                        # In all other cases, assume measure tech. follows
                        # baseline tech., accounting for special handling of
                        # like-for-like HP replacements to avoid double
                        # counting across heating and cooling stock (this
                        # handles a case where a measure applies to a HP but
                        # hasn't been flagged as a HP measure above based on
                        # its name)
                        else:
                            tech_name_chk_b, tech_name_chk_e = (
                                mskeys[-2] for n in range(2))
                            # Given like-for-like HP replacement, anchor
                            # baseline and measure refrigerant emissions
                            # on the cooling end use and set flag to zero
                            # out baseline and measure refrigerant
                            # emissions for the heating end use
                            # (to avoid double counting of refrigerants
                            # since HP stock units are processed for each
                            # of the heating and cooling end uses)
                            if (mskeys[-2] is not None and
                                    "HP" in mskeys[-2]):
                                if mskeys[4] == "cooling":
                                    zero_b_r_flag, zero_m_r_flag = (
                                        False for n in range(2))
                                else:
                                    zero_b_r_flag, zero_m_r_flag = (
                                        True for n in range(2))
                            else:
                                zero_b_r_flag, zero_m_r_flag = (
                                    False for n in range(2))
                    # Refrigeration measure; special handling needed
                    # for commercial
                    elif mskeys[4] == "refrigeration":
                        # Residential case; use mseg technology name as-is
                        # to key in refrigerants data; pull identical
                        # refrigerant data for baseline and efficient cases
                        if bldg_name_chk == "residential":
                            tech_name_chk_b, tech_name_chk_e = (
                                mskeys[-2] for n in range(2))
                        # Commercial case; centralized refrigeration is
                        # anchored on supermarket display cases (compressor
                        # racks/condensers are included in EIA's display case
                        # estimates to avoid double counting); all other
                        # commercial refrigeration equipment has dedicated
                        # data in the refigerants input file
                        else:
                            if mskeys[-2] == \
                                    "Commercial Supermarket Display Cases":
                                tech_name_chk_b, tech_name_chk_e = (
                                    "Centralized refrigeration" for
                                    n in range(2))
                            elif mskeys[-2] in [
                                "Commercial Walk-In Freezers",
                                "Commercial Walk-In Refrigerators",
                                "Commercial Reach-In Refrigerators",
                                "Commercial Reach-In Freezers",
                                "Commercial Ice Machines",
                                "Commercial Beverage Merchandisers",
                                "Commercial Refrigerated Vending Machines"
                                    ]:
                                tech_name_chk_b, tech_name_chk_e = (
                                    mskeys[-2] for n in range(2))
                            else:
                                raise ValueError(
                                    "Unexpected commercial refrigeration "
                                    "technology encountered for segment " +
                                    mskeys)
                        zero_b_r_flag, zero_m_r_flag = (
                            False for n in range(2))
                    else:
                        tech_name_chk_b, tech_name_chk_e = (
                            mskeys[-2] for n in range(2))
                        zero_b_r_flag, zero_m_r_flag = (
                            False for n in range(2))

                    # Attempt to pull refrigerant data dict for the base case
                    # technology; if no data available set to None
                    try:
                        f_refr_b = self.handyvars.fug_emissions[
                            "refrigerants"]["refrigerant_data_by_tech_type"][
                                bldg_name_chk][tech_name_chk_b]
                    except KeyError:
                        f_refr_b = None
                    # Attempt to pull refrigerant data dict for the measure
                    # case; if no data available set to None
                    try:
                        f_refr_e = self.handyvars.fug_emissions[
                            "refrigerants"]["refrigerant_data_by_tech_type"][
                                bldg_name_chk][tech_name_chk_e]
                    except KeyError:
                        f_refr_e = None
                else:
                    f_refr_e, f_refr_b = (None for n in range(2))
                    zero_b_r_flag, zero_m_r_flag = (False for n in range(2))

                # Finalize dict of supporting fugitive refrigerant emissions
                # data and flags for the baseline and measure cases
                if any([x is not None for x in [f_refr_b, f_refr_e]]):
                    f_refr = {"baseline": [f_refr_b, zero_b_r_flag],
                              "efficient": [f_refr_e, zero_m_r_flag]}
                else:
                    f_refr = None

                # Calculate fugitive emissions totals by year for supply
                # chain methane leakage. If methane leakage assessment is
                # desired and measure applies to natural gas microsegments,
                # all microsegments for that measure are prepared with
                # fugitive methane emissions data in their output dict (
                # including dicts of zeros for any non-NG msegs the measure
                # applies to. If no assessment of fugitive methane emissions
                # is desired, set to None
                if opts.fugitive_emissions is not False and \
                    opts.fugitive_emissions[0] in ['1', '3'] and mskeys[3] == \
                        "natural gas":
                    # Create variables for converting natural gas energy
                    # to volume and mass
                    mmbtu_to_mcf = 0.0009643202
                    methane_gram_per_mcf = 20200
                    methane_100yr_GWP = 28
                    mmt_conv = 1 / 1000000000
                    add_fmeth = {key: val * mmbtu_to_mcf * lkg_rate *
                                 methane_gram_per_mcf * methane_100yr_GWP *
                                 mmt_conv for key, val in
                                 add_energy.items()}
                elif opts.fugitive_emissions is not False and \
                        opts.fugitive_emissions[0] in ['1', '3']:
                    add_fmeth = {key: 0 for key in add_energy.keys()}
                else:
                    add_fmeth = None

                # Check for time-sensitive efficiency valuation (e.g., a
                # measure has time sensitive features and/or the user has
                # optionally specified time sensitive output metrics or sector-
                # level load shapes for the given year). If this type of
                # valuation is necessary and the current microsegment pertains
                # to electricity, develop the factors needed to reweight
                # energy/cost/carbon data to reflect hourly changes in energy
                # load, energy price, and average carbon emissions across the
                # desired annual or sub-annual time horizon; also pull hourly
                # fraction of annual load data needed to calculate sector-level
                # shapes below if the user has specified the '--sect_shapes'
                # option and simulation is in a year where such shapes are
                # desired (per self.handyvars.aeo_years_summary attribute).
                if (opts.tsv_metrics is not False or
                    self.tsv_features is not None or (
                        calc_sect_shapes is True and
                        yr in self.handyvars.aeo_years_summary)) and (
                        mskeys[0] == "primary" and (
                            (mskeys[3] == "electricity") or
                            (self.fuel_switch_to == "electricity"))):
                    # Set the appropriate TSV data to use for the current
                    # microsegment; when assuming a high grid decarbonization
                    # case, the user can choose to assess time sensitive
                    # emissions and cost factors in non-fuel switching
                    # microsegments using base-case data (e.g., before
                    # additional grid decarbonization)
                    if opts.grid_decarb is not False and \
                        tsv_data_nonfs is not None and (
                            self.fuel_switch_to is None or (
                            self.fuel_switch_to == "electricity" and
                            "electricity" in mskeys)):
                        tsv_data = tsv_data_nonfs
                    else:
                        tsv_data = tsv_data_init

                    # Pull TSV scaling fractions and shapes
                    tsv_scale_fracs, tsv_shapes = self.gen_tsv_facts(
                        tsv_data, mskeys, bldg_sect, convert_data, opts,
                        cost_energy_meas)
                else:
                    tsv_scale_fracs = {
                        "energy": {"baseline": 1, "efficient": 1},
                        "cost": {"baseline": 1, "efficient": 1},
                        "carbon": {"baseline": 1, "efficient": 1}}
                    tsv_shapes = None

                # After calculations are complete, if measure fuel switches
                # from fossil to electricity and the current baseline
                # microsegment indicates fossil fuel, set all 8760 baseline
                # values to zero (it is assumed that sector-level load shapes
                # will only be relevant to the electricity system, which will
                # not see a baseline load that is satisfied by non-electric
                # fuel – e.g. in this view, measures that fuel switch add
                # load to the electricity system that wasn't there before)
                if self.fuel_switch_to == "electricity" and \
                        "electricity" not in mskeys:
                    # Set baseline load shapes to zero in cases where these
                    # fractions have been calculated
                    if calc_sect_shapes is True and tsv_shapes is not None:
                        tsv_shapes["baseline"] = [0 for x in range(8760)]
                    # Set baseline TSV scaling fractions to 1
                    for x in ["energy", "cost", "carbon"]:
                        tsv_scale_fracs[x]["baseline"] = 1
                for adopt_scheme in self.handyvars.adopt_schemes:
                    # Update total, competed, and efficient stock, energy,
                    # carbon and baseline/measure cost info. based on adoption
                    # scheme
                    [add_stock_total, add_energy_total, add_carb_total,
                     add_fmeth_total, add_frefr_total, add_stock_total_meas,
                     add_energy_total_eff, add_carb_total_eff,
                     add_fmeth_total_eff, add_frefr_total_eff,
                     add_stock_compete, add_energy_compete, add_carb_compete,
                     add_fmeth_compete, add_frefr_compete,
                     add_stock_compete_meas, add_energy_compete_eff,
                     add_carb_compete_eff, add_fmeth_compete_eff,
                     add_frefr_compete_eff, add_stock_cost, add_energy_cost,
                     add_carb_cost, add_stock_cost_meas, add_energy_cost_eff,
                     add_carb_cost_eff, add_stock_cost_compete,
                     add_energy_cost_compete, add_carb_cost_compete,
                     add_stock_cost_compete_meas, add_energy_cost_compete_eff,
                     add_carb_cost_compete_eff, add_fs_energy_eff_remain,
                     add_fs_carb_eff_remain, add_fs_energy_cost_eff_remain,
                     mkt_scale_frac_fin, warn_list] = \
                        self.partition_microsegment(
                            adopt_scheme, diffuse_params, mskeys, bldg_sect,
                            sqft_subst, mkt_scale_frac, new_constr, add_stock,
                            add_energy, add_carb, add_fmeth, f_refr,
                            cost_base, cost_meas, cost_energy_base,
                            cost_energy_meas, rel_perf, life_base, life_meas,
                            site_source_conv_base, site_source_conv_meas,
                            intensity_carb_base, intensity_carb_meas,
                            energy_total_scnd, tsv_scale_fracs, tsv_shapes,
                            opts, contrib_mseg_key, ctrb_ms_pkg_prep, hp_rate,
                            retro_rate_mseg, calc_sect_shapes, lkg_fmeth_base,
                            lkg_fmeth_meas, warn_list)

                    # Remove minor HVAC equipment stocks in cases where major
                    # HVAC tech. is also covered by the measure definition, as
                    # well as double counted stock and stock cost for equipment
                    # measures that apply to more than one end use that
                    # includes heating or cooling. In the latter cases, anchor
                    # stock/cost on on heating end use tech.,
                    # provided heating is included, because they are
                    # generally of greatest interest for the stock of measures
                    # like ASHPs and span fuels (e.g., electric resistance, gas
                    # furnace, oil furnace, etc.). If heating is not covered,
                    # anchor on the cooling end use technologies. This
                    # adjustment covers all measures that apply across
                    # heating/cooling (and possibly other) end uses
                    if sqft_subst != 1 and ((
                        # Multiple techs. including minor HVAC tech.
                        any([x in mskeys for x in
                             self.handyvars.minor_hvac_tech]) and not
                        all([x in self.handyvars.minor_hvac_tech for
                             x in self.technology["primary"]])) or (
                        # Multiple end uses
                        len(ms_lists[3]) > 1 and ((
                            "heating" in ms_lists[3] and
                            "heating" not in mskeys) or (
                            "heating" not in ms_lists[3] and
                            "cooling" in ms_lists[3] and
                                "cooling" not in mskeys)))):
                        add_stock_total, add_stock_compete, \
                            add_stock_total_meas, add_stock_compete_meas, \
                            add_stock_cost, add_stock_cost_compete, \
                            add_stock_cost_meas, \
                            add_stock_cost_compete_meas = ({
                                yr: 0 for yr in self.handyvars.aeo_years}
                                for n in range(8))

                    # Combine stock/energy/carbon/cost/lifetime updating info.
                    # into a dict. Note that baseline lighting lifetimes are
                    # adjusted by the stock of the contributing microsegment
                    # such that a total weighted baseline lifetime may be
                    # calculated below for the measure across all contributing
                    # microsegments
                    add_dict = {
                        "stock": {
                            "total": {
                                "all": add_stock_total,
                                "measure": add_stock_total_meas},
                            "competed": {
                                "all": add_stock_compete,
                                "measure": add_stock_compete_meas}},
                        "energy": {
                            "total": {
                                "baseline": add_energy_total,
                                "efficient": add_energy_total_eff},
                            "competed": {
                                "baseline": add_energy_compete,
                                "efficient": add_energy_compete_eff}},
                        "carbon": {
                            "total": {
                                "baseline": add_carb_total,
                                "efficient": add_carb_total_eff},
                            "competed": {
                                "baseline": add_carb_compete,
                                "efficient": add_carb_compete_eff}},
                        "cost": {
                            "stock": {
                                "total": {
                                    "baseline": add_stock_cost,
                                    "efficient": add_stock_cost_meas},
                                "competed": {
                                    "baseline": add_stock_cost_compete,
                                    "efficient": add_stock_cost_compete_meas}},
                            "energy": {
                                "total": {
                                    "baseline": add_energy_cost,
                                    "efficient": add_energy_cost_eff},
                                "competed": {
                                    "baseline": add_energy_cost_compete,
                                    "efficient": add_energy_cost_compete_eff}},
                            "carbon": {
                                "total": {
                                    "baseline": add_carb_cost,
                                    "efficient": add_carb_cost_eff},
                                "competed": {
                                    "baseline": add_carb_cost_compete,
                                    "efficient": add_carb_cost_compete_eff}}},
                        "lifetime": {
                            "baseline": {
                                yr: life_base[yr] * add_stock_total[yr] for
                                yr in self.handyvars.aeo_years},
                            "measure": life_meas}}

                    # Check fugitive emissions option settings and update
                    # dict with fugitive emissions, broken out by the source
                    # of the fugitive emissions; note that entries to this
                    # dict will be None when the given type of fugitive
                    # emissions is not selected (e.g., all under the methane
                    # key will be None when only refrigerants are being
                    # assessed, and vice versa)
                    if opts.fugitive_emissions is not False:
                        add_dict['fugitive emissions'] = {
                            "methane": {
                                "total": {
                                    "baseline": add_fmeth_total,
                                    "efficient": add_fmeth_total_eff},
                                "competed": {
                                    "baseline": add_fmeth_compete,
                                    "efficient": add_fmeth_compete_eff}},
                            "refrigerants": {
                                "total": {
                                    "baseline": add_frefr_total,
                                    "efficient": add_frefr_total_eff},
                                "competed": {
                                    "baseline": add_frefr_compete,
                                    "efficient": add_frefr_compete_eff}}}

                    # Using the key chain for the current microsegment,
                    # determine the output climate zone, building type, and end
                    # use breakout categories to which the current microsegment
                    # applies

                    # Establish applicable climate zone breakout
                    for cz in self.handyvars.out_break_czones.items():
                        if mskeys[1] in cz[1]:
                            out_cz = cz[0]
                    # Establish applicable building type breakout
                    for bldg in self.handyvars.out_break_bldgtypes.items():
                        if all([x in bldg[1] for x in [
                                mskeys[2], mskeys[-1]]]):
                            out_bldg = bldg[0]
                    # Establish applicable end use breakout
                    for eu in self.handyvars.out_break_enduses.items():
                        # * Note: The 'other' microsegment end
                        # use may map to either the 'Refrigeration' output
                        # breakout or the 'Other' output breakout, depending on
                        # the technology type specified in the measure
                        # definition. Also note that 'supply' side
                        # heating/cooling microsegments map to the
                        # 'Heating (Equip.)'/'Cooling (Equip.)' end uses, while
                        # 'demand' side heating/cooling microsegments map to
                        # the 'Heating (Env.)'/'Cooling (Env.) end use, with
                        # the exception of 'demand' side heating/cooling
                        # microsegments that represent waste heat from lights -
                        # these are categorized as part of the 'Lighting' end
                        # use
                        if mskeys[4] == "other":
                            if mskeys[5] == "freezers":
                                out_eu = "Refrigeration"
                            else:
                                out_eu = "Other"
                        elif mskeys[4] in eu[1]:
                            if (eu[0] in ["Heating (Equip.)",
                                          "Cooling (Equip.)"] and
                                mskeys[5] == "supply") or (
                                eu[0] in ["Heating (Env.)",
                                          "Cooling (Env.)"] and
                                mskeys[5] == "demand" and
                                mskeys[0] == "primary") or (
                                eu[0] not in ["Heating (Equip.)",
                                              "Cooling (Equip.)",
                                              "Heating (Env.)",
                                              "Cooling (Env.)"]):
                                out_eu = eu[0]
                        elif "lighting gain" in mskeys:
                            out_eu = "Lighting"
                    # If applicable, establish fuel type breakout (electric vs.
                    # non-electric); note – only applicable to end uses that
                    # are at least in part fossil-fired
                    if (len(self.handyvars.out_break_fuels.keys()) != 0) and (
                            out_eu in self.handyvars.out_break_eus_w_fsplits):
                        # Flag for detailed fuel type breakout
                        detail = len(self.handyvars.out_break_fuels.keys()) > 2
                        # Establish breakout of fuel type that is being
                        # reduced (e.g., through efficiency or fuel switching
                        # away from the fuel)
                        for f in self.handyvars.out_break_fuels.items():
                            if mskeys[3] in f[1]:
                                # Special handling for other fuel tech.,
                                # under detailed fuel type breakouts; this
                                # tech. may fit into multiple fuel categories
                                if detail and mskeys[3] == "other fuel":
                                    # Assign coal/kerosene tech.
                                    if f[0] == "Distillate/Other" and (
                                        mskeys[-2] is not None and any([
                                            x in mskeys[-2] for x in [
                                            "coal", "kerosene"]])):
                                        out_fuel_save = f[0]
                                    # Assign wood tech.
                                    elif f[0] == "Biomass" and (
                                        mskeys[-2] is not None and "wood" in
                                            mskeys[-2]):
                                        out_fuel_save = f[0]
                                    # All other tech. goes to propane
                                    elif f[0] == "Propane":
                                        out_fuel_save = f[0]
                                else:
                                    out_fuel_save = f[0]
                        # Establish breakout of fuel type that is being added
                        # to via fuel switching, if applicable
                        if self.fuel_switch_to == "electricity" and \
                                out_fuel_save != "Electric":
                            out_fuel_gain = "Electric"
                        elif self.fuel_switch_to not in [None, "electricity"] \
                                and out_fuel_save == "Electric":
                            # Check for detailed fuel types
                            if detail:
                                for f in \
                                        self.handyvars.out_break_fuels.items():
                                    # Special handling for other fuel tech.,
                                    # under detailed fuel type breakouts; this
                                    # tech. may fit into multiple fuel cats.
                                    if self.fuel_switch_to in f[1] and \
                                            mskeys[3] == "other fuel":
                                        # Assign coal/kerosene tech.
                                        if f[0] == "Distillate/Other" and (
                                            mskeys[-2] is not None and any([
                                                x in mskeys[-2] for x in [
                                                "coal", "kerosene"]])):
                                            out_fuel_gain = f[0]
                                        # Assign wood tech.
                                        elif f[0] == "Biomass" and (
                                            mskeys[-2] is not None and "wood"
                                                in mskeys[-2]):
                                            out_fuel_gain = f[0]
                                        # All other tech. goes to propane
                                        elif f[0] == "Propane":
                                            out_fuel_gain = f[0]
                                    elif self.fuel_switch_to in f[1]:
                                        out_fuel_gain = f[0]
                            else:
                                out_fuel_gain = "Non-Electric"
                        else:
                            out_fuel_gain = ""
                    else:
                        out_fuel_save, out_fuel_gain = ("" for n in range(2))

                    # Given the contributing microsegment's applicable climate
                    # zone, building type, and end use categories, add the
                    # microsegment's energy/ecost/carbon baseline, efficient
                    # energy/ecost/carbon, and energy/ecost/carbon savings val.
                    # to the appropriate leaf node of the dictionary used to
                    # store measure output breakout information. * Note: the
                    # values in this dictionary will be normalized in run.py by
                    # the measure's energy/ecost/carbon baseline, efficient
                    # energy/ecost/carbon, and energy/ecost/carbon savings
                    # totals (post-competition) to yield the fractions of
                    # measure energy, carbon, and cost markets/savings that are
                    # attributable to each climate zone, building type, and
                    # end use that the measure applies to
                    try:
                        # Create a shorthand for baseline and efficient
                        # energy/carbon/cost data to add to the breakout dict
                        base_data = [add_energy_total, add_energy_cost,
                                     add_carb_total]
                        eff_data = [add_energy_total_eff, add_energy_cost_eff,
                                    add_carb_total_eff]
                        # For a fuel switching case where the user desires that
                        # the outputs be split by fuel, create shorthands for
                        # any efficient energy/carbon/cost that remains with
                        # the baseline fuel
                        if self.fuel_switch_to is not None and out_fuel_save:
                            eff_data_fs = [add_fs_energy_eff_remain,
                                           add_fs_energy_cost_eff_remain,
                                           add_fs_carb_eff_remain]
                            # Record the efficient energy that has not yet fuel
                            # switched and total efficient energy for the
                            # current mseg for later use in packaging and/or
                            # competing measures
                            self.eff_fs_splt[adopt_scheme][
                                str(contrib_mseg_key)] = {
                                    "energy": [add_fs_energy_eff_remain,
                                               add_energy_total_eff],
                                    "cost": [add_fs_energy_cost_eff_remain,
                                             add_energy_cost_eff],
                                    "carbon": [add_fs_carb_eff_remain,
                                               add_carb_total_eff]}
                        # Handle case where output breakout includes fuel type
                        # breakout or not
                        if out_fuel_save:
                            # Update results for the baseline fuel; handle
                            # case where results for the current region, bldg.,
                            # end use, and fuel have not yet been initialized
                            try:
                                for yr in self.handyvars.aeo_years:
                                    for ind, key in enumerate([
                                            "energy", "cost", "carbon"]):
                                        # Baseline; add in baseline data
                                        # as-is
                                        self.markets[adopt_scheme][
                                            "mseg_out_break"][key][
                                            "baseline"][out_cz][out_bldg][
                                            out_eu][out_fuel_save][yr] += \
                                            base_data[ind][yr]
                                        # Efficient and savings; if there is
                                        # fuel switching, only the portion
                                        # of the efficient case results that
                                        # have not yet switched (due to stock
                                        # turnover limitations) remain, and
                                        # savings are the delta between what
                                        # remains unswitched in the efficient
                                        # case and the baseline
                                        if not out_fuel_gain:
                                            self.markets[adopt_scheme][
                                                "mseg_out_break"][key][
                                                "efficient"][out_cz][out_bldg][
                                                out_eu][out_fuel_save][yr] += \
                                                eff_data[ind][yr]
                                            self.markets[adopt_scheme][
                                                "mseg_out_break"][key][
                                                "savings"][out_cz][out_bldg][
                                                out_eu][out_fuel_save][yr] += (
                                                    base_data[ind][yr] -
                                                    eff_data[ind][yr])
                                        else:
                                            self.markets[adopt_scheme][
                                                "mseg_out_break"][key][
                                                "efficient"][out_cz][out_bldg][
                                                out_eu][out_fuel_save][yr] += \
                                                eff_data_fs[ind][yr]
                                            self.markets[adopt_scheme][
                                                "mseg_out_break"][key][
                                                "savings"][out_cz][out_bldg][
                                                out_eu][out_fuel_save][yr] += (
                                                    base_data[ind][yr] -
                                                    eff_data_fs[ind][yr])
                            except KeyError:
                                for ind, key in enumerate([
                                        "energy", "cost", "carbon"]):
                                    # Baseline; add in baseline data
                                    # as-is
                                    self.markets[adopt_scheme][
                                        "mseg_out_break"][key][
                                        "baseline"][out_cz][out_bldg][
                                        out_eu][out_fuel_save] = {
                                            yr: base_data[ind][yr] for
                                            yr in self.handyvars.aeo_years}
                                    # Efficient and savings; if there is fuel
                                    # switching, only the portion of the
                                    # efficient case results that have not yet
                                    # switched (due to stock turnover
                                    # limitations) remain, and savings are the
                                    # delta between what remains unswitched in
                                    # the efficient case and the baseline
                                    if not out_fuel_gain:
                                        self.markets[adopt_scheme][
                                            "mseg_out_break"][key][
                                            "efficient"][out_cz][out_bldg][
                                            out_eu][out_fuel_save] = {
                                                yr: eff_data[ind][yr] for
                                                yr in self.handyvars.aeo_years}
                                        self.markets[adopt_scheme][
                                            "mseg_out_break"][key][
                                            "savings"][out_cz][out_bldg][
                                                out_eu][out_fuel_save] = {
                                                yr: (base_data[ind][yr] -
                                                     eff_data[ind][yr]) for
                                                yr in self.handyvars.aeo_years}
                                    else:
                                        self.markets[adopt_scheme][
                                            "mseg_out_break"][key][
                                            "efficient"][out_cz][out_bldg][
                                            out_eu][out_fuel_save] = {
                                                yr: eff_data_fs[ind][yr] for
                                                yr in self.handyvars.aeo_years}
                                        self.markets[adopt_scheme][
                                            "mseg_out_break"][key][
                                            "savings"][out_cz][out_bldg][
                                                out_eu][out_fuel_save] = {
                                                yr: (base_data[ind][yr] -
                                                     eff_data_fs[ind][yr]) for
                                                yr in self.handyvars.aeo_years}
                            # In a fuel switching case, update results for
                            # the fuel being switched/added to
                            if out_fuel_gain:
                                # Handle case where results for the current
                                # region, bldg., end use, and fuel have not yet
                                # been initialized
                                try:
                                    for yr in self.handyvars.aeo_years:
                                        for ind, key in enumerate([
                                                "energy", "cost", "carbon"]):
                                            # Note: no need to add to baseline
                                            # for fuel being switched to,
                                            # which remains zero

                                            # Efficient and savings; efficient
                                            # case energy/emissions/cost that
                                            # do not remain with the baseline
                                            # fuel are added to the switched to
                                            # fuel and represented as negative
                                            # savings for the switched to fuel
                                            self.markets[adopt_scheme][
                                                "mseg_out_break"][key][
                                                "efficient"][out_cz][out_bldg][
                                                out_eu][out_fuel_gain][yr] += \
                                                (eff_data[ind][yr] -
                                                 eff_data_fs[ind][yr])
                                            self.markets[adopt_scheme][
                                                "mseg_out_break"][key][
                                                "savings"][out_cz][out_bldg][
                                                out_eu][out_fuel_gain][yr] -= (
                                                    eff_data[ind][yr] -
                                                    eff_data_fs[ind][yr])
                                except KeyError:
                                    for ind, key in enumerate([
                                            "energy", "cost", "carbon"]):
                                        # Baseline for the fuel being switched
                                        # to is initialized as zero
                                        self.markets[adopt_scheme][
                                            "mseg_out_break"][key][
                                            "baseline"][out_cz][out_bldg][
                                            out_eu][out_fuel_gain] = {
                                                yr: 0 for yr in
                                                self.handyvars.aeo_years}
                                        # Efficient and savings; efficient
                                        # case energy/emissions/cost that
                                        # do not remain with the baseline
                                        # fuel are added to the switched to
                                        # fuel and represented as negative
                                        # savings for the switched to fuel
                                        self.markets[adopt_scheme][
                                            "mseg_out_break"][key][
                                            "efficient"][out_cz][out_bldg][
                                            out_eu][out_fuel_gain] = {
                                                yr: (eff_data[ind][yr] -
                                                     eff_data_fs[ind][yr]) for
                                                yr in self.handyvars.aeo_years}
                                        self.markets[adopt_scheme][
                                            "mseg_out_break"][key][
                                            "savings"][out_cz][out_bldg][
                                                out_eu][out_fuel_gain] = {
                                                yr: -(eff_data[ind][yr] -
                                                      eff_data_fs[ind][yr]) for
                                                yr in self.handyvars.aeo_years}
                        else:
                            # Handle case where results for the current region,
                            # bldg., end use, and fuel have not yet been
                            # initialized
                            try:
                                for yr in self.handyvars.aeo_years:
                                    for ind, key in enumerate([
                                            "energy", "cost", "carbon"]):
                                        self.markets[adopt_scheme][
                                            "mseg_out_break"][key][
                                            "baseline"][out_cz][out_bldg][
                                            out_eu][yr] += base_data[ind][yr]
                                        self.markets[adopt_scheme][
                                            "mseg_out_break"][key][
                                            "efficient"][out_cz][out_bldg][
                                            out_eu][yr] += eff_data[ind][yr]
                                        self.markets[adopt_scheme][
                                            "mseg_out_break"][key][
                                            "savings"][out_cz][out_bldg][
                                            out_eu][yr] += (
                                                base_data[ind][yr] -
                                                eff_data[ind][yr])
                            except KeyError:
                                for ind, key in enumerate([
                                        "energy", "cost", "carbon"]):
                                    self.markets[adopt_scheme][
                                        "mseg_out_break"][key]["baseline"][
                                        out_cz][out_bldg][out_eu] = {
                                            yr: base_data[ind][yr] for
                                            yr in self.handyvars.aeo_years}
                                    self.markets[adopt_scheme][
                                        "mseg_out_break"][key]["efficient"][
                                        out_cz][out_bldg][out_eu] = {
                                            yr: eff_data[ind][yr] for
                                            yr in self.handyvars.aeo_years}
                                    self.markets[adopt_scheme][
                                        "mseg_out_break"][key]["savings"][
                                        out_cz][out_bldg][out_eu] = {
                                            yr: (base_data[ind][yr] -
                                                 eff_data[ind][yr]) for
                                            yr in self.handyvars.aeo_years}

                    # Yield warning if current contributing microsegment cannot
                    # be mapped to an output breakout category
                    except KeyError:
                        verboseprint(
                            opts.verbose,
                            "Baseline market key chain: '" +
                            str(mskeys) +
                            "' for ECM '" + self.name + "' does not map to "
                            "output breakout categories, thus will not "
                            "be reflected in output breakout data")

                    # Record contributing microsegment data needed for ECM
                    # competition in the analysis engine
                    contrib_mseg_key_str = str(contrib_mseg_key)

                    # Case with no existing 'windows' contributing microsegment
                    # for the current climate zone, building type, fuel type,
                    # and end use (create new 'contributing mseg keys and
                    # values' and 'competed choice parameters' microsegment
                    # information)
                    if contrib_mseg_key_str not in self.markets[adopt_scheme][
                        "mseg_adjust"][
                            "contributing mseg keys and values"].keys():
                        # Register contributing microsegment information for
                        # later use in determining savings overlaps for
                        # measures that apply to this microsegment
                        self.markets[adopt_scheme]["mseg_adjust"][
                            "contributing mseg keys and values"][
                            contrib_mseg_key_str] = add_dict
                        # Register choice parameters associated with
                        # contributing microsegment for later use in
                        # apportioning out various technology options across
                        # competed stock
                        self.markets[adopt_scheme]["mseg_adjust"][
                            "competed choice parameters"][
                            contrib_mseg_key_str] = choice_params
                    # Case with existing 'windows' contributing microsegment
                    # for the current climate zone, building type, fuel type,
                    # and end use (add to existing 'contributing mseg keys and
                    # values' information)
                    else:
                        self.markets[adopt_scheme]["mseg_adjust"][
                            "contributing mseg keys and values"][
                            contrib_mseg_key_str] = self.add_keyvals(
                                self.markets[adopt_scheme]["mseg_adjust"][
                                    "contributing mseg keys and values"][
                                    contrib_mseg_key_str], add_dict)

                    # Market scaling fraction comes out of
                    # "partition_microsegment" function in dict format, broken
                    # by year; reformat as single value if values for each year
                    # are identical
                    if all([round(x[1], 3) == round(mkt_scale_frac_fin[
                            self.handyvars.aeo_years[0]], 3) for
                            x in mkt_scale_frac_fin.items()]):
                        mkt_scale_frac_fin = mkt_scale_frac_fin[
                            self.handyvars.aeo_years[0]]

                    # Record the sub-market scaling fraction associated with
                    # the current contributing microsegment
                    self.markets[adopt_scheme]["mseg_adjust"][
                        "contributing mseg keys and values"][
                        contrib_mseg_key_str]["sub-market scaling"] = \
                        mkt_scale_frac_fin

                    # Add all updated contributing microsegment stock, energy
                    # carbon, cost, and lifetime information to existing master
                    # mseg dict and move to next iteration of the loop through
                    # key chains
                    self.markets[adopt_scheme]["master_mseg"] = \
                        self.add_keyvals(self.markets[adopt_scheme][
                            "master_mseg"], add_dict)

        # Print warnings
        if len(warn_list) > 0:
            for warn in list(set(warn_list)):
                print(warn)
        # Print suppressed incentives warning
        if len(suppress_incent) > 0:
            warnings.warn("Incentives found for measure '" + self.name +
                          "' markets but cannot be applied due to "
                          "performance units for incentives that are "
                          "inconsistent with the baseline or measure "
                          "equipment performance units. Check that measure "
                          "uses standard performance units recommended here: "
                          "https://scout-bto.readthedocs.io/en/latest/"
                          "ecm_reference.html#energy-efficiency-units")
        # Further normalize a measure's lifetime and stock information (where
        # the latter is based on square footage) to the number of microsegments
        # that contribute to the measure's overall master microsegment and
        # have valid stock/energy and cost/performance/lifetime data
        if valid_keys_stk_energy != 0 and valid_keys_cpl != 0:

            for adopt_scheme in self.handyvars.adopt_schemes:
                # Calculate overall average baseline and measure lifetimes
                for yr in self.handyvars.aeo_years:
                    # Divide summed baseline lifetimes * stock values for
                    # contributing microsegments by total stock across all
                    # contributing microsegments
                    if self.markets[adopt_scheme][
                            "master_mseg"]["stock"]["total"]["all"][yr] != 0:
                        self.markets[adopt_scheme]["master_mseg"]["lifetime"][
                            "baseline"][yr] = self.markets[adopt_scheme][
                            "master_mseg"]["lifetime"]["baseline"][yr] / \
                            self.markets[adopt_scheme][
                            "master_mseg"]["stock"]["total"]["all"][yr]
                # Divide summed measure lifetimes by total number of
                # contributing microsegment key chains with valid
                # cost/performance/lifetime data
                self.markets[adopt_scheme][
                    "master_mseg"]["lifetime"]["measure"] = \
                    self.markets[adopt_scheme]["master_mseg"][
                    "lifetime"]["measure"] / valid_keys_cpl

                # Remove stock multipliers from lifetime values in
                # contributing microsegment (values used in competition later)

                # Shorthand for contributing microsegment information
                contrib_msegs = self.markets[adopt_scheme]["mseg_adjust"][
                    "contributing mseg keys and values"]
                for key in contrib_msegs.keys():
                    contrib_msegs[key]["lifetime"]["baseline"] = {yr: (
                        contrib_msegs[key]["lifetime"]["baseline"][yr] /
                        contrib_msegs[key]["stock"]["total"]["all"][yr]) if
                        contrib_msegs[key]["stock"]["total"]["all"][yr] != 0
                        else 10 for yr in self.handyvars.aeo_years}

                # In microsegments where square footage must be used as stock,
                # the square footages (or number of households, in the
                # residential case) cannot be summed to calculate the master
                # microsegment stock values (as is the case when using no. of
                # units).  For example, 1000 Btu of cooling and heating in the
                # same 1000 square foot building should not yield 2000 total
                # square feet of stock in the master microsegment even though
                # there are two contributing microsegments in this case
                # (heating and cooling). This is remedied by dividing summed
                # square footage values by (# valid key chains / (# czones * #
                # bldg types * # structure types)), where the numerator refers
                # to the number of full dict key chains that contributed to the
                # mseg stock, energy, and cost calcs, and the denominator
                # reflects the breakdown of square footage (or number of
                # households) by climate zone, building type, and the
                # structure type that the measure applies to.
                if sqft_subst == 1:
                    # Determine number of structure types the measure applies
                    # to (could be just new, just existing, or both)
                    if isinstance(self.structure_type, list) and len(
                            self.structure_type) > 1:
                        structure_types = 2
                    else:
                        structure_types = 1
                    # Create a factor for reduction of msegs with ft^2 floor
                    # area stock
                    reduce_num = valid_keys_cpl / (
                        len(ms_lists[0]) * len(ms_lists[1]) * structure_types)
                    # Adjust master microsegment by above factor
                    self.markets[adopt_scheme]["master_mseg"] = \
                        self.div_keyvals_float_restrict(self.markets[
                            adopt_scheme]["master_mseg"], reduce_num)
                    # Adjust all recorded microsegments that contributed to the
                    # master microsegment by above factor
                    contrib_msegs = self.div_keyvals_float_restrict(
                        contrib_msegs, reduce_num)
                else:
                    reduce_num = 1

        # Generate an error message when no contributing microsegments
        # with a full set of stock/energy and cost/performance/lifetime data
        # have been found for the measure's master microsegment
        elif self.remove is False:
            raise KeyError(
                "No data retrieved for applicable baseline market "
                "definition of ECM '" + self.name + "'")

        # Print update on measure status

        # If not in verbose mode, suppress summaries about missing data;
        # otherwise, summarize the extent of the missing data
        if opts is not None and opts.verbose is True:
            # Summarize percentage of baseline stock and energy
            # data that were missing (if any)
            if valid_keys_stk_energy == valid_keys:
                bstk_msg = ""
            else:
                bstk_msg = "\n" + " - " + str(
                    round((1 - (valid_keys_stk_energy / valid_keys)) *
                          100)) + \
                    " % of baseline technologies were missing valid" + \
                    " baseline stock or energy data and" + \
                    " removed from analysis"
            # Summarize percentage of baseline cost, performance, and lifetime
            # data that were missing (if any)
            if valid_keys_cpl == valid_keys_stk_energy:
                bcpl_msg = ""
            else:
                bcpl_msg = "\n" + " - " + str(
                    round((1 - (valid_keys_cpl / valid_keys_stk_energy)) *
                          100)) + \
                    " % of baseline technologies were missing valid" + \
                    " baseline cost, performance, or lifetime data and" + \
                    " removed from analysis"
            # Summarize percentage of baseline consumer choice data that
            # were missing (if any)
            if valid_keys_consume == valid_keys_stk_energy:
                bcc_msg = ""
            else:
                bcc_msg = "\n" + " - " + str(
                    round((1 - (valid_keys_consume / valid_keys_stk_energy)) *
                          100)) + \
                    " % of baseline technologies were missing" + \
                    " valid consumer choice data"
            cc_msg = ""
        else:
            # Missing baseline stock and energy data, cost, performance, and
            # lifetime data and consumer data summaries are blank
            bstk_msg, bcpl_msg, bcc_msg = ("" for n in range(3))
            # If one or more conversion to ECM unit cost has been made, note
            # this in the update message
            if cost_converts == 0:
                cc_msg = ""
            else:
                cc_msg = " (cost units converted)"

        # Print message to console; if in verbose mode, print to new line,
        # otherwise append to existing message on the console
        if opts is not None and opts.verbose is True:
            print("ECM '" + self.name + "' successfully updated" +
                  bstk_msg + bcpl_msg + bcc_msg + cc_msg)
        else:
            print("Success" + bstk_msg + bcpl_msg + bcc_msg + cc_msg)

    def gen_tsv_facts(
            self, tsv_data, mskeys, bldg_sect, cost_conv, opts,
            cost_energy_meas):
        """Set annual re-weighting factors and hourly load fractions for TSV.

        Args:
            tsv_data (data): Time-resolved load/energy cost/emissions data.
            mskeys (tuple): Microsegment information needed to key load shapes.
            bldg_sect (string): Building sector of the current microsegment.
            cost_conv (dict): Conversion factors, EPlus->Scout building types.
            opts (object): Stores user-specified execution options.
            cost_energy_meas (dict): Annual retail electricity rates for the
                region the measure applies to.

        Returns:
            Dict of microsegment-specific energy, cost, and emissions annual
            re-weighting factors that reflect time-sensitive evaluation of
            energy efficiency and associated energy costs/carbon emissions, and
            (if desired by user) hourly fractions of annual energy load.
        """

        # If time-sensitive valuation is needed and energy, cost, and carbon
        # re-weighting and load shape information does not already exist for
        # the current combination of region, building type, and end use, set
        # energy load shapes for given climate, building type, and end use
        if self.handyvars.tsv_hourly_lafs is not None and \
            self.handyvars.tsv_hourly_lafs[mskeys[1]][bldg_sect][
                mskeys[2]][mskeys[4]] is None:
            # Primary microsegment case: find the load shape associated with
            # the primary end use
            if mskeys[0] == "primary":
                # First, assume there is a load shape for the current end use
                try:
                    # Do not assign the "other" load shape directly to the
                    # "other" end use in Scout, as this end use has
                    # sub-categories (e.g., dishwashing, pool pumps/heaters,
                    # etc.) that should be used to key in the appropriate load
                    # shape information
                    if mskeys[4] == "other":
                        raise (KeyError)
                    else:
                        # Set load data end use key for use in 'apply_tsv'
                        eu = mskeys[4]
                        # Key in the appropriate load shape data
                        load_fact = tsv_data["load"][bldg_sect][eu]
                except KeyError:
                    # If there is no load shape for the current end use, handle
                    # the resultant error differently for res./com.
                    if bldg_sect == "residential":
                        # Secondary heating maps to heating load shape
                        if mskeys[4] == "secondary heating":
                            eu = "heating"
                        # Computers and TVs map to plug loads load shape
                        elif mskeys[4] in ["computers", "TVs"]:
                            eu = "plug loads"
                        # Clothes drying technology maps to clothes drying
                        elif mskeys[4] == "drying":
                            eu = "clothes drying"
                        # Other end use maps to various load shapes
                        elif mskeys[4] == "other":
                            # Dishwasher technology maps to dishwasher
                            if mskeys[5] == "dishwasher":
                                eu = "dishwasher"
                            # Clothes washing tech. maps to clothes washing
                            elif mskeys[5] == "clothes washing":
                                eu = "clothes washing"
                            # Pool heaters maps to pool heaters
                            elif mskeys[5] == "pool heaters":
                                eu = "pool heaters"
                            # Pool pumps map to pool pumps
                            elif mskeys[5] == "pool pumps":
                                eu = "pool pumps"
                            # Freezers map to refrigeration
                            elif mskeys[5] == "freezers":
                                eu = "refrigeration"
                            # Portable electric spas (e.g. hot tubs) map
                            # to portable electric spas
                            elif mskeys[5] == "portable electric spas":
                                eu = "portable electric spas"
                            # All other maps to other
                            else:
                                eu = "plug loads"
                        # In all other cases, error
                        else:
                            raise KeyError(
                                "The following baseline segment could not be "
                                "mapped to any baseline load shape in the "
                                "Scout database: " + str(mskeys))
                    elif bldg_sect == "commercial":
                        # For commercial PCs/non-PC office equipment and MELs,
                        # use the load shape for plug loads
                        if mskeys[4] in ["PCs", "non-PC office equipment",
                                         "MELs", "cooking"]:
                            eu = "plug loads"
                        # In all other cases, error
                        else:
                            raise KeyError(
                                "The following baseline segment could not be "
                                "mapped to any baseline load shape in the "
                                "Scout database: " + str(mskeys))
                    # Key in the appropriate load shape data
                    load_fact = tsv_data["load"][bldg_sect][eu]

            # Commercial secondary lighting microsegment case: use the lighting
            # load shape for secondary heating/cooling impacts from lighting
            else:
                eu = "lighting"
                # Key in the appropriate load shape data
                load_fact = tsv_data["load"][bldg_sect][eu]

            # Find weights needed to map ASHRAE/IECC climate zones to EMM
            # region, and EnergyPlus building type to Scout building type

            # Find ASHRAE/IECC -> EMM weighting factors for current EMM region
            ash_czone_wts = [[x["ASHRAE"], x[mskeys[1]]] for x in
                             self.handyvars.ash_emm_map if (
                                numpy.isfinite(x[mskeys[1]]) and
                                x[mskeys[1]] != 0)]
            # Check to ensure that region weighting factors sum to 1
            if round(sum([x[1] for x in ash_czone_wts]), 2) != 1:
                raise ValueError(
                    "ASHRAE climate -> EMM region weights for region " +
                    mskeys[1] + " and building type " + mskeys[2] +
                    " do not sum to 1")

            # Find EnergyPlus -> Scout building type weighting factors for
            # current building type. Note that residential buildings map
            # directly to the ResStock single family building type
            if bldg_sect == "commercial":
                eplus_bldg_wts = cost_conv["building type conversions"][
                    "conversion data"]["value"]["commercial"][mskeys[2]]
                # Handle case where there is no EnergyPlus analogue for a Scout
                # commercial building type (e.g., for the other or unspecified
                # Scout building type and the health care Scout building type)
                if eplus_bldg_wts is None:
                    if mskeys[2] in ["other", "unspecified"]:
                        eplus_bldg_wts = {"MediumOfficeDetailed": 1}
                    elif mskeys[2] == "health care":
                        eplus_bldg_wts = {"Hospital": 1}
            else:
                if mskeys[2] == "single family home":
                    eplus_bldg_wts = {"SF": 1}
                elif mskeys[2] == "multi family home":
                    eplus_bldg_wts = {"MF": 1}
                else:
                    eplus_bldg_wts = {"MH": 1}
            # Check to ensure that building type weighting factors sum to 1
            if round(sum([x[1] for x in eplus_bldg_wts.items()]), 2) != 1:
                raise ValueError(
                    "EPlus building -> Scout building mapping weights for "
                    "region " + mskeys[1] + " and building type " + mskeys[2] +
                    " do not sum to 1")

            # Generate appropriate 8760 price and emissions scaling shapes if
            # either measure TSV features are present or the user desires
            # TSV metrics outputs; assume these shapes are not necessary if
            # the user only desires sector-level load shapes
            if (opts.sect_shapes is True and opts.tsv_metrics is False and
                    self.tsv_features is None):
                cost_fact_hourly, carbon_fact_hourly, cost_yr_map, \
                    carb_yr_map = (None for n in range(4))
            else:
                # Set time-varying electricity price scaling factors for the
                # EMM region (dict with keys distinguished by year, *CURRENTLY*
                # every two years beginning in 2018). Assume that
                # only 41% of the electricity price is subject to scaling based
                # on historical allocation of fixed vs. variable/volumetric
                # retail electricity costs in
                # https://www.nrel.gov/docs/fy22osti/78224.pdf. Check for cases
                # where multiplication of scaling factor by retail rate will
                # result in negative prices and force such cases to zero.
                if self.tsv_hourly_price[mskeys[1]] is None:
                    cost_fact_hourly, self.tsv_hourly_price[
                        mskeys[1]] = ({} for n in range(2))
                    # Loop through all years in price shape data and record
                    # final scaling factors
                    for yr in tsv_data["price"][
                            "electricity price shapes"].keys():
                        # Since scaling factor calculation depends on retail
                        # energy rates ('cost_energy_meas'), which are only
                        # available for AEO year range, check for inclusion of
                        # year from price shape data in AEO range; if not in
                        # range, set all price scaling factors to 1 for year
                        if yr in self.handyvars.aeo_years:
                            cost_fact_hourly[yr], \
                                self.tsv_hourly_price[
                                mskeys[1]][yr] = ([(
                                    (0.59 + 0.41 * x) if (
                                        cost_energy_meas[yr] *
                                        (0.59 + 0.41 * x) >= 0) else 0) for
                                    x in tsv_data["price"][
                                     "electricity price shapes"][yr][
                                     mskeys[1]]] for n in range(2))
                        else:
                            cost_fact_hourly[yr], \
                                self.tsv_hourly_price[
                                mskeys[1]][yr] = ([1] * 8760 for n in range(2))
                else:
                    cost_fact_hourly = \
                        self.tsv_hourly_price[mskeys[1]]
                # Set TSV data -> AEO year mapping to use in preparing cost
                # scaling factors
                cost_yr_map = tsv_data["price_yr_map"]
                # Set time-varying emissions scaling factors for the EMM
                # region (dict with keys distinguished by year, *CURRENTLY*
                # every two years beginning in 2018)
                if self.tsv_hourly_emissions[mskeys[1]] is None:
                    carbon_fact_hourly,  self.tsv_hourly_emissions[
                        mskeys[1]] = ({yr: tsv_data["emissions"][
                            "average carbon emissions rates"][yr][
                            mskeys[1]] for yr in tsv_data["emissions"][
                            "average carbon emissions rates"].keys()} for
                            n in range(2))
                else:
                    carbon_fact_hourly = self.tsv_hourly_emissions[
                        mskeys[1]]
                # Set TSV data -> AEO year mapping to use in preparing
                # emissions scaling factors
                carb_yr_map = tsv_data["emissions_yr_map"]

            # Use 8760 load shape information, combined with 8760 price and
            # emissions shape information above, to calculate factors that
            # modify annually-determined baseline and efficient energy, cost,
            # and carbon totals such that they reflect sub-annual assessment
            # of these totals
            updated_tsv_fracs, updated_tsv_shapes = self.apply_tsv(
                load_fact, ash_czone_wts, eplus_bldg_wts, cost_fact_hourly,
                carbon_fact_hourly, mskeys, bldg_sect, eu, opts, cost_yr_map,
                carb_yr_map)
            # Set adjustment factors for current combination of
            # region, building type, and end use such that they
            # need not be calculated again for this combination in
            # subsequent technology microsegments
            self.handyvars.tsv_hourly_lafs[mskeys[1]][bldg_sect][
                mskeys[2]][mskeys[4]] = {
                    "annual adjustment fractions": copy.deepcopy(
                        updated_tsv_fracs),
                    "hourly shapes": copy.deepcopy(updated_tsv_shapes)}
        elif self.handyvars.tsv_hourly_lafs is not None:
            updated_tsv_fracs, updated_tsv_shapes = [
                copy.deepcopy(x) for x in [
                    self.handyvars.tsv_hourly_lafs[mskeys[1]][bldg_sect][
                        mskeys[2]][mskeys[4]]["annual adjustment fractions"],
                    self.handyvars.tsv_hourly_lafs[mskeys[1]][bldg_sect][
                        mskeys[2]][mskeys[4]]["hourly shapes"]]]
        else:
            updated_tsv_fracs = {
                "energy": {"baseline": 1, "efficient": 1},
                "cost": {"baseline": 1, "efficient": 1},
                "carbon": {"baseline": 1, "efficient": 1}}
            updated_tsv_shapes = None

        return updated_tsv_fracs, updated_tsv_shapes

    def apply_tsv(self, load_fact, ash_cz_wts, eplus_bldg_wts,
                  cost_fact_hourly, carbon_fact_hourly, mskeys, bldg_sect,
                  eu, opts, cost_yr_map, carb_yr_map):
        """Apply time varying efficiency levels to base load profile.

        Args:
            load_fact (dict): Hourly energy load fractions of annual load.
            ash_cz_wts (list): Factors to map ASH climates -> EMM regions.
            eplus_bldg_wts (dict): Factors to map EPlus -> Scout bldg. types.
            cost_fact_hourly (list): 8760 electricity price scaling factors.
            carbon_fact_hourly (list): 8760 emissions scaling factors.
            mskeys (tuple): Microsegment information.
            bldg_sect (str): Building sector flag (residential/commercial).
            eu (str): End use for keying time sensitive load data.
            opts (object): Stores user-specified execution options.
            cost_yr_map (dict): Mapping 8760 TSV price data years -> AEO years.
            carb_yr_map (dict): Mapping 8760 TSV carbon data yrs. -> AEO years.

        Returns:
            Dict of microsegment-specific energy, cost, and emissions re-
            weighting factors that reflect time-sensitive evaluation of energy
            efficiency and associated energy costs/carbon emissions; list
            with hourly fractions of annual baseline and efficient energy use
            (if desired by the user)
        """

        # Initialize overall factors to use in scaling annually-determined
        # baseline and efficient energy, cost and emissions data

        # Note: energy scaling data is not broken out by projection year
        energy_scale_base, energy_scale_eff = (0 for n in range(2))

        # Initialize hourly fractions of annual baseline and efficient energy
        # if sector-level load shape information is desired by the user
        if opts.sect_shapes is True:
            energy_base_shape, energy_eff_shape = ([
                0 for x in range(8760)] for n in range(2))
        # Initialize carbon/cost scaling factor variables, but only if
        # either measure TSV features are present or the user desires
        # TSV metrics outputs; assume these shapes are not necessary if
        # the user only desires sector-level load shapes
        if opts.tsv_metrics is not False or self.tsv_features is not None:
            # Initial format of cost/carbon scaling factor data (broken out by
            # years available in the 8760 TSV cost/carbon input data)
            cost_scale_base, cost_scale_eff = (
                {yr: 0 for yr in cost_yr_map.keys()} for n in range(2))
            carb_scale_base, carb_scale_eff = (
                {yr: 0 for yr in carb_yr_map.keys()} for n in range(2))
            # Final format of cost/carbon scaling factor data (broken out by
            # AEO years)
            cost_scale_base_aeo, cost_scale_eff_aeo, carb_scale_base_aeo, \
                carb_scale_eff_aeo = (
                    {yr: 0 for yr in self.handyvars.aeo_years} for
                    n in range(4))

        # Set the user-specified time-sensitive valuation features
        # for the current ECM; handle cases where this parameter is
        # broken out by EMM region, building type, and/or end use.
        # If no time sensitive valuation features are given, reflect
        # this with an empty dict

        if self.tsv_features is not None:
            if (type(self.tsv_features) is dict and all([
                x not in self.handyvars.tsv_feature_types for x in
                    self.tsv_features.keys()])):

                # By EMM region, building type, and end use
                try:
                    tsv_adjustments = self.tsv_features[mskeys[1]][mskeys[2]][
                        mskeys[4]]
                # By EMM region and building type
                except KeyError:
                    try:
                        tsv_adjustments = self.tsv_features[mskeys[1]][
                            mskeys[2]]
                    # By EMM region and end use
                    except KeyError:
                        try:
                            tsv_adjustments = self.tsv_features[mskeys[1]][
                                mskeys[4]]
                        # By building type and end use
                        except KeyError:
                            try:
                                tsv_adjustments = self.tsv_features[mskeys[2]][
                                    mskeys[4]]
                            # By EMM region
                            except KeyError:
                                try:
                                    tsv_adjustments = self.tsv_features[
                                        mskeys[1]]
                                # By building type
                                except KeyError:
                                    try:
                                        tsv_adjustments = self.tsv_features[
                                            mskeys[2]]
                                    # By end use
                                    except KeyError:
                                        try:
                                            tsv_adjustments = \
                                                self.tsv_features[
                                                    mskeys[4]]
                                        # Not broken out by any of these
                                        except KeyError:
                                            raise KeyError(
                                                "Unexpected breakout of "
                                                "tsv_features parameter for"
                                                "measure " + self.name +
                                                ". If this parameter is "
                                                "broken out by region, "
                                                "building type, or "
                                                "end use, ensure that ALL "
                                                "categories of these "
                                                "variables are correctly "
                                                "represented in the breakout.")
            else:
                tsv_adjustments = self.tsv_features
        else:
            tsv_adjustments = {}

        # Loop through all EPlus building types (which commercial load profiles
        # are broken out by) that map to the current Scout building type
        for bldg in eplus_bldg_wts.keys():
            # Find the appropriate key in the load shape information for the
            # current EnergyPlus building type
            try:
                load_fact_bldg_key = [
                        x for x in load_fact.keys() if (bldg in load_fact[x][
                            "represented building types"] or load_fact[x][
                            "represented building types"] == "all")][0]
            except IndexError:
                # Check for possible naming inconsistency between certain
                # building type names used in baseline load data vs. expected
                # reference case building type names
                if bldg in [
                    "RetailStandalone", "MediumOfficeDetailed",
                        "LargeOfficeDetailed"]:
                    # Account for possible use of StandAlone naming in
                    # baseline load shapes file (vs. expected 'Standalone')
                    if bldg == "RetailStandalone":
                        bldg_adj = "RetailStandAlone"
                    # Account for possible use of MediumOffice naming
                    # in in baseline load shapes file (vs. expected
                    # 'MediumOfficeDetailed')
                    elif bldg == "MediumOffice":
                        bldg_adj = "MediumOfficeDetailed"
                    # Account for possible use of LargeOffice naming
                    # in in baseline load shapes file (vs. expected
                    # 'LargeOfficeDetailed')
                    elif bldg == "LargeOffice":
                        bldg_adj = "LargeOfficeDetailed"
                    else:
                        bldg_adj = None
                        raise ValueError(
                            "No representative baseline load shape data "
                            "available for building type '" + bldg +
                            "'. Check './supporting_data/tsv_data/tsv_load.gz "
                            "to ensure that this building type name is "
                            "correctly listed under 'represented building "
                            "types key")
                # Redo search for appropriate bldg key in load shape data
                load_fact_bldg_key = [
                    x for x in load_fact.keys() if (bldg_adj in load_fact[x][
                        "represented building types"] or load_fact[x][
                        "represented building types"] == "all")][0]

            # Calculations for TSV metric outputs and/or any measures with
            # time sensitive valuation features are based on data with ASHRAE
            # climate zone breakouts, which must be mapped into the EMM
            # region breakouts of the Scout microsegments. In these cases,
            # find list of applicable ASHRAE regions and weights for mapping
            # into the EMM region of the current microsegment, then implement
            # the mapping via for loop below. In cases where TSV metrics
            # or measure features are not present, set the list to a single
            # region of "None" with weight of 1 (effectively avoiding mapping)
            if opts.tsv_metrics is not False or len(
                    tsv_adjustments.keys()) != 0:
                # Ensure that all applicable ASHRAE climate zones are
                # represented in the keys for time sensitive metrics data; if
                # a zone is not represented, remove it from the weighting and
                # renormalize weights
                ash_cz_wts = [
                    [x[0], x[1]] for x in ash_cz_wts if x[0] in
                    self.handyvars.tsv_climate_regions]
                # Ensure that ASHRAE climate zone weightings sum to 1
                ash_cz_renorm = sum([x[1] for x in ash_cz_wts])
                if round(ash_cz_renorm, 2) != 1:
                    ash_cz_wts = [[x[0], (x[1] / ash_cz_renorm)] for
                                  x in ash_cz_wts]
            else:
                ash_cz_wts = [[None, 1]]

            # Loop through all ASHRAE/IECC climate zones (which load profiles
            # are broken out by) that map to the current EMM region
            for cz in ash_cz_wts:
                # Set the climate zone key to use in keying in savings
                # shape information
                load_fact_climate_key = cz[0]
                # Flag for cases where multiple sets of system load shape
                # information are germane to the current climate zone
                if cz[0] and type(self.handyvars.cz_emm_map[cz[0]]) == int:
                    mult_sysshp = False
                    mult_sysshp_key_metrics, mult_sysshp_key_save = (
                        "set 1" for n in range(2))
                elif cz[0] and type(self.handyvars.cz_emm_map[cz[0]]) == dict:
                    mult_sysshp = True
                    # Given multiple possible system load shape data
                    # sets, find the key for the set that is representative
                    # of the current EMM region; initialize separate keys to
                    # use in retrieving TSV metrics data vs. savings shape
                    # data (the approaches to keying in these different types
                    # of data differ below)
                    mult_sysshp_key_metrics, mult_sysshp_key_save = ([
                        y[0] for y in self.handyvars.cz_emm_map[cz[0]].items()
                        if self.handyvars.emm_name_num_map[mskeys[1]]
                        in y[1][1]][0] for n in range(2))
                elif not cz[0]:  # Case with no TSV mapping from ASH -> EMM
                    mult_sysshp, mult_sysshp_key_metrics, \
                        mult_sysshp_key_save = (None for n in range(3))
                else:
                    raise ValueError(
                        "Unable to determine representative utility system "
                        "load data for climate " + cz[0])

                # Set the weighting factor to map the current EPlus building
                # and ASHRAE climate (which measure savings shapes may
                # be broken out by) to Scout building and EMM region,
                # and set the appropriate baseline load shape (8760 hourly
                # fractions of annual load, broken out by EPlus building type
                # and EMM region)
                # Set the weighting factor; note EPlus/Scout building types map
                # 1:1 for residential and thus no building type weighting is
                # necessary here
                if bldg_sect == "commercial":
                    emm_adj_wt = eplus_bldg_wts[bldg] * cz[1]
                else:
                    emm_adj_wt = cz[1]

                # Set the baseline load shape

                # Handle case where the baseline load shape is not broken out
                # by EMM region
                try:
                    base_load_hourly = load_fact[
                        load_fact_bldg_key]["load shape"][mskeys[1]]
                except (KeyError, TypeError):
                    base_load_hourly = load_fact[
                        load_fact_bldg_key]["load shape"]
                # Ensure that retrieved baseline load data are expected length
                if len(base_load_hourly) != 8760:
                    raise ValueError(
                        "Baseline load data are of unexpected length " +
                        "(" + str(len(base_load_hourly)) + " elements) for " +
                        "end use " + mskeys[4] + ", EPlus building type " +
                        bldg + ", and EMM region " + mskeys[1] + ". Check "
                        "file ./supporting_data/tsv_data/tsv_load.gz to "
                        "ensure that 8760 data values are available for this "
                        "microsegment")

                # Initialize efficient load shape as equal to base load
                eff_load_hourly = copy.deepcopy(base_load_hourly)

                # Loop through all time-varying efficiency features in sorted
                # order, applying each successively to the base load shape
                for a in [x for x in sorted(tsv_adjustments.keys())]:

                    # Set the applicable days on which time-varying efficiency
                    # impact applies; if no start and stop information is
                    # specified by the user, assume the impact applies all year
                    try:
                        # Set the starting and stopping days for the impact
                        start_stop_dy = [
                            tsv_adjustments[a][x] for x in [
                                "start_day", "stop_day"]]
                        # Generate a list of all days in which the
                        # time-varying efficiency impact applies
                        try:
                            applicable_days = list(
                                range(start_stop_dy[0] - 1, start_stop_dy[1]))
                        # Error sets applicable day range to the full year
                        except TypeError:
                            # Assume two day ranges are specified
                            try:
                                # Start/stop set 1
                                start_1 = start_stop_dy[0][0]
                                stop_1 = start_stop_dy[1][0]
                                # Start/stop set 2
                                start_2 = start_stop_dy[0][1]
                                stop_2 = start_stop_dy[1][1]
                                # Patch together two day ranges
                                applicable_days = list(
                                    range(start_1 - 1, stop_1)) + list(
                                    range(start_2 - 1, stop_2))
                            except TypeError:
                                applicable_days = range(365)
                    except (TypeError, KeyError):
                        applicable_days = range(365)

                    # Set the applicable hours in which to apply the
                    # time-varying efficiency impact; if no start and stop
                    # information is specified by the user, assume the impact
                    # applies across all 24 hours
                    try:
                        # Set the starting and stopping hours for the impact
                        start_stop_hr = [
                            tsv_adjustments[a][x] for x in [
                                "start_hour", "stop_hour"]]
                        # Generate a list of all hours of day in which the
                        # time-varying efficiency impact applies
                        try:
                            # Handle case where user specifies an overnight
                            # start/stop time (e.g., 11AM to 5AM)
                            if start_stop_hr[0] <= start_stop_hr[1]:
                                applicable_hrs = list(range(
                                    start_stop_hr[0] - 1, start_stop_hr[1]))
                            else:
                                applicable_hrs = list(range(
                                    0, start_stop_hr[1])) + \
                                    list(range(start_stop_hr[0], 24))
                        # Error sets applicable hour range to all day
                        except TypeError:
                            applicable_hrs = list(range(0, 24))
                    except (TypeError, KeyError):
                        applicable_hrs = list(range(0, 24))

                    # Set enumerate object for applicable day/hour ranges
                    enumerate_list = list(
                        enumerate(itertools.product(
                            range(365), range(24))))

                    # Apply time-varying impacts based on type of time-varying
                    # efficiency feature(s) specified for the measure

                    # "Shed" time-varying efficiency feature applies a %
                    # hourly load reduction across the user-specified hours
                    if "shed" in a:
                        # Determine the relative impact of the shed on the
                        # baseline load (specified as a change fraction)
                        # during the specified hour range
                        try:
                            rel_save_tsv = tsv_adjustments[a][
                                "relative energy change fraction"]
                        except KeyError:
                            rel_save_tsv = 0
                        # Reflect the shed impacts on efficient load shape
                        # across all relevant hours of the year
                        eff_load_hourly = [base_load_hourly[i] * (
                            1 - rel_save_tsv) if (
                            x in applicable_days and y in applicable_hrs) else
                            eff_load_hourly[i] for
                            i, (x, y) in enumerate_list]
                    # "Shift" time-varying efficiency features move a certain
                    # percentage of baseline load from one time period into
                    # another time period
                    elif "shift" in a:
                        # Set the number of hours earlier to shift the load
                        offset_hrs = tsv_adjustments[a]["offset_hrs_earlier"]
                        # If the user has not specified a time range for the
                        # load shifting, assume the measure shifts the entire
                        # load shape earlier by the number of hours set above
                        if len(applicable_hrs) == 24:
                            # Reflect load shifting in efficient load shape
                            # across all 8760 hours of the year; the initial
                            # efficient load in hour X is now the load in hour
                            # X minus user-specified hour offset
                            eff_load_hourly = [
                                base_load_hourly[i + offset_hrs] if ((
                                    i + offset_hrs) <= 8759 and
                                    x in applicable_days) else
                                base_load_hourly[(y + offset_hrs) - 24] for
                                i, (x, y) in enumerate_list]
                        # If the user has specified a time range for the load
                        # shifting, shift the load in accordance with range
                        else:
                            # Determine the relative amount of baseline load
                            # to shift out of the specified time window
                            try:
                                rel_save_tsv = tsv_adjustments[a][
                                    "relative energy change fraction"]
                            except KeyError:
                                rel_save_tsv = 0
                            # Determine the hour range to shift the load to (
                            # take the user-specified hour range and shift it
                            # backward by the user-specified number of hours
                            # to reflect e.g., pre-heating or cooling)
                            new_start, new_end = [
                                (x - offset_hrs) if (
                                    x - offset_hrs) >= 0 else
                                ((x - offset_hrs) + 24) for x in [
                                    min(applicable_hrs), max(applicable_hrs)]]
                            # Handle case where load is shifted to an overnight
                            # start/stop time (e.g., 11AM-5AM the next morning)
                            if new_start <= new_end:
                                hrs_to_shift_to = list(
                                    range(new_start, new_end + 1))
                            else:
                                hrs_to_shift_to = \
                                    list(range(new_start, 24)) + \
                                    list(range(0, new_end + 1))

                            # Reflect load shifting impacts on efficient load
                            # shape across all 8670 hours of the year; take the
                            # user-specified % of load in the user-specified
                            # hour range and move it X hours earlier, where X
                            # is determined by the "offset_hours" parameter
                            eff_load_hourly = [
                                    (base_load_hourly[i] + (
                                        base_load_hourly[
                                            i + offset_hrs] *
                                        rel_save_tsv)) if (
                                        ((i + offset_hrs) <= 8759 and
                                         x in applicable_days)
                                        and y in hrs_to_shift_to and
                                        y not in applicable_hrs) else
                                    (base_load_hourly[i] * (
                                      1 - rel_save_tsv) + (base_load_hourly[
                                        i + offset_hrs] *
                                        rel_save_tsv)) if (
                                        ((i + offset_hrs) <= 8759 and
                                         x in applicable_days)
                                        and y in hrs_to_shift_to and
                                        y in applicable_hrs) else
                                    (base_load_hourly[i] +
                                     base_load_hourly[
                                        (y + offset_hrs) - 24] * rel_save_tsv)
                                    if (y in hrs_to_shift_to and
                                        y not in applicable_hrs and
                                        x in applicable_days) else
                                    (base_load_hourly[i] * (
                                        1 - rel_save_tsv) + base_load_hourly[
                                        (y + offset_hrs) - 24] * rel_save_tsv)
                                    if (y in hrs_to_shift_to and
                                        y in applicable_hrs and
                                        x in applicable_days) else
                                    eff_load_hourly[i] * (
                                        1 - rel_save_tsv) if (
                                        y in applicable_hrs and
                                        x in applicable_days) else
                                    eff_load_hourly[i] for i, (x, y) in
                                    enumerate_list]

                    # "Shape" time-sensitive efficiency features reshape
                    # the baseline load shape in accordance with custom load
                    # savings shape information specified either for a typical
                    # day or across all 8760 hours of the year
                    elif "shape" in a:
                        # Custom daily load savings shape information is
                        # applied across all applicable days of the year
                        if "custom_daily_savings" in \
                            tsv_adjustments[a].keys() and tsv_adjustments[a][
                                "custom_daily_savings"] is not None:
                            # Set the custom daily savings shape (each element
                            # of the list represents hourly savings fraction)
                            custom_save_shape = \
                                tsv_adjustments[a]["custom_daily_savings"]
                            # Reflect custom load savings in efficient load
                            # shape
                            eff_load_hourly = [
                                base_load_hourly[i] * (
                                    1 - custom_save_shape[y]) if (
                                    x in applicable_days) else
                                eff_load_hourly[i] for
                                i, (x, y) in enumerate_list]

                        # Custom annual load savings shape information contains
                        # savings fractions for all 8760 hours of the year
                        elif "custom_annual_savings" in \
                            tsv_adjustments[a].keys() and tsv_adjustments[a][
                                "custom_annual_savings"] is not None:
                            # Set the custom 8760 savings shape data, which are
                            # stored in a dict initialized with the measure
                            # from a CSV, where the dict will be keyed in by
                            # the current combination of end use, building
                            # type, and climate zone to yield the 8760 shape
                            try:
                                custom_hr_save_shape = tsv_adjustments[a][
                                    "custom_annual_savings"][eu][
                                    load_fact_bldg_key][load_fact_climate_key][
                                    mult_sysshp_key_save]
                            except KeyError:
                                # Try again with the assumption that there is
                                # only one savings shape version per climate,
                                # building, and end use combination (e.g.,
                                # this will be the case for EE measures, vs.
                                # DR measures which may respond to more than
                                # one peak period range)
                                try:
                                    custom_hr_save_shape = tsv_adjustments[a][
                                        "custom_annual_savings"][eu][
                                        load_fact_bldg_key][
                                        load_fact_climate_key]["set 1"]
                                except KeyError:
                                    custom_hr_save_shape = []

                            # Ensure that the resultant custom savings shape
                            # for the current combination of climate zone,
                            # building type, and end use is not zero elements
                            # or all zeroes in the baseline. If the savings
                            # shape is zero elements or has an all zeroes
                            # baseline, warn the user (in verbose mode) that
                            # no savings shape data were found for the current
                            # climate/building/end use combination and that the
                            # savings will be assumed to be zero relative to
                            # the baseline load.
                            if len(custom_hr_save_shape) == 0 or sum(
                                    custom_hr_save_shape[
                                        "CSV base frac. annual"]) == 0:
                                verboseprint(
                                    opts.verbose,
                                    "Measure '" + self.name + "', requires "
                                    "custom savings shape data, but none were "
                                    "found or all values were zero for the "
                                    "combination of climate "
                                    "zone " + load_fact_climate_key +
                                    " (system load " + mult_sysshp_key_save +
                                    "), building type " +
                                    load_fact_bldg_key + ", and end use " +
                                    eu + ". Assuming savings are zero for "
                                    "this combination. If this is "
                                    "unexpected, check that 8760 hourly "
                                    "savings fractions are available for "
                                    "all baseline market segments the "
                                    "measure applies to in "
                                    f"{fp.ECM_DEF / 'energy_plus_data' / 'savings_shapes'}.")
                            else:
                                # Develop an adjustment from the generic
                                # baseline load shape for the current climate,
                                # building type, and end use combination to the
                                # baseline load shape that is specific to the
                                # measure in question, which the measure load
                                # shape is calculated relative to in input CSVs
                                meas_base_adj = [(
                                    custom_hr_save_shape[
                                        "CSV base frac. annual"][x] /
                                    base_load_hourly[x]) if (
                                        numpy.isfinite(base_load_hourly[x]) and
                                        base_load_hourly[x] != 0) else 1
                                    for x in range(8760)]
                                # Pull in relative hourly savings fractions to
                                # apply to baseline to get to efficient shape
                                hr_chg = custom_hr_save_shape[
                                    "CSV relative change"]
                                # Apply hourly baseline adjustment and relative
                                # load change to derive efficient shape
                                eff_load_hourly = [(
                                    base_load_hourly[x] * meas_base_adj[x] *
                                    hr_chg[x]) for x in range(8760)]
                                # Ensure all efficient load fractions are
                                # greater than zero
                                eff_load_hourly = [
                                    eff_load_hourly[x] if
                                    eff_load_hourly[x] >= 0
                                    else 0 for x in range(8760)]
                        else:
                            # Throw an error if the load reshaping operation
                            # name is invalid
                            raise KeyError(
                                "Invalid load reshaping operation name for "
                                "measure: " + self.name + ". Valid reshaping "
                                "operations include 'custom_daily_savings' "
                                "or 'custom_annual_savings'.")
                    # Yield error if unexpected TSV feature type is present
                    else:
                        raise KeyError(
                            "Invalid feature type present in tsv_features "
                            "attribute for measure " + self.name +
                            " - valid types include: " +
                            str(self.handyvars.tsv_feature_types))
                # Update hourly fractions of annual baseline and efficient
                # energy to reflect baseline hourly load shape plus effects of
                # time-sensitive measure features on the baseline load (if any)
                if opts.sect_shapes is True:
                    # Base load weighted by contribution of climate for load
                    # to EMM region
                    base_ls_wt = [
                        x * emm_adj_wt for x in base_load_hourly]
                    # Efficient load weighted by contribution of climate for
                    # load to current EMM region
                    eff_ls_wt = [
                        x * emm_adj_wt for x in eff_load_hourly]
                    # Add to existing base load fractions (across all climates
                    # that overlap with the current EMM region)
                    energy_base_shape = [
                        energy_base_shape[x] + base_ls_wt[x] for
                        x in range(len(energy_base_shape))]
                    # Add to existing efficient load fractions (across all
                    # climates that overlap with the current EMM region)
                    energy_eff_shape = [
                        energy_eff_shape[x] + eff_ls_wt[x] for
                        x in range(len(energy_eff_shape))]

                # Further adjust baseline and efficient load shapes
                # to account for time sensitive valuation (TSV) output metrics
                if opts.tsv_metrics is not False:

                    # Set legible name for each TSV metrics input

                    # Output type (energy/power)
                    if opts.tsv_metrics[0] == "1":
                        output = "energy"
                    else:
                        output = "power"
                    # Applicable hours of focus (all/peak/take)
                    if opts.tsv_metrics[1] == "1":
                        hours = "all"
                    elif opts.tsv_metrics[1] == "2":
                        hours = "peak"
                    else:
                        hours = "take"
                    # Applicable season of focus (summer/winter/intermediate)
                    if opts.tsv_metrics[2] == '1':
                        season = "summer"
                    elif opts.tsv_metrics[2] == '2':
                        season = "winter"
                    elif opts.tsv_metrics[2] == '3':
                        season = "intermediate"
                    # Type of calculation (sum/max/avg depending on output)
                    if output == "energy" and opts.tsv_metrics[3] == "1":
                        calc = "sum"
                    elif output == "power" and opts.tsv_metrics[3] == "1":
                        calc = "max"
                    else:
                        calc = "avg"
                    # Days to perform operations over (applicable to sum
                    # and averaging calculations)
                    if opts.tsv_metrics[-1] == "1":
                        days = "all"
                    elif opts.tsv_metrics[-1] == "2":
                        days = "weekdays"
                    elif opts.tsv_metrics[-1] == "3":
                        days = "weekends"
                    else:
                        days = None

                    # Set applicable day range

                    # Sum or average calcs (spans multiple days)
                    if calc == "sum" or calc == "avg":
                        tsv_metrics_days = self.handyvars.tsv_metrics_data[
                            "season days"][days][season]
                    # Maximum calc type (pertains only to a peak day)
                    else:
                        tsv_metrics_days = [
                            self.handyvars.tsv_metrics_data[
                                "peak days"][season][cz[0]]]

                    # Set applicable daily hour range
                    # NOTE: for now, use peak/take periods from 2050 only

                    # Sum or average calc type (spans multiple hours)
                    if calc in ["sum", "avg"]:
                        # All hours of the day
                        if hours == "all":
                            tsv_metrics_hrs = list(range(1, 25))
                        # Peak hours only
                        elif hours == "peak":
                            # Handle one set of net load shape data differently
                            # than multiple sets
                            if mult_sysshp is False:
                                tsv_metrics_hrs = \
                                    self.handyvars.tsv_metrics_data[
                                        "system load hours"][season]["2050"][
                                        cz[0]]["peak range"]
                            else:
                                tsv_metrics_hrs = \
                                    self.handyvars.tsv_metrics_data[
                                        "system load hours"][season]["2050"][
                                        cz[0]][mult_sysshp_key_metrics][
                                        "peak range"]
                        # Take hours only
                        else:
                            # Handle one set of net load shape data differently
                            # than multiple sets
                            if mult_sysshp is False:
                                tsv_metrics_hrs = \
                                    self.handyvars.tsv_metrics_data[
                                        "system load hours"][season]["2050"][
                                        cz[0]]["take range"]
                            else:
                                tsv_metrics_hrs = \
                                    self.handyvars.tsv_metrics_data[
                                        "system load hours"][season]["2050"][
                                        cz[0]][mult_sysshp_key_metrics][
                                        "take range"]
                    # Maximum calc type (pertains only to a max/min hour)
                    else:
                        # All hours (assume max peak hour) or max peak hour
                        if hours == "all" or hours == "peak":
                            # Handle one set of net load shape data differently
                            # than multiple sets
                            if mult_sysshp is False:
                                tsv_metrics_hrs = [
                                    self.handyvars.tsv_metrics_data[
                                        "system load hours"][season]["2050"][
                                        cz[0]]["max"]]
                            else:
                                tsv_metrics_hrs = [
                                    self.handyvars.tsv_metrics_data[
                                        "system load hours"][season]["2050"][
                                        cz[0]][mult_sysshp_key_metrics]["max"]]
                        # Min take hour
                        else:
                            # Handle one set of net load shape data differently
                            # than multiple sets
                            if mult_sysshp is False:
                                tsv_metrics_hrs = [
                                    self.handyvars.tsv_metrics_data[
                                        "system load hours"][season]["2050"][
                                        cz[0]]["min"]]
                            else:
                                tsv_metrics_hrs = [
                                    self.handyvars.tsv_metrics_data[
                                        "system load hours"][season]["2050"][
                                        cz[0]][mult_sysshp_key_metrics]["min"]]

                    # Determine number of days to average over; if peak day
                    # only, set this number to one
                    if calc == 'avg':
                        # Power (single hour) output case; divide by total
                        # applicable hours for the appropriate season times
                        # total applicable days
                        if output == "power":
                            avg_len = len(tsv_metrics_days) * \
                                      len(tsv_metrics_hrs)
                        # Energy (multiple hour) output case; divide by total
                        # applicable days
                        else:
                            avg_len = len(tsv_metrics_days)
                    else:
                        avg_len = 1

                    # Adjust the baseline and efficient loads to reflect
                    # only the hourly values that fall within the applicable
                    # hour and day ranges from above; set all inapplicable
                    # values to zero (to maintain full 8760 list length)
                    base_load_hourly, eff_load_hourly = [[
                        (x[i] / avg_len) if (
                            (d + 1) in tsv_metrics_days and
                            (h + 1) in tsv_metrics_hrs)
                        else 0 for
                        i, (d, h) in
                        self.handyvars.tsv_metrics_data["hourly index"]] for
                        x in [base_load_hourly, eff_load_hourly]]

                    # Sum across all 8760 hourly baseline and efficient load
                    # values to arrive at final factor used to rescale
                    # annually-determined energy totals
                    energy_scale_base += numpy.sum([
                        x * emm_adj_wt for x in base_load_hourly])
                    energy_scale_eff += numpy.sum([
                        x * emm_adj_wt for x in eff_load_hourly])
                else:
                    # If no tsv metrics are specified, annually-determined
                    # baseline energy total requires no rescaling (8760
                    # baseline hourly values sum to 1)
                    energy_scale_base = 1
                    # Sum across all 8760 hourly efficient load values
                    # to arrive at final factor used to rescale annually-
                    # determined energy totals
                    energy_scale_eff += numpy.sum([
                        x * emm_adj_wt for x in eff_load_hourly])

        # Finalize carbon/cost scaling factor variables, but only if
        # either measure TSV features are present or the user desires
        # TSV metrics outputs; assume these shapes are not necessary if
        # the user only desires sector-level load shapes
        if opts.tsv_metrics is not False or self.tsv_features is not None:

            # Calculate baseline/efficient cost rescaling factors as the sums
            # of the hourly baseline/efficient load shape multiplied by the
            # hourly price scaling factors; calculate across available
            # projection years for the price scaling factors
            for yr in cost_scale_base.keys():
                cost_scale_base[yr] += numpy.sum([
                    x * y for x, y in zip(
                        base_load_hourly, cost_fact_hourly[yr])])
                cost_scale_eff[yr] += numpy.sum([
                    x * y for x, y in zip(
                        eff_load_hourly, cost_fact_hourly[yr])])

            # Calculate baseline/efficient emissions rescaling factors as the
            # sums of the hourly baseline/efficient load shape multiplied by
            # the hourly emissions scaling factors; calculate across available
            # projection years for the emissions scaling factors
            for yr in carb_scale_base.keys():
                carb_scale_base[yr] += numpy.sum([
                    x * y for x, y in zip(
                        base_load_hourly, carbon_fact_hourly[yr])])
                carb_scale_eff[yr] += numpy.sum([
                    x * y for x, y in zip(
                        eff_load_hourly, carbon_fact_hourly[yr])])

            # Extend price/emissions factors across all years in the AEO time
            # horizon
            for yr in self.handyvars.aeo_years:
                # Find year in the cost scaling factors data that maps to the
                # current AEO year
                tsv_yr_cost = [
                    x[0] for x in cost_yr_map.items() if yr in x[1]][0]
                # Find year in the emissions scaling factors data that maps to
                # the current AEO year
                tsv_yr_carb = [
                    x[0] for x in carb_yr_map.items() if yr in x[1]][0]
                # Finalize the cost/emissions scaling factors data for the
                # current AEO year

                # APPROACH 1: TAKE THE DATA AS IS
                cost_scale_base_aeo[yr], cost_scale_eff_aeo[yr], \
                    carb_scale_base_aeo[yr], carb_scale_eff_aeo[yr] = [
                    cost_scale_base[tsv_yr_cost],
                    cost_scale_eff[tsv_yr_cost],
                    carb_scale_base[tsv_yr_carb],
                    carb_scale_eff[tsv_yr_carb]]
                # # APPROACH 2: RUNNING MEAN
                # if yr == self.handyvars.aeo_years[0]:
                #     cost_scale_base_aeo[yr], cost_scale_eff_aeo[yr], \
                #         carb_scale_base_aeo[yr], carb_scale_eff_aeo[yr] = [
                #         cost_scale_base[tsv_yr_cost],
                #         cost_scale_eff[tsv_yr_cost],
                #         carb_scale_base[tsv_yr_carb],
                #         carb_scale_eff[tsv_yr_carb]]
                # else:
                #     cost_scale_base_aeo[yr], cost_scale_eff_aeo[yr], \
                #         carb_scale_base_aeo[yr], carb_scale_eff_aeo[yr] = [
                #         x * (1 / len(self.handyvars.aeo_years)) + y * (
                #             1 - (1 / len(self.handyvars.aeo_years))) for
                #         x, y in zip([
                #             cost_scale_base[tsv_yr_cost],
                #             cost_scale_eff[tsv_yr_cost],
                #             carb_scale_base[tsv_yr_carb],
                #             carb_scale_eff[tsv_yr_carb]], [
                #             cost_scale_base_aeo[str(int(yr) - 1)],
                #             cost_scale_eff_aeo[str(int(yr) - 1)],
                #             carb_scale_base_aeo[str(int(yr) - 1)],
                #             carb_scale_eff_aeo[str(int(yr) - 1)]])]
        else:
            cost_scale_base_aeo, cost_scale_eff_aeo, carb_scale_base_aeo, \
                carb_scale_eff_aeo = (1 for n in range(4))

        # After calculations are complete, if measure fuel switches from
        # fossil to electricity and the current baseline microsegment indicates
        # fossil fuel, set all 8760 baseline values to zero (it is assumed that
        # sector-level load shapes will only be relevant to the electricity
        # system, which will not see a baseline load that is satisfied by
        # non-electric fuel (from this perspective, measures that fuel switch
        # heating or water heating to electricity add load to the electricity
        # system that wasn't there before)
        if self.fuel_switch_to == "electricity" and \
                "electricity" not in mskeys and opts.sect_shapes is True:
            energy_base_shape = [0 for x in range(len(energy_base_shape))]

        # Return hourly fractions of annual baseline and efficient energy
        # if sector-level load shape information is desired by the user
        if opts.sect_shapes is True:
            updated_tsv_shapes = {
                "baseline": energy_base_shape,
                "efficient": energy_eff_shape}
        else:
            updated_tsv_shapes = None
        # Return the final energy, cost, and emissions rescaling factors
        # to use in modifying annually-determined energy, cost, and emissions
        # totals such that they reflect sub-annual assessment of these totals
        updated_tsv_fracs = {
            "energy": {
                "baseline": energy_scale_base,
                "efficient": energy_scale_eff},
            "cost": {
                "baseline": cost_scale_base_aeo,
                "efficient": cost_scale_eff_aeo
            },
            "carbon": {
                "baseline": carb_scale_base_aeo,
                "efficient": carb_scale_eff_aeo
            }}

        return updated_tsv_fracs, updated_tsv_shapes

    def convert_costs(self, convert_data, bldg_sect, mskeys, cost_meas,
                      cost_meas_units, cost_base_units, verbose):
        """Convert measure cost to comparable baseline cost units.

        Args:
            convert_data (dict): Measure cost unit conversions.
            bldg_sect (string): Applicable building sector for measure cost.
            mskeys (tuple): Full applicable market microsegment information for
                measure cost (mseg type->reg->bldg->fuel->end use->technology
                type->structure type).
            cost_meas (float): Initial user-defined measure cost.
            cost_meas_units (string): Initial user-defined measure cost units.
            cost_base_units (string): Comparable baseline cost units.
            verbose (bool or NoneType): Determines whether to print all
                user warnings and messages.

        Returns:
            Updated measure costs and cost units that are consistent with
            baseline technology cost units.

        Raises:
            KeyError: If no cost conversion data are available for
                the particular measure microsegment
            ValueError: If initial user-defined measure cost units are
                determined to be invalid/unsupported.
        """
        # Separate cost units into the cost year and everything else
        cost_meas_units_unpack, cost_base_units_unpack = [re.search(
            r'(\d*)(.*)', x) for x in [cost_meas_units, cost_base_units]]
        # Establish measure and baseline cost year
        cost_meas_yr, cost_base_yr = \
            cost_meas_units_unpack.group(1), cost_base_units_unpack.group(1)
        # Where there is no measure or base cost year given in units, assume
        # current year (which will be the first in the model time horizon)
        if not cost_meas_yr:
            cost_meas_yr = self.handyvars.aeo_years[0]
        if not cost_base_yr:
            cost_base_yr = self.handyvars.aeo_years[0]
            # Ensure that final baseline units include year addition
            cost_base_units_fin = cost_base_yr + cost_base_units
        else:
            cost_base_units_fin = cost_base_units
        # Establish measure and baseline cost units (excluding cost year)
        cost_meas_noyr, cost_base_noyr = \
            cost_meas_units_unpack.group(2), cost_base_units_unpack.group(2)

        # Set flag for special case where residential non-envelope cost #
        # units are given in $/ft^2 floor and must be converted to $/unit;
        # this case only requires a cost year conversion (conversion between
        # $/ft^@ floor and $/unit handled separately from this function)
        res_sf_unit = (
            bldg_sect == "residential" and "$/ft^2 floor" in
            cost_meas_noyr and "$/unit" in cost_base_noyr)

        # If the cost units of the measure are inconsistent with the baseline
        # cost units (with the cost year removed), map measure cost units to
        # baseline cost units
        if cost_meas_noyr != cost_base_noyr and res_sf_unit is False:
            # Retrieve end use-specific cost unit conversion
            # data. Conversion data should be formatted in a list to simplify
            # its handling in subsequent operations

            # Find top-level key for retrieving data from cost translation
            # dictionary
            top_keys = convert_data['cost unit conversions'].keys()
            try:
                # Handle special case of secondary heating, which should be
                # mapped to heating end use cost translation data
                if mskeys[4] == "secondary heating":
                    top_key = [x for x in top_keys if "heating" in x][0]
                # Otherwise, use microsegment end use to pull top-level key
                else:
                    top_key = [x for x in top_keys if mskeys[4] in x][0]
            except IndexError:
                raise KeyError("No conversion data for ECM '" +
                               self.name + "' cost units '" +
                               cost_meas_units + "'")

            # Handle special case where top-level key equals heating/cooling
            if top_key == "heating and cooling":
                # Determine whether HVAC equipment (supply) or envelope
                # (demand) microsegment is being addressed
                if "demand" in mskeys:
                    htcl_key = "demand"
                else:
                    htcl_key = "supply"
                # Pull out appropriate HVAC equipment or envelope cost
                # conversion data
                if htcl_key == "supply":
                    # Retrieve supply-side heating/cooling conversion data
                    supply_dat = convert_data[
                        'cost unit conversions'][top_key][htcl_key]
                    # Handle special case of secondary heating, which
                    # should be mapped to heating end use cost data
                    try:
                        if mskeys[4] == "secondary heating":
                            supply_key = [
                                x for x in supply_dat.keys() if "heating" in x
                                and cost_meas_noyr in [
                                    supply_dat[x]["revised units"],
                                    supply_dat[x]["original units"]]][0]
                        # Otherwise, use microsegment end use to pull key
                        else:
                            supply_key = [
                                x for x in supply_dat.keys() if mskeys[4] in x
                                and cost_meas_noyr in [
                                    supply_dat[x]["revised units"],
                                    supply_dat[x]["original units"]]][0]
                    except IndexError:
                        raise KeyError("No conversion data for ECM '" +
                                       self.name + "' cost units " +
                                       cost_meas_units + "'")
                    # Finalize data
                    convert_units_data = [
                        convert_data['cost unit conversions'][top_key][
                            htcl_key][supply_key]]
                else:
                    # Retrieve demand-side heating/cooling conversion data
                    demand_keys = self.handyvars.cconv_tech_mltstage_map
                    try:
                        demand_key = next(
                            x for x in demand_keys.keys() if
                            cost_meas_noyr in demand_keys[x]['key'])
                    except StopIteration:
                        raise KeyError("No conversion data for ECM '" +
                                       self.name + "' cost units" +
                                       cost_meas_units + "'")
                    convert_units_data = [
                        convert_data['cost unit conversions'][top_key][
                            htcl_key][x] for x in
                        demand_keys[demand_key]["conversion stages"]]
            else:
                # Retrieve conversion data for costs outside of the
                # heating/cooling cases
                convert_units_data = [
                    convert_data['cost unit conversions'][top_key]]

            # Finalize cost conversion information retrieved above

            # Initialize a final cost conversion value for the microsegment
            convert_units = 1
            # Loop through list of conversion data, multiplying initial
            # conversion value by successive list elements and weighting
            # results as needed to map conversion factors for multiple
            # non-Scout building types to the single Scout building type of
            # the current microsegment
            for cval in convert_units_data:
                if cost_meas_noyr not in cval["revised units"]:
                    new_units = 'revised units'
                    invert_cval = False
                else:
                    new_units = 'original units'
                    invert_cval = True

                # Case where conversion data is split by building sector
                if isinstance(cval['conversion factor']['value'], dict):
                    # Restrict to building sector of current microsegment
                    cval_bldgtyp = \
                        cval['conversion factor']['value'][bldg_sect]
                    # Case where conversion data is further nested by
                    # Scout building type and EnergyPlus building type
                    if isinstance(cval_bldgtyp, dict) and isinstance(
                            cval_bldgtyp[mskeys[2]], dict):
                        # Develop weighting factors to map conversion data
                        # from multiple EnergyPlus building types to the
                        # single Scout building type of the current
                        # microsegment

                        # Invert conversion in case where measure cost
                        # units are found under the "revised units" key (
                        # e.g., if the conversion should be reversed)
                        if invert_cval is False:
                            cval_bldgtyp = cval_bldgtyp[mskeys[2]].values()
                        else:
                            cval_bldgtyp = [
                                1 / x for x
                                in cval_bldgtyp[mskeys[2]].values()]
                        bldgtyp_wts = convert_data[
                            'building type conversions'][
                            'conversion data']['value'][bldg_sect][
                            mskeys[2]].values()
                        convert_units *= sum([a * b for a, b in zip(
                            cval_bldgtyp, bldgtyp_wts)])
                    # Case where conversion data is further nested by
                    # Scout building type
                    elif isinstance(cval_bldgtyp, dict):
                        # Invert conversion in case where measure cost
                        # units are found under the "revised units" key (
                        # e.g., if the conversion should be reversed)
                        if invert_cval is False:
                            cval_bldgtyp = cval_bldgtyp[mskeys[2]]
                        else:
                            cval_bldgtyp = 1 / cval_bldgtyp[mskeys[2]]
                        convert_units *= cval_bldgtyp
                    # Case where conversion data is not nested further
                    else:
                        # Invert conversion in case where measure cost
                        # units are found under the "revised units" key (
                        # e.g., if the conversion should be reversed)
                        if invert_cval is True:
                            cval_bldgtyp = 1 / cval_bldgtyp
                        convert_units *= cval_bldgtyp
                else:
                    if invert_cval is False:
                        convert_units *= cval['conversion factor']['value']
                    else:
                        convert_units *= (
                            1 / cval['conversion factor']['value'])
        else:
            convert_units = 1
            new_units = None

        # Map the year of measure cost units to the year of baseline cost units

        # If measure and baseline cost years are inconsistent, map measure
        # to baseline cost year using Consumer Price Index (CPI) data
        if cost_meas_yr != cost_base_yr:
            # Set full CPI dataset
            cpi = self.handyvars.consumer_price_ind
            # Find array of rows in CPI dataset associated with the measure
            # cost year
            cpi_row_meas = [x[1] for x in cpi if cost_meas_yr in x['DATE']]
            # Average across all rows for a year, or if year wasn't found,
            # choose the latest available row in the data
            if len(cpi_row_meas) == 0:
                cpi_row_meas = cpi[-1][1]
            else:
                cpi_row_meas = numpy.mean(cpi_row_meas)
            # Find array of rows in CPI dataset associated with the baseline
            # cost year
            cpi_row_base = [x[1] for x in cpi if cost_base_yr in x['DATE']]
            # Average across all rows for a year, or if year wasn't found,
            # choose the latest available row in the data
            if len(cpi_row_base) == 0:
                cpi_row_base = cpi[-1][1]
            else:
                cpi_row_base = numpy.mean(cpi_row_base)
            # Calculate year conversion ratio
            convert_yr = cpi_row_base / cpi_row_meas
        else:
            convert_yr = 1

        # Apply finalized cost conversion and year conversion factors
        # to measure costs to map to baseline cost units
        cost_meas_fin = cost_meas * convert_units * convert_yr

        # Adjust initial measure cost units to reflect the conversion (should
        # now be consistent with baseline cost units, this is checked in
        # a subsequent step of the 'fill_mkts' routine)

        # Cost year/unit adjustment involving multiple conversion stages
        # (use the revised measure units for the final conversion stage)
        if new_units and isinstance(convert_units_data, list):
            cost_meas_units_fin = cost_base_yr + \
                convert_units_data[-1][new_units]
        # Cost year/unit adjustments involving a single stage of conversion
        elif new_units:
            cost_meas_units_fin = cost_base_yr + \
                convert_units_data[new_units]
        # Cost year adjustment only
        else:
            cost_meas_units_fin = cost_base_yr + cost_meas_noyr

        # Case where cost conversion has succeeded
        if cost_meas_units_fin == cost_base_units_fin or res_sf_unit is True:
            # If in verbose mode, notify user of cost conversion details
            if verbose:
                # Set base user message
                if not isinstance(cost_meas, numpy.ndarray):
                    user_message = "ECM '" + self.name + \
                        "' cost converted from " + \
                        str(cost_meas) + " " + cost_meas_units + " to " + \
                        str(round(cost_meas_fin, 3)) + " " + \
                        cost_meas_units_fin
                else:
                    user_message = "ECM '" + self.name + \
                        "' cost converted from " + \
                        str(numpy.mean(cost_meas)) + " " + cost_meas_units + \
                        " to " + str(round(numpy.mean(cost_meas_fin), 3)) + \
                        " " + cost_meas_units_fin
                # Add building type information to base message in cases where
                # cost conversion depends on building type (e.g., for envelope
                # components)
                if (cost_meas_noyr in self.handyvars.cconv_bybldg_units or
                        isinstance(self.installed_cost, dict)):
                    user_message += " for building type '" + mskeys[2] + "'"

                # Print user message
                print(user_message)
        # Case where cost conversion has not succeeded
        else:
            raise ValueError(
                "ECM '" + self.name + "' cost units '" +
                str(cost_meas_units_fin) + "' not equal to base units '" +
                str(cost_base_units_fin) + "'")

        return cost_meas_fin, cost_meas_units_fin, cost_base_units_fin

    def partition_microsegment(
            self, adopt_scheme, diffuse_params, mskeys, bldg_sect, sqft_subst,
            mkt_scale_frac, new_constr, stock_total_init, energy_total_init,
            carb_total_init, fmeth_total_init, f_refr, cost_base, cost_meas,
            cost_energy_base, cost_energy_meas, rel_perf, life_base, life_meas,
            site_source_conv_base, site_source_conv_meas, intensity_carb_base,
            intensity_carb_meas, energy_total_scnd, tsv_adj_init,
            tsv_shapes, opts, contrib_mseg_key, ctrb_ms_pkg_prep, hp_rate,
            retro_rate_mseg, calc_sect_shapes, lkg_fmeth_base, lkg_fmeth_meas,
            warn_list):
        """Find total, competed, and efficient portions of a mkt. microsegment.

        Args:
            adopt_scheme (string): Assumed consumer adoption scenario.
            diffuse_params (NoneType): Parameters relating to the 'adjusted
                adoption' consumer choice model (currently a placeholder).
            mskeys (tuple): Dictionary key information for the currently
                partitioned market microsegment (mseg type->reg->bldg->
                fuel->end use->technology type->structure type).
            bldg_sect (str): Flag for residential or commercial.
            sqft_subst (int): Flag for use of square feet as a stock basis.
            mkt_scale_frac (float): Microsegment scaling fraction (used to
                break market microsegments into more granular sub-markets).
            new_constr (dict): Data needed to determine the portion of the
                total microsegment stock that is added in each year.
            stock_total_init (dict): Baseline technology stock, by year.
            energy_total_init (dict): Baseline microsegment primary energy use,
                by year.
            carb_total_init (dict): Baseline microsegment carbon emissions,
                by year.
            fmeth_total_init (dict or NoneType): Baseline microsegment
                fugitive emissions from methane, by year.
            f_refr (dict or NoneType): Supporting data for calculating fugitive
                emissions from refrigerants for the current microsegment.
            cost_base (dict): Baseline technology installed cost, by year.
            cost_meas (float): Measure installed cost, by year.
            cost_energy_base (dict): Baseline fuel cost, by year.
            cost_energy_meas (dict): Measure fuel cost, by year.
            rel_perf (float): Measure performance relative to baseline.
            life_base (dict): Baseline technology lifetime.
            life_meas (float): Measure lifetime.
            site_source_conv_base (dict): Baseline fuel site-source conversion,
                by year.
            site_source_conv_meas (dict): Measure fuel site-source conversion,
                by year.
            intensity_carb_base (dict): Baseline fuel carbon intensity,
                by year.
            intensity_carb_meas (dict): Measure fuel carbon intensity, by year.
            tsv_adj_init (dict): Adj. for time sensitive efficiency valuation.
            tsv_shapes (dict): 8760 hourly adjustments (sum to tsv_adj values)
            opts (object): Stores user-specified execution options.
            contrib_mseg_key (tuple): The same as mskeys, but adjusted to merge
                windows solar/conduction msegs into "windows" if applicable.
            ctrb_ms_pkg_prep (list): Names of measures that contribute to pkgs.
            hp_rate (dict): Exogenous rate of conversion of the baseline mseg
                to HPs, if applicable.
            retro_rate_mseg (dict): Microsegment-specific retrofit rate.
            calc_sect_shapes (boolean): Flag for sector-shape calculations.
            lkg_fmeth_base (float): Methane leakage for baseline mseg tech.,
                if applicable.
            lkg_fmeth_meas (float): Methane leakage for measure tech. that is
                replacing baseline mseg, if applicable.

        Returns:
            Total, total-efficient, competed, and competed-efficient
            stock, energy, carbon, and cost market microsegments by year; for
            fuel switching measures, also reports out any remaining
            (unswitched) energy, carbon, and cost segments by year.
        """
        # Initialize stock, energy, and carbon mseg partition dicts, where the
        # dict keys will be years in the modeling time horizon
        stock_total, stock_total_sbmkt, energy_total, carb_total, \
            energy_total_sbmkt, carb_total_sbmkt, stock_total_meas, \
            stock_comp_cum_sbmkt, stock_total_hp_convert, energy_total_eff, \
            carb_total_eff, stock_compete, stock_compete_meas, \
            stock_compete_sbmkt, stock_comp_hp_convert, stock_comp_hp_remain, \
            energy_compete, carb_compete, energy_compete_sbmkt, \
            carb_compete_sbmkt, energy_compete_eff, carb_compete_eff, \
            stock_total_cost, energy_total_cost, carb_total_cost, \
            stock_total_cost_eff, energy_total_eff_cost, \
            carb_total_eff_cost, stock_compete_cost, energy_compete_cost, \
            carb_compete_cost, stock_compete_cost_eff, \
            energy_compete_cost_eff, carb_compete_cost_eff, \
            mkt_scale_frac_fin = ({} for n in range(35))

        # Case needing fugitive methane assessment where current mseg
        # has fugitive methane
        if opts.fugitive_emissions is not False and \
            opts.fugitive_emissions[0] in ['1', '3'] and any([
                x[1] != 0 for x in fmeth_total_init.items()]):
            # Initialize fugitive methane dicts for further calculations
            fmeth_total_sbmkt, fmeth_total, fmeth_total_eff, fmeth_compete, \
                fmeth_compete_sbmkt, fmeth_compete_eff = ({} for n in range(6))
            # Flag further assessment of fugitive methane in this function
            f_meth_assess = True
        # Case needing fugitive methane assessment where current mseg
        # is not relevant to fugitive methane calcs.
        elif opts.fugitive_emissions is not False and \
                opts.fugitive_emissions[0] in ['1', '3']:
            # Set fugitive methane dicts to all zeros
            fmeth_total_sbmkt, fmeth_total, fmeth_total_eff, fmeth_compete, \
                fmeth_compete_sbmkt, fmeth_compete_eff = (
                    {yr: 0 for yr in self.handyvars.aeo_years} for
                    n in range(6))
            # No further assessment of fugitive methane required in function
            f_meth_assess = ""
        # Case not needing fugitive methane assessment
        else:
            # All fugitive methane dicts are set to None
            fmeth_total_sbmkt, fmeth_total, fmeth_total_eff, fmeth_compete, \
                fmeth_compete_sbmkt, fmeth_compete_eff = (
                    None for n in range(6))
            # No further assessment of fugitive methane required in function
            f_meth_assess = ""

        # Case needing fugitive refrigerants assessment where current mseg
        # has fugitive refrigerants
        if opts.fugitive_emissions is not False and \
            opts.fugitive_emissions[0] in ['2', '3'] and \
                f_refr is not None:
            # Initialize fugitive refrigerant dicts for further calculations
            frefr_total, frefr_total_eff, frefr_compete, frefr_compete_eff = (
                {} for n in range(4))
            # Flag further assessment of fugitive refrigerants in this function
            f_refr_assess = True
        # Case needing fugitive refrigerant assessment where current mseg
        # is not relevant to fugitive refrigerant calcs.
        elif opts.fugitive_emissions is not False and \
                opts.fugitive_emissions[0] in ['2', '3']:
            # Set fugitive refrigerants dicts to all zeros
            frefr_total, frefr_total_eff, frefr_compete, frefr_compete_eff = ({
                    yr: 0 for yr in self.handyvars.aeo_years} for
                    n in range(4))
            # No further assessment of fugitive refrig. required in function
            f_refr_assess = ""
        # Case not needing fugitive refrigerant assessment
        else:
            # All fugitive refrigerant dicts are set to None
            frefr_total, frefr_total_eff, frefr_compete,  frefr_compete_eff = (
                None for n in range(4))
            # No further assessment of fugitive refrig. required in function
            f_refr_assess = ""

        # Initialize the portion of microsegment already captured by the
        # efficient measure as 0, the cumulative portion of the microsegment
        # already competed by the current year as 0, the cumulative portion
        # of a fuel switching microsegment already converted to a HP by current
        # year as 0, and a flag for whether full saturation with competed
        # measures has not been achieved as True
        meas_cum_frac = 0
        comp_cum_frac = 0
        stock_total_hp_convert_frac = 0
        turnover_cap_not_reached = True

        # Initialize flag for whether measure is on the market in a given year
        # as false
        measure_on_mkt, measure_exited_mkt = ("" for n in range(2))

        # Initialize variables that capture the portion of baseline
        # energy, carbon, and energy cost that remains with the baseline fuel
        # (for fuel switching measures only)
        fs_energy_eff_remain, fs_carb_eff_remain, fs_energy_cost_eff_remain = (
            {yr: 0 for yr in self.handyvars.aeo_years} for n in range(3))

        # In cases where secondary microsegments are present, initialize a
        # dict of year-by-year secondary microsegment adjustment information
        # that will be used to scale down the secondary microsegment(s) in
        # accordance with the portion of the total applicable primary stock
        # that is captured by the measure in each year
        if energy_total_scnd is not False or mskeys[0] == "secondary":
            # Set short names for secondary adjustment information dicts
            secnd_adj_sbmkt = self.markets[adopt_scheme]["mseg_adjust"][
                "secondary mseg adjustments"]["sub-market"]
            secnd_adj_stk = self.markets[adopt_scheme]["mseg_adjust"][
                "secondary mseg adjustments"]["stock-and-flow"]
            secnd_adj_mktshr = self.markets[adopt_scheme]["mseg_adjust"][
                "secondary mseg adjustments"]["market share"]
            # Determine a dictionary key indicating the climate zone, building
            # type, and structure type that is shared by the primary
            # microsegment and secondary microsegment
            secnd_mseg_adjkey = str((mskeys[1], mskeys[2], mskeys[-1]))
            # If no year-by-year secondary microsegment adjustment information
            # exists for the given climate zone, building type, and structure
            # type, initialize all year-by-year adjustment values as 0
            if secnd_mseg_adjkey not in secnd_adj_stk[
                    "original energy (total)"].keys():
                # Initialize sub-market secondary adjustment information

                # Initialize original primary microsegment stock information
                secnd_adj_sbmkt["original energy (total)"][
                    secnd_mseg_adjkey] = {
                        key: energy_total_scnd[key] for key in
                        self.handyvars.aeo_years}
                # Initialize sub-market adjusted microsegment stock information
                secnd_adj_sbmkt["adjusted energy (sub-market)"][
                    secnd_mseg_adjkey] = dict.fromkeys(
                        self.handyvars.aeo_years, 0)

                # Initialize stock-and-flow secondary adjustment information

                # Initialize original primary microsegment stock information
                secnd_adj_stk["original energy (total)"][secnd_mseg_adjkey] = \
                    dict.fromkeys(self.handyvars.aeo_years, 0)
                # Initialize previously captured primary microsegment stock
                # information
                secnd_adj_stk["adjusted energy (previously captured)"][
                    secnd_mseg_adjkey] = dict.fromkeys(
                        self.handyvars.aeo_years, 0)
                # Initialize competed primary microsegment stock information
                secnd_adj_stk["adjusted energy (competed)"][
                    secnd_mseg_adjkey] = dict.fromkeys(
                        self.handyvars.aeo_years, 0)
                # Initialize competed and captured primary microsegment stock
                # information
                secnd_adj_stk["adjusted energy (competed and captured)"][
                    secnd_mseg_adjkey] = dict.fromkeys(
                        self.handyvars.aeo_years, 0)

                # Initialize market share secondary adjustment information

                # Initialize original total captured stock information
                secnd_adj_mktshr["original energy (total captured)"][
                    secnd_mseg_adjkey] = dict.fromkeys(
                        self.handyvars.aeo_years, 0)
                # Initialize original competed and captured stock information
                secnd_adj_mktshr["original energy (competed and captured)"][
                    secnd_mseg_adjkey] = dict.fromkeys(
                        self.handyvars.aeo_years, 0)
                # Initialize adjusted total captured stock information
                secnd_adj_mktshr["adjusted energy (total captured)"][
                    secnd_mseg_adjkey] = dict.fromkeys(
                        self.handyvars.aeo_years, 0)
                # Initialize adjusted competed and captured stock information
                secnd_adj_mktshr["adjusted energy (competed and captured)"][
                    secnd_mseg_adjkey] = dict.fromkeys(
                        self.handyvars.aeo_years, 0)

        # In cases where no secondary heating/cooling microsegment is present,
        # set secondary microsegment adjustment key to None
        else:
            secnd_mseg_adjkey = None

        # Set time sensitive energy scaling factors for all baseline stock
        # (does not depend on year)
        tsv_energy_base = tsv_adj_init["energy"]["baseline"]
        # Commercial equipment stock numbers are in units of annual
        # demand served; cost and (if applicable) refrigerant charge
        # values are in units of hourly service capacity; to multiply the
        # former by the latter, a conversion factor is needed to translate
        # stock from unit service demand to unit service capacity. Note that
        # this conversion is set to 1 in the case where no stock (service
        # demand) data could be pulled for the given equipment type/end use/
        # building type.
        if (bldg_sect == "commercial" and mskeys[2] != "unspecified") and \
                sqft_subst != 1 and (
                mskeys[4] not in self.handyvars.com_eqp_eus_nostk):
            # Use try/except to handle missing capacity factor data
            try:
                # Set appropriate capacity factor (TBtu delivered service
                # for hours of actual operation / TBtu service running at
                # full capacity for all hours of the year)
                cap_fact_mseg = self.handyvars.cap_facts["data"][mskeys[2]][mskeys[4]]
                # Conversion: (1) divides stock (service delivered) by
                # the capacity factor (service delivered per year /
                # service per year @ full capacity) to get to service per
                # year @ full capacity; (2) divides by 8760 hours / year
                # to get to service per hour at full capacity; (3)
                # multiplies by 1e9 to get from service demand units of
                # TBtu/h (heating/cooling), giga-lm (lighting) or giga-
                # CFM (ventilation) to the baseline/measure cost unit
                # denominators of kBtu/h, 1000 lm, and 1000 CFM
                stk_serv_cap_cnv = (1 / cap_fact_mseg) * (1 / 8760) * 1e9
            except (KeyError):
                raise KeyError(
                    "Microsegment '" + str(mskeys) + "' "
                    "requires capacity factor data that are missing")
        else:
            stk_serv_cap_cnv = 1

        # DIFFUSION COEFFICIENTS
        # 1) Initialize dictionary
        years_diff_fraction_dictionary = {}
        # 2) Let us check if the diffusion coefficients are defined:
        try:
            self.diffusion
        except (NameError, AttributeError):
            # If not present, we set it to 1
            for year in self.handyvars.aeo_years:
                years_diff_fraction_dictionary[str(year)] = 1
        else:
            # 3) Check if diffusion parameters are defined as fractions
            if ('fraction_' in list(self.diffusion.keys())[0]):
                try:
                    # The diffusion fraction dictionary is converted to a pandas dataframe
                    df = pd.DataFrame(self.diffusion.items(), columns=['years', 'diff'])
                    df['years'] = df['years'].str.replace('fraction_', '')
                    if str(self.handyvars.aeo_years[0]) not in df['years']:
                        df.loc[len(df.index), :] = [str(self.handyvars.aeo_years[0]), None]
                    if str(self.handyvars.aeo_years[-1]) not in df['years']:
                        df.loc[len(df.index), :] = [str(self.handyvars.aeo_years[-1]), None]
                    # The years column is used as index
                    df['years'] = pd.to_datetime(df['years'])
                    df.index = df['years']
                    df.drop(['years'], axis=1, inplace=True)
                    # Force all values to be floats
                    df["diff"] = pd.to_numeric(df["diff"], downcast="float")
                    # The data are resampled yearly
                    df = df.resample('YE').mean()
                    # If there is any value greater than 1, set it to 1
                    if (df['diff'] > 1).any():
                        warn_list.append("WARNING: Some declared diffusion fractions are greater"
                                         " than 1. Their value has been changed to 1.")
                        df.loc[df['diff'] > 1, 'diff'] = 1
                    # if there is any value smaller than 0, set it to 0
                    if (df['diff'] < 0).any():
                        warn_list.append("WARNING: Some declared diffusion fractions are smaller"
                                         " than 0. Their value has been changed to 0.")
                        df.loc[df['diff'] < 0, 'diff'] = 0
                    # The data are interpolated to fill up values for each year
                    df = df.interpolate(method='linear',
                                        limit_direction='both',
                                        limit_area='inside')
                    # if values for the first and for the last years are not specified, the first
                    # declared value is used for all the first years and the last declared value
                    # is used for all the last years.
                    if df['diff'].isnull().values.any():
                        warn_list.append("WARNING: Not enough data were provided for first and"
                                         " last years of the considered simulation period.\n"
                                         "\tThe simulation will continue assuming plausible"
                                         " diffusion fraction values.")
                        df = df.interpolate(method='linear').bfill()
                    # The time span for the diffusion fraction is limited to the simulation period
                    df = df[(
                             df.index.year >= int(self.handyvars.aeo_years[0])
                            ) &
                            (
                             df.index.year <
                             (int(self.handyvars.aeo_years[-1])
                              + 1)
                            )]
                    fractions = df['diff'].to_list()
                    # for year in range_years:
                    for i in range(0, len(self.handyvars.aeo_years)):
                        years_diff_fraction_dictionary[
                                        str(self.handyvars.aeo_years[i])
                                                      ] = fractions[i]

                except (NameError, AttributeError, ValueError):
                    # This takes care of fractions defined
                    # as strings not convertible to floats
                    warn_list.append('WARNING: Diffusion parameters are not '
                                     'properly defined in the measure\n==>'
                                     'diffusion parameters set to 1 for'
                                     ' every year.')
                    for year in self.handyvars.aeo_years:
                        years_diff_fraction_dictionary[str(year)] = 1
            # 4) check if diffusion parameters are defined as
            # p and q for Bass Diffusion Model
            elif ('bass_model_p' in self.diffusion.keys())\
                &\
                 ('bass_model_q' in self.diffusion.keys()):
                try:
                    p = float(self.diffusion['bass_model_p'])
                    q = float(self.diffusion['bass_model_q'])
                except ValueError:
                    warn_list.append('WARNING: Diffusion parameters are not '
                                     'properly defined in the measure\n==>'
                                     'diffusion parameters set to 1 for'
                                     ' every year.')
                    # If not present, we set it to 1
                    for year in self.handyvars.aeo_years:
                        years_diff_fraction_dictionary[str(year)] = 1
                else:
                    for i in range(0, len(self.handyvars.aeo_years)):
                        # Bass diffusion model
                        value = (1 - math.exp(-(p+q)*(float(
                            self.handyvars.aeo_years[i]
                            ) - float(self.handyvars.aeo_years[0]))))\
                            /\
                            ((1 + (q/p) * math.exp(-(p+q)*(
                                float(self.handyvars.aeo_years[i])
                                - float(self.handyvars.aeo_years[0])))))
                        years_diff_fraction_dictionary[str(
                            self.handyvars.aeo_years[i])] = value
            else:
                warn_list.append('WARNING: Diffusion parameters are not '
                                 'properly defined in the measure\n==>'
                                 'diffusion parameters set to 1 for'
                                 ' every year.')
                # 5) If not present, we set it to 1
                for year in self.handyvars.aeo_years:
                    years_diff_fraction_dictionary[str(year)] = 1

        # Loop through and update stock, energy, and carbon mseg partitions for
        # each year in the modeling time horizon
        for yr in self.handyvars.aeo_years:
            # Reset flag for whether measure is on the market in current year
            # and for whether measure has exited the market
            if (int(yr) >= self.market_entry_year and
                    int(yr) < self.market_exit_year):
                measure_on_mkt = True
                measure_exited_mkt = ""
            elif int(yr) >= self.market_exit_year:
                measure_on_mkt = ""
                measure_exited_mkt = True
            else:
                measure_on_mkt, measure_exited_mkt = ("" for n in range(2))

            # Set time sensitive cost/emissions scaling factors for all
            # baseline stock; handle cases where these factors are/are not
            # broken out by AEO projection year
            try:
                tsv_ecost_base, tsv_carb_base = [
                    tsv_adj_init[x]["baseline"][yr] for
                    x in ["cost", "carbon"]]
            except TypeError:
                tsv_ecost_base, tsv_carb_base = [
                    tsv_adj_init[x]["baseline"] for
                    x in ["cost", "carbon"]]

            # If measure is not on the market, force efficient TSV scaling
            # factors to baseline profiles
            if measure_on_mkt:
                tsv_energy_eff = tsv_adj_init["energy"]["efficient"]
                # Set time sensitive cost/emissions scaling factors for all
                # efficient stock; handle cases where these factors are/are not
                # broken out by AEO projection year
                try:
                    tsv_ecost_eff, tsv_carb_eff = [
                        tsv_adj_init[x]["efficient"][yr] for
                        x in ["cost", "carbon"]]
                except TypeError:
                    tsv_ecost_eff, tsv_carb_eff = [
                        tsv_adj_init[x]["efficient"] for x
                        in ["cost", "carbon"]]
            else:
                tsv_energy_eff, tsv_ecost_eff, tsv_carb_eff = [
                    tsv_energy_base, tsv_ecost_base, tsv_carb_base]

            # For secondary microsegments only, update sub-market scaling
            # fraction, based on any sub-market scaling in the associated
            # primary microsegment
            if mskeys[0] == "secondary":
                # Adjust sub-market scaling fraction
                if secnd_adj_sbmkt["original energy (total)"][
                        secnd_mseg_adjkey][yr] != 0:
                    mkt_scale_frac = secnd_adj_sbmkt[
                        "adjusted energy (sub-market)"][
                        secnd_mseg_adjkey][yr] / \
                        secnd_adj_sbmkt["original energy (total)"][
                        secnd_mseg_adjkey][yr]

            # Total stock, energy, and carbon markets after accounting for any
            # sub-market scaling
            stock_total_sbmkt[yr] = stock_total_init[yr] * mkt_scale_frac
            energy_total_sbmkt[yr] = energy_total_init[yr] * mkt_scale_frac
            carb_total_sbmkt[yr] = carb_total_init[yr] * mkt_scale_frac
            # Total fugitive emissions markets after accounting for any
            # sub-market scaling
            if f_meth_assess:
                fmeth_total_sbmkt[yr] = fmeth_total_init[yr] * mkt_scale_frac

            # Calculate new, replacement, and retrofit fractions for the
            # baseline and efficient stock as applicable. * Note: these
            # fractions are 0 for secondary microsegments
            if mskeys[0] == "primary":
                # Calculate the portions of existing baseline and efficient
                # stock that are up for replacement

                # Update base/measure captured replacement fraction

                # Set an indicator for when the stock previously captured
                # after the beginning of the modeling time horizon begins to
                # turnover, using the baseline lifetime (zero or negative
                # value indicates turnover)
                yrs_until_prevcapt_tover = int(life_base[yr]) - (
                    int(yr) - int(sorted(self.handyvars.aeo_years)[0]))

                if type(yrs_until_prevcapt_tover) != numpy.ndarray:
                    prev_capt_turnover = yrs_until_prevcapt_tover <= 0
                else:
                    prev_capt_turnover = all(yrs_until_prevcapt_tover <= 0)

                # Case where the current microsegment applies to new
                # structures
                if mskeys[-1] == "new":
                    # Calculate the newly added portion of the total new
                    # stock (where 'new' is all stock added since year one of
                    # the modeling time horizon)

                    # After the first year in the modeling time horizon,
                    # the newly added stock fraction is the difference in
                    # stock between the current and previous year divided
                    # by the current year's stock (assuming it is not zero)
                    if yr != self.handyvars.aeo_years[0] and \
                            stock_total_sbmkt[yr] != 0:
                        # Handle case where data for previous year are
                        # unavailable; set newly added fraction to 1
                        try:
                            # Handle case where new stock is attentuating
                            # rather than growing; set new added fraction to 0
                            if stock_total_sbmkt[yr] >= stock_total_sbmkt[
                                    str(int(yr) - 1)]:
                                new_frac = (
                                    stock_total_sbmkt[yr] - stock_total_sbmkt[
                                        str(int(yr) - 1)]) / \
                                    stock_total_sbmkt[yr]
                            else:
                                new_frac = 0
                        except KeyError:
                            new_frac = 1
                    # For the first year in the modeling time horizon, the
                    # newly added stock fraction is 1
                    elif yr == self.handyvars.aeo_years[0]:
                        new_frac = 1
                    # If the total new stock in the current year is zero,
                    # the newly added stock fraction is also zero
                    else:
                        new_frac = 0

                    # If the previously captured baseline stock has
                    # begun to turnover, the replacement rate equals the
                    # ratio of baseline stock captured in a given previous
                    # year over the total stock in the current year
                    if prev_capt_turnover:
                        # Determine the previous year in which
                        # the new stock was captured by the base/measure
                        prev_capt_yr = int(yr) - int(life_base[yr])
                        # Ensure that year of previous capture is not
                        # earlier than earliest year in the AEO range
                        if prev_capt_yr >= int(self.handyvars.aeo_years[0]):
                            # Calculate the ratio of the new stock
                            # previously captured by the base/measure tech.
                            # to the total new stock in the current year

                            # Previously captured year is not first
                            # AEO year and non-zero denominator
                            if (str(prev_capt_yr) !=
                                self.handyvars.aeo_years[0]) and \
                                    stock_total_sbmkt[yr] != 0:
                                # Handle case where stock was attentuating
                                # rather than growing in year where new stock
                                # was captured
                                if stock_total_sbmkt[str(prev_capt_yr)] >= \
                                    stock_total_sbmkt[
                                        str(prev_capt_yr - 1)]:
                                    repl_frac = (
                                        stock_total_sbmkt[str(prev_capt_yr)] -
                                        stock_total_sbmkt[
                                            str(prev_capt_yr - 1)]
                                        ) / stock_total_sbmkt[yr]
                                else:
                                    repl_frac = (
                                        stock_total_sbmkt[str(prev_capt_yr)] /
                                        stock_total_sbmkt[yr])
                                retro_frac = 0
                            # Previously captured year is first
                            # AEO year and non-zero denominator
                            elif stock_total_sbmkt[yr] != 0:
                                repl_frac = (
                                    stock_total_sbmkt[str(prev_capt_yr)] /
                                    stock_total_sbmkt[yr])
                                retro_frac = 0
                            else:
                                repl_frac, retro_frac = (
                                    0 for n in range(2))
                        else:
                            repl_frac, retro_frac = (0 for n in range(2))
                    # Otherwise, the replacement/retrofit rates are zero
                    else:
                        repl_frac, retro_frac = (0 for n in range(2))

                # Case where the current microsegment applies to
                # existing structures
                else:
                    # Set newly added stock fraction to zero (not applicable
                    # for existing structures)
                    new_frac = 0

                    # Case where not all existing stock has been captured
                    # yet; allow both regular replacements and retrofits
                    if turnover_cap_not_reached:
                        repl_frac = (1 / life_base[yr])
                        retro_frac = retro_rate_mseg[yr]
                    # Case where all existing stock has been captured
                    # but stock previously captured after the start of
                    # the modeling horizon is turning over; allow only
                    # regular replacement of that stock, no retrofits
                    elif prev_capt_turnover:
                        repl_frac = (1 / life_base[yr])
                        retro_frac = 0
                    # Otherwise, the turnover rates are zero
                    else:
                        repl_frac, retro_frac = (0 for n in range(2))
            else:
                new_frac, repl_frac, retro_frac = (0 for n in range(3))
                prev_capt_turnover = False

            # Determine the fraction of total stock, energy, and carbon
            # in a given year post-sub-mkt scaling that the measure will
            # compete for given the microsegment type and adoption scenario

            # Secondary microsegment (competed fraction tied to the associated
            # primary microsegment)
            if mskeys[0] == "secondary" and secnd_mseg_adjkey is not None \
                and secnd_adj_stk["original energy (total)"][
                    secnd_mseg_adjkey][yr] != 0:
                comp_frac_sbmkt = secnd_adj_stk[
                    "adjusted energy (competed)"][
                    secnd_mseg_adjkey][yr] / secnd_adj_sbmkt[
                    "adjusted energy (sub-market)"][secnd_mseg_adjkey][yr]
            # Primary microsegment in the first year of a technical
            # potential scenario (all stock competed)
            elif mskeys[0] == "primary" and \
                    adopt_scheme == "Technical potential":
                comp_frac_sbmkt = 1
            # Primary microsegment not in the first year where current
            # microsegment applies to new structure type
            elif mskeys[0] == "primary":
                # Total competed stock fraction is sum of new, replacement,
                # and retrofit turnover fractions calculated above
                comp_frac_sbmkt = new_frac + repl_frac + retro_frac
            else:
                comp_frac_sbmkt = 0

            # Ensure that competed fraction never exceeds 1; handle cases
            # with/without lifetime distribution
            try:
                if comp_frac_sbmkt > 1:
                    comp_frac_sbmkt = 1
            # Handle case with retro. rate distributions
            except ValueError:
                if any(comp_frac_sbmkt > 1):
                    comp_frac_sbmkt[numpy.where(comp_frac_sbmkt > 1)] = 1

            # Diffusion of electric HPs according to pre-determined rates
            if hp_rate and mskeys[0] == "primary":
                # Find the annual fraction of the total stock that is converted
                # to a heat pump; set to 100% for technical potential,
                # otherwise set to the new/replacement stock (and, if user
                # desires, retrofit) times externally determined rate

                # In a technical potential case, all stock converts to HPs
                if adopt_scheme == "Technical potential":
                    # Frac. total stock that converted to HP in year
                    annual_hp_convert_frac = 1
                    # Frac. total stock that remains with base fuel in year
                    annual_hp_remain_frac = 0
                # Otherwise, stock is converted according to external rate
                else:
                    # Determine what portion of the retrofit rate in each
                    # year converts to HPs based on user cmd line inputs
                    if opts.exog_hp_rates[1] == '1':  # All retrofits to HPs
                        # Converted retrofits
                        retro_convert = retro_frac
                        # Remaining retrofits
                        retro_remain = 0
                    elif opts.exog_hp_rates[1] == '2':  # Frac. retro. to HPs
                        # Converted retrofits
                        retro_convert = retro_frac * hp_rate[yr]
                        # Remaining retrofits
                        retro_remain = retro_frac * (1 - hp_rate[yr])

                    # If full conversion has not yet been achieved, calculate
                    # annual conversion fraction on the basis of external
                    # rate; otherwise, assume all units are converted
                    if stock_total_hp_convert_frac != 1:
                        # Frac. total stock that converted to HP in year
                        annual_hp_convert_frac = (new_frac + repl_frac) * \
                            hp_rate[yr] + retro_convert
                        # Frac. total stock that remains with base fuel in year
                        annual_hp_remain_frac = (new_frac + repl_frac) * (
                            1 - hp_rate[yr]) + retro_remain
                    else:
                        annual_hp_convert_frac = \
                            new_frac + repl_frac + retro_convert
                        annual_hp_remain_frac = 0

                # Find the annual stock that is converted to the HP measure
                stock_comp_hp_convert[yr] = \
                    annual_hp_convert_frac * stock_total_sbmkt[yr]
                # Find the annual stock competed but remains /w base fuel
                stock_comp_hp_remain[yr] = \
                    annual_hp_remain_frac * stock_total_sbmkt[yr]
                # Find the total stock (cumulative) that has been converted
                # to HPs through the current year
                if yr == self.handyvars.aeo_years[0]:
                    stock_total_hp_convert[yr] = stock_comp_hp_convert[yr]
                # If stock is not already all captured, add competed stock
                else:
                    stock_total_hp_convert[yr] = (
                        stock_total_hp_convert[str(int(yr) - 1)]
                        + stock_comp_hp_convert[yr])
                # Ensure converted HP stock never goes above 1
                if stock_total_hp_convert[yr] > stock_total_sbmkt[yr]:
                    stock_total_hp_convert[yr] = stock_total_sbmkt[yr]

                # Find the fraction of the total stock that has been converted
                # to HPs through the current year
                if stock_total_sbmkt[yr] != 0:
                    stock_total_hp_convert_frac = \
                        stock_total_hp_convert[yr] / stock_total_sbmkt[yr]
                else:
                    stock_total_hp_convert_frac = 0

                # Finalize the fraction of the total stock (cumulative) that
                # has either been converted to or away from the current mseg
                # through the current year, as well as the fraction of the
                # converted stock that was competed in the current year

                # Case where the measure's microsegment is being added to
                # by the HP conversion (e.g., electric ASHP or HPWH measure)
                if (self.fuel_switch_to == "electricity" or
                        "electricity" in mskeys):
                    # Cumulative fraction converted to HPs
                    diffuse_frac = stock_total_hp_convert_frac
                    # Fraction of total converted stock that was
                    # converted in the current year
                    if stock_total_hp_convert[yr] != 0:
                        comp_frac_diffuse = stock_comp_hp_convert[yr] / \
                            stock_total_hp_convert[yr]
                    else:
                        comp_frac_diffuse = 0
                # Case where the measure's microsegment is being eroded
                # by the pre-determined conversion to HPs (e.g. gas efficiency)
                else:
                    # Cumulative fraction remaining after conversion to HPs
                    diffuse_frac = (1 - stock_total_hp_convert_frac)
                    # Fraction of total remaining stock that was
                    # retained in the current year
                    if (stock_total_sbmkt[yr] -
                            stock_total_hp_convert[yr]) != 0:
                        comp_frac_diffuse = stock_comp_hp_remain[yr] / (
                            stock_total_sbmkt[yr] -
                            stock_total_hp_convert[yr])
                    else:
                        comp_frac_diffuse = 0
            # All other measure diffusion cases
            else:
                # Currently no diffusion scaling
                diffuse_frac = 1
                # Competed fraction is that calculated above for the mseg
                # after applying submkt scaling
                comp_frac_diffuse = comp_frac_sbmkt

            # Ensure that competed diffusion fraction is always between 0 and 1
            if type(comp_frac_diffuse) != numpy.ndarray:
                if comp_frac_diffuse > 1:
                    comp_frac_diffuse = 1
                elif comp_frac_diffuse < 0:
                    comp_frac_diffuse = 0
            elif type(comp_frac_diffuse) == numpy.ndarray:
                if any(comp_frac_diffuse > 1):
                    comp_frac_diffuse[numpy.where(comp_frac_diffuse > 1)] = 1
                elif any(comp_frac_diffuse < 0):
                    comp_frac_diffuse[numpy.where(comp_frac_diffuse < 0)] = 0

            # Multiply diffusion fractions calculated above
            # (to represent exogenous HP switching rates, if applicable)
            # by further fraction to represent slow diffusion of information
            # for emerging technologies
            diffuse_frac *= years_diff_fraction_dictionary[yr]

            # If the measure is on the market, the competed fraction that
            # is captured by the measure is the same as the competed fraction
            # above (all competed stock goes to measure); otherwise, this
            # fraction is set to zero to preclude measure impacts on baseline
            if measure_on_mkt:
                comp_frac_diffuse_meas = comp_frac_diffuse
            else:
                comp_frac_diffuse_meas = 0

            # Flag a case where the measure is not currently on the market,
            # but the measure has previously captured stock that is now
            # turning over; in this case, the previously captured measure stock
            # should be decremented by the current year's competed stock value
            # (see use of this below)
            if not measure_on_mkt and prev_capt_turnover:
                decrmnt_meas_capt_stk = True
            else:
                decrmnt_meas_capt_stk = False

            # If applicable, pull baseline and efficient case refrigerant
            # leakage emissions on a per unit basis
            if f_refr_assess:
                # Baseline tech. unit-level refrigerant emissions
                if f_refr["baseline"][0] is not None:
                    # Find key to use in pulling global warming potential
                    # value for the typical baseline tech. refrigerant; typical
                    # refrigerants may be stored as a dict keyed in by year
                    if type(f_refr["baseline"][0]["typ_refrigerant"]) == dict:
                        r_key_b = f_refr["baseline"][0]["typ_refrigerant"][[
                            y for y in f_refr["baseline"][0][
                                "typ_refrigerant"].keys() if
                            int(yr) >= int(y)][-1]]
                    else:
                        r_key_b = f_refr["baseline"][0]["typ_refrigerant"]
                    # Use key above to pull baseline tech. refrigerant GWP
                    try:
                        base_gwp_yr = self.handyvars.fug_emissions[
                            "refrigerants"]["refrigerant_GWP100"][r_key_b]
                    except KeyError:
                        raise ValueError(
                            "Refrigerant '" + r_key_b +
                            "' has no available GWP data in supporting "
                            f"file {fp.CONVERT_DATA / 'fugitive_emissions_convert.json'}")
                else:
                    base_gwp_yr = ""
                # Measure tech. unit-level refrigerant emissions
                if f_refr["efficient"][0] is not None:
                    # Case where user assumes measure uses low GWP refrigerant
                    if opts.fugitive_emissions[1] == '2':
                        # Low GWP refrigerant may be specified as a measure
                        # attribute; first try to pull from this attribute
                        try:
                            self.low_gwp_refrigerant
                            # User may set low GWP refrigerant to a default
                            # choice in the measure definition, or otherwise
                            # will provide the low GWP refrigerant name as a
                            # single string or in a dict keyed by year
                            if self.low_gwp_refrigerant in [
                                None, "default"] or \
                                    type(self.low_gwp_refrigerant) not in [
                                        str, dict]:
                                low_gwp_usr = ""
                            else:
                                low_gwp_usr = self.low_gwp_refrigerant
                        except AttributeError:
                            low_gwp_usr = ""
                        # Find key to use in pulling global warming potential
                        # value for the measure tech. refrigerant; this is
                        # either provided by the user or otherwise available
                        # in the supporting refrigerants data file
                        if low_gwp_usr:
                            # Handle low gwp broken out by year in the
                            # user input
                            if type(low_gwp_usr) == dict:
                                r_key_e = low_gwp_usr[
                                     [y for y in low_gwp_usr.keys() if
                                      int(yr) >= int(y)][-1]]
                            else:
                                r_key_e = low_gwp_usr
                        else:
                            # Low GWP refrigerants may be stored as a dict
                            # keyed in by year
                            if type(f_refr["efficient"][0][
                                    "low_gwp_refrigerant"]) == dict:
                                r_key_e = f_refr["efficient"][0][
                                    "low_gwp_refrigerant"][
                                     [y for y in f_refr["efficient"][0][
                                      "low_gwp_refrigerant"].keys() if
                                      int(yr) >= int(y)][-1]]
                            else:
                                r_key_e = f_refr["efficient"][0][
                                    "low_gwp_refrigerant"]
                    # Case where user assumes measure uses typical refrigerant
                    else:
                        # Typical refrigerants may be stored as a dict keyed
                        # in by year
                        if type(f_refr[
                                "efficient"][0]["typ_refrigerant"]) == dict:
                            r_key_e = f_refr["efficient"][0][
                                "typ_refrigerant"][
                                [y for y in f_refr["efficient"][0][
                                  "typ_refrigerant"].keys() if
                                 int(yr) >= int(y)][-1]]
                        else:
                            r_key_e = \
                                f_refr["efficient"][0]["low_gwp_refrigerant"]
                    # Use key above to pull measure refrigerant GWP
                    try:
                        meas_gwp_yr = self.handyvars.fug_emissions[
                            "refrigerants"]["refrigerant_GWP100"][r_key_e]
                    except KeyError:
                        raise ValueError(
                            "Refrigerant '" + r_key_e +
                            "' has no available GWP data in supporting "
                            f"file {fp.CONVERT_DATA / 'fugitive_emissions_convert.json'}")
                else:
                    meas_gwp_yr = ""

                # Calculate per unit baseline and measure refrigerant
                # emissions; calculation is:
                # ((typ. unit charge (kg) * typ. unit leakage (%) +
                #  (typ. unit charge (kg) * typ. unit leakage (%))/lifetime) *
                # refrigerant 100 year GWP * (1 MMT / 1e9 kg)
                # NOTE: use baseline lifetime to annualize end of life
                # refrigerant leakage across baseline/measure cases

                # Per unit baseline tech. refrigerant emissions in the current
                # year; if not available, set to zero
                if base_gwp_yr:
                    base_f_r_unit_yr = (
                        f_refr["baseline"][0]["typ_charge"] *
                        f_refr["baseline"][0]["typ_ann_leak_pct"] + ((
                            f_refr["baseline"][0]["typ_charge"] *
                            f_refr["baseline"][0]["EOL_leak_pct"]) /
                            life_base[yr])) * base_gwp_yr * (1 / 1e9)
                else:
                    base_f_r_unit_yr = 0

                # Weighted overall (for stock captured in current year and all
                # previous years) and previously captured per unit
                # baseline tech. refrigerant emissions

                # If first year in modeling time horizon, initialize
                # overall/previously captured unit baseline refrigerant
                # emissions as the same as the first year's values
                if yr == self.handyvars.aeo_years[0]:
                    base_f_r_unit_overall, base_f_r_unit_capt = (
                        copy.deepcopy(base_f_r_unit_yr) for n in range(2))
                # Otherwise calculate overall unit baseline refrigerant
                # emissions by weighing competed baseline stock from current
                # year with stock from previous years
                else:
                    # Set overall per unit refrigerant emissions; use
                    # competition fraction to weight current vs. previous
                    # year baseline unit refrigerant emissions
                    base_f_r_unit_overall = \
                        base_f_r_unit_yr * comp_frac_diffuse + \
                        base_f_r_unit_capt * (1 - comp_frac_diffuse)

                # Per unit measure tech. refrigerant emissions; if
                # not available, set to zero
                if meas_gwp_yr:
                    meas_f_r_unit_yr = (
                        f_refr["efficient"][0]["typ_charge"] *
                        f_refr["efficient"][0]["typ_ann_leak_pct"] + ((
                            f_refr["efficient"][0]["typ_charge"] *
                            f_refr["efficient"][0]["EOL_leak_pct"]) /
                            life_base[yr])) * meas_gwp_yr * (1 / 1e9)
                else:
                    meas_f_r_unit_yr = 0
                # Set measure unit refrigerant emissions relative to the
                # baseline unit refrigerant emissions (used below to
                # calculate total efficient refrigerant emissions relative
                # to total baseline refrigerant emissions)
                try:
                    rel_frefr_yr = meas_f_r_unit_yr / base_f_r_unit_yr
                except ZeroDivisionError:  # Handle zero refrigerant data
                    rel_frefr_yr = 1
            else:
                base_f_r_unit_yr, base_f_r_unit_overall, \
                    base_f_r_unit_capt = ("" for n in range(3))

            # Final total stock, energy, and carbon markets after accounting
            # for any diffusion/conversion dynamics that restrict a measure's
            # access to it's full baseline market (after sub-mkt scaling), as
            # well as any adjustments to account for time-sensitive effects
            stock_total[yr] = stock_total_sbmkt[yr] * diffuse_frac
            energy_total[yr] = \
                energy_total_sbmkt[yr] * diffuse_frac * tsv_energy_base
            carb_total[yr] = \
                carb_total_sbmkt[yr] * diffuse_frac * tsv_carb_base
            # Final total fugitive emissions markets after accounting for
            # diffusion/conversion dynamics/accounting for time-sensitive
            # effects (the latter only relevant to methane)
            if f_meth_assess:
                fmeth_total[yr] = \
                    fmeth_total_sbmkt[yr] * diffuse_frac * tsv_energy_base
            if f_refr_assess:
                # Check for flag to force baseline refrig. emissions to zero
                if f_refr["baseline"][1] is True:
                    frefr_total[yr] = 0
                else:
                    # For total baseline fugitive refrigerants, multiply
                    # number of total stock units by baseline fugitive
                    # refrigerant emissions per unit for the overall stock,
                    # also accounting for any required stock unit conversions)
                    frefr_total[yr] = \
                        stock_total_sbmkt[yr] * stk_serv_cap_cnv * \
                        base_f_r_unit_overall * diffuse_frac
            # Finalize the sub-market scaling fraction as the initial sub-
            # market scaling fraction from the measure definition times
            # the diffusion fraction by year
            mkt_scale_frac_fin[yr] = mkt_scale_frac * diffuse_frac

            # Re-apportion total baseline microsegment energy across all 8760
            # hours of the year, if necessary (supports sector-level savings
            # shapes)

            # Only update sector-level shapes for certain years of focus;
            # ensure that load shape information is available for the
            # update and if not, yield an error message. NOTE: divided out by
            # previously applied TSV factors to avoid double counting of
            # impacts in the baseline case
            if calc_sect_shapes is True and yr in \
                self.handyvars.aeo_years_summary and \
                    tsv_shapes is not None:
                self.sector_shapes[adopt_scheme][mskeys[1]][yr]["baseline"] = [
                    self.sector_shapes[adopt_scheme][mskeys[1]][yr][
                        "baseline"][x] + tsv_shapes["baseline"][x] *
                    (energy_total[yr] / tsv_energy_base) for x in range(8760)]
            elif calc_sect_shapes is True and tsv_shapes is None and (
                    mskeys[0] == "primary" and (
                        (mskeys[3] == "electricity") or
                        (self.fuel_switch_to == "electricity"))):
                raise ValueError(
                    "Missing hourly fraction of annual load data for "
                    "baseline energy use segment: " + str(mskeys) + ". ")

            # Competed stock, energy, and carbon markets after accounting for
            # any sub-market scaling
            stock_compete_sbmkt[yr] = stock_total_sbmkt[yr] * comp_frac_sbmkt
            energy_compete_sbmkt[yr] = energy_total_sbmkt[yr] * comp_frac_sbmkt
            carb_compete_sbmkt[yr] = carb_total_sbmkt[yr] * comp_frac_sbmkt
            # Competed fugitive emissions markets after accounting for
            # any sub-market scaling
            if f_meth_assess:
                fmeth_compete_sbmkt[yr] = fmeth_total_sbmkt[yr] * \
                    comp_frac_sbmkt

            # Final competed stock, energy, and carbon markets after accounting
            # for any diffusion/conversion dynamics that restrict a measure's
            # access to it's full baseline market (after sub-mkt scaling), as
            # well as any adjustments to account for time-sensitive effects
            stock_compete[yr] = \
                stock_total_sbmkt[yr] * diffuse_frac * comp_frac_diffuse
            energy_compete[yr] = \
                energy_total_sbmkt[yr] * diffuse_frac * comp_frac_diffuse * \
                tsv_energy_base
            carb_compete[yr] = \
                carb_total_sbmkt[yr] * diffuse_frac * comp_frac_diffuse * \
                tsv_carb_base
            # Final competed fugitive emissions markets after accounting for
            # diffusion/conversion dynamics that restrict a measure's
            # access to it's full baseline market (after sub-mkt scaling), as
            # well as any adjustments to account for time-sensitive effects
            # (only relevant to methane)
            if f_meth_assess:
                fmeth_compete[yr] = fmeth_total_sbmkt[yr] * diffuse_frac * \
                    comp_frac_diffuse * tsv_energy_base
            if f_refr_assess:
                # Check for flag to force baseline refrig. emissions to zero
                if f_refr["baseline"][1] is True:
                    frefr_compete[yr] = 0
                else:
                    # For competed baseline fugitive refrigerants, multiply
                    # number of competed stock units by baseline fugitive
                    # refrigerant emissions per unit for the current year
                    # stock, also accounting for any required stock unit
                    # conversions)
                    frefr_compete[yr] = stock_total_sbmkt[yr] * \
                        stk_serv_cap_cnv * base_f_r_unit_yr * diffuse_frac * \
                        comp_frac_diffuse

            # Final competed stock captured by the measure; special handling
            # for stock of measures that have exited the market to ensure that
            # captured stock continues to be assessed after market exit – the
            # effects of the market exit on captured stock are reflected later
            # in the competition routine, run.py
            if not measure_exited_mkt:
                stock_compete_meas[yr] = stock_total_sbmkt[yr] * \
                    diffuse_frac * comp_frac_diffuse_meas
            else:
                stock_compete_meas[yr] = stock_total_sbmkt[yr] * \
                    diffuse_frac * comp_frac_diffuse

            # For primary microsegments only, update portion of stock captured
            # by efficient measure in previous years
            if mskeys[0] == "primary" and yr != self.handyvars.aeo_years[0]:
                # Set previous year key
                prev_yr = str(int(yr) - 1)

                # Handle case where captured efficient stock total
                # is a point value
                try:
                    if (stock_total[yr] - stock_compete[yr]) != 0:
                        meas_cum_frac = \
                            stock_total_meas[prev_yr] / (
                                stock_total[yr] - stock_compete[yr])
                    if (stock_total_sbmkt[yr] - stock_compete_sbmkt[yr]) != 0:
                        comp_cum_frac = stock_comp_cum_sbmkt[prev_yr] / \
                            (stock_total_sbmkt[yr] - stock_compete_sbmkt[yr])
                # Handle case where captured efficient stock total
                # is a numpy array
                except ValueError:
                    if all((stock_total[yr] - stock_compete[yr]) != 0):
                        meas_cum_frac = (
                            stock_total_meas[prev_yr] /
                            (stock_total[yr] - stock_compete[yr]))
                    if all((stock_total_sbmkt[yr] -
                            stock_compete_sbmkt[yr]) != 0):
                        comp_cum_frac = (stock_comp_cum_sbmkt[prev_yr] / (
                            stock_total_sbmkt[yr] - stock_compete_sbmkt[yr]))

                # Ensure neither fraction goes above 1

                # Cumulative measure-captured fraction
                if type(meas_cum_frac) != numpy.ndarray and meas_cum_frac > 1:
                    meas_cum_frac = 1
                elif type(meas_cum_frac) == numpy.ndarray and \
                        any(meas_cum_frac > 1):
                    meas_cum_frac[numpy.where(meas_cum_frac > 1)] = 1
                # Cumulative competed-captured fraction
                if type(comp_cum_frac) != numpy.ndarray and comp_cum_frac > 1:
                    comp_cum_frac = 1
                elif type(comp_cum_frac) == numpy.ndarray and \
                        any(comp_cum_frac > 1):
                    comp_cum_frac[numpy.where(comp_cum_frac > 1)] = 1

                # For non-technical potential scenarios for existing stock,
                # determine whether a cap on existing stock turnover has been
                # reached (TP has measure eligible for all stock
                # each year, and new stock is always growing so the turnover
                # cap is not applicable)
                if "existing" in mskeys and \
                        adopt_scheme != "Technical potential":
                    if type(comp_cum_frac) != numpy.ndarray:
                        turnover_cap_not_reached = comp_cum_frac < 1
                    else:
                        turnover_cap_not_reached = all(comp_cum_frac < 1)

            # In the case of a primary microsegment with secondary effects,
            # update the information needed to scale down the secondary
            # microsegment(s) by a sub-market fraction and previously captured,
            # competed, and competed and captured stock fractions for the
            # primary microsegment
            if mskeys[0] == "primary" and mskeys[4] == "lighting" and \
                    secnd_mseg_adjkey is not None:
                # Sub-market stock
                secnd_adj_sbmkt["adjusted energy (sub-market)"][
                    secnd_mseg_adjkey][yr] += energy_total_sbmkt[yr]
                # Total stock
                secnd_adj_stk[
                    "original energy (total)"][secnd_mseg_adjkey][yr] += \
                    energy_total_init[yr]
                # Previously captured stock
                secnd_adj_stk["adjusted energy (previously captured)"][
                    secnd_mseg_adjkey][yr] += \
                    energy_total_sbmkt[yr] * meas_cum_frac
                # Competed stock
                secnd_adj_stk["adjusted energy (competed)"][
                    secnd_mseg_adjkey][yr] += \
                    energy_total_sbmkt[yr] * comp_frac_sbmkt
                # Competed and captured stock
                secnd_adj_stk["adjusted energy (competed and captured)"][
                    secnd_mseg_adjkey][yr] += \
                    energy_total_sbmkt[yr] * comp_frac_sbmkt

            # For secondary microsegments only, update the portion of
            # associated primary microsegment stock that has been captured by
            # the measure in previous years
            if mskeys[0] == "secondary":
                # Adjust previously captured efficient fraction
                if secnd_adj_stk["original energy (total)"][
                        secnd_mseg_adjkey][yr] != 0:
                    meas_cum_frac = secnd_adj_stk[
                        "adjusted energy (previously captured)"][
                        secnd_mseg_adjkey][yr] / secnd_adj_stk[
                        "original energy (total)"][secnd_mseg_adjkey][yr]
                else:
                    meas_cum_frac = 0

            # Determine the amount of existing stock that has already
            # been captured by the measure up until the current year;
            # subsequently, update the number of total and competed stock units
            # captured by the measure to reflect additions from the current
            # year

            # First year in the modeling time horizon
            if yr == self.handyvars.aeo_years[0]:
                stock_total_meas[yr] = stock_compete_meas[yr]
                stock_comp_cum_sbmkt[yr] = stock_compete_sbmkt[yr]
            # Subsequent year in modeling time horizon
            else:
                # Technical potential case where the measure has entered the
                # market: the stock captured by the measure should equal the
                # total stock (measure captures all stock)
                if adopt_scheme == "Technical potential" and (
                        measure_on_mkt or measure_exited_mkt):
                    stock_total_meas[yr] = stock_total[yr]
                    stock_comp_cum_sbmkt[yr] = stock_total_sbmkt[yr]
                # Non-technical potential case
                elif adopt_scheme != "Technical potential":
                    # Handle case where data for previous year are
                    # unavailable; set total captured stock to current year's
                    # competed/captured stock
                    try:
                        # Update total number of stock units captured by the
                        # measure specifically and an efficient alternative (
                        # reflects all previously captured stock +
                        # captured competed stock from the current year).
                        if not decrmnt_meas_capt_stk:
                            stock_total_meas[yr] = stock_total_meas[
                                str(int(yr) - 1)] + stock_compete_meas[yr]
                        else:
                            stock_total_meas[yr] = stock_total_meas[
                                str(int(yr) - 1)] - stock_compete[yr]
                        stock_comp_cum_sbmkt[yr] = stock_comp_cum_sbmkt[
                            str(int(yr) - 1)] + stock_compete_sbmkt[yr]
                    except KeyError:
                        if not decrmnt_meas_capt_stk:
                            stock_total_meas[yr] = stock_compete_meas[yr]
                        else:
                            stock_total_meas[yr] = 0
                        stock_comp_cum_sbmkt[yr] = stock_compete_sbmkt[yr]
                # All other cases, including technical potential case where
                # measure is not on the market (stock goes immediately to zero)
                else:
                    stock_total_meas[yr] = 0
                    stock_comp_cum_sbmkt[yr] = 0

            # Ensure stock captured by measure never exceeds total stock
            # and that it never goes below zero

            # Handle case where stock captured by measure is an array
            if type(stock_total_meas[yr]) == numpy.ndarray and \
                    any(stock_total_meas[yr] > stock_total[yr]) is True:
                stock_total_meas[yr][numpy.where(
                    stock_total_meas[yr] > stock_total[yr])] = stock_total[yr]
            elif type(stock_total_meas[yr]) == numpy.ndarray and \
                    any(stock_total_meas[yr] < 0) is True:
                stock_total_meas[yr][numpy.where(
                    stock_total_meas[yr] < 0)] = 0
            # Handle case where stock captured by measure is point val
            elif type(stock_total_meas[yr]) != numpy.ndarray and \
                    stock_total_meas[yr] > stock_total[yr]:
                stock_total_meas[yr] = stock_total[yr]
            elif type(stock_total_meas[yr]) != numpy.ndarray and \
                    stock_total_meas[yr] < 0:
                stock_total_meas[yr] = 0

            # Set the weighted overall relative performance and per unit
            # measure tech. refrigerant emissions (if applicable) for stock
            # captured in current year and all previous years.

            # Set to the relative performance/unit measure refrigerant
            # emissions of the current year only for all years through
            # market entry
            if int(yr) <= self.market_entry_year:
                # Update overall and previously captured measure stock RP
                rel_perf_overall, rel_perf_capt = (
                    copy.deepcopy(rel_perf[yr]) for n in range(2))
                # Update overall and previously captured measure unit
                # refrigerant emissions, if needed
                if f_refr_assess:
                    rel_frefr_overall, rel_frefr_capt = (
                        copy.deepcopy(rel_frefr_yr) for n in range(2))
            # Set a measure stock turnover weight to use in balancing
            # current measure year's relative performance and fugitive
            # refrigerant emissions per unit (if applicable) with values
            # for measure stock captured in all previous years since market
            # entry; weight is equal to the ratio of total competed measure
            # stock in the current year to total previously captured
            # measure stock across all previous years
            else:
                if ((type(stock_total_meas[yr]) != numpy.ndarray and
                     stock_total_meas[yr] != 0) or (
                    type(stock_total_meas[yr]) == numpy.ndarray and
                        all(stock_total_meas[yr]) != 0)):
                    to_m_stk = (
                        stock_compete_meas[yr] / stock_total_meas[yr])
                else:
                    to_m_stk = 0
                # Ensure the measure turnover weight never exceeds 1
                if (type(to_m_stk) != numpy.ndarray) and \
                        to_m_stk > 1:
                    to_m_stk = 1
                elif (type(to_m_stk) == numpy.ndarray) and any(
                        to_m_stk > 1):
                    to_m_stk[numpy.where(to_m_stk > 1)] = 1
                # Update overall measure stock RP
                rel_perf_overall = (rel_perf[yr] * to_m_stk +
                                    rel_perf_capt * (1 - to_m_stk))
                # Update overall and previously captured measure unit
                # refrigerant emissions, if needed
                if f_refr_assess:
                    rel_frefr_overall = (rel_frefr_yr * to_m_stk +
                                         rel_frefr_capt * (1 - to_m_stk))

            # Update total-efficient and competed-efficient energy and
            # carbon, where "efficient" signifies the total and competed
            # energy/carbon remaining after measure implementation plus
            # non-competed energy/carbon.

            # Set common variables for the efficient energy calculations

            # Competed energy captured by measure
            energy_tot_comp_meas = energy_total_sbmkt[yr] * diffuse_frac * \
                tsv_energy_eff * comp_frac_diffuse_meas * \
                rel_perf[yr] * (
                    site_source_conv_meas[yr] / site_source_conv_base[yr])
            # Competed energy not captured by measure
            energy_tot_comp_base = energy_total_sbmkt[yr] * diffuse_frac * \
                tsv_energy_base * (
                    comp_frac_diffuse - comp_frac_diffuse_meas)
            # Uncompeted energy captured by measure
            energy_tot_uncomp_meas = energy_total_sbmkt[yr] * diffuse_frac * \
                tsv_energy_eff * (1 - comp_frac_diffuse) * meas_cum_frac * \
                rel_perf_capt * (
                    site_source_conv_meas[yr] / site_source_conv_base[yr])
            # Uncompeted energy not captured by measure
            energy_tot_uncomp_base = energy_total_sbmkt[yr] * diffuse_frac * \
                tsv_energy_base * (1 - comp_frac_diffuse) * (
                    1 - meas_cum_frac)

            # Competed-efficient energy
            energy_compete_eff[yr] = energy_tot_comp_meas + \
                energy_tot_comp_base
            # Total-efficient energy
            energy_total_eff[yr] = energy_compete_eff[yr] + \
                energy_tot_uncomp_meas + energy_tot_uncomp_base

            # Re-apportion total efficient microsegment energy across all 8760
            # hours of the year, if necessary (supports sector-level savings
            # shapes)
            if calc_sect_shapes is True and \
                yr in self.handyvars.aeo_years_summary and \
                    tsv_shapes is not None:
                # For fuel switching measures where a fossil baseline segment
                # (to be switched to electricity) is present, only represent
                # the measure-captured (switched to/added) loads in the
                # efficient-case sector shape; otherwise, represent both
                # measure-captured and remaining baseline loads for
                # electricity microsegments. NOTE: divided out by previously
                # applied TSV factors to avoid double counting of impacts
                # in the efficient case
                if self.fuel_switch_to == "electricity" and \
                        "electricity" not in mskeys:
                    self.sector_shapes[adopt_scheme][mskeys[1]][yr][
                        "efficient"] = [self.sector_shapes[adopt_scheme][
                            mskeys[1]][yr]["efficient"][x] + ((
                                energy_tot_comp_meas +
                                energy_tot_uncomp_meas) / tsv_energy_eff) *
                            tsv_shapes["efficient"][x]
                        for x in range(8760)]
                else:
                    self.sector_shapes[adopt_scheme][mskeys[1]][yr][
                        "efficient"] = [self.sector_shapes[adopt_scheme][
                            mskeys[1]][yr]["efficient"][x] + ((
                                energy_tot_comp_meas +
                                energy_tot_uncomp_meas) / tsv_energy_eff) *
                            tsv_shapes["efficient"][x] + ((
                                energy_tot_comp_base +
                                energy_tot_uncomp_base) / tsv_energy_base) *
                            tsv_shapes["baseline"][x]
                        for x in range(8760)]
            # Anticipate and handle case with base carbon intensity of zero for
            # electricity; in this case, assume the measure and baseline
            # intensity is the same (zero intensity is only possible for
            # electricity; assume measures will not be switched away from
            # electricity)
            try:
                intensity_carb_ratio = (
                    intensity_carb_meas[yr] / intensity_carb_base[yr])
            except ZeroDivisionError:
                intensity_carb_ratio = 1

            # Set common variables for the carbon calculations

            # Competed carbon captured by measure
            carb_tot_comp_meas = carb_total_sbmkt[yr] * diffuse_frac * \
                tsv_carb_eff * comp_frac_diffuse_meas * rel_perf[yr] * \
                (site_source_conv_meas[yr] / site_source_conv_base[yr]) * \
                intensity_carb_ratio
            # Competed carbon not captured by measure
            carb_tot_comp_base = carb_total_sbmkt[yr] * diffuse_frac * \
                tsv_carb_base * (
                    comp_frac_diffuse - comp_frac_diffuse_meas)
            # Uncompeted carbon captured by measure
            carb_tot_uncomp_meas = carb_total_sbmkt[yr] * diffuse_frac * \
                tsv_carb_eff * (1 - comp_frac_diffuse) * meas_cum_frac * \
                rel_perf_capt * (
                    site_source_conv_meas[yr] / site_source_conv_base[yr]) * \
                intensity_carb_ratio
            # Uncompeted carbon not captured by measure
            carb_tot_uncomp_base = carb_total_sbmkt[yr] * diffuse_frac * \
                tsv_carb_base * (1 - comp_frac_diffuse) * (
                    1 - meas_cum_frac)

            # Competed-efficient carbon
            carb_compete_eff[yr] = carb_tot_comp_meas + carb_tot_comp_base
            # Total-efficient energy
            carb_total_eff[yr] = carb_compete_eff[yr] + \
                carb_tot_uncomp_meas + carb_tot_uncomp_base

            # Set common variables for the fugitive emissions calculations

            # Methane
            if f_meth_assess:
                # Anticipate and handle case with base carbon intensity of zero
                # for electricity; in this case, assume the measure/baseline
                # intensity is the same (zero intensity is only possible for
                # electricity; assume measures will not be switched away from
                # electricity)
                try:
                    lkg_fmeth_ratio = (lkg_fmeth_meas / lkg_fmeth_base)
                except ZeroDivisionError:
                    lkg_fmeth_ratio = 1

                # Competed fugitive methane captured by measure
                fmeth_tot_comp_meas = fmeth_total_sbmkt[yr] * diffuse_frac * \
                    tsv_energy_eff * comp_frac_diffuse_meas * rel_perf[yr] * \
                    (site_source_conv_meas[yr] / site_source_conv_base[yr]) * \
                    lkg_fmeth_ratio
                # Competed fugitive methane not captured by measure
                fmeth_tot_comp_base = fmeth_total_sbmkt[yr] * diffuse_frac * \
                    tsv_energy_base * (
                        comp_frac_diffuse - comp_frac_diffuse_meas)
                # Uncompeted fugitive methane captured by measure
                fmeth_tot_uncomp_meas = fmeth_total_sbmkt[yr] * \
                    diffuse_frac * tsv_energy_eff * (1 - comp_frac_diffuse) * \
                    meas_cum_frac * rel_perf_capt * (
                        site_source_conv_meas[yr] /
                        site_source_conv_base[yr]) * \
                    lkg_fmeth_ratio
                # Uncompeted fugitive methane not captured by measure
                fmeth_tot_uncomp_base = fmeth_total_sbmkt[yr] * \
                    diffuse_frac * tsv_energy_base * \
                    (1 - comp_frac_diffuse) * (1 - meas_cum_frac)
                # Competed-efficient fugitive methane
                fmeth_compete_eff[yr] = fmeth_tot_comp_meas + \
                    fmeth_tot_comp_base
                # Total-efficient fugitive methane
                fmeth_total_eff[yr] = fmeth_compete_eff[yr] + \
                    fmeth_tot_uncomp_meas + fmeth_tot_uncomp_base
            # Refrigerants
            if f_refr_assess:
                # Check for flag to force measure refrig. emissions to zero
                if f_refr["efficient"][1] is True:
                    frefr_tot_comp_meas, frefr_tot_uncomp_meas = (
                        0 for n in range(2))
                else:
                    # # Competed fugitive refrigerants captured by measure
                    frefr_tot_comp_meas = stock_total_sbmkt[yr] * \
                        stk_serv_cap_cnv * diffuse_frac * \
                        comp_frac_diffuse_meas * base_f_r_unit_yr * \
                        rel_frefr_yr
                    # Uncompeted fugitive refrigerants captured by measure
                    frefr_tot_uncomp_meas = stock_total_sbmkt[yr] * \
                        stk_serv_cap_cnv * diffuse_frac * (
                            1 - comp_frac_diffuse) * \
                        meas_cum_frac * base_f_r_unit_capt * rel_frefr_capt
                # Check for flag to force baseline refrig. emissions to zero
                if f_refr["baseline"][1] is True:
                    frefr_tot_comp_base, frefr_tot_uncomp_base = (
                        0 for n in range(2))
                else:
                    # Competed fugitive refrigerants not captured by measure
                    frefr_tot_comp_base = stock_total_sbmkt[yr] * \
                        stk_serv_cap_cnv * diffuse_frac * \
                        (comp_frac_diffuse - comp_frac_diffuse_meas) * \
                        base_f_r_unit_yr
                    # Uncompeted fugitive refrigerants not captured by measure
                    frefr_tot_uncomp_base = stock_total_sbmkt[yr] * \
                        stk_serv_cap_cnv * diffuse_frac * (
                            1 - comp_frac_diffuse) * (
                            1 - meas_cum_frac) * base_f_r_unit_capt
                # Competed-efficient fugitive refrigerants
                frefr_compete_eff[yr] = frefr_tot_comp_meas + \
                    frefr_tot_comp_base
                # Total-efficient fugitive refrigerants
                frefr_total_eff[yr] = frefr_compete_eff[yr] + \
                    frefr_tot_uncomp_meas + frefr_tot_uncomp_base
                # Reset previously captured baseline refrigerant emissions for
                # next year to that of overall baseline stock in current yr.
                base_f_r_unit_capt, rel_frefr_capt = [
                    base_f_r_unit_overall, rel_frefr_overall]

            # Update total and competed stock, energy, and carbon
            # costs

            # Baseline cost of the competed and total stock; anchor this on
            # the stock captured by the measure to allow direct comparison
            # with measure stock costs
            stock_compete_cost[yr] = \
                (stock_compete_meas[yr] * stk_serv_cap_cnv) * cost_base[yr]
            stock_total_cost[yr] = \
                (stock_total_meas[yr] * stk_serv_cap_cnv) * cost_base[yr]
            # Total and competed-efficient stock cost for add-on and
            # full service measures. * Note: the baseline technology installed
            # cost must be added to the measure installed cost in the case of
            # an add-on measure type
            if self.measure_type == "add-on":
                # Competed-efficient stock cost (add-on measure)
                stock_compete_cost_eff[yr] = \
                    (stock_compete_meas[yr] * stk_serv_cap_cnv) * (
                    cost_meas[yr] + cost_base[yr])
                # Total-efficient stock cost (add-on measure)
                stock_total_cost_eff[yr] = \
                    (stock_total_meas[yr] * stk_serv_cap_cnv) * (
                    cost_meas[yr] + cost_base[yr])
            else:
                # Competed-efficient stock cost (full service measure)
                stock_compete_cost_eff[yr] = \
                    (stock_compete_meas[yr] * stk_serv_cap_cnv) * cost_meas[yr]
                # Total-efficient stock cost (full service measure)
                stock_total_cost_eff[yr] = \
                    (stock_total_meas[yr] * stk_serv_cap_cnv) * cost_meas[yr]

            # Competed baseline energy cost
            energy_compete_cost[yr] = energy_total_sbmkt[yr] * \
                diffuse_frac * comp_frac_diffuse * cost_energy_base[yr] * \
                tsv_ecost_base
            # Total baseline energy cost
            energy_total_cost[yr] = energy_total_sbmkt[yr] * \
                diffuse_frac * cost_energy_base[yr] * tsv_ecost_base

            # Set common variables for the energy cost calculations

            # Competed energy cost captured by measure
            energy_cost_tot_comp_meas = energy_total_sbmkt[yr] * \
                diffuse_frac * tsv_ecost_eff * comp_frac_diffuse_meas * \
                rel_perf[yr] * (
                    site_source_conv_meas[yr] / site_source_conv_base[yr]) * \
                cost_energy_meas[yr]
            # Competed energy cost remaining with baseline
            energy_cost_tot_comp_base = energy_total_sbmkt[yr] * \
                diffuse_frac * tsv_ecost_base * (
                    comp_frac_diffuse - comp_frac_diffuse_meas) * \
                cost_energy_base[yr]
            # Total energy cost captured by measure
            energy_cost_tot_uncomp_meas = energy_total_sbmkt[yr] * \
                diffuse_frac * tsv_ecost_eff * (
                    1 - comp_frac_diffuse) * meas_cum_frac * rel_perf_capt * (
                    site_source_conv_meas[yr] / site_source_conv_base[yr]) * \
                cost_energy_meas[yr]
            # Total energy cost remaining with baseline
            energy_cost_tot_uncomp_base = energy_total_sbmkt[yr] * \
                diffuse_frac * tsv_ecost_base * (
                    1 - comp_frac_diffuse) * (1 - meas_cum_frac) * \
                cost_energy_base[yr]

            # Competed-efficient energy cost
            energy_compete_cost_eff[yr] = energy_cost_tot_comp_meas + \
                energy_cost_tot_comp_base
            # Total-efficient energy cost
            energy_total_eff_cost[yr] = energy_compete_cost_eff[yr] + \
                energy_cost_tot_uncomp_meas + energy_cost_tot_uncomp_base

            # Competed baseline carbon cost
            carb_compete_cost[yr] = carb_compete[yr] * \
                self.handyvars.ccosts[yr]
            # Total baseline carbon cost
            carb_total_cost[yr] = carb_total[yr] * self.handyvars.ccosts[yr]

            # Competed carbon-efficient cost
            carb_compete_cost_eff[yr] = \
                carb_compete_eff[yr] * self.handyvars.ccosts[yr]
            # Total carbon-efficient cost
            carb_total_eff_cost[yr] = \
                carb_total_eff[yr] * self.handyvars.ccosts[yr]

            # Reset previously captured measure relative performance for next
            # year to that of the overall stock in the current year
            rel_perf_capt = rel_perf_overall

            # For fuel switching measures where exogenous HP conversion
            # rates have NOT been specified only, record the portion of total
            # baseline energy, carbon, and energy cost that remains with the
            # baseline fuel in the given year; for fuel switching measures
            # with exogenous HP conversion rates specified, no baseline
            # energy/carbon/cost will remain with the baseline fuel b/c of the
            # way the markets are specified (the baseline for such measures is
            # constrained to only the energy/carbon/cost that switches over
            # in each year); for non-fuel switching measures, this variable is
            # not used further in the routine
            if self.fuel_switch_to is not None and not hp_rate:
                fs_energy_eff_remain[yr] = \
                    energy_tot_comp_base + energy_tot_uncomp_base
                fs_carb_eff_remain[yr] = \
                    carb_tot_comp_base + carb_tot_uncomp_base
                fs_energy_cost_eff_remain[yr] = \
                    energy_cost_tot_comp_base + energy_cost_tot_uncomp_base

        # Return partitioned stock, energy, and cost mseg information
        return [stock_total, energy_total, carb_total, fmeth_total,
                frefr_total, stock_total_meas, energy_total_eff,
                carb_total_eff, fmeth_total_eff, frefr_total_eff,
                stock_compete, energy_compete, carb_compete,
                fmeth_compete, frefr_compete, stock_compete_meas,
                energy_compete_eff, carb_compete_eff, fmeth_compete_eff,
                frefr_compete_eff, stock_total_cost, energy_total_cost,
                carb_total_cost, stock_total_cost_eff, energy_total_eff_cost,
                carb_total_eff_cost, stock_compete_cost, energy_compete_cost,
                carb_compete_cost, stock_compete_cost_eff,
                energy_compete_cost_eff, carb_compete_cost_eff,
                fs_energy_eff_remain, fs_carb_eff_remain,
                fs_energy_cost_eff_remain, mkt_scale_frac_fin, warn_list]

    def check_mkt_inputs(self):
        """Check for valid applicable baseline market inputs for a measure.

        Note:
            The inputs are checked against a list of valid baseline market
            input names, determined from the 'valid_mktnames' attribute of the
            'UsefulVars' object type.

        Raises:
            ValueError: If 'technology_type' attribute is improperly formatted
                or input names are not in the list of valid names.
        """
        # Initialize the list of input names to check
        check_list = []
        # Loop through all inputs related to a measure's applicable baseline
        # market and add input names to the list
        for x in [
            self.climate_zone, self.bldg_type, self.structure_type,
            self.fuel_type, self.end_use, self.technology_type,
                self.technology]:
            # Handle input values formatted as dicts, lists, or strings
            if isinstance(x, dict):
                [check_list.extend(x[ms]) if isinstance(x[ms], list) else
                 check_list.append(x[ms]) for ms in ["primary", "secondary"]]
            elif isinstance(x, list):
                check_list.extend(x)
            elif isinstance(x, str) or x is None:
                check_list.append(x)
            else:
                raise ValueError(
                    "ECM '" + self.name + "'applicable baseline market "
                    "input in unexpected format (need dict, list, or string)")
        # Find subset of input names that are not in the list of valid names
        invalid_names = [
            y for y in check_list if y not in self.handyvars.valid_mktnames]
        # If invalid names are discovered, report them in an error message

        # Special case: ECM uses region names that are inconsistent with
        # higher-level region settings for the simulation
        if len(invalid_names) > 0 and ((
            type(self.climate_zone) == list and all([
                x in self.climate_zone for x in invalid_names])) or (
            type(self.climate_zone) == str and all([
                x == self.climate_zone for x in invalid_names]))):
            raise ValueError(
                f"'climate_zone' input name(s) for ECM '{self.name}' ({str(invalid_names)})"
                " inconsistent with region settings for the simulation run. Either revise the"
                " 'climate_zone' input or the simulation's region setting (default is EMM regions,"
                " with alternates specified via '--alt_regions' command line option) to ensure"
                " consistency")
        # All other cases
        elif len(invalid_names) > 0:
            raise ValueError(
                "Input names in the following list are invalid for ECM '" +
                self.name + "': " + str(invalid_names))

    def fill_attr(self):
        """Fill out missing ECM attribute data.

        Note:
            Handles the following:
            a) Adds a 'windows conduction' technology market when only
            'windows solar' is defined for the ECM, and vice versa (assuming
            that both pertain to the same window unit).
            b) Splits the 'energy_efficiency', 'energy_efficiency_units',
            'energy_fuel_type', 'end_use', 'technology', and 'technology_type'
            attributes into 'primary' and 'secondary' keys, ensuring that
            secondary end use impacts are properly processed by the remainder
            of the 'fill_mkts' routine. Note: if the user has not specified
            any secondary end use impacts, secondary key values for all
            relevant attributes are set to None.
            c) Fills out any attributes marked 'all'. For example, if a user
            has specified 'all' for the 'climate_zone' measure attribute, this
            function will translate that entry into the full list of climate
            zones: ['AIA_CZ1', 'AIA_CZ2', 'AIA_CZ3', 'AIA_CZ4', 'AIA_CZ5'].
        """
        # Ensure that 'windows conduction' and 'windows solar' markets are
        # always defined together in the technology attribute
        if not isinstance(self.technology, list) and \
                self.technology is not None and "windows" in self.technology:
            self.technology = ["windows conduction", "windows solar"]
        elif isinstance(self.technology, list) and any([
                x is not None and "windows" in x for x in self.technology]):
            self.technology.extend(
                list(set(["windows conduction", "windows solar"]) -
                     set(self.technology)))

        # Split attributes relevant to representing secondary end use impacts
        # into 'primary' and 'secondary' keys

        # Case where user has flagged secondary end use impacts for the measure
        # via the 'end_use' attribute; fill in secondary 'fuel_type',
        # 'technology', and 'technology_type' information. Note: it is assumed
        # the user has already filled in secondary performance information via
        # the 'energy_efficiency' and 'energy_efficiency_units' attributes
        if isinstance(self.end_use, dict):
            # Assume secondary effects pertain to all fuel types and
            # technologies associated with the given secondary end use(s)
            self.fuel_type, self.technology = [{
                "primary": x, "secondary": "all"} for
                x in [self.fuel_type, self.technology]]
            # Assume secondary heating/cooling effects pertain to demand-side
            # heating/cooling technologies (e.g., secondary 'technology_type'
            # is set to 'demand')
            if (isinstance(self.end_use["secondary"], list) and any([
                x not in ["heating", "secondary heating", "cooling"] for
                x in self.end_use["secondary"]])) or (
                not isinstance(self.end_use["secondary"], list) and
                self.end_use["secondary"] not in [
                    "heating", "secondary heating", "cooling"]):
                self.technology_type = {
                    "primary": self.technology_type, "secondary": "supply"}
            else:
                self.technology_type = {
                    "primary": self.technology_type, "secondary": "demand"}
        # Case where the user has not flagged any secondary end use impacts;
        # set 'secondary' key values for relevant attributes to None
        else:
            self.energy_efficiency, self.energy_efficiency_units, \
                self.fuel_type, self.end_use, self.technology, \
                self.technology_type = [
                    {"primary": x, "secondary": None} for x in [
                        self.energy_efficiency, self.energy_efficiency_units,
                        self.fuel_type, self.end_use, self.technology,
                        self.technology_type]]

        # Fill out an 'all' climate zone input
        if self.climate_zone == 'all' or 'all' in self.climate_zone:
            self.climate_zone = self.handyvars.in_all_map["climate_zone"]

        # Fill out an 'all' structure type input
        if self.structure_type == 'all' or 'all' in self.structure_type:
            self.structure_type = self.handyvars.in_all_map["structure_type"]

        # Fill out an 'all' building type, fuel type, end use, and/or
        # technology input. Note that these attributes are affected by whether
        # or not the user has flagged secondary end use impacts
        for mseg_type in [
                x[0] for x in self.end_use.items() if x[1] is not None]:
            # Fill out a building type input that is marked 'all',
            # 'all residential' or 'all commercial', or is formatted as a list
            # with certain elements containing 'all' (e.g.,
            # ['all residential,' 'assembly', 'education'])
            if self.bldg_type == "all" or (
                "all" in self.bldg_type and
                self.bldg_type != "small office") or (
                type(self.bldg_type) == list and any([
                    "all" in b for b in self.bldg_type if
                    b != "small office"])):
                # Record the initial 'bldg_type' attribute the user has defined
                # for the measure before this attribute is reset below
                map_bldgtype_orig = self.bldg_type
                # Reset the measure 'bldg_type' attribute as a list that does
                # not contain any elements including 'all' (e.g., if the
                # 'bldg_type' attribute value was initially 'all', 'all
                # residential', or 'all commercial', it would be reset as a
                # blank list; if the 'bldg_type' attribute value was initially
                # ['all residential', 'assembly', 'education'], it would be
                # reset as ['assembly', 'education'])
                if self.bldg_type == 'all' or 'all' in self.bldg_type:
                    self.bldg_type = []
                else:
                    self.bldg_type = [
                        b for b in self.bldg_type if (
                            'all' not in b or b == 'small office')]
                # Fill 'bldg_type' attribute. Note that the comprehension below
                # handles 'all', 'all residential', or 'all commercial' values
                # for the initial user-defined 'bldg_type' attribute as well as
                # a list value for this attribute that contains elements with
                # 'all'(e.g., ['all residential,' 'assembly', 'education'])
                [self.bldg_type.extend(b[1]) for b in
                    self.handyvars.in_all_map["bldg_type"].items() if
                    b[1] not in self.bldg_type and (
                        map_bldgtype_orig == "all" or b[0] in
                        map_bldgtype_orig or
                        any([b[0] in borig for borig in map_bldgtype_orig if
                            'all ' in borig]))]
                # Record the measure's applicability to the residential and/or
                # the commercial building sectors in a list (for use in filling
                # 'fuel_type,' 'end_use,' and 'technology' attributes below)
                bldgsect_list = [x[0] for x in self.handyvars.in_all_map[
                    "bldg_type"].items() if any([
                        bt in x[1] for bt in self.bldg_type])]

                # For an 'all' building type case, measure 'installed_cost,'
                # 'cost_units,' 'energy_efficiency,' 'energy_efficiency_units,'
                # and/or 'product_lifetime' attributes may be formatted as
                # dictionaries broken out by building sector (e.g., with
                # 'all residential' and/or 'all commercial' keys). This part of
                # the code replaces such keys and their associated values with
                # the appropriate set of building types. For example a
                # ('all commercial', 1) key, value pair would be replaced with
                # [('assembly', 1), ('education', 1), ...]
                for attr in [self.installed_cost, self.cost_units,
                             self.energy_efficiency[mseg_type],
                             self.energy_efficiency_units[mseg_type],
                             self.product_lifetime]:
                    # Check whether attribute is a dict before moving further
                    if isinstance(attr, dict):
                        # Loop through each building sector and the building
                        # type names in that sector. If the sector has been
                        # assigned an 'all' breakout for the current attribute,
                        # add new key, value pairs to the attribute dict where
                        # the building type name is each key and the original
                        # value for the building sector is the value for the
                        # new pair
                        for b in bldgsect_list:
                            # Check whether the sector is assigned an 'all'
                            # breakout in the attribute (e.g., 'all
                            # residential')
                            sect_val = [
                                x[1] for x in attr.items() if b in x[0]]
                            # If sector is assigned 'all' breakout, add
                            # appropriate building type keys and values to
                            # attribute dict
                            if sect_val:
                                for kn in self.handyvars.in_all_map[
                                        "bldg_type"][b]:
                                    attr[kn] = sect_val[0]
                        # Remove the original 'all residential' and
                        # 'all commercial' branches from the attribute dict
                        del_keys = [x for x in attr.keys() if x in [
                                    'all residential', 'all commercial']]
                        for dk in del_keys:
                            del (attr[dk])
            # If there is no 'all' building type input, still record the
            # measure's applicability to the residential and/or
            # the commercial building sector in a list (for use in filling out
            # 'fuel_type,' 'end_use,' and 'technology' attributes below)
            else:
                bldgsect_list = [b[0] for b in self.handyvars.in_all_map[
                    "bldg_type"].items() if any([
                        bta == self.bldg_type for bta in b[1]]) or
                    any([bta in b[1] for bta in self.bldg_type])]

            # Fill out an 'all' fuel type input
            if self.fuel_type[mseg_type] == 'all' or \
                    'all' in self.fuel_type[mseg_type]:
                # Reset measure 'fuel_type' attribute as a list and fill with
                # all fuels for the measure's applicable building sector(s)
                self.fuel_type[mseg_type] = []
                for b in bldgsect_list:
                    [self.fuel_type[mseg_type].append(f) for f in
                     self.handyvars.in_all_map["fuel_type"][b] if
                     f not in self.fuel_type[mseg_type]]

            # Record the measure's applicable fuel type(s) in a list (for use
            # in filling out 'end_use,' and 'technology' attributes below)
            if not isinstance(self.fuel_type[mseg_type], list):
                fueltype_list = [self.fuel_type[mseg_type]]
            else:
                fueltype_list = self.fuel_type[mseg_type]

            # Fill out an 'all' end use input
            if self.end_use[mseg_type] == 'all' or \
                    'all' in self.end_use[mseg_type]:
                # Reset measure 'end_use' attribute as a list and fill with
                # all end uses for the measure's applicable building sector(s)
                # and fuel type(s)
                self.end_use[mseg_type] = []
                for b in bldgsect_list:
                    for f in [x for x in fueltype_list if
                              x in self.handyvars.in_all_map["fuel_type"][b]]:
                        [self.end_use[mseg_type].append(e) for e in
                         self.handyvars.in_all_map["end_use"][b][f] if
                         e not in self.end_use[mseg_type]]

            # Record the measure's applicable end use(s) in a list (for use in
            # filling out 'technology' attribute below)
            if not isinstance(self.end_use[mseg_type], list):
                enduse_list = [self.end_use[mseg_type]]
            else:
                enduse_list = self.end_use[mseg_type]

            # Fill out a technology input that is marked 'all' or is
            # formatted as a list with certain elements containing 'all' (e.g.,
            # ['all heating,' 'central AC', 'room AC'])
            if self.technology[mseg_type] == 'all' or \
                self.technology[mseg_type] == ['all'] or \
                (type(self.technology[mseg_type]) == list and any([
                 t is not None and 'all ' in t for t in
                 self.technology[mseg_type]])) or (
                    type(self.technology[mseg_type]) == str and
                    self.technology[mseg_type] is not None and
                    'all ' in self.technology[mseg_type]):
                # Record the initial 'technology' attribute the user has
                # defined for the measure before this attribute is reset below
                map_tech_orig = self.technology[mseg_type]
                # Reset the measure 'technology' attribute as a list that does
                # not contain any elements including 'all' (e.g., if the
                # 'technology' attribute value was initially 'all', it would be
                # reset as a blank list; if the 'technology' attribute value
                # was initially ['all heating', 'central AC', 'room AC'], it
                # would be reset as ['central AC', 'room AC'])
                if self.technology[mseg_type] == 'all':
                    self.technology[mseg_type] = []
                elif isinstance(self.technology[mseg_type], list):
                    self.technology[mseg_type] = [
                        t for t in self.technology[mseg_type] if
                        ('all ' not in t and t != 'all')]
                else:
                    self.technology[mseg_type] = [self.technology[mseg_type]]

                # Fill 'technology' attribute
                for b in bldgsect_list:
                    # Case concerning a demand-side technology, for which the
                    # set of 'all' technologies depends only on the measure's
                    # applicable building sector(s)
                    if self.technology_type[mseg_type] == "demand":
                        [self.technology[mseg_type].append(t) for t in
                            self.handyvars.in_all_map["technology"][b][
                            self.technology_type[mseg_type]] if t not in
                            self.technology[mseg_type]]
                    # Case concerning a supply-side technology, for which the
                    # set of 'all' technologies depends on the measure's
                    # applicable building sector(s), fuel type(s), and
                    # end use(s)
                    else:
                        for f in [ft for ft in fueltype_list if
                                  ft in self.handyvars.in_all_map[
                                "fuel_type"][b]]:
                            for e in [eu for eu in enduse_list if eu in
                                      self.handyvars.in_all_map[
                                    "end_use"][b][f]]:
                                # Note that the comprehension below handles
                                # both an 'all' value for the initial
                                # user-defined 'technology' attribute and a
                                # list value for this attribute that contains
                                # elements with 'all' (e.g., ['all heating',
                                # 'central AC', 'room AC']); ensure that
                                # secondary heating end use is not conflated
                                # with the heating end use
                                [self.technology[mseg_type].append(t) for
                                 t in self.handyvars.in_all_map["technology"][
                                 b][self.technology_type[mseg_type]][f][e] if
                                 t not in self.technology[mseg_type] and
                                 (map_tech_orig == "all" or
                                  map_tech_orig == ["all"] or any([(
                                     ("secondary heating" not in torig and
                                      e in torig) or (
                                      "secondary heating" in torig
                                      and e == "secondary heating")) for
                                    torig in map_tech_orig if
                                    'all ' in torig])) or e in map_tech_orig]

    def create_keychain(self, mseg_type):
        """Create list of dictionary keys used to find baseline microsegments.

        Args:
            mseg_type (string): Identifies the type of baseline microsegments
                to generate keys for ('primary' or 'secondary').

        Returns:
            List of key chains to use in retreiving data for the measure's
            applicable baseline market microsegments.
        """
        # Ensure that all variables relevant to forming key chains are lists
        self.climate_zone, self.bldg_type, self.fuel_type[mseg_type], \
            self.end_use[mseg_type], self.technology_type[mseg_type], \
            self.technology[mseg_type], self.structure_type = [
                [x] if not isinstance(x, list) else x for x in [
                    self.climate_zone, self.bldg_type,
                    self.fuel_type[mseg_type], self.end_use[mseg_type],
                    self.technology_type[mseg_type],
                    self.technology[mseg_type], self.structure_type]]
        # Flag heating/cooling end use microsegments. For heating/cooling
        # cases, an extra 'supply' or 'demand' key is required in the key
        # chain; this key indicates the supply-side and demand-side variants
        # of heating/ cooling technologies (e.g., ASHP for the former,
        # envelope air sealing for the latter).
        ht_cl_euses = ["heating", "secondary heating", "cooling"]

        # Case with heating and/or cooling microsegments
        if any([x in ht_cl_euses for x in self.end_use[mseg_type]]):
            # Format measure end use attribute as numpy array
            eu = numpy.array(self.end_use[mseg_type])
            # Set a list of heating and/or cooling end uses
            eu_hc = list(eu[numpy.where([x in ht_cl_euses for x in eu])])
            # Set a list of all other end uses
            eu_non_hc = list(eu[numpy.where([
                x not in ht_cl_euses for x in eu])])
            # Set a list including all measure microsegment attributes,
            # constraining the 'end_use' attribute to only heating/cooling
            # end uses
            ms_lists = [
                self.climate_zone, self.bldg_type, self.fuel_type[mseg_type],
                eu_hc, self.technology_type[mseg_type],
                self.technology[mseg_type]]
            # Generate a list of all possible combinations of the elements
            # in 'ms_lists' above
            ms_iterable_init = list(itertools.product(*ms_lists))
            # If there are also non-heating/cooling microsegments, set
            # a list including all measure microsegment attributes,
            # constraining the 'end_use' attribute to only non-heating/cooling
            # end uses
            if len(eu_non_hc) > 0:
                ms_lists_add = [self.climate_zone, self.bldg_type,
                                self.fuel_type[mseg_type], eu_non_hc,
                                self.technology[mseg_type]]
                # Generate a list of all possible combinations of the
                # elements in 'ms_lists_add' above and add this list
                # to 'ms_iterable_init', also adding the elements of
                # 'ms_lists_add' to the initial elements of 'ms_lists'
                # if not already present
                ms_iterable_init.extend(
                    list(itertools.product(*ms_lists_add)))
                for ind, ms_lists_add_i in enumerate(ms_lists_add):
                    if ind != (len(ms_lists_add)-1):
                        ms_lists[ind].extend([
                            x for x in ms_lists_add_i if
                            x not in ms_lists[ind]])
                    # Special handling for technology attribute, which
                    # is the last element in 'ms_lists_add' but the second
                    # to last element in 'ms_lists'
                    elif ind == (len(ms_lists_add)-1):
                        ms_lists[(ind + 1)].extend([
                            x for x in ms_lists_add_i if
                            x not in ms_lists[(ind + 1)]])

        # Case without heating or cooling microsegments
        else:
            # Set a list including all measure microsegment attributes
            ms_lists = [self.climate_zone, self.bldg_type,
                        self.fuel_type[mseg_type], self.end_use[mseg_type],
                        self.technology[mseg_type]]
            # Generate a list of all possible combinations of the elements
            # in 'ms_lists' above
            ms_iterable_init = list(itertools.product(*ms_lists))

        # Add primary or secondary microsegment type indicator to beginning
        # of each key chain and the applicable structure type (new or existing)
        # to the end of each key chain

        # Case where measure applies to both new and existing structures
        # (final ms_iterable list length is double that of ms_iterable_init)
        if len(self.structure_type) > 1:
            ms_iterable1, ms_iterable2 = ([] for n in range(2))
            for i in range(0, len(ms_iterable_init)):
                ms_iterable1.append((mseg_type,) + ms_iterable_init[i] +
                                    (self.structure_type[0], ))
                ms_iterable2.append((mseg_type,) + ms_iterable_init[i] +
                                    (self.structure_type[1], ))
            ms_iterable = ms_iterable1 + ms_iterable2
        # Case where measure applies to only new or existing structures
        # (final ms_iterable list length is same as that of ms_iterable_init)
        else:
            ms_iterable = []
            for i in range(0, len(ms_iterable_init)):
                ms_iterable.append(((mseg_type, ) + ms_iterable_init[i] +
                                    (self.structure_type[0], )))

        # Output list of key chains
        return ms_iterable, ms_lists

    def find_scnd_overlp(self, vint_frac, ss_conv, dict1, energy_tot):
        """Find total lighting energy for climate/building/structure type.

        Note:
            Primary/secondary microsegments are linked by climate zone,
            building type, and structure type (new/existing).

        Args:
            vint_frac (dict): New/existing fraction by year.
            ss_conv (dict): Site/source conversion factors by year.
            dict1 (dict): Dict of lighting stock/energy data for given
            climate zone, building type, and structure type.
            energy_tot (float): Total lighting energy value.

        Returns:
            Total lighting energy for given climate, building, and structure
            type.
        """
        for k, i in dict1.items():
            # Ignore stock data
            if k == "stock":
                continue
            # Proceed further into nested dict if terminal node with stock/
            # energy data has not been reached
            elif isinstance(i, dict) and k != "energy":
                self.find_scnd_overlp(
                    vint_frac, ss_conv, i, energy_tot)
            # If terminal node energy data has been reached, add energy
            # data to the total energy use variable
            elif k == "energy":
                try:  # Handle blank lighting incandescent energy dicts
                    for yr in self.handyvars.aeo_years:
                        energy_tot[yr] += dict1["energy"][yr] * \
                            vint_frac[yr] * ss_conv[yr]
                except KeyError:
                    pass
            # Raise error if no energy data are available
            else:
                raise ValueError(
                    'No energy data available for total '
                    'lighting energy use calculation')

        return energy_tot

    def add_keyvals(self, dict1, dict2):
        """Add key values of two dicts together.

        Note:
            Dicts must be identically structured.

        Args:
            dict1 (dict): First dictionary to add.
            dict2 (dict): Second dictionary to add.

        Returns:
            Single dictionary of combined values.

        Raises:
            KeyError: When added dict keys do not match.
        """
        for (k, i), (k2, i2) in zip(
                sorted(dict1.items()), sorted(dict2.items())):
            if k == k2:
                if isinstance(i, dict):
                    self.add_keyvals(i, i2)
                else:
                    if dict1[k] is None:
                        dict1[k] = copy.deepcopy(dict2[k2])
                    else:
                        dict1[k] = dict1[k] + dict2[k]
            else:
                raise KeyError("When adding together two dicts "
                               "for ECM '" + self.name +
                               "' update, dict key structures "
                               "do not match")
        return dict1

    def add_keyvals_restrict(self, dict1, dict2):
        """Add key values of two dicts, with restrictions.

        Note:
            Restrict the addition of 'lifetime' information. This
            function is used to merge baseline microsegments for
            windows conduction and windows solar components; the
            lifetimes for these components will be the same and
            need not be added and averaged later, as is the case
            for summed lifetime information yielded by 'add_keyvals'.
            Dicts must be identically structured.

        Args:
            dict1 (dict): First dictionary to add.
            dict2 (dict): Second dictionary to add.

        Returns:
            Single dictionary of combined values.

        Raises:
            KeyError: When added dict keys do not match.
        """
        for (k, i), (k2, i2) in zip(
                sorted(dict1.items()), sorted(dict2.items())):
            if k == k2 and k == "lifetime":
                continue
            elif k == k2 and k != "lifetime":
                if isinstance(i, dict):
                    self.add_keyvals(i, i2)
                else:
                    if dict1[k] is None:
                        dict1[k] = copy.deepcopy(dict2[k2])
                    else:
                        dict1[k] = dict1[k] + dict2[k]
            else:
                raise KeyError("When adding together two dicts "
                               "for ECM '" + self.name +
                               "' update, dict key structures "
                               "do not match")
        return dict1

    def div_keyvals(self, dict1, dict2):
        """Divide key values of one dict by analogous values of another.

        Note:
            This function is used to generate partitioning fractions
            for key measure results. In that case the function divides a
            measure's climate, building, and end use-specific
            baseline energy use by its total baseline energy use
            (both organized by year). Dicts must be identically structured.

        Args:
            dict1 (dict): Dict with values to divide.
            dict2 (dict): Dict with factors to divide values of first dict by.

        Returns:
            An updated version of the first dict with all original
            values divided by the analogous values in the second dict.
        """
        for (k, i) in dict1.items():
            if isinstance(i, dict):
                self.div_keyvals(i, dict2)
            else:
                # Handle total energy use of zero
                if ((type(dict2[k]) == numpy.ndarray and all(dict2[k]) != 0) or
                        (type(dict2[k]) != numpy.ndarray and dict2[k] != 0)):
                    dict1[k] = dict1[k] / dict2[k]
                else:
                    dict1[k] = 0
        return dict1

    def div_keyvals_float(self, dict1, reduce_num):
        """Divide a dict's key values by a single number.

        Args:
            dict1 (dict): Dictionary with values to divide.
            reduce_num (float): Factor to divide dict values by.

        Returns:
            An updated dict with all original values divided by the
            number.

        """
        for (k, i) in dict1.items():
            if isinstance(i, dict):
                self.div_keyvals_float(i, reduce_num)
            else:
                # Handle zero values
                if ((type(reduce_num) == numpy.ndarray and
                     all(reduce_num) != 0) or (
                        type(reduce_num) != numpy.ndarray and
                        reduce_num != 0)):
                    dict1[k] = dict1[k] / reduce_num
                else:
                    dict1[k] = 0
        return dict1

    def div_keyvals_float_restrict(self, dict1, reduce_num):
        """Divide a dict's key values by a factor, with restrictions.

        Note:
            This function handles the special case where square footage
            is used as microsegment stock and double counted stock/stock
            cost must be factored out. As this special case only concerns
            microsegment stock and stock cost numbers, the function
            does not apply the input factor to microsegment energy,
            carbon, and lifetime information in the dict.

        Args:
            dict1 (dict): Dictionary with values to divide.
            reduce_num (float): Number to divide dict values by.

        Returns:
            An updated dict with all non-restricted original values divided
            by the number.
        """
        for (k, i) in dict1.items():
            # Do not divide any energy, carbon, lifetime, or sub-market
            # scaling information
            if (k == "energy" or k == "carbon" or k == "lifetime" or
                    k == "sub-market scaling" or k == "fugitive emissions"):
                continue
            else:
                if isinstance(i, dict):
                    self.div_keyvals_float_restrict(i, reduce_num)
                else:
                    dict1[k] = dict1[k] / reduce_num
        return dict1

    def adj_pkg_mseg_keyvals(self, cmsegs, base_adj, eff_adj, eff_adj_c,
                             base_eff_flag, comp_flag):
        """ Adjust contributing microsegments in a package to remove overlaps.

        Args:
            cmsegs (dict): Contributing microsegment data to adjust.
            base_adj (dict): Adjustment factors for baseline mseg values.
            eff_adj (dict): Adjustment factors for total efficient mseg values.
            eff_adj_c (dict): Adj. factors for competed efficient mseg values.
            base_eff_flag (boolean): Flag to determine whether current
                adjustment is to a baseline or efficient value.
            comp_flag (boolean): Flag to determine whether current adjustment
                is to a total or competed value.

        Returns:
            Dict of adjusted contributing microsegments for a measure in a
            package that accounts for/removes overlaps with the contributing
            microsegments of other measures in the package.
        """

        # Move down the dict until a non-dict terminal value is encountered
        for (k, i) in cmsegs.items():
            # If a "baseline" or "efficient" key is encountered, flag the
            # adjustment appropriately. Note that these keys will only be
            # encountered for "energy" and "carbon" msegs; other msegs are
            # not broken out by these keys and will yield a flag of None.
            if k in ["baseline", "efficient"]:
                base_eff_flag = k
            # If a "competed" key is encountered, flag the adjustment
            # appropriately (in some cases competed values are adjusted
            # differently than total values)
            if k in "competed":
                comp_flag = True
            if isinstance(i, dict):
                self.adj_pkg_mseg_keyvals(
                    i, base_adj, eff_adj, eff_adj_c,
                    base_eff_flag, comp_flag)
            else:
                # Set the appropriate overlap adjustment factor based on
                # the flag for baseline/efficient adjustments; if this flag
                # is None, set the adjustment to the baseline value
                if base_eff_flag == "baseline" and k in \
                        self.handyvars.aeo_years:
                    # Handle cases where the baseline adjustment is not
                    # broken out by year
                    try:
                        adj_fact = base_adj[k]
                    except TypeError:
                        adj_fact = base_adj
                elif base_eff_flag == "efficient" and k in \
                        self.handyvars.aeo_years:
                    # If a competed efficient value adjustment is flagged, use
                    # the appropriate competed adjustment fraction; otherwise,
                    # use the total efficient adjustment fraction
                    if comp_flag:
                        eff_adj_fact = eff_adj_c
                    else:
                        eff_adj_fact = eff_adj
                    # Handle cases where the efficient adjustment is not
                    # broken out by year
                    try:
                        adj_fact = eff_adj_fact[k]
                    except TypeError:
                        adj_fact = eff_adj_fact
                else:
                    try:
                        adj_fact = base_adj[k]
                    except TypeError:
                        adj_fact = base_adj
                # Apply the adjustment to the mseg value
                cmsegs[k] = cmsegs[k] * adj_fact

        return cmsegs

    def rand_list_gen(self, distrib_info, nsamples):
        """Generate N samples from a given probability distribution.

        Args:
            distrib_info (list): Distribution type and parameters.
            nsamples (int): Number of samples to draw from distribution.

        Returns:
            Numpy array of samples from the input distribution.

        Raises:
            ValueError: When unsupported probability distribution is present.
        """
        # Generate a list of randomly generated numbers using the
        # distribution name and parameters provided in "distrib_info".
        # Check that the correct number of parameters is specified for
        # each distribution.
        if len(distrib_info) == 3 and distrib_info[0] == "normal":
            rand_list = numpy.random.normal(distrib_info[1],
                                            distrib_info[2], nsamples)
        elif len(distrib_info) == 3 and distrib_info[0] == "lognormal":
            rand_list = numpy.random.lognormal(distrib_info[1],
                                               distrib_info[2], nsamples)
        elif len(distrib_info) == 3 and distrib_info[0] == "uniform":
            rand_list = numpy.random.uniform(distrib_info[1],
                                             distrib_info[2], nsamples)
        elif len(distrib_info) == 3 and distrib_info[0] == "gamma":
            rand_list = numpy.random.gamma(distrib_info[1],
                                           distrib_info[2], nsamples)
        elif len(distrib_info) == 3 and distrib_info[0] == "weibull":
            rand_list = numpy.random.weibull(distrib_info[1], nsamples)
            rand_list = distrib_info[2] * rand_list
        elif len(distrib_info) == 4 and distrib_info[0] == "triangular":
            rand_list = numpy.random.triangular(distrib_info[1],
                                                distrib_info[2],
                                                distrib_info[3], nsamples)
        else:
            raise ValueError(
                "Unsupported input distribution specification for ECM '" +
                self.name + "'")

        return rand_list

    def fill_perf_dict(
            self, perf_dict, eplus_perf_array, vintage_weights,
            base_cols, eplus_bldg_types):
        """Fill an empty dict with updated measure performance information.

        Note:
            Use structured array data drawn from an EnergyPlus output file
            and building type/vintage weighting data to fill a dictionary of
            final measure performance information.

        Args:
            perf_dict (dict): Empty dictionary to fill with EnergyPlus-based
                performance information broken down by climate zone, building
                type/vintage, fuel type, and end use.
            eplus_perf_array (numpy recarray): Structured array of EnergyPlus
                energy savings information for the Measure.
            vintage_weights (dict): Square-footage-derived weighting factors
                for each EnergyPlus building vintage type.
            eplus_bldg_types (dict): Scout-EnergyPlus building type mapping
                data, including weighting factors needed to map multiple
                EnergyPlus building types to a single Scout building type.
                Drawn from EPlusMapDicts object's 'bldgtype' attribute.

        Returns:
            A measure performance dictionary filled with relative energy
            savings values from an EnergyPlus simulation output file.

        Raises:
            KeyError: If an EnergyPlus category name cannot be mapped to the
                input perf_dict keys.
            ValueError: If weights used to map multiple EnergyPlus reference
                building types to a single Scout building type do not sum to 1.
        """
        # Instantiate useful EnergyPlus-Scout mapping dicts
        handydicts = EPlusMapDicts()

        # Set the header of the EnergyPlus input array (used when reducing
        # columns of the array to the specific performance categories being
        # updated below)
        eplus_header = list(eplus_perf_array.dtype.names)

        # Loop through zeroed out measure performance dictionary input and
        # update the values with data from the EnergyPlus input array
        for key, item in perf_dict.items():
            # If current dict item is itself a dict, reduce EnergyPlus array
            # based on the current dict key and proceed further down the dict
            # levels
            if isinstance(item, dict):
                # Microsegment type level (no update to EnergyPlus array
                # required)
                if key in ['primary', 'secondary']:
                    updated_perf_array = eplus_perf_array
                # Climate zone level
                elif key in handydicts.czone.keys():
                    # Reduce EnergyPlus array to only rows with climate zone
                    # currently being updated in the performance dictionary
                    updated_perf_array = eplus_perf_array[numpy.where(
                        eplus_perf_array[
                            'climate_zone'] == handydicts.czone[key])].copy()
                    if len(updated_perf_array) == 0:
                        raise KeyError(
                            "EPlus climate zone name not found for ECM '" +
                            self.name + "'")
                # Building type level
                elif key in handydicts.bldgtype.keys():
                    # Determine relevant EnergyPlus building types for current
                    # Scout building type
                    eplus_bldg_types = handydicts.bldgtype[key]
                    if sum(eplus_bldg_types.values()) != 1:
                        raise ValueError(
                            "EPlus building type weights do not sum to 1 "
                            "for ECM '" + self.name + "'")
                    # Reduce EnergyPlus array to only rows with building type
                    # relevant to current Scout building type
                    updated_perf_array = eplus_perf_array[numpy.in1d(
                        eplus_perf_array['building_type'],
                        list(eplus_bldg_types.keys()))].copy()
                    if len(updated_perf_array) == 0:
                        raise KeyError(
                            "EPlus building type name not found for ECM '" +
                            self.name + "'")
                # Fuel type level
                elif key in handydicts.fuel.keys():
                    # Reduce EnergyPlus array to only columns with fuel type
                    # currently being updated in the performance dictionary,
                    # plus bldg. type/vintage, climate, and measure columns
                    colnames = base_cols + [
                        x for x in eplus_header if handydicts.fuel[key] in x]
                    if len(colnames) == len(base_cols):
                        raise KeyError(
                            "EPlus fuel type name not found for ECM '" +
                            self.name + "'")
                    updated_perf_array = eplus_perf_array[colnames].copy()
                # End use level
                elif key in handydicts.enduse.keys():
                    # Reduce EnergyPlus array to only columns with end use
                    # currently being updated in the performance dictionary,
                    # plus bldg. type/vintage, climate, and measure columns
                    colnames = base_cols + [
                        x for x in eplus_header if x in handydicts.enduse[
                            key]]
                    if len(colnames) == len(base_cols):
                        raise KeyError(
                            "EPlus end use name not found for ECM '" +
                            self.name + "'")
                    updated_perf_array = eplus_perf_array[colnames].copy()
                else:
                    raise KeyError(
                        "Invalid performance dict key for ECM '" +
                        self.name + "'")

                # Given updated EnergyPlus array, proceed further down the
                # dict level hierarchy
                self.fill_perf_dict(
                    item, updated_perf_array, vintage_weights,
                    base_cols, eplus_bldg_types)
            else:
                # Reduce EnergyPlus array to only rows with structure type
                # currently being updated in the performance dictionary
                # ('new' or 'retrofit')
                if key in handydicts.structure_type.keys():
                    # A 'new' structure type will match only one of the
                    # EnergyPlus building vintage names
                    if key == "new":
                        updated_perf_array = eplus_perf_array[numpy.where(
                            eplus_perf_array['template'] ==
                            handydicts.structure_type['new'])].copy()
                    # A 'retrofit' structure type will match multiple
                    # EnergyPlus building vintage names
                    else:
                        updated_perf_array = eplus_perf_array[numpy.in1d(
                            eplus_perf_array['template'], list(
                                handydicts.structure_type[
                                    'retrofit'].keys()))].copy()
                    if len(updated_perf_array) == 0 or \
                       (key == "new" and
                        len(numpy.unique(updated_perf_array[
                            'template'])) != 1 or key == "retrofit" and
                        len(numpy.unique(updated_perf_array[
                            'template'])) != len(
                            handydicts.structure_type["retrofit"].keys())):
                        raise ValueError(
                            "EPlus vintage name not found for ECM '" +
                            self.name + "'")
                else:
                    raise KeyError(
                        "Invalid performance dict key for ECM '" +
                        self.name + "'")

                # Separate filtered array into the rows representing measure
                # consumption and those representing baseline consumption
                updated_perf_array_m, updated_perf_array_b = [
                    updated_perf_array[updated_perf_array[
                        'measure'] != 'none'],
                    updated_perf_array[updated_perf_array[
                        'measure'] == 'none']]
                # Ensure that a baseline consumption row exists for every
                # measure consumption row retrieved
                if len(updated_perf_array_m) != len(updated_perf_array_b):
                    raise ValueError(
                        "Lengths of ECM and baseline EPlus data arrays "
                        "are unequal for ECM '" + self.name + "'")
                # Initialize total measure and baseline consumption values
                val_m, val_b = (0 for n in range(2))

                # Weight and combine the measure/baseline consumption values
                # left in the EnergyPlus arrays; subtract total measure
                # consumption from baseline consumption and divide by baseline
                # consumption to reach relative savings value for the current
                # dictionary branch
                for ind in range(0, len(updated_perf_array_m)):
                    row_m, row_b = [
                        updated_perf_array_m[ind], updated_perf_array_b[ind]]
                    # Loop through remaining columns with consumption data
                    for n in eplus_header:
                        if row_m[n].dtype.char != 'S' and \
                                row_m[n].dtype.char != 'U':
                            # Find appropriate building type to weight
                            # consumption data points by
                            eplus_bldg_type_wt_row_m, \
                                eplus_bldg_type_wt_row_b = [
                                    eplus_bldg_types[row_m['building_type']],
                                    eplus_bldg_types[row_b['building_type']]]
                            # Weight consumption data points by factors for
                            # appropriate building type and vintage
                            row_m_val, row_b_val = [(
                                row_m[n] * eplus_bldg_type_wt_row_m *
                                vintage_weights[row_m['template'].copy()]),
                                (row_b[n] * eplus_bldg_type_wt_row_b *
                                 vintage_weights[row_b['template'].copy()])]
                            # Add weighted measure consumption data point to
                            # total measure consumption
                            val_m += row_m_val
                            # Add weighted baseline consumption data point to
                            # total base consumption
                            val_b += row_b_val
                    # Find relative savings if total baseline use != zero
                    if val_b != 0:
                        end_key_val = (val_b - val_m) / val_b
                    else:
                        end_key_val = 0

                # Update the current dictionary branch value to the final
                # measure relative savings value derived above
                perf_dict[key] = round(end_key_val, 3)

        return perf_dict

    def create_perf_dict(self, msegs):
        """Create dict to fill with updated measure performance information.

        Note:
            Given a measure's applicable climate zone, building type,
            structure type, fuel type, and end use, create a dict of zeros
            with a hierarchy that is defined by these measure properties.

        Args:
            msegs (dict): Baseline microsegment stock and energy use
                information to use in validating categorization of
                measure performance information.

        Returns:
            Empty dictionary to fill with EnergyPlus-based performance
            information broken down by climate zone, building type/vintage,
            fuel type, and end use.
        """
        # Initialize performance dict
        perf_dict_empty = {"primary": None, "secondary": None}
        # Create primary dict structure from baseline market properties
        perf_dict_empty["primary"] = self.create_nested_dict(
            msegs, "primary")

        # Create secondary dict structure from baseline market properties
        # (if needed)
        if isinstance(self.end_use, dict):
            perf_dict_empty["secondary"] = self.create_nested_dict(
                msegs, "secondary")

        return perf_dict_empty

    def create_nested_dict(self, msegs, mseg_type):
        """Create a nested dictionary based on a pre-defined branch structure.

        Note:
            Create a nested dictionary with a structure that is defined by a
            measure's applicable baseline market, with end leaf node values set
            to zero.

        Args:
            msegs (dict): Baseline microsegment stock and energy use
                information to use in validating categorization of
                measure performance information.
            mseg_type (string): Primary or secondary microsegment type flag.

        Returns:
            Nested dictionary of zeros with desired branch structure.
        """
        # Initialize output dictionary
        output_dict = {}
        # Establish levels of the dictionary key hierarchy from measure's
        # applicable baseline market information
        keylevels = [
            self.climate_zone, self.bldg_type, self.fuel_type[mseg_type],
            self.end_use[mseg_type], self.structure_type]
        # Find all possible dictionary key chains from the above key level
        # info.
        dict_keys = list(itertools.product(*keylevels))
        # Remove all natural gas cooling key chains (EnergyPlus output
        # files do not include a column for natural gas cooling)
        dict_keys = [x for x in dict_keys if not (
            'natural gas' in x and 'cooling' in x)]

        # Use an input dictionary with valid baseline microsegment information
        # to check that each of the microsegment key chains generated above is
        # valid for the current measure; if not, remove each invalid key chain
        # from further operations

        # Initialize a list of valid baseline microsegment key chains for the
        # measure
        dict_keys_fin = []
        # Loop through the initial set of candidate key chains generated above
        for kc in dict_keys:
            # Copy the input dictionary containing valid key chains
            dict_check = copy.deepcopy(msegs)
            # Loop through all keys in the candidate key chain and move down
            # successive levels of the input dict until either the end of the
            # key chain is reached or a key is not found in the list of valid
            # keys for the current input dict level. In the former case, the
            # resultant dict will point to all technologies associated with the
            # current key chain (e.g., ASHP, LFL, etc.) If none of these
            # technologies are found in the list of technologies covered by the
            # measure, the key chain is deemed invalid
            for ind, key in enumerate(kc):
                # If key is found in the list of valid keys for the current
                # input microsegment dict level, move on to next level in the
                # dict; otherwise, break the current loop
                if key in dict_check.keys():
                    dict_check = dict_check[key]
                    # In the case of heating or cooling end uses, an additional
                    # 'technology type' key must be accounted for ('supply' or
                    # 'demand')
                    if key in ['heating', 'cooling']:
                        dict_check = \
                            dict_check[self.technology_type[mseg_type]]
                else:
                    break

            # If any of the technology types listed in the measure definition
            # are found in the keys of the dictionary yielded by the above
            # loop, add the key chain to the list that is used to define the
            # final nested dictionary output (e.g., the key chain is valid)
            if any([x in self.technology[mseg_type]
                   for x in dict_check.keys()]):
                dict_keys_fin.append(kc)

        # Loop through each of the valid key chains and create an
        # associated path in the dictionary, terminating with a zero value
        # to be updated in a subsequent routine with EnergyPlus output data
        for kc in dict_keys_fin:
            current_level = output_dict
            for ind, elem in enumerate(kc):
                if elem not in current_level and (ind + 1) != len(kc):
                    current_level[elem] = {}
                elif elem not in current_level and (ind + 1) == len(kc):
                    current_level[elem] = 0
                current_level = current_level[elem]

        return output_dict

    def build_array(self, eplus_coltyp, files_to_build):
        """Assemble EnergyPlus data from one or more CSVs into a record array.

        Args:
            eplus_coltypes (list): Expected EnergyPlus variable data types.
            files_to_build (CSV objects): CSV files of EnergyPlus energy
                consumption information under measure and baseline cases.

        Returns:
            Structured array of EnergyPlus energy savings information for the
            Measure.
        """
        # Loop through CSV file objects and import/add to record array
        for ind, f in enumerate(files_to_build):
            # Read in CSV file to array
            eplus_file = numpy.genfromtxt(f, names=True, dtype=eplus_coltyp,
                                          delimiter=",", missing_values='')
            # Find only those rows in the array that represent
            # completed simulation runs for the measure of interest
            eplus_file = eplus_file[(eplus_file[
                'measure'] == self.energy_efficiency['EnergyPlus file']) |
                (eplus_file['measure'] == 'none') &
                (eplus_file['status'] == 'completed normal')]
            # Initialize or add to a master array that covers all CSV data
            if ind == 0:
                eplus_perf_array = eplus_file
            else:
                eplus_perf_array = \
                    numpy.concatenate((eplus_perf_array, eplus_file))

        return eplus_perf_array


class MeasurePackage(Measure):
    """Set up a class representing packaged efficiency measures as objects.

    The MeasurePackage class is a subclass of the Measure class.

    Attributes:
        handyvars (object): Global variables useful across class methods.
        contributing_ECMs (list): List of measures to package.
        contributing_ECMs_eqp (list): Subset of packaged measures that applies
            to equipment improvements.
        contributing_ECMs_env (list): Subset of packaged measures that applies
            to envelope improvements.
        pkg_env_costs (boolean): Flag for whether or not to include contrib.
            envelope measure costs in the package (if applicable).
        pkg_env_cost_convert_data (dict): Data needed to convert envelope
            cost data to match HVAC equipment cost units in commercial cases.
        name (string): Package name.
        benefits (dict): Percent improvements in energy savings and/or cost
            reductions from packaging measures.
        remove (boolean): Determines whether package should be removed from
            analysis engine due to insufficient market source data.
        usr_opts (dict): Records several user command line input
            selections that affect measure outputs.
        market_entry_year (int): Earliest year of market entry across all
            measures in the package.
        market_exit_year (int): Latest year of market exit across all
            measures in the package.
        yrs_on_mkt (list): List of years that the measure is active on market.
        meas_typ (string): "full service" or "add-on".
        climate_zone (list): Applicable climate zones for package.
        bldg_type (list): Applicable building types for package.
        structure_type (list): Applicable structure types for package.
        fuel_type (dict): Applicable primary fuel type for package.
        end_use (dict): Applicable primary end use type for package.
        technology (list): Applicable technologies for package.
        technology_type (dict): Applicable technology types (supply/demand) for
            package.
        fuel_switch_to (str, None): Fuel switch to flag.
        tech_switch_to (str, None): Technology switch to flag.
        markets (dict): Data grouped by adoption scheme on:
            a) 'master_mseg': a package's master market microsegments (stock,
               energy, carbon, cost),
            b) 'mseg_adjust': all microsegments that contribute to each master
               microsegment (required later for measure competition).
            c) 'mseg_out_break': master microsegment breakdowns by key
               variables (e.g., climate zone, building type, end use, etc.)
        eff_fs_splt (dict): Data needed to determine the fuel splits of
            efficient case results for fuel switching measures.
        htcl_overlaps (dict): Dict to store data on heating/cooling overlaps
            across contributing equipment vs. envelope ECMs that apply to the
            same region, building type/vintage, fuel type, and end use.
        sector_shapes (dict): Sector-level hourly baseline and efficient load
            shapes by adopt scheme, EMM region, and year.

    """

    def __init__(self, measure_list_package, p, bens, handyvars, handyfiles,
                 opts, convert_data):
        self.name = p
        self.handyvars = handyvars
        self.contributing_ECMs = copy.deepcopy(measure_list_package)
        # Check to ensure energy output settings for all measures that
        # contribute to the package are identical
        if not all([all([m.usr_opts[x] ==
                         self.contributing_ECMs[0].usr_opts[x] for
                         x in self.contributing_ECMs[0].usr_opts.keys()])
                    for m in self.contributing_ECMs[1:]]):
            raise ValueError(
                "Package '" + self.name + "' attempts to merge measures with "
                "different energy output settings; re-prepare package's "
                "contributing ECMs to ensure these settings are identical")
        self.contributing_ECMs_eqp = [
            m for m in self.contributing_ECMs if
            m.technology_type["primary"][0] == "supply"]
        # Raise error if package does not contain any equipment ECMs;
        # packages are oriented around these ECMs
        if len(self.contributing_ECMs_eqp) == 0:
            raise ValueError(
                "ECM package " + self.name + " has no contributing ECMs; "
                "envelope components should be packaged within a single ECM "
                "definition rather than via the packaging feature")
        # Raise error if package is not merging measures of the type expected
        # to be supported by packaged – integration of HVAC equipment and
        # envelope and/or controls, or integration of lighting equipment and
        # controls; also raise error if package is merging equipment measures
        # with inconsistent fuel switching settings
        elif not (all([all([x in ["heating", "secondary heating",
                                  "ventilation", "cooling"] for x in
                            m.end_use["primary"]])
                       for m in self.contributing_ECMs_eqp]) or (
                  all([m.end_use["primary"] == "lighting" for
                       m in self.contributing_ECMs_eqp]) and
                  any([m.measure_type == "add-on" for
                       m in self.contributing_ECMs_eqp]))):
            raise ValueError(
                "ECM package " + self.name + " contains unsupported measures. "
                "Packages are meant to account for interactions between HVAC "
                "equipment and envelope and/or between HVAC and/or lighting "
                "equipment and controls; remove all contributing measures "
                "that are outside of this scope")
        elif not all([m.fuel_switch_to ==
                      self.contributing_ECMs_eqp[0].fuel_switch_to
                      for m in self.contributing_ECMs_eqp[1:]]):
            raise ValueError(
                "ECM package " + self.name + " merges measures with different "
                "'fuel_switch_to' attribute values; ensure the value for this "
                "attribute is set consistently across all packaged measures")

        self.contributing_ECMs_env = [
            m for m in self.contributing_ECMs if
            m.technology_type["primary"][0] == "demand"]
        # If there are envelope measures in the package, register settings
        # around whether to include the costs of those measures in pkg. costs
        if len(self.contributing_ECMs_env) != 0:
            self.pkg_env_costs = opts.pkg_env_costs
        else:
            self.pkg_env_costs = False
        self.pkg_env_cost_convert_data = convert_data
        # Check to ensure that measure name is proper length for plotting
        # *** COMMENT FOR NOW ***
        # if len(self.name) > 40:
        #     raise ValueError(
        #         "ECM '" + self.name + "' name must be <= 40 characters")
        self.benefits = bens
        self.remove = False
        # Set energy outputs and fuel/tech switching properties to that of the
        # first equipment measure; consistency in these properties across
        # measures in the package was checked for above
        self.usr_opts = self.contributing_ECMs_eqp[0].usr_opts
        self.fuel_switch_to = self.contributing_ECMs_eqp[0].fuel_switch_to
        self.tech_switch_to = self.contributing_ECMs_eqp[0].tech_switch_to
        # Set market entry year as earliest of all the packaged eqp. measures
        if any([x.market_entry_year is None or (int(
                x.market_entry_year) < int(x.handyvars.aeo_years[0])) for x in
               self.contributing_ECMs_eqp]):
            self.market_entry_year = int(handyvars.aeo_years[0])
        else:
            self.market_entry_year = min([
                x.market_entry_year for x in self.contributing_ECMs_eqp])
        # Set market exit year is latest of all the packaged eqp. measures
        if any([x.market_exit_year is None or (int(
                x.market_exit_year) > (int(x.handyvars.aeo_years[0]) + 1)) for
                x in self.contributing_ECMs_eqp]):
            self.market_exit_year = int(handyvars.aeo_years[-1]) + 1
        else:
            self.market_exit_year = max([
                x.market_entry_year for x in self.contributing_ECMs_eqp])
        self.yrs_on_mkt = [
            str(i) for i in range(
                self.market_entry_year, self.market_exit_year)]
        if all([m.measure_type == "full service" for m in
                self.contributing_ECMs_eqp]):
            self.meas_typ = "full service"
        elif all([m.measure_type == "add-on" for m in
                  self.contributing_ECMs_eqp]):
            self.meas_typ = "add-on"
        else:
            self.meas_typ = "full service"
        self.climate_zone, self.bldg_type, self.structure_type = (
                [] for n in range(3))
        self.fuel_type, self.end_use, self.technology, \
            self.technology_type = (
                {"primary": [], "secondary": None} for n in range(4))
        self.markets = {}
        self.eff_fs_splt = {
            a_s: {} for a_s in handyvars.adopt_schemes}
        self.sector_shapes = None
        self.htcl_overlaps = {adopt: {"keys": [], "data": {}} for
                              adopt in handyvars.adopt_schemes}
        for adopt_scheme in handyvars.adopt_schemes:
            self.markets[adopt_scheme] = {
                "master_mseg": {
                    "stock": {
                        "total": {
                            "all": None, "measure": None},
                        "competed": {
                            "all": None, "measure": None}},
                    "energy": {
                        "total": {
                            "baseline": None, "efficient": None},
                        "competed": {
                            "baseline": None, "efficient": None}},
                    "carbon": {
                        "total": {
                            "baseline": None, "efficient": None},
                        "competed": {
                            "baseline": None, "efficient": None}},
                    "cost": {
                        "stock": {
                            "total": {
                                "baseline": None, "efficient": None},
                            "competed": {
                                "baseline": None, "efficient": None}},
                        "energy": {
                            "total": {
                                "baseline": None, "efficient": None},
                            "competed": {
                                "baseline": None, "efficient": None}},
                        "carbon": {
                            "total": {
                                "baseline": None, "efficient": None},
                            "competed": {
                                "baseline": None, "efficient": None}}},
                    "lifetime": {"baseline": None, "measure": None}},
                "mseg_adjust": {
                    "contributing mseg keys and values": {},
                    "competed choice parameters": {},
                    "secondary mseg adjustments": {
                        "sub-market": {
                            "original energy (total)": {},
                            "adjusted energy (sub-market)": {}},
                        "stock-and-flow": {
                            "original energy (total)": {},
                            "adjusted energy (previously captured)": {},
                            "adjusted energy (competed)": {},
                            "adjusted energy (competed and captured)": {}},
                        "market share": {
                            "original energy (total captured)": {},
                            "original energy (competed and captured)": {},
                            "adjusted energy (total captured)": {},
                            "adjusted energy (competed and captured)": {}}}},
                "mseg_out_break": {key: {
                    "baseline": copy.deepcopy(self.handyvars.out_break_in),
                    "efficient": copy.deepcopy(self.handyvars.out_break_in),
                    "savings": copy.deepcopy(self.handyvars.out_break_in)} for
                    key in ["energy", "carbon", "cost"]}}

            # Add fugitive emissions key to output dict if fugitive
            # emissions option is set
            if self.usr_opts["fugitive_emissions"] is not False:
                # Initialize methane/refrigerant values as zero when that
                # fugitive option is being assessed (in which case new data
                # will be summed into the zero dicts), and as None otherwise

                # Check for methane assessment/initialization of zero dict
                if self.usr_opts["fugitive_emissions"][0] in ['1', '3']:
                    init_meth = {yr: 0 for yr in self.handyvars.aeo_years}
                else:
                    init_meth = None
                # Check for refrigerants assessment/initialization of zero dict
                if self.usr_opts["fugitive_emissions"][0] in ['2', '3']:
                    init_refr = {yr: 0 for yr in self.handyvars.aeo_years}
                else:
                    init_refr = None
                # Organize methane and refrigerants dict under broader key
                self.markets[adopt_scheme]["master_mseg"][
                    "fugitive emissions"] = {
                        "methane": {
                            "total": {
                                "baseline": copy.deepcopy(init_meth),
                                "efficient": copy.deepcopy(init_meth)},
                            "competed": {
                                "baseline": copy.deepcopy(init_meth),
                                "efficient": copy.deepcopy(init_meth)}},
                        "refrigerants": {
                            "total": {
                                "baseline": copy.deepcopy(init_refr),
                                "efficient": copy.deepcopy(init_refr)},
                            "competed": {
                                "baseline": copy.deepcopy(init_refr),
                                "efficient": copy.deepcopy(init_refr)}}}

    def merge_measures(self, opts):
        """Merge the markets information of multiple individual measures.

        Attributes:
            opts (object): Stores user-specified execution options.

        Note:
            Combines the 'markets' attributes of each individual measure into
            a packaged 'markets' attribute.

        Returns:
            Updated 'markets' attribute for a packaged measure that combines
            the 'markets' attributes of multiple individual measures.
        """
        # Initialize a list of dicts for storing measure microsegment data as
        # it is looped through and updated
        mseg_dat_rec = [{} for n in range(len(self.contributing_ECMs_eqp))]

        # When a user wishes to package envelope costs with equipment costs
        # in a package, pull a common set of measure-captured stock figures
        # to use in calculating the envelope costs to add to the package
        # later, across all equipment measures in the package
        if opts.pkg_env_costs is not False and len(
                self.contributing_ECMs_env) > 0:
            # Initialize dict of common stock data for all equipment
            # measures in the package
            common_stk = {a_s: {} for a_s in self.handyvars.adopt_schemes}
            # Loop through all equipment measures in the package
            for m in self.contributing_ECMs_eqp:
                # Loop through all adoption scenarios
                for a_s in self.handyvars.adopt_schemes:
                    # Shorthand deep copy of measure stock data
                    stk_cpy = copy.deepcopy(m.markets[a_s]["mseg_adjust"][
                        "contributing mseg keys and values"])
                    # Loop through all contributing msegs for measure
                    for cm in stk_cpy.keys():
                        # If contributing mseg is not already
                        # represented in the common measure-captured stock
                        # data, add stock data for measure/adoption scn./
                        # contributing mseg
                        if cm not in common_stk.keys():
                            common_stk[a_s][cm] = {met: {yr: stk_cpy[
                                cm]["stock"][met]["measure"][yr]
                                for yr in self.handyvars.aeo_years} for
                                met in ["total", "competed"]}
                        # If contributing mseg is already represented
                        # in the common stock data, add stock data for
                        # contributing measure/adoption scn./contributing mseg
                        # only if the data values are higher than those already
                        # stored (e.g., pull the highest measure-captured
                        # stock values from equipment measures to use in
                        # adding envelope costs to the package later)
                        else:
                            common_stk[a_s][cm] = {met: {yr: stk_cpy[
                                cm]["stock"][met]["measure"][yr] if
                                stk_cpy[cm]["stock"][met]["measure"][yr] >
                                common_stk[a_s][cm][met][yr] else
                                common_stk[a_s][cm][met][yr]
                                for yr in self.handyvars.aeo_years} for
                                met in ["total", "competed"]}
        else:
            common_stk = None

        # Loop through each measure and either adjust and record its attributes
        # for further processing or directly add its attributes to the merged
        # package measure definition
        for ind, m in enumerate(self.contributing_ECMs_eqp):
            # Add measure climate zones
            self.climate_zone.extend(
                list(set(m.climate_zone) - set(self.climate_zone)))
            # Add measure building types
            self.bldg_type.extend(
                list(set(m.bldg_type) - set(self.bldg_type)))
            # Add measure structure types
            self.structure_type.extend(
                list(set(m.structure_type) - set(self.structure_type)))
            # Add measure fuel types
            self.fuel_type["primary"].extend(
                list(set(m.fuel_type["primary"]) -
                     set(self.fuel_type["primary"])))
            # Add measure end uses
            self.end_use["primary"].extend(
                list(set(m.end_use["primary"]) -
                     set(self.end_use["primary"])))
            # Add measure technologies
            self.technology["primary"].extend(
                list(set(m.technology["primary"]) -
                     set(self.technology["primary"])))
            # Add measure technology types
            self.technology_type["primary"].extend(
                list(set(m.technology_type["primary"]) -
                     set(self.technology_type["primary"])))
            # Initialize sector-level load shape information for the package
            # if this info has not yet already been initialized and measure
            # in package has sector shape information
            if not self.sector_shapes and m.sector_shapes:
                self.sector_shapes = {
                    a_s: {reg: {yr: {"baseline": [0 for n in range(8760)],
                                     "efficient": [0 for n in range(8760)]} for
                                yr in self.handyvars.aeo_years_summary} for
                          reg in self.climate_zone}
                    for a_s in self.handyvars.adopt_schemes}

            # Update measure contributing mseg data by adoption scheme
            for adopt_scheme in self.handyvars.adopt_schemes:
                # Set contributing microsegment data for individual measure;
                # deep copy to ensure that initial measure data are retained
                # throughout the updates
                msegs_meas_init = copy.deepcopy(
                    m.markets[adopt_scheme]["mseg_adjust"])
                # Set output breakout data for individual measure (used to
                # break out results by climate, building, and end use)
                mseg_out_break_init = copy.deepcopy(
                    m.markets[adopt_scheme]["mseg_out_break"])
                # Set contributing microsegment data to update for package
                # under the current adoption scheme
                msegs_pkg_init = self.markets[adopt_scheme]["mseg_adjust"]
                # Initialize variables to track adjustments to sector load
                # shapes for each individual measure to support packaging,
                # if sector-level load shapes are being generated for the
                # package and individual measure sector shapes are available;
                # otherwise set to None
                if self.sector_shapes and m.sector_shapes:
                    sect_shp_e_init, sect_shp_e_fin = ({
                        reg: {yr: {"baseline": 0, "efficient": 0}
                              for yr in self.handyvars.aeo_years_summary}
                        for reg in m.climate_zone} for n in range(2))
                else:
                    sect_shp_e_init, sect_shp_e_fin = (None for n in range(2))
                # Loop through/update all microsegment data for measure
                for k in msegs_meas_init.keys():
                    # Remove any direct overlaps between the measure's
                    # contributing msegs and those of other measures in the
                    # package (e.g., msegs with the exact same key information)
                    if k == "contributing mseg keys and values":
                        for cm in msegs_meas_init[k].keys():
                            # If applicable, develop shorthand for data needed
                            # to split efficient-case energy by fuel type
                            if len(m.eff_fs_splt[adopt_scheme].keys()) != 0 \
                                and cm in m.eff_fs_splt[
                                    adopt_scheme].keys():
                                fs_eff_splt = m.eff_fs_splt[adopt_scheme][cm]
                                # Add fuel split information to that for the
                                # overall package if not already reflected
                                # in this package attribute
                                if cm not in self.eff_fs_splt[
                                        adopt_scheme].keys():
                                    self.eff_fs_splt[adopt_scheme][cm] = \
                                        copy.deepcopy(fs_eff_splt)
                            else:
                                fs_eff_splt = None
                            # Convert mseg string to list for further calcs.
                            key_list = literal_eval(cm)
                            # Add to initial annual electricity use that
                            # concerns the measure's sector shape, before
                            # package adjustments
                            if sect_shp_e_init is not None:
                                # If applicable, pull fuel split data for
                                # energy variable to use in sect shape calcs
                                if fs_eff_splt is not None:
                                    fs_eff_splt_sect_shape = \
                                        fs_eff_splt["energy"]
                                else:
                                    fs_eff_splt_sect_shape = None
                                # Add pre-adjusted energy use for the current
                                # mseg to tracking data for sector shape
                                # adjustments
                                sect_shp_e_init[key_list[1]] = \
                                    self.add_sectshape_energy(
                                        cm, sect_shp_e_init[key_list[1]],
                                        msegs_meas_init[k][cm]["energy"][
                                            "total"], fs_eff_splt_sect_shape,
                                        m.fuel_switch_to)
                            # Merge directly overlapping measure mseg data
                            msegs_meas_init[k][cm], mseg_out_break_init = \
                                self.merge_direct_overlaps(
                                    msegs_meas_init[k][cm], cm, adopt_scheme,
                                    mseg_out_break_init, m.name,
                                    m.fuel_switch_to, m.measure_type,
                                    fs_eff_splt, key_list)
                    # Add all other contributing microsegment data for
                    # the measure
                    elif k in ["competed choice parameters",
                               "secondary mseg adjustments"]:
                        self.update_dict(msegs_pkg_init[k], msegs_meas_init[k])
                # Record adjusted contributing microsegment, output
                # breakout data, and sector shapes data for the measure for
                # subsequent use in removing any overlaps between heating/
                # cooling equipment and envelope measures (if applicable) and
                # finalizing the measure's contribution to the package
                mseg_dat_rec[ind][adopt_scheme] = {
                    "name": m.name,
                    "microsegments": msegs_meas_init,
                    "breakouts": mseg_out_break_init,
                    "sect_shp_orig": m.sector_shapes,
                    "sect_shp_e_init": sect_shp_e_init,
                    "sect_shp_e_fin": sect_shp_e_fin,
                    "fuel_switch_to": m.fuel_switch_to,
                    "eff_fuel_splits": m.eff_fs_splt[adopt_scheme],
                    "regions": m.climate_zone}

        # Loop through all previously adjusted/recorded microsegment and
        # outbreak data for the measures that contribute to the package;
        # if necessary, remove remaining overlaps between heating/cooling
        # equipment and envelope measures in the package; finalize the
        # contributing data for the package
        for m in mseg_dat_rec:
            # Update measure contributing mseg data by adoption scheme
            for adopt_scheme in self.handyvars.adopt_schemes:
                # Set shorthands for contributing mseg, outbreak, sector shapes
                # data (if applicable), and package data to finalize
                msegs_meas_fin = m[adopt_scheme]["microsegments"][
                    "contributing mseg keys and values"]
                mseg_out_break_fin = m[adopt_scheme]["breakouts"]
                # Set shorthand for data used to track annual electricity use
                # that concerns the measure's sector shape after package
                # adjustments
                if self.sector_shapes:
                    sect_shp_e_fin = m[adopt_scheme]["sect_shp_e_fin"]
                else:
                    sect_shp_e_fin = None
                # Set shorthand for final merged data for the current mseg
                msegs_pkg_fin = self.markets[adopt_scheme]["mseg_adjust"][
                    "contributing mseg keys and values"]
                # Loop through/update all microsegment data for measure
                for cm in msegs_meas_fin.keys():
                    # If applicable develop shorthand for data needed
                    # to split efficient-case energy by fuel type
                    if len(m[adopt_scheme][
                        "eff_fuel_splits"].keys()) != 0 and \
                            cm in m[adopt_scheme]["eff_fuel_splits"].keys():
                        fs_eff_splt = m[adopt_scheme]["eff_fuel_splits"][cm]
                    else:
                        fs_eff_splt = None
                    # Convert mseg string to list for further calcs.
                    key_list = literal_eval(cm)
                    # Further adjust equipment msegs to account for
                    # overlapping envelope performance improvements
                    msegs_meas_fin[cm], mseg_out_break_fin = \
                        self.merge_htcl_overlaps(
                            msegs_meas_fin[cm], cm, adopt_scheme,
                            mseg_out_break_fin, m[adopt_scheme]["name"],
                            m[adopt_scheme]["fuel_switch_to"], fs_eff_splt,
                            key_list, opts, common_stk)
                    # Apply any additional energy savings and/or installed cost
                    # benefits from packaging
                    msegs_meas_fin[cm] = self.apply_pkg_benefits(
                        msegs_meas_fin[cm], adopt_scheme, cm)
                    # Add to final annual electricity use that concerns the
                    # measure's sector shape, after package adjustments
                    if sect_shp_e_fin is not None:
                        # If applicable, pull fuel split data for
                        # energy variable to use in sect shape calcs
                        if fs_eff_splt is not None:
                            fs_eff_splt_sect_shape = fs_eff_splt["energy"]
                        else:
                            fs_eff_splt_sect_shape = None
                        # Add post-adjustment energy use for the current
                        # mseg to tracking data for sector shape adjustments
                        sect_shp_e_fin[key_list[1]] = \
                            self.add_sectshape_energy(
                                cm, sect_shp_e_fin[key_list[1]],
                                msegs_meas_fin[cm]["energy"]["total"],
                                fs_eff_splt_sect_shape,
                                m[adopt_scheme]["fuel_switch_to"])
                    # Add finalized contributing microsegment information for
                    # individual measure to the packaged measure, creating a
                    # new contributing microsegment key if one does not already
                    # exist for the package
                    if cm not in msegs_pkg_fin.keys():
                        msegs_pkg_fin[cm] = msegs_meas_fin[cm]
                    else:
                        msegs_pkg_fin[cm] = self.add_keyvals(
                            msegs_pkg_fin[cm], msegs_meas_fin[cm])
                # Generate a dictionary including data on how much of the
                # packaged measure's baseline energy/carbon/cost is attributed
                # to each of the output climate zones, building types, and end
                # uses it applies to
                for v in ["energy", "carbon", "cost"]:
                    for s in ["baseline", "efficient", "savings"]:
                        self.merge_out_break(self.markets[adopt_scheme][
                            "mseg_out_break"][v][s],
                            mseg_out_break_fin[v][s])

                # Adjust individual measure's contributing sector shape
                # information to account for overlaps with other measures in
                # the package and add to the overall package sector shape
                if sect_shp_e_fin:
                    self.sector_shapes[adopt_scheme] = {reg: {yr: {s: [
                        # Add in measure sector shape data, adjusted to
                        # account for any changes in annual electricity use
                        # after packaging, to the package sector shape
                        self.sector_shapes[adopt_scheme][reg][yr][s][x] +
                        m[adopt_scheme]["sect_shp_orig"][
                            adopt_scheme][reg][yr][s][x] * (
                            (m[adopt_scheme]["sect_shp_e_fin"][reg][yr][s] /
                             m[adopt_scheme]["sect_shp_e_init"][reg][yr][s]
                             ) if m[adopt_scheme][
                                "sect_shp_e_fin"][reg][yr][s] != 0 else 1)
                        for x in range(8760)]
                        for s in ["baseline", "efficient"]}
                        for yr in self.handyvars.aeo_years_summary}
                        # Ensure that only package regions concerning currently
                        # added individual measure are looped through
                        for reg in self.climate_zone if reg in m[
                            adopt_scheme]["regions"]}

        # Generate a packaged master microsegment based on the contributing
        # microsegment information defined above
        for adopt_scheme in self.handyvars.adopt_schemes:
            # Loop through all contributing microsegments for the packaged
            # measure and add to the packaged master microsegment
            for k in (self.markets[adopt_scheme]["mseg_adjust"][
                    "contributing mseg keys and values"].keys()):
                self.add_keyvals(
                    self.markets[adopt_scheme]["master_mseg"],
                    self.markets[adopt_scheme]["mseg_adjust"][
                        "contributing mseg keys and values"][k])
            # Set measures to use in calculating average package lifetimes: all
            # non add-on equipment measures if non add-ons are part of package,
            # and otherwise all add-on measures
            if all([m.measure_type == "add-on" for
                    m in self.contributing_ECMs_eqp]):
                life_avg_meas = self.contributing_ECMs_eqp
            else:
                life_avg_meas = [m for m in self.contributing_ECMs_eqp if
                                 m.measure_type != "add-on"]
            # Average lifetimes
            self.markets[adopt_scheme]["master_mseg"]["lifetime"][
                "measure"] = numpy.mean([
                    m.markets[adopt_scheme]["master_mseg"]["lifetime"][
                        "measure"] for m in life_avg_meas])
            self.markets[adopt_scheme]["master_mseg"]["lifetime"][
                "baseline"] = {
                    yr: numpy.mean([m.markets[adopt_scheme][
                        "master_mseg"]["lifetime"]["baseline"][yr]
                        for m in life_avg_meas])
                    for yr in self.handyvars.aeo_years}

    def add_sectshape_energy(self, cm, sect_shape_init, add_energy,
                             fs_eff_splt_energy, fuel_switch_to):
        """Add annual energy to a dict that is used to pkg. sector shapes.

        Args:
            cm (tuple): Microsegment key describing the contributing
                microsegment currently being added (e.g. reg->bldg, etc.)
            sect_shape_init (dict): Annual energy use tracking dataset to add
                the current mseg energy to, used in adjusting measure sector
                shapes for packaging.
            add_energy (dict): Mseg energy data to add to tracking data.
            fs_eff_splt_energy (dict): If applicable, the fuel splits for
                efficient-case measure energy use.
            fuel_switch_to (string): Indicator of which baseline fuel the
                measure switches to (if applicable).

        Returns:
            Updated energy, carbon, and energy cost output breakouts adjusted
            to account for removal of overlaps between measure and other
            measures in a package.
        """

        # Update the total annual electricity use covered by a measure's sector
        # shape in a given region to reflect the current packaged microsegment
        sect_shape_energy_updated = {yr: {
            s: sect_shape_init[yr][s] +
            # Add energy data for current mseg, subtracting out any efficient-
            # case energy that remains with fossil fuel (not applicable to
            # sector shapes)
            ((add_energy[s][yr] * (1 - (fs_eff_splt_energy[0][yr] /
                                        fs_eff_splt_energy[1][yr]))
              # Pull in fuel split as needed
              if (fs_eff_splt_energy and s == "efficient" and
                  "electricity" not in cm
                  and fs_eff_splt_energy[1][yr] != 0)
              else add_energy[s][yr])
             # Only add energy data to sector shapes if data concerns
             # electricity mseg or fuel switching from fossil is occurring and
             # the data concern the efficient case (base would have had no
             # elec. to add)
             if ("electricity" in cm or (
                fuel_switch_to == "electricity" and s == "efficient"))
             else 0) for s in ["baseline", "efficient"]}
            for yr in self.handyvars.aeo_years_summary
        }

        return sect_shape_energy_updated

    def htcl_adj_rec(self, opts):
        """Record overlaps in heating/cooling eqp. and env. energy.

        Args:
            opts (object): Stores user-specified execution options.
        """

        # If there are both heating/cooling equipment and envelope measures in the package,
        # continue further to check for overlaps
        if not self.contributing_ECMs_eqp or not self.contributing_ECMs_env:
            return

        parsed_keys = set()
        # Loop through the heating/cooling equipment measures, check for
        # overlaps with envelope measures, and record the overlaps
        for m in self.contributing_ECMs_eqp:
            # Record unique data for each adoption scheme
            for adopt_scheme in self.handyvars.adopt_schemes:
                # Use shorthand for measure contributing microsegment data
                msegs_meas = copy.deepcopy(m.markets[adopt_scheme][
                    "mseg_adjust"]["contributing mseg keys and values"])

                # Loop through all contributing microsegment keys for the equipment measure that
                # apply to heating/cooling end uses and have not previously been parsed for
                # overlapping data
                hvac_keys = [key for key in msegs_meas.keys() if any(e in key for e in [
                    "heating", "cooling", "secondary heating"])]
                for cm_key in hvac_keys:
                    if cm_key in parsed_keys:
                        continue
                    # Record that the contributing microsegment key has been parsed
                    parsed_keys.add(cm_key)

                    # Translate the contributing microsegment key (a str) to list type
                    keys = literal_eval(str(cm_key))
                    # Extract region, building type/vintage, fuel type, & end use from the key list
                    cm_key_match = [str(x) for x in [keys[1], keys[2], keys[-1], keys[3], keys[4]]]

                    # Determine which, if any, envelope ECMs overlap with the region, building
                    # type/vintage, fuel type, and end use for the current contributing mseg for
                    # the equipment ECM
                    dmd_match_ECMs = [
                        x for x in self.contributing_ECMs_env if
                        any([all([k in z for k in cm_key_match]) for z in
                            x.markets[adopt_scheme]["mseg_adjust"][
                            "contributing mseg keys and values"].keys()])]

                    if not dmd_match_ECMs:
                        continue

                    # If an overlap is identified, record all necessary
                    # data for the current contributing microsegment
                    # across both the equipment and envelope side
                    # that are needed to remove the overlap subsequently
                    cm_keys_dmd = [[x for x in z.markets[adopt_scheme][
                        "mseg_adjust"][
                        "contributing mseg keys and values"].keys()
                        if all([k in x for k in cm_key_match])] for
                        z in dmd_match_ECMs]
                    # Record envelope energy savings across all
                    # envelope measures that overlap with current mseg
                    dmd_save = {yr: sum([sum([(
                        dmd_match_ECMs[m].markets[adopt_scheme][
                            "mseg_adjust"][
                            "contributing mseg keys and values"][
                            cm_keys_dmd[m][k]]["energy"][
                            "total"]["baseline"][yr] -
                        dmd_match_ECMs[m].markets[adopt_scheme][
                            "mseg_adjust"][
                            "contributing mseg keys and values"][
                            cm_keys_dmd[m][k]]["energy"][
                            "total"]["efficient"][yr]) for k in range(
                                len(cm_keys_dmd[m]))]) for
                        m in range(len(dmd_match_ECMs))]) for yr in
                        self.handyvars.aeo_years}

                    # Record envelope energy baselines across all
                    # envelope measures that overlap with current mseg
                    dmd_base = {yr: sum([sum([
                        dmd_match_ECMs[m].markets[adopt_scheme][
                            "mseg_adjust"][
                            "contributing mseg keys and values"][
                            cm_keys_dmd[m][k]]["energy"][
                            "total"]["baseline"][yr] for k in range(
                                len(cm_keys_dmd[m]))]) for
                        m in range(len(dmd_match_ECMs))]) for yr in
                        self.handyvars.aeo_years}

                    # If the user opts to include envelope costs in
                    # the total costs of the HVAC/envelope package,
                    # record those overlapping costs
                    if opts.pkg_env_costs is not False:
                        dmd_stk_cost = [[
                            dmd_match_ECMs[m].markets[adopt_scheme][
                                "mseg_adjust"][
                                "contributing mseg keys and values"][
                                cm_keys_dmd[m][k]]["cost"]["stock"] for
                            k in range(len(cm_keys_dmd[m]))] for
                            m in range(len(dmd_match_ECMs))]
                        dmd_stk = [[
                            dmd_match_ECMs[m].markets[adopt_scheme][
                                "mseg_adjust"][
                                "contributing mseg keys and values"][
                                cm_keys_dmd[m][k]]["stock"] for
                            k in range(len(cm_keys_dmd[m]))] for
                            m in range(len(dmd_match_ECMs))]
                    else:
                        dmd_stk_cost, dmd_stk = (
                            None for n in range(2))

                    # Translate key used to identify overlaps to str
                    cm_key_store = str(cm_key_match)
                    # Record the overlap data if it has not already
                    # been recorded for the current overlapping
                    # region, building type, building vintage,
                    # fuel, and end use
                    if cm_key_store not in self.htcl_overlaps[
                            adopt_scheme]["data"].keys():
                        # Data include the overlapping energy savings
                        # and baselines recorded for the equipment/
                        # envelope measures above, as well as the
                        # total energy use that could have overlapped,
                        # pulled from pre-calculated values
                        self.htcl_overlaps[adopt_scheme]["data"][
                                cm_key_store] = {
                            "affected savings": dmd_save,
                            "total affected": dmd_base,
                            "stock costs": dmd_stk_cost,
                            "stock": dmd_stk}

    def merge_direct_overlaps(
            self, msegs_meas, cm_key, adopt_scheme, mseg_out_break_adj,
            name_meas, fuel_switch_to, meas_typ, fs_eff_splt, key_list):
        """Adjust measure mseg data to address direct overlaps in package.

        Args:
            msegs_meas (dict): Data for the contributing microsegment of an
                individual measure that is being merged into the package.
            cm_key (tuple): Microsegment key describing the contributing
                microsegment currently being added (e.g. reg->bldg, etc.)
            adopt_scheme (string): Assumed consumer adoption scenario.
            mseg_out_break_adj (dict): Contributing measure baseline energy/
                carbon/cost baseline and savings data split by climate zone,
                building type, and end use.
            name_meas (string): Measure name.
            sect_shapes_meas (dict or NoneType): Sector-level baseline and
                efficient load shape information for the current contributing
                microsegment, if applicable (otherwise, None).
            fuel_switch_to (string): Indicator of which baseline fuel the
                measure switches to (if applicable).
            meas_typ (str): Used to flag add-on measures, which have special
                handling for merging in stock cost data.
            fs_eff_splt (dict): If applicable, the fuel splits for efficient-
                case measure energy/carb/cost (used to adj. output breakouts).
            key_list (list): List of keys with microsegment info. (primary/
                secondary, region, bldg, fuel, end use, tech type, vintage)

        Returns:
            Updated contributing microsegment and out break info. for the
            equip. measure that accounts for direct overlaps between the
            measure and other contributing equip. measures in the package
            given the current contributing microsegment information (e.g.,
            an exact key match).
        """

        # Pull keys from the current contributing microsegment info. that
        # can be used to match across heating/cooling equip/env measures
        htcl_key_match = str([str(x) for x in [
            key_list[1], key_list[2], key_list[-1], key_list[3],
            key_list[4]]])
        # If the keys are not found in the package attribute that stores data
        # needed to adjust across equip/env msegs, set key to empty string
        if htcl_key_match not in self.htcl_overlaps[
                adopt_scheme]["data"].keys():
            htcl_key_match = ""
        # Determine what other measures in the package, if any, share an
        # exact match of the current contributing microsegment key information
        # for the individual measure; exclude the measure itself from this list
        overlap_meas = [
            x for x in self.contributing_ECMs_eqp if cm_key in x.markets[
                adopt_scheme]["mseg_adjust"][
                "contributing mseg keys and values"].keys() and
            x.name != name_meas]
        # If an exact match with other measures is identified for the current
        # contributing microsegment, update the contributing microsegment data
        # to account for/remove direct overlaps with other measures
        if len(overlap_meas) != 0:
            # Make a copy of the mseg info. that is unaffected by subsequent
            # operations in the loop
            msegs_meas_init = copy.deepcopy(msegs_meas)
            # Find base and efficient adjustment fractions
            base_adj, eff_adj, eff_adj_c = self.find_base_eff_adj_fracs(
                msegs_meas_init, cm_key, adopt_scheme,
                name_meas, htcl_key_match, overlap_meas)
            # Adjust energy, carbon, and energy/carbon cost data based on
            # savings contribution of the measure and overlapping measure(s)
            # in this contributing microsegment, as well as the relative
            # performance of the overlapping measure(s)
            for k in ["energy", "carbon"]:
                # Make adjustments to energy/carbon/cost microsegments
                mseg_adj, mseg_cost_adj, tot_base_orig, tot_eff_orig, \
                    tot_save_orig, tot_base_orig_ecost, tot_eff_orig_ecost, \
                    tot_save_orig_ecost = self.make_base_eff_adjs(
                        k, cm_key, msegs_meas, base_adj, eff_adj, eff_adj_c)
                # Make adjustments to energy/carbon/cost output breakouts
                mseg_out_break_adj = self.find_adj_out_break_cats(
                    k, cm_key, mseg_adj, mseg_cost_adj, mseg_out_break_adj,
                    tot_base_orig, tot_eff_orig, tot_save_orig,
                    tot_base_orig_ecost, tot_eff_orig_ecost,
                    tot_save_orig_ecost, key_list, fuel_switch_to, fs_eff_splt)
            # Special handling for cost merge when add-on measure is packaged
            if meas_typ == "add-on":
                # Determine whether any of the measures overlapping with the
                # packaged measure are full service measures
                fullserv_overlp = any([
                    m.measure_type == "full service" for m in overlap_meas])
                # If the add-on measure overlaps with one or more full service
                # measures, subtract out the cost of the baseline stock, which
                # is embedded in the add-on measure's efficient cost estimate
                # but already covered by whatever full service equipment
                # measure(s) the add-on measure is being coupled with
                if fullserv_overlp is True:
                    # Remove baseline stock costs from efficient stock costs
                    msegs_meas["cost"]["stock"]["total"]["efficient"], \
                        msegs_meas["cost"]["stock"][
                        "competed"]["efficient"] = [{
                            yr: (msegs_meas["cost"]["stock"][s][
                                "efficient"][yr] -
                                 msegs_meas["cost"]["stock"][s][
                                 "baseline"][yr])
                            for yr in self.handyvars.aeo_years}
                            for s in ["total", "competed"]]
                    # Set baseline stock costs to zero
                    msegs_meas["cost"]["stock"]["total"]["baseline"], \
                        msegs_meas["cost"]["stock"]["competed"]["baseline"] = [
                            {yr: 0 for yr in self.handyvars.aeo_years}
                        for n in range(2)]

            # Adjust stock data to be consistent with the
            # energy-based baseline adjustment fractions calculated above
            self.adj_pkg_mseg_keyvals(
                msegs_meas["stock"], base_adj, base_adj, base_adj,
                base_eff_flag=None, comp_flag=None)

            # If necessary, adjust fugitive emissions data

            # Methane data adjusted consistent with energy data adjustment
            if self.usr_opts["fugitive_emissions"] is not False and \
                    self.usr_opts["fugitive_emissions"][0] in ['1', '3']:
                self.adj_pkg_mseg_keyvals(
                    msegs_meas["fugitive emissions"]["methane"],
                    base_adj, eff_adj, eff_adj_c, base_eff_flag=None,
                    comp_flag=None)
            # Refrigerants data adjusted consistent with stock data adjustment
            if self.usr_opts["fugitive_emissions"] is not False and \
                    self.usr_opts["fugitive_emissions"][0] in ['2', '3']:
                self.adj_pkg_mseg_keyvals(
                    msegs_meas["fugitive emissions"]["refrigerants"],
                    base_adj, base_adj, base_adj, base_eff_flag=None,
                    comp_flag=None)

            # Adjust measure sub-market scaling for contributing microsegment
            # such that when recombined into the package, the maximum
            # sub-market scaling fraction across overlapping measures in
            # the current microsegment is yielded

            # Case where all contributing measures have point values for
            # sub-market scaling fractions (not broken out by year)
            if not isinstance(msegs_meas["sub-market scaling"], dict) and all([
                not isinstance(x.markets[adopt_scheme]["mseg_adjust"][
                    "contributing mseg keys and values"][cm_key][
                        "sub-market scaling"], dict) for x in overlap_meas]):
                # Maximum of current measure's sub-market scaling fraction
                # and sub-market fractions of all other contrib. measures
                max_submkt_scale = max(
                    [msegs_meas["sub-market scaling"], max([
                        x.markets[adopt_scheme]["mseg_adjust"][
                            "contributing mseg keys and values"][cm_key][
                            "sub-market scaling"] for x in overlap_meas])]) / \
                    (len(overlap_meas) + 1)
            # Case where at least one contributing measure has sub-market
            # scaling fractions broken out by year
            else:
                # If current measure's sub-market scaling fraction is not
                # broken out by year, extend it across years
                if not isinstance(msegs_meas["sub-market scaling"], dict):
                    current_meas_sbmkt = {
                        yr: msegs_meas["sub-market scaling"] for yr in
                        self.handyvars.aeo_years}
                # Otherwise, set as is
                else:
                    current_meas_sbmkt = msegs_meas["sub-market scaling"]

                # Initialize list of dicts broken out by years for the sub-
                # market scaling fractions of overlapping measures
                overlap_submkt_dicts = []
                # Loop through all overlapping measures
                for x in overlap_meas:
                    # Shorthand for sub-market scaling information
                    overlap_submkt_init = x.markets[adopt_scheme][
                        "mseg_adjust"]["contributing mseg keys and values"][
                        cm_key]["sub-market scaling"]
                    # If overlapping measure's sub-market scaling fraction is
                    # not broken out by year, extend it across years
                    if not isinstance(overlap_submkt_init, dict):
                        overlap_submkt_dicts.append({
                            yr: overlap_submkt_init for yr in
                            self.handyvars.aeo_years})
                    # Otherwise, set as is
                    else:
                        overlap_submkt_dicts.append(overlap_submkt_init)

                # Find the maximum sub-market scaling fraction in each year
                # and set such that package sub-market scaling fraction equals
                # this fraction when contributing markets are recombined across
                # measures after adjustment
                max_submkt_scale = {
                    yr: None for yr in self.handyvars.aeo_years}
                for yr in max_submkt_scale.keys():
                    max_submkt_scale[yr] = max([current_meas_sbmkt[yr], max(
                        [x[yr] for x in overlap_submkt_dicts])]) / \
                        (len(overlap_meas) + 1)

            # Set final sub-market scaling fraction as maximum across all
            # contributing measures
            msegs_meas["sub-market scaling"] = max_submkt_scale

        return msegs_meas, mseg_out_break_adj

    def merge_htcl_overlaps(
            self, msegs_meas, cm_key, adopt_scheme,
            mseg_out_break_adj, name_meas, fuel_switch_to, fs_eff_splt,
            key_list, opts, common_stk):

        """Adjust measure mseg data to address equip/env overlaps in package.

        Args:
            msegs_meas (dict): Data for the contributing microsegment of an
                individual measure that is being merged into the package,
                post-adjustment to consider overlaps across equipment measures
                in the package.
            cm_key (tuple): Microsegment key describing the contributing
                microsegment currently being added (e.g. reg->bldg, etc.)
            adopt_scheme (string): Assumed consumer adoption scenario.
            mseg_out_break_adj (dict): Contributing measure baseline energy/
                carbon/cost baseline and savings data split by climate zone,
                building type, and end use.
            name_meas (string): Measure name.
            fuel_switch_to (string): Indicator of which baseline fuel the
                measure switches to (if applicable).
            fs_eff_splt (dict): If applicable, the fuel splits for efficient-
                case measure energy/carb/cost (used to adj. output breakouts).
            key_list (list): List of keys with microsegment info. (primary/
                secondary, region, bldg, fuel, end use, tech type, vintage).
            opts (object): Stores user-specified execution options.
            common_stk (dict or NoneType): Common, pre-adjusted measure-
                captured stock data for the current contributing microsegment.

        Returns:
            Updated contributing microsegment and out break info. for the
            equip. measure that accounts for indirect overlaps between the
            measure and other heating/cooling envelope measures in the package
            given the current contributing microsegment information (e.g., key
            match by region, bldg. type/vintage, fuel, and end use).
        """
        # Pull keys from the current contributing microsegment info. that
        # can be used to match across heating/cooling equip/env measures
        htcl_key_match = str([str(x) for x in [
            key_list[1], key_list[2], key_list[-1], key_list[3], key_list[4]]])
        # If the keys are not found in the package attribute that stores data
        # needed to adjust across equip/env msegs, stop operation
        if htcl_key_match in self.htcl_overlaps[
                adopt_scheme]["data"].keys():
            # Make a copy of the mseg info. that is unaffected by subsequent
            # operations in the loop
            msegs_meas_init = copy.deepcopy(msegs_meas)
            # Find base and efficient adjustment fractions; directly
            # overlapping measures are none in this case
            base_adj, eff_adj, eff_adj_c = self.find_base_eff_adj_fracs(
                    msegs_meas_init, cm_key, adopt_scheme, name_meas,
                    htcl_key_match, overlap_meas="")
            # Adjust energy, carbon, and energy/carbon cost data based on
            # savings contribution of the measure and overlapping measure(s)
            # in this contributing microsegment, as well as the relative
            # performance of the overlapping measure(s)
            for k in ["energy", "carbon"]:
                # Make adjustments to energy/carbon/cost microsegments
                mseg_adj, mseg_cost_adj, tot_base_orig, tot_eff_orig, \
                    tot_save_orig, tot_base_orig_ecost, tot_eff_orig_ecost, \
                    tot_save_orig_ecost = self.make_base_eff_adjs(
                        k, cm_key, msegs_meas, base_adj, eff_adj, eff_adj_c)
                # Make adjustments to energy/carbon/cost output breakouts
                mseg_out_break_adj = self.find_adj_out_break_cats(
                    k, cm_key, mseg_adj, mseg_cost_adj, mseg_out_break_adj,
                    tot_base_orig, tot_eff_orig, tot_save_orig,
                    tot_base_orig_ecost, tot_eff_orig_ecost,
                    tot_save_orig_ecost, key_list, fuel_switch_to, fs_eff_splt)

            # If necessary, adjust fugitive emissions data

            # Methane data adjusted consistent with energy data adjustment
            # No adjustment to refrigerant data required, as envelope
            # overlaps only affect energy use/methane
            if self.usr_opts["fugitive_emissions"] is not False and \
                    self.usr_opts["fugitive_emissions"][0] in ['1', '3']:
                self.adj_pkg_mseg_keyvals(
                    msegs_meas["fugitive emissions"]["methane"],
                    base_adj, eff_adj, eff_adj_c, base_eff_flag=None,
                    comp_flag=None)
            # If desired by the user, incorporate envelope stock costs,
            # provided that the total measure stock for the current equipment
            # microsegment anchor is not zero across the time horizon
            if opts.pkg_env_costs is not False and not all([
                    msegs_meas["stock"]["total"]["measure"][yr] == 0 for yr in
                    self.handyvars.aeo_years]):
                mseg_cost_adj = self.add_env_costs_to_pkg(
                    msegs_meas, adopt_scheme, htcl_key_match, key_list, cm_key,
                    common_stk)

        return msegs_meas, mseg_out_break_adj

    def add_env_costs_to_pkg(
            self, msegs_meas, adopt_scheme, htcl_key_match, key_list, cm_key,
            common_stk):
        """Reflect envelope stock costs in HVAC/envelope package data.

        Args:
            msegs_meas (dict): Data for the contributing microsegment of an
                individual measure that is being merged into the package,
                post-adjustment to consider overlaps across equipment measures
                in the package.
            adopt_scheme (string): Assumed consumer adoption scenario.
            htcl_key_match (string): Matching mseg keys to use in pulling
                overlapping HVAC equipment and envelope data.
            key_list (string): Mseg key list.
            cm_key (tuple): Microsegment key describing the contributing
                microsegment currently being added (e.g. reg->bldg, etc.)
            common_stk (dict or NoneType): Common, pre-adjusted measure-
                captured stock data for the current contributing microsegment.

        Returns:
            Updated contributing microsegment info. for the equip. measure
            that accounts for envelope stock costs.
        """

        # Shorthand for overlapping envelope stock cost data
        overlp_data_stock_cost = self.htcl_overlaps[adopt_scheme]["data"][
            htcl_key_match]["stock costs"]
        # Shorthand for overlapping envelope stock data
        overlp_data_stock = self.htcl_overlaps[adopt_scheme]["data"][
            htcl_key_match]["stock"]

        # Loop through env. measures that overlap with current equip. measure
        for olm in range(len(overlp_data_stock_cost)):
            # Shorthands for overlapping stock cost/stock data for the
            # currently looped through envelope measure
            olm_sc, olm_s = [
                overlp_data_stock_cost[olm], overlp_data_stock[olm]]
            # Initialize shorthands for total and competed stock cost/stock
            # data by year for the currently looped through envelope measure
            tot_stk_eff, tot_stk_cost_eff, tot_stk_base, tot_stk_cost_base, \
                comp_stk_eff, comp_stk_cost_eff, comp_stk_base, \
                comp_stk_cost_base, = ({
                    yr: 0 for yr in self.handyvars.aeo_years}
                    for n in range(8))
            # Loop through all contributing microsegments for the overlapping
            # envelope measure and add to initialized stock cost/stock data
            for c_mseg in range(len(olm_sc)):
                tot_stk_eff, tot_stk_cost_eff, tot_stk_base, \
                    tot_stk_cost_base, comp_stk_eff, comp_stk_cost_eff, \
                    comp_stk_base, comp_stk_cost_base = [{
                        yr: (x[yr] + y[yr])
                        for yr in self.handyvars.aeo_years}
                        for x, y in zip([
                            tot_stk_eff, tot_stk_cost_eff,
                            tot_stk_base, tot_stk_cost_base,
                            comp_stk_eff, comp_stk_cost_eff,
                            comp_stk_base, comp_stk_cost_base], [
                            olm_s[c_mseg]["total"]["measure"],
                            olm_sc[c_mseg]["total"]["efficient"],
                            olm_s[c_mseg]["total"]["all"],
                            olm_sc[c_mseg]["total"]["baseline"],
                            olm_s[c_mseg]["competed"]["measure"],
                            olm_sc[c_mseg]["competed"]["efficient"],
                            olm_s[c_mseg]["competed"]["all"],
                            olm_sc[c_mseg]["competed"]["baseline"]])]

            # Determine whether current mseg pertains to residential or
            # commercial buildings
            if any([x in htcl_key_match for x in [
                    "single family home", "mobile home",
                    "multi family home"]]):
                bldg_sect = "residential"
            else:
                bldg_sect = "commercial"
            # For commercial microsegments, envelope stock will be in units
            # of ft^2 floor; pull factor to convert these costs to the
            # units of kBtu/h heating or cooling service demand that the HVAC
            # equipment measures will be using
            if bldg_sect == "commercial":
                # Initial conversion from ft^2 floor to service capacity units.
                if "heating" in htcl_key_match:
                    stk_cnv_1 = (
                        self.pkg_env_cost_convert_data[
                            "cost unit conversions"]["heating and cooling"][
                            "supply"]["heating equipment"][
                            "conversion factor"]["value"])

                elif "cooling" in htcl_key_match:
                    stk_cnv_1 = (
                        self.pkg_env_cost_convert_data[
                            "cost unit conversions"]["heating and cooling"][
                            "supply"]["cooling equipment"][
                            "conversion factor"]["value"])
                else:
                    raise ValueError(
                        "Non-heating or cooling end use encountered when "
                        "translating envelope measure cost data unit to "
                        "match those of packaged HVAC equipment measures")
                # Secondary conversion from hourly service capacity to annual
                # service demand units
                try:
                    # Set appropriate capacity factor (TBtu delivered service
                    # for hours of actual operation / TBtu service running at
                    # full capacity for all hours of the year), keyed in
                    # by building type and end use service
                    cap_fact_mseg = self.handyvars.cap_facts["data"][key_list[2]][key_list[4]]
                    # Conversion: (1) divides by 1e9 to get from capacity
                    # units of kBtu to service demand units of TBtu (2)
                    # multiplies by 8760 to translate units from per hour full
                    # capacity to per year full capacity (3) multiplies
                    # by capacity factor (service delivered per year /
                    # service per year @ full capacity) to get to final units
                    # of heating or cooling service delivered per year
                    stk_cnv_2 = (1 / 1e9) * (8760) * cap_fact_mseg
                    # Adjust conversion to account for capacity factor
                    convert_env_to_hvac_stk_units = stk_cnv_1 * stk_cnv_2
                except (KeyError):
                    raise KeyError(
                        "Microsegment '" + str(htcl_key_match) + "' "
                        "requires capacity factor data that are missing")

            # For residential microsegments, envelope stock will already
            # have been converted to # homes, which is considered 1:1 with
            # the HVAC equipment measure units (assume one HVAC unit per home)
            else:
                convert_env_to_hvac_stk_units = 1

            # Convert stock totals for envelope measures to same basis
            # as stock of HVAC equip. measures, using conversion from above
            tot_stk_eff, comp_stk_eff = [
                {yr: s_e[yr] * convert_env_to_hvac_stk_units for
                 yr in self.handyvars.aeo_years} for s_e in [
                    tot_stk_eff, comp_stk_eff]]
            # Pull equipment stock totals to use in normalizing costs; note
            # that these values are pre-adjustment for overlaps across multiple
            # equipment measures in the package (e.g., they are the total
            # measure-captured stock for the current overlapping
            # microsegment, to ensure this calculation is consistent across
            # all overlapping equipment measures)
            tot_stk_eff_hvac_unadj, comp_stk_eff_hvac_unadj = [{
                yr: s_u[yr] for yr in self.handyvars.aeo_years} for s_u in [
                    common_stk[adopt_scheme][cm_key]["total"],
                    common_stk[adopt_scheme][cm_key]["competed"]]]
            # Develop factors to map number of envelope measure stock
            # units to number of HVAC measure stock units; note that when
            # the captured stock for the envelope measure is greater than
            # that of the HVAC measure, assume envelope stock (# units of
            # equipment it applies to, for residential, or units demand served,
            # for commercial) equals the HVAC equipment stock it is being
            # merged with
            tot_env_to_hvac_stk = {
                yr: (tot_stk_eff[yr] / tot_stk_eff_hvac_unadj[yr]) if (
                    tot_stk_eff[yr] < tot_stk_eff_hvac_unadj[yr]) else (
                        1 if tot_stk_eff_hvac_unadj[yr] != 0 else 0)
                for yr in self.handyvars.aeo_years}
            comp_env_to_hvac_stk = {
                yr: (comp_stk_eff[yr] / comp_stk_eff_hvac_unadj[yr]) if (
                    comp_stk_eff[yr] < comp_stk_eff_hvac_unadj[yr]) else (
                        1 if comp_stk_eff_hvac_unadj[yr] != 0 else 0)
                for yr in self.handyvars.aeo_years}

            # Set efficient and baseline envelope costs, normalized by the
            # stock (converted above to same units as HVAC equipment), to add
            # to the unit costs of the HVAC equipment measure; note that in
            # both cases, aggregate costs are anchored on the measure-captured
            # stock numbers
            env_cost_eff_tot_unit = {
                yr: ((tot_stk_cost_eff[yr] / tot_stk_eff[yr]) *
                     tot_env_to_hvac_stk[yr]) if tot_stk_eff[yr] != 0
                else 0 for yr in self.handyvars.aeo_years}
            env_cost_base_tot_unit = {
                yr: ((tot_stk_cost_base[yr] / tot_stk_eff[yr]) *
                     tot_env_to_hvac_stk[yr]) if tot_stk_eff[yr] != 0
                else 0 for yr in self.handyvars.aeo_years}
            env_cost_eff_comp_unit = {
                yr: ((comp_stk_cost_eff[yr] / comp_stk_eff[yr]) *
                     comp_env_to_hvac_stk[yr]) if comp_stk_eff[yr] != 0
                else 0 for yr in self.handyvars.aeo_years}
            env_cost_base_comp_unit = {
                yr: ((comp_stk_cost_base[yr] / comp_stk_eff[yr]) *
                     comp_env_to_hvac_stk[yr]) if comp_stk_eff[yr] != 0
                else 0 for yr in self.handyvars.aeo_years}

            # Adjust total efficient stock cost to account for envelope
            msegs_meas["cost"]["stock"]["total"]["efficient"] = {
                yr: msegs_meas["cost"]["stock"]["total"][
                     "efficient"][yr] + env_cost_eff_tot_unit[yr] *
                msegs_meas["stock"]["total"]["measure"][yr] for
                yr in self.handyvars.aeo_years}
            # Adjust total baseline stock cost to account for envelope
            msegs_meas["cost"]["stock"]["total"]["baseline"] = {
                yr: msegs_meas["cost"]["stock"]["total"][
                     "baseline"][yr] + env_cost_base_tot_unit[yr] *
                msegs_meas["stock"]["total"]["measure"][yr] for
                yr in self.handyvars.aeo_years}
            # Adjust competed efficient stock cost to account for envelope
            msegs_meas["cost"]["stock"]["competed"]["efficient"] = {
                yr: msegs_meas["cost"]["stock"]["competed"][
                    "efficient"][yr] + env_cost_eff_comp_unit[yr] *
                msegs_meas["stock"]["competed"]["measure"][yr] for
                yr in self.handyvars.aeo_years}
            # Adjust competed baseline stock cost to account for envelope
            msegs_meas["cost"]["stock"]["competed"]["baseline"] = {
                yr: msegs_meas["cost"]["stock"]["competed"][
                     "baseline"][yr] + env_cost_base_comp_unit[yr] *
                msegs_meas["stock"]["competed"]["measure"][yr] for
                yr in self.handyvars.aeo_years}

        return msegs_meas

    def find_base_eff_adj_fracs(self, msegs_meas, cm_key, adopt_scheme,
                                name_meas, htcl_key_match, overlap_meas):
        """Calculate overlap adjustments for measure mseg in a package.

        Args:
            msegs_meas (dict): Data for the contributing microsegment of an
                individual measure that is being merged into the package.
            cm_key (tuple): Microsegment key describing the contributing
                microsegment currently being added (e.g. reg->bldg, etc.)
            adopt_scheme (string): Assumed consumer adoption scenario.
            name_meas (string): Measure name.
            htcl_key_match (string): Matching mseg keys to use in pulling
                overlapping HVAC equipment and envelope data.
            overlap_meas (list or str): List of measure objs that also apply
                to the current contributing microsegment; will be an empty
                string when heating/cooling equipment and envelope overlaps
                are being addressed.

        Returns:
            Fractions for adjusting baseline and efficient energy, carbon,
            and cost data for the current contributing microsegment to
            account for overlaps with other measures.
        """

        # Develop baseline/efficient data adjustment fractions for two
        # overlap cases:

        # Case 1: overlap between heating/cooling equipment and
        # envelope microsegments with required adjustment data available;
        # here, microsegments will not overlap exactly, but will be linked by
        # region, building type/vintage, fuel, and end use, and note that the
        # resultant adjustment fractions will be applied to the equipment
        # measure's energy, energy cost and carbon data only
        if not overlap_meas and htcl_key_match:
            # Initialize estimates of the portion of total potential
            # overlapping energy use in the current microsegment that is
            # affected by measure(s) of the overlapping tech type; the savings
            # of these overlapping measure(s); the portion of total overlapping
            # savings that the current measure contributes; and the relative
            # performance of the overlapping measure(s)
            save_overlp_htcl, rp_overlp_htcl = ({
                yr: 1 for yr in self.handyvars.aeo_years} for n in range(2))
            # Set shorthand for total savings and affected energy use
            # for the overlapping tech type in the current contributing mseg
            overlp_data = self.htcl_overlaps[adopt_scheme]["data"][
                htcl_key_match]

            # Loop through all years in the modeling time horizon and use
            # data above to develop baseline/efficient adjustment factors
            for yr in self.handyvars.aeo_years:
                # Ignore numpy divide by zero errors and handle NaNs
                with numpy.errstate(all='ignore'):
                    # Find relative savings fraction for the overlapping
                    # measure(s); handle zero denominator
                    try:
                        save_overlp_htcl[yr] = (
                            overlp_data["affected savings"][yr] /
                            overlp_data["total affected"][yr])
                    except ZeroDivisionError:
                        save_overlp_htcl[yr] = 0
                    # Handle numpy NaNs
                    if not numpy.isfinite(save_overlp_htcl[yr]):
                        save_overlp_htcl[yr] = 0
                    # Set relative performance for the overlapping measure(s)
                    rp_overlp_htcl[yr] = 1 - save_overlp_htcl[yr]

                    # Ensure that envelope savings are never negative and that
                    # they never exceed 100%
                    if overlp_data["affected savings"][yr] < 0 or \
                            rp_overlp_htcl[yr] > 1:
                        rp_overlp_htcl[yr] = 1
                    elif overlp_data["affected savings"][yr] > 0 and \
                            rp_overlp_htcl[yr] < 0:
                        rp_overlp_htcl[yr] = 0

            # Overlapping envelope measures only affect the efficient case
            # energy/carbon results for an HVAC equipment measure; set base
            # adjustment to 1
            base_adj = {yr: 1 for yr in self.handyvars.aeo_years}
            # The relative performance of the overlapping envelope measures
            # will be used to adjust down efficient case energy/carbon for the
            # HVAC equipment measures
            eff_adj, eff_adj_comp = ({
                yr: rp_overlp_htcl[yr] for yr in self.handyvars.aeo_years}
                for n in range(2))
        # Case 2: direct overlap between two equipment measures (same mseg)
        elif overlap_meas:
            # Initialize an additional adjustment to account for sub-market
            # scaling fractions across overlapping measure(s)
            sbmkt_wt_meas = {yr: 0 for yr in self.handyvars.aeo_years}
            # Establish common baseline microsegment values for calculations
            # below; baseline microsegments will not be identical in cases
            # where one or more of the overlapping measures has a sub-market
            # scaling fraction, and the inapplicable portion of the common
            # baseline must be accounted for across measures
            common_base, common_base_comp = ({} for n in range(2))
            sbmkts_all = {}
            for yr in self.handyvars.aeo_years:
                # Record the baseline microsegments for all
                # measures in the given year (e.g., the current measure being
                # adjusted and those that overlap for the given baseline
                # microsegment), and store in a list
                # Total
                base_mkts_all = [
                    msegs_meas["energy"]["total"]["baseline"][yr]]
                base_mkts_all.extend([
                    x.markets[adopt_scheme]["mseg_adjust"][
                        "contributing mseg keys and values"][cm_key][
                        "energy"]["total"]["baseline"][yr] for
                    x in overlap_meas])
                # Competed
                base_mkts_all_comp = [
                    msegs_meas["energy"]["competed"]["baseline"][yr]]
                base_mkts_all_comp.extend([
                    x.markets[adopt_scheme]["mseg_adjust"][
                        "contributing mseg keys and values"][cm_key][
                        "energy"]["competed"]["baseline"][yr] for
                    x in overlap_meas])
                # Common baseline is set to the max baseline mseg value across
                # measures (since differences are driven by down-scaling)
                # Total
                common_base[yr] = max(base_mkts_all)
                # Competed
                common_base_comp[yr] = max(base_mkts_all_comp)
                # Record the difference between the baseline mseg value of each
                # measure and the common baseline value
                sbmkts_all[yr] = [common_base[yr] - x for x in base_mkts_all]
            # Find total savings of the measure being adjusted
            save_meas = {yr: (
                msegs_meas["energy"]["total"]["baseline"][yr] -
                msegs_meas["energy"]["total"]["efficient"][yr]) for
                yr in self.handyvars.aeo_years}
            # Find total savings of overlapping measure(s); store in list
            save_overlp = {yr: [(
                overlap_meas[x].markets[adopt_scheme]["mseg_adjust"][
                    "contributing mseg keys and values"][cm_key][
                    "energy"]["total"]["baseline"][yr] -
                overlap_meas[x].markets[adopt_scheme]["mseg_adjust"][
                    "contributing mseg keys and values"][cm_key][
                    "energy"]["total"]["efficient"][yr]) for x in range(
                    len(overlap_meas))] for yr in
                self.handyvars.aeo_years}
            # Find competed savings of overlapping measure(s); store in list
            save_comp_overlp = {yr: [(
                overlap_meas[x].markets[adopt_scheme]["mseg_adjust"][
                    "contributing mseg keys and values"][cm_key][
                    "energy"]["competed"]["baseline"][yr] -
                overlap_meas[x].markets[adopt_scheme]["mseg_adjust"][
                    "contributing mseg keys and values"][cm_key][
                    "energy"]["competed"]["efficient"][yr]) for x in range(
                    len(overlap_meas))] for yr in
                self.handyvars.aeo_years}
            # Establish the ratio of the measure's savings in the
            # current microsegment to those of the overlapping measure(s)
            save_wt_meas = {yr: (
                (abs(save_meas[yr]) / (
                    abs(save_meas[yr]) + abs(sum(save_overlp[yr])))) if (
                    abs(save_meas[yr]) + abs(sum(save_overlp[yr]))) != 0
                else 0) for yr in self.handyvars.aeo_years}
            # Establish the ratio of each of the overlapping measure's savings
            # to total savings across the overlapping measures (e.g., excluding
            # the current measure being adjusted)
            save_wt_overlp = {
                yr: [save_overlp[yr][x] / sum(save_overlp[yr]) if
                     sum(save_overlp[yr]) != 0 else 0 for x in range(
                     len(save_overlp[yr]))] for yr in self.handyvars.aeo_years}
            # If applicable, update the sub-market adjustment for each year
            # in the analysis time horizon
            for yr in self.handyvars.aeo_years:
                # Only update sub-market adjustment in cases where at least
                # one of the measures being compared has a sub-market scaling
                # fraction and it is not the current measure being adjusted
                if sum(sbmkts_all[yr]) != 0 and sbmkts_all[yr][0] == 0:
                    # Determine the total savings for all overlapping measures
                    # that do not have sub-market scaling fractions for the
                    # current microsegment
                    save_overlp_sbmkt = {yr: [
                        save_overlp[yr][(x - 1)] if (sbmkts_all[yr][x] == 0)
                        else 0 for x in range(1, len(sbmkts_all[yr]))]
                        for yr in self.handyvars.aeo_years}
                    # Use the savings of the current measure being adjusted
                    # to determine its share of the common baseline that is
                    # inapplicable due to sub-market scaling across overlapping
                    # measures; if there are no savings to compare, assign
                    # evenly across all measures without sub-market scaling;
                    # handle zero denominator

                    # Ignore numpy divide by zero errors and handle NaNs
                    with numpy.errstate(all='ignore'):
                        try:
                            sbmkt_save_wt_meas = (abs(save_meas[yr]) / (
                                abs(save_meas[yr]) +
                                abs(sum(save_overlp_sbmkt[yr]))))
                        except ZeroDivisionError:
                            sbmkt_save_wt_meas = (
                                1 / len([x for x in sbmkts_all[yr] if x == 0]))
                        # Handle numpy NaNs
                        if not numpy.isfinite(sbmkt_save_wt_meas):
                            sbmkt_save_wt_meas = (
                                1 / len([x for x in sbmkts_all[yr] if x == 0]))

                    # Determine the total fraction of the common baseline
                    # that overlapping measures do not apply to due to
                    # sub-market scaling fractions
                    # Record the savings-based fractions of the current
                    # baseline microsegment for each measure being considered
                    save_all_yr = [save_meas[yr]]
                    save_all_yr.extend(save_overlp[yr])
                    save_all_wts = {
                        yr: [(x / sum(save_all_yr)) if sum(save_all_yr) != 0
                             else (1 / (len(overlap_meas) + 1))
                             for x in save_all_yr]}
                    # Multiply the fractions calculated above by each
                    # measure's inapplicable portion of the common baseline,
                    # and normalize by the common baseline
                    sbmkt_base_frac = sum(
                        [sbmkts_all[yr][x] * save_all_wts[yr][x] for
                         x in range(len(sbmkts_all[yr]))]) / common_base[yr]

                    # Calculate additional fraction of the common baseline to
                    # assign current measure to account for sub-market scaling
                    sbmkt_wt_meas[yr] = sbmkt_save_wt_meas * sbmkt_base_frac
            # Final baseline adjustment factor is determined by the measure's
            # fractional contribution to total overlapping savings, plus
            # any adjustment to account for sub-market scaling across
            # overlapping measures; if the savings fraction is zero, assume
            # even weight across overlapping measures after confirming the
            # overlapping measure savings are all zero, or otherwise set to
            # the sub-market adjustment fraction
            base_adj = {
                yr: (save_wt_meas[yr] + sbmkt_wt_meas[yr]) if
                save_wt_meas[yr] != 0 else ((
                    1 / (len(overlap_meas) + 1) + sbmkt_wt_meas[yr])
                    if sum(save_overlp[yr]) == 0 else sbmkt_wt_meas[yr]) for
                yr in self.handyvars.aeo_years}

            # Determine relative performance of the current measure in the
            # cumulatively competed-captured portion of the stock for each year
            rp_comp_meas = {yr: (1 - (
                # Measure savings (cumulative)
                save_meas[yr] / (
                    # Find ratio of measure savings to cumulatively competed
                    # energy use in the given year
                    # Cumulatively competed-captured fraction for measure
                    (msegs_meas["stock"]["total"]["measure"][yr] /
                     msegs_meas["stock"]["total"]["all"][yr]) *
                    # Total baseline energy for measure
                    msegs_meas["energy"]["total"]["baseline"][yr])))
                if (msegs_meas["stock"]["total"]["measure"][yr] != 0 and
                    msegs_meas["stock"]["total"]["all"][yr] != 0 and
                    msegs_meas["energy"]["total"]["baseline"][yr] != 0) else 1
                for yr in self.handyvars.aeo_years}

            # Calculate overall relative performance of overlapping measures (
            # excluding the current measure being adjusted) in the total and
            # competed efficient stock

            # Total efficient relative performance adjustment; note that this
            # factor is ultimately applied to the measure's efficient data
            # after already considering adjustment by the baseline factor
            # above; thus, it is calculated relative to the efficient data
            # post-adjustment by that factor
            rp_overlp = {yr: (1 - (
                # Total weighted savings from overlapping measures (cumulative)
                # calculated relative to current measure's baseline
                (((sum([x * y for x, y in zip(
                    save_wt_overlp[yr], save_overlp[yr])]) / common_base[yr]) *
                  msegs_meas["energy"]["total"]["baseline"][yr]) *
                 # Scale overlapping savings to reflect baseline adjustment
                 # for the current measure, and again to reflect the
                 # relative performance of the current measure in the competed
                 # stock
                 base_adj[yr] * rp_comp_meas[yr]) /
                # Find ratio of adjusted overlapping savings to the
                # current measure's efficient data post-baseline adjustment
                (msegs_meas["energy"]["total"]["efficient"][yr] *
                 base_adj[yr])))
                if (msegs_meas["energy"]["total"]["efficient"][yr] *
                    base_adj[yr] != 0) else 1
                for yr in self.handyvars.aeo_years}
            # Competed efficient relative performance adjustment; this will
            # adjust the current measure's competed-efficient results by the
            # relative performance of the overlapping measure(s)
            rp_overlp_comp = {yr: (1 - (
                # Total savings from overlapping measures (cumulative)
                sum([x * y for x, y in zip(
                    save_wt_overlp[yr], save_comp_overlp[yr])]) /
                common_base_comp[yr]))
                if common_base_comp[yr] != 0 else 1
                for yr in self.handyvars.aeo_years}

            # Final efficient adjustment factor adds multiplication of the
            # relative performance of the overlapping measure(s) to the
            # initial adjustment calculation
            # Implement total efficient adjustment
            eff_adj = {yr: base_adj[yr] * rp_overlp[yr]
                       for yr in self.handyvars.aeo_years}
            # Implement competed efficient adjustment
            eff_adj_comp = {yr: base_adj[yr] * rp_overlp_comp[yr]
                            for yr in self.handyvars.aeo_years}
        # If neither case 1 or 2 above, set baseline/efficient adjustments to 1
        else:
            base_adj, eff_adj, eff_adj_comp = (
              {yr: 1 for yr in self.handyvars.aeo_years} for n in range(2))

        return base_adj, eff_adj, eff_adj_comp

    def make_base_eff_adjs(
            self, k, cm_key, msegs_meas, base_adj, eff_adj, eff_adj_c):
        """Apply overlap adjustments for measure mseg in a package.

        Args:
            k (str): Data type indicator ("energy" or "carbon")
            cm_key (tuple): Microsegment key describing the contributing
                microsegment currently being added (e.g. reg->bldg, etc.)
            msegs_meas (dict): Data for the contributing microsegment of an
                individual measure that is being merged into the package.
            base_adj (dict): Overlap adjustments for baseline data.
            eff_adj (dict): Overlap adjustments for total efficient data.
            eff_adj_c (dict): Overlap adjustments for competed efficient data.

        Returns:
            Adjusted baseline/efficient energy and carbon data that accounts
            account for overlaps between a given measure and other measures
            in a package.
        """

        # Initialize variables used to track pre-adjusted mseg data
        tot_base_orig, tot_eff_orig, tot_save_orig, tot_base_orig_ecost, \
            tot_eff_orig_ecost, tot_save_orig_ecost = ('' for n in range(6))
        # Create shorthand for stock/energy/carbon/lifetime msegs data
        mseg_adj = msegs_meas[k]
        # Create shorthand for cost data
        mseg_cost_adj = msegs_meas["cost"][k]
        # Total baseline energy/carbon
        tot_base_orig = copy.deepcopy(mseg_adj["total"]["baseline"])
        # Total efficient energy/carbon
        tot_eff_orig = copy.deepcopy(mseg_adj["total"]["efficient"])
        # Total energy/carbon savings
        tot_save_orig = {yr: (
            copy.deepcopy(mseg_adj["total"]["baseline"][yr]) -
            copy.deepcopy(mseg_adj["total"]["efficient"][yr]))
            for yr in self.handyvars.aeo_years}
        # Record total energy cost data before adjustment
        if k == "energy" and mseg_cost_adj:
            # Total baseline energy cost
            tot_base_orig_ecost = copy.deepcopy(
                mseg_cost_adj["total"]["baseline"])
            # Total efficient energy cost
            tot_eff_orig_ecost = copy.deepcopy(
                mseg_cost_adj["total"]["efficient"])
            # Total energy cost savings
            tot_save_orig_ecost = {yr: (
                copy.deepcopy(mseg_cost_adj["total"]["baseline"][yr]) -
                copy.deepcopy(mseg_cost_adj["total"]["efficient"][yr]))
                for yr in self.handyvars.aeo_years}
        # Adjust msegs using base/efficient adjustment fractions
        self.adj_pkg_mseg_keyvals(
            mseg_adj, base_adj, eff_adj, eff_adj_c, base_eff_flag=None,
            comp_flag=None)
        # If applicable, adjust energy/carbon cost msegs
        if mseg_cost_adj:
            self.adj_pkg_mseg_keyvals(
                mseg_cost_adj, base_adj, eff_adj, eff_adj_c,
                base_eff_flag=None, comp_flag=None)

        return mseg_adj, mseg_cost_adj, tot_base_orig, tot_eff_orig, \
            tot_save_orig, tot_base_orig_ecost, tot_eff_orig_ecost, \
            tot_save_orig_ecost

    def find_adj_out_break_cats(
            self, k, cm_key, msegs_ecarb, msegs_ecarb_cost, mseg_out_break_adj,
            tot_base_orig, tot_eff_orig, tot_save_orig, tot_base_orig_ecost,
            tot_eff_orig_ecost, tot_save_orig_ecost, key_list, fuel_switch_to,
            fs_eff_splt):
        """Adjust output breakouts after removing energy/carbon data overlaps.

        Args:
            k (str): Data type indicator ("energy" or "carbon")
            cm_key (tuple): Microsegment key describing the contributing
                microsegment currently being added (e.g. reg->bldg, etc.)
            msegs_ecarb (dict): Shorthand for energy/carbon data.
            msegs_ecarb_cost (dict): Shorthand for energy/carbon cost data.
            mseg_out_break_adj (dict): Initial output breakout data.
            tot_base_orig (dict): Unadjusted baseline energy/carbon data.
            tot_eff_orig (dict): Unadjusted efficient energy/carbon data.
            tot_save_orig (dict): Unadjusted energy/carbon savings data.
            tot_base_orig_ecost (dict): Unadjusted base energy cost data.
            tot_eff_orig_ecost (dict): Unadjusted efficient energy cost data.
            tot_save_orig_ecost (dict): Unadjusted energy cost savings data.
            key_list (list): List of microsegment keys.
            fuel_switch_to (string): Indicator of which baseline fuel the
                measure switches to (if applicable).
            fs_eff_splt (dict): If applicable, the fuel splits for efficient-
                case measure energy/carb/cost (used to adj. output breakouts).

        Returns:
            Updated energy, carbon, and energy cost output breakouts adjusted
            to account for removal of overlaps between measure and other
            measures in a package.
        """

        # Establish applicable climate zone breakout
        for cz in self.handyvars.out_break_czones.items():
            if key_list[1] in cz[1]:
                out_cz = cz[0]
        # Establish applicable building type breakout
        for bldg in self.handyvars.out_break_bldgtypes.items():
            if all([x in bldg[1] for x in [
                    key_list[2], key_list[-1]]]):
                out_bldg = bldg[0]
        # Establish applicable end use breakout
        for eu in self.handyvars.out_break_enduses.items():
            # * Note: The 'other' microsegment end
            # use may map to either the 'Refrigeration' output
            # breakout or the 'Other' output breakout, depending on
            # the technology type specified in the measure
            # definition. Also note that 'supply' side
            # heating/cooling microsegments map to the
            # 'Heating (Equip.)'/'Cooling (Equip.)' end uses, while
            # 'demand' side heating/cooling microsegments map to
            # the 'Heating (Env.)'/'Cooling (Env.) end use, with the
            # exception of 'demand' side heating/cooling microsegments that
            # represent waste heat from lights - these are
            # categorized as part of the 'Lighting' end use
            if key_list[4] == "other":
                if key_list[5] == "freezers":
                    out_eu = "Refrigeration"
                else:
                    out_eu = "Other"
            elif key_list[4] in eu[1]:
                if (eu[0] in ["Heating (Equip.)", "Cooling (Equip.)"] and
                    key_list[5] == "supply") or (
                    eu[0] in ["Heating (Env.)", "Cooling (Env.)"] and
                    key_list[5] == "demand" and key_list[0] == "primary") or (
                    eu[0] not in ["Heating (Equip.)", "Cooling (Equip.)",
                                  "Heating (Env.)", "Cooling (Env.)"]):
                    out_eu = eu[0]
            elif "lighting gain" in key_list:
                out_eu = "Lighting"
        # If applicable, establish fuel type breakout (electric vs.
        # non-electric); note – only applicable to end uses that
        # are at least in part fossil-fired
        if len(self.handyvars.out_break_fuels.keys()) != 0 and (
                out_eu in self.handyvars.out_break_eus_w_fsplits):
            # Flag for detailed fuel type breakout
            detail = len(self.handyvars.out_break_fuels.keys()) > 2
            # Establish breakout of fuel type that is being
            # reduced (e.g., through efficiency or fuel switching
            # away from the fuel)
            for f in self.handyvars.out_break_fuels.items():
                if key_list[3] in f[1]:
                    # Special handling for other fuel tech.,
                    # under detailed fuel type breakouts; this
                    # tech. may fit into multiple fuel categories
                    if detail and key_list[3] == "other fuel":
                        # Assign coal/kerosene tech.
                        if f[0] == "Distillate/Other" and (
                            key_list[-2] is not None and any([
                                x in key_list[-2] for x in [
                                "coal", "kerosene"]])):
                            out_fuel_save = f[0]
                        # Assign wood tech.
                        elif f[0] == "Biomass" and (
                            key_list[-2] is not None and "wood" in
                                key_list[-2]):
                            out_fuel_save = f[0]
                        # Assign all other tech. to propane
                        elif f[0] == "Propane":
                            out_fuel_save = f[0]
                    else:
                        out_fuel_save = f[0]
            # Establish breakout of fuel type that is being added
            # to via fuel switching, if applicable
            if fuel_switch_to == "electricity" and out_fuel_save != "Electric":
                out_fuel_gain = "Electric"
            elif fuel_switch_to not in [None, "electricity"] and \
                    out_fuel_save == "Electric":
                # Check for detailed fuel types
                if detail:
                    for f in self.handyvars.out_break_fuels.items():
                        # Special handling for other fuel tech.,
                        # under detailed fuel type breakouts; this
                        # tech. may fit into multiple fuel cats.
                        if self.fuel_switch_to in f[1] and \
                                key_list[3] == "other fuel":
                            # Assign coal/kerosene tech.
                            if f[0] == "Distillate/Other" and (
                                key_list[-2] is not None and any([
                                    x in key_list[-2] for x in [
                                    "coal", "kerosene"]])):
                                out_fuel_gain = f[0]
                            # Assign wood tech.
                            elif f[0] == "Biomass" and (
                                key_list[-2] is not None and "wood" in
                                    key_list[-2]):
                                out_fuel_gain = f[0]
                            # Assign all other tech. to propane
                            elif f[0] == "Propane":
                                out_fuel_gain = f[0]
                        elif self.fuel_switch_to in f[1]:
                            out_fuel_gain = f[0]
                else:
                    out_fuel_gain = "Non-Electric"
            else:
                out_fuel_gain = ""
        else:
            out_fuel_save, out_fuel_gain = ("" for n in range(2))

        # Shorthands for data used to adjust original output breakouts
        base_orig, eff_orig, save_orig, base_adj, eff_adj = [
            tot_base_orig, tot_eff_orig, tot_save_orig, msegs_ecarb[
                "total"]["baseline"], msegs_ecarb["total"]["efficient"]]
        # If necessary, shorthands for data used to adjust original cost output
        if all([x for x in [tot_base_orig_ecost, tot_eff_orig_ecost,
                            tot_save_orig_ecost]]):
            # Shorthands for data used to adjust original output breakouts
            base_cost_orig, eff_cost_orig, save_cost_orig, base_cost_adj, \
                eff_cost_adj = [tot_base_orig_ecost, tot_eff_orig_ecost,
                                tot_save_orig_ecost,
                                msegs_ecarb_cost["total"]["baseline"],
                                msegs_ecarb_cost["total"]["efficient"]]

        # Fuel switching case
        if out_fuel_save and out_fuel_gain:
            # Pull the fraction of efficient-case energy/carbon that remains w/
            # the original fuel in each year for the contributing measure/mseg
            fs_eff_splt_var = {
                yr: (fs_eff_splt[k][0][yr] / fs_eff_splt[k][1][yr]) if
                fs_eff_splt[k][1][yr] != 0 else 1
                for yr in self.handyvars.aeo_years}
            # Energy/carbon; original fuel
            mseg_out_break_adj[k]["baseline"][
                out_cz][out_bldg][out_eu][out_fuel_save], \
                mseg_out_break_adj[k]["efficient"][
                    out_cz][out_bldg][out_eu][out_fuel_save], \
                mseg_out_break_adj[k]["savings"][
                    out_cz][out_bldg][out_eu][out_fuel_save] = [
                # Remove adjusted baseline
                {yr: mseg_out_break_adj[k]["baseline"][
                    out_cz][out_bldg][out_eu][out_fuel_save][yr] - (
                        base_orig[yr] - base_adj[yr]) for
                 yr in self.handyvars.aeo_years},
                # Remove adjusted efficient case that remains with base fuel
                {yr: mseg_out_break_adj[k]["efficient"][
                    out_cz][out_bldg][out_eu][out_fuel_save][yr] - (
                        eff_orig[yr] - eff_adj[yr]) * fs_eff_splt_var[yr] for
                 yr in self.handyvars.aeo_years},
                # Savings is difference between adjusted baseline and efficient
                # and is subtracted from existing savings for baseline fuel (
                # e.g., savings becomes less positive)
                {yr: mseg_out_break_adj[k]["savings"][
                    out_cz][out_bldg][out_eu][out_fuel_save][yr] - (
                        (base_orig[yr] - base_adj[yr]) -
                        (eff_orig[yr] - eff_adj[yr]) * fs_eff_splt_var[yr]) for
                 yr in self.handyvars.aeo_years}]
            # Energy/carbon; switched to fuel
            mseg_out_break_adj[k]["efficient"][
                out_cz][out_bldg][out_eu][out_fuel_gain], \
                mseg_out_break_adj[k]["savings"][
                    out_cz][out_bldg][out_eu][out_fuel_gain] = [
                # Note: Baseline case remains zero (no baseline before switch)

                # Remove adjusted efficient case that switched from base fuel
                {yr: mseg_out_break_adj[k]["efficient"][
                    out_cz][out_bldg][out_eu][out_fuel_gain][yr] - ((
                        eff_orig[yr] - eff_adj[yr]) * (
                        1 - fs_eff_splt_var[yr]))
                 for yr in self.handyvars.aeo_years},
                # Adjusted efficient is added to the existing savings for
                # baseline fuel (e.g., savings becomes less negative)
                {yr: mseg_out_break_adj[k]["savings"][
                    out_cz][out_bldg][out_eu][out_fuel_gain][yr] + ((
                        eff_orig[yr] - eff_adj[yr]) * (
                        1 - fs_eff_splt_var[yr]))
                 for yr in self.handyvars.aeo_years}]
            # Energy costs
            if all([x for x in [tot_base_orig_ecost, tot_eff_orig_ecost,
                                tot_save_orig_ecost]]):
                # Pull the fraction of efficient-case energy cost that remains
                # w/ the orig. fuel in each year for the contrib. measure/mseg
                fs_eff_splt_cost = {
                    yr: (fs_eff_splt["cost"][0][yr] /
                         fs_eff_splt["cost"][1][yr]) if
                    fs_eff_splt["cost"][1][yr] != 0 else 1
                    for yr in self.handyvars.aeo_years}

                # Energy cost; original fuel
                mseg_out_break_adj["cost"]["baseline"][
                    out_cz][out_bldg][out_eu][out_fuel_save], \
                    mseg_out_break_adj["cost"]["efficient"][
                        out_cz][out_bldg][out_eu][out_fuel_save], \
                    mseg_out_break_adj["cost"]["savings"][
                        out_cz][out_bldg][out_eu][out_fuel_save] = [
                    # Remove adjusted baseline
                    {yr: mseg_out_break_adj["cost"]["baseline"][
                        out_cz][out_bldg][out_eu][out_fuel_save][yr] - (
                            base_cost_orig[yr] - base_cost_adj[yr]) for
                     yr in self.handyvars.aeo_years},
                    # Remove adjusted efficient case that remains with base
                    # fuel
                    {yr: mseg_out_break_adj["cost"]["efficient"][
                        out_cz][out_bldg][out_eu][out_fuel_save][yr] - (
                            eff_cost_orig[yr] - eff_cost_adj[yr]) *
                     fs_eff_splt_cost[yr] for yr in self.handyvars.aeo_years},
                    # Adjusted savings is difference between adjusted baseline
                    # and efficient and is subtracted from existing savings for
                    # baseline fuel (e.g., savings becomes less positive)
                    {yr: mseg_out_break_adj["cost"]["savings"][
                        out_cz][out_bldg][out_eu][out_fuel_save][yr] - (
                            (base_cost_orig[yr] - base_cost_adj[yr]) -
                            (eff_cost_orig[yr] - eff_cost_adj[yr]) *
                     fs_eff_splt_cost[yr]) for yr in self.handyvars.aeo_years}]
                # Switched to fuel
                mseg_out_break_adj["cost"]["efficient"][
                    out_cz][out_bldg][out_eu][out_fuel_gain], \
                    mseg_out_break_adj["cost"]["savings"][
                        out_cz][out_bldg][out_eu][out_fuel_gain] = [
                    # Note: Baseline case remains zero (no baseline before
                    # switch)

                    # Remove adjusted efficient case that switched from base
                    # fuel
                    {yr: mseg_out_break_adj["cost"]["efficient"][
                        out_cz][out_bldg][out_eu][out_fuel_gain][yr] - ((
                            eff_cost_orig[yr] - eff_cost_adj[yr]) * (
                            1 - fs_eff_splt_cost[yr])) for
                     yr in self.handyvars.aeo_years},
                    # Adjusted efficient is added to the existing savings for
                    # baseline fuel (e.g., savings becomes less negative)
                    {yr: mseg_out_break_adj["cost"]["savings"][
                        out_cz][out_bldg][out_eu][out_fuel_gain][yr] + ((
                            eff_cost_orig[yr] - eff_cost_adj[yr]) * (
                            1 - fs_eff_splt_cost[yr])) for
                     yr in self.handyvars.aeo_years}]
        # Fuel splits without fuel switching
        elif out_fuel_save:
            # Energy/carbon
            mseg_out_break_adj[k]["baseline"][
                out_cz][out_bldg][out_eu][out_fuel_save], \
                mseg_out_break_adj[k]["efficient"][
                    out_cz][out_bldg][out_eu][out_fuel_save], \
                mseg_out_break_adj[k]["savings"][
                    out_cz][out_bldg][out_eu][out_fuel_save] = [
                # Remove adjusted baseline
                {yr: mseg_out_break_adj[k]["baseline"][
                    out_cz][out_bldg][out_eu][out_fuel_save][yr] - (
                        base_orig[yr] - base_adj[yr]) for
                    yr in self.handyvars.aeo_years},
                # Remove adjusted efficient
                {yr: mseg_out_break_adj[k]["efficient"][
                    out_cz][out_bldg][out_eu][out_fuel_save][yr] - (
                        eff_orig[yr] - eff_adj[yr]) for
                 yr in self.handyvars.aeo_years},
                # Adjusted savings is difference between adjusted
                # baseline/efficient
                {yr: mseg_out_break_adj[k]["savings"][
                    out_cz][out_bldg][out_eu][out_fuel_save][yr] - (
                    save_orig[yr] - (base_adj[yr] - eff_adj[yr])) for
                 yr in self.handyvars.aeo_years}]
            # Energy costs
            if all([x for x in [tot_base_orig_ecost, tot_eff_orig_ecost,
                                tot_save_orig_ecost]]):
                mseg_out_break_adj["cost"]["baseline"][
                    out_cz][out_bldg][out_eu][out_fuel_save], \
                    mseg_out_break_adj["cost"]["efficient"][
                        out_cz][out_bldg][out_eu][out_fuel_save], \
                    mseg_out_break_adj["cost"]["savings"][
                        out_cz][out_bldg][out_eu][out_fuel_save] = [
                    # Remove adjusted baseline
                    {yr: mseg_out_break_adj["cost"]["baseline"][
                            out_cz][out_bldg][out_eu][out_fuel_save][yr] - (
                        base_cost_orig[yr] - base_cost_adj[yr]) for
                     yr in self.handyvars.aeo_years},
                    # Remove adjusted efficient
                    {yr: mseg_out_break_adj["cost"]["efficient"][
                        out_cz][out_bldg][out_eu][out_fuel_save][yr] - (
                        eff_cost_orig[yr] - eff_cost_adj[yr]) for
                     yr in self.handyvars.aeo_years},
                    # Adjusted savings is difference between adjusted
                    # baseline/efficient
                    {yr: mseg_out_break_adj["cost"]["savings"][
                        out_cz][out_bldg][out_eu][out_fuel_save][yr] - (
                        save_cost_orig[yr] - (
                            base_cost_adj[yr] - eff_cost_adj[yr])) for
                     yr in self.handyvars.aeo_years}]
        # All other cases
        else:
            # Energy/carbon
            mseg_out_break_adj[k]["baseline"][out_cz][out_bldg][out_eu], \
                mseg_out_break_adj[k]["efficient"][out_cz][out_bldg][out_eu], \
                mseg_out_break_adj[k]["savings"][out_cz][out_bldg][out_eu] = [
                # Remove adjusted baseline
                {yr: mseg_out_break_adj[k]["baseline"][
                    out_cz][out_bldg][out_eu][yr] - (
                        base_orig[yr] - base_adj[yr]) for
                    yr in self.handyvars.aeo_years},
                # Remove adjusted efficient
                {yr: mseg_out_break_adj[k]["efficient"][
                    out_cz][out_bldg][out_eu][yr] - (
                        eff_orig[yr] - eff_adj[yr]) for
                 yr in self.handyvars.aeo_years},
                # Adjusted savings is difference between adjusted
                # baseline/efficient
                {yr: mseg_out_break_adj[k]["savings"][
                    out_cz][out_bldg][out_eu][yr] - (
                    save_orig[yr] - (base_adj[yr] - eff_adj[yr])) for
                 yr in self.handyvars.aeo_years}]
            # Energy costs
            if all([x for x in [tot_base_orig_ecost, tot_eff_orig_ecost,
                                tot_save_orig_ecost]]):
                mseg_out_break_adj["cost"]["baseline"][
                    out_cz][out_bldg][out_eu], \
                    mseg_out_break_adj["cost"]["efficient"][
                        out_cz][out_bldg][out_eu], \
                    mseg_out_break_adj["cost"]["savings"][
                        out_cz][out_bldg][out_eu] = [
                    # Remove adjusted baseline
                    {yr: mseg_out_break_adj["cost"]["baseline"][
                            out_cz][out_bldg][out_eu][yr] - (
                        base_cost_orig[yr] - base_cost_adj[yr]) for
                     yr in self.handyvars.aeo_years},
                    # Remove adjusted efficient
                    {yr: mseg_out_break_adj["cost"]["efficient"][
                        out_cz][out_bldg][out_eu][yr] - (
                        eff_cost_orig[yr] - eff_cost_adj[yr]) for
                     yr in self.handyvars.aeo_years},
                    # Adjusted savings is difference between adjusted
                    # baseline/efficient
                    {yr: mseg_out_break_adj["cost"]["savings"][
                        out_cz][out_bldg][out_eu][yr] - (
                        save_cost_orig[yr] - (
                            base_cost_adj[yr] - eff_cost_adj[yr])) for
                     yr in self.handyvars.aeo_years}]

        return mseg_out_break_adj

    def update_dict(self, dict1, dict2):
        """Merge data from one dict into another dict.

        Note:
            Adds all branches in the second dict that are not already in
            the first dict to the first dict, given that the first dict
            is either blank or includes data nested under keys that include
            'primary' or 'secondary' (indicating the intended level of the
            data merge).

        Args:
            dict1 (dict): Dict to merge data into.
            dict2 (dict): Dict to merge data from.
        """
        # If the first dict is nested and the intended level of the data
        # merge has not yet been reached, proceed further down its branches
        if len(dict1.keys()) != 0 and all([
                "," not in x for x in dict1.keys()]):
            for (k, i), (k2, i2) in zip(
                    dict1.items(), dict2.items()):
                self.update_dict(i, i2)
        # If the first dict is not nested, or the dict is nested but the
        # intended level of the data merge has been reached, merge the
        # second dict's data into the first dict
        else:
            dict1.update(dict2)

    def apply_pkg_benefits(
            self, msegs_meas, adopt_scheme, cm_key):
        """Apply additional energy savings or cost benefits from packaging.

        Attributes:
            msegs_meas (dict): Data for the contributing microsegment of an
                individual measure that is being merged into the package.
            adopt_scheme (string): Assumed consumer adoption scenario.
            cm_key (tuple): Microsegment key describing the contributing
                microsegment currently being added (e.g. reg->bldg, etc.)

        Note:
            Users may define additional percent energy savings improvements or
            installed cost reductions for a measure package; this function
            applies those benefits to the package's energy, carbon, and cost
            data.

        Args:
            msegs_meas (dict): Original energy, carbon, and cost data to
                apply packaging benefits to (if applicable)

        Returns:
            Updated energy, carbon, and cost data for the packaged measure
            that reflects the additional user-defined energy savings and
            installed cost benefits for the package.
        """
        # Set additional energy savings and cost benefits defined by the user
        # for the current package being updated
        energy_ben = self.benefits["energy savings increase"]
        cost_ben = self.benefits["cost reduction"]

        # If additional energy savings benefits are not None and are non-zero,
        # apply them to the measure's energy, carbon, and energy/carbon costs
        if energy_ben not in [None, 0]:
            for x in ["energy", "carbon"]:
                for cs in ["competed", "total"]:
                    # Set short variable names for efficient
                    # energy and carbon data
                    eff = msegs_meas[x][cs]["efficient"]
                    # Apply the additional energy savings % benefit to the
                    # efficient energy and carbon data (disallow result
                    # below zero)
                    msegs_meas[x][cs]["efficient"] = {
                        key: eff[key] * (1 - energy_ben)
                        for key in self.handyvars.aeo_years}
                    # Set short variable names for baseline and efficient
                    # energy and carbon cost data
                    eff_c = msegs_meas["cost"][x][cs]["efficient"]
                    # Apply the additional energy savings % benefit to the
                    # efficient energy and carbon cost data (disallow result
                    # below zero)
                    msegs_meas["cost"][x][cs]["efficient"] = {
                        key: eff_c[key] * (1 - energy_ben)
                        for key in self.handyvars.aeo_years}

        # If additional installed cost benefits are not None and are non-zero,
        # apply them to the measure's stock cost
        if cost_ben not in [None, 0]:
            for cs in ["competed", "total"]:
                msegs_meas["cost"]["stock"][cs]["efficient"] = {
                    key: msegs_meas["cost"]["stock"][cs]["efficient"][key] *
                    (1 - cost_ben) for key in self.handyvars.aeo_years}

        return msegs_meas

    def merge_out_break(self, pkg_brk, meas_brk):
        """Merge output breakout data for an individual measure into a package.

        Note:
            The 'markets' attribute of an individual measure to be merged
            into a package includes partitioning fractions needed to breakout
            the measure's output markets/savings by key variables (e.g.,
            climate zone, building type, end use, etc.). These fractions are
            based on the portion of the measure's total energy use market that
            is attributable to each breakout variable (e.g., 50% of the
            measure's total energy market is attributed to the heating end
            use). This function unnormalizes the measure's breakout fractions
            and adds the resultant energy use sub-market to a new set of
            breakout information for a measure package. Output breakout dicts
            must be identically structured for the merge to work.

        Args:
            pkg_brk (dict): Packaged measure output breakout information to
                merge individual measure breakout information into.
            meas_brk (dict): Individual measure output breakout information
                to merge into the package breakout information.

        Returns:
            Updated output breakout information for the packaged measure
            that incorporates the individual measure's breakout information.
        """
        for (k, i), (k2, i2) in zip(
                sorted(pkg_brk.items()), sorted(meas_brk.items())):
            if isinstance(i2, dict) and (
                    sorted(list(i2.keys())) != self.handyvars.aeo_years):
                self.merge_out_break(i, i2)
            else:
                if k == k2 and (isinstance(i, dict) == isinstance(i2, dict)):
                    # If individual measure output breakout data is
                    # available to add to the current terminal leaf
                    # node for the package, either set the leaf node value
                    # to the unnormalized breakout data if no data already
                    # exist, or otherwise add the unnormalized breakout data
                    # for the individual measure to that of the package
                    if len(i.keys()) == 0:
                        pkg_brk[k] = {yr: i2[yr] for
                                      yr in self.handyvars.aeo_years}
                    else:
                        pkg_brk[k] = {yr: pkg_brk[k][yr] + i2[yr] for
                                      yr in self.handyvars.aeo_years}
                else:
                    raise KeyError(
                        "Output data dicts to merge for ECM '" + self.name +
                        " are not identically structured")
        return pkg_brk


def prepare_measures(measures, convert_data, msegs, msegs_cpl, handyvars,
                     handyfiles, cbecs_sf_byvint, tsv_data, base_dir, opts,
                     ctrb_ms_pkg_prep, tsv_data_nonfs):
    """Finalize measure markets for subsequent use in the analysis engine.

    Note:
        Determine which in a list of measures require updates to finalize
        stock, energy, carbon, and cost markets for further use in the
        analysis engine; instantiate these measures as Measure objects;
        execute the necessary updates for each object; and update the
        original list of measures accordingly.

    Args:
        measures (list): List of dicts with efficiency measure attributes.
        convert_data (dict): Measure cost unit conversion data.
        msegs (dict): Baseline microsegment stock and energy use.
        msegs_cpl (dict): Baseline technology cost, performance, and lifetime.
        handyvars (object): Global variables of use across Measure methods.
        handyfiles (object): Input files of use across Measure methods.
        cbecs_sf_byvint (dict): Commercial square footage by vintage data.
        tsv_data (dict): Data needed for time sensitive efficiency valuation.
        base_dir (string): Base directory.
        opts (object): Stores user-specified execution options.
        ctrb_ms_pkg_prep (list): Names of measures that contribute to pkgs.
        tsv_data_nonfs (dict): If applicable, base-case TSV data to apply to
            non-fuel switching measures under a high decarb. scenario.

    Returns:
        A list of dicts, each including a set of measure attributes that has
        been prepared for subsequent use in the analysis engine.

    Raises:
        ValueError: If more than one Measure object matches the name of a
            given input efficiency measure.
    """
    print('Initializing measures...', end="", flush=True)
    # Translate user options to a dictionary for further use in Measures
    opts_dict = vars(opts)
    # Initialize Measure() objects based on 'measures_update' list
    meas_update_objs = [Measure(base_dir, handyvars, handyfiles, opts_dict, **m) for m in measures]
    print("Complete")

    # Fill in EnergyPlus-based performance information for Measure objects
    # with a 'From EnergyPlus' flag in their 'energy_efficiency' attribute

    # Handle a superfluous 'undefined' key in the ECM performance field that is
    # generated by the 'Add ECM' web form in certain cases *** NOTE: WILL
    # FIX IN FUTURE UI VERSION ***
    if any([isinstance(m.energy_efficiency, dict) and (
        "undefined" in m.energy_efficiency.keys() and
        m.energy_efficiency["undefined"] == "From EnergyPlus") for
            m in meas_update_objs]):
        # NOTE: Comment out operations related to the import of ECM performance
        # data from EnergyPlus and yield an error until these operations are
        # fully supported in the future

        # Set default directory for EnergyPlus simulation output files
        # eplus_dir = base_dir + '/ecm_definitions/energyplus_data'
        # # Set EnergyPlus global variables
        # handyeplusvars = EPlusGlobals(eplus_dir, cbecs_sf_byvint)
        # # Fill in EnergyPlus-based measure performance information
        # [m.fill_eplus(
        #     msegs, eplus_dir, handyeplusvars.eplus_coltypes,
        #     handyeplusvars.eplus_files, handyeplusvars.eplus_vintage_weights,
        #     handyeplusvars.eplus_basecols) for m in meas_update_objs
        #     if 'EnergyPlus file' in m.energy_efficiency.keys()]
        raise ValueError(
            'One or more ECMs require EnergyPlus data for ECM performance; '
            'EnergyPlus-based ECM performance data are currently unsupported.')
    # Finalize 'markets' attribute for all Measure objects
    [m.fill_mkts(
        msegs, msegs_cpl, convert_data, tsv_data, opts, ctrb_ms_pkg_prep,
        tsv_data_nonfs)
     for m in meas_update_objs]
    return meas_update_objs


def prepare_packages(packages, meas_update_objs, meas_summary,
                     handyvars, handyfiles, base_dir, opts, convert_data):
    """Combine multiple measures into a single packaged measure.

    Args:
        packages (dict): Names of packages and measures that comprise them.
        meas_update_objs (dict): Attributes of individual efficiency measures.
        meas_summary (): List of dicts including previously prepared ECM data.
        handyvars (object): Global variables of use across Measure methods.
        handyfiles (object): Input files of use across Measure methods.
        base_dir (string): Base directory.
        opts (object): Stores user-specified execution options.
        convert_data (dict): Measure cost unit conversion data.

    Returns:
        A dict with packaged measure attributes that can be added to the
        existing measures database.
    """
    # Run through each unique measure package and merge the measures that
    # contribute to this package
    for p in packages:
        # Notify user that measure is being updated
        print("Updating ECM '" + p["name"] + "'...", end="", flush=True)

        # Establish a list of names for measures that contribute to the
        # package
        package_measures = p["contributing_ECMs"]
        # Determine the subset of all previously initialized measure
        # objects that contribute to the current package
        measure_list_package = [
            x for x in meas_update_objs if x.name in package_measures]
        # Determine which contributing measures have not yet been
        # initialized as objects
        measures_to_add = [mc for mc in package_measures if mc not in [
            x.name for x in measure_list_package]]
        # Initialize any missing contributing measure objects and add to
        # the existing list of contributing measure objects for the package
        for m in measures_to_add:
            # Load and set high level summary data for the missing measure
            meas_summary_data = [x for x in meas_summary if x["name"] == m]
            if len(meas_summary_data) == 1:
                # Translate user options to a dictionary for further use in
                # Measures
                opts_dict = vars(opts)
                # Initialize the missing measure as an object
                meas_obj = Measure(
                    base_dir, handyvars, handyfiles, opts_dict,
                    **meas_summary_data[0])
                # Reset measure technology type and total energy (used to
                # normalize output breakout fractions) to their values in the
                # high level summary data (reformatted during initialization)
                meas_obj.technology_type = meas_summary_data[0][
                    "technology_type"]
                # Assemble folder path for measure competition data
                meas_folder_name = handyfiles.ecm_compete_data
                # Assemble file name for measure competition data
                meas_file_name = meas_obj.name + ".pkl.gz"
                # Load and set competition data for the missing measure object
                with gzip.open(meas_folder_name / meas_file_name, 'r') as zp:
                    try:
                        meas_comp_data = pickle.load(zp)
                    except Exception as e:
                        raise Exception(
                            "Error reading in competition data of " +
                            "contributing ECM '" + meas_obj.name +
                            "' for package '" + p["name"] + "': " +
                            str(e)) from None
                for adopt_scheme in handyvars.adopt_schemes:
                    meas_obj.markets[adopt_scheme]["master_mseg"] = \
                        meas_summary_data[0]["markets"][adopt_scheme][
                            "master_mseg"]
                    meas_obj.markets[adopt_scheme]["mseg_adjust"] = \
                        meas_comp_data[adopt_scheme]
                    meas_obj.markets[adopt_scheme]["mseg_out_break"] = \
                        meas_summary_data[0]["markets"][adopt_scheme][
                            "mseg_out_break"]
                # Add missing measure object to the existing list
                measure_list_package.append(meas_obj)
            # Raise an error if no existing data exist for the missing
            # contributing measure
            elif len(meas_summary_data) == 0:
                raise ValueError(
                    "Contributing ECM '" + m +
                    "' cannot be added to package '" + p["name"] +
                    "' due to missing attribute data for this ECM")
            else:
                raise ValueError(
                    "More than one set of attribute data for " +
                    "contributing ECM '" + m + "'; ECM cannot be added to" +
                    "package '" + p["name"])

        # Determine which (if any) measure objects that contribute to
        # the package are invalid due to unacceptable input data sourcing
        measure_list_package_rmv = [
            x for x in measure_list_package if x.remove is True]

        # Warn user of no valid measures to package
        if len(measure_list_package_rmv) > 0:
            warnings.warn("WARNING (CRITICAL): Package '" + p["name"] +
                          "' removed due to invalid contributing ECM(s)")
            packaged_measure = False
        # Update package if valid contributing measures are available
        else:
            # Instantiate measure package object based on packaged measure
            # subset above
            packaged_measure = MeasurePackage(
                measure_list_package, p["name"], p["benefits"],
                handyvars, handyfiles, opts, convert_data)
            # Record heating/cooling equipment and envelope overlaps in
            # package after confirming that envelope measures are present
            if len(packaged_measure.contributing_ECMs_env) > 0:
                packaged_measure.htcl_adj_rec(opts)
            # Merge measures in the package object
            packaged_measure.merge_measures(opts)
            # Print update on measure status
            print("Success")

        # Add the new packaged measure to the measure list (if it exists)
        # for further evaluation like any other regular measure
        if packaged_measure is not False:
            meas_update_objs.append(packaged_measure)

    return meas_update_objs


def split_clean_data(meas_prepped_objs):
    """Reorganize and remove data from input Measure objects.

    Note:
        The input Measure objects have updated data, which must
        be reorganized/condensed for the purposes of writing out
        to JSON files.

    Args:
        meas_prepped_objs (object): Measure objects with data to
            be split in to separate dicts or removed.

    Returns:
        Three to four lists of dicts, one containing competition data for
        each updated measure, one containing high level summary
        data for each updated measure, another containing sector shape
        data for each measure (if applicable), and a final one containing
        efficient fuel split data, as applicable to fuel switching measures
        when the user has required fuel splits.
    """
    # Initialize lists of measure competition/summary data
    meas_prepped_compete = []
    meas_prepped_summary = []
    meas_prepped_shapes = []
    meas_eff_fs_splt = []
    # Loop through all Measure objects and reorganize/remove the
    # needed data.
    for m in meas_prepped_objs:
        # Initialize a reorganized measure competition data dict and efficient
        # fuel split data dict
        comp_data_dict, fs_splits_dict, shapes_dict = ({} for n in range(3))
        # Retrieve measure contributing microsegment data that
        # is relevant to markets competition in the analysis
        # engine, then remove these data from measure object
        for adopt_scheme in m.handyvars.adopt_schemes:
            # Delete contributing microsegment data that are
            # not relevant to competition in the analysis engine
            del m.markets[adopt_scheme]["mseg_adjust"][
                "secondary mseg adjustments"]["sub-market"]
            del m.markets[adopt_scheme]["mseg_adjust"][
                "secondary mseg adjustments"]["stock-and-flow"]
            # Add remaining contributing microsegment data to
            # competition data dict, then delete from measure
            comp_data_dict[adopt_scheme] = \
                m.markets[adopt_scheme]["mseg_adjust"]
            del m.markets[adopt_scheme]["mseg_adjust"]
            # If applicable, add efficient fuel split data to fuel split data
            # dict
            if len(m.eff_fs_splt[adopt_scheme].keys()) != 0:
                fs_splits_dict[adopt_scheme] = \
                    m.eff_fs_splt[adopt_scheme]
            # If applicable, add sector shape data
            if m.sector_shapes is not None and len(
                    m.sector_shapes[adopt_scheme].keys()) != 0:
                shapes_dict["name"] = m.name
                shapes_dict[adopt_scheme] = \
                    m.sector_shapes[adopt_scheme]
        # Delete info. about efficient fuel splits for fuel switch measures
        del m.eff_fs_splt
        # Delete info. about sector shapes
        del m.sector_shapes

        # Append updated competition data from measure to
        # list of competition data across all measures
        meas_prepped_compete.append(comp_data_dict)
        # Append fuel switching split information, if applicable
        meas_eff_fs_splt.append(fs_splits_dict)
        # Append sector shape information, if applicable
        meas_prepped_shapes.append(shapes_dict)
        # Delete 'handyvars' measure attribute (not relevant to
        # analysis engine)
        del m.handyvars
        # Delete 'tsv_features' measure attributes
        # (not relevant) for individual measures
        if not isinstance(m, MeasurePackage):
            del m.tsv_features
        # For measure packages, replace 'contributing_ECMs'
        # objects list with a list of these measures' names and remove
        # unnecessary heating/cooling equip/env overlap data
        if isinstance(m, MeasurePackage):
            m.contributing_ECMs = [
                x.name for x in m.contributing_ECMs]
            del m.htcl_overlaps
            del m.contributing_ECMs_eqp
            del m.contributing_ECMs_env
        # Append updated measure __dict__ attribute to list of
        # summary data across all measures
        meas_prepped_summary.append(m.__dict__)

    return meas_prepped_compete, meas_prepped_summary, meas_prepped_shapes, \
        meas_eff_fs_splt


def custom_formatwarning(msg, *a):
    """Given a warning object, return only the warning message."""
    return str(msg) + '\n'


def verboseprint(verbose, msg):
    """Print input message when the code is run in verbose mode.

    Args:
        verbose (boolean): Indicator of verbose mode
        msg (string): Message to print to console when in verbose mode

    Returns:
        Printed console message when code is run in verbose mode.
    """
    print(msg) if verbose else lambda *a, **k: None


def tsv_cost_carb_yrmap(tsv_data, aeo_years):
    """Map 8760 TSV cost/carbon data years to AEO years.

    Args:
        tsv_data: TSV cost or carbon input datasets.
        aeo_years: AEO year range.

    Returns:
        Mapping between TSV cost/carbon data years and AEO years.
    """

    # Set up a matrix mapping each AEO year to the years available in the
    # TSV data

    # Pull available years from TSV data
    tsv_yrs = list(sorted(tsv_data.keys()))
    # Establish the mapping from available TSV years to AEO years
    tsv_yr_map = {
        yr_tsv: [str(x) for x in range(
            int(yr_tsv), int(tsv_yrs[ind + 1]))]
        if (ind + 1) < len(tsv_yrs) else [str(x) for x in range(
            int(yr_tsv), int(aeo_years[-1]) + 1)]
        for ind, yr_tsv in enumerate(tsv_yrs)
    }
    # Prepend AEO years preceding the start year in the TSV data, if needed
    if (aeo_years[0] not in tsv_yr_map[tsv_yrs[0]]):
        yrs_to_prepend = range(int(aeo_years[0]), min([
            int(x) for x in tsv_yr_map[tsv_yrs[0]]]))
        tsv_yr_map[tsv_yrs[0]] = [str(x) for x in yrs_to_prepend] + \
            tsv_yr_map[tsv_yrs[0]]

    return tsv_yr_map


def downselect_packages(existing_pkgs: list[dict], pkg_subset: list) -> list:
    if "*" in pkg_subset:
        return existing_pkgs
    downselected_pkgs = [pkg for pkg in existing_pkgs if pkg["name"] in pkg_subset]

    return downselected_pkgs


def format_console_list(list_to_format):
    return [f"  {elem}\n" for elem in list_to_format]


def retrieve_valid_ecms(packages: list,
                        opts: argparse.NameSpace,  # noqa: F821
                        handyfiles: UsefulInputFiles) -> list:
    """Determine full list of individual measure JSON names that 1) contribute to selected
        packages in opts.ecm_packages, or 2) are included in opts.ecm_files, and 3) exist in the
        ecm definitions directory (opts.ecm_directory)

    Args:
        packages (list): List of valid packages
        opts (argparse.NameSpace): object storing user responses
        handyfiles (UsefulInputFiles): object storing input filepaths

    Returns:
        list: filtered list of ECMs that meet the criteria above
    """

    contributing_ecms = {
        ecm for pkg in packages for ecm in pkg["contributing_ECMs"]}
    opts.ecm_files.extend([ecm for ecm in contributing_ecms if ecm not in opts.ecm_files])
    valid_ecms = [
        x for x in handyfiles.indiv_ecms.iterdir() if x.suffix == ".json" and
        'package_ecms' not in x.name and x.stem in opts.ecm_files]

    return valid_ecms


def filter_invalid_packages(packages: list[dict],
                            ecms: list,
                            pkgs_filtered: bool) -> list[dict]:
    """Identify and filter packages whose ECMs are not all present in the individual ECM set

    Args:
        packages (list[dict]): List of packages imported from package_ecms.json
        ecms (list): List of ECM definitions file names
        pkgs_filtered (bool): Indicate whether the packages have been filtered via the
            ecm_packages argument

    Returns:
        filtered_packages (list[dict]): Packages list with invalid packages filtered out
    """

    invalid_pkgs = [pkg["name"] for pkg in packages if not
                    set(pkg["contributing_ECMs"]).issubset(set(ecms))]
    if invalid_pkgs:
        package_opt_txt = ""
        if pkgs_filtered:
            package_opt_txt = "specified with the ecm_packages argument "
        invalid_pkgs_txt = format_console_list(invalid_pkgs)
        msg = (f"WARNING: Packages in package_ecms.json {package_opt_txt}have contributing ECMs"
               " that are not present among ECM definitions. The following packages will not be"
               f" executed: \n{''.join(invalid_pkgs_txt)}")
        warnings.warn(msg)
    filtered_packages = [pkg for pkg in packages if pkg["name"] not in invalid_pkgs]

    return filtered_packages


def update_active_measures(run_setup: dict, to_active: list = [], to_inactive: list = []) -> dict:
    """Update active/inactive values of the run_setup dictionary

    Args:
        run_setup (dict): dictionary to be used as the analysis engine setup file
        to_active (list, optional): measures or packages to set to active and remove from
            inactive. Defaults to [].
        to_inactive (list, optional): measures or packages to set to inactive and remove from
            active. Defaults to [].

    Returns:
        dict: run_setup data with updated active and inactive lists
    """
    active_set = set(run_setup["active"])
    inactive_set = set(run_setup["inactive"])

    # Set active and remove from inactive
    active_set.update(to_active)
    inactive_set.difference_update(to_active)

    # Set inactive and remove from active
    active_set.difference_update(to_inactive)
    inactive_set.update(to_inactive)

    run_setup["active"] = list(active_set)
    run_setup["inactive"] = list(inactive_set)

    return run_setup


def initialize_run_setup(input_files: UsefulInputFiles) -> dict:
    """Reads in analysis engine setup file, run_setup.json, and initializes values. If the file
        exists and has measures set to 'active', those will be moved to 'inactive'. If the file
        does not exist, return a dictionary with empty 'active' and 'inactive' lists.

    Args:
        input_files (UsefulInputFiles): UsefulInputFiles instance

    Returns:
        dict: run_setup data with active and inactive lists
    """
    try:
        am = open(input_files.run_setup, 'r')
        try:
            run_setup = json.load(am, object_pairs_hook=OrderedDict)
        except ValueError as e:
            raise ValueError(
                f"Error reading in '{input_files.run_setup}': {str(e)}") from None
        am.close()
        # Initalize all measures as inactive
        run_setup = update_active_measures(run_setup, to_inactive=run_setup["active"])
    except FileNotFoundError:
        run_setup = {"active": [], "inactive": []}

    return run_setup


def main(opts: argparse.NameSpace):  # noqa: F821
    """Import and prepare measure attributes for analysis engine.

    Note:
        Determine which measure definitions in an 'ecm_definitions'
        sub-folder are new or edited; prepare the cost, performance, and
        markets attributes for these measures for use in the analysis
        engine; and write prepared data to analysis engine input files.

    Args:
        opts (argparse.NameSpace): argparse object containing the argument attributes
    """

    # Set current working directory
    base_dir = getcwd()

    # Custom format all warning messages (ignore everything but
    # message itself) *** Note: sometimes yields error; investigate ***
    # warnings.formatwarning = custom_formatwarning
    # Instantiate useful input files object
    handyfiles = UsefulInputFiles(opts)

    # UNCOMMENT WITH ISSUE 188
    # # Ensure that all AEO-based JSON data are drawn from the same AEO version
    # if len(numpy.unique([splitext(x)[0][-4:] for x in [
    #         handyfiles.msegs_in, handyfiles.msegs_cpl_in,
    #         handyfiles.metadata]])) > 1:
    #     raise ValueError("Inconsistent AEO version used across input files")

    # Instantiate useful variables object
    handyvars = UsefulVars(base_dir, handyfiles, opts)

    # Import file to write prepared measure attributes data to for
    # subsequent use in the analysis engine (if file does not exist,
    # provide empty list as substitute, since file will be created
    # later when writing ECM data)
    try:
        es = open(handyfiles.ecm_prep, 'r')
        try:
            meas_summary = json.load(es)
        except ValueError as e:
            raise ValueError(f"Error reading in '{handyfiles.ecm_prep}': {str(e)}") from None
        es.close()
        # Flag if the ecm_prep file already exists
        ecm_prep_exists = True
    except FileNotFoundError:
        meas_summary = []
        ecm_prep_exists = ""

    # Import packages JSON, filter as needed
    with open(handyfiles.ecm_packages, 'r') as mpk:
        try:
            meas_toprep_package_init = json.load(mpk)
        except ValueError as e:
            raise ValueError(
                f"Error reading in ECM package '{handyfiles.ecm_packages}': {str(e)}") from None
    meas_toprep_package_init = downselect_packages(meas_toprep_package_init, opts.ecm_packages)

    # If applicable, import file to write prepared measure sector shapes to
    # (if file does not exist, provide empty list as substitute, since file
    # will be created later when writing ECM data)
    if opts.sect_shapes is True:
        try:
            es_ss = open(handyfiles.ecm_prep_shapes, 'r')
            try:
                meas_shapes = json.load(es_ss)
            except ValueError:
                raise ValueError(f"Error reading in '{handyfiles.ecm_prep_shapes}'")
            es_ss.close()
        except FileNotFoundError:
            meas_shapes = []

    # Determine full list of individual measure JSON names
    meas_toprep_indiv_names = retrieve_valid_ecms(meas_toprep_package_init, opts, handyfiles)

    # Initialize list of all individual measures that require updates
    meas_toprep_indiv = []
    # Initialize list of individual measures that require an update due to
    # a change in their individual definition (and not a change in the
    # definition or other contributing measures of a package they are a
    # part of, if applicable)
    meas_toprep_indiv_nopkg = []

    # If user desires isolation of envelope impacts within envelope/HVAC
    # packages, develop a list that indicates which individual ECMs contribute
    # to which package(s); this info. is needed for making copies of certain
    # ECMs and ECM packages that serve as counterfactuals for the isolation of
    # envelope impacts within packages
    if opts.health_costs is True or opts.pkg_env_sep is True:
        # Initialize list to track ECM packages and contributing ECMs
        ctrb_ms_pkg_all = []
        # Initialize list to track ECM packages that should be copied as
        # counterfactuals for isolating envelope impacts
        pkg_copy_flag = []
        # Add package/contributing ECM information to list
        for p in meas_toprep_package_init:
            ctrb_ms_pkg_all.append([p["name"], p["contributing_ECMs"]])
        if opts.pkg_env_sep is True:
            # Import separate file that will ultimately store all
            # counterfactual package data for later use
            try:
                ecf = open(handyfiles.ecm_prep_env_cf, 'r')
                try:
                    meas_summary_env_cf = json.load(ecf)
                except ValueError as e:
                    raise ValueError(
                        f"Error reading in '{handyfiles.ecm_prep_env_cf}': {str(e)}") from None
                ecf.close()
                # In some cases, individual ECMs may be defined and written to
                # the counterfactual package data; these ECMs should be added
                # to the list of previously prepared individual ECMs so that
                # they are not prepared again if their definitions haven't
                # been updated
                meas_summary_env_cf_indiv = [
                    m for m in meas_summary_env_cf if
                    "contributing_ECMs" not in m.keys()]
                if len(meas_summary_env_cf_indiv) != 0:
                    meas_summary = meas_summary + meas_summary_env_cf_indiv
            except FileNotFoundError:
                meas_summary_env_cf = []
            # If applicable, import separate file that will store
            # counterfactual package sector shape data
            try:
                ecf_ss = open(handyfiles.ecm_prep_env_cf, 'r')
                try:
                    meas_shapes_env_cf = json.load(ecf_ss)
                except ValueError:
                    raise ValueError(
                        f"Error reading in '{handyfiles.ecm_prep_env_cf_shapes}'") from None
                ecf_ss.close()
            except FileNotFoundError:
                meas_shapes_env_cf = []
        else:
            meas_summary_env_cf, meas_shapes_env_cf = (
                None for n in range(2))
    else:
        ctrb_ms_pkg_all, pkg_copy_flag, meas_summary_env_cf, \
            meas_shapes_env_cf = (None for n in range(4))

    # Import all individual measure JSONs
    for mi in meas_toprep_indiv_names:
        with open(handyfiles.indiv_ecms / mi, 'r') as jsf:
            try:
                # Load each JSON into a dict
                meas_dict = json.load(jsf)
                # Shorthand for previously prepared measured data that match
                # current measure
                match_in_prep_file = [y for y in meas_summary if (
                    "contributing_ECMs" not in y.keys() and
                    y["name"] == meas_dict["name"]) or (
                    "contributing_ECMs" in y.keys() and
                    meas_dict["name"] in y["contributing_ECMs"])]
                # Determine whether dict should be added to list of individual
                # measure definitions to update. Add a measure dict to the list
                # requiring further preparation if: a) measure is in package
                # (may be removed from update later) b) measure JSON time stamp
                # indicates it has been modified since the last run of
                # 'ecm_prep.py' c) measure name is not already included in
                # database of prepared measure attributes ('/generated/ecm_prep.json'); d)
                # measure does not already have competition data prepared for
                # it (in '/generated/ecm_competition_data' folder), or
                # or e) command line arguments applied to the measure are not
                # consistent with those reported out the last time the measure
                # was prepared (based on 'usr_opts' attribute), excepting
                # the 'verbose', 'yaml', and 'ecm_directory' options, which have no bearing
                # on results
                compete_files = [x for x in handyfiles.ecm_compete_data.iterdir() if not
                                 x.name.startswith('.')]
                ignore_opts = ["verbose", "yaml", "ecm_directory", "ecm_files", "ecm_files_user",
                               "ecm_packages", "ecm_files_regex"]
                update_indiv_ecm = ((ecm_prep_exists and stat(
                    handyfiles.indiv_ecms / mi).st_mtime > stat(
                    handyfiles.ecm_prep).st_mtime) or
                    (len(match_in_prep_file) == 0 or (
                        "(CF)" not in meas_dict["name"] and all([all([
                            x["name"] != Path(y.stem).stem for y in
                            compete_files]) for
                            x in match_in_prep_file])) or
                        (opts is None and not all([all([
                         m["usr_opts"][k] is False
                         for k in m["usr_opts"].keys()]) for
                            m in match_in_prep_file])) or
                        (not all([all([m["usr_opts"][x] ==
                                      vars(opts)[x] for x in [
                            k for k in vars(opts).keys() if
                            k not in ignore_opts]]) for m in
                            match_in_prep_file]))))
                # Add measure to tracking of individual measures needing update
                # independent of required updates to packages they are a
                # part of (if applicable)
                if update_indiv_ecm:
                    meas_toprep_indiv_nopkg.append(meas_dict["name"])
                # Register name of package measure is a part of, if applicable
                meas_in_pkgs = any([
                    meas_dict["name"] in pkg["contributing_ECMs"] for pkg in
                    meas_toprep_package_init])
                if update_indiv_ecm or meas_in_pkgs:
                    # Check to ensure that tech switching information is
                    # available, if needed; otherwise throw a warning
                    # about this measure

                    # Check for tech. switch attribute, if not there set NA
                    try:
                        meas_dict["tech_switch_to"]
                    except KeyError:
                        meas_dict["tech_switch_to"] = "NA"
                    # If tech switching information is None unexpectedly
                    # (e.g., for a measure that switches fuels, or from
                    # resistance-based tech. to HPs, or to LEDs), prompt user
                    # to provide this information and rerun
                    if meas_dict["tech_switch_to"] == "NA" and (
                            meas_dict["fuel_switch_to"] is not None or (
                                any([x in meas_dict["name"] for x in [
                                    "LED", "solid state",
                                    "Solid State", "SSL"]]) or
                                (any([x in meas_dict["name"] for x in [
                                        "HP", "heat pump", "Heat Pump"]]) and (
                                    meas_dict["technology"] is not None and
                                    (any([
                                        x in handyvars.resist_ht_wh_tech for
                                        x in meas_dict["technology"]]) or
                                     meas_dict["technology"] in
                                     handyvars.resist_ht_wh_tech))))):
                        # Print missing tech switch info. warning
                        raise ValueError(
                            "Measure is missing expected technology switching "
                            "info.; add to 'tech_switch_to' attribute in the "
                            "measure definition JSON and rerun ecm_prep, "
                            "e.g.:\n"
                            "'tech_switch_to': 'ASHP' (switch to ASHP)\n"
                            "'tech_switch_to': 'GSHP' (switch to GSHP)\n"
                            "'tech_switch_to': 'HPWH' (switch to HPWH)\n"
                            "'tech_switch_to': 'electric cooking' "
                            "(switch to electric cooking)\n"
                            "'tech_switch_to': 'electric drying' "
                            "(switch to electric drying)\n"
                            "'tech_switch_to': 'LEDs' "
                            "(switch to LED lighting)\n"
                            " Alternatively, set 'tech_switch_to' to null "
                            "if no tech switching is meant to be represented")
                    else:
                        tech_swtch = meas_dict["tech_switch_to"]

                    # Append measure dict to list of measure definitions
                    # to update if it meets the above criteria
                    meas_toprep_indiv.append(meas_dict)
                    # Add copies of the measure that examine multiple scenarios
                    # of public health cost data additions, assuming the
                    # measure is not already a previously prepared copy
                    # that reflects these additions (judging by name)
                    if opts.health_costs is True and \
                            "PHC" not in meas_dict["name"]:
                        # Check to ensure that the measure applies to the
                        # electric fuel type (or switches to it); if not, do
                        # not prepare additional versions of the measure with
                        # health costs
                        if ((((type(meas_dict["fuel_type"]) is not list) and
                              meas_dict["fuel_type"] not in [
                            "electricity", "all"]) or ((type(
                                meas_dict["fuel_type"]) is list) and all([
                                x not in ["electricity", "all"] for x in
                                meas_dict["fuel_type"]]))) and
                                meas_dict["fuel_switch_to"] != "electricity"):
                            # Warn the user that ECMs that do not apply to the
                            # electric fuel type will not be prepared with
                            # public cost health adders
                            warnings.warn(
                                "WARNING: " + meas_dict["name"] + " does not "
                                "apply to the electric fuel type; versions of "
                                "this ECM with low/high public health cost "
                                "adders will not be prepared.")
                        else:
                            for scn in handyvars.health_scn_names:
                                # Determine unique measure copy name
                                new_name = meas_dict["name"] + "-" + scn[0]
                                # Copy the measure
                                new_meas = copy.deepcopy(meas_dict)
                                # Set the copied measure name to the name above
                                new_meas["name"] = new_name
                                # Append the copied measure to list of measure
                                # definitions to update
                                meas_toprep_indiv.append(new_meas)
                                # Add measure to tracking of individual
                                # measures needing update independent of
                                # required updates to packages they are a
                                # part of (if applicable)
                                if update_indiv_ecm:
                                    meas_toprep_indiv_nopkg.append(
                                        new_meas["name"])
                                # Flag the package(s) that the measure that was
                                # copied contributes to; this package will be
                                # copied as well
                                pkgs_to_copy = [
                                    x[0] for x in ctrb_ms_pkg_all if
                                    meas_dict["name"] in x[1]]
                                # Add the package name, the package copy name,
                                # the name of the original measure that
                                # contributes to the package, and the measure
                                # copy name
                                for p in pkgs_to_copy:
                                    # Set pkg copy name
                                    new_pkg_name = p + "-" + scn[0]
                                    pkg_copy_flag.append([
                                        p, new_pkg_name,
                                        meas_dict["name"], new_name])

                    # Check for whether a reference case analogue measure
                    # should be added (user option is present, measure is in
                    # ESTAR/IECC/90.1 tier, measure applies to equipment
                    # not envelope components, and either the
                    # measure does not switch equipment types or it switches
                    # and exogenous switching rates are not used)
                    if opts is not None and opts.add_typ_eff is True and \
                        any([x in meas_dict["name"] for x in [
                            "ENERGY STAR", "ESTAR", "IECC", "90.1"]]) and (
                            not ((isinstance(meas_dict["technology"], list)
                                  and all([x in handyvars.demand_tech for
                                           x in meas_dict["technology"]])) or
                                 meas_dict["technology"] in
                                 handyvars.demand_tech)) and (
                            not tech_swtch or opts.exog_hp_rates is False):
                        add_ref_meas = True
                    else:
                        add_ref_meas = ""

                    # Add copies of ESTAR, IECC, or 90.1 measures that
                    # downgrade to typical/BAU efficiency levels; exclude typ.
                    # /BAU fuel switching measures that are assessed under
                    # exogenously determined FS rates; such measures must be
                    # manually defined by the user and are handled like normal
                    # measures; also exclude typical windows/envelope measures,
                    # as these are already baked into the energy use totals for
                    # typical/BAU HVAC equipment measures
                    if add_ref_meas:
                        # Find substring in existing measure name to replace
                        if "ENERGY STAR" in meas_dict["name"]:
                            name_substr = "ENERGY STAR"
                        elif "ESTAR" in meas_dict["name"]:
                            name_substr = "ESTAR"
                        elif "IECC c. 2021" in meas_dict["name"]:
                            name_substr = "IECC c. 2021"
                        elif "90.1 c. 2019" in meas_dict["name"]:
                            name_substr = "90.1 c. 2019"
                        else:
                            name_substr = ""
                        # Determine unique measure copy name
                        if name_substr:
                            new_name = meas_dict["name"].replace(
                                name_substr, "Ref. Case")
                        else:
                            new_name = meas_dict["name"] + " Ref. Case"
                        # Copy the measure
                        new_meas = copy.deepcopy(meas_dict)
                        # Set the copied measure name to the name above
                        new_meas["name"] = new_name
                        opts.ecm_files.append(new_meas["name"])
                        # If measure was set to fuel switch without exogenous
                        # FS rates, reset typical/BAU analogue FS to None (
                        # e.g., such that for an ASHP FS measure, a typical/
                        # BAU fossil-based heating analogue is created
                        # for later competition with that FS measure). Also
                        # ensure that no tech switching is specified for
                        # consistency w/ fuel_switch_to
                        if (meas_dict["fuel_switch_to"] is not None and
                                opts.exog_hp_rates is False):
                            new_meas["fuel_switch_to"], \
                                new_meas["tech_switch_to"] = (
                                    None for n in range(2))
                        # Append the copied measure to list of measure
                        # definitions to update
                        meas_toprep_indiv.append(new_meas)
                        # Add measure to tracking of individual
                        # measures needing update independent of
                        # required updates to packages they are a
                        # part of (if applicable)
                        if update_indiv_ecm:
                            meas_toprep_indiv_nopkg.append(new_meas["name"])
                    # If desired by user, add copies of HVAC equipment measures
                    # that are part of packages; these measures will be
                    # assigned no relative performance improvement and
                    # added to copies of those HVAC/envelope packages, to serve
                    # as counter-factuals that allow isolation of envelope
                    # impacts within each package
                    if opts is not None and opts.pkg_env_sep is True and len(
                        ctrb_ms_pkg_all) != 0 and (any(
                            [meas_dict["name"] in x[1] for
                             x in ctrb_ms_pkg_all])) and (
                        (isinstance(meas_dict["end_use"], list) and any([
                            x in ["heating", "cooling"] for
                            x in meas_dict["end_use"]])) or
                            meas_dict["end_use"] in
                            ["heating", "cooling"]) and (
                        not ((isinstance(meas_dict["technology"], list)
                              and all([x in handyvars.demand_tech for
                                       x in meas_dict["technology"]])) or
                             meas_dict["technology"] in
                             handyvars.demand_tech)):
                        # Determine measure copy name, CF for counterfactual
                        new_name = meas_dict["name"] + " (CF)"
                        # Copy the measure
                        new_meas = copy.deepcopy(meas_dict)
                        # Set the copied measure name to the name above
                        new_meas["name"] = new_name
                        # Append the copied measure to list of measure
                        # definitions to update
                        meas_toprep_indiv.append(new_meas)
                        # Add measure to tracking of individual
                        # measures needing update independent of
                        # required updates to packages they are a
                        # part of (if applicable)
                        if update_indiv_ecm:
                            meas_toprep_indiv_nopkg.append(new_meas["name"])
                        # Flag the package(s) that the measure that was copied
                        # contributes to; this package will be copied as well
                        # to produce the final counterfactual data
                        pkgs_to_copy = [x[0] for x in ctrb_ms_pkg_all if
                                        meas_dict["name"] in x[1]]
                        # Add the package name, the package copy name,
                        # the name of the original measure that contributes
                        # to the package, and the measure copy name
                        for p in pkgs_to_copy:
                            # Set pkg copy name
                            new_pkg_name = p + " (CF)"
                            pkg_copy_flag.append([
                                p, new_pkg_name, meas_dict["name"], new_name])
            except ValueError as e:
                raise ValueError(
                    "Error reading in ECM '" + mi + "': " +
                    str(e)) from None

    # Find package measure definitions that are new or were edited since
    # the last time 'ecm_prep.py' routine was run, or are comprised of
    # individual measures that are new or were edited since the last time
    # 'ecm_prep.py' routine was run

    # Initialize list of measure package dicts to prepare
    meas_toprep_package = []
    # Initialize a list to track which individual ECMs contribute to packages
    ctrb_ms_pkg_prep = []
    # Identify all previously prepared measure packages
    meas_prepped_pkgs = [mpkg for mpkg in meas_summary if "contributing_ECMs" in mpkg.keys()]
    # Identify and filter packages whose ECMs are not all present in ECM list
    ecm_pkg_filtered = False
    if opts.ecm_packages is not None:
        ecm_pkg_filtered = True
    ecm_names = [meas.stem for meas in meas_toprep_indiv_names]
    meas_toprep_package_init = filter_invalid_packages(meas_toprep_package_init,
                                                       ecm_names,
                                                       ecm_pkg_filtered)

    # Write initial data for run_setup.json
    # Import analysis engine setup file
    run_setup = initialize_run_setup(handyfiles)

    # Set contributing ECMs as inactive in run_setup and throw warning, set all others as active
    ctrb_ms = [ecm for pkg in meas_toprep_package_init for ecm in pkg["contributing_ECMs"]]
    non_ctrb_ms = [ecm for ecm in opts.ecm_files if ecm not in ctrb_ms]
    excluded_ind_ecms = [ecm for ecm in opts.ecm_files_user if ecm in ctrb_ms]
    run_setup = update_active_measures(run_setup,
                                       to_active=non_ctrb_ms,
                                       to_inactive=excluded_ind_ecms)
    if excluded_ind_ecms:
        excluded_ind_ecms_txt = format_console_list(excluded_ind_ecms)
        warnings.warn("The following ECMs were selected to be prepared, but due to their"
                      " presence in one or more packages, they will not be run individually and"
                      " will only be included as part of the package(s):"
                      f"\n{''.join(excluded_ind_ecms_txt)}")

    # Set packages to active in run_setup
    valid_packages = [pkg["name"] for pkg in meas_toprep_package_init]
    run_setup = update_active_measures(run_setup, to_active=valid_packages)

    # Loop through each package dict in the current list and determine which
    # of these package measures require further preparation
    for m in meas_toprep_package_init:
        # Determine the subset of previously prepared package measures
        # with the same name as the current package measure
        m_exist = [me for me in meas_prepped_pkgs if me["name"] == m["name"]]

        # Add a package dict to the list requiring further preparation after first checking if all
        # of the package's contributing measures have been updated, then if: a) the package is
        # new, b) package does not already have competition data prepared for it, c) package
        # "contributing_ECMs" and/or "benefits" parameters have been edited from a previous
        # version, or d) package was prepared with different settings around including envelope
        # costs (if applicable) than in the current run

        # Check for existing competition data for the package (condition b)
        name_mask = all(m["name"] != Path(y.stem).stem for y in
                        handyfiles.ecm_compete_data.iterdir())
        exst_ecms_mask = exst_engy_save_mask = exst_cost_red_mask = False
        exst_pkg_env_mask_1 = exst_pkg_env_mask_2 = False
        # Check for differences in the specification of the previously prepared
        # package measure and the current package measure of the same name (condition c and d)
        if len(m_exist) == 1:
            exst_ecms_mask = (sorted(m["contributing_ECMs"]) !=
                              sorted(m_exist[0]["contributing_ECMs"]))
            # Difference in expected package energy savings
            exst_engy_save_mask = (m["benefits"]["energy savings increase"] !=
                                   m_exist[0]["benefits"]["energy savings increase"])
            exst_cost_red_mask = (m["benefits"]["cost reduction"] !=
                                  m_exist[0]["benefits"]["cost reduction"])
            exst_pkg_env_mask_1 = (opts is not None and opts.pkg_env_costs is not False and
                                   m_exist[0]["pkg_env_costs"] is False)
            exst_pkg_env_mask_2 = (opts is None or opts.pkg_env_costs is False and
                                   m_exist[0]["pkg_env_costs"] is not False)

        # Check for conditions that would indicate a package needs to be processed
        # (condition a and previously inspected conditions b, c, and d)
        if len(m_exist) == 0 or name_mask or \
                ((exst_ecms_mask or exst_engy_save_mask or exst_cost_red_mask) or
                 (exst_pkg_env_mask_1 or exst_pkg_env_mask_2)):

            meas_toprep_package.append(m)
            # Add contributing ECMs to those needing updates
            ctrb_ms_pkg_prep.extend(m["contributing_ECMs"])
            # If package is flagged as needing a copy to serve as a
            # counterfactual for isolating envelope impacts, make the copy
            if pkg_copy_flag:
                pkg_item = [x for x in pkg_copy_flag if x[0] == m["name"]]
            else:
                pkg_item = []
            if len(pkg_item) > 0:
                # Determine unique package copy name
                new_pkg_name = pkg_item[0][1]
                # Copy the package
                new_pkg = copy.deepcopy(m)
                # Set the copied package name to the name above
                new_pkg["name"] = new_pkg_name
                # Parse new package data to find information about revised
                # names in contributing ECM set
                for p in pkg_item:
                    # Replace original ECM names from the package's list of contributing ECMs with
                    # those of the ECM copies such that data for these copies will be pulled into
                    # the package assessment
                    for ind, ecm in enumerate(new_pkg["contributing_ECMs"]):
                        if ecm in p:
                            new_pkg["contributing_ECMs"][ind] = p[3]
                # Append the copied package measure to list of measure definitions to update, and
                # also update the list of individual measures that contribute to packages being
                # prepared
                meas_toprep_package.append(new_pkg)
                ctrb_ms_pkg_prep.extend(new_pkg["contributing_ECMs"])

        # Raise an error if the current package matches the name of
        # multiple previously prepared packages
        elif len(m_exist) > 1:
            raise ValueError(
                "Multiple existing ECM names match '" + m["name"] + "'")

    # Remove measures previously added to the list solely b/c of their
    # membership in a package that do not belong to a package needing updates
    meas_toprep_indiv = [m for m in meas_toprep_indiv if any([
        m["name"] in x for x in [meas_toprep_indiv_nopkg, ctrb_ms_pkg_prep]])]

    print("\nImporting supporting data...", end="", flush=True)
    # If one or more measure definition is new or has been edited, proceed
    # further with 'ecm_prep.py' routine; otherwise end the routine
    if len(meas_toprep_indiv) > 0 or len(meas_toprep_package) > 0:
        # Import baseline microsegments
        if opts.alt_regions in ['EMM', 'State']:  # Extract EMM/state files
            bjszip = handyfiles.msegs_in
            with gzip.GzipFile(bjszip, 'r') as zip_ref:
                msegs = json.loads(zip_ref.read().decode('utf-8'))
        else:
            with open(handyfiles.msegs_in, 'r') as msi:
                try:
                    msegs = json.load(msi)
                except ValueError as e:
                    raise ValueError(
                        "Error reading in '" +
                        handyfiles.msegs_in + "': " + str(e)) from None
        # Import baseline cost, performance, and lifetime data
        bjszip = handyfiles.msegs_cpl_in
        with gzip.GzipFile(bjszip, 'r') as zip_ref:
            msegs_cpl = json.loads(zip_ref.read().decode('utf-8'))
        # Import measure cost unit conversion data
        with open(handyfiles.cost_convert_in, 'r') as cc:
            try:
                convert_data = json.load(cc)
            except ValueError as e:
                raise ValueError(
                    f"Error reading in '{handyfiles.cost_convert_in}': {str(e)}") from None
        # Import CBECs square footage by vintage data (used to map EnergyPlus
        # commercial building vintages to Scout building vintages)
        with open(handyfiles.cbecs_sf_byvint, 'r') as cbsf:
            try:
                cbecs_sf_byvint = json.load(cbsf)[
                    "commercial square footage by vintage"]
            except ValueError as e:
                raise ValueError(
                    f"Error reading in '{handyfiles.cbecs_sf_byvint}': {str(e)}") from None
        if (opts.alt_regions == 'EMM' and ((
                opts.tsv_metrics is not False or any([
                ("tsv_features" in m.keys() and m["tsv_features"] is not None)
                for m in meas_toprep_indiv])) or
                opts is not None and opts.sect_shapes is True)):
            # Import load, price, and emissions shape data needed for time
            # sensitive analysis of measure energy efficiency impacts
            tsv_l = handyfiles.tsv_load_data
            tsv_l_zip = tsv_l.with_suffix('.gz')
            with gzip.GzipFile(tsv_l_zip, 'r') as zip_ref_l:
                tsv_load_data = json.loads(zip_ref_l.read().decode('utf-8'))
            # When sector shapes are specified and no other time sensitive
            # valuation or features are present, assume that hourly price
            # and emissions data will not be needed
            if ((opts.sect_shapes is True)
                and opts.tsv_metrics is False and all([(
                    "tsv_features" not in m.keys() or
                    m["tsv_features"] is None) for m in meas_toprep_indiv])):
                tsv_data, tsv_data_nonfs = ({
                    "load": tsv_load_data, "price": None,
                    "price_yr_map": None, "emissions": None,
                    "emissions_yr_map": None} for n in range(2))
            else:
                tsv_c = handyfiles.tsv_cost_data
                tsv_c_zip = tsv_c.with_suffix('.gz')
                with gzip.GzipFile(tsv_c_zip, 'r') as zip_ref_c:
                    tsv_cost_data = \
                        json.loads(zip_ref_c.read().decode('utf-8'))
                # Case where the user assesses time sensitive cost
                # factors for before grid decarbonization for non-fuel
                # switching measures
                if handyfiles.tsv_cost_data_nonfs is not None:
                    tsv_c_nonfs = handyfiles.tsv_cost_data_nonfs
                    tsv_c_nonfs_zip = tsv_c_nonfs.with_suffix('.gz')
                    with gzip.GzipFile(tsv_c_nonfs_zip, 'r') as \
                            zip_ref_nonfs_c:
                        tsv_cost_nonfs_data = \
                            json.loads(zip_ref_nonfs_c.read().decode('utf-8'))
                else:
                    tsv_cost_nonfs_data = None

                tsv_cb = handyfiles.tsv_carbon_data
                tsv_cb_zip = tsv_cb.with_suffix('.gz')
                with gzip.GzipFile(tsv_cb_zip, 'r') as zip_ref_cb:
                    tsv_carbon_data = \
                        json.loads(zip_ref_cb.read().decode('utf-8'))
                # Case where the user assesses time sensitive emissions
                # factors for before grid decarbonization for non-fuel
                # switching measures
                if handyfiles.tsv_carbon_data_nonfs is not None:
                    tsv_cb_nonfs = handyfiles.tsv_carbon_data_nonfs
                    tsv_cb_nonfs_zip = tsv_cb_nonfs.with_suffix('.gz')
                    with gzip.GzipFile(tsv_cb_nonfs_zip, 'r') as \
                            zip_ref_nonfs_cb:
                        tsv_carbon_nonfs_data = \
                            json.loads(zip_ref_nonfs_cb.read().decode('utf-8'))
                else:
                    tsv_carbon_nonfs_data = None

                # Map years available in 8760 TSV cost/carbon data to AEO yrs.
                tsv_cost_yrmap = tsv_cost_carb_yrmap(
                    tsv_cost_data["electricity price shapes"],
                    handyvars.aeo_years)
                tsv_carbon_yrmap = tsv_cost_carb_yrmap(
                    tsv_carbon_data["average carbon emissions rates"],
                    handyvars.aeo_years)
                # Stitch together load shape, cost, emissions, and year
                # mapping datasets
                tsv_data = {
                    "load": tsv_load_data, "price": tsv_cost_data,
                    "price_yr_map": tsv_cost_yrmap,
                    "emissions": tsv_carbon_data,
                    "emissions_yr_map": tsv_carbon_yrmap}
                # Case where the user assesses time sensitive emissions/cost
                # factors for before grid decarbonization for non-fuel
                # switching measures
                if all([x is not None for x in [
                        tsv_cost_nonfs_data, tsv_carbon_nonfs_data]]):
                    tsv_data_nonfs = {
                        "load": tsv_load_data, "price": tsv_cost_nonfs_data,
                        "price_yr_map": tsv_cost_yrmap,
                        "emissions": tsv_carbon_nonfs_data,
                        "emissions_yr_map": tsv_carbon_yrmap}
                else:
                    tsv_data_nonfs = None

        else:
            tsv_data, tsv_data_nonfs = (None for n in range(2))

        print("Complete")

        # Prepare new or edited measures for use in analysis engine
        meas_prepped_objs = prepare_measures(
            meas_toprep_indiv, convert_data, msegs, msegs_cpl, handyvars,
            handyfiles, cbecs_sf_byvint, tsv_data, base_dir, opts,
            ctrb_ms_pkg_prep, tsv_data_nonfs)

        # Prepare measure packages for use in analysis engine (if needed)
        if meas_toprep_package:
            meas_prepped_objs = prepare_packages(
                meas_toprep_package, meas_prepped_objs, meas_summary,
                handyvars, handyfiles, base_dir, opts, convert_data)

        print("All ECM updates complete; finalizing data...",
              end="", flush=True)
        # Split prepared measure data into subsets needed to set high-level
        # measure attributes information and to execute measure competition
        # in the analysis engine
        meas_prepped_compete, meas_prepped_summary, meas_prepped_shapes, \
            meas_eff_fs_splt = split_clean_data(meas_prepped_objs)

        # Add all prepared high-level measure information to existing
        # high-level data and to list of active measures for analysis;
        # ensure that high-level data for measures that contribute to
        # packages are not written out, with the exception of HVAC
        # measures in a package that the user has requested be written
        # out for eventual competition with the packages they contribute to
        for m_i, m in enumerate([x for x in meas_prepped_summary]):
            is_in_package = m["name"] in ctrb_ms_pkg_prep
            is_supply_tech = m["technology_type"]["primary"][0] == "supply"
            # Measure does not serve as counterfactual for isolating
            # envelope impacts within packages
            if "(CF)" not in m["name"] and (
                    not is_in_package or (opts.pkg_env_costs == '1' and is_supply_tech)):
                # Measure has been prepared from existing case (replace
                # high-level data for measure)
                if m["name"] in [x["name"] for x in meas_summary]:
                    [x.update(m) for x in meas_summary if x["name"] == m["name"]]
                # Measure is new (add high-level data for measure)
                else:
                    meas_summary.append(m)
                # Repeat for sector shapes, if applicable; exclude sector
                # shapes for individual measures that are part of packages
                if opts.sect_shapes is True:
                    # Shorthand for measure sector shapes data object
                    m_ss = meas_prepped_shapes[m_i]
                    if len(m_ss.keys()) != 0:
                        # Measure has been prepared from existing case (replace
                        # high-level data for measure)
                        if m_ss["name"] in [x["name"] for x in meas_shapes]:
                            [x.update(m_ss) for x in meas_shapes if
                             x["name"] == m_ss["name"]]
                        # Measure is new (add high-level data for measure)
                        else:
                            meas_shapes.append(m_ss)
                # Remove measures from active list; when public health costs are assumed, only
                # the "high" health costs versions of prepared measures remain active
                if opts.health_costs is True and "PHC-EE (high)" not in m["name"]:
                    run_setup = update_active_measures(run_setup, to_inactive=[m["name"]])
            # Measure serves as counterfactual for isolating envelope impacts
            # within packages; append data to separate list, which will
            # be written to a separate ecm_prep file
            elif opts.pkg_env_sep is True and (
                    not is_in_package or (opts.pkg_env_costs == '1' and is_supply_tech)):
                # Measure has been prepared from existing case (replace
                # high-level data for measure)
                if m["name"] in [x["name"] for x in meas_summary_env_cf]:
                    [x.update(m) for x in meas_summary_env_cf if
                     x["name"] == m["name"]]
                # Measure is new (add high-level data for measure)
                else:
                    meas_summary_env_cf.append(m)
                    # Repeat for sector shapes, if applicable; exclude sector
                    # shapes for individual measures that are part of packages
                    if opts.sect_shapes is True:
                        # Shorthand for measure sector shapes data object
                        m_ss = meas_prepped_shapes[m_i]
                        if len(m_ss.keys()) != 0:
                            meas_shapes_env_cf.append(m_ss)

        # Notify user that all measure preparations are completed
        print('Writing output data...')

        # Write prepared measure competition data and (if applicable) efficient
        # fuel switching splits by microsegment to zipped JSONs
        for ind, m in enumerate(meas_prepped_objs):
            # Ensure that competed data is not written out for
            # counterfactual measures or measures that contribute to
            # packages, with the exception of HVAC measures in a package that
            # the user has requested be written out for eventual competition
            # with the packages they contribute to
            if "(CF)" not in m.name and (
                    m.name not in ctrb_ms_pkg_prep or (
                    opts.pkg_env_costs == '1' and
                    m.technology_type["primary"][0] == "supply")):
                # Assemble file name for measure competition data
                meas_file_name = m.name + ".pkl.gz"
                # Assemble folder path for measure competition data
                comp_folder_name = handyfiles.ecm_compete_data
                with gzip.open(comp_folder_name / meas_file_name, 'w') as zp:
                    pickle.dump(meas_prepped_compete[ind], zp, -1)
                if len(meas_eff_fs_splt[ind].keys()) != 0:
                    # Assemble path for measure efficient fs split data
                    fs_splt_folder_name = handyfiles.ecm_eff_fs_splt_data
                    with gzip.open(fs_splt_folder_name / meas_file_name, 'w') as zp:
                        pickle.dump(meas_eff_fs_splt[ind], zp, -1)
        # Write prepared high-level measure attributes data to JSON
        with open(handyfiles.ecm_prep, "w") as jso:
            json.dump(meas_summary, jso, indent=2, cls=MyEncoder)
        # If applicable, write sector shape data to JSON
        if opts.sect_shapes is True:
            with open(handyfiles.ecm_prep_shapes, "w") as jso:
                json.dump(meas_shapes, jso, indent=2, cls=MyEncoder)

        # Write prepared high-level counterfactual measure attributes data to
        # JSON (e.g., a separate file with data that will be used to isolate
        # the effects of envelope within envelope/HVAC packages)
        if opts is not None and opts.pkg_env_sep is True and \
                meas_summary_env_cf is not None:
            with open(handyfiles.ecm_prep_env_cf, "w") as jso:
                json.dump(meas_summary_env_cf, jso, indent=2, cls=MyEncoder)
            # If applicable, write out envelope counterfactual sector shapes
            if opts.sect_shapes is True:
                with open(handyfiles.ecm_prep_env_cf_shapes, "w") as jso:
                    json.dump(meas_shapes_env_cf, jso, indent=2, cls=MyEncoder)

        # Write metadata for consistent use later in the analysis engine
        with open(handyfiles.glob_vars, "w") as jso:
            glob_vars = {
                "adopt_schemes": handyvars.adopt_schemes,
                "retro_rate": handyvars.retro_rate,
                "aeo_years": handyvars.aeo_years,
                "discount_rate": handyvars.discount_rate,
                "out_break_czones": handyvars.out_break_czones,
                "out_break_bldg_types": handyvars.out_break_bldgtypes,
                "out_break_enduses": handyvars.out_break_enduses,
                "out_break_fuels": handyvars.out_break_fuels,
                "out_break_eus_w_fsplits": handyvars.out_break_eus_w_fsplits
            }
            json.dump(glob_vars, jso, indent=2, cls=MyEncoder)
    else:
        print('No new ECM updates available')

    # Write lists of active/inactive measures to be used in the analysis engine
    with open(handyfiles.run_setup, "w") as jso:
        json.dump(run_setup, jso, indent=2)


if __name__ == "__main__":
    import time
    start_time = time.time()
    opts = ecm_args()

    main(opts)
    hours, rem = divmod(time.time() - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("--- Runtime: %s (HH:MM:SS.mm) ---" %
          "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
