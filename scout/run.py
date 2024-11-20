#!/usr/bin/env python3
from __future__ import annotations
import json
import numpy
import copy
from numpy.linalg import LinAlgError
from collections import OrderedDict, defaultdict
import gzip
import pickle
from ast import literal_eval
import math
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy_financial as npf
from datetime import datetime
from scout.plots import run_plot
from scout.config import FilePaths as fp
from scout.config import Config
import warnings


class UsefulInputFiles(object):
    """Class of input files to be opened by this routine.

    Attributes:
        glob_vars (str) = Global run settings from ecm_prep.
        msegs_in (JSON): Database of baseline microsegment stock/energy.
        meas_summary_data (tuple): High-level measure summary data.
        meas_compete_data (tuple): Contributing microsegment data needed
            for measure competition.
        meas_eff_fs_splt_data (tuple): Folder with data needed to determine the
            fuel splits of efficient case results for fuel switching measures.
        active_measures (str): Measures that are active for the analysis.
        meas_engine_out_ecms (tuple): Individual measure output summaries.
        meas_engine_out_agg (tuple): Portfolio output summaries.
        comp_fracs_out (tuple): Competition adjustment fractions (if required)
        cpi_data (tuple). Consumer Price Index (CPI) data.
        htcl_totals (tuple): Heating/cooling energy totals by climate zone,
            building type, and structure type.
    """

    def __init__(self, energy_out, regions, grid_decarb):
        self.glob_vars = fp.GENERATED / "glob_run_vars.json"
        self.meas_summary_data = fp.GENERATED / "ecm_prep.json"
        self.meas_compete_data = fp.ECM_COMP
        self.meas_eff_fs_splt_data = fp.EFF_FS_SPLIT
        self.active_measures = fp.GENERATED / "run_setup.json"
        self.meas_engine_out_ecms = fp.RESULTS / "ecm_results.json"
        self.meas_engine_out_agg = fp.RESULTS / "agg_results.json"
        self.comp_fracs_out = fp.RESULTS / "comp_fracs.json"
        self.cpi_data = fp.CONVERT_DATA / "cpi.csv"
        # Set heating/cooling energy totals file conditional on: 1) regional
        # breakout used, and 2) whether site energy data, source energy data
        # (fossil equivalent site-source conversion), or source energy data
        # (captured energy site-source conversion) are needed
        if regions == "AIA":
            self.msegs_in = fp.STOCK_ENERGY / "mseg_res_com_cz.json"
            if energy_out[0] == "site":
                self.htcl_totals = fp.STOCK_ENERGY / "htcl_totals-site.json"
            elif energy_out[0] == "fossil_equivalent":
                # Further condition the file based on whether a high grid
                # decarb case has been selected by the user
                if grid_decarb is True:
                    self.htcl_totals = fp.STOCK_ENERGY / "htcl_totals_decarb.json"
                else:
                    self.htcl_totals = fp.STOCK_ENERGY / "htcl_totals.json"
            elif energy_out[0] == "captured":
                self.htcl_totals = fp.STOCK_ENERGY / "htcl_totals-ce.json"
            else:
                raise ValueError(
                    "Unsupported user option type (site, source "
                    "(fossil fuel equivalent), and source (captured "
                    "energy) are currently supported)")
        elif regions == "EMM":
            self.msegs_in = fp.STOCK_ENERGY / "mseg_res_com_emm.gz"
            if energy_out[0] == "site":
                self.htcl_totals = fp.STOCK_ENERGY / "htcl_totals-site_emm.json"
            elif energy_out[0] == "fossil_equivalent":
                # Further condition the file based on whether a high grid
                # decarb case has been selected by the user
                if grid_decarb is True:
                    self.htcl_totals = fp.STOCK_ENERGY / "htcl_totals_emm_decarb.json"
                else:
                    self.htcl_totals = fp.STOCK_ENERGY / "htcl_totals_emm.json"
            elif energy_out[0] == "captured":
                self.htcl_totals = fp.STOCK_ENERGY / "htcl_totals-ce_emm.json"
            else:
                raise ValueError(
                    "Unsupported user option type (site, source "
                    "(fossil fuel equivalent), and source (captured "
                    "energy) are currently supported)")
        elif regions == "State":
            self.msegs_in = fp.STOCK_ENERGY / "mseg_res_com_state.gz"
            if energy_out[0] == "site":
                self.htcl_totals = fp.STOCK_ENERGY / "htcl_totals-site_state.json"
            elif energy_out[0] == "fossil_equivalent":
                self.htcl_totals = fp.STOCK_ENERGY / "htcl_totals_state.json"
            elif energy_out[0] == "captured":
                self.htcl_totals = fp.STOCK_ENERGY / "htcl_totals-ce_state.json"
            else:
                raise ValueError(
                    "Unsupported user option type (site, source "
                    "(fossil fuel equivalent), and source (captured "
                    "energy) are currently supported)")
        else:
            raise ValueError("Unsupported regional breakout (" + regions + ")")
        # Use the user-specified grid_decarb flag and energy
        # calculation method to determine which site-source
        # conversions to select
        if grid_decarb is not False:
            self.ss_data = fp.CONVERT_DATA / "site_source_co2_conversions-100by2035.json"
        elif energy_out[0] == "captured":
            self.ss_data = fp.CONVERT_DATA / "site_source_co2_conversions-ce.json"
        else:
            self.ss_data = fp.CONVERT_DATA / "site_source_co2_conversions.json"
        # Use the user-specified grid_decarb flag and region selection
        # to select the correct electricity price and CO2 intensity data
        if regions == 'EMM':
            if grid_decarb is not False:
                self.elec_price_co2 = fp.CONVERT_DATA / "emm_region_emissions_prices-100by2035.json"
            else:
                self.elec_price_co2 = fp.CONVERT_DATA / "emm_region_emissions_prices.json"
        elif regions == 'State':
            self.elec_price_co2 = fp.CONVERT_DATA / "state_emissions_prices.json"
        else:
            if grid_decarb is not False:
                self.elec_price_co2 = fp.CONVERT_DATA / "site_source_co2_conversions-100by2035.json"
            else:
                if energy_out[0] == 'captured':
                    self.elec_price_co2 = fp.CONVERT_DATA / "site_source_co2_conversions-ce.json"
                else:
                    self.elec_price_co2 = fp.CONVERT_DATA / "site_source_co2_conversions.json"


class UsefulVars(object):
    """Class of variables that are used globally across functions.

    Attributes:
        adopt_schemes (list): Possible consumer adoption scenarios.
        retro_rate (float): Rate at which existing stock is retrofitted.
        aeo_years (list) = Modeling time horizon.
        discount_rate (float): General rate to use in discounting cash flows.
        com_timeprefs (dict): Time preference premiums for commercial adopters.
        out_break_czones (OrderedDict): Maps measure climate zone names to
            the climate zone categories used in summarizing measure outputs.
        out_break_bldgtypes (OrderedDict): Maps measure building type names to
            the building sector categories used in summarizing measure outputs.
        out_break_enduses (OrderedDict): Maps measure end use names to
            the end use categories used in summarizing measure outputs.
        out_break_fuels (OrderedDict): Maps measure fuel types to fuel type
            categories used in summarizing measure outputs.
        regions (str): Regions to use in geographically breaking out the data.
        region_inout_namepairs (dict): Input/output region name pairs.
        common_cost_yr (str) = Common year for all cost calculations.
        cost_convert (dict) = Conversions between cost year for various cost
            metrics (stock, energy, carbon) and common cost year.
    """

    def __init__(self, handyfiles):
        # Pull in global variable settings from ecm_prep
        with open(handyfiles.glob_vars, 'r') as gv:
            try:
                gvars = json.load(gv)
            except ValueError:
                raise ValueError(f"Error reading in '{handyfiles.glob_vars}'")
        self.adopt_schemes = gvars["adopt_schemes"]
        self.retro_rate = gvars["retro_rate"]
        self.aeo_years = gvars["aeo_years"]
        self.discount_rate = gvars["discount_rate"]
        self.out_break_czones = gvars["out_break_czones"]
        self.out_break_bldgtypes = gvars["out_break_bldg_types"]
        self.out_break_enduses = gvars["out_break_enduses"]
        self.out_break_fuels = gvars["out_break_fuels"]
        self.out_break_eus_w_fsplits = gvars["out_break_eus_w_fsplits"]
        # Set commercial time prefs and region in/out name pairs as unique
        # attributes for the UsefulVars class in run.py
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
        self.region_inout_namepairs = {
            "AIA": [
                ('AIA CZ1', 'AIA_CZ1'), ('AIA CZ2', 'AIA_CZ2'),
                ('AIA CZ3', 'AIA_CZ3'), ('AIA CZ4', 'AIA_CZ4'),
                ('AIA CZ5', 'AIA_CZ5')],
            "EMM": [(x, x) for x in [
                'TRE', 'FRCC', 'MISW', 'MISC', 'MISE', 'MISS',
                'ISNE', 'NYCW', 'NYUP', 'PJME', 'PJMW', 'PJMC',
                'PJMD', 'SRCA', 'SRSE', 'SRCE', 'SPPS', 'SPPC',
                'SPPN', 'SRSG', 'CANO', 'CASO', 'NWPP', 'RMRG', 'BASN']],
            "State": [(x, x) for x in [
                'AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL',
                'GA', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME',
                'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH',
                'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI',
                'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI',
                'WY']]}
        # Import CPI data to use in cost conversions
        try:
            cpi = numpy.genfromtxt(
                handyfiles.cpi_data,
                names=True, delimiter=',',
                dtype=[('DATE', 'U10'), ('VALUE', '<f8')])
            # Ensure that consumer price date is in expected format
            if len(cpi['DATE'][0]) != 10:
                raise ValueError("CPI date format should be YYYY-MM-DD")
        except ValueError as e:
            raise ValueError(
                "Error reading in '" +
                handyfiles.cpi_data + "': " + str(e)) from None
        # Years of the baseline stock cost, energy cost, and carbon cost data
        # *** Note: could eventually be pulled from baseline data files
        # rather than hardcoded ***
        cost_yrs = {"stock": "2017", "energy": "2021", "carbon": "2020"}
        # Desired common cost year; use year previous to current one to
        # ensure that full year of CPI data is available for averaging
        self.common_cost_yr = str(datetime.today().year - 1)
        # Initialize dict of conversions to reach common cost basis
        self.cost_convert = {key: None for key in cost_yrs.keys()}
        # Find array of rows in CPI dataset associated with current year
        cpi_row_cmn = [
            x[1] for x in cpi if self.common_cost_yr in x['DATE']]
        # If year is not found in CPI data, default to last available
        # CPI index value in the dataset; otherwise, average across all
        # values for that year
        if len(cpi_row_cmn) == 0:
            cpi_row_cmn = cpi[-1][1]
        else:
            cpi_row_cmn = numpy.mean(cpi_row_cmn)
        # Loop through each metric and develop conversions of costs between
        # year of costs for given metric and common cost year developed above
        for metr in cost_yrs.keys():
            # Find array of rows in CPI dataset associated with the metric
            # cost year
            cpi_row_metr = [x[1] for x in cpi if cost_yrs[metr] in x['DATE']]
            # If year is not found in CPI data, default to last
            # available CPI index value in the dataset; otherwise, average
            # across all values for that year
            if len(cpi_row_metr) == 0:
                cpi_row_metr = cpi[-1][1]
            else:
                cpi_row_metr = numpy.mean(cpi_row_metr)
            # Calculate ratio of metric year CPI index to common year CPI index
            self.cost_convert[metr] = cpi_row_cmn / cpi_row_metr


class Measure(object):
    """Class representing individual efficiency measures.

    Attributes:
        **kwargs: Arbitrary keyword arguments used to fill measure attributes
            from an input dictionary.
        update_results (dict): Flags markets, savings, and financial metric
            outputs that have yet to be finalized by the analysis engine.
        eff_fs_splt (tuple): Data needed to determine the fuel splits of
            efficient case results for fuel switching measures.
        markets (dict): Data grouped by adoption scheme on:
            a) 'master_mseg': a measure's master market microsegments (stock,
               energy, carbon, cost),
            b) 'mseg_adjust': all microsegments that contribute to each master
               microsegment (required later for measure competition).
            c) 'mseg_out_break': master microsegment breakdowns by key
               variables (e.g., climate zone, building type, end use, etc.)
        savings (dict): Energy, carbon, and stock, energy, and carbon cost
            savings for measure over baseline technology case.
        financial_metrics (dict): Measure financial metrics.
    """

    def __init__(self, handyvars, **kwargs):
        # Read Measure object attributes from measures input JSON
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.savings, self.financial_metrics = ({} for n in range(2))
        self.update_results = {"savings": {}, "financial metrics": True}
        self.eff_fs_splt = {}
        # Determine whether fugitive emissions should be assessed for the
        # measure based on usr_opts attribute; handle case where this
        # attribute is not present
        try:
            if self.usr_opts["fugitive_emissions"] is not False and isinstance(
                    self.usr_opts["fugitive_emissions"], list):
                # Methane only
                if self.usr_opts["fugitive_emissions"][0] == '1':
                    self.fug_e = ["methane"]
                # Refrigerants only
                elif self.usr_opts["fugitive_emissions"][0] == '2':
                    self.fug_e = ["refrigerants"]
                # Methane and refrigerants
                elif self.usr_opts["fugitive_emissions"][0] == '3':
                    self.fug_e = ["methane", "refrigerants"]
            else:
                self.fug_e = ""
        except (AttributeError, KeyError):
            self.fug_e = ""
        # Convert any master market microsegment data formatted as lists to
        # numpy arrays
        self.convert_to_numpy(self.markets)
        # Pull high-level markets data for adoption schemes; high-level
        # technical potential (TP) market data will be available and should be
        # pulled in here even if a user has declined to simulate this scheme;
        # TP data are used to set unit-level metrics in the competition
        if "Technical potential" not in handyvars.adopt_schemes:
            adopt_schemes_highlvl_mkts = handyvars.adopt_schemes + \
                ["Technical potential"]
        else:
            adopt_schemes_highlvl_mkts = handyvars.adopt_schemes
        for adopt_scheme in adopt_schemes_highlvl_mkts:
            # Initialize 'uncompeted' and 'competed' versions of
            # Measure markets (initially, they are identical)
            self.markets[adopt_scheme] = {
                "uncompeted": copy.deepcopy(self.markets[adopt_scheme]),
                "competed": copy.deepcopy(self.markets[adopt_scheme])}
            self.update_results["savings"][
                adopt_scheme] = {"uncompeted": True, "competed": True}
            self.savings[adopt_scheme] = {
                "uncompeted": {
                    "stock": {
                        "cost savings": None},
                    "energy": {
                        "savings": None,
                        "cost savings": None},
                    "carbon": {
                        "savings": None,
                        "cost savings": None}
                    },
                "competed": {
                    "stock": {
                        "cost savings": None},
                    "energy": {
                        "savings": None,
                        "cost savings": None},
                    "carbon": {
                        "savings": None,
                        "cost savings": None}
                    }}
            # Append a key to the savings dict for fugitive emissions data in
            # the case where those data are being assessed for the measure
            if self.fug_e:
                for met in ["uncompeted", "competed"]:
                    self.savings[adopt_scheme][met]["fugitive emissions"] = {
                        "methane": {"savings": None},
                        "refrigerants": {"savings": None}}
            self.financial_metrics = {
                "unit cost": {
                    "stock cost": {
                        "residential": None,
                        "commercial": None
                    },
                    "energy cost": {
                        "residential": None,
                        "commercial": None
                    },
                    "carbon cost": {
                        "residential": None,
                        "commercial": None
                    }},
                "irr (w/ energy costs)": None,
                "irr (w/ energy and carbon costs)": None,
                "payback (w/ energy costs)": None,
                "payback (w/ energy and carbon costs)": None,
                "cce": None,
                "cce (w/ carbon cost benefits)": None,
                "ccc": None,
                "ccc (w/ energy cost benefits)": None
                }

    def convert_to_numpy(self, markets):
        """Convert terminal/leaf node lists in a dict to numpy arrays.

        Args:
            markets (dict): Input dict with possible lists at terminal/leaf
                nodes.
        """
        for k, i in markets.items():
            if isinstance(i, dict):
                self.convert_to_numpy(i)
            else:
                if isinstance(markets[k], list):
                    markets[k] = numpy.array(markets[k])


class Engine(object):
    """Class representing a collection of efficiency measures.

    Attributes:
        handyvars (object): Global variables useful across class methods.
        measures (list): List of active measure objects to be analyzed.
        output_ecms (OrderedDict): Summary results by active measure.
        output_all (OrderedDict): Summary results across all active measures;
            also stores data on energy output type (site, source (fossil
            equivalent site-source) or source (captured energy site-source).
    """

    def __init__(self, handyvars, opts, measure_objects, energy_out, brkout):
        self.handyvars = handyvars
        self.opts = opts
        self.measures = measure_objects
        self.output_ecms, self.output_all = (OrderedDict() for n in range(2))
        self.output_all["All ECMs"] = OrderedDict([
            ("Markets and Savings (Overall)", OrderedDict())])
        self.output_all["Energy Output Type"] = energy_out
        self.output_all["Output Resolution"] = brkout
        # Initialize competition adjustment fraction dict, if required by user
        if self.opts.report_cfs is True:
            self.output_ecms_cfs = {}
        else:
            self.output_ecms_cfs = None
        for adopt_scheme in self.handyvars.adopt_schemes:
            self.output_all["All ECMs"]["Markets and Savings (Overall)"][
                adopt_scheme] = OrderedDict()
        for m in self.measures:
            # Fill dict with competition adjustment data if needed
            if self.output_ecms_cfs is not None:
                # Organize data by measure name and by variable type – energy,
                # carbon, energy costs, and stock. Energy, carbon, and energy
                # costs will be further resolved by result type (baseline,
                # efficient, and savings) as well as region.
                self.output_ecms_cfs[m.name] = {
                    key: {mt: {
                        reg: 0 for reg in self.handyvars.out_break_czones}
                        for mt in [
                        "baseline", "efficient", "savings"]} for key in [
                        "energy", "carbon", "cost"]}
                self.output_ecms_cfs[m.name]["stock"] = {"measure": None}
            # Set measure climate zone, building sector, and end use
            # output category names for use in filtering and/or breaking
            # out results
            czones, bldgtypes, end_uses = ([] for n in range(3))
            # Find measure climate zone output categories
            for cz in self.handyvars.out_break_czones.items():
                if any([x in cz[1] for x in m.climate_zone]) and \
                        cz[0] not in czones:
                    czones.append(cz[0])
            # Find measure building sector output categories
            for bldg in self.handyvars.out_break_bldgtypes.items():
                if any([x in bldg[1] for x in m.bldg_type]) and \
                        bldg[0] not in bldgtypes:
                    bldgtypes.append(bldg[0])
            # Find measure end use output categories
            for euse in self.handyvars.out_break_enduses.items():
                # Find primary end use categories
                if any([x in euse[1] for x in m.end_use["primary"]]) and \
                        euse[0] not in end_uses:
                    # * Note: classify special freezers ECM case as
                    # 'Refrigeration'; classify 'supply' side heating/cooling
                    # ECMs as 'Heating (Equip.)'/'Cooling (Equip.)' and
                    # 'demand' side heating/cooling ECMs as 'Envelope'
                    if (euse[0] == "Refrigeration" and
                        ("refrigeration" in m.end_use["primary"] or
                         "freezers" in m.technology)) or (
                        euse[0] != "Refrigeration" and ((
                            euse[0] in ["Heating (Equip.)",
                                        "Cooling (Equip.)"] and
                            "supply" in m.technology_type["primary"]) or (
                            euse[0] in ["Heating (Env.)", "Cooling (Env.)"] and
                            "demand" in m.technology_type["primary"]) or (
                            euse[0] not in [
                                "Heating (Equip.)", "Cooling (Equip.)",
                                "Heating (Env.)", "Cooling (Env.)"]))):
                        end_uses.append(euse[0])
                # Assign secondary heating/cooling microsegments that
                # represent waste heat from lights to the 'Lighting' end use
                # category
                if m.end_use["secondary"] is not None and any([
                    x in m.end_use["secondary"] for x in [
                        "heating", "cooling"]]) and "Lighting" not in end_uses:
                    end_uses.append("Lighting")

            # Set measure climate zone(s), building sector(s), and end use(s)
            # as filter variables
            self.output_ecms[m.name] = OrderedDict([
                ("Filter Variables", OrderedDict([
                    ("Applicable Regions", czones),
                    ("Applicable Building Classes", bldgtypes),
                    ("Applicable End Uses", end_uses)])),
                ("Markets and Savings (Overall)", OrderedDict()),
                ("Markets and Savings (by Category)", OrderedDict()),
                ("Financial Metrics", OrderedDict())])
            for adopt_scheme in self.handyvars.adopt_schemes:
                # Initialize measure overall markets and savings
                self.output_ecms[m.name]["Markets and Savings (Overall)"][
                    adopt_scheme] = OrderedDict()
                # Initialize measure markets and savings broken out by climate
                # zone, building sector, and end use categories
                self.output_ecms[m.name]["Markets and Savings (by Category)"][
                    adopt_scheme] = OrderedDict()
                # Initialize measure financial metrics
                self.output_ecms[m.name]["Financial Metrics"] = OrderedDict()

    def calc_savings_metrics(self, adopt_scheme, comp_scheme):
        """Calculate and update measure savings and financial metrics.

        Notes:
            Given information on measure master microsegments for
            each projection year, determine capital, energy, and carbon
            cost savings; energy and carbon savings; and the net present
            value, internal rate of return, simple payback, cost of
            conserved energy, and cost of conserved carbon for the measure.

        Args:
            adopt_scheme (string): Assumed consumer adoption scenario.
            comp_scheme (string): Assumed measure competition scenario.
        """
        # Find all active measures that require savings updates
        measures_update = [m for m in self.measures if m.update_results[
            "savings"][adopt_scheme][comp_scheme] is True]

        # Update measure savings and associated financial metrics
        for m in measures_update:
            # Initialize energy/energy cost savings, carbon/
            # carbon cost savings dicts
            scostsave_tot, esave_tot, ecostsave_tot, csave_tot, \
                ccostsave_tot = ({
                    yr: None for yr in self.handyvars.aeo_years} for
                    n in range(5))
            # Initialize methane savings dict if fugitive emissions from
            # methane leaks are being assessed for the measure
            if m.fug_e and "methane" in m.fug_e:
                meth_save_tot = {
                    yr: None for yr in self.handyvars.aeo_years}
            else:
                meth_save_tot = ""
            # Initialize refrigerants savings dict if fugitive emissions from
            # refrigerant leaks are being assessed for the measure
            if m.fug_e and "refrigerants" in m.fug_e:
                refr_save_tot = {
                    yr: None for yr in self.handyvars.aeo_years}
            else:
                refr_save_tot = ""
            # Shorthand for data used to determine uncompeted and competed
            # savings by adoption scheme
            markets_save = m.markets[adopt_scheme][comp_scheme]["master_mseg"]

            # Calculate measure energy/carbon savings, capital cost savings,
            # and energy/carbon cost savings for each projection year
            for yr in self.handyvars.aeo_years:
                # Calculate total annual energy/carbon and capital/energy/
                # carbon cost savings for the measure vs. baseline. Total
                # savings reflect the impact of all measure adoptions
                # simulated up until and including the current year
                esave_tot[yr] = \
                    markets_save["energy"]["total"]["baseline"][yr] - \
                    markets_save["energy"]["total"]["efficient"][yr]
                csave_tot[yr] = \
                    markets_save["carbon"]["total"]["baseline"][yr] - \
                    markets_save["carbon"]["total"]["efficient"][yr]
                # Note: convert stock, energy, and carbon costs to common
                # year dollars
                scostsave_tot[yr] = (
                    markets_save["cost"]["stock"]["total"]["baseline"][yr] -
                    markets_save["cost"]["stock"]["total"]["efficient"][yr]
                    ) * self.handyvars.cost_convert["stock"]
                ecostsave_tot[yr] = (
                    markets_save["cost"]["energy"]["total"]["baseline"][yr] -
                    markets_save["cost"]["energy"]["total"]["efficient"][yr]
                    ) * self.handyvars.cost_convert["energy"]
                ccostsave_tot[yr] = (
                    markets_save["cost"]["carbon"]["total"]["baseline"][yr] -
                    markets_save["cost"]["carbon"]["total"]["efficient"][yr]
                    ) * self.handyvars.cost_convert["carbon"]
                # Calculate fugitive methane emissions savings if applicable
                if meth_save_tot:
                    meth_save_tot[yr] = \
                        markets_save["fugitive emissions"]["methane"][
                            "total"]["baseline"][yr] - \
                        markets_save["fugitive emissions"]["methane"][
                            "total"]["efficient"][yr]
                # Calculate fugitive refrigerant emissions savings if
                # applicable
                if refr_save_tot:
                    refr_save_tot[yr] = \
                        markets_save["fugitive emissions"][
                            "refrigerants"]["total"]["baseline"][yr] - \
                        markets_save["fugitive emissions"][
                            "refrigerants"]["total"]["efficient"][yr]

            # Record final measure savings figures (across all years)

            # Set measure savings dict to update
            save = m.savings[adopt_scheme][comp_scheme]
            # Update capital cost savings
            save["stock"]["cost savings"] = scostsave_tot
            # Update energy and energy cost savings
            save["energy"]["savings"] = esave_tot
            save["energy"]["cost savings"] = ecostsave_tot
            # Update carbon and carbon cost savings
            save["carbon"]["savings"] = csave_tot
            save["carbon"]["cost savings"] = ccostsave_tot
            # Update fugitive methane emissions savings if applicable
            if meth_save_tot:
                save["fugitive emissions"][
                    "methane"]["savings"] = meth_save_tot
            # Update fugitive refrigerant emissions savings if applicable
            if refr_save_tot:
                save["fugitive emissions"][
                    "refrigerants"]["savings"] = refr_save_tot

            # Set measure savings for the current adoption and competition
            # schemes to finalized status
            m.update_results["savings"][adopt_scheme][comp_scheme] = False

            # Update measure financial metrics if they have not already been
            # finalized (these metrics remain constant across
            # all consumer adoption and measure competition schemes)
            if m.update_results["financial metrics"] is True:
                # Shorthand for data used to determine financial metrics at the
                # unit-level (since metrics at this level do not vary based on
                # competition or adoption dynamics, use only uncompeted
                # technical potential data for the calculations)
                markets_uc = m.markets["Technical potential"]["uncompeted"][
                    "master_mseg"]

                # Initialize per unit measure stock, energy, and carbon costs;
                # per unit energy and carbon cost savings; per unit energy and
                # carbon savings; unit stock, energy, and carbon costs to use
                # in residential and commercial competition calculations; and
                # financial metrics (irr, payback, cce, ccc)
                scostbase_unit, scostmeas_delt_unit, scostmeas_unit, \
                    ecost_meas_unit, ccost_meas_unit, \
                    ecostsave_unit, ccostsave_unit, esave_unit, \
                    csave_unit, stock_unit_cost_res, stock_unit_cost_com, \
                    energy_unit_cost_res, energy_unit_cost_com, \
                    carb_unit_cost_res, carb_unit_cost_com, irr_e, irr_ec, \
                    payback_e, payback_ec, cce, cce_bens, ccc, ccc_bens = ({
                        yr: None for yr in self.handyvars.aeo_years} for
                        n in range(23))

                # Calculate per unit stock costs, energy and carbon savings,
                # and energy and carbon cost savings for each projection year;
                # base calculations on competed stock in each year
                for yr in self.handyvars.aeo_years:

                    # Baseline capital cost
                    stock_base_cost_cmp = \
                        markets_uc["cost"]["stock"]["competed"]["baseline"][yr]
                    # Measure capital cost
                    stock_meas_cost_cmp = markets_uc["cost"]["stock"][
                        "competed"]["efficient"][yr]
                    # Energy savings
                    esave_cmp = \
                        markets_uc["energy"]["competed"]["baseline"][yr] - \
                        markets_uc["energy"]["competed"]["efficient"][yr]
                    # Carbon savings
                    csave_cmp = \
                        markets_uc["carbon"]["competed"]["baseline"][yr] - \
                        markets_uc["carbon"]["competed"]["efficient"][yr]
                    # Baseline energy costs
                    ecost_base_cmp = markets_uc["cost"]["energy"]["competed"][
                        "baseline"][yr]
                    # Measure energy cost
                    ecost_meas_cmp = markets_uc["cost"]["energy"][
                        "competed"]["efficient"][yr]
                    # Baseline carbon costs
                    ccost_base_cmp = markets_uc["cost"]["carbon"]["competed"][
                        "baseline"][yr]
                    # Measure carbon cost
                    ccost_meas_cmp = markets_uc["cost"]["carbon"][
                        "competed"]["efficient"][yr]
                    # Energy cost savings
                    ecostsave_cmp = ecost_base_cmp - ecost_meas_cmp
                    # Carbon cost savings
                    ccostsave_cmp = ccost_base_cmp - ccost_meas_cmp
                    # Number of applicable baseline stock units
                    nunits_cmp = \
                        markets_uc["stock"]["competed"]["all"][yr]
                    # Number of applicable stock units capt. by measure
                    nunits_meas_cmp = \
                        markets_uc["stock"]["competed"]["measure"][yr]

                    # Calculate per unit baseline capital cost and incremental
                    # measure capital cost (used in financial metrics
                    # calculations below); set these values to zero for
                    # years in which total number of base/meas units is zero
                    if nunits_cmp != 0 and (
                        not isinstance(nunits_meas_cmp, numpy.ndarray) and
                        nunits_meas_cmp != 0 or
                            isinstance(nunits_meas_cmp, numpy.ndarray) and all(
                                nunits_meas_cmp) != 0):
                        # Per unit baseline capital cost; note that these costs
                        # are aggregated as a baseline counterfactual for all
                        # units captured by the measure and therefore must be
                        # normalized by the number of measure-captured units
                        scostbase_unit[yr] = (
                            stock_base_cost_cmp / nunits_cmp) * \
                            self.handyvars.cost_convert["stock"]
                        # Per unit measure total capital cost
                        scostmeas_unit[yr] = (
                            stock_meas_cost_cmp / nunits_meas_cmp) * \
                            self.handyvars.cost_convert["stock"]
                        # Per unit measure incremental capital cost
                        scostmeas_delt_unit[yr] = (
                            scostbase_unit[yr] - scostmeas_unit[yr])
                        # Per unit measure energy savings
                        esave_unit[yr] = esave_cmp / nunits_meas_cmp
                        # Per unit measure carbon savings
                        csave_unit[yr] = csave_cmp / nunits_meas_cmp
                        # Per unit measure energy costs
                        ecost_meas_unit[yr] = (
                            ecost_meas_cmp / nunits_meas_cmp) * \
                            self.handyvars.cost_convert["energy"]
                        # Per unit measure carbon costs
                        ccost_meas_unit[yr] = (
                            ccost_meas_cmp / nunits_meas_cmp) * \
                            self.handyvars.cost_convert["carbon"]
                        # Per unit measure energy cost savings
                        ecostsave_unit[yr] = (
                            ecostsave_cmp / nunits_meas_cmp) * \
                            self.handyvars.cost_convert["energy"]
                        # Per unit measure carbon cost savings
                        ccostsave_unit[yr] = (
                            ccostsave_cmp / nunits_meas_cmp) * \
                            self.handyvars.cost_convert["carbon"]

                    # Set the lifetime of the baseline technology for
                    # comparison with measure lifetime
                    life_base = markets_uc["lifetime"]["baseline"][yr]
                    # Ensure that baseline lifetime is at least 1 year
                    if isinstance(life_base, numpy.ndarray) and \
                            any(life_base) < 1:
                        life_base[numpy.where(life_base) < 1] = 1
                    elif not isinstance(life_base, numpy.ndarray) and \
                            life_base < 1:
                        life_base = 1
                    # Set lifetime of the measure
                    life_meas = markets_uc["lifetime"]["measure"]
                    # Ensure that measure lifetime is at least 1 year
                    if isinstance(life_meas, numpy.ndarray) and \
                            any(life_meas) < 1:
                        life_meas[numpy.where(life_meas) < 1] = 1
                    elif not isinstance(life_meas, numpy.ndarray) and \
                            life_meas < 1:
                        life_meas = 1

                    # Calculate measure financial metrics

                    # If the total baseline stock is zero or no measure units
                    # have been captured for a given year, set finance metrics
                    # to 999
                    if nunits_cmp == 0 or (
                        not isinstance(nunits_meas_cmp, numpy.ndarray) and
                        nunits_meas_cmp == 0 or
                            isinstance(nunits_meas_cmp, numpy.ndarray) and all(
                                nunits_meas_cmp) == 0):
                        if yr == self.handyvars.aeo_years[0]:
                            stock_unit_cost_res[yr], \
                                energy_unit_cost_res[yr], \
                                carb_unit_cost_res[yr], \
                                stock_unit_cost_com[yr], \
                                energy_unit_cost_com[yr], \
                                carb_unit_cost_com[yr], \
                                irr_e[yr], irr_ec[yr], payback_e[yr], \
                                payback_ec[yr], cce[yr], cce_bens[yr], \
                                ccc[yr], ccc_bens[yr] = [
                                    None for n in range(6)] + [
                                    999 for n in range(8)]
                        else:
                            yr_prev = str(int(yr) - 1)
                            stock_unit_cost_res[yr], \
                                energy_unit_cost_res[yr], \
                                carb_unit_cost_res[yr], \
                                stock_unit_cost_com[yr], \
                                energy_unit_cost_com[yr], \
                                carb_unit_cost_com[yr], \
                                irr_e[yr], irr_ec[yr], \
                                payback_e[yr], payback_ec[yr], \
                                cce[yr], cce_bens[yr], \
                                ccc[yr], ccc_bens[yr] = [x[yr_prev] for x in [
                                    stock_unit_cost_res, energy_unit_cost_res,
                                    carb_unit_cost_res, stock_unit_cost_com,
                                    energy_unit_cost_com, carb_unit_cost_com,
                                    irr_e, irr_ec, payback_e, payback_ec, cce,
                                    cce_bens, ccc, ccc_bens]]
                    # Otherwise, check whether any financial metric calculation
                    # inputs that can be arrays are in fact arrays
                    elif any(isinstance(x, numpy.ndarray) for x in [
                            scostmeas_delt_unit[yr], esave_unit[yr],
                            life_meas]):
                        # Make copies of the above stock, energy/carbon/cost
                        # variables for possible further manipulation below
                        # before using as inputs to "metric update" function
                        scostmeas_delt_unit_tmp, esave_tmp_unit, \
                            ecostsave_tmp_unit, csave_tmp_unit, \
                            ccostsave_tmp_unit, life_meas_tmp, \
                            scost_meas_tmp, ecost_meas_tmp, ccost_meas_tmp = [
                                scostmeas_delt_unit[yr], esave_unit[yr],
                                ecostsave_unit[yr], csave_unit[yr],
                                ccostsave_unit[yr], life_meas,
                                scostmeas_unit[yr],
                                ecost_meas_unit[yr],
                                ccost_meas_unit[yr]]

                        # Ensure consistency in length of all "metric_update"
                        # inputs that can be arrays

                        # Determine the length that any array inputs to
                        # "metric_update" should consistently have
                        len_arr = next((len(item) for item in [
                            scostmeas_delt_unit[yr], esave_unit[yr],
                            life_meas] if isinstance(
                                item, numpy.ndarray)), None)

                        # Ensure all array inputs to "metric_update" are of the
                        # above length

                        # Check capital cost inputs
                        if not isinstance(
                                scostmeas_delt_unit_tmp, numpy.ndarray):
                            scostmeas_delt_unit_tmp = numpy.repeat(
                                scostmeas_delt_unit_tmp, len_arr)
                            scost_meas_tmp = numpy.repeat(
                                scost_meas_tmp, len_arr)
                        # Check energy/energy cost and carbon/cost savings
                        # inputs
                        if not isinstance(
                                esave_tmp_unit, numpy.ndarray):
                            esave_tmp_unit = numpy.repeat(
                                esave_tmp_unit, len_arr)
                            ecostsave_tmp_unit = \
                                numpy.repeat(ecostsave_tmp_unit, len_arr)
                            csave_tmp_unit = numpy.repeat(
                                csave_tmp_unit, len_arr)
                            ccostsave_tmp_unit = \
                                numpy.repeat(ccostsave_tmp_unit, len_arr)
                            ecost_meas_tmp = numpy.repeat(
                                ecost_meas_tmp, len_arr)
                            ccost_meas_tmp = numpy.repeat(
                                ccost_meas_tmp, len_arr)
                        # Check measure lifetime input
                        if not isinstance(life_meas_tmp, numpy.ndarray):
                            life_meas_tmp = numpy.repeat(
                                life_meas_tmp, len_arr)

                        # Initialize numpy arrays for financial metrics outputs
                        stock_unit_cost_res[yr], energy_unit_cost_res[yr], \
                            carb_unit_cost_res[yr], stock_unit_cost_com[yr], \
                            energy_unit_cost_com[yr], carb_unit_cost_com[yr], \
                            irr_e[yr], irr_ec[yr], payback_e[yr], \
                            payback_ec[yr], cce[yr], cce_bens[yr], ccc[yr], \
                            ccc_bens[yr] = (numpy.repeat(None, len(
                                scostmeas_delt_unit_tmp)) for v in range(14))

                        # Run measure energy/carbon/cost savings and lifetime
                        # inputs through "metric_update" function to yield
                        # financial metric outputs. To handle inputs that are
                        # arrays, use a for loop to generate an output for each
                        # input array element one-by-one and append it to the
                        # appropriate output list. Note that lifetime float
                        # values are translated to integers, and all
                        # energy, carbon, and energy/carbon cost savings values
                        # are normalized by total applicable stock units
                        for x in range(0, len(scostmeas_delt_unit_tmp)):
                            stock_unit_cost_res[yr][x], \
                                energy_unit_cost_res[yr][x], \
                                carb_unit_cost_res[yr][x], \
                                stock_unit_cost_com[yr][x], \
                                energy_unit_cost_com[yr][x], \
                                carb_unit_cost_com[yr][x], \
                                irr_e[yr][x], irr_ec[yr][x], \
                                payback_e[yr][x], payback_ec[yr][x], \
                                cce[yr][x], cce_bens[yr][x], ccc[yr][x], \
                                ccc_bens[yr][x] = self.metric_update(
                                    m, int(round(life_base)),
                                    int(round(life_meas_tmp[x])),
                                    scostbase_unit[yr],
                                    scostmeas_delt_unit_tmp[x],
                                    esave_tmp_unit[x], ecostsave_tmp_unit[x],
                                    csave_tmp_unit[x], ccostsave_tmp_unit[x],
                                    scost_meas_tmp[x], ecost_meas_tmp[x],
                                    ccost_meas_tmp[x])
                    else:
                        # Run measure energy/carbon/cost savings and lifetime
                        # inputs through "metric_update" function to yield
                        # financial metric outputs. Note that lifetime float
                        # values are translated to integers, and all
                        # energy, carbon, and energy/carbon cost savings values
                        # are normalized by total applicable stock units
                        stock_unit_cost_res[yr], energy_unit_cost_res[yr], \
                            carb_unit_cost_res[yr], stock_unit_cost_com[yr], \
                            energy_unit_cost_com[yr], carb_unit_cost_com[yr], \
                            irr_e[yr], irr_ec[yr], payback_e[yr], \
                            payback_ec[yr], cce[yr], cce_bens[yr], ccc[yr], \
                            ccc_bens[yr] = \
                            self.metric_update(
                                m, int(round(life_base)),
                                int(round(life_meas)), scostbase_unit[yr],
                                scostmeas_delt_unit[yr], esave_unit[yr],
                                ecostsave_unit[yr], csave_unit[yr],
                                ccostsave_unit[yr],
                                scostmeas_unit[yr],
                                ecost_meas_unit[yr],
                                ccost_meas_unit[yr])

                # Set measure financial metrics dict to update (across years)
                metrics_finance = m.financial_metrics
                # Update unit capital and operating costs
                metrics_finance["unit cost"]["stock cost"]["residential"], \
                    metrics_finance["unit cost"]["stock cost"][
                    "commercial"] = [stock_unit_cost_res, stock_unit_cost_com]
                metrics_finance["unit cost"]["energy cost"]["residential"], \
                    metrics_finance["unit cost"]["energy cost"][
                    "commercial"] = [energy_unit_cost_res,
                                     energy_unit_cost_com]
                metrics_finance["unit cost"]["carbon cost"]["residential"], \
                    metrics_finance["unit cost"]["carbon cost"][
                    "commercial"] = [carb_unit_cost_res, carb_unit_cost_com]
                # Update internal rate of return
                metrics_finance["irr (w/ energy costs)"] = irr_e
                metrics_finance["irr (w/ energy and carbon costs)"] = irr_ec
                # Update payback period
                metrics_finance["payback (w/ energy costs)"] = payback_e
                metrics_finance["payback (w/ energy and carbon costs)"] = \
                    payback_ec
                # Update cost of conserved energy
                metrics_finance["cce"] = cce
                metrics_finance["cce (w/ carbon cost benefits)"] = cce_bens
                # Update cost of conserved carbon
                metrics_finance["ccc"] = ccc
                metrics_finance["ccc (w/ energy cost benefits)"] = ccc_bens

                # Set measure consumer-level metrics to finalized status
                m.update_results["financial metrics"] = False

    def metric_update(self, m, life_base, life_meas, scost_base,
                      scost_meas_delt, esave, ecostsave, csave, ccostsave,
                      scost_meas, ecost_meas, ccost_meas):
        """Calculate measure financial metrics for a given year.

        Notes:
            Calculate internal rate of return, simple payback, and cost of
            conserved energy/carbon from cash flows and energy/carbon
            savings across the measure lifetime. In the cash flows, represent
            the benefits of longer lifetimes for lighting equipment ECMs over
            comparable baseline technologies.

        Args:
            m (object): Measure object.
            nunits (int): Total competed baseline units in a given year.
            nunits_meas (int): Total competed units captured by measure in
                given year.
            life_base (float): Baseline technology lifetime.
            life_meas (float): Measure lifetime.
            scost_base (float): Per unit baseline capital cost in given year.
            scost_meas_delt (float): Per unit incremental capital
                cost for measure over baseline unit in given year.
            esave (float): Per unit annual energy savings over measure
                lifetime, starting in given year.
            ecostsave (float): Per unit annual energy cost savings over
                measure lifetime, starting in a given year.
            csave (float): Per unit annual avoided carbon emissions over
                measure lifetime, starting in given year.
            ccostsave (float): Per unit annual carbon cost savings over
                measure lifetime, starting in a given year.
            scost_meas (float): Per unit measure capital cost in given year.
            ecost_meas (float): Per unit measure energy cost in given year.
            ccost_meas (float): Per unit measure carbon cost in given year.

        Returns:
            Consumer and portfolio-level financial metrics for the given
            measure cost savings inputs.
        """
        # Develop four initial cash flow scenarios over the measure life:
        # 1) Cash flows considering capital costs only
        # 2) Cash flows considering capital costs and energy costs
        # 3) Cash flows considering capital costs and carbon costs
        # 4) Cash flows considering capital, energy, and carbon costs

        # For lighting equipment ECMs only: determine when over the course of
        # the ECM lifetime (if at all) a cost gain is realized from an avoided
        # purchase of the baseline lighting technology due to longer measure
        # lifetime; store this information in a list of year indicators for
        # subsequent use below.  Example: an LED bulb lasts 30 years compared
        # to a baseline bulb's 10 years, meaning 3 purchases of the baseline
        # bulb would have occurred by the time the LED bulb has reached the
        # end of its life.
        added_stockcost_gain_yrs = []
        if (life_meas > life_base) and ("lighting" in m.end_use[
            "primary"]) and (m.measure_type == "full service") and (
                m.technology_type["primary"] == "supply"):
            for i in range(1, life_meas):
                if i % life_base == 0:
                    added_stockcost_gain_yrs.append(i - 1)

        # If the measure lifetime is less than 1 year, set it to 1 year
        # (a minimum for measure lifetime to work in below calculations)
        if life_meas < 1:
            life_meas = 1

        # Construct capital cost cash flows across measure life

        # Initialize incremental and total capital cost cash flows with
        # upfront incremental and total capital cost
        cashflows_s_delt = numpy.array(scost_meas_delt)
        cashflows_s_tot = numpy.array(scost_meas)

        for life_yr in range(0, life_meas):
            # Check whether an avoided cost of the baseline technology should
            # be added for given year; if not, set this term to zero
            if life_yr in added_stockcost_gain_yrs:
                scost_life = scost_base
            else:
                scost_life = 0

            # Add avoided capital costs as appropriate (e.g., for an LED
            # lighting measure with a longer lifetime than the comparable
            # baseline lighting technology)
            cashflows_s_delt = numpy.append(cashflows_s_delt, scost_life)
            cashflows_s_tot = numpy.append(cashflows_s_tot, scost_life)

        # Construct complete incremental and total energy and carbon cash
        # flows across measure lifetime. First term (reserved for initial
        # investment) is zero
        cashflows_e_delt, cashflows_c_delt = [
            numpy.append(0, [x] * life_meas) for x in [ecostsave, ccostsave]]
        cashflows_e_tot, cashflows_c_tot = [
            numpy.append(0, [x] * life_meas) for x in [ecost_meas, ccost_meas]]

        # Calculate net present values (NPVs) using the above cashflows
        npv_s_delt, npv_e_delt, npv_c_delt = [
            npf.npv(self.handyvars.discount_rate, x) for x in [
                cashflows_s_delt, cashflows_e_delt, cashflows_c_delt]]

        # Develop arrays of energy and carbon savings across measure
        # lifetime (for use in cost of conserved energy and carbon calcs).
        # First term (reserved for initial investment figure) is zero, and
        # each array is normalized by number of captured stock units
        esave_array = numpy.append(0, [esave] * life_meas)
        csave_array = numpy.append(0, [csave] * life_meas)

        # Calculate Net Present Value and annuity equivalent Net Present Value
        # of the above energy and carbon savings
        npv_esave = npf.npv(self.handyvars.discount_rate, esave_array)
        npv_csave = npf.npv(self.handyvars.discount_rate, csave_array)

        # Calculate portfolio-level financial metrics

        # Calculate cost of conserved energy w/ and w/o carbon cost savings
        # benefits. Restrict denominator values less than or equal to zero
        if npv_esave > 0:
            cce = (-npv_s_delt / npv_esave)
            cce_bens = (-(npv_s_delt + npv_c_delt) / npv_esave)
        else:
            cce, cce_bens = [999 for n in range(2)]

        # Calculate cost of conserved carbon w/ and w/o energy cost savings
        # benefits. Restrict denominator values less than or equal to zero
        if npv_csave > 0:
            ccc = (-npv_s_delt / (npv_csave * 1000000))
            ccc_bens = (-(npv_s_delt + npv_e_delt) /
                        (npv_csave * 1000000))
        else:
            ccc, ccc_bens = [999 for n in range(2)]

        # Calculate internal rate of return and simple payback for capital
        # + energy and capital + energy + carbon cash flows.  Use try/
        # except to handle cases where IRR/payback cannot be calculated

        # IRR and payback given capital + energy cash flows
        try:
            irr_e = npf.irr(cashflows_s_delt + cashflows_e_delt)
            if not math.isfinite(irr_e):
                raise (ValueError)
        except (ValueError, LinAlgError):
            irr_e = 999
        try:
            payback_e = self.payback(cashflows_s_delt + cashflows_e_delt)
        except (ValueError, LinAlgError):
            payback_e = 999
        # IRR and payback given capital + energy + carbon cash flows
        try:
            irr_ec = npf.irr(
                cashflows_s_delt + cashflows_e_delt + cashflows_c_delt)
            if not math.isfinite(irr_ec):
                raise (ValueError)
        except (ValueError, LinAlgError):
            irr_ec = 999
        try:
            payback_ec = \
                self.payback(
                    cashflows_s_delt + cashflows_e_delt + cashflows_c_delt)
        except (ValueError, LinAlgError):
            payback_ec = 999

        # Set unit capital and operating costs using the above
        # cashflows for later use in measure competition calculations. For
        # residential sector measures, unit costs are simply the unit-level
        # capital and operating costs for the measure.  For commerical
        # sector measures, unit costs are translated to life cycle capital
        # and operating costs across the measure lifetime using multiple
        # discount rate levels that reflect various degrees of risk
        # tolerance observed amongst commercial adopters.  These discount
        # rate levels are imported from commercial AEO demand module data.

        # Populate unit costs for residential sector
        # Check whether measure applies to residential sector
        if any([x in ["single family home", "multi family home",
                      "mobile home"] for x in m.bldg_type]):
            unit_cost_s_res, unit_cost_e_res, unit_cost_c_res = [
                scost_meas, ecost_meas, ccost_meas]
        # If measure does not apply to residential sector, set residential
        # unit costs to 'None'
        else:
            unit_cost_s_res, unit_cost_e_res, unit_cost_c_res = (
                None for n in range(3))

        # Populate unit costs for commercial sector
        # Check whether measure applies to commercial sector
        if any([x not in ["single family home", "multi family home",
                          "mobile home"] for x in m.bldg_type]):
            unit_cost_s_com, unit_cost_e_com, unit_cost_c_com = (
                {} for n in range(3))
            # Set unit cost values under 7 discount rate categories
            try:
                for ind, tps in enumerate(
                        self.handyvars.com_timeprefs["rates"]):
                    unit_cost_s_com["rate " + str(ind + 1)], \
                        unit_cost_e_com["rate " + str(ind + 1)], \
                        unit_cost_c_com["rate " + str(ind + 1)] = \
                        [npf.npv(tps, x) for x in [
                         cashflows_s_tot, cashflows_e_tot,
                         cashflows_c_tot]]
                    if any([not math.isfinite(x) for x in [
                            unit_cost_s_com["rate " + str(ind + 1)],
                            unit_cost_e_com["rate " + str(ind + 1)],
                            unit_cost_c_com["rate " + str(ind + 1)]]]):
                        raise (ValueError)
            except ValueError:
                unit_cost_s_com, unit_cost_e_com, unit_cost_c_com = (
                    None for n in range(3))
        # If measure does not apply to commercial sector, set commercial
        # unit costs to 'None'
        else:
            unit_cost_s_com, unit_cost_e_com, unit_cost_c_com = (
                None for n in range(3))

        # Return all updated economic metrics
        return unit_cost_s_res, unit_cost_e_res, unit_cost_c_res, \
            unit_cost_s_com, unit_cost_e_com, unit_cost_c_com, irr_e, \
            irr_ec, payback_e, payback_ec, cce, cce_bens, ccc, ccc_bens

    def payback(self, cashflows):
        """Calculate simple payback period.

        Notes:
            Calculate the simple payback period given an input list of
            cash flows, which may be uneven.

        Args:
            cashflows (list): Cash flows across measure lifetime.

        Returns:
            Simple payback period for the input cash flows.
        """
        # Separate initial investment and subsequent cash flows
        # from "cashflows" input; extend cashflows up until 100 years
        # out to ensure calculation of all paybacks under 100 years
        investment, cashflows = cashflows[0], list(
            cashflows[1:]) + [cashflows[-1]] * (100 - len(cashflows[1:]))
        # If initial investment is positive, payback = 0
        if investment >= 0:
            payback_val = 0
        else:
            # Find absolute value of initial investment to compare
            # subsequent cash flows against
            investment = abs(investment)
            # Initialize cumulative cashflow and # years tracking
            total, years, cumulative = 0, 0, []
            # Add to years and cumulative cashflow trackers while cumulative
            # cashflow < investment
            for cashflow in cashflows:
                total += cashflow
                if total < investment:
                    years += 1
                cumulative.append(total)
            # If investment pays back within the measure lifetime,
            # calculate this payback period in years
            if years < len(cashflows):
                a = years
                # Case where payback period < 1 year
                if (years - 1) < 0:
                    b = investment
                    c = cumulative[0]
                # Case where payback period >= 1 year
                else:
                    b = investment - cumulative[years - 1]
                    c = cumulative[years] - cumulative[years - 1]
                payback_val = a + (b / c)
            # If investment does not pay back within measure lifetime,
            # set payback period to artifically high number
            else:
                payback_val = 999

        # Return updated payback period value in years
        return payback_val

    def compete_measures(self, adopt_scheme, htcl_totals):
        """Compete/apportion total stock/energy/carbon/cost across measures.

        Notes:
            Adjust each competing measure's 'baseline' and 'efficient'
            energy/carbon/cost to reflect either a) direct competition between
            measures, or b) the indirect effects of measure competition.

        Args:
            adopt_scheme (string): Assumed consumer adoption scenario.
        """
        # Establish list of key chains and supporting competition data for all
        # stock/energy/carbon/cost microsegments that contribute to a measure's
        # total stock/energy/carbon/cost microsegments, across active measures
        mseg_keys, mkts_adj = ([] for n in range(2))
        for x in self.measures:
            mseg_keys.extend(x.markets[adopt_scheme]["competed"][
                "mseg_adjust"]["contributing mseg keys and values"].keys())
            mkts_adj.append(x.markets[adopt_scheme]["competed"]["mseg_adjust"])

        # Establish list of unique key chains in mseg_keys list above,
        # ensuring that all 'primary' microsegments (e.g., relating to direct
        # equipment replacement) are ordered and updated before 'secondary'
        # microsegments (e.g., relating to indirect effects of equipment
        # replacement, such as reduced waste heat from changes in lighting)
        msegs = sorted(numpy.unique(mseg_keys))

        # Initialize a dict used to store data on overlaps between supply-side
        # heating/cooling ECMs (e.g., HVAC equipment) and demand-side
        # heating/cooling ECMs (e.g., envelope). If the current set of ECMs
        # does not affect both supply-side and demand-side heating/cooling
        # markets, this dict is set to None
        if any(["supply" in x for x in msegs]) and \
           any(["demand" in x for x in msegs]):
            htcl_adj_data = {"supply": {}, "demand": {}}
        else:
            htcl_adj_data = None

        # Run through all unique contributing microsegments in the above list,
        # determining how the initial measure stock/energy/carbon/cost data
        # associated with each should be adjusted to reflect the effects of
        # measure competition
        for msu in msegs:

            # Determine the subset of measures that pertain to the current
            # contributing microsegment
            measures_adj = [self.measures[x] for x in range(
                0, len(self.measures)) if msu in mkts_adj[x][
                "contributing mseg keys and values"].keys()]
            # Create short name for all ECM competition data pertaining to
            # current contributing microsegment
            msu_mkts = [m.markets[adopt_scheme]["competed"][
                "mseg_adjust"]["contributing mseg keys and values"][msu] for
                m in measures_adj]

            # If the current contributing microsegment is of the 'primary'
            # type, directly compete the microsegment across applicable
            # measures
            if "primary" in msu:
                # If multiple measures are competing for the primary
                # microsegment, determine the market shares of each competing
                # measure and adjust primary stock/energy/carbon/cost
                # totals for each measure accordingly, using separate market
                # share modeling routines for residential/commercial sectors.
                if len(measures_adj) > 1 and any(x in msu for x in (
                        'single family home', 'multi family home',
                        'mobile home')):
                    self.compete_res_primary(measures_adj, msu, adopt_scheme)
                elif len(measures_adj) > 1 and all(x not in msu for x in (
                        'single family home', 'multi family home',
                        'mobile home')):
                    self.compete_com_primary(measures_adj, msu, adopt_scheme)
            # If the current contributing microsegment is of the 'secondary'
            # type, adjust the microsegment across applicable measures as
            # needed to reflect competition of associated primary
            # contributing microsegment(s) for each measure
            elif "secondary" in msu:
                # Determine the climate zone, building type, and structure type
                # needed to link the secondary microsegment and associated3
                # primary microsegment(s)
                cz_bldg_struct = literal_eval(msu)
                secnd_mseg_adjkey = str((
                    cz_bldg_struct[1], cz_bldg_struct[2], cz_bldg_struct[-1]))
                # Determine the subset of measures pertaining to the given
                # secondary microsegment that require total energy/carbon/cost
                # adjustments due to changes in associated primary
                # microsegment(s) (note that secondary microsegments do not
                # affect stock totals, only energy/carbon and associated costs)
                measures_adj_scnd = [self.measures[x] for x in range(
                    0, len(self.measures)) if self.measures[x] in
                    measures_adj and any(
                    [(y[1] > 0) for y in mkts_adj[x][
                        "secondary mseg adjustments"]["market share"][
                        "original energy (total captured)"][
                        secnd_mseg_adjkey].items()])]
                # If at least one applicable measure requires adjustments to
                # total secondary energy/carbon/cost, proceed with the
                # adjustment calculation
                if len(measures_adj_scnd) > 0:
                    self.secondary_adj(measures_adj_scnd, msu,
                                       secnd_mseg_adjkey, adopt_scheme)

            # For any contributing microsegment that pertains to heating or
            # cooling, record data needed for additional adjustments to remove
            # overlaps between the supply-side and demand-side of heating
            # and cooling energy (note that supply-side and demand-side heating
            # and cooling ECMs are not directly competed). NOTE: EXCLUDE
            # SECONDARY HEATING/COOLING MICROSEGMENTS FOR NOW UNTIL
            # REASONABLE APPROACH FOR ADJUSTING THESE IS IMPLEMENTED

            # Ensure the current contributing microsegment pertains to
            # heating or cooling (marked by 'supply' or 'demand' keys) and
            # that both supply and demand-side ECMs are present in the analysis
            if ('primary' in msu and
                ('supply' in msu or 'demand' in msu)) and \
                    htcl_adj_data is not None:
                htcl_adj_data = self.htcl_adj_rec(
                    htcl_adj_data, msu, msu_mkts, htcl_totals)

        # Once all direct competition is finished, remove all recorded
        # overlapping energy use and associated carbon/costs between
        # supply-side and demand-side heating and cooling ECMs, provided both
        # are present in the analysis
        if htcl_adj_data is not None:
            # Find the subset of ECMs that applies to heating and cooling
            measures_htcl_adj = [m for m in self.measures if any([
                z[0] in ["heating", "cooling", "secondary heating"] for
                z in m.end_use.values() if z is not None])]

            # Remove energy, carbon, and cost overlaps between supply-side and
            # demand-side heating/cooling ECMs
            self.htcl_adj(measures_htcl_adj, adopt_scheme, htcl_adj_data)

    def compete_res_primary(self, measures_adj, mseg_key, adopt_scheme):
        """Apportion stock/energy/carbon/cost across residential measures.

        Notes:
            Determine the shares of a given market microsegment that are
            captured by a series of residential efficiency measures that
            compete for this market microsegment.

        Args:
            measures_adj (list): Competing residential measure objects.
            mseg_key (string): Competed market microsegment information
                (mseg type->czone->bldg->fuel->end use->technology type
                 ->structure type).
            adopt_scheme (string): Assumed consumer adoption scenario.
        """
        # Initialize list of dicts that each store the annual market fractions
        # captured by competing measures; also initialize a dict that sums
        # market fractions by year across competing measures (used to normalize
        # the measure market fractions such that they all sum to 1)
        mkt_fracs = [{} for meas in range(0, len(measures_adj))]
        mkt_fracs_tot = dict.fromkeys(self.handyvars.aeo_years, 0)

        # Loop through competing measures and calculate market shares for each
        # based on their annualized capital and operating costs.

        # Set abbreviated names for the dictionaries containing measure
        # capital and operating cost values, accessed further below

        # Unit capital cost dictionary
        unit_cost_s_in = [m.financial_metrics["unit cost"]["stock cost"][
            "residential"] for m in measures_adj]
        # Unit operating cost dictionary
        unit_cost_e_in = [m.financial_metrics["unit cost"]["energy cost"][
            "residential"] for m in measures_adj]

        # Find the year range in which at least one measure that applies
        # to the competed primary microsegment is on the market
        years_on_mkt_all = []
        [years_on_mkt_all.extend(x.yrs_on_mkt) for x in measures_adj]
        years_on_mkt_all = numpy.unique(years_on_mkt_all)

        # Set market entry years for all competing measures
        mkt_entry_yrs = [
            m.market_entry_year for m in measures_adj]

        # Loop through competing measures and calculate market shares for
        # each based on their annualized capital and operating costs
        for ind, m in enumerate(measures_adj):
            # Set measure markets and market adjustment information

            # Loop through all years in time horizon
            for yr in self.handyvars.aeo_years:
                # Ensure measure is on the market in given year
                if yr in m.yrs_on_mkt:
                    # Set measure capital and operating cost inputs. * Note:
                    # operating cost is set to just energy costs (for now), but
                    # could be expanded to include maintenance and carbon costs

                    # Set capital cost (handle as numpy array or point value)
                    if isinstance(unit_cost_s_in[ind][yr], numpy.ndarray):
                        cap_cost = numpy.zeros(len(unit_cost_s_in[ind][yr]))
                        for i in range(0, len(unit_cost_s_in[ind][yr])):
                            cap_cost[i] = unit_cost_s_in[ind][yr][i]
                    else:
                        cap_cost = unit_cost_s_in[ind][yr]
                    # Set operating cost (handle as numpy array or point value)
                    if isinstance(unit_cost_e_in[ind][yr], numpy.ndarray):
                        op_cost = numpy.zeros(len(unit_cost_e_in[ind][yr]))
                        for i in range(0, len(unit_cost_e_in[ind][yr])):
                            op_cost[i] = unit_cost_e_in[ind][yr][i]
                    else:
                        op_cost = unit_cost_e_in[ind][yr]

                    # Calculate measure market fraction using log-linear
                    # regression equation that takes capital/operating
                    # costs as inputs

                    # Handle case where cost is None
                    try:
                        # Calculate weighted sum of incremental capital and
                        # operating costs
                        sum_wt = cap_cost * \
                            m.markets[adopt_scheme]["competed"][
                                "mseg_adjust"]["competed choice parameters"][
                                str(mseg_key)]["b1"][yr] + op_cost * \
                            m.markets[adopt_scheme]["competed"]["mseg_adjust"][
                                "competed choice parameters"][
                                str(mseg_key)]["b2"][yr]

                        # Guard against cases with very low weighted sums of
                        # incremental capital and operating costs
                        if not isinstance(sum_wt, numpy.ndarray) and \
                                sum_wt < -500:
                            sum_wt = -500
                        elif isinstance(sum_wt, numpy.ndarray) and any([
                                x < -500 for x in sum_wt]):
                            sum_wt = [-500 if x < -500 else x for x in sum_wt]

                        # Calculate market fraction
                        mkt_fracs[ind][yr] = numpy.exp(sum_wt)
                    except TypeError:
                        mkt_fracs[ind][yr] = 0

                    # Add calculated market fraction to mkt fraction sum
                    mkt_fracs_tot[yr] = \
                        mkt_fracs_tot[yr] + mkt_fracs[ind][yr]

        # Loop through competing measures to normalize their calculated
        # market shares to the total market share sum; use normalized
        # market shares to make adjustments to each measure's master
        # microsegment values
        for ind, m in enumerate(measures_adj):
            # Calculate annual market share fraction for the measure and
            # adjust measure's master microsegment values accordingly
            for yr in self.handyvars.aeo_years:
                # Ensure measure is on the market in given year; if not,
                # the measure either splits the market with other
                # competing measures if none of those measures is on
                # the market either, or else has a market share of zero
                if yr in m.yrs_on_mkt:
                    if ((not isinstance(mkt_fracs_tot[yr], numpy.ndarray) and
                         mkt_fracs_tot[yr] != 0) or (
                        isinstance(mkt_fracs_tot[yr], numpy.ndarray) and all(
                            mkt_fracs_tot[yr] != 0))):
                        mkt_fracs[ind][yr] = \
                            mkt_fracs[ind][yr] / mkt_fracs_tot[yr]
                    else:
                        mkt_fracs[ind][yr] = 1 / len(measures_adj)
                elif yr not in years_on_mkt_all:
                    mkt_fracs[ind][yr] = 1 / len(measures_adj)
                else:
                    mkt_fracs[ind][yr] = 0

        # Check for competing ECMs that apply to but a fraction of the competed
        # market, and apportion the remaining fraction of this market across
        # the other competing ECMs
        added_sbmkt_fracs = self.find_added_sbmkt_fracs(
            mkt_fracs, measures_adj, mseg_key, adopt_scheme, years_on_mkt_all)

        # Loop through competing measures and apply competed market shares
        # and gains from sub-market fractions to each ECM's total energy,
        # carbon, and cost impacts
        for ind, m in enumerate(measures_adj):
            # Set measure markets and market adjustment information
            # Establish starting energy/carbon/cost totals, energy/carbon/cost
            # results breakout information, and current contributing primary
            # energy/carbon/cost information for measure
            mast, adj_out_break, adj, mast_list_base, mast_list_eff, \
                adj_list_eff, adj_list_base, adj_stk_trk = \
                self.compete_adj_dicts(m, mseg_key, adopt_scheme)
            for yr in self.handyvars.aeo_years:
                # Make the adjustment to the measure's stock/energy/carbon/
                # cost totals and breakouts based on its updated competed
                # market share and stock turnover rates
                self.compete_adj(
                    mkt_fracs[ind], added_sbmkt_fracs[ind], mast,
                    adj_out_break, adj, mast_list_base, mast_list_eff,
                    adj_list_eff, adj_list_base, yr, mseg_key, m, adopt_scheme,
                    mkt_entry_yrs, adj_stk_trk)

    def compete_com_primary(self, measures_adj, mseg_key, adopt_scheme):
        """Apportion stock/energy/carbon/cost across commercial measures.

        Notes:
            Determines the shares of a given market microsegment that are
            captured by a series of commerical efficiency measures that
            compete for this market microsegment.

        Args:
            measures_adj (list): Competing commercial measure objects.
            mseg_key (string): Competed market microsegment information
                (mseg type->czone->bldg->fuel->end use->technology type
                 ->structure type).
            adopt_scheme (string): Assumed consumer adoption scenario.
        """
        # Initialize list of dicts that each store the annual market fractions
        # captured by competing measures; also initialize a dict that records
        # the total annualized capital + operating costs for each measure
        # and discount rate level (used to choose which measure is adopted
        # under each discount rate level)
        mkt_fracs = [{} for meas in range(0, len(measures_adj))]
        tot_cost = [{} for meas in range(0, len(measures_adj))]

        # Calculate the total annualized cost (capital + operating) needed to
        # determine market shares below

        # Set abbreviated names for the dictionaries containing measure
        # capital and operating cost values, accessed further below

        # Unit stock cost dictionary
        unit_cost_s_in = [m.financial_metrics["unit cost"]["stock cost"][
                     "commercial"] for m in measures_adj]
        # Unit operating cost dictionary
        unit_cost_e_in = [m.financial_metrics["unit cost"]["energy cost"][
            "commercial"] for m in measures_adj]

        # Find the year range in which at least one measure that applies
        # to the competed primary microsegment is on the market
        years_on_mkt_all = []
        [years_on_mkt_all.extend(x.yrs_on_mkt) for x in measures_adj]
        years_on_mkt_all = numpy.unique(years_on_mkt_all)

        # Set market entry years for all competing measures
        mkt_entry_yrs = [
            m.market_entry_year for m in measures_adj]

        # Initialize a flag that indicates whether any competing measures
        # have arrays of annualized capital and/or operating costs rather
        # than point values (resultant of distributions on measure inputs),
        # for each year in the range above
        length_array = numpy.repeat(0, len(self.handyvars.aeo_years))

        # Loop through all years in time horizon
        for ind_l, yr in enumerate(self.handyvars.aeo_years):
            # Determine whether any of the competing measures have
            # arrays of annualized capital and/or operating costs for
            # the given year; if so, find the array length. * Note: all
            # array lengths should be equal to the 'nsamples' variable
            # defined in 'ecm_prep.py'
            if any([isinstance(x[yr], numpy.ndarray) or
                    isinstance(y[yr], numpy.ndarray) for
                    x, y in zip(unit_cost_s_in, unit_cost_e_in)]) is True:
                length_array[ind_l] = next(
                    (len(x[yr]) or len(y[yr]) for x, y in
                     zip(unit_cost_s_in, unit_cost_e_in) if isinstance(
                        x[yr], numpy.ndarray) or isinstance(
                            y[yr], numpy.ndarray)),
                    length_array[ind_l])

        # Loop through competing measures and calculate market shares for
        # each based on their annualized capital and operating costs
        for ind, m in enumerate(measures_adj):
            # Set measure markets and market adjustment information
            # Loop through all years in time horizon
            for ind_l, yr in enumerate(self.handyvars.aeo_years):
                # Ensure measure is on the market in given year
                if yr in m.yrs_on_mkt:
                    # Set measure capital and operating cost inputs. * Note:
                    # operating cost is set to just energy costs (for now), but
                    # could be expanded to include maintenance and carbon costs

                    # Handle cases where capital and/or operating cost inputs
                    # are specified as arrays for at least one of the competing
                    # measures. In this case, the capital and operating costs
                    # for all measures must be formatted consistently as arrays
                    # of the same length
                    if length_array[ind_l] > 0:
                        cap_cost, op_cost = ([
                            {} for n in range(length_array[ind_l])] for
                            n in range(2))
                        for i in range(length_array[ind_l]):
                            # Set capital cost input array
                            if isinstance(
                                    unit_cost_s_in[ind][yr], numpy.ndarray):
                                cap_cost[i] = unit_cost_s_in[ind][yr][i]
                            else:
                                cap_cost[i] = unit_cost_s_in[ind][yr]
                            # Set operating cost input array
                            if isinstance(
                                    unit_cost_e_in[ind][yr], numpy.ndarray):
                                op_cost[i] = unit_cost_e_in[ind][yr][i]
                            else:
                                op_cost[i] = unit_cost_e_in[ind][yr]
                        # Sum capital and operating cost arrays and add to the
                        # total cost dict entry for the given measure
                        tot_cost[ind][yr] = [
                            [] for n in range(length_array[ind_l])]
                        # Handle case where cost is None
                        try:
                            for c_l in range(0, len(tot_cost[ind][yr])):
                                for dr in sorted(cap_cost[c_l].keys()):
                                    tot_cost[ind][yr][c_l].append(
                                        cap_cost[c_l][dr] + op_cost[c_l][dr])
                        except AttributeError:
                            pass
                    # Handle cases where capital and/or operating cost inputs
                    # are specified as point values for all competing measures
                    else:
                        # Set capital cost point value
                        cap_cost = unit_cost_s_in[ind][yr]
                        # Set operating cost point value
                        op_cost = unit_cost_e_in[ind][yr]

                        # Sum capital and operating cost point values and add
                        # to the total cost dict entry for the given measure
                        tot_cost[ind][yr] = []
                        # Handle case where cost is None
                        try:
                            for dr in sorted(cap_cost.keys()):
                                tot_cost[ind][yr].append(
                                    cap_cost[dr] + op_cost[dr])
                        except AttributeError:
                            pass

        # Loop through competing measures and use total annualized capital
        # + operating costs to determine the overall share of the market
        # that is captured by each measure; use market shares to make
        # adjustments to each measure's master microsegment values
        for ind, m in enumerate(measures_adj):
            # Calculate annual market share fraction for the measure and
            # adjust measure's master microsegment values accordingly

            # Loop through all years in time horizon
            for ind_l, yr in enumerate(self.handyvars.aeo_years):
                # Ensure measure is on the market in given year; if not,
                # the measure either splits the market with other
                # competing measures if none of those measures is on
                # the market either, or else has a market share of zero
                if yr in m.yrs_on_mkt:
                    # Set the fractions of commericial adopters who fall into
                    # each discount rate category for this particular
                    # microsegment
                    mkt_dists = m.markets[adopt_scheme]["competed"][
                        "mseg_adjust"]["competed choice parameters"][
                            str(mseg_key)]["rate distribution"][yr]
                    # For each discount rate category, find which measure has
                    # the lowest annualized cost and assign that measure the
                    # share of commercial market adopters defined for that
                    # category above

                    # Handle cases where capital and/or operating cost inputs
                    # are specified as lists for at least one of the competing
                    # measures.
                    if length_array[ind_l] > 0 and len(
                            tot_cost[ind][yr][0]) != 0:
                        mkt_fracs[ind][yr] = [
                            [] for n in range(length_array[ind_l])]
                        for c_l in range(length_array[ind_l]):
                            for ind2, dr in enumerate(
                                    tot_cost[ind][yr][c_l]):
                                # Find the lowest annualized cost for the
                                # set of competing measures/discount bin
                                min_val = min([
                                    tot_cost[x][yr][c_l][ind2] for x in
                                    range(0, len(measures_adj)) if
                                    (yr in tot_cost[x].keys() and
                                     len(tot_cost[x][yr][0]) != 0)])
                                # Determine how many competing measures
                                # have the lowest annualized cost under
                                # the given discount rate bin
                                min_val_ecms = [
                                    x for x in range(0, len(measures_adj)) if
                                    (yr in tot_cost[x].keys() and
                                     len(tot_cost[x][yr][0]) != 0) and
                                    tot_cost[x][yr][c_l][ind2] == min_val]
                                # If the current measure has the lowest
                                # annualized cost, assign it appropriate
                                # market share for current discount rate
                                # category being looped through, divided by
                                # total number of competing measures that
                                # share the lowest annualized cost
                                if tot_cost[ind][yr][c_l][ind2] == min_val:
                                    mkt_fracs[ind][yr][c_l].append(
                                        mkt_dists[ind2] /
                                        len(min_val_ecms))
                                # Otherwise, set its market share for that
                                # discount rate bin to zero
                                else:
                                    mkt_fracs[ind][yr][c_l].append(0)
                            mkt_fracs[ind][yr][c_l] = sum(
                                mkt_fracs[ind][yr][c_l])
                        # Convert market fractions list to numpy array for
                        # use in compete_adj function below
                        mkt_fracs[ind][yr] = numpy.array(
                            mkt_fracs[ind][yr])
                    # Handle cases where capital and/or operating cost inputs
                    # are specified as point values for all competing measures
                    elif length_array[ind_l] == 0:
                        if len(tot_cost[ind][yr]) != 0:
                            mkt_fracs[ind][yr] = []
                            for ind2, dr in enumerate(tot_cost[ind][yr]):
                                # Find the lowest annualized cost for the given
                                # set of competing measures and discount bin
                                min_val = min([
                                    tot_cost[x][yr][ind2] for x in
                                    range(0, len(measures_adj)) if (
                                        yr in tot_cost[x].keys() and
                                        len(tot_cost[x][yr]) != 0)])
                                # Determine how many of the competing measures
                                # have the lowest annualized cost under
                                # the given discount rate bin
                                min_val_ecms = [
                                    x for x in range(0, len(measures_adj)) if
                                    (yr in tot_cost[x].keys() and
                                     len(tot_cost[x][yr]) != 0) and
                                    tot_cost[x][yr][ind2] == min_val]
                                # If the current measure has the lowest
                                # annualized cost, assign it the appropriate
                                # market share for the current discount rate
                                # category being looped through, divided by the
                                # total number of competing measures that share
                                # the lowest annualized cost
                                if tot_cost[ind][yr][ind2] == min_val:
                                    mkt_fracs[ind][yr].append(
                                        mkt_dists[ind2] / len(min_val_ecms))
                                # Otherwise, set its market share for that
                                # discount rate bin to zero
                                else:
                                    mkt_fracs[ind][yr].append(0)
                            mkt_fracs[ind][yr] = sum(mkt_fracs[ind][yr])
                        else:
                            mkt_fracs[ind][yr] = 0
                    else:
                        mkt_fracs[ind][yr] = 0
                elif yr not in years_on_mkt_all:
                    if len(measures_adj) > 1:
                        mkt_fracs[ind][yr] = 1 / len(measures_adj)
                    else:
                        mkt_fracs[ind][yr] = 0
                else:
                    mkt_fracs[ind][yr] = 0

        # Check for competing ECMs that apply to but a fraction of the competed
        # market, and apportion the remaining fraction of this market across
        # the other competing ECMs
        added_sbmkt_fracs = self.find_added_sbmkt_fracs(
            mkt_fracs, measures_adj, mseg_key, adopt_scheme, years_on_mkt_all)

        # Loop through competing measures and apply competed market shares
        # and gains from sub-market fractions to each ECM's total energy,
        # carbon, and cost impacts
        for ind, m in enumerate(measures_adj):
            # Set measure markets and market adjustment information
            # Establish starting energy/carbon/cost totals, energy/carbon/cost
            # results breakout information, and current contributing primary
            # energy/carbon/cost information for measure
            mast, adj_out_break, adj, mast_list_base, mast_list_eff, \
                adj_list_eff, adj_list_base, adj_stk_trk = \
                self.compete_adj_dicts(m, mseg_key, adopt_scheme)
            for yr in self.handyvars.aeo_years:
                # Make the adjustment to the measure's stock/energy/carbon/
                # cost totals and breakouts based on its updated competed
                # market share and stock turnover rates
                self.compete_adj(
                    mkt_fracs[ind], added_sbmkt_fracs[ind], mast,
                    adj_out_break, adj, mast_list_base, mast_list_eff,
                    adj_list_eff, adj_list_base, yr, mseg_key, m, adopt_scheme,
                    mkt_entry_yrs, adj_stk_trk)

    def find_added_sbmkt_fracs(
            self, mkt_fracs, measures_adj, mseg_key, adopt_scheme,
            years_on_mkt_all):
        """Add to competed ECM market shares to account for sub-market scaling.

        Notes:
            In cases where one or more competing ECM only applies to a fraction
            of its applicable baseline market (e.g., it is assigned with a
            sub-market scaling fraction/fractions), the fraction of the market
            that the ECM(s) does not apply to should be apportioned across the
            other competing ECMs and added to their competed market shares.
            This function determines the added fraction that should be added to
            each ECM's competed market share.

        Args:
            mkt_fracs (list): ECM market shares (before considering sub-mkts.).
            measures_adj (object): Competing ECM objects.
            mseg_key (string): Competed market microsegment information.
            adopt_scheme (string): Assumed consumer adoption scenario.
            years_on_mkt_all (list): List of years in which at least one
                competing measure is on the market.

        Returns:
            Market fractions to add to each ECM's competed market share to
            reflect sub-market scaling in one or more competing ECMs.
        """
        # Set the fraction of the competed market that each ECM does not apply
        # to (to be apportioned across the other competing ECMs);
        # this fraction is broken out by year, and draws from sub-market
        # scaling information for competing measures that may or may not
        # also already be broken out by year
        noapply_sbmkt_fracs = [
            {yr: 1 - m.markets[adopt_scheme]["competed"]["mseg_adjust"][
                "contributing mseg keys and values"][mseg_key][
                "sub-market scaling"] for yr in self.handyvars.aeo_years}
            if not isinstance(m.markets[adopt_scheme]["competed"][
                "mseg_adjust"]["contributing mseg keys and values"][mseg_key][
                "sub-market scaling"], dict)
            else {yr: 1 - m.markets[adopt_scheme]["competed"]["mseg_adjust"][
                "contributing mseg keys and values"][mseg_key][
                "sub-market scaling"][yr] for yr in self.handyvars.aeo_years}
            for m in measures_adj]
        # Set a list used to verify that ECM gaining market share is on the
        # market in a given year
        yrs_on_mkt = [m.yrs_on_mkt for m in measures_adj]
        # Set the total number of ECMs being competed
        len_compete = len(noapply_sbmkt_fracs)

        # If all of the competing ECMs apply to the full competed segment,
        # added market shares due to sub-market scaling are set to zero
        if all([x == 0 for x in noapply_sbmkt_fracs]):
            added_sbmkt_fracs = [{yr: 0 for yr in self.handyvars.aeo_years} for
                                 n in range(len_compete)]
        else:
            # For each competed ECM, set the total unaffected market segment
            # across all years in the analysis
            noapply_sbsbmkt_distrib_fracs_yr = [{
                yr: noapply_sbmkt_fracs[ind][yr] * mkt_fracs[ind][yr] for
                yr in self.handyvars.aeo_years} for
                     ind in range(len(measures_adj))]

            # Initialize a list of dicts where each dict represents the
            # additional market fraction an ECM should receive to reflect the
            # presence of sub-market scaling in the competing ECM set
            added_sbmkt_fracs = [{yr: 0 for yr in self.handyvars.aeo_years} for
                                 n in range(len_compete)]
            # Loop through all competing ECMs, determining how to distribute
            # the portion of the competed segment that the ECM does not apply
            # to (if any) across other competing ECMs in the analysis
            for m in range(len_compete):
                # Loop through all years in the analysis
                for yr in self.handyvars.aeo_years:
                    # Set the current measure's unused portion of the segment
                    # to redistribute in the current year
                    seg_redist = noapply_sbsbmkt_distrib_fracs_yr[m][yr]
                    # Determine which of the other competing ECMs is eligible
                    # to receive the current ECM's inapplicable segment
                    # portion. NOTE: it is assumed that competing ECMs that
                    # also do not apply to the entire segment are ineligible,
                    # as are ECMs that are not on the market in a year where
                    # at least one other competing ECM is on the market
                    distrib_inds = [1 if (noapply_sbmkt_fracs[mc][yr] == 0 and
                                          (yr not in years_on_mkt_all or (
                                           yr in years_on_mkt_all and
                                           yr in yrs_on_mkt[mc])))
                                    else 0 for mc in range(0, len_compete)]

                    # Case where one or more competing ECMs applies to the full
                    # competed segment, but the market shares for these ECMs
                    # are all zero
                    if (not isinstance(
                            mkt_fracs[0][yr], numpy.ndarray) and all(
                        [(mkt_fracs[x][yr] == 0) for
                            x in range(0, len(distrib_inds)) if
                            distrib_inds[x] == 1])) or \
                       (isinstance(mkt_fracs[0][yr], numpy.ndarray) and all(
                        [all([mkt_fracs[x][yr][y] == 0 for
                             y in range(len(mkt_fracs[x][yr]))]) for
                            x in range(0, len(distrib_inds)) if
                            distrib_inds[x] == 1])):
                        # Set weights to use in distributing the current ECM's
                        # inapplicable segment portion across all other
                        # competing ECMs that apply to the full competed
                        # segment; since in this case the market shares for
                        # these other ECMs are all zero, set weights such that
                        # the re-distribution is even across these other ECMs
                        if sum(distrib_inds) == 0:
                            sbmkt_distrib_fracs_yr = [
                                0 for n in range(len_compete)]
                        else:
                            even_frac = 1 / sum(distrib_inds)
                            sbmkt_distrib_fracs_yr = [
                                even_frac if distrib_inds[mc] == 1
                                else 0 for mc in range(0, len_compete)]
                    # All other cases
                    else:
                        # Set weights to use in distributing the current ECM's
                        # inapplicable segment portion across all other
                        # competing ECMs that apply to the full competed
                        # segment, based on each ECM's competed market share
                        sbmkt_distrib_fracs_yr = [
                            mkt_fracs[mc][yr] if distrib_inds[mc] == 1
                            else 0 for mc in range(0, len_compete)]
                        # Re-normalize the weighting factors to ensure that
                        # they sum to 1
                        if (not isinstance(
                                sbmkt_distrib_fracs_yr[0], numpy.ndarray)
                            and sum(sbmkt_distrib_fracs_yr) != 0) or \
                           (isinstance(
                                sbmkt_distrib_fracs_yr[0], numpy.ndarray)
                            and all([sum(sbmkt_distrib_fracs_yr[x]) != 0 for
                                    x in range(len(sbmkt_distrib_fracs_yr))])):
                            sbmkt_distrib_fracs_yr = [
                                x / sum(sbmkt_distrib_fracs_yr) for
                                x in sbmkt_distrib_fracs_yr]
                        else:
                            sbmkt_distrib_fracs_yr = [
                                0 for n in range(len(sbmkt_distrib_fracs_yr))]

                    # Loop through all competing ECMs and set the portion of
                    # the current ECM's inapplicable segment that goes to each
                    for mn in range(len_compete):
                        # For each competing ECM and year, multiply the total
                        # inapplicable segment fraction by the ECM's
                        # re-distribution weights calculated above
                        try:
                            added_sbmkt_fracs[mn][yr] += (
                                seg_redist * (sbmkt_distrib_fracs_yr[mn]))
                        except FloatingPointError:  # Handle small numbers
                            pass

        return added_sbmkt_fracs

    def secondary_adj(
            self, measures_adj, mseg_key, secnd_mseg_adjkey, adopt_scheme):
        """Adjust secondary microsegments to account for primary competition.

        Notes:
            Adjust a measure's secondary energy/carbon/cost totals to reflect
            the updated market shares calculated for an associated primary
            microsegment.

        Args:
            measures_adj (list): Competing commercial measure objects.
            mseg_key (string): Competed market microsegment information
                (mseg type->czone->bldg->fuel->end use->technology type
                 ->structure type).
            adopt_scheme (string): Assumed consumer adoption scenario.
        """
        # Loop through all measures that apply to the current contributing
        # secondary microsegment
        for ind, m in enumerate(measures_adj):
            # Establish starting energy/carbon/cost totals and current
            # contributing secondary energy/carbon/cost information for measure
            mast, adj_out_break, adj, mast_list_base, mast_list_eff, \
                adj_list_eff, adj_list_base, adj_stk_trk = \
                self.compete_adj_dicts(m, mseg_key, adopt_scheme)

            # Adjust secondary energy/carbon/cost totals based on the measure's
            # competed market share for an associated primary contributing
            # microsegment

            # Loop through all years where at least one measure that applies
            # to the secondary microsegment is on the market
            for yr in self.handyvars.aeo_years:
                # Find the appropriate market share adjustment information
                # for the given secondary climate zone, building type, and
                # structure type in the measure's 'mseg_adjust' attribute
                # and scale down the energy/carbon/cost totals accordingly
                secnd_adj_mktshr = m.markets[adopt_scheme]["competed"][
                    "mseg_adjust"]["secondary mseg adjustments"][
                    "market share"]
                # Calculate the competed and total market share adjustment
                # factors to apply to the measure secondary energy/carbon/cost
                # totals, where the 'competed' share considers the effects
                # of primary microsegment competition for the current year
                # only, and the 'total' share considers the effects of
                # primary microsegment competition for the current and all
                # previous years the measure was on the market

                # Set competed market share adjustment
                if secnd_adj_mktshr[
                        "original energy (competed and captured)"][
                        secnd_mseg_adjkey][yr] != 0:
                    adj_frac_comp = secnd_adj_mktshr[
                        "adjusted energy (competed and captured)"][
                        secnd_mseg_adjkey][yr] / secnd_adj_mktshr[
                        "original energy (competed and captured)"][
                        secnd_mseg_adjkey][yr]
                # Set competed market share adjustment to zero if total
                # originally captured baseline stock is zero for
                # current year
                else:
                    adj_frac_comp = 0

                # Set total market share adjustment
                if secnd_adj_mktshr["original energy (total captured)"][
                        secnd_mseg_adjkey][yr] != 0:
                    adj_frac_t = secnd_adj_mktshr[
                        "adjusted energy (total captured)"][
                        secnd_mseg_adjkey][yr] / secnd_adj_mktshr[
                        "original energy (total captured)"][
                        secnd_mseg_adjkey][yr]
                # Set total market share adjustment to zero if total
                # originally captured baseline stock is zero for
                # current year
                else:
                    adj_frac_t = 0

                # Adjust baseline energy/cost/carbon, efficient energy/
                # cost/carbon, and energy/cost/carbon savings totals
                # grouped by climate zone, building type, end use, and (
                # if applicable) fuel type by the appropriate fraction;
                # adjust based on segment of current microsegment that was
                # removed from competition
                for var in ["energy", "cost", "carbon"]:
                    # Update baseline and efficient results provided neither is
                    # None or all zeros
                    vs_list = [
                        v if (
                            adj_out_break["base fuel"][var][v] is not None and
                            ((not isinstance(adj_out_break["base fuel"][
                                var][v][yr], numpy.ndarray) and any([
                                    adj_out_break[
                                        "base fuel"][var][v][yr] != 0])) or (
                                isinstance(adj_out_break[
                                    "base fuel"][var][v][yr], numpy.ndarray)
                             and any([any([adj_out_break[
                                "base fuel"][var][v][yr] != 0])])) for
                                yr in adj_out_break[
                                    "base fuel"][var][v].keys()))
                        else "" for v in ["baseline", "efficient"]]
                    # Energy data may include unique efficient captured
                    # tracking if efficient breakout data are present
                    if "efficient" in vs_list and var == "energy" and \
                            "efficient-captured" in adj_out_break[
                            "base fuel"]["energy"].keys():
                        vs_list.append("efficient-captured")
                    for var_sub in [x for x in vs_list if x]:
                        # Select correct fuel split data

                        # Baseline data all with original fuel
                        if var_sub == "baseline":
                            fs_eff_splt_var = 1
                        # Efficient-captured energy all with switched to fuel
                        elif var_sub == "efficient-captured":
                            fs_eff_splt_var = 0
                        # Efficient energy may be split across base/switched
                        # to fuel
                        else:
                            fs_eff_splt_var = adj_out_break[
                                "efficient fuel splits"][var][yr]

                        # Handle extra key on the adjusted microsegment data
                        # for the cost variables ("energy")
                        if var == "cost":
                            adj_out_break["base fuel"][var][var_sub][yr] = \
                                adj_out_break[
                                    "base fuel"][var][var_sub][yr] - (
                                adj[var]["energy"]["total"][var_sub][yr]) * (
                                1 - adj_frac_t) * fs_eff_splt_var
                        else:
                            # Handle efficient captured energy case for fuel
                            # switching, where no base fuel data will be
                            # reported (go to next variable in loop)
                            try:
                                adj_out_break[
                                    "base fuel"][var][var_sub][yr] = \
                                    adj_out_break[
                                        "base fuel"][var][var_sub][yr] - (
                                    adj[var]["total"][var_sub][yr]) * (
                                    1 - adj_frac_t) * fs_eff_splt_var
                            except KeyError:
                                continue

                    # Update savings results
                    # Handle extra key on the adjusted microsegment data
                    # for the cost variables ("energy")
                    if var == "cost":
                        adj_out_break["base fuel"][var]["savings"][yr] = \
                            adj_out_break["base fuel"][var]["savings"][yr] - ((
                                adj[var]["energy"]["total"]["baseline"][yr] -
                                adj[var]["energy"]["total"]["efficient"][yr]
                                ) * (1 - adj_frac_t) * adj_out_break[
                                    "efficient fuel splits"][var][yr])
                    else:
                        adj_out_break["base fuel"][var]["savings"][yr] = \
                            adj_out_break["base fuel"][var]["savings"][yr] - ((
                                adj[var]["total"]["baseline"][yr] -
                                adj[var]["total"]["efficient"][yr]) * (
                                1 - adj_frac_t) * adj_out_break[
                                    "efficient fuel splits"][var][yr])

                    # If the measure involves fuel switching and the user
                    # has broken out results by fuel type, make adjustments
                    # to the efficient, and savings results for the
                    # switched to fuel
                    if adj_out_break["switched fuel"][var][
                            "efficient"] is not None:
                        # Handle extra key on the adjusted microsegment
                        # data for the cost variables ("energy")
                        if var == "cost":
                            # Update efficient result
                            adj_out_break["switched fuel"][var][
                                "efficient"][yr] = \
                                adj_out_break["switched fuel"][var][
                                    "efficient"][yr] - (
                                adj[var]["energy"]["total"][
                                    "efficient"][yr]) * (
                                1 - adj_frac_t) * (1 - adj_out_break[
                                    "efficient fuel splits"][var][yr])
                            # Update savings result; note that savings
                            # for a switched to fuel will be negative and
                            # thus the adjustment to microsegment data
                            # post-competition should be added to the
                            # original savings breakout results
                            adj_out_break["switched fuel"][var][
                                "savings"][yr] = \
                                adj_out_break["switched fuel"][var][
                                    "savings"][yr] + (
                                adj[var]["energy"]["total"][
                                    "efficient"][yr]) * (
                                1 - adj_frac_t) * (1 - adj_out_break[
                                    "efficient fuel splits"][var][yr])
                        else:
                            # Update efficient result
                            # Energy data may include efficient-captured
                            # tracking
                            if var == "energy" and "efficient-captured" in \
                                adj_out_break["switched fuel"][
                                    "energy"].keys():
                                vs_list = ["efficient", "efficient-captured"]
                            else:
                                vs_list = ["efficient"]
                            # Loop through efficient and (if applicable)
                            # efficient-captured data and update
                            for var_sub in vs_list:
                                # Efficient data subject to fuel splits (some
                                # may remain with original fuel)
                                if var_sub == "efficient":
                                    fs_eff_splt_var = adj_out_break[
                                        "efficient fuel splits"][var][yr]
                                # Efficient-captured data not subject to fuel
                                # splits (by definition all energy with
                                # switched to fuel)
                                else:
                                    fs_eff_splt_var = 0
                                adj_out_break["switched fuel"][var][
                                    var_sub][yr] = \
                                    adj_out_break["switched fuel"][var][
                                        var_sub][yr] - (
                                    adj[var]["total"][var_sub][yr]) * (
                                        1 - adj_frac_t) * (1 - fs_eff_splt_var)
                            # Update savings result
                            adj_out_break["switched fuel"][var][
                                "savings"][yr] = \
                                adj_out_break["switched fuel"][var][
                                    "savings"][yr] + (
                                adj[var]["total"]["efficient"][yr]) * (
                                1 - adj_frac_t) * (1 - adj_out_break[
                                    "efficient fuel splits"][var][yr])

                # Adjust total and competed baseline and efficient
                # data by the appropriate secondary adjustment factor
                for x in ["baseline", "efficient"]:
                    # Determine appropriate adjustment data to use
                    # for baseline or efficient case
                    if x == "baseline":
                        mastlist, adjlist = [mast_list_base, adj_list_base]
                    else:
                        mastlist, adjlist = [mast_list_eff, adj_list_eff]
                    # Adjust the total and competed energy, carbon, and
                    # associated cost savings by the secondary adjustment
                    # factor, both overall and for the current
                    # contributing microsegment
                    mast["cost"]["energy"]["total"][x][yr], \
                        mast["cost"]["carbon"]["total"][x][yr], \
                        mast["energy"]["total"][x][yr], \
                        mast["carbon"]["total"][x][yr] = [
                            x[yr] - (y[yr] * (1 - adj_frac_t)) for x, y in
                            zip(mastlist[1:5], adjlist[1:5])]
                    mast["cost"]["energy"]["competed"][x][yr], \
                        mast["cost"]["carbon"]["competed"][x][yr], \
                        mast["energy"]["competed"][x][yr], \
                        mast["carbon"]["competed"][x][yr] = [
                            x[yr] - (y[yr] * (1 - adj_frac_comp)) for x, y in
                            zip(mastlist[6:10], adjlist[6:10])]
                    adj["cost"]["energy"]["total"][x][yr], \
                        adj["cost"]["carbon"]["total"][x][yr], \
                        adj["energy"]["total"][x][yr], \
                        adj["carbon"]["total"][x][yr] = [
                            (x[yr] * adj_frac_t) for x in adjlist[1:5]]
                    adj["cost"]["energy"]["competed"][x][yr], \
                        adj["cost"]["carbon"]["competed"][x][yr], \
                        adj["energy"]["competed"][x][yr], \
                        adj["carbon"]["competed"][x][yr] = [
                            (x[yr] * adj_frac_comp) for x in adjlist[6:10]]
                    # Adjust fugitive methane emissions if applicable
                    if m.fug_e and "methane" in m.fug_e:
                        # Total
                        mast["fugitive emissions"]["methane"][
                            "total"][x][yr] = mastlist[10][yr] - (
                                adjlist[10][yr] * (1 - adj_frac_t))
                        adj["fugitive emissions"]["methane"][
                            "total"][x][yr] = adjlist[10][yr] * adj_frac_t
                        # Competed
                        mast["fugitive emissions"]["methane"][
                            "competed"][x][yr] = mastlist[11][yr] - (
                                adjlist[11][yr] * (1 - adj_frac_comp))
                        adj["fugitive emissions"]["methane"][
                            "competed"][x][yr] = \
                            adjlist[11][yr] * adj_frac_comp

    def htcl_adj_rec(self, htcl_adj_data, msu, msu_mkts, htcl_totals):
        """Record overlaps in heating/cooling supply and demand-side energy.

        Args:
            htcl_adj_data (dict): Overlapping supply or demand-side heating/
                cooling energy use data to update with energy data for current
                contributing microsegment.
            msu (string): Current contributing microsegment key chain.
            msu_mkts (dict): Energy, carbon, and cost data for the current
                contributing microsegment, across all ECMs that apply to
                this microsegment.
            htcl_totals: Energy use totals for all possible climate zone,
                building type, and structure type combinations.

        Returns:
            Updated supply or demand-side heating/cooling overlap data that
            include overlapping energy use data from the current contributing
            microsegment.
        """
        # Establish criteria for matching a supply-side heating/
        # cooling microsegment with a demand-side heating/cooling
        # microsegment (same climate zone, building type, and structure
        # type)

        # Convert contributing microsegment key chain string to a list
        keys = literal_eval(msu)
        # Pull out climate zone, building type, structure type, fuel type,
        # and end use
        msu_split = [str(x) for x in [keys[1], keys[2], keys[-1],
                                      keys[3], keys[4]]]
        # Convert climate zone, building type, structure type, fuel type,
        # and end use data into a string, to be used as a dict key below
        msu_split_key = str(msu_split)
        # Set the technology type of the current heating/cooling microsegment
        # ('supply' or 'demand')
        tech_typ = keys[-3]

        # Determine whether overlapping heating/cooling energy use
        # data have already been initialized for the given climate
        # zone, building type, structure type, fuel type, and end use
        # combination; if not, initialize with overlapping energy use data for
        # all ECMs that apply to the current contributing microsegment; if
        # so, add the overlapping data to what is already there
        if msu_split_key not in htcl_adj_data[tech_typ].keys():
            htcl_adj_data[tech_typ][msu_split_key] = {
                # Record total potential overlapping supply-side
                # and demand-side heating/cooling energy use for
                # the given climate zone, building type,
                # structure type, fuel type, and end use combination
                "total": htcl_totals[msu_split[0]][msu_split[1]][
                    msu_split[2]][msu_split[3]][msu_split[4]],
                # Record the overlapping energy use that is actually
                # affected by the current contributing microsegment,
                # across all ECMs that apply to this microsegment
                "total affected": {yr: sum([(
                    m["energy"]["total"]["baseline"][yr]) for
                    m in msu_mkts]) for
                    yr in self.handyvars.aeo_years},
                # Record the savings in the overlapping energy use
                # affected by the current contributing microsegment,
                # across all ECMs that apply to this microsegment
                "affected savings": {yr: sum([(
                    m["energy"]["total"]["baseline"][yr] -
                    m["energy"]["total"]["efficient"][yr]) for
                    m in msu_mkts]) for yr in self.handyvars.aeo_years}}
        else:
            for yr in self.handyvars.aeo_years:
                # Add to affected overlapping energy use
                htcl_adj_data[tech_typ][msu_split_key][
                    "total affected"][yr] += sum([(
                        m["energy"]["total"]["baseline"][yr]) for
                        m in msu_mkts])
                # Add to affected overlapping energy use savings
                htcl_adj_data[tech_typ][msu_split_key][
                    "affected savings"][yr] += sum([(
                        m["energy"]["total"]["baseline"][yr] -
                        m["energy"]["total"]["efficient"][yr]) for
                        m in msu_mkts])

        return htcl_adj_data

    def htcl_adj(self, measures_htcl_adj, adopt_scheme, htcl_adj_data):
        """Remove heating/cooling supply-demand energy/carbon/cost overlaps.

        Notes:
            These additional adjustments are required to remove overlaps
            across supply-side and demand-side heating/cooling energy

        Args:
            measures_adj (list): Measures requiring supply-demand
                adjustments to energy/carbon/cost totals.
            adopt_scheme (string): Assumed consumer adoption scenario.
            htcl_adj_data (dict): Overlapping supply or demand-side heating/
                cooling energy use data to use in scaling down energy/carbon/
                cost overlaps.
        """
        # Loop through all ECMs requiring additional energy/carbon/cost
        # adjustments
        for m in measures_htcl_adj:
            # Determine the subset of ECM contributing microsegment keys that
            # apply to supply-side or demand-side heating/cooling. NOTE:
            # EXCLUDE SECONDARY HEATING/COOLING MICROSEGMENTS FOR NOW UNTIL
            # REASONABLE APPROACH FOR ADJUSTING THESE IS IMPLEMENTED
            htcl_keys = [k for k in m.markets[adopt_scheme]["competed"][
                "mseg_adjust"]["contributing mseg keys and values"].keys() if
                "primary" in k and ("supply" in k or "demand" in k)]
            # Loop through the ECM's supply-side or demand-side heating/cooling
            # contributing microsegments and scale down energy, carbon, and
            # cost data for that microsegment to remove previously recorded
            # overlaps across the heating/cooling supply-side and demand-side
            for mseg in htcl_keys:
                # Convert contributing microsegment key chain string to a list
                keys = literal_eval(mseg)
                # Pull out climate zone, building type, structure type,
                # fuel type, and end use
                msu_split = [str(x) for x in [keys[1], keys[2], keys[-1],
                                              keys[3], keys[4]]]
                # Convert climate zone, building type, structure type, fuel
                # type, and end use data into a string, to be used as a dict
                # key below
                msu_split_key = str(msu_split)
                # Set the technology type of the current microsegment, as well
                # as the technology types of overlapping microsegments (e.g.,
                # if the current microsegment is on the supply-side of
                # heating/cooling, overlapping microsegments are on the demand
                # side, and vice versa)
                if 'supply' in mseg:
                    tech_typ, tech_typ_overlp = ["supply", "demand"]
                else:
                    tech_typ, tech_typ_overlp = ["demand", "supply"]
                # If no overlapping energy use data exist for the current
                # microsegment's climate zone, building type, structure
                # type, fuel type, and end use combination, move to next
                # contributing microsegment
                if msu_split_key not in htcl_adj_data[tech_typ].keys():
                    continue

                # If overlapping energy use data do exist for the current
                # microsegment's climate zone, building type, structure
                # type, fuel type, and end use combination, create short name
                # for current microsegment energy data dict and overlapping
                # energy data dict; Note: if no overlapping energy data dict
                # can be found, move to next heating/cooling contributing
                # microsegment in for loop
                tech_data = htcl_adj_data[tech_typ][msu_split_key]
                try:
                    overlp_data = htcl_adj_data[tech_typ_overlp][msu_split_key]
                except KeyError:
                    continue
                # Establish set of dicts used to adjust the contributing
                # microsegment energy, carbon, and cost data and master energy,
                # carbon, and cost data to remove the overlaps
                mast, adj_out_break, adj, mast_list_base, mast_list_eff, \
                    adj_list_eff, adj_list_base, adj_stk_trk = \
                    self.compete_adj_dicts(m, mseg, adopt_scheme)
                # Adjust contributing and master energy/carbon/cost
                # data to remove recorded supply-demand overlaps
                for yr in self.handyvars.aeo_years:
                    # Find the fraction of total possibly overlapping
                    # heating/cooling energy for the given climate zone,
                    # building type, and structure type combination that is
                    # actually affected by ECMs in the analysis (e.g., if
                    # looping through a supply-side contributing microsegment,
                    # this is the portion of total energy affected by demand-
                    # side microsegments in the analysis, and vice versa)
                    if overlp_data["total"][yr] != 0:
                        affected_frac = (overlp_data["total affected"][yr] /
                                         overlp_data["total"][yr])
                    else:
                        affected_frac = 0
                    # Find overall relative performance for the technology
                    # type of the current contributing microsegment in the
                    # given climate zone, building type, and structure type
                    # combination
                    if (not isinstance(
                            tech_data["total affected"][yr], numpy.ndarray) and
                        tech_data["total affected"][yr] != 0) or (
                        isinstance(tech_data[
                            "total affected"][yr], numpy.ndarray) and
                        all([x != 0 for x in
                             tech_data["total affected"][yr]])):
                        rel_perf_tech = (1 - (
                            tech_data["affected savings"][yr] /
                            tech_data["total affected"][yr]))
                    else:
                        rel_perf_tech = 1
                    # Find overall relative performance for the overlapping
                    # technology type in the given climate zone, building
                    # type, and structure type combination
                    if (not isinstance(overlp_data["total affected"][yr],
                                       numpy.ndarray) and
                        overlp_data["total affected"][yr] != 0) or (
                        isinstance(overlp_data[
                            "total affected"][yr], numpy.ndarray) and
                        all([x != 0 for x in
                             overlp_data["total affected"][yr]])):
                        rel_perf_tech_overlp = (1 - (
                            overlp_data["affected savings"][yr] /
                            overlp_data["total affected"][yr]))
                    else:
                        rel_perf_tech_overlp = 1
                    # Calculate the ratio of relative performances between the
                    # current microsegment and overlapping microsegments'
                    # technology types in the given climate zone, building
                    # type, and structure type combination; ensure that
                    # neither performance value is negative for the comparison
                    if (all([not isinstance(x, numpy.ndarray) for x in [
                        rel_perf_tech, rel_perf_tech_overlp]]) and
                        (abs(1 - rel_perf_tech) +
                         abs(1 - rel_perf_tech_overlp) != 0)) or (
                        any([isinstance(x, numpy.ndarray) for x in [
                            rel_perf_tech, rel_perf_tech_overlp]]) and
                        all([x != 0 for x in (
                            abs(1 - rel_perf_tech) +
                            abs(1 - rel_perf_tech_overlp))])):
                        save_ratio = abs(1 - rel_perf_tech) / (abs(
                            1 - rel_perf_tech) + abs(1 - rel_perf_tech_overlp))
                    else:
                        save_ratio = 0.5

                    # Calculate baseline and efficient adjustment fractions

                    # Adjust baseline data to reflect the fraction of energy
                    # use affected by the overlapping microsegments, plus the
                    # portion of affected energy use saved by the overlapping
                    # microsegments
                    adj_frac_base = (1 - affected_frac) + \
                        affected_frac * save_ratio

                    # Adjust efficient data in the same way as baseline data,
                    # but with additional consideration for the energy savings
                    # benefits of the overlapping microsegments
                    adj_frac_eff = (1 - affected_frac) + \
                        affected_frac * save_ratio * rel_perf_tech_overlp

                    # Use the baseline/efficient adjustment fractions above to
                    # adjust the ECM's current contributing and master energy,
                    # carbon, and cost data and remove any overlaps
                    for x in ["baseline", "efficient"]:
                        # Determine appropriate adjustment data and factors to
                        # use for baseline or efficient case
                        if x == "baseline":
                            mastlist, adjlist = [mast_list_base, adj_list_base]
                            # Set adj. fraction from above to baseline case
                            adj_frac = adj_frac_base
                        else:
                            mastlist, adjlist = [mast_list_eff, adj_list_eff]
                            # Set adj. fraction from above to efficient case
                            adj_frac = adj_frac_eff
                        # Adjust the total and competed energy, carbon, and
                        # associated cost data for both the ECM's current
                        # contributing microsegment and master microsegment
                        mast["cost"]["energy"]["total"][x][yr], \
                            mast["cost"]["carbon"]["total"][x][yr], \
                            mast["energy"]["total"][x][yr], \
                            mast["carbon"]["total"][x][yr] = [
                                x[yr] - (y[yr] * (1 - adj_frac))
                                for x, y in zip(mastlist[1:5], adjlist[1:5])]
                        mast["cost"]["energy"]["competed"][x][yr], \
                            mast["cost"]["carbon"]["competed"][x][yr], \
                            mast["energy"]["competed"][x][yr], \
                            mast["carbon"]["competed"][x][yr] = [
                                x[yr] - (y[yr] * (1 - adj_frac))
                                for x, y in zip(mastlist[6:10], adjlist[6:10])]
                        # Adjust fugitive methane emissions if applicable
                        if m.fug_e and "methane" in m.fug_e:
                            # Total
                            mast["fugitive emissions"]["methane"][
                                "total"][x][yr] = mastlist[10][yr] - (
                                    adjlist[10][yr] * (1 - adj_frac))
                            # Competed
                            mast["fugitive emissions"]["methane"][
                                "competed"][x][yr] = mastlist[11][yr] - (
                                    adjlist[11][yr] * (1 - adj_frac))

                    # Adjust baseline energy/cost/carbon, efficient energy/
                    # cost/carbon, and energy/cost/carbon savings totals
                    # grouped by climate zone, building type, end use, and (
                    # if applicable) fuel type by the appropriate fraction;
                    # adjust based on segment of current microsegment that was
                    # removed from competition
                    for var in ["energy", "cost", "carbon"]:
                        # Update baseline and efficient results provided
                        # neither is None or all zeros
                        vs_list = [
                            v if (
                                adj_out_break["base fuel"][var][v] is not None
                                and
                                ((not isinstance(adj_out_break["base fuel"][
                                    var][v][yr], numpy.ndarray) and any([
                                        adj_out_break[
                                            "base fuel"][var][v][yr] != 0])) or
                                 (isinstance(adj_out_break[
                                        "base fuel"][var][v][yr],
                                        numpy.ndarray)
                                 and any([any([adj_out_break[
                                    "base fuel"][var][v][yr] != 0])])) for
                                    yr in adj_out_break[
                                        "base fuel"][var][v].keys()))
                            else "" for v in ["baseline", "efficient"]]
                        # Energy data may include unique efficient captured
                        # tracking if efficient breakout data are present
                        if "efficient" in vs_list and var == "energy" and \
                                "efficient-captured" in adj_out_break[
                                "base fuel"]["energy"].keys():
                            vs_list.append("efficient-captured")
                        for var_sub in [x for x in vs_list if x]:
                            # Set appropriate post-competition adjustment frac.
                            # and fuel split data
                            if var_sub == "baseline":
                                adj_frac_t = adj_frac_base
                                # Baseline data all with original fuel
                                fs_eff_splt_var = 1
                            elif var_sub == "efficient-captured":
                                adj_frac_t = adj_frac_eff
                                # Efficient-captured energy all with switched
                                # to fuel
                                fs_eff_splt_var = 0
                            else:
                                adj_frac_t = adj_frac_eff
                                # Efficient energy may be split across base/
                                # switched to fuel
                                fs_eff_splt_var = adj_out_break[
                                    "efficient fuel splits"][var][yr]

                            # Handle extra key on the adjusted microsegment
                            # data for the cost variables ("energy")
                            if var == "cost":
                                adj_out_break[
                                    "base fuel"][var][var_sub][yr] = \
                                    adj_out_break[
                                        "base fuel"][var][var_sub][yr] - (
                                    adj[var]["energy"][
                                        "total"][var_sub][yr]) * (
                                        1 - adj_frac_t) * fs_eff_splt_var
                            else:
                                # Handle efficient-captured energy case for
                                # fuel switching, where no base fuel data will
                                # be reported (go to next variable in loop)
                                try:
                                    adj_out_break[
                                        "base fuel"][var][var_sub][yr] = \
                                        adj_out_break["base fuel"][var][
                                            var_sub][yr] - (
                                        adj[var]["total"][var_sub][yr]) * (
                                        1 - adj_frac_t) * fs_eff_splt_var
                                except KeyError:
                                    continue

                        # Update savings results
                        # Handle extra key on the adjusted microsegment data
                        # for the cost variables ("energy")
                        if var == "cost":
                            adj_out_break["base fuel"][var]["savings"][yr] = \
                                adj_out_break["base fuel"][var][
                                    "savings"][yr] - (
                                    ((adj[var]["energy"]["total"][
                                        "baseline"][yr] * (
                                        1 - adj_frac_base)) -
                                     (adj[var]["energy"]["total"][
                                        "efficient"][yr]) * (
                                        1 - adj_frac_eff) * adj_out_break[
                                            "efficient fuel splits"][var][yr]))
                        else:
                            adj_out_break["base fuel"][var]["savings"][yr] = \
                                adj_out_break["base fuel"][var][
                                    "savings"][yr] - (
                                    ((adj[var]["total"]["baseline"][yr] * (
                                        1 - adj_frac_base)) -
                                     (adj[var]["total"]["efficient"][yr]) * (
                                        1 - adj_frac_eff) * adj_out_break[
                                            "efficient fuel splits"][var][yr]))

                        # If the measure involves fuel switching and the user
                        # has broken out results by fuel type, make adjustments
                        # to the efficient, and savings results for the
                        # switched to fuel
                        if adj_out_break["switched fuel"]["energy"][
                                "baseline"] is not None:
                            # Handle extra key on the adjusted microsegment
                            # data for the cost variables ("energy")
                            if var == "cost":
                                # Update efficient result
                                adj_out_break["switched fuel"][var][
                                    "efficient"][yr] = \
                                    adj_out_break["switched fuel"][var][
                                        "efficient"][yr] - (
                                    adj[var]["energy"]["total"][
                                        "efficient"][yr]) * (
                                    1 - adj_frac_eff) * (1 - adj_out_break[
                                        "efficient fuel splits"][var][yr])
                                # Update savings result; note that savings
                                # for a switched to fuel will be negative and
                                # thus the adjustment to microsegment data
                                # post-competition should be added to the
                                # original savings breakout results
                                adj_out_break["switched fuel"][var][
                                    "savings"][yr] = \
                                    adj_out_break["switched fuel"][var][
                                        "savings"][yr] + (
                                    adj[var]["energy"]["total"][
                                        "efficient"][yr]) * (
                                    1 - adj_frac_eff) * (1 - adj_out_break[
                                        "efficient fuel splits"][var][yr])
                            else:
                                # Update efficient result
                                # Energy data may include efficient-captured
                                # tracking
                                if var == "energy" and \
                                    "efficient-captured" in \
                                    adj_out_break["switched fuel"][
                                        "energy"].keys():
                                    vs_list = ["efficient",
                                               "efficient-captured"]
                                else:
                                    vs_list = ["efficient"]
                                # Loop through efficient and (if applicable)
                                # efficient-captured data and update
                                for var_sub in vs_list:
                                    # Efficient data subject to fuel splits
                                    # (some may remain with original fuel)
                                    if var_sub == "efficient":
                                        fs_eff_splt_var = adj_out_break[
                                            "efficient fuel splits"][var][yr]
                                    # Efficient-captured data not subject to
                                    # fuel splits (by definition all energy w/
                                    # switched to fuel)
                                    else:
                                        fs_eff_splt_var = 0
                                    adj_out_break["switched fuel"][var][
                                        var_sub][yr] = \
                                        adj_out_break["switched fuel"][var][
                                            var_sub][yr] - (
                                        adj[var]["total"][var_sub][yr]) * (
                                            1 - adj_frac_eff) * (
                                            1 - fs_eff_splt_var)
                                # Update savings result
                                adj_out_break["switched fuel"][var][
                                    "savings"][yr] = \
                                    adj_out_break["switched fuel"][var][
                                        "savings"][yr] + (
                                    adj[var]["total"]["efficient"][yr]) * (
                                    1 - adj_frac_eff) * (1 - adj_out_break[
                                        "efficient fuel splits"][var][yr])

    def compete_adj_dicts(self, m, mseg_key, adopt_scheme):
        """Set the initial measure market data needed to adjust for overlaps.

        Notes:
            Establishes a measure's initial stock/energy/carbon/cost totals
            (pre-competition) and the contributing microsegment information
            needed to adjust these totals following measure competition.

        Args:
            m (object): Measure needing market overlap adjustments.
            mseg_key (string): Key for competed market microsegment.
            adopt_scheme (string): Assumed consumer adoption scenario.

        Returns:
            Lists of initial measure master microsegment data and contributing
            microsegment data needed to adjust for competition across measures.
        """

        # Using the key chain for the current microsegment, determine the
        # output climate zone, building type, and end use breakout categories
        # to which the current microsegment applies (uncompeted data in this
        # combination of categories will be adjusted to reflect competition)

        # Convert microsegment string to a list
        key_list = literal_eval(mseg_key)
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
            # the 'Envelope' end use, with the exception of
            # 'demand' side heating/cooling microsegments that
            # represent waste heat from lights - these are
            # categorized as part of the 'Lighting' end use
            if key_list[4] == "other":
                if key_list[5] == "freezers":
                    out_eu = "Refrigeration"
                else:
                    out_eu = "Other"
            elif key_list[4] in eu[1]:
                if (eu[0] in ["Heating (Equip.)",
                              "Cooling (Equip.)"] and
                    key_list[5] == "supply") or (
                    eu[0] in ["Heating (Env.)",
                              "Cooling (Env.)"] and
                    key_list[5] == "demand" and
                    key_list[0] == "primary") or (
                    eu[0] not in ["Heating (Equip.)",
                                  "Cooling (Equip.)",
                                  "Heating (Env.)",
                                  "Cooling (Env.)"]):
                    out_eu = eu[0]
            elif "lighting gain" in key_list:
                out_eu = "Lighting"

        # If applicable, establish fuel type breakout
        if len(self.handyvars.out_break_fuels.keys()) != 0 and out_eu in \
                self.handyvars.out_break_eus_w_fsplits:
            # Flag for detailed fuel type breakout
            detail = len(self.handyvars.out_break_fuels.keys()) > 2
            # Establish breakout of fuel type that is being
            # reduced (e.g., through efficiency or fuel switching
            # away from the fuel)
            for f in self.handyvars.out_break_fuels.items():
                if key_list[3] in f[1]:
                    # Special handling for other fuel tech.,
                    # under detailed fuel type breakouts; this
                    # tech. may fit into multiple fuel cats.
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
                        # All other tech. goes to propane
                        elif f[0] == "Propane":
                            out_fuel_save = f[0]
                    else:
                        out_fuel_save = f[0]
            # Establish breakout of fuel type that is being added
            # to via fuel switching, if applicable
            if m.fuel_switch_to == "electricity" and \
                    out_fuel_save != "Electric":
                out_fuel_gain = "Electric"
            elif m.fuel_switch_to not in [None, "electricity"] and \
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

        # Organize relevant starting master microsegment values into a list
        mast = m.markets[adopt_scheme]["competed"]["master_mseg"]
        # Set total-baseline and competed-baseline overall
        # stock/energy/carbon/cost totals to be updated in the
        # 'compete_adj', 'secondary_adj', and 'htcl_adj' functions
        mast_list_base = [
            mast["cost"]["stock"]["total"]["baseline"],
            mast["cost"]["energy"]["total"]["baseline"],
            mast["cost"]["carbon"]["total"]["baseline"],
            mast["energy"]["total"]["baseline"],
            mast["carbon"]["total"]["baseline"],
            mast["cost"]["stock"]["competed"]["baseline"],
            mast["cost"]["energy"]["competed"]["baseline"],
            mast["cost"]["carbon"]["competed"]["baseline"],
            mast["energy"]["competed"]["baseline"],
            mast["carbon"]["competed"]["baseline"]]
        # Set total-efficient and competed-efficient overall
        # stock/energy/carbon/cost totals to be updated in the
        # 'compete_adj', 'secondary_adj', and 'htcl_adj' functions
        mast_list_eff = [
            mast["cost"]["stock"]["total"]["efficient"],
            mast["cost"]["energy"]["total"]["efficient"],
            mast["cost"]["carbon"]["total"]["efficient"],
            mast["energy"]["total"]["efficient"],
            mast["carbon"]["total"]["efficient"],
            mast["cost"]["stock"]["competed"]["efficient"],
            mast["cost"]["energy"]["competed"]["efficient"],
            mast["cost"]["carbon"]["competed"]["efficient"],
            mast["energy"]["competed"]["efficient"],
            mast["carbon"]["competed"]["efficient"]]
        # Efficient energy data may include tracking of the portion captured
        # by the measure; if present, update the master data with this variable
        # and flag this for subsequent operations
        try:
            mast_list_eff.append(mast["energy"]["total"]["efficient-captured"])
            eff_capt = True
        except KeyError:
            eff_capt = False

        # Add total-baseline and competed-baseline overall fugitive emissions
        # totals to the lists above, if applicable

        # Add fugitive emissions from methane if applicable
        if m.fug_e and "methane" in m.fug_e:
            mast_list_base.extend([
                mast["fugitive emissions"]["methane"][
                    "total"]["baseline"],
                mast["fugitive emissions"]["methane"][
                    "competed"]["baseline"]])
            mast_list_eff.extend([
                mast["fugitive emissions"]["methane"][
                    "total"]["efficient"],
                mast["fugitive emissions"]["methane"][
                    "competed"]["efficient"]])
        # Add fugitive emissions from refrigerants if applicable
        if m.fug_e and "refrigerants" in m.fug_e:
            mast_list_base.extend([
                mast["fugitive emissions"]["refrigerants"][
                    "total"]["baseline"],
                mast["fugitive emissions"]["refrigerants"][
                    "competed"]["baseline"]])
            mast_list_eff.extend([
                mast["fugitive emissions"]["refrigerants"][
                    "total"]["efficient"],
                mast["fugitive emissions"]["refrigerants"][
                    "competed"]["efficient"]])

        # Initialize shorthand dict for baseline energy/cost/carbon, efficient
        # energy/cost/carbon, and energy/cost/carbon savings breakout
        # information for the current measure that falls under the climate
        # zone, building type, end use, and (if applicable) fuel type
        # categories of the currently competed microsegment
        adj_out_break = {
            "base fuel": {
                "energy": {
                    "baseline": None, "efficient": None, "savings": None},
                "cost": {
                    "baseline": None, "efficient": None, "savings": None},
                "carbon": {
                    "baseline": None, "efficient": None, "savings": None}},
            "switched fuel": {
                "energy": {
                    "baseline": None, "efficient": None, "savings": None},
                "cost": {
                    "baseline": None, "efficient": None, "savings": None},
                "carbon": {
                    "baseline": None, "efficient": None, "savings": None}},
            "efficient fuel splits": {
                "energy": None, "carbon": None, "cost": None
            }
        }

        # Breakout data may include reporting of efficient-captured energy;
        # initialize if needed
        if eff_capt:
            adj_out_break["base fuel"]["energy"]["efficient-captured"], \
                adj_out_break["switched fuel"]["energy"][
                    "efficient-captured"] = (None for n in range(2))

        # Add data from the current microsegment as appropriate to the
        # categories in the output breakout dict initialized above

        # Case where output breakouts are split by fuel
        if out_fuel_save:
            for var in ["energy", "cost", "carbon"]:
                # Determine list of metrics to loop through; note that energy data
                # include unique tracking of efficient-captured energy data
                if var != "energy":
                    var_list = ["baseline", "efficient", "savings"]
                else:  # efficient captured data for energy
                    var_list = ["baseline", "efficient",
                                "efficient-captured", "savings"]
                # Adjust energy/carbon/cost data
                for var_sub in var_list:
                    # Handle case where potential efficient-captured energy
                    # data are not present
                    try:
                        adj_out_break["base fuel"][var][var_sub] = \
                            m.markets[adopt_scheme]["competed"]["mseg_out_break"][
                                var][var_sub][out_cz][out_bldg][out_eu][out_fuel_save]
                    except KeyError as ke:
                        if var_sub == "efficient-captured":
                            continue
                        else:
                            raise ke
                # Case with fuel switching
                if out_fuel_gain:
                    # Adjust stock/energy/carbon/cost data
                    for var_sub in var_list:
                        adj_out_break["switched fuel"][var][var_sub] = \
                            m.markets[adopt_scheme]["competed"][
                                "mseg_out_break"][var][var_sub][out_cz][
                                out_bldg][out_eu][out_fuel_gain]
                    # Set previously stored fuel splits for efficient case
                    # results (e.g., the efficient case may reflect some
                    # energy/carb/cost that remains with the baseline fuel
                    # type and thus is not applicable to the adjustment of the
                    # fuel being switched to)
                    fs_eff_splt = m.eff_fs_splt[adopt_scheme][mseg_key]
                    # Pull the fraction of efficient-case energy/cost/carbon
                    # that remains w/ the original fuel in each year for the
                    # contributing measure/mseg
                    adj_out_break["efficient fuel splits"][var] = {
                        yr: (fs_eff_splt[var][0][yr] /
                             fs_eff_splt[var][1][yr]) if
                        fs_eff_splt[var][1][yr] != 0 else 1
                        for yr in self.handyvars.aeo_years}
                else:
                    # All efficient energy remains with original base fuel type
                    # if there is no fuel switching
                    for var in ["energy", "cost", "carbon"]:
                        adj_out_break["efficient fuel splits"][var] = {
                            yr: 1 for yr in self.handyvars.aeo_years}
        # Case where output breakouts are not split by fuel
        else:
            # Adjust energy/carbon/cost data
            for var in ["energy", "cost", "carbon"]:
                # Determine list of metrics to loop through
                if var != "energy":
                    var_list = ["baseline", "efficient", "savings"]
                else:  # efficient captured data for energy
                    var_list = ["baseline", "efficient",
                                "efficient-captured", "savings"]
                for var_sub in var_list:
                    # Handle case where potential efficient-captured energy
                    # data are not present
                    try:
                        adj_out_break["base fuel"][var][var_sub] = \
                            m.markets[adopt_scheme]["competed"]["mseg_out_break"][
                                var][var_sub][out_cz][out_bldg][out_eu]
                    except KeyError as ke:
                        if var_sub == "efficient-captured":
                            continue
                        else:
                            raise ke
                # No adjustment to efficient results required to account for
                # fuel splits
                adj_out_break["efficient fuel splits"][var] = {
                    yr: 1 for yr in self.handyvars.aeo_years}

        # Set up lists that will be used to determine the energy, carbon,
        # and cost totals associated with the contributing microsegment that
        # must be adjusted to reflect measure competition/interaction
        adj = m.markets[adopt_scheme]["competed"]["mseg_adjust"][
            "contributing mseg keys and values"][mseg_key]

        # Set up separate set of stock data needed to determine stock turnover
        # adjustments as part of the measure competition calculations

        # Handle case where an equipment measure has multiple end uses that
        # include heating and/or cooling and the contributing microsegment
        # needs to be anchored on the stock turnover of the heating equipment
        # (or cooling if not available). Such measures might include, for
        # example, HP measures, HVAC + envelope packages, or controls measures
        # spanning heating/cooling and other end uses
        if len(m.end_use) > 1 and "demand" not in mseg_key and ((
            "heating" in m.end_use["primary"] and "heating" not in mseg_key)
            or ("heating" not in m.end_use["primary"] and (
                "cooling" in m.end_use["primary"] and
                "cooling" not in mseg_key))):
            # Decompose contributing microsegment key information into a list,
            # to be modified per comment above
            key_list = list(literal_eval(mseg_key))
            # Strip any additional information that is added to the
            # EIA technology name to further distinguish msegs with exogenous
            # rates and/or specific heating and cooling pairings
            if "-" in key_list[-2]:
                tch_apnd = ("-" + key_list[-2].split("-")[-1])
            else:
                tch_apnd = ""
            # Determine the building type of the contributing microsegment
            if any([x in mseg_key for x in [
                    "single family home", "mobile home",
                    "multi family home"]]):
                mseg_bldg_sect = "residential"
            else:
                mseg_bldg_sect = "commercial"
            # Case 1: heating is in the measure end uses, while heating is not
            # in the current contributing microsegment
            if "heating" in m.end_use["primary"] and (
                    "heating" not in mseg_key):
                # Reset end use
                key_list[4] = "heating"
                # Ensure the contributing microsegment information is
                # structured to have the same "supply" key as the heating
                # microsegment that will ultimately be pulled for stock
                # turnover calculations
                if "supply" not in key_list:
                    key_list = key_list[:5] + ["supply"] + key_list[5:]
                # Residential case
                if mseg_bldg_sect == "residential":
                    # Non-cooling tech. or cooling tech. is non-HP; find
                    # appropriate heating tech. to switch to
                    if any([x in mseg_key for x in [
                            "room AC", "central AC"]]) or \
                            "cooling" not in mseg_key:
                        # Set tech. to first in list of heating
                        # technologies that the measure applies to, and set
                        # the fuel as appropriate to the selected tech.
                        if "resistance heat" in m.technology["primary"]:
                            # Reset tech.
                            key_list[-2] = "resistance heat"
                            # Reset fuel
                            key_list[3] = "electricity"
                        else:
                            # Initialize list of heating technologies that
                            # would be expected for a non-HP cooling tech.
                            tech_search = [x for x in [
                                "furnace (NG)", "boiler (NG)",
                                "furnace (distillate)", "boiler (distillate)",
                                "furnace (LPG)", "furnace (kerosene)",
                                "stove (wood)"] if x
                                in m.technology["primary"]]
                            # If the microsegment is non-cooling (e.g.,
                            # secondary heating), expand to all commercial
                            # heating tech.
                            if "cooling" not in mseg_key:
                                tech_search.extend(["ASHP", "GSHP", "NGHP"])
                            if len(tech_search) == 0:
                                raise ValueError(
                                    "Contributing microsegment " + mseg_key +
                                    " for measure " + m.name +
                                    " has unexpected heating technology "
                                    "information for stock turnover "
                                    "calculations")
                            else:
                                # Reset tech.
                                key_list[-2] = tech_search[0]
                                # Reset fuel
                                if "NG" in tech_search[0]:
                                    key_list[3] = "natural gas"
                                elif "distillate" in tech_search[0]:
                                    key_list[3] = "distillate"
                                elif any([x in tech_search[0] for x in [
                                        "LPG", "kerosene", "wood"]]):
                                    key_list[3] = "other fuel"
                                else:
                                    key_list[3] = "electricity"
                    # Cooling tech. is HP; heating tech. is identical and no
                    # further action is needed
                    elif any([x in mseg_key for x in [
                            "ASHP", "GSHP", "NGHP"]]):
                        pass
                    # If unexpected tech. is present, throw error
                    else:
                        raise ValueError(
                            "Contributing microsegment " + mseg_key +
                            " for measure " + m.name + " has "
                            "unexpected technology information for stock "
                            "turnover calculations")
                # Commercial case
                else:
                    # Non-cooling tech. or cooling tech. is non-HP; find
                    # appropriate heating tech. to switch to
                    if any([x in mseg_key for x in [
                            "rooftop_AC", "scroll_chiller",
                            "res_type_central_AC", "reciprocating_chiller",
                            "centrifugal_chiller", "wall-window_room_AC",
                            "screw_chiller", "gas_chiller",
                            "gas_eng-driven_RTAC"]]) or \
                            "cooling" not in mseg_key:
                        # Set tech. to first in list of heating
                        # technologies that the measure applies to, and set
                        # the fuel as appropriate to the selected tech.

                        # Initialize list of heating technologies that would
                        # be expected for a non-HP cooling tech.
                        tech_search = [x for x in [
                            "elec_boiler", "electric_res-heat", "gas_boiler",
                            "gas_furnace", "oil_boiler", "oil_furnace"] if x in
                            m.technology["primary"]]
                        # If the microsegment is non-cooling (e.g.,
                        # ventilation), expand to all commercial heating tech.
                        if "cooling" not in mseg_key:
                            tech_search.extend([
                                "rooftop_ASHP-heat", "comm_GSHP-heat",
                                "gas_eng-driven_RTHP-heat",
                                "res_type_gasHP-heat"])
                        if len(tech_search) == 0:
                            raise ValueError(
                                "Contributing microsegment " + mseg_key +
                                " for measure " + m.name + " has unexpected "
                                "heating technology information for stock "
                                "turnover calculations")
                        else:
                            # Reset tech.
                            key_list[-2] = tech_search[0]
                            # Reset fuel
                            if "elec" in tech_search[0] or any([
                                    x in tech_search[0] for
                                    x in ["ASHP", "GSHP"]]):
                                key_list[3] = "electricity"
                            elif "gas" in tech_search[0]:
                                key_list[3] = "natural gas"
                            else:
                                key_list[3] = "distillate"
                    # Cooling tech. is HP; select analogous heating HP tech.
                    # name
                    elif "comm_GSHP-cool" in mseg_key:
                        key_list[-2] = "comm_GSHP-heat"
                    elif "rooftop_ASHP-cool" in mseg_key:
                        key_list[-2] = "rooftop_ASHP-heat"
                    elif "gas_eng-driven_RTHP-cool" in mseg_key:
                        key_list[-2] = "gas_eng-driven_RTHP-heat"
                    elif "res_type_gasHP-cool" in mseg_key:
                        key_list[-2] = "res_type_gasHP-heat"
                    # If unexpected tech. is present, throw error
                    else:
                        raise ValueError(
                            "Contributing microsegment " + mseg_key +
                            " for measure " + m.name + " has "
                            "unexpected technology information for stock "
                            "turnover calculations")
            # Case 2: heating is not in the measure end uses, cooling is in the
            # measure end uses, and cooling is not in the current contributing
            # microsegment
            elif "heating" not in m.end_use["primary"] and (
                    "cooling" in m.end_use["primary"] and "cooling" not in
                    mseg_key):
                # Reset end use
                key_list[4] = "cooling"
                # Ensure the contributing microsegment information is
                # structured to have the same "supply" key as the cooling
                # microsegment that will ultimately be pulled for stock
                # turnover calculations
                if "supply" not in key_list:
                    key_list = key_list[:5] + ["supply"] + key_list[5:]
                # Residential case
                if mseg_bldg_sect == "residential":
                    # Set tech. to first in list of cooling
                    # technologies that the measure applies to, and set
                    # the fuel as appropriate to the selected tech.
                    tech_search = [x for x in [
                        "central AC", "ASHP", "GSHP", "NGHP", "room AC"] if
                        x in m.technology["primary"]]
                    if len(tech_search) == 0:
                        raise ValueError(
                            "Contributing microsegment " + mseg_key +
                            " for measure " + m.name +
                            " has unexpected cooling technology "
                            "information for stock turnover calculations")
                    else:
                        # Reset tech.
                        key_list[-2] = tech_search[0]
                        # Reset fuel
                        if tech_search[0] == "NGHP":
                            key_list[3] = "natural gas"
                        else:
                            key_list[3] = "electricity"
                # Commercial case
                else:
                    # Set tech. to first in list of cooling
                    # technologies that the measure applies to, and set
                    # the fuel as appropriate to the selected tech.
                    tech_search = [x for x in [
                         "rooftop_AC", "rooftop_ASHP-cool",
                         "reciprocating_chiller", "scroll_chiller",
                         "centrifugal_chiller", "screw_chiller",
                         "res_type_central_AC", "comm_GSHP-cool",
                         "gas_eng-driven_RTAC", "gas_chiller",
                         "res_type_gasHP-cool", "gas_eng-driven_RTHP-cool",
                         "wall-window_room_AC"] if x in
                        m.technology["primary"]]
                    if len(tech_search) == 0:
                        raise ValueError(
                            "Contributing microsegment " + mseg_key +
                            " for measure " + m.name +
                            " has unexpected cooling technology "
                            "information for stock turnover calculations")
                    else:
                        # Reset tech.
                        key_list[-2] = tech_search[0]
                        # Reset fuel
                        if "gas" in tech_search[0]:
                            key_list[3] = "natural gas"
                        else:
                            key_list[3] = "electricity"
            else:
                raise ValueError(
                    "Contributing microsegment " + mseg_key +
                    " for measure " + m.name +
                    " has unexpected information for stock turnover "
                    "calculations")
            # After making the adjustments above, convert the modified
            # contributing microsegment information back into a string
            # to use in keying in needed stock data
            mseg_key_stk_trk = str(tuple(key_list))
            # For HP measures with exogenously-specified switching
            # rates or representing replacement of specific heating and
            # cooling pairs, the heating technology will be specified in
            # contributing microsegment data with an appended competition info.
            # to distinguish such considerations; develop alternate stock data
            # keys to switch to to handle this case
            if tch_apnd:
                key_list_alt1 = copy.deepcopy(key_list)
                key_list_alt1[-2] = (key_list_alt1[-2] + tch_apnd)
                mseg_key_stk_trk_alt1 = str(tuple(key_list_alt1))
            else:
                mseg_key_stk_trk_alt1 = None
        # Handle all other cases, where stock data will be available for
        # the contributing microsegment to be adjusted as-is
        else:
            # Use contributing microsegment info. as-is to key in stock data
            mseg_key_stk_trk = mseg_key
            # Alternate stock data key does not apply in this case
            mseg_key_stk_trk_alt1 = None

        # Pull data for stock turnover calculations for the current
        # contributing microsegment, using the data key information from above;
        # note that these calculations rely on pre-competition (unadjusted)
        # stock data that will be common to all measures that compete for
        # the microsegment

        # Handle case where data are keyed in with additional tech. information
        # that distinguishes segments with exogenous switching rates and
        # specifics heating/cooling pairs (use alternate data key from above)
        try:
            adj_stk_trk = m.markets[adopt_scheme]["uncompeted"]["mseg_adjust"][
                "contributing mseg keys and values"][mseg_key_stk_trk]["stock"]
        except KeyError:
            try:
                adj_stk_trk = m.markets[adopt_scheme]["uncompeted"][
                    "mseg_adjust"]["contributing mseg keys and values"][
                    mseg_key_stk_trk_alt1]["stock"]
            except KeyError:
                # Handle case where expected microsegment stock data to be
                # linked to the stock turnover calculations for the current
                # microsegment is not available; key in stock data with the
                # current microsegment stock info.
                try:
                    adj_stk_trk = m.markets[adopt_scheme]["uncompeted"][
                        "mseg_adjust"][
                        "contributing mseg keys and values"][
                        mseg_key]["stock"]
                except KeyError:
                    raise ValueError(
                        "Stock turnover data could not be keyed in "
                        "for contributing microsegment " + mseg_key +
                        " for measure " + m.name + " using the "
                        "keys " + mseg_key_stk_trk + ", " +
                        mseg_key_stk_trk_alt1 + ", or" + mseg_key)

        # Set total-baseline and competed-baseline contributing microsegment
        # stock/energy/carbon/cost totals to be updated in the
        # 'compete_adj', 'secondary_adj', and 'htcl_adj' functions
        adj_list_base = [
            adj["cost"]["stock"]["total"]["baseline"],
            adj["cost"]["energy"]["total"]["baseline"],
            adj["cost"]["carbon"]["total"]["baseline"],
            adj["energy"]["total"]["baseline"],
            adj["carbon"]["total"]["baseline"],
            adj["cost"]["stock"]["competed"]["baseline"],
            adj["cost"]["energy"]["competed"]["baseline"],
            adj["cost"]["carbon"]["competed"]["baseline"],
            adj["energy"]["competed"]["baseline"],
            adj["carbon"]["competed"]["baseline"]]
        # Set total-efficient and competed-efficient contributing microsegment
        # stock/energy/carbon/cost totals to be updated in the
        # 'compete_adj', 'secondary_adj', and 'htcl_adj' functions
        adj_list_eff = [
            adj["cost"]["stock"]["total"]["efficient"],
            adj["cost"]["energy"]["total"]["efficient"],
            adj["cost"]["carbon"]["total"]["efficient"],
            adj["energy"]["total"]["efficient"],
            adj["carbon"]["total"]["efficient"],
            adj["cost"]["stock"]["competed"]["efficient"],
            adj["cost"]["energy"]["competed"]["efficient"],
            adj["cost"]["carbon"]["competed"]["efficient"],
            adj["energy"]["competed"]["efficient"],
            adj["carbon"]["competed"]["efficient"]]
        # Set efficient-captured energy data if needed/present
        if eff_capt:
            adj_list_eff.append(adj["energy"]["total"]["efficient-captured"])

        # Add total-baseline and competed-baseline contributing microsegment
        # fugitive emissions totals to the lists above, if applicable

        # Add fugitive emissions from methane if applicable
        if m.fug_e and "methane" in m.fug_e:
            adj_list_base.extend([
                adj["fugitive emissions"]["methane"][
                    "total"]["baseline"],
                adj["fugitive emissions"]["methane"][
                    "competed"]["baseline"]])
            adj_list_eff.extend([
                adj["fugitive emissions"]["methane"][
                    "total"]["efficient"],
                adj["fugitive emissions"]["methane"][
                    "competed"]["efficient"]])
        # Add fugitive emissions from refrigerants if applicable
        if m.fug_e and "refrigerants" in m.fug_e:
            adj_list_base.extend([
                adj["fugitive emissions"]["refrigerants"][
                    "total"]["baseline"],
                adj["fugitive emissions"]["refrigerants"][
                    "competed"]["baseline"]])
            adj_list_eff.extend([
                adj["fugitive emissions"]["refrigerants"][
                    "total"]["efficient"],
                adj["fugitive emissions"]["refrigerants"][
                    "competed"]["efficient"]])

        return mast, adj_out_break, adj, mast_list_base, mast_list_eff, \
            adj_list_eff, adj_list_base, adj_stk_trk

    def compete_adj(
            self, adj_fracs, added_sbmkt_fracs, mast,
            adj_out_break, adj, mast_list_base, mast_list_eff, adj_list_eff,
            adj_list_base, yr, mseg_key, measure, adopt_scheme, mkt_entry_yrs,
            adj_stk_trk):
        """Scale down measure totals to reflect competition.

        Notes:
            Scale stock/energy/carbon/cost totals associated with the current
            contributing market microsegment by the measure's market share for
            this microsegment; reflect these scaled down contributing
            microsegment totals in the measure's overall
            stock/energy/carbon/cost totals.

        Args:
            adj_fracs (dict): Competed market share(s) for the measure.
            added_sbmkt_fracs: Additional market share from sub-market scaling.
            mast (dict): Initial overall stock/energy/carbon/cost totals to
                adjust based on competed market share(s).
            mast_brk_base (dict): Baseline energy use data for the measure/
                microsegment broken out by climate, building, end use.
            mast_brk_eff (dict): Efficient energy use data for the measure/
                microsegment broken out by climate, building, end use.
            mast_brk_save (dict): Energy savings data for the measure/
                microsegment broken out by climate, building, end use.
            adj (dict): Contributing microsegment data to use in adjusting
                overall stock/energy/carbon/cost following competition.
            mast_list_base (dict): Overall 'baseline' scenario
                stock/energy/carbon/cost totals.
            mast_list_eff (dict): Overall 'efficient' scenario
                stock/energy/carbon/cost totals.
            adj_list_eff (dict): Contributing microsegment 'efficient' scenario
                stock/energy/carbon/cost totals.
            adj_list_base (dict): Contributing microsegment 'baseline' scenario
                stock/energy/carbon/cost totals.
            yr (string): Current year in modeling time horizon.
            mseg_key (string): Key for competed market microsegment.
            measure (object): Measure needing competition adjustments.
            adopt_scheme (string): Assumed consumer adoption scenario.
            mkt_entry_yrs (list): Mkt. entry years for all competing measures.
            adj_stk_trk (dict): Stock data used to calculate baseline turnover
                rates and stock-weighted market shares in the given year.
        """
        # Set market shares for the competed stock in the current year, and
        # for the weighted combination of the competed stock for the current
        # and all previous years. Handle this calculation differently for
        # primary and secondary microsegment types

        # Set primary microsegment competed and total weighted market shares

        # Calculate the competed stock market share weight based on turnover
        # in the given year; also calculate the total cumulative captured
        # stock share for the measure up until the current year

        # Competed stock market share (adjustment for current year only)
        adj_c = adj_fracs[yr] + added_sbmkt_fracs[yr]

        # Determine whether efficient-captured energy is being reported
        eff_capt = ("efficient-captured" in adj["energy"]["total"].keys())

        # For non-technical potential cases only, add a flag for measures
        # with market entry years that begin after the minimum market entry
        # year across competing measures. Such measures' efficient data require
        # further adjustment to ensure that the relative performance of the
        # captured stock after applying market shares is consistent with that
        # of the captured stock before the market share scaling was applied
        if adopt_scheme != "Technical potential" and int(
                measure.market_entry_year) > min(mkt_entry_yrs):
            # Add flag
            delay_entry_adj = True
            # Initialize dicts used to make the required adjustment to the
            # measure's efficient data
            rp_adj, save_c, tot_c = ({
                v: 0 for v in ["energy", "carbon", "cost"]} for n in range(3))
            # Initialize tracker of cumulative competed stock (including
            # in years before measure entered market) for use in subsequent
            # measure-captured stock adjustment for measures that enter the
            # market after the minimum market entry year across competing
            # measures; if efficient-captured energy is reported, also
            # initialize trackers of the stock that is measure-captured
            # and competed since measure entered market
            cum_compete_stk = 0
            if eff_capt:
                stk_capt_since_entry, stk_cmp_since_entry = (
                    0 for n in range(2))
        else:
            delay_entry_adj = False
            rp_adj, save_c, tot_c = (None for n in range(3))

        # Weight the market share adjustment for the stock captured by the
        # measure in the current year against that of the stock captured
        # by the measure in all previous years, yielding a total weighted
        # market share adjustment
        if int(yr) < min(mkt_entry_yrs):
            adj_frac_t = adj_fracs[yr] + added_sbmkt_fracs[yr]
        else:
            # Determine the subset of all years leading up to current year in
            # the modeling time horizon
            weighting_yrs = sorted([
                x for x in adj_fracs.keys() if
                (int(x) <= int(yr) and int(x) >= min(mkt_entry_yrs))])

            # Loop through the above set of years, successively updating the
            # weighted market share using a simple moving average
            for ind, wyr in enumerate(weighting_yrs):
                # For non-technical potential cases, calculate the market
                # share weight based on competed stock turnover in the given
                # year, and (if necessary) track the cumulative relative
                # performance of the stock the measure competes for each year.
                # For a technical potential case, market share weight is 1
                # since all stock is competed in each year
                if adopt_scheme != "Technical potential":
                    # Stock turnover rate is ratio of competed to total stock
                    # in given weighting year; handle zero total stock
                    if adj_stk_trk["total"]["all"][wyr] != 0:
                        wt_comp_wyr = (adj_stk_trk["competed"]["all"][wyr] /
                                       adj_stk_trk["total"]["all"][wyr])
                    else:
                        wt_comp_wyr = 0
                    # For measures with delayed market entry, add current year
                    # competed stock value to tracker of cumulative competed
                    # stock (including in years before measure entered market)
                    # and, if efficient-captured energy is reported and measure
                    # has entered market, add current year measure-captured
                    # and competed stock values to trackers of those
                    # variables since measure entered market
                    if delay_entry_adj:
                        cum_compete_stk += adj_stk_trk["competed"]["all"][wyr]
                        # Ensure cumulative competed stock never exceeds
                        # base case total stock projection (competed stock
                        # includes measure-on-measure replacements)
                        if cum_compete_stk > adj_stk_trk["total"]["all"][yr]:
                            cum_compete_stk = adj_stk_trk["total"]["all"][yr]
                        if eff_capt and \
                                int(wyr) >= int(measure.market_entry_year):
                            stk_capt_since_entry += \
                                adj_stk_trk["competed"]["measure"][wyr]
                            stk_cmp_since_entry += \
                                adj_stk_trk["competed"]["all"][wyr]

                    # If needed, update efficient data adjustment for measures
                    # with delayed market entry; adjustment represents the
                    # relative performance of all stock the measure competes
                    # for and captures in the time it is on the market
                    if delay_entry_adj and \
                            adj_stk_trk["total"]["measure"][wyr] > 0:
                        # Calculate the relative performance for each output
                        # metric of interest
                        for v in ["energy", "carbon", "cost"]:
                            # Handle unique cost data structure; focus on
                            # energy costs for this adjustment
                            if v == "cost":
                                # Update numerator in relative performance
                                # calculation (competed-captured savings)
                                save_c[v] += (adj[v][
                                    "energy"]["competed"]["baseline"][wyr] -
                                    adj[v][
                                    "energy"]["competed"]["efficient"][wyr])
                                # Update denominator in relative performance
                                # calculation (baseline competed-captured)
                                tot_c[v] += (adj[v][
                                    "energy"]["competed"]["baseline"][wyr])
                            # Energy/carbon metrics
                            else:
                                # Update numerator in relative performance
                                # calculation (competed-captured savings)
                                save_c[v] += (
                                    adj[v]["competed"]["baseline"][wyr] -
                                    adj[v]["competed"]["efficient"][wyr])
                                # Update denominator in relative performance
                                # calculation (baseline competed-captured)
                                tot_c[v] += (
                                    adj[v]["competed"]["baseline"][wyr])

                else:
                    wt_comp_wyr = 1

                # Set the measure's long-run market share for the current
                # weighting year
                mms_lr = adj_fracs[wyr] + added_sbmkt_fracs[wyr]

                # First year in competed time horizon or any year in a
                # technical potential scenario; weighted market share equals
                # market share for the captured stock in this year only (a
                # "long run" market share value assuming 100% stock turnover)
                if ind == 0 or adopt_scheme == "Technical potential":
                    adj_frac_t = mms_lr
                # Subsequent year for a max adoption potential scenario;
                # weighted market share averages market share for captured
                # stock in current year and all previous years
                else:
                    # Weighted market share equals the "long run" market share
                    # for the current year weighted by the fraction of the
                    # total market that is competed, plus any market share
                    # captured in previous years
                    adj_frac_t = (1 - wt_comp_wyr) * adj_frac_t + \
                        wt_comp_wyr * mms_lr
                    # Ensure that total weighted market share is never above 1
                    if not isinstance(adj_frac_t, numpy.ndarray) and \
                            adj_frac_t > 1:
                        adj_frac_t = 1
                    elif isinstance(adj_frac_t, numpy.ndarray):
                        adj_frac_t[numpy.where(adj_frac_t > 1)] = 1

        # Initialize baseline and efficient data market share adjustment
        # fractions using the overall adjustment fraction calculated above
        adj_t_b, adj_t_e = ({
            v: adj_frac_t for v in [
                "stock", "energy", "energy-captured", "carbon", "cost"]}
            for n in range(2))

        # If necessary, implement adjustment to ensure that measure-captured
        # portion of total stock and relative performance of measure
        # post-market share adjustment is consistent with its measure-captured
        # portion of stock/relative performance pre-market share adjustment
        if delay_entry_adj:
            # Make unique calculation for each output metric of interest
            for var in ["stock", "energy", "carbon", "cost"]:
                if var == "stock":
                    # Develop a factor that adjusts measure-captured stock
                    # values to account for missed competed stock in years
                    # before measure entered market
                    try:
                        b_e_ratio = cum_compete_stk / \
                            adj[var]["total"]["measure"][yr]
                    except (ZeroDivisionError, FloatingPointError):
                        b_e_ratio = 1
                    adj_t_e[var] = adj_t_b[var] * b_e_ratio
                # Cumulative competed relative performance equals cumulatively
                # competed-captured savings for the metric divided by
                # cumulatively competed-captured baseline; handle zero
                # denominator
                else:
                    if tot_c[var] != 0:
                        rp_adj[var] = 1 - (save_c[var] / tot_c[var])
                    else:
                        rp_adj[var] = 1

                    # Since the relative performance adjustment is calculated
                    # relative to baseline data, develop a ratio that first
                    # translates efficient to baseline values, before the
                    # relative performance factor is applied

                    # Handle cost data structure separately, focus on energy
                    # costs
                    if var == "cost":
                        try:
                            b_e_ratio = \
                                adj[var]["energy"]["total"]["baseline"][yr] / \
                                adj[var]["energy"]["total"]["efficient"][yr]
                        except (ZeroDivisionError, FloatingPointError):
                            b_e_ratio = 1
                    # Energy/carbon data
                    else:
                        try:
                            b_e_ratio = adj[var]["total"]["baseline"][yr] / \
                                adj[var]["total"]["efficient"][yr]
                        except (ZeroDivisionError, FloatingPointError):
                            b_e_ratio = 1

                    # Ensure that calculated ratio is a finite number
                    if (isinstance(b_e_ratio, numpy.ndarray) and not all(
                        numpy.isfinite(b_e_ratio))) or (
                        not isinstance(b_e_ratio, numpy.ndarray) and not
                            numpy.isfinite(b_e_ratio)):
                        b_e_ratio = 1

                    # Further scale efficient market share adjustment fraction
                    # on the basis of the factors calculated above
                    adj_t_e[var] = adj_t_b[var] * b_e_ratio * rp_adj[var]

                    # In the special case of efficient energy use where
                    # captured efficient energy is reported, develop a factor
                    # that adjusts efficient-captured energy values to account
                    # for missed competed stock in years before the measure was
                    # on the market
                    if var == "energy" and eff_capt:
                        try:
                            # Ratio to adjust efficient-captured to efficient
                            # energy total
                            e_ec_ratio = adj[var]["total"]["efficient"][yr] / \
                                adj[var]["total"]["efficient-captured"][yr]
                            # Ratio to determine what portion of efficient
                            # energy was captured by measure since market entry
                            # based on stock totals
                            stk_capt_since_entry_ratio = \
                                stk_capt_since_entry / stk_cmp_since_entry
                        except (ZeroDivisionError, FloatingPointError):
                            e_ec_ratio, stk_capt_since_entry_ratio = 1, 1
                        # Final ratio adjusts efficient-captured data to
                        # efficient data and then scales down based on what
                        # portion of efficient is measure-captured
                        adj_t_e["energy-captured"] = \
                            adj_t_e[var] * e_ec_ratio * \
                            stk_capt_since_entry_ratio

        # For a primary microsegment with secondary effects, record market
        # share information that will subsequently be used to adjust associated
        # secondary microsegments and associated energy/carbon/cost totals
        if len(measure.markets[adopt_scheme]["competed"]["mseg_adjust"][
                "secondary mseg adjustments"]["market share"][
                "original energy (total captured)"].keys()) > 0:
            # Determine the climate zone, building type, and structure
            # type for the current contributing primary microsegment from the
            # microsegment key chain information and use as the key for linking
            # the primary and its associated secondary microsegment
            cz_bldg_struct = literal_eval(mseg_key)
            secnd_mseg_adjkey = str((
                cz_bldg_struct[1], cz_bldg_struct[2], cz_bldg_struct[-1]))

            if secnd_mseg_adjkey in measure.markets[adopt_scheme][
                "competed"]["mseg_adjust"][
                "secondary mseg adjustments"]["market share"][
                    "original energy (total captured)"].keys():
                # Record original and adjusted primary stock numbers as part of
                # the measure's 'mseg_adjust' attribute
                secnd_adj_mktshr = measure.markets[adopt_scheme][
                    "competed"]["mseg_adjust"]["secondary mseg adjustments"][
                    "market share"]
                # Total captured energy
                secnd_adj_mktshr["original energy (total captured)"][
                    secnd_mseg_adjkey][yr] += \
                    adj["energy"]["total"]["efficient"][yr]
                # Competed and captured energy
                secnd_adj_mktshr["original energy (competed and captured)"][
                    secnd_mseg_adjkey][yr] += \
                    adj["energy"]["competed"]["efficient"][yr]
                # Adjusted total captured energy
                secnd_adj_mktshr["adjusted energy (total captured)"][
                    secnd_mseg_adjkey][yr] += (
                        adj["energy"]["total"]["efficient"][yr] *
                        adj_t_e["stock"])
                # Adjusted competed and captured energy
                secnd_adj_mktshr["adjusted energy (competed and captured)"][
                    secnd_mseg_adjkey][yr] += (
                        adj["energy"]["competed"]["efficient"][yr] * adj_c)

        # Adjust baseline energy/cost/carbon, efficient energy/cost/carbon, and
        # energy/cost/carbon savings totals grouped by climate zone, building
        # type, end use, and (if applicable) fuel type by the appropriate
        # fraction; adjust based on segment of current microsegment that was
        # removed from competition
        for var in ["energy", "cost", "carbon"]:
            # Update baseline and efficient results provided
            # neither is None or all zeros
            vs_list = [v if (
                adj_out_break["base fuel"][var][v] is not None and (
                    (not isinstance(adj_out_break["base fuel"][
                        var][v][yr], numpy.ndarray) and any([
                            adj_out_break["base fuel"][var][v][yr] != 0])) or
                    (isinstance(
                        adj_out_break["base fuel"][var][v][yr], numpy.ndarray)
                     and any([any([adj_out_break[
                        "base fuel"][var][v][yr] != 0])])) for yr in
                    adj_out_break["base fuel"][var][v].keys()))
                else "" for v in ["baseline", "efficient"]]
            # Energy data may include unique efficient captured tracking
            # if efficient breakout data are present
            if "efficient" in vs_list and var == "energy" and \
                    "efficient-captured" in adj_out_break[
                    "base fuel"]["energy"].keys():
                vs_list.append("efficient-captured")
            for var_sub in [x for x in vs_list if x]:
                # Adjustment fraction unique to baseline/efficient results
                if var_sub == "baseline":
                    adj_t = adj_t_b[var]
                else:
                    # Draw unique adjustment fraction for efficient-captured
                    # energy results
                    if var == "energy" and var_sub == "efficient-captured":
                        adj_t = adj_t_e["energy-captured"]
                    else:
                        adj_t = adj_t_e[var]

                # Select correct fuel split data
                if var_sub != "baseline":
                    fs_eff_splt_var = adj_out_break[
                        "efficient fuel splits"][var][yr]
                else:
                    fs_eff_splt_var = 1

                # Handle extra key on the adjusted microsegment data for the
                # cost variables ("energy")
                if var == "cost":
                    adj_out_break["base fuel"][var][var_sub][yr] = \
                        adj_out_break["base fuel"][var][var_sub][yr] - (
                        adj[var]["energy"]["total"][var_sub][yr]) * (
                        1 - adj_t) * fs_eff_splt_var
                else:
                    # Handle efficient-captured energy case for fuel switching,
                    # where no base fuel data will be reported (skip to next
                    # variable)
                    try:
                        adj_out_break["base fuel"][var][var_sub][yr] = \
                            adj_out_break["base fuel"][var][var_sub][yr] - (
                            adj[var]["total"][var_sub][yr]) * (
                                1 - adj_t) * fs_eff_splt_var
                    except KeyError:
                        continue

            # Update savings results
            # Handle extra key on the adjusted microsegment data for the cost
            # variables ("energy")
            if var == "cost":
                adj_out_break["base fuel"][var]["savings"][yr] = \
                    adj_out_break["base fuel"][var]["savings"][yr] - ((
                        adj[var]["energy"]["total"]["baseline"][yr] * (
                            1 - adj_t_b[var]) -
                        adj[var]["energy"]["total"]["efficient"][yr] * (
                            1 - adj_t_e[var]) * adj_out_break[
                            "efficient fuel splits"][var][yr]))
            else:
                adj_out_break["base fuel"][var]["savings"][yr] = \
                    adj_out_break["base fuel"][var]["savings"][yr] - ((
                        adj[var]["total"]["baseline"][yr] * (
                            1 - adj_t_b[var]) -
                        adj[var]["total"]["efficient"][yr] * (
                            1 - adj_t_e[var]) * adj_out_break[
                            "efficient fuel splits"][var][yr]))

            # If the measure involves fuel switching and the user has broken
            # out results by fuel type, make adjustments to the efficient, and
            # savings results for the switched to fuel
            if adj_out_break["switched fuel"][var]["efficient"] is not None:
                # Handle extra key on the adjusted microsegment data for
                # the cost variables ("energy")
                if var == "cost":
                    # Update efficient result
                    adj_out_break["switched fuel"][var]["efficient"][yr] = \
                        adj_out_break["switched fuel"][var][
                            "efficient"][yr] - (
                        adj[var]["energy"]["total"][
                            "efficient"][yr]) * (
                        1 - adj_t_e[var]) * (1 - adj_out_break[
                            "efficient fuel splits"][var][yr])
                    # Update savings result; note that savings
                    # for a switched to fuel will be negative and
                    # thus the adjustment to microsegment data
                    # post-competition should be added to the
                    # original savings breakout results
                    adj_out_break["switched fuel"][var]["savings"][yr] = \
                        adj_out_break["switched fuel"][var][
                            "savings"][yr] + (
                        adj[var]["energy"]["total"][
                            "efficient"][yr]) * (
                        1 - adj_t_e[var]) * (1 - adj_out_break[
                            "efficient fuel splits"][var][yr])
                else:
                    # Update efficient result
                    # Energy data may include efficient-captured tracking
                    if var == "energy" and "efficient-captured" in \
                        adj_out_break["switched fuel"][
                            "energy"].keys():
                        vs_list = ["efficient", "efficient-captured"]
                    else:
                        vs_list = ["efficient"]
                    # Loop through efficient and (if applicable) efficient-
                    # captured data and update
                    for var_sub in vs_list:
                        if var_sub == "efficient":
                            # Efficient competition adjustment fraction
                            adj_t = adj_t_e[var]
                            # Efficient fuel splits
                            fs_eff_splt_var = adj_out_break[
                                "efficient fuel splits"][var][yr]
                        elif var_sub == "efficient-captured":
                            # Efficient-captured comp. adj. fraction
                            adj_t = adj_t_e["energy-captured"]
                            # Efficient fuel splits
                            fs_eff_splt_var = adj_out_break[
                                "efficient-captured fuel splits"][var][yr]
                        adj_out_break["switched fuel"][var][
                            var_sub][yr] = adj_out_break[
                            "switched fuel"][var][var_sub][yr] - (
                            adj[var]["total"][var_sub][yr]) * (
                            1 - adj_t) * (1 - fs_eff_splt_var)
                    # Update savings result
                    adj_out_break["switched fuel"][var]["savings"][yr] = \
                        adj_out_break["switched fuel"][var][
                            "savings"][yr] + (
                        adj[var]["total"]["efficient"][yr]) * (
                        1 - adj_t) * (1 - adj_out_break[
                            "efficient fuel splits"][var][yr])

        # Adjust the total and competed baseline stock captured, both overall
        # and for the current contributing microsegment

        # Overall total baseline stock
        mast["stock"]["total"]["all"][yr] = \
            mast["stock"]["total"]["all"][yr] - \
            (adj["stock"]["total"]["all"][yr] * (1 - adj_t_b["stock"]))
        # Competed total baseline stock
        mast["stock"]["competed"]["all"][yr] = \
            mast["stock"]["competed"]["all"][yr] - \
            (adj["stock"]["competed"]["all"][yr] * (1 - adj_c))
        # Current contributing mseg total baseline stock
        adj["stock"]["total"]["all"][yr] = \
            adj["stock"]["total"]["all"][yr] * adj_t_b["stock"]
        # Current contributing mseg competed baseline stock
        adj["stock"]["competed"]["all"][yr] = \
            adj["stock"]["competed"]["all"][yr] * adj_c

        # Adjust the total and competed stock captured by the measure and
        # associated measure and base-case cost totals for that captured
        # stock by the appropriate measure market share, both overall and
        # for the current contributing microsegment.

        # Overall total measure stock
        mast["stock"]["total"]["measure"][yr] = \
            mast["stock"]["total"]["measure"][yr] - \
            adj["stock"]["total"]["measure"][yr] * (1 - adj_t_e["stock"])
        # Overall total baseline stock cost
        mast["cost"]["stock"]["total"]["baseline"][yr] = \
            mast["cost"]["stock"]["total"]["baseline"][yr] - \
            adj["cost"]["stock"]["total"]["baseline"][yr] * (
                1 - adj_t_b["stock"])
        # Overall total measure stock cost
        mast["cost"]["stock"]["total"]["efficient"][yr] = \
            mast["cost"]["stock"]["total"]["efficient"][yr] - \
            adj["cost"]["stock"]["total"]["efficient"][yr] * (
                1 - adj_t_e["stock"])
        # Overall competed measure stock
        mast["stock"]["competed"]["measure"][yr] = \
            mast["stock"]["competed"]["measure"][yr] - \
            adj["stock"]["competed"]["measure"][yr] * (1 - adj_c)
        # Overall competed baseline stock cost
        mast["cost"]["stock"]["competed"]["baseline"][yr] = \
            mast["cost"]["stock"]["competed"]["baseline"][yr] - \
            adj["cost"]["stock"]["competed"]["baseline"][yr] * (1 - adj_c)
        # Overall competed measure stock cost
        mast["cost"]["stock"]["competed"]["efficient"][yr] = \
            mast["cost"]["stock"]["competed"]["efficient"][yr] - \
            adj["cost"]["stock"]["competed"]["efficient"][yr] * (
                1 - adj_c)
        # Current contributing mseg total measure stock
        adj["stock"]["total"]["measure"][yr] = \
            adj["stock"]["total"]["measure"][yr] * adj_t_e["stock"]
        # Current contributing mseg total baseline stock cost
        adj["cost"]["stock"]["total"]["baseline"][yr] = \
            adj["cost"]["stock"]["total"]["baseline"][yr] * \
            adj_t_b["stock"]
        # Current contributing mseg total measure stock cost
        adj["cost"]["stock"]["total"]["efficient"][yr] = \
            adj["cost"]["stock"]["total"]["efficient"][yr] * \
            adj_t_e["stock"]
        # Current contributing mseg competed measure stock
        adj["stock"]["competed"]["measure"][yr] = \
            adj["stock"]["competed"]["measure"][yr] * adj_c
        # Current contributing mseg competed baseline stock cost
        adj["cost"]["stock"]["competed"]["baseline"][yr] = \
            adj["cost"]["stock"]["competed"]["baseline"][yr] * adj_c
        # Current contributing mseg competed measure stock cost
        adj["cost"]["stock"]["competed"]["efficient"][yr] = \
            adj["cost"]["stock"]["competed"]["efficient"][yr] * adj_c

        # Adjust total and competed baseline and efficient energy, carbon,
        # and cost data by measure market share
        for x in ["baseline", "efficient"]:
            # Determine appropriate adjustment data to use
            # for baseline or efficient case
            if x == "baseline":
                mastlist, adjlist = [mast_list_base, adj_list_base]
                adj_t = adj_t_b
            else:
                mastlist, adjlist = [mast_list_eff, adj_list_eff]
                adj_t = adj_t_e

            # Adjust the total and competed energy, carbon, and associated cost
            # savings by the appropriate measure market share, both overall
            # and for the current contributing microsegment

            # Adjust total energy/carbon costs using energy cost adj factors
            mast["cost"]["energy"]["total"][x][yr], \
                mast["cost"]["carbon"]["total"][x][yr] = [
                x[yr] - (y[yr] * (1 - adj_t["cost"])) for x, y in
                zip(mastlist[1:3], adjlist[1:3])]
            adj["cost"]["energy"]["total"][x][yr], \
                adj["cost"]["carbon"]["total"][x][yr] = [
                (x[yr] * adj_t["cost"]) for x in adjlist[1:3]]
            # Adjust energy/carbon
            mast["energy"]["total"][x][yr], \
                mast["carbon"]["total"][x][yr] = [
                x[yr] - (y[yr] * (1 - adj_t[v])) for x, y, v in
                zip(mastlist[3:5], adjlist[3:5], ["energy", "carbon"])]
            adj["energy"]["total"][x][yr], \
                adj["carbon"]["total"][x][yr] = [
                (x[yr] * adj_t[v]) for x, v in zip(
                    adjlist[3:5], ["energy", "carbon"])]
            # Adjust efficient-captured energy if these data are present
            if x == "efficient":
                try:
                    mast["energy"]["total"]["efficient-captured"][yr] = \
                        mastlist[10][yr] - (
                        adjlist[10][yr] * (1 - adj_t["energy-captured"]))
                    adj["energy"]["total"]["efficient-captured"][yr] = (
                        adjlist[10][yr] * adj_t["energy-captured"])
                except (KeyError, IndexError):
                    pass

            # Adjust total emissions from fugitive emissions if applicable
            if measure.fug_e:
                # Adjust fugitive methane emissions results if applicable
                if measure.fug_e and "methane" in measure.fug_e:
                    # Total
                    mast["fugitive emissions"]["methane"][
                        "total"][x][yr] = mastlist[-4][yr] - (
                            adjlist[-4][yr] * (1 - adj_t["energy"]))
                    adj["fugitive emissions"]["methane"][
                        "total"][x][yr] = adjlist[-4][yr] * adj_t["energy"]
                    # Competed
                    mast["fugitive emissions"]["methane"][
                        "competed"][x][yr] = mastlist[-3][yr] - (
                            adjlist[-3][yr] * (1 - adj_c))
                    adj["fugitive emissions"]["methane"]["competed"][x][yr] = \
                        adjlist[-3][yr] * adj_c
                # Adjust fugitive refrigerant emissions results if
                # applicable
                if measure.fug_e and "refrigerants" in measure.fug_e:
                    # Total
                    mast["fugitive emissions"]["refrigerants"][
                        "total"][x][yr] = mastlist[-2][yr] - (
                            adjlist[-2][yr] * (1 - adj_t_b["stock"]))
                    adj["fugitive emissions"]["refrigerants"][
                        "total"][x][yr] = adjlist[-2][yr] * adj_t_b["stock"]
                    # Competed
                    mast["fugitive emissions"]["refrigerants"][
                        "competed"][x][yr] = mastlist[-1][yr] - (
                            adjlist[-1][yr] * (1 - adj_c))
                    adj["fugitive emissions"]["refrigerants"][
                        "competed"][x][yr] = adjlist[-1][yr] * adj_c

            # Adjust competed energy/carbon and energy/carbon costs
            mast["cost"]["energy"]["competed"][x][yr], \
                mast["cost"]["carbon"]["competed"][x][yr], \
                mast["energy"]["competed"][x][yr], \
                mast["carbon"]["competed"][x][yr] = [
                    x[yr] - (y[yr] * (1 - adj_c)) for x, y in
                    zip(mastlist[6:10], adjlist[6:10])]
            adj["cost"]["energy"]["competed"][x][yr], \
                adj["cost"]["carbon"]["competed"][x][yr], \
                adj["energy"]["competed"][x][yr], \
                adj["carbon"]["competed"][x][yr] = [
                    (x[yr] * adj_c) for x in adjlist[6:10]]

    def finalize_outputs(
            self, adopt_scheme, trim_out, trim_yrs):
        """Prepare selected measure outputs to write to a summary JSON file.

        Args:
            adopt_scheme (string): Consumer adoption scenario to summarize
                outputs for.
            trim_out (boolean): Flag for trimmed down results file.
            trim_yrs (list): Optional list of years to focus results on.
        """
        # Initialize markets and savings totals across all ECMs
        summary_vals_all_ecms = [{
            yr: 0 for yr in self.handyvars.aeo_years} for n in range(12)]
        # Initialize fugitive markets and savings totals across all ECMs
        # as None (re-initialized below in case where fugitive emissions are
        # assessed)
        summary_vals_all_ecms_f_e = None
        # Set up subscript translator for carbon variable strings
        sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        # If user has specified a reduced results file size, check for whether
        # years of focus should be used
        if trim_yrs is not False:
            focus_yrs = [str(x) for x in trim_yrs]
        else:
            focus_yrs = self.handyvars.aeo_years
        # Initialize markets and savings totals across all ECMs

        # Set total number of market variables that could be reported across ECMs,
        # determined as the number of variables in summary_vals other than the
        # 'metrics_finance' variables
        n_vars_all = 13
        # Initialize summary variable values at zero
        summary_vals_all_ecms = [{
            yr: 0 for yr in focus_yrs} for n in range(n_vars_all)]
        # Initialize variable that aggregates total and incremental
        # stock cost for deploying measures across all measures, provided
        # the user has chosen to report those data. Structure this variable
        # as a list of two dicts, where the first dict will store total cost
        # data and the second will store incremental cost data
        if self.opts.report_stk is True:
            stk_cost_all_ecms = [{yr: 0 for yr in focus_yrs} for n in range(2)]
            # Set stock cost units (common to all ECMs)
            stk_cost_key_tot, stk_cost_key_inc = [
                x + self.handyvars.common_cost_yr + "$)" for
                x in ["Total Measure Stock Cost (",
                      "Incremental Measure Stock Cost ("]]
        else:
            stk_cost_all_ecms, stk_cost_key_tot, stk_cost_key_inc = (
                "" for n in range(3))
        # Loop through all measures and populate above dict of summary outputs
        for m in self.measures:
            # Set competed measure markets and savings and financial metrics
            mkts = m.markets[adopt_scheme]["competed"]["master_mseg"]
            # Set shorthand for efficient-captured market data if these are
            # available, and if they are not, set shorthand to None
            eff_capt = mkts["energy"]["total"].get("efficient-captured", "")
            save = m.savings[adopt_scheme]["competed"]
            metrics_finance = m.financial_metrics

            # Group baseline/efficient markets, savings, and financial
            # metrics into list for updates
            summary_vals = [
                mkts["energy"]["total"]["baseline"],
                mkts["carbon"]["total"]["baseline"],
                mkts["cost"]["energy"]["total"]["baseline"],
                mkts["cost"]["carbon"]["total"]["baseline"],
                mkts["energy"]["total"]["efficient"],
                eff_capt,
                mkts["carbon"]["total"]["efficient"],
                mkts["cost"]["energy"]["total"]["efficient"],
                mkts["cost"]["carbon"]["total"]["efficient"],
                save["energy"]["savings"],
                save["energy"]["cost savings"],
                save["carbon"]["savings"],
                save["carbon"]["cost savings"],
                metrics_finance["cce"],
                metrics_finance["cce (w/ carbon cost benefits)"],
                metrics_finance["ccc"],
                metrics_finance["ccc (w/ energy cost benefits)"],
                metrics_finance["irr (w/ energy costs)"],
                metrics_finance["irr (w/ energy and carbon costs)"],
                metrics_finance["payback (w/ energy costs)"],
                metrics_finance["payback (w/ energy and carbon costs)"]]
            # Order the year entries in the above markets, savings,
            # and portfolio metrics outputs
            summary_vals_init = [OrderedDict(
                sorted(x.items())) for x in summary_vals]
            # Apply focus year range, if applicable
            summary_vals = [{
                yr: summary_vals_init[v][yr] for yr in focus_yrs}
                for v in range(len(summary_vals))]
            # Add ECM markets and savings totals to totals across all ECMs
            summary_vals_all_ecms = [{
                yr: summary_vals_all_ecms[v][yr] + summary_vals[v][yr] for
                yr in focus_yrs} for v in range(0, n_vars_all)]

            # If fugitive emissions are being assessed for the measure,
            # initialize a summary list for those data
            if m.fug_e:
                # Set baseline and efficient fugitive emissions results
                summary_vals_f_e = [
                    mkts["fugitive emissions"][x]["total"][y] for x in
                        ["methane", "refrigerants"] for y in
                        ["baseline", "efficient"]]
                # Set fugitive emissions savings results
                summary_vals_f_e.extend([
                    save["fugitive emissions"][x]["savings"] for x in
                    ["methane", "refrigerants"]])
                # Order the year entries in the above markets, savings,
                # and portfolio metrics outputs
                summary_vals_init_f_e = [OrderedDict(
                    sorted(x.items())) if x is not None else None
                    for x in summary_vals_f_e]
                # Apply focus year range, if applicable
                for ind, v in enumerate(summary_vals_f_e):
                    # Exclude None values from operation (e.g.,
                    # when methane is assessed without refrigerants,
                    # refrigerant values will be None and vice versa)
                    if v is not None:
                        summary_vals_f_e[ind] = {
                            yr: summary_vals_init_f_e[ind][yr]
                            for yr in focus_yrs}
                # If summary list of fugitive emissions results across all
                # ECMs has not already been re-initialized, do so while adding
                # the fugitive emissions data for the current measure
                if summary_vals_all_ecms_f_e is None:
                    summary_vals_all_ecms_f_e = []
                    for ind, v in enumerate(summary_vals_f_e):
                        # Exclude None values from operation
                        if v is not None:
                            summary_vals_all_ecms_f_e.append({
                                yr: summary_vals_f_e[ind][yr] for
                                yr in focus_yrs})
                        else:
                            summary_vals_all_ecms_f_e.append(v)
                # Otherwise, add fugitive emissions data for the current
                # measure to the previously initialized sum of fugitive
                # emissions data across all ECMs
                else:
                    for ind, v in enumerate(summary_vals_f_e):
                        # Exclude None values from operation
                        if v is not None:
                            summary_vals_all_ecms_f_e[ind] = {
                                yr: (summary_vals_all_ecms_f_e[ind][yr] +
                                     summary_vals_f_e[ind][yr]) for
                                yr in focus_yrs}
            # Find mean and 5th/95th percentile values of each output
            # (note: if output is point value, all three of these values
            # will be the same)

            # Mean of outputs
            energy_base_avg, carb_base_avg, energy_cost_base_avg, \
                carb_cost_base_avg, energy_eff_avg, energy_eff_capt_avg, carb_eff_avg, \
                energy_cost_eff_avg, carb_cost_eff_avg, energy_save_avg, \
                energy_costsave_avg, carb_save_avg, carb_costsave_avg, \
                cce_avg, cce_c_avg, ccc_avg, ccc_e_avg, \
                irr_e_avg, irr_ec_avg, payback_e_avg, \
                payback_ec_avg = [{
                    k: numpy.mean(v) if v is not None else None
                    for k, v in z.items()} for z in summary_vals]
            # 5th percentile of outputs
            energy_base_low, carb_base_low, energy_cost_base_low, \
                carb_cost_base_low, energy_eff_low, energy_eff_capt_low, carb_eff_low, \
                energy_cost_eff_low, carb_cost_eff_low, energy_save_low, \
                energy_costsave_low, carb_save_low, carb_costsave_low, \
                cce_low, cce_c_low, ccc_low, ccc_e_low, \
                irr_e_low, irr_ec_low, payback_e_low, payback_ec_low = [{
                    k: numpy.percentile(v, 5) if v is not None else None
                    for k, v in z.items()} for z in summary_vals]
            # 95th percentile of outputs
            energy_base_high, carb_base_high, energy_cost_base_high, \
                carb_cost_base_high, energy_eff_high, energy_eff_capt_high, carb_eff_high, \
                energy_cost_eff_high, carb_cost_eff_high, energy_save_high, \
                energy_costsave_high, carb_save_high, carb_costsave_high, \
                cce_high, cce_c_high, ccc_high, ccc_e_high, \
                irr_e_high, irr_ec_high, payback_e_high, payback_ec_high = [{
                    k: numpy.percentile(v, 95) if v is not None else None
                    for k, v in z.items()} for z in summary_vals]

            # Record updated markets and savings in Engine 'output'
            # attribute; initialize markets/savings breakouts by category as
            # total markets/savings (e.g., not broken out in any way). These
            # initial values will be adjusted by breakout fractions below.
            # If user desires pared down outputs, only report savings values.
            if trim_out is False:
                self.output_ecms[m.name]["Markets and Savings (Overall)"][
                    adopt_scheme], self.output_ecms[m.name][
                        "Markets and Savings (by Category)"][
                        adopt_scheme] = (OrderedDict([
                            ("Baseline Energy Use (MMBtu)", energy_base_avg),
                            ("Efficient Energy Use (MMBtu)", energy_eff_avg),
                            ("Baseline CO2 Emissions (MMTons)".translate(sub),
                                carb_base_avg),
                            ("Efficient CO2 Emissions (MMTons)".translate(sub),
                                carb_eff_avg),
                            ("Baseline Energy Cost (USD)",
                             energy_cost_base_avg),
                            ("Efficient Energy Cost (USD)",
                             energy_cost_eff_avg),
                            ("Baseline CO2 Cost (USD)".translate(sub),
                                carb_cost_base_avg),
                            ("Efficient CO2 Cost (USD)".translate(sub),
                                carb_cost_eff_avg),
                            ("Energy Savings (MMBtu)", energy_save_avg),
                            ("Energy Cost Savings (USD)", energy_costsave_avg),
                            ("Avoided CO2 Emissions (MMTons)".
                                translate(sub), carb_save_avg),
                            ("CO2 Cost Savings (USD)".
                                translate(sub), carb_costsave_avg)]) for
                            n in range(2))
                # Add efficient-captured data to reporting if present
                if eff_capt and energy_eff_capt_avg is not None:
                    self.output_ecms[m.name][
                        "Markets and Savings (Overall)"][adopt_scheme][
                        "Efficient Energy Use, Measure (MMBtu)"], \
                        self.output_ecms[m.name][
                            "Markets and Savings (by Category)"][
                            adopt_scheme][
                            "Efficient Energy Use, Measure (MMBtu)"] = (
                                energy_eff_capt_avg for n in range(2))

                # Record updated (post-competed) fugitive emissions results
                # for individual ECM if applicable
                if m.fug_e:
                    # Record updated baseline/efficient methane results
                    self.output_ecms[m.name]["Markets and Savings (Overall)"][
                        adopt_scheme][
                        "Baseline Fugitive Methane (MMTons CO2e)"], \
                        self.output_ecms[m.name][
                            "Markets and Savings (Overall)"][adopt_scheme][
                        "Efficient Fugitive Methane (MMTons CO2e)"] = \
                        summary_vals_f_e[0:2]
                    # Record updated methane savings results
                    self.output_ecms[m.name]["Markets and Savings (Overall)"][
                        adopt_scheme][
                        "Fugitive Methane Savings (MMTons CO2e)"] = \
                        summary_vals_f_e[4]
                    # Record updated baseline/efficient refrigerant results
                    self.output_ecms[m.name]["Markets and Savings (Overall)"][
                        adopt_scheme][
                        "Baseline Fugitive Refrigerants (MMTons CO2e)"], \
                        self.output_ecms[m.name][
                            "Markets and Savings (Overall)"][adopt_scheme][
                        "Efficient Fugitive Refrigerants (MMTons CO2e)"] = \
                        summary_vals_f_e[2:4]
                    # Record updated refrigerant savings results
                    self.output_ecms[m.name]["Markets and Savings (Overall)"][
                        adopt_scheme][
                        "Fugitive Refrigerants Savings (MMTons CO2e)"] = \
                        summary_vals_f_e[5]

                # Record list of baseline variable names for use in finalizing
                # output breakouts below
                mkt_base_keys = [
                    "Baseline Energy Use (MMBtu)",
                    "Baseline CO2 Emissions (MMTons)".translate(sub),
                    "Baseline Energy Cost (USD)",
                    "Baseline CO2 Cost (USD)".translate(sub)]
                # Record list of efficient variable names for use in finalizing
                # output breakouts below
                mkt_eff_keys = [
                    "Efficient Energy Use (MMBtu)",
                    "Efficient CO2 Emissions (MMTons)".translate(sub),
                    "Efficient Energy Cost (USD)",
                    "Efficient CO2 Cost (USD)".translate(sub)]
                # Add efficient-captured to efficient breakout names if present
                if eff_capt:
                    mkt_eff_keys.append(
                        "Efficient Energy Use, Measure (MMBtu)")
                # Record list of savings variable names for use in finalizing
                # output breakouts below
                save_keys = [
                    "Energy Savings (MMBtu)",
                    "Avoided CO2 Emissions (MMTons)".translate(sub),
                    "Energy Cost Savings (USD)",
                    "CO2 Cost Savings (USD)".translate(sub)]
            else:
                self.output_ecms[m.name]["Markets and Savings (Overall)"][
                    adopt_scheme], self.output_ecms[m.name][
                        "Markets and Savings (by Category)"][
                        adopt_scheme] = (OrderedDict([
                            ("Baseline Energy Use (MMBtu)", energy_base_avg),
                            ("Efficient Energy Use (MMBtu)", energy_eff_avg),
                            ("Baseline CO2 Emissions (MMTons)".translate(sub),
                                carb_base_avg),
                            ("Efficient CO2 Emissions (MMTons)".translate(sub),
                                carb_eff_avg),
                            ("Energy Savings (MMBtu)", energy_save_avg),
                            ("Avoided CO2 Emissions (MMTons)".
                                translate(sub), carb_save_avg)
                            ]) for n in range(2))
                if eff_capt and energy_eff_capt_avg is not None:
                    self.output_ecms[m.name][
                        "Markets and Savings (Overall)"][adopt_scheme][
                        "Efficient Energy Use, Measure (MMBtu)"], \
                        self.output_ecms[m.name][
                            "Markets and Savings (by Category)"][
                            adopt_scheme][
                            "Efficient Energy Use, Measure (MMBtu)"] = (
                                energy_eff_capt_avg for n in range(2))
                # Record list of baseline variable names for use in finalizing
                # output breakouts below
                mkt_base_keys = [
                    "Baseline Energy Use (MMBtu)",
                    "Baseline CO2 Emissions (MMTons)".translate(sub)]
                # Record list of efficient variable names for use in finalizing
                # output breakouts below
                mkt_eff_keys = [
                    "Efficient Energy Use (MMBtu)",
                    "Efficient CO2 Emissions (MMTons)".translate(sub)]
                # Add efficient-captured to efficient breakout names if present
                if eff_capt:
                    mkt_eff_keys.append(
                        "Efficient Energy Use, Measure (MMBtu)")
                # Record list of savings variable names for use in finalizing
                # output breakouts below
                save_keys = [
                    "Energy Savings (MMBtu)",
                    "Avoided CO2 Emissions (MMTons)".translate(sub)]

                # Record updated (post-competed) fugitive emissions results
                # for individual ECM if applicable
                if m.fug_e:
                    # Record updated baseline/efficient methane results
                    self.output_ecms[m.name]["Markets and Savings (Overall)"][
                        adopt_scheme][
                        "Baseline Fugitive Methane (MMTons CO2e)"], \
                        self.output_ecms[m.name][
                            "Markets and Savings (Overall)"][adopt_scheme][
                        "Efficient Fugitive Methane (MMTons CO2e)"] = \
                        summary_vals_f_e[0:2]
                    # Record updated methane savings results
                    self.output_ecms[m.name]["Markets and Savings (Overall)"][
                        adopt_scheme][
                        "Fugitive Methane Savings (MMTons CO2e)"] = \
                        summary_vals_f_e[4]
                    # Record updated baseline/efficient refrigerant results
                    self.output_ecms[m.name]["Markets and Savings (Overall)"][
                        adopt_scheme][
                        "Baseline Fugitive Refrigerants (MMTons CO2e)"], \
                        self.output_ecms[m.name][
                            "Markets and Savings (Overall)"][adopt_scheme][
                        "Efficient Fugitive Refrigerants (MMTons CO2e)"] = \
                        summary_vals_f_e[2:4]
                    # Record updated refrigerant savings results
                    self.output_ecms[m.name]["Markets and Savings (Overall)"][
                        adopt_scheme][
                        "Fugitive Refrigerants Savings (MMTons CO2e)"] = \
                        summary_vals_f_e[5]

            # If competition adjustment fractions must be reported, find/store
            # those data
            if self.opts.report_cfs is True:
                # Loop through and pull energy/energy cost/carbon factors
                for k in ["energy", "cost", "carbon"]:
                    for mt in ["baseline", "efficient", "savings"]:
                        dat_c, dat_uc = [
                            m.markets[adopt_scheme][scn][
                                "mseg_out_break"][k][mt] for scn in [
                                "competed", "uncompeted"]]
                        # Loop through all regions, building types, and
                        # end uses in the output breakout data to calculate
                        # the competition factors
                        for reg in self.handyvars.out_break_czones:
                            # Initialize competed/uncompeted data dicts
                            tot_uc, tot_c = ({yr: 0 for yr in focus_yrs} for
                                             n in range(2))
                            for b in self.handyvars.out_break_bldgtypes:
                                for eu in self.handyvars.out_break_enduses:
                                    # Ensure that data exist for the
                                    # current combination of region,
                                    # building type and end use
                                    try:
                                        dat_c_ms, dat_uc_ms = [
                                            dat_c[reg][b][eu],
                                            dat_uc[reg][b][eu]]
                                    except KeyError:
                                        continue
                                    # Loop through focus years and pull
                                    # data to add to post-competition
                                    # totals; handle case where data are
                                    # split by fuel type
                                    try:
                                        # Competed totals
                                        tot_c = {
                                            yr: tot_c[yr] + dat_c_ms[yr]
                                            for yr in focus_yrs}
                                        # Uncompeted totals
                                        tot_uc = {
                                            yr: tot_uc[yr] + dat_uc_ms[yr]
                                            for yr in focus_yrs}
                                    except KeyError:
                                        try:
                                            dat_c_ms_add, dat_uc_ms_add = [{
                                                yr: sum([d[x][yr] if yr in
                                                         d[x].keys() else 0 for
                                                         x in ["Electric",
                                                               "Non-Electric"]
                                                         ])
                                                for yr in focus_yrs} for d in [
                                                    dat_c_ms, dat_uc_ms]]
                                            # Competed totals
                                            tot_c = {
                                                yr: tot_c[yr] +
                                                dat_c_ms_add[yr]
                                                for yr in focus_yrs}
                                            # Uncompeted totals
                                            tot_uc = {yr: tot_uc[yr] +
                                                      dat_uc_ms_add[yr]
                                                      for yr in focus_yrs}
                                        except KeyError:
                                            continue
                            # Finalize energy/carbon/cost scaling fraction
                            # for meas/metric/case/region
                            self.output_ecms_cfs[m.name][k][mt][reg] = {
                                yr: (tot_c[yr] / tot_uc[yr]) if
                                tot_uc[yr] != 0 else 1 for yr in focus_yrs}
                            # Check for and if possible handle energy/carbon/
                            # cost competition fractions that are not between
                            # 0 and 1
                            yrs_v_adj = [
                                yr for yr in self.output_ecms_cfs[
                                    m.name][k][mt][reg].keys()
                                if (self.output_ecms_cfs[
                                        m.name][k][mt][reg][yr] > 1 or
                                    self.output_ecms_cfs[
                                        m.name][k][mt][reg][yr] < 0)]
                            for yva in yrs_v_adj:
                                # Screen for small savings
                                if (tot_c[yva] > -1 and tot_c[yva] < 1):
                                    self.output_ecms_cfs[
                                        m.name][k][mt][reg][yva] = 0

                # Finalize stock scaling fraction for measure
                stk_c, stk_uc = [
                    m.markets[adopt_scheme][scn]["master_mseg"]["stock"][
                        "total"]["measure"] for scn in [
                        "competed", "uncompeted"]]
                self.output_ecms_cfs[m.name]["stock"] = {
                    yr: (stk_c[yr] / stk_uc[yr]) if stk_uc[yr] != 0 else 1
                    for yr in focus_yrs}
                # Check for and if possible handle stock competition fractions
                # that are not between 0 and 1
                yrs_v_adj = [
                    yr for yr in self.output_ecms_cfs[m.name]["stock"].keys()
                    if (self.output_ecms_cfs[m.name]["stock"][yr] > 1 or
                        self.output_ecms_cfs[m.name]["stock"][yr] < 0)]
                for yva in yrs_v_adj:
                    # Screen for small numbers/artifacts
                    if (stk_c[yva] > -1 and stk_c[yva] < 1):
                        self.output_ecms_cfs[m.name]["stock"][yva] = 0

            # Normalize the baseline energy/carbon/cost and efficient energy/
            # carbon/cost for the measure that falls into each of the climate,
            # building type, and end use output categories by the total
            # baseline energy/carbon/cost and efficient energy/carbon/cost for
            # the measure (all post-competition); this yields fractions to use
            # in apportioning energy, carbon, and cost results by category

            # Energy
            # Calculate baseline energy fractions by output breakout category
            frac_base_energy = self.out_break_walk(
                m.markets[adopt_scheme]["competed"]["mseg_out_break"][
                    "energy"]["baseline"], energy_base_avg, focus_yrs,
                divide=True)
            # Calculate efficient energy fractions by output breakout category
            frac_eff_energy = self.out_break_walk(
                m.markets[adopt_scheme]["competed"]["mseg_out_break"][
                    "energy"]["efficient"], energy_eff_avg, focus_yrs,
                divide=True)
            # Calculate efficient captured energy fractions by output breakout
            # category if efficient-captured energy data are present
            if eff_capt:
                frac_eff_energy_capt = self.out_break_walk(
                    m.markets[adopt_scheme]["competed"]["mseg_out_break"][
                        "energy"]["efficient-captured"], energy_eff_capt_avg,
                    focus_yrs, divide=True)
            # Cost
            # Calculate baseline energy cost fractions by output breakout
            # category
            frac_base_cost = self.out_break_walk(
                m.markets[adopt_scheme]["competed"]["mseg_out_break"][
                    "cost"]["baseline"], energy_cost_base_avg,
                focus_yrs, divide=True)
            # Calculate efficient energy cost fractions by output breakout
            # category
            frac_eff_cost = self.out_break_walk(
                m.markets[adopt_scheme]["competed"]["mseg_out_break"][
                    "cost"]["efficient"], energy_cost_eff_avg,
                focus_yrs, divide=True)
            # Carbon
            # Calculate baseline carbon fractions by output breakout category
            frac_base_carb = self.out_break_walk(
                m.markets[adopt_scheme]["competed"]["mseg_out_break"][
                    "carbon"]["baseline"], carb_base_avg, focus_yrs,
                divide=True)
            # Calculate efficient carbon fractions by output breakout category
            frac_eff_carb = self.out_break_walk(
                m.markets[adopt_scheme]["competed"]["mseg_out_break"][
                    "carbon"]["efficient"], carb_eff_avg, focus_yrs,
                divide=True)

            # Create shorthand variable for results by breakout category
            mkt_save_brk = self.output_ecms[m.name][
                "Markets and Savings (by Category)"][adopt_scheme]
            # Create combined list of baseline and efficient variables to
            # loop through below in finalizing baseline/efficient breakouts
            mkt_keys = mkt_base_keys + mkt_eff_keys
            # Apply output breakout fractions to total baseline and efficient
            # energy, carbon, and cost results initialized above
            for k in mkt_keys:
                # Apply baseline partitioning fractions to baseline values
                if "Baseline" in k:
                    # Energy results
                    if "Energy Use" in k:
                        mkt_save_brk[k] = self.out_break_walk(
                            copy.deepcopy(frac_base_energy), mkt_save_brk[k],
                            focus_yrs, divide=False)
                    # Energy cost results
                    elif "Energy Cost" in k:
                        mkt_save_brk[k] = self.out_break_walk(
                            copy.deepcopy(frac_base_cost), mkt_save_brk[k],
                            focus_yrs, divide=False)
                    # Carbon results
                    else:
                        mkt_save_brk[k] = self.out_break_walk(
                            copy.deepcopy(frac_base_carb), mkt_save_brk[k],
                            focus_yrs, divide=False)
                # Apply efficient partitioning fractions to efficient values
                elif any([x in k for x in ["Efficient", "Measure"]]):
                    # Energy results excluding efficient captured
                    if "Energy Use" in k and "Measure" not in k:
                        mkt_save_brk[k] = self.out_break_walk(
                            copy.deepcopy(frac_eff_energy), mkt_save_brk[k],
                            focus_yrs, divide=False)
                    # Efficient captured energy results
                    elif eff_capt and "Energy Use" in k and "Measure" in k:
                        mkt_save_brk[k] = self.out_break_walk(
                            copy.deepcopy(frac_eff_energy_capt),
                            mkt_save_brk[k],
                            focus_yrs, divide=False)
                    # Energy cost results
                    elif "Energy Cost" in k:
                        mkt_save_brk[k] = self.out_break_walk(
                            copy.deepcopy(frac_eff_cost), mkt_save_brk[k],
                            focus_yrs, divide=False)
                    # Carbon results
                    else:
                        mkt_save_brk[k] = self.out_break_walk(
                            copy.deepcopy(frac_eff_carb), mkt_save_brk[k],
                            focus_yrs, divide=False)
            # Assess final output breakouts of savings as the difference
            # between finalized baseline and efficient breakouts from above
            for ind_k, k in enumerate(save_keys):
                # Copy baseline breakouts dict to use in establishing the
                # structure of the final savings output breakouts dict
                orig_dict_struct = copy.deepcopy(
                    mkt_save_brk[mkt_base_keys[ind_k]])
                # Loop through all nested levels of the dict above; when
                # reaching terminal nodes, finalize savings values as
                # difference between finalized baseline and efficient results
                mkt_save_brk[k] = self.out_break_walk_subtr(
                    orig_dict_struct, mkt_save_brk[mkt_base_keys[ind_k]],
                    mkt_save_brk[mkt_eff_keys[ind_k]], focus_yrs)

            # Record low and high estimates on markets, if available and
            # user has not specified trimmed output
            if trim_out is not False:
                # Set shorter name for markets and savings output dict
                mkt_sv = self.output_ecms[m.name][
                    "Markets and Savings (Overall)"][adopt_scheme]
                # Record low and high baseline market values
                if energy_base_avg != energy_base_low:
                    # for x in [output_dict_overall, output_dict_bycat]:
                    mkt_sv["Baseline Energy Use (low) (MMBtu)"] = \
                        energy_base_low
                    mkt_sv["Baseline Energy Use (high) (MMBtu)"] = \
                        energy_base_high
                    mkt_sv["Baseline CO2 Emissions (low) (MMTons)".
                           translate(sub)] = carb_base_low
                    mkt_sv["Baseline CO2 Emissions (high) (MMTons)".
                           translate(sub)] = carb_base_high
                    mkt_sv["Baseline Energy Cost (low) (USD)"] = \
                        energy_cost_base_low
                    mkt_sv["Baseline Energy Cost (high) (USD)"] = \
                        energy_cost_base_high
                    mkt_sv["Baseline CO2 Cost (low) (USD)".translate(sub)] = \
                        carb_cost_base_low
                    mkt_sv["Baseline CO2 Cost (high) (USD)".translate(sub)] = \
                        carb_cost_base_high
                # Record low and high efficient market values
                if energy_eff_avg != energy_eff_low:
                    # for x in [output_dict_overall, output_dict_bycat]:
                    mkt_sv["Efficient Energy Use (low) (MMBtu)"] = \
                        energy_eff_low
                    mkt_sv["Efficient Energy Use (high) (MMBtu)"] = \
                        energy_eff_high
                    mkt_sv["Efficient CO2 Emissions (low) (MMTons)".
                           translate(sub)] = carb_eff_low
                    mkt_sv["Efficient CO2 Emissions (high) (MMTons)".
                           translate(sub)] = carb_eff_high
                    mkt_sv["Efficient Energy Cost (low) (USD)"] = \
                        energy_cost_eff_low
                    mkt_sv["Efficient Energy Cost (high) (USD)"] = \
                        energy_cost_eff_high
                    mkt_sv["Efficient CO2 Cost (low) (USD)".translate(sub)] = \
                        carb_cost_eff_low
                    mkt_sv[
                        "Efficient CO2 Cost (high) (USD)".translate(sub)] = \
                        carb_cost_eff_high
                    # Record efficient-captured data if present
                    if eff_capt:
                        mkt_sv[
                            "Efficient Energy Use, Measure (low) (MMBtu)"] = \
                            energy_eff_capt_low
                        mkt_sv[
                            "Efficient Energy Use, Measure (high) (MMBtu)"] = \
                            energy_eff_capt_high

            # Record updated financial metrics in Engine 'output' attribute;
            # yield low and high estimates on the metrics if available
            if trim_out is not False and cce_avg != cce_low:
                self.output_ecms[m.name]["Financial Metrics"] = OrderedDict([
                        ("Cost of Conserved Energy ($/MMBtu saved)",
                            cce_avg),
                        ("Cost of Conserved Energy (low) ($/MMBtu saved)",
                            cce_low),
                        ("Cost of Conserved Energy (high) ($/MMBtu saved)",
                            cce_high),
                        (("Cost of Conserved CO2 "
                          "($/MTon CO2 avoided)").
                         translate(sub), ccc_avg),
                        (("Cost of Conserved CO2 (low) "
                          "($/MTon CO2 avoided)").
                         translate(sub), ccc_low),
                        (("Cost of Conserved CO2 (high) "
                          "($/MTon CO2 avoided)").
                         translate(sub), ccc_high),
                        ("IRR (%)", irr_e_avg),
                        ("IRR (low) (%)", irr_e_low),
                        ("IRR (high) (%)", irr_e_high),
                        ("Payback (years)", payback_e_avg),
                        ("Payback (low) (years)", payback_e_low),
                        ("Payback (high) (years)", payback_e_high)])
            else:
                self.output_ecms[m.name]["Financial Metrics"] = OrderedDict([
                        ("Cost of Conserved Energy ($/MMBtu saved)",
                            cce_avg),
                        (("Cost of Conserved CO2 "
                          "($/MTon CO2 avoided)").
                         translate(sub), ccc_avg),
                        ("IRR (%)", irr_e_avg),
                        ("Payback (years)", payback_e_avg)])

            # If a user desires measure market penetration percentages as an
            # output, calculate and report these fractions
            if self.opts.mkt_fracs is True:
                # Calculate market penetration percentages for the current
                # measure and scenario; divide post-competition measure stock
                # by the total stock that the measure could possibly affect
                mkt_fracs = {yr: round(
                    ((mkts["stock"]["total"]["measure"][yr] / m.markets[
                      adopt_scheme]["uncompeted"]["master_mseg"]["stock"][
                      "total"]["all"][yr]) * 100), 1) if m.markets[
                      adopt_scheme]["uncompeted"]["master_mseg"]["stock"][
                      "total"]["all"][yr] != 0 else 0 for
                    yr in focus_yrs}
                # Calculate average and low/high penetration fractions
                mkt_fracs_avg = {
                    k: numpy.mean(v) for k, v in mkt_fracs.items()}
                mkt_fracs_low = {
                    k: numpy.percentile(v, 5) for k, v in mkt_fracs.items()}
                mkt_fracs_high = {
                    k: numpy.percentile(v, 95) for k, v in mkt_fracs.items()}
                # Set the average market penetration fraction output
                self.output_ecms[m.name]["Markets and Savings (Overall)"][
                    adopt_scheme]["Stock Penetration (%)"] = mkt_fracs_avg
                # Set low/high market penetration fractions (as applicable)
                if mkt_fracs_avg != mkt_fracs_low:
                    self.output_ecms[m.name]["Markets and Savings (Overall)"][
                        adopt_scheme]["Stock Penetration (low) (%)"] = \
                        mkt_fracs_low
                    self.output_ecms[m.name]["Markets and Savings (Overall)"][
                        adopt_scheme]["Stock Penetration (high) (%)"] = \
                        mkt_fracs_high
            # If a user desires stock data as an output, calculate and report
            # these data for the baseline and measure cases
            if self.opts.report_stk is True:
                # Determine correct units to use for stock reporting

                # Envelope tech.; use units of ft^2 floor
                if "demand" in m.technology_type["primary"]:
                    if any([x in m.bldg_type for x in [
                        "single family home", "multi family home",
                            "mobile home"]]):
                        stk_units = "(# homes served)"
                    else:
                        stk_units = "(ft^2 floor served)"
                # Non-envelope residential tech.; use equipment units
                elif any([x in m.bldg_type for x in [
                    "single family home", "multi family home",
                        "mobile home"]]):
                    stk_units = "(units equipment)"
                # Non-envelope commercial tech.; units vary by end use
                else:
                    # If measure affects heating, units are always in terms
                    # of heating service
                    if "heating" in m.end_use["primary"]:
                        stk_units = "(TBtu heating served)"
                    # If measure affects cooling but does not affect heating,
                    # units are always in terms of cooling service
                    elif "cooling" in m.end_use["primary"]:
                        stk_units = "(TBtu cooling served)"
                    elif "lighting" in m.end_use["primary"]:
                        stk_units = "(giga-lm-years served)"
                    elif "ventilation" in m.end_use["primary"]:
                        stk_units = "(giga-CFM-years served)"
                    elif any([x in m.end_use["primary"] for x in [
                            "water heating", "refrigeration", "cooking"]]):
                        # Find end use name
                        eu = [x for x in [
                            "water heating", "refrigeration", "cooking"]
                            if x in m.end_use["primary"]][0]
                        stk_units = "(TBtu " + eu + " served)"
                    # Computers and other equipment in units of ft^2 floor
                    else:
                        stk_units = "(ft^2 floor served)"
                # Set baseline and measure stock keys, including units
                base_stk_uc_key, base_stk_c_key, meas_stk_key = [
                    x + stk_units for x in [
                        "Baseline Stock (Uncompeted)",
                        "Baseline Stock (Competed)",
                        "Measure Stock (Competed)"]]
                # Report baseline stock and stock cost figures (all baseline
                # stock/stock cost that the measure could affect/capture);
                # report both uncompeted/competed versions
                stk_base_uc = {yr: m.markets[adopt_scheme][
                    "uncompeted"]["master_mseg"]["stock"][
                    "total"]["all"][yr] for yr in focus_yrs}
                stk_base_c = {
                    yr: mkts["stock"]["total"]["all"][yr] for yr in focus_yrs}
                stk_cost_base = {yr: mkts["cost"]["stock"]["total"][
                    "baseline"][yr] for yr in focus_yrs}
                self.output_ecms[m.name]["Markets and Savings (Overall)"][
                    adopt_scheme][base_stk_uc_key] = stk_base_uc
                self.output_ecms[m.name]["Markets and Savings (Overall)"][
                    adopt_scheme][base_stk_c_key] = stk_base_c
                # Report measure stock and stock cost figures
                stk_meas, stk_cost_meas = [{yr: mkts["stock"]["total"][
                        "measure"][yr] for yr in focus_yrs},
                    {yr: mkts["cost"]["stock"]["total"][
                        "efficient"][yr] for yr in focus_yrs}]
                # Calculate average and low/high measure stock and stock
                # cost figures
                stk_meas_avg, stk_cost_meas_avg = [{
                    k: numpy.mean(v) for k, v in sv.items()} for sv in [
                        stk_meas, stk_cost_meas]]
                stk_meas_low, stk_cost_meas_low = [{
                    k: numpy.percentile(v, 5) for k, v in sv.items()} for sv
                    in [stk_meas, stk_cost_meas]]
                stk_meas_high, stk_cost_meas_high = [{
                    k: numpy.percentile(v, 95) for k, v in stk_meas.items()}
                    for sv in [stk_meas, stk_cost_meas]]
                # Set the average measure stock output
                self.output_ecms[m.name]["Markets and Savings (Overall)"][
                    adopt_scheme][meas_stk_key] = stk_meas_avg
                # Set the average measure stock cost output
                self.output_ecms[m.name]["Markets and Savings (Overall)"][
                    adopt_scheme][stk_cost_key_tot] = stk_cost_meas_avg
                # Set the average measure incremental stock cost output
                self.output_ecms[m.name]["Markets and Savings (Overall)"][
                    adopt_scheme][stk_cost_key_inc] = {
                        yr: (stk_cost_meas_avg[yr] - stk_cost_base[yr])
                        for yr in focus_yrs}
                # Update stock cost data across all ECMs with data for
                # current ECM
                for yr in focus_yrs:
                    # Add to total stock cost data (first element of list)
                    stk_cost_all_ecms[0][yr] += stk_cost_meas_avg[yr]
                    # Add to inc. stock cost data (second element of list)
                    stk_cost_all_ecms[1][yr] += (
                        stk_cost_meas_avg[yr] - stk_cost_base[yr])
                # Set low/high measure stock outputs (as applicable)
                if stk_meas_avg != stk_meas_low:
                    meas_stk_key_low, stk_cost_key_tot_low, \
                        stk_cost_key_inc_low = [x + " (low)" for x in [
                            meas_stk_key, stk_cost_key_tot, stk_cost_key_inc]]
                    meas_stk_key_high, stk_cost_key_tot_high, \
                        stk_cost_key_inc_high = [x + " (high)" for x in [
                            meas_stk_key, stk_cost_key_tot, stk_cost_key_inc]]
                    self.output_ecms[m.name]["Markets and Savings (Overall)"][
                        adopt_scheme][meas_stk_key_low] = stk_meas_low
                    self.output_ecms[m.name]["Markets and Savings (Overall)"][
                        adopt_scheme][meas_stk_key_high] = stk_meas_high
                    # Set the low measure stock cost output
                    self.output_ecms[m.name]["Markets and Savings (Overall)"][
                        adopt_scheme][stk_cost_key_tot_low] = \
                        stk_meas_low
                    # Set the low measure incremental stock cost output
                    self.output_ecms[m.name]["Markets and Savings (Overall)"][
                        adopt_scheme][stk_cost_key_inc_low] = {
                        yr: (stk_cost_meas_low[yr] - stk_cost_base[yr])
                        for yr in focus_yrs}
                    # Set the high measure stock cost output
                    self.output_ecms[m.name]["Markets and Savings (Overall)"][
                        adopt_scheme][stk_cost_key_tot_high] = \
                        stk_cost_meas_high
                    # Set the high measure incremental stock cost output
                    self.output_ecms[m.name]["Markets and Savings (Overall)"][
                        adopt_scheme][stk_cost_key_inc_high] = {
                        yr: (stk_cost_meas_high[yr] - stk_meas_high[yr])
                        for yr in focus_yrs}

        # Find mean and 5th/95th percentile values of each market/savings
        # total across all ECMs (note: if total is point value, all three of
        # these values will be the same)

        # Mean of outputs across all ECMs
        energy_base_all_avg, carb_base_all_avg, energy_cost_base_all_avg, \
            carb_cost_base_all_avg, energy_eff_all_avg, energy_eff_all_capt_avg, \
            carb_eff_all_avg, energy_cost_eff_all_avg, carb_cost_eff_all_avg, \
            energy_save_all_avg, energy_costsave_all_avg, carb_save_all_avg, \
            carb_costsave_all_avg = [{
                k: numpy.mean(v) if v is not None else v
                for k, v in z.items()} for z in summary_vals_all_ecms]
        # 5th percentile of outputs across all ECMs
        energy_base_all_low, carb_base_all_low, energy_cost_base_all_low, \
            carb_cost_base_all_low, energy_eff_all_low, energy_eff_all_capt_low, \
            carb_eff_all_low, energy_cost_eff_all_low, carb_cost_eff_all_low, \
            energy_save_all_low, energy_costsave_all_low, carb_save_all_low, \
            carb_costsave_all_low = [{
                k: numpy.percentile(v, 5) if v is not None else v
                for k, v in z.items()} for z in summary_vals_all_ecms]
        # 95th percentile of outputs across all ECMs
        energy_base_all_high, carb_base_all_high, energy_cost_base_all_high, \
            carb_cost_base_all_high, energy_eff_all_high, energy_eff_all_capt_high, \
            carb_eff_all_high, energy_cost_eff_all_high, carb_cost_eff_all_high, \
            energy_save_all_high, energy_costsave_all_high, \
            carb_save_all_high, carb_costsave_all_high = [{
                k: numpy.percentile(v, 95) if v is not None else v
                for k, v in z.items()} for z in summary_vals_all_ecms]

        # Record mean markets and savings across all ECMs
        self.output_all["All ECMs"]["Markets and Savings (Overall)"][
            adopt_scheme] = OrderedDict([
                ("Baseline Energy Use (MMBtu)", energy_base_all_avg),
                ("Baseline CO2 Emissions (MMTons)".translate(sub),
                 carb_base_all_avg),
                ("Baseline Energy Cost (USD)", energy_cost_base_all_avg),
                ("Baseline CO2 Cost (USD)".translate(sub),
                 carb_cost_base_all_avg),
                ("Energy Savings (MMBtu)", energy_save_all_avg),
                ("Energy Cost Savings (USD)", energy_costsave_all_avg),
                ("Avoided CO2 Emissions (MMTons)".translate(sub),
                 carb_save_all_avg),
                ("CO2 Cost Savings (USD)".translate(sub),
                 carb_costsave_all_avg),
                ("Efficient Energy Use (MMBtu)", energy_eff_all_avg),
                ("Efficient CO2 Emissions (MMTons)".translate(sub),
                 carb_eff_all_avg),
                ("Efficient Energy Cost (USD)", energy_cost_eff_all_avg),
                ("Efficient CO2 Cost (USD)".translate(sub),
                 carb_cost_eff_all_avg)])
        # Record efficient-captured data across all ECMs if present
        if eff_capt and energy_eff_all_capt_avg is not None:
            self.output_all["All ECMs"]["Markets and Savings (Overall)"][
                adopt_scheme]["Efficient Energy Use, Measure (MMBtu)"] = \
                energy_eff_all_capt_avg

        # Record updated (post-competed) fugitive emissions results across all
        # ECMs if applicable
        if summary_vals_all_ecms_f_e is not None:
            # Record updated baseline/efficient methane results
            self.output_all["All ECMs"]["Markets and Savings (Overall)"][
                adopt_scheme]["Baseline Fugitive Methane (MMTons CO2e)"], \
                self.output_all["All ECMs"][
                    "Markets and Savings (Overall)"][adopt_scheme][
                "Efficient Fugitive Methane (MMTons CO2e)"] = \
                summary_vals_all_ecms_f_e[0:2]
            # Record updated methane savings results
            self.output_all["All ECMs"]["Markets and Savings (Overall)"][
                adopt_scheme]["Fugitive Methane Savings (MMTons CO2e)"] = \
                summary_vals_all_ecms_f_e[4]
            # Record updated baseline/efficient refrigerant results
            self.output_all["All ECMs"]["Markets and Savings (Overall)"][
                adopt_scheme][
                    "Baseline Fugitive Refrigerants (MMTons CO2e)"], \
                self.output_all["All ECMs"][
                    "Markets and Savings (Overall)"][adopt_scheme][
                "Efficient Fugitive Refrigerants (MMTons CO2e)"] = \
                summary_vals_all_ecms_f_e[2:4]
            # Record updated refrigerant savings results
            self.output_all["All ECMs"]["Markets and Savings (Overall)"][
                adopt_scheme][
                    "Fugitive Refrigerants Savings (MMTons CO2e)"] = \
                summary_vals_all_ecms_f_e[5]

        # If necessary, record stock costs across all ECMs
        if self.opts.report_stk is True and stk_cost_all_ecms:
            self.output_all["All ECMs"]["Markets and Savings (Overall)"][
                adopt_scheme][stk_cost_key_tot] = \
                stk_cost_all_ecms[0]
            self.output_all["All ECMs"]["Markets and Savings (Overall)"][
                adopt_scheme][stk_cost_key_inc] = \
                stk_cost_all_ecms[1]

        # Record low/high estimates on efficient markets across all ECMs, if
        # available and user has not specified trimmed output
        if trim_out is not False and energy_eff_all_avg != energy_eff_all_low:

            # Set shorter name for markets and savings output dict across all
            # ECMs
            mkt_sv_all = self.output_all["All ECMs"][
                "Markets and Savings (Overall)"][adopt_scheme]

            mkt_sv_all["Efficient Energy Use (low) (MMBtu)"] = \
                energy_eff_all_low
            mkt_sv_all["Efficient Energy Use (high) (MMBtu)"] = \
                energy_eff_all_high
            mkt_sv_all["Efficient CO2 Emissions (low) (MMTons)".
                       translate(sub)] = carb_eff_all_low
            mkt_sv_all["Efficient CO2 Emissions (high) (MMTons)".
                       translate(sub)] = carb_eff_all_high
            mkt_sv_all["Efficient Energy Cost (low) (USD)"] = \
                energy_cost_eff_all_low
            mkt_sv_all["Efficient Energy Cost (high) (USD)"] = \
                energy_cost_eff_all_high
            mkt_sv_all["Efficient CO2 Cost (low) (USD)".translate(sub)] = \
                carb_cost_eff_all_low
            mkt_sv_all["Efficient CO2 Cost (high) (USD)".translate(sub)] = \
                carb_cost_eff_all_high
            # Record low/high efficient-captured data across all ECMs if
            # present
            if eff_capt and all([x is not None for x in [
                    energy_cost_eff_all_low, energy_cost_eff_all_high]]):
                self.output_all["All ECMs"]["Markets and Savings (Overall)"][
                    adopt_scheme][
                    "Efficient Energy Use, Measure (low) (MMBtu)"], \
                    self.output_all["All ECMs"][
                    "Markets and Savings (Overall)"][adopt_scheme][
                    "Efficient Energy Use, Measure (high) (MMBtu)"] = \
                    [energy_eff_all_capt_low, energy_eff_all_capt_high]

    def out_break_walk(self, adjust_dict, adjust_vals, focus_yrs, divide):
        """Partition measure results by climate, building sector, and end use.

        Args:
            adjust_dict (dict): Results partitioning structure and fractions
                for climate zone, building sector, and end use.
            adjust_vals (dict): Unpartitioned energy, carbon, and cost
                markets/savings.
            focus_yrs (list): Optional years of focus within overall yr. range
            divide (boolean): Optional flag to divide terminal values instead
                of multiplying them (the default option).

        Returns:
            Measure baseline or efficient results partitioned by climate,
            building sector, end use, and possibly fuel type.
        """
        for (k, i) in sorted(adjust_dict.items()):
            if isinstance(i, dict) and len(i.keys()) > 0:
                self.out_break_walk(i, adjust_vals, focus_yrs, divide)
            elif isinstance(i, dict):
                del adjust_dict[k]
            elif k in focus_yrs:
                # Apply appropriate climate zone/building type/end use
                # partitioning fraction to the overall market/savings
                # value
                if divide is False:
                    adjust_dict[k] = adjust_dict[k] * adjust_vals[k]
                else:
                    if adjust_vals[k] != 0:
                        adjust_dict[k] = adjust_dict[k] / adjust_vals[k]
                    else:
                        adjust_dict[k] = 0
            else:
                del adjust_dict[k]
        return adjust_dict

    def out_break_walk_subtr(self, orig_dict, base_val, eff_val, focus_yrs):
        """Subtract efficient from base values in nested dicts to get savings.

        Args:
            orig_dict (dict): The final dict/dict structure to be produced.
            base_val (dict): Baseline values in nested dict.
            eff_val (list): Efficient values in nested dict.
            focus_yrs (list): Optional years of focus within overall yr. range

        Returns:
            Measure savings results partitioned by climate, building sector,
            end use, and possibly fuel type.
        """
        for (k, i) in sorted(orig_dict.items()):
            if isinstance(i, dict) and len(i.keys()) > 0:
                self.out_break_walk_subtr(
                    i, base_val[k], eff_val[k], focus_yrs)
            elif isinstance(i, dict):
                del orig_dict[k]
            elif k in focus_yrs:
                # Subtract efficient from baseline values at terminal nodes
                orig_dict[k] = base_val[k] - eff_val[k]
            else:
                del orig_dict[k]
        return orig_dict


def measure_opts_match(option_dicts: list[dict]) -> bool:
    """Checks if a list of measure options have common argument values, excluding those that
        do not influence final results

    Args:
        option_dicts (list[dict]): List of dictionaries containing user options for measures

    Returns:
        bool: if True, then all options dicts are alike, otherwise False
    """

    ignore_opts = ["verbose", "yaml", "ecm_directory", "ecm_files", "ecm_files_user",
                   "ecm_packages", "ecm_files_regex"]
    keys_to_check = [key for key in option_dicts[0].keys() if key not in ignore_opts]
    if any(opts[x] != option_dicts[0][x] for opts in option_dicts[1:] for x in keys_to_check):
        return False

    return True


def main(opts: argparse.NameSpace):  # noqa: F821
    """Import, finalize, and write out measure savings and financial metrics.

    Note:
        Import measures from a JSON, calculate competed and uncompeted
        savings and financial metrics for each measure, and write a summary
        of key results to an output JSON.
    """

    # Set function that only prints message when in verbose mode
    verboseprint = print if opts.verbose else lambda *a, **k: None

    # Raise numpy errors as exceptions
    numpy.seterr('raise')
    # Initialize user opts variable (elements: S-S calculation method;
    # daily hour range of focus for TSV metrics (all hours, peak, low demand
    # hours); output type for TSV metrics (energy or power); calculation type
    # for TSV metrics (sum, max, avg); season of focus for TSV metrics (summer,
    # winter, intermediate)
    energy_out = ["fossil_equivalent", "NA", "NA", "NA", "NA"]
    # Instantiate useful input files object (fossil fuel equivalency method
    # used by default to calculate site-source conversions, with no TSV metrics
    # and AIA regions and a baseline grid scenario)
    handyfiles = UsefulInputFiles(
        energy_out=energy_out, regions="AIA", grid_decarb=False)
    # Instantiate useful variables object
    handyvars = UsefulVars(handyfiles)

    # If a user desires trimmed down results, collect information about whether
    # they want to restrict to certain years of focus
    if opts.trim_results is True:
        # Flag trimmed results format
        trim_out = True
        trim_yrs = []
        while trim_yrs is not False and ((len(trim_yrs) == 0) or any([
            x < int(handyvars.aeo_years[0]) or x > int(handyvars.aeo_years[-1])
                for x in trim_yrs])):
            # Initialize focus year range input
            trim_yrs_init = input(
                "Enter years of focus for the outputs, with a space in "
                "between each (or hit return to use all years): ")
            # Finalize focus year range input; if not provided, assume False
            if trim_yrs_init:
                trim_yrs = list(map(int, trim_yrs_init.split()))
                if any([x < int(handyvars.aeo_years[0]) or
                        x > int(handyvars.aeo_years[-1]) for x in trim_yrs]):
                    print('Please try again. Enter focus years between '
                          + handyvars.aeo_years[0] + ' and ' +
                          handyvars.aeo_years[-1])
            else:
                trim_yrs = False
    else:
        trim_out, trim_yrs = (False for n in range(2))

    # Import measure files
    with open(handyfiles.meas_summary_data, 'r') as mjs:
        try:
            meas_summary = json.load(mjs)
        except ValueError as e:
            raise ValueError(
                f"Error reading in '{handyfiles.meas_summary_data}': {str(e)}") from None

    # Import list of all unique active measures
    with open(handyfiles.active_measures, 'r') as am:
        try:
            run_setup = json.load(am)
            active_meas_all = numpy.unique(run_setup["active"])
        except ValueError as e:
            raise ValueError(
                f"Error reading in '{handyfiles.active_measures}': {str(e)}") from None
    print('ECM attributes data load complete')

    active_ecms_w_jsons = 0
    # Check that all ECM names included in the active list have a
    # matching ECM definition in ./ecm_definitions; warn users about ECMs
    # that do not have a matching ECM definition, which will be excluded
    for mn in active_meas_all:
        if mn not in [m["name"] for m in meas_summary]:
            warnings.warn(
                "WARNING: ECM '" + mn + "' in 'run_setup.json' active " +
                "list does not match any of the ECM names found in " +
                f"{fp.ECM_DEF} JSONs and will not be simulated")
        else:
            active_ecms_w_jsons += 1

    # Warn the user if skipped ECMs remain in the run_setup file (may need
    # to be edited and re-prepared)
    if len(run_setup["skipped"]) != 0:
        warnings.warn(
            f"WARNING: Run setup file ({handyfiles.active_measures}) "
            "indicates ECM preparation routine skipped over some measures. "
            "Check names of these measures under the 'skipped' key within "
            "this setup file and if needed, edit their measure definitions "
            f"in {fp.ECM_DEF} and re-prepare via ecm_prep.")

    # After verifying that there are active measures to simulate with
    # corresponding JSON definitions, loop through measures data in JSON,
    # initialize objects for all measures that are active and valid
    if active_ecms_w_jsons == 0:
        raise (ValueError("No active measures found; ensure that the " +
                          "'active' list in run_setup.json is not empty " +
                          "and that all active measure names match those " +
                          "found in the 'name' field for corresponding " +
                          "measure definitions in ./ecm_definitions"))
    else:
        measures_objlist = [
            Measure(handyvars, **m) for m in meas_summary if
            m["name"] in active_meas_all and m["remove"] is False]

    # Check to ensure that all active/valid measure definitions used consistent
    # user option settings
    try:
        if not measure_opts_match([m.usr_opts for m in measures_objlist]):
            raise ValueError(
                "Attempting to compete measures with different user option settings. To address"
                f" this issue, ensure that all active ECMs in {fp.GENERATED / 'run_setup.json'}"
                " were prepared using the same command line options, or delete the file"
                " ./supporting_data/ and rerun ecm_prep.py with desired command line options.")
    except AttributeError:
        raise ValueError(
            "One or more active ECMs lacks information needed to determine what energy units or"
            " conversions were used in its definition. To address this issue, delete the file"
            " ./supporting_data/ and rerun ecm_prep.py with desired command line options.")

    # Set a flag for detailed building types
    if measures_objlist[0].usr_opts["detail_brkout"] == '1':
        brkout = "detail"
    elif measures_objlist[0].usr_opts["detail_brkout"] == '2':
        brkout = "detail_reg"
    elif measures_objlist[0].usr_opts["detail_brkout"] == '3':
        brkout = "detail_bldg"
    elif measures_objlist[0].usr_opts["detail_brkout"] == '4':
        brkout = "detail_fuel"
    elif measures_objlist[0].usr_opts["detail_brkout"] == '5':
        brkout = "detail_reg_bldg"
    elif measures_objlist[0].usr_opts["detail_brkout"] == '6':
        brkout = "detail_reg_fuel"
    elif measures_objlist[0].usr_opts["detail_brkout"] == '7':
        brkout = "detail_bldg_fuel"
    else:
        brkout = "basic"

    # Set a flag for the type of user option desired (site, source-fossil
    # fuel equivalent, source-captured energy)
    if measures_objlist[0].usr_opts["site_energy"] is True:
        # Set user option to site energy
        energy_out[0] = "site"
    elif measures_objlist[0].usr_opts["captured_energy"] is True:
        # Set user option to source energy using captured energy S-S
        energy_out[0] = "captured"
    else:
        # Otherwise, set user option to source energy, fossil equivalent S-S
        energy_out[0] = "fossil_equivalent"
    # Set a flag for TSV metrics
    if measures_objlist[0].usr_opts["tsv_metrics"] is not False:
        # TSV metrics - Hour range
        if measures_objlist[0].usr_opts["tsv_metrics"][1] == "1":
            energy_out[1] = "All"
        elif measures_objlist[0].usr_opts["tsv_metrics"][1] == "2":
            energy_out[1] = "Pk."
        else:
            energy_out[1] = "Low"
        # TSV metrics  - Output type
        if measures_objlist[0].usr_opts["tsv_metrics"][0] == "1":
            energy_out[2] = "Prd."
        else:
            energy_out[2] = "Hr."
        # TSV metrics - Calc type
        if measures_objlist[0].usr_opts["tsv_metrics"][3] == "1" and \
                measures_objlist[0].usr_opts["tsv_metrics"][0] == "1":
            energy_out[3] = "Sum."
        elif measures_objlist[0].usr_opts["tsv_metrics"][3] == "1":
            energy_out[3] = "Max."
        else:
            energy_out[3] = "Avg."
        # TSV metrics - Season (S - Summer, W - Winter, I - Intermediate)
        if measures_objlist[0].usr_opts["tsv_metrics"][2] == "1":
            energy_out[4] = "(S)"
        elif measures_objlist[0].usr_opts["tsv_metrics"][2] == "2":
            energy_out[4] = "(W)"
        else:
            energy_out[4] = "(I)"
    else:
        energy_out[1:] = ("NA" for n in range(len(energy_out) - 1))

    # Set a flag for geographical breakout (currently possible to breakout
    # by AIA climate zone, NEMS EMM region, or state).
    if measures_objlist[0].usr_opts["alt_regions"] == "EMM":
        regions = "EMM"
    elif measures_objlist[0].usr_opts["alt_regions"] == "State":
        regions = "State"
    else:  # Otherwise, set regional breakdown to AIA climate zones
        regions = "AIA"

    # Set a flag for the assumption of a high grid decarbonization case on
    # the supply-side, which is relevant to site-source conversion factors
    # and the selection of heating/cooling totals for use in adjusting
    # envelope/HVAC overlaps
    if measures_objlist[0].usr_opts["grid_decarb"] is not False:
        grid_decarb = True
    else:
        grid_decarb = False

    # Re-instantiate useful input files object when site energy is output
    # instead of the default source energy or regional breakdown other than
    # default AIA climate zone breakdown is chosen or a high grid decarb.
    # scheme is assumed
    if energy_out[0] != "fossil_equivalent" or regions != "AIA" or \
            grid_decarb is True:
        handyfiles = UsefulInputFiles(energy_out, regions, grid_decarb)
    # Re-instantiate useful variables object when regional breakdown other
    # than the default AIA climate zone breakdown is chosen
    if regions != "AIA":
        handyvars = UsefulVars(handyfiles)

    # Load and set competition data for active measure objects; suppress
    # new line if not in verbose mode ('Data load complete' is appended to
    # this message on the same line of the console upon data load completion)
    if opts.verbose:
        print('Importing ECM competition data...')
    else:
        print('Importing ECM competition data...', end="", flush=True)

    for m in measures_objlist:
        # Assemble file name for measure competition data
        meas_file_name = m.name + ".pkl.gz"
        # Assemble folder path for measure competition data
        comp_folder_name = handyfiles.meas_compete_data
        with gzip.open(comp_folder_name / meas_file_name, 'r') as zp:
            try:
                meas_comp_data = pickle.load(zp)
            except Exception as e:
                raise Exception(
                    f"Error reading in competition data of ECM '{m.name}': {str(e)}") from None
        # Assemble folder path for measure efficient fuel split data
        fs_splt_folder_name = handyfiles.meas_eff_fs_splt_data
        try:
            with gzip.open(fs_splt_folder_name / meas_file_name, 'r') as zp:
                meas_eff_fs_data = pickle.load(zp)
        except FileNotFoundError:
            meas_eff_fs_data = None
        for adopt_scheme in handyvars.adopt_schemes:
            # Reset measure microsegment data attribute to imported values;
            # initialize an uncompeted and post-competition copy of these data
            # (the former of which will be used to establish a common set of
            # stock turnover constraints in the competition, the latter of
            # which will be adjusted by the competition)
            m.markets[adopt_scheme]["uncompeted"]["mseg_adjust"] = \
                meas_comp_data[adopt_scheme]
            m.markets[adopt_scheme]["competed"]["mseg_adjust"] = \
                copy.deepcopy(
                    m.markets[adopt_scheme]["uncompeted"]["mseg_adjust"])
            # Reset measure fuel split attribute to imported values
            m.eff_fs_splt = meas_eff_fs_data
        # Print data import message for each ECM if in verbose mode
        verboseprint("Imported ECM '" + m.name + "' competition data")

    # Import total absolute heating and cooling energy use data, used in
    # removing overlaps between supply-side and demand-side heating/cooling
    # ECMs in the analysis
    with open(handyfiles.htcl_totals, 'r') as msi:
        try:
            htcl_totals = json.load(msi)
        except ValueError as e:
            raise ValueError(
                f"Error reading in '{handyfiles.htcl_totals}': {str(e)}") from None

    # Print message to console; if in verbose mode, print to new line,
    # otherwise append to existing message on the console
    if opts.verbose:
        print('ECM competition data load complete')
    else:
        print('Data load complete')

    # Instantiate an Engine object using active measures list
    a_run = Engine(handyvars, opts, measures_objlist, energy_out, brkout)

    # Calculate uncompeted and competed measure savings and financial
    # metrics, and write key outputs to JSON file
    for adopt_scheme in handyvars.adopt_schemes:
        # Calculate each measure's uncompeted savings and metrics,
        # and print progress update to user
        print("Calculating uncompeted '" + adopt_scheme +
              "' savings/metrics...", end="", flush=True)
        a_run.calc_savings_metrics(adopt_scheme, "uncompeted")
        print("Calculations complete")
        # Update each measure's competed markets to reflect the
        # removal of savings overlaps with competing measures,
        # and print progress update to user
        print("Competing ECMs for '" + adopt_scheme + "' scenario...",
              end="", flush=True)
        a_run.compete_measures(adopt_scheme, htcl_totals)
        print("Competition complete")
        # Calculate each measure's competed measure savings and metrics
        # using updated competed markets, and print progress update to user
        print("Calculating competed '" + adopt_scheme +
              "' savings/metrics...", end="", flush=True)
        a_run.calc_savings_metrics(adopt_scheme, "competed")
        print("Calculations complete")
        print("Finalizing results...", end="", flush=True)
        # Write selected outputs to a summary JSON file for post-processing
        a_run.finalize_outputs(adopt_scheme, trim_out, trim_yrs)
        print("Results finalized")

    # Notify user that all analysis engine calculations are completed
    print("All calculations complete; writing output data...", end="",
          flush=True)

    # Import baseline microsegments
    if regions in ['EMM', 'State']:  # Extract compressed EMM/state data
        bjszip = handyfiles.msegs_in
        with gzip.GzipFile(bjszip, 'r') as zip_ref:
            msegs = json.loads(zip_ref.read().decode('utf-8'))
    else:
        with open(handyfiles.msegs_in, 'r') as msi:
            try:
                msegs = json.load(msi)
            except ValueError as e:
                raise ValueError(
                    f"Error reading in '{handyfiles.msegs_in}': {str(e)}") from None

    # Import site-source conversions
    with open(handyfiles.ss_data, 'r') as ss:
        try:
            cost_ss_carb = json.load(ss)
            ss_conv = cost_ss_carb['electricity'][
                'site to source conversion']['data']
        except ValueError as e:
            raise ValueError(
                f"Error reading in '{handyfiles.ss_data}': {str(e)}") from None

    # Import electricity price and CO2 emissions intensity
    with open(handyfiles.elec_price_co2, 'r') as ece:
        try:
            elec_cost_carb = json.load(ece)
        except ValueError as e:
            raise ValueError(
                f"Error reading in '{handyfiles.elec_price_co2}': + {str(e)}") from None
    # Extract separate price and CO2 emissions intensity variables
    try:
        elec_carb = elec_cost_carb['CO2 intensity of electricity']['data']
        elec_cost = elec_cost_carb['End-use electricity price']['data']
        fmt = True  # Boolean for indicating data key substructure
    except KeyError:
        # Data are structured as in the site_source_co2_conversions files
        elec_carb = elec_cost_carb['electricity']['CO2 intensity']['data']
        elec_cost = elec_cost_carb['electricity']['price']['data']
        fmt = False

    # Determine regions and building types used by active measures for
    # aggregating onsite generation data
    czgrp = set([cz for m in meas_summary
                 if m["name"] in active_meas_all and m["remove"] is False
                 for cz in m['climate_zone']])
    czgrp = sorted(czgrp)
    btgrp = [bt for m in meas_summary
             if m["name"] in active_meas_all and m["remove"] is False
             for bt in m['bldg_type']]
    # Drop multi family and mobile homes, along with commercial unspecified
    # building type; no onsite generation data provided for these bldg. types
    btgrp = set([
        bt for bt in btgrp if bt not in [
            'mobile home', 'multi family home', 'unspecified']])
    btgrp = sorted(btgrp)
    # Set up recursively extensible empty dict to populate with onsite
    # generation data
    def variable_depth_dict(): return defaultdict(variable_depth_dict)
    osg_temp = variable_depth_dict()

    # Aggregate onsite generation data
    osg = {k: 0 for k in handyvars.aeo_years}
    osgcarb = {k: 0 for k in handyvars.aeo_years}
    osgcost = {k: 0 for k in handyvars.aeo_years}
    for cz in czgrp:
        for bt in btgrp:
            z = msegs[cz][bt]['electricity']['onsite generation']['energy']
            # Get onsite generation and adjust by appropriate factor
            # unless site user opts are expected
            if not measures_objlist[0].usr_opts["site_energy"]:
                z = {k: z.get(k, 0)*ss_conv.get(k, 0)
                     for k in handyvars.aeo_years}
            # Get building sector from building type
            if bt in ["single family home", "multi family home",
                      "mobile home"]:
                bt_bin = 'residential'
            else:
                bt_bin = 'commercial'
            # Get CO2 intensity and electricity cost data and convert units
            if fmt:  # Data (and data structure) from emm_region files
                # Convert Mt/TWh to Mt/MMBtu
                carbtmp = {k: elec_carb[cz].get(k, 0)/3.41214e6
                           for k in elec_carb[cz].keys()}
                # Convert $/kWh to $/MMBtu
                costtmp = {k: elec_cost[bt_bin][cz].get(k, 0)/3.41214e-3
                           for k in elec_cost[bt_bin][cz].keys()}
            else:
                if not measures_objlist[0].usr_opts["site_energy"]:
                    # Convert Mt/quads to Mt/MMBtu
                    carbtmp = {k: elec_carb[bt_bin].get(k, 0)/1e9
                               for k in elec_carb[bt_bin].keys()}
                    costtmp = elec_cost[bt_bin]
                else:
                    # Convert Mt/quads to Mt/MMBtu and account for need to
                    # add site-source conversion factor to the carbon and
                    # cost multiplications
                    carbtmp = {k: (elec_carb[bt_bin].get(k, 0)/1e9) *
                               ss_conv.get(k, 0) for k in
                               elec_carb[bt_bin].keys()}
                    costtmp = {k: elec_cost[bt_bin].get(k, 0) *
                               ss_conv.get(k, 0) for k in
                               elec_cost[bt_bin].keys()}
            # Report out onsite generation and corresponding emissions
            # and energy cost savings at the AEO building type level
            osg_temp['Energy (MMBtu)']['By Category'][cz][bt] = z
            osg_temp['CO\u2082 Emissions (MMTons)']['By Category'][cz][bt] = {
                k: z.get(k, 0)*carbtmp.get(k, 0) for k in handyvars.aeo_years}
            osg_temp['Energy Cost (USD)']['By Category'][cz][bt] = {
                k: z.get(k, 0)*costtmp.get(k, 0) for k in handyvars.aeo_years}
            # Calculate total annual onsite generation and corresponding
            # emissions and energy cost savings
            osg = {k: osg.get(k, 0) + z.get(k, 0)
                   for k in handyvars.aeo_years}
            osgcarb = {k: osgcarb.get(k, 0) + z.get(k, 0)*carbtmp.get(k, 0)
                       for k in handyvars.aeo_years}
            osgcost = {k: osgcost.get(k, 0) + z.get(k, 0)*costtmp.get(k, 0)
                       for k in handyvars.aeo_years}
    # Report out onsite generation and corresponding emissions and
    # energy cost savings in aggregate
    osg_temp['Energy (MMBtu)']['Overall'] = osg
    osg_temp['CO\u2082 Emissions (MMTons)']['Overall'] = osgcarb
    osg_temp['Energy Cost (USD)']['Overall'] = osgcost

    # Add onsite generation data as additional measure-level data
    # written with ECM results output
    a_run.output_ecms['On-site Generation'] = osg_temp

    # Recursively navigate dictionary and round values
    def round_values(data, precision):
        if isinstance(data, dict):
            for k, v in data.items():
                data[k] = round_values(v, precision)
        elif isinstance(data, float):
            data = round(data*1.5, precision)
        return data

    a_run.output_ecms = round_values(a_run.output_ecms, 6)
    a_run.output_all = round_values(a_run.output_all, 6)

    # Write summary outputs for individual measures to a JSON
    with open(handyfiles.meas_engine_out_ecms, "w") as jso:
        json.dump(a_run.output_ecms, jso, indent=2)
    # Write summary outputs across all measures to a JSON
    with open(handyfiles.meas_engine_out_agg, "w") as jso:
        json.dump(a_run.output_all, jso, indent=2)
    print("Data writing complete")
    # Write competition adjustment fractions to a JSON, if applicable
    if a_run.output_ecms_cfs is not None:
        with open(handyfiles.comp_fracs_out, "w") as jso:
            json.dump(a_run.output_ecms_cfs, jso, indent=2)

    # Do not plot for the case where a user has trimmed down the results
    # (not all data required for the plots will be available)
    if opts.trim_results is False:
        # Notify user that the output data are being plotted
        print("Plotting output data...", end="", flush=True)
        # Execute plots
        run_plot(meas_summary, a_run, handyvars, measures_objlist, regions)
        print("Plotting complete")


def parse_args(args: list = None) -> argparse.NameSpace:  # noqa: F821
    """Parse arguments for run.py using Config class

    Args:
        args (list, optional): run.py input arguments, if not provided, command line arguments
            will be used in Config. Defaults to None.

    Returns:
        argparse.NameSpace: Arguments object to be used in main()
    """

    # Retrieve config file and CLI arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    config = Config(parser, "run", args)
    opts = config.parse_args()

    return opts


if __name__ == '__main__':
    import time
    start_time = time.time()
    opts = parse_args()
    main(opts)

    hours, rem = divmod(time.time() - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("--- Runtime: %s (HH:MM:SS.mm) ---" %
          "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
