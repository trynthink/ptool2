from __future__ import annotations
import yaml
import copy
import sys
import warnings
import logging
from pathlib import Path
from jsonschema import validate


class LogConfig:
    """Configure the logger
    """

    @staticmethod
    def configure_logging():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


LogConfig.configure_logging()


class FilePaths:
    # Package data:
    _root_dir = Path(__file__).resolve().parents[0]  # package install dir
    SUPPORTING_DATA = _root_dir / "supporting_data"
    THERMAL_LOADS = SUPPORTING_DATA / "thermal_loads_data"
    CONVERT_DATA = SUPPORTING_DATA / "convert_data"
    STOCK_ENERGY = SUPPORTING_DATA / "stock_energy_tech_data"
    TSV_DATA = SUPPORTING_DATA / "tsv_data"
    SUB_FED = SUPPORTING_DATA / "sub_fed"

    # Non-package data:
    _parent_dir = Path.cwd()  # parent dir of repo
    ECM_DEF = _parent_dir / "ecm_definitions"
    GENERATED = _parent_dir / "generated"
    ECM_COMP = GENERATED / "ecm_competition_data"
    EFF_FS_SPLIT = GENERATED / "eff_fs_splt_data"
    INPUTS = _parent_dir / "inputs"
    RESULTS = _parent_dir / "results"
    PLOTS = RESULTS / "plots"
    METADATA_PATH = INPUTS / "metadata.json"

    # Store baseline FilePaths attributes
    _base_paths = {
        attr: value
        for attr, value in locals().items()
        if not callable(value) and not attr.startswith("__")
    }

    @classmethod
    def set_paths(cls, paths_to_update: dict):
        """Update class attributes to new values. If the filepath variable being updated has
            dependent filepaths downstream, then those will also be updated, assuming the same
            relative relationship (e.g., PLOTS remains one level inside of RESULTS)

        Args:
            paths_to_update (dict): Update dictionary, where attributes are keys and
                file paths are values.
        """

        downstream_map = {"GENERATED": ["ECM_COMP", "EFF_FS_SPLIT"],
                          "INPUTS": ["METADATA_PATH"],
                          "RESULTS": ["PLOTS"]}

        for var, new_path in paths_to_update.items():
            if var[0] == "_":
                warnings.warn(f"Changing the value of `FilePaths.{var}` is not allowed. The"
                              " variable will remain unchanged.")
                continue

            original_path = getattr(cls, var)

            # Update downstream paths
            for downstream_path in downstream_map.get(var, []):
                diff = getattr(cls, downstream_path).relative_to(original_path)
                setattr(cls, downstream_path, new_path / diff)
                getattr(cls, downstream_path).mkdir(parents=True, exist_ok=True)

            # Set new path
            setattr(cls, var, new_path)
            getattr(cls, var).mkdir(parents=True, exist_ok=True)

    @classmethod
    def reset_base_paths(cls):
        """Reset filepath attributes to the baseline values defined at class definition"""
        for attr, val in cls._base_paths.items():
            setattr(cls, attr, val)


class Config:
    """Methods to handle yml configuration files, including writing argparse arguments based on
        the schema (supporting_data/config_schema.yml) and enabling validation of inputted
        configuration files.

    Attributes:
        parser (argparse.ArgumentParser): Argument parser
        key (str): Workflow step on which to validate and set arguments {ecm_prep, run}
        cli_args (list): Arguments passed directly to CLI
        schema_file (Pathlib.filepath): Location of the configuration schema yml
        schema_data (dict): Parsed yml schema data
        config_args (dict): Parsed arguments from the yml input file
        args (argparse.Namespace): Arguments object to be used in ecm_prep.py or run.py,
            which includes validated config args and cli inputs
    """

    def __init__(self, parser: argparse.ArgumentParser,  # noqa: F821
                 key: str, cli_args: list = None):
        self.parser = parser
        self.key = key
        if cli_args is None:
            self.cli_args = sys.argv[1:]
        else:
            self.cli_args = cli_args
        self.schema_file = FilePaths.SUPPORTING_DATA / "config_schema.yml"
        self.schema_data = Config.load_config(self.schema_file)

        # Generate argparse, set initial cli args
        schema_block = self.schema_data.get("properties", {}).get(self.key, {})
        self.initialize_argparse(parser)
        self.create_argparse(parser, schema_block)
        self.set_config_args(self.cli_args)

    @staticmethod
    def load_config(filepath):
        with open(filepath, "r") as file:
            return yaml.safe_load(file)

    def _validate(self, input_data: dict, schema_data: dict = None):
        """Validate argument data against schema data

        Args:
            input_data (dict): Data to validate.
            schema_data (dict, optional): Data to validate against, "None" value will use the yml
                schema data in supporting_data/config_schema.
        """

        if not schema_data:
            schema_data = self.schema_data
        validate(input_data, schema_data)

    def _resolve_filepath_args(self):
        """Resolves filepaths depending on whether they are set in the yml or via the CLI.
        """

        fp_arg_dict = {"ecm_directory": "ECM_DEF", "results_directory": "RESULTS"}
        for dir_arg, fp_var in fp_arg_dict.items():
            dir_val = getattr(self.args, dir_arg, None)
            if not dir_val:
                continue
            if Path(dir_val).is_absolute():
                FilePaths.set_paths({fp_var: Path(dir_val)})
                continue

            # Set relative to cwd, if applicable
            if "--" + dir_arg in self.cli_args:
                cwd = Path.cwd()
                resolved_path = (cwd / dir_val).resolve()
            # Set relative to yml location, if applicable
            elif dir_arg in self.config_args.get(self.key, {}).keys():
                yml_dir = self.args.yaml.resolve().parents[0]
                resolved_path = (yml_dir / dir_val).resolve()

            setattr(self.args, dir_arg, resolved_path)

            # Update FilePaths with new directory and affected downstream paths
            FilePaths.set_paths({fp_var: resolved_path})

    def initialize_argparse(self, parser: argparse.ArgumentParser):  # noqa: F821
        """Initialize argument parser with yaml argument, as this is not included in the schema

        Args:
            parser (argparse.ArgumentParser): Parser to which arguments are added
        """
        parser.add_argument(
            "-y",
            "--yaml",
            type=Path,
            help=("Path to YAML configuration file, arguments passed directly to the command "
                  "line will take priority over arguments in this file")
            )

    def set_config_args(self, cli_args: list[str] = []):
        """If there is a config file provided, store data and validate

        Args:
            cli_args (list[str], optional): Command line arguments. Defaults to [].
        """

        self.config_args = {}
        self.args = self.parser.parse_args(cli_args)
        if self.args.yaml:
            self.config_args = Config.load_config(self.args.yaml)
            self._validate(self.config_args, self.schema_data)

    def parse_args(self) -> argparse.Namespace:  # noqa: F821
        """Parse and store arguments from config file and the command line. CLI arguments take
            precedence over config file arguments. Once parsed, arguments are checked for invalid
            combinations not covered by the schema.

        Returns:
            argparse.Namespace: Arguments object to be used in ecm_prep.py or run.py
        """

        # Update args with yaml arguments
        if self.config_args:
            config_key_args = self.config_args.get(self.key, {})
            self.args = self.update_args(self.args, config_key_args)

        # Ensure command-line args take precedence
        self.args = self.parser.parse_args(self.cli_args, namespace=self.args)

        # Resolve relative filepaths (--ecm_directory and --results_directory)
        self._resolve_filepath_args()

        # Check for valid arguments
        self.check_dependencies(self.args)

        return self.args

    def update_args(self,
                    existing_args: argparse.NameSpace,  # noqa: F821
                    new_args: dict) -> argparse.NameSpace:  # noqa: F821
        """Update argparse arguments NameSpace with args dictionary

        Args:
            existing_args (argparse.NameSpace): Arguments object to be updated
            new_args (dict): Arguments and argument values used to update

        Returns:
            argparse.NameSpace: Updated version of existing_args
        """

        for k, v in new_args.items():
            if isinstance(v, dict):
                self.update_args(existing_args, v)
            else:
                existing_args.__dict__.update({k: v})

        return existing_args

    def check_dependencies(self, args: argparse.NameSpace):  # noqa: F821
        """Custom checks to test for invalid combinations of arguments. These checks apply to
            arguments populated by the configuration file and the CLI.

        Args:
            args (argparse.NameSpace): Arguments object to be passed to ecm_prep.py or run.py

        Raises:
            ValueError: If detailed breakout option is not compatible with split_fuel arg
        """

        if self.key == "ecm_prep":
            # tsv metrics
            if bool(args.tsv_type) ^ bool(args.tsv_daily_hr_restrict):
                raise ValueError(
                    "Both `tsv_type` and `tsv_daily_hr_restrict` must be provided if running tsv"
                    " metrics")
            if args.tsv_daily_hr_restrict in ["peak", "low"] and not args.tsv_sys_shape_case:
                raise ValueError(
                    "The `tsv_sys_shape_case` argument must be provided if `tsv_daily_hr_restrict`"
                    " is one of 'peak' or 'low'.")
            if args.tsv_type == "energy" and not (args.tsv_energy_agg and args.tsv_average_days):
                raise ValueError(
                    "`tsv_energy_agg` and `tsv_average_days` must be specified if `tsv_type` is"
                    " 'energy'.")
            if args.tsv_type == "power" and not args.tsv_power_agg:
                raise ValueError(
                    "`tsv_power_agg` must be specified if `tsv_type` is 'power'.")
            if args.tsv_power_agg == "average" and not args.tsv_average_days:
                raise ValueError(
                    "`tsv_average_days` must be specified if `tsv_power_agg` is 'average'.")

            if bool(args.grid_decarb_level) ^ bool(args.grid_assessment_timing):
                raise ValueError(
                    "Both `grid_decarb_level` and `grid_assessment_timing` must be provided if"
                    " assessing grid decarbonization")

            # detailed breakout
            if "fuel types" in args.detail_brkout and args.split_fuel:
                raise ValueError(
                    "Detailed breakout (`detail_brkout`) cannot include `fuel types` if"
                    " split_fuel==True")

            # retrofits
            if args.retrofit_type == "increasing" and (not args.retrofit_multiplier or
                                                       not args.retrofit_mult_year):
                raise ValueError(
                    "`retrofit_multiplier` and `retrofit_mult_year` must be specified if"
                    " `retrofit_type` is 'increasing'.")

            # fugitive emissions
            if ("typical refrigerant" in args.fugitive_emissions and
                    "low-gwp refrigerant" in args.fugitive_emissions):
                raise ValueError(
                    "The `fugitive_emissions` argument can only accept one of 'typical"
                    " refrigerant' and 'low-gwp refrigerant'")

    def create_argparse(self, parser: argparse.ArgumentParser,  # noqa: F821
                        schema_data: dict, group: str = None):
        """Extracts arguments from the config schema and writes argparse arguments. This method
            populates information for the --help flag of ecm_prep.py and run.py and enables
            passing arguments directly via the command line.

        Args:
            parser (argparse.ArgumentParser): Parser to which arguments are added
            schema_data (dict): Schema data; object is expected to have a `type` key at a minimum.
                If the type is an `object`, the method will be called recursively using
                `properties` value as schema_data
            group (str, optional): The group to which arguments belong. Defaults to None.

        Raises:
            ValueError: If more than one data type is provided for an argument, excluding nulls
            ValueError: If data type for an argument does not include one of {string, integer,
                number, boolean, array, object}
        """

        # Subset arguments that belong to groups in the schema
        if group:
            parser = group
        arguments = copy.deepcopy(schema_data).get("properties", {})

        # Iterate through schema key-values and populate arg parser
        for arg_name, data in arguments.items():
            arg_type = data["type"]
            if type(arg_type) == list:
                if "null" in arg_type:
                    arg_type.remove("null")
                if len(arg_type) > 1:
                    raise ValueError("Multiple argument data types is not currently supported")
                arg_type = arg_type[0]
            arg_choices = data.get("enum")
            arg_help = data.get("description")
            arg_default = data.get("default")
            arg_arr_choices = data.get("items", {}).get("enum")

            # If applicable, write string argument choices to description
            if arg_type == "string" and arg_choices:
                while None in arg_choices:
                    arg_choices.remove(None)
                arg_help += f" Allowed values are {{{', '.join(arg_choices)}}}"
                parser.add_argument(
                    f"--{arg_name}",
                    choices=arg_choices,
                    help=arg_help,
                    default=arg_default,
                    metavar="",
                )
            elif arg_type == "string":
                parser.add_argument(f"--{arg_name}", type=str, help=arg_help, default=arg_default)
            elif arg_type == "integer":
                parser.add_argument(f"--{arg_name}", type=int, help=arg_help, default=arg_default)
            elif arg_type == "number":
                parser.add_argument(f"--{arg_name}", type=float, help=arg_help, default=arg_default)
            elif arg_type == "boolean":
                parser.add_argument(f"--{arg_name}", action="store_true", help=arg_help)
            # If applicable, write allowable array choices to description
            elif arg_type == "array":
                if arg_arr_choices:
                    arg_help += f" Allowed values are 0 or more of {{{', '.join(arg_arr_choices)}}}"
                parser.add_argument(
                    f"--{arg_name}",
                    nargs="*",
                    choices=arg_arr_choices,
                    help=arg_help,
                    default=arg_default,
                    metavar="",
                )
            # If belonging to a group, recursively call function and populate arguments
            elif arg_type == "object" and "properties" in data:
                new_group = parser.add_argument_group(arg_name)
                self.create_argparse(parser, data, group=new_group)
            else:
                raise ValueError(f"Unsupported argument type: {arg_type}")
