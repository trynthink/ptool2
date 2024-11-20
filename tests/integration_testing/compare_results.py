import pandas as pd
import argparse
import json
import re
from pathlib import Path


class ScoutCompare():
    """Class to compare results from  Scout workflow run. Comparisons are saved as csv files to
        summarize differences in results json files (agg_results.json, ecm_results.json) and/or
        summary report files (Summary_Data-TP.xlsx, Summary_Data-MAP.xlsx)
    """

    @staticmethod
    def load_json(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    @staticmethod
    def load_summary_report(file_path):
        reports = pd.read_excel(file_path, sheet_name=None, index_col=list(range(5)))
        return reports

    def compare_dict_keys(self, dict1, dict2, paths, path='', key_diffs=None):
        """Compares nested keys across two dictionaries by recursively searching each level

        Args:
            dict1 (dict): baseline dictionary to compare
            dict2 (dict): new dictionary to compare
            paths (list): paths to the original files from which the dictionaries are imported
            path (str, optional): current dictionary path at whcih to compare. Defaults to ''.
            key_diffs (pd.DataFrame, optional): existing summary of difference. Defaults to None.

        Returns:
            pd.DataFrame: summary of differences specifying the file, the unique keys, and the
                path that key is found at.
        """
        if key_diffs is None:
            key_diffs = pd.DataFrame(columns=["Results file", "Unique key", "Found at"])
        keys1 = set(dict1.keys())
        keys2 = set(dict2.keys())
        only_in_dict1 = keys1 - keys2
        only_in_dict2 = keys2 - keys1

        if only_in_dict1:
            new_row = pd.DataFrame({"Results file": f"{paths[0].parent.name}/{paths[0].name}",
                                    "Unique key": str(only_in_dict1),
                                    "Found at": path[2:]}, index=[0])
            key_diffs = pd.concat([key_diffs, new_row], ignore_index=True)
        if only_in_dict2:
            new_row = pd.DataFrame({"Results file": f"{paths[1].parent.name}/{paths[1].name}",
                                    "Unique key": str(only_in_dict2),
                                    "Found at": path[2:]}, index=[0])
            key_diffs = pd.concat([key_diffs, new_row], ignore_index=True)

        for key in keys1.intersection(keys2):
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                key_diffs = self.compare_dict_keys(dict1[key],
                                                   dict2[key],
                                                   paths,
                                                   path=f"{path}: {key}",
                                                   key_diffs=key_diffs)

        return key_diffs

    def compare_dict_values(self, dict1, dict2, percent_threshold=10, abs_threshold=1000):
        """Compares values across two dictionary by recursively searching keys until identifying
            values at common paths. Both thresholds must be met to report results.

        Args:
            dict1 (dict): baseline dictionary to compare
            dict2 (dict): new dictionary to compare
            percent_threshold (int, optional): the percent difference threshold at which
                                               differences are reported. Defaults to 10.
            abs_threshold (int, optional): the abosolute difference threshold at which differences
                                           are reported. Defaults to 10.

        Returns:
            pd.DataFrame: summary of percent differences that meet thresholds
        """
        diff_report = {}

        def compare_recursive(d1, d2, path=""):
            for key in d1.keys():
                current_path = f"{path}['{key}']"
                if isinstance(d1[key], dict) and key in d2:
                    compare_recursive(d1[key], d2[key], current_path)
                elif isinstance(d1[key], (int, float)) and key in d2:
                    if isinstance(d2[key], (int, float)):
                        val1 = d1[key]
                        val2 = d2[key]
                        if val1 != 0:
                            percent_change = ((val2 - val1) / val1) * 100
                            if (abs(percent_change) >= percent_threshold) and \
                                    (abs(val1) >= abs_threshold or abs(val2) >= abs_threshold):
                                diff_report[current_path] = percent_change

        compare_recursive(dict1, dict2)
        return diff_report

    def split_json_key_path(self, path):
        keys = re.findall(r"\['(.*?)'\]", path)
        if len(keys) == 5:
            keys[4:4] = [None, None, None]
        return keys

    def write_dict_key_report(self, diff_report, output_path):
        if diff_report.empty:
            return
        diff_report.to_csv(output_path, index=False)
        print(f"Wrote dictionary key report to {output_path}")

    def write_dict_value_report(self, diff_report, output_path):
        col_headers = [
            "ECM",
            "Markets and Savings Type",
            "Adoption Scenario",
            "Results Scenario",
            "Climate Zone",
            "Building Class",
            "End Use",
            "Year"
        ]
        df = pd.DataFrame(columns=["Results path"], data=list(diff_report.keys()))
        if df.empty:
            return
        df[col_headers] = df["Results path"].apply(self.split_json_key_path).apply(pd.Series)
        df["Percent difference"] = [round(diff, 2) for diff in diff_report.values()]
        df = df.dropna(axis=1, how="all")
        df.to_csv(output_path, index=False)
        print(f"Wrote dictionary value report to {output_path}")

    def compare_jsons(self, json1_path, json2_path, output_dir=True):
        """Compare two jsons and report differences in keys and in values

        Args:
            json1_path (Path): baseline json file to compare
            json2_path (Path): new json file to compare
            write_reports (bool, optional): _description_. Defaults to True.
        """
        json1 = self.load_json(json1_path)
        json2 = self.load_json(json2_path)

        # Compare differences in json keys
        key_diffs = self.compare_dict_keys(json1, json2, [json1_path, json2_path])
        if output_dir is None:
            output_dir = json2_path.parent
        self.write_dict_key_report(key_diffs, output_dir / f"{json2_path.stem}_key_diffs.csv")

        # Compare differences in json values
        val_diffs = self.compare_dict_values(json1, json2)
        self.write_dict_value_report(val_diffs, output_dir / f"{json2_path.stem}_value_diffs.csv")

    def compare_summary_reports(self, report1_path, report2_path, output_dir=None):
        """Compare Summary_Data-TP.xlsx and Summary_Data-MAP.xlsx with baseline files

        Args:
            report1_path (Path): baseline summary report to compare
            report2_path (Path): new summary report to compare
            output_dir (Path, optional): _description_. Defaults to None.
        """

        reports1 = self.load_summary_report(report1_path)
        reports2 = self.load_summary_report(report2_path)
        if output_dir is None:
            output_dir = report2_path.parent
        output_path = output_dir / f"{report2_path.stem}_percent_diffs.xlsx"
        with pd.ExcelWriter(output_path) as writer:
            for (output_type, report1), (_, report2) in zip(reports1.items(), reports2.items()):
                diff = ((report2 - report1)/report1).round(2)
                diff = diff.reset_index()
                diff.to_excel(writer, sheet_name=output_type, index=False)
        print(f"Wrote Summary_Data percent difference report to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare results files for Scout.")
    parser.add_argument("--json-baseline", type=Path, help="Path to the baseline JSON file")
    parser.add_argument("--json-new", type=Path, help="Path to the new JSON file")
    parser.add_argument("--summary-baseline", type=Path,
                        help="Path to the baseline summary report (Excel file)")
    parser.add_argument("--summary-new", type=Path,
                        help="Path to the new summary report (Excel file)")
    parser.add_argument("--new-dir", type=Path, help="Directory containing files to compare")
    parser.add_argument("--base-dir", type=Path, help="Directory containing files to compare")
    parser.add_argument("--threshold", type=float, default=10,
                        help="Threshold for percent difference")
    args = parser.parse_args()

    compare = ScoutCompare()
    if args.base_dir and args.new_dir:
        # Compare all files
        base_dir = args.base_dir.resolve()
        new_dir = args.new_dir.resolve()
        agg_json_base = base_dir / "agg_results.json"
        agg_json_new = new_dir / "agg_results.json"
        compare.compare_jsons(agg_json_base, agg_json_new, output_dir=new_dir)
        ecm_json_base = base_dir / "ecm_results.json"
        ecm_json_new = new_dir / "ecm_results.json"
        compare.compare_jsons(ecm_json_base, ecm_json_new, output_dir=new_dir)

        summary_tp_base = base_dir / "Summary_Data-TP.xlsx"
        summary_tp_new = new_dir / "plots" / "tech_potential" / "Summary_Data-TP.xlsx"
        compare.compare_summary_reports(summary_tp_base, summary_tp_new, output_dir=new_dir)
        summary_map_base = base_dir / "Summary_Data-MAP.xlsx"
        summary_map_new = new_dir / "plots" / "max_adopt_potential" / "Summary_Data-MAP.xlsx"
        compare.compare_summary_reports(summary_map_base, summary_map_new, output_dir=new_dir)
    else:
        # Compare only as specified by the arguments
        if args.json_baseline and args.json_new:
            compare.compare_jsons(args.json_baseline, args.json_new)
        if args.summary_baseline and args.summary_new:
            compare.compare_summary_reports(args.summary_baseline, args.summary_new)


if __name__ == "__main__":
    main()
