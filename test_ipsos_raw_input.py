"""Data tests for SDD raw input from IPSOS

These are pytest tests for a raw .SAV SPSS file, that check values based on
metadata saved in \\ic\ic_dme_dfs\QME_PROD\LIFE\LIFE_SDD\Publication\RAP\Metadata\Latest
To test a different .sav file or use different metadata, change the command line
arguments --sdd_file and --sdd_metadata.

These tests are written based on the 2018 SAV input file, using metadata from that
file. If there are substanstive changes to the format (for example responses are no
longer integer codes) then these tests may have to be rewritten.

For testing a new iteration of SDD, run the file run_import_validation with a new
sdd_file and sdd_metadata location (initally metadata will be copied from 2018).
Then iteratively update the metadata or request new data from IPSOS as issues are
found and either corrected, or the metadata is updated.
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from sdd_code.utilities.data_import import import_sav_values
from sdd_code.utilities.metadata import convert_sdd_dtypes
from sdd_code.utilities.parameters import DROP_COLUMNS, PUPIL_DATA_PATH, META_DIR

# Set any test constants
NUM_ROWS = 13664


# sdd_file is set via command line arguments, defaults to None
# session scoping to cache dataframes, enable fast testing
@pytest.fixture(scope="session")
def sdd_all(sdd_file: str) -> pd.DataFrame:
    """Get main sdd output table and output to test"""
    if sdd_file:
        data = convert_sdd_dtypes(import_sav_values(sdd_file, DROP_COLUMNS))
    else:
        file_path = PUPIL_DATA_PATH
        data = convert_sdd_dtypes(import_sav_values(file_path, DROP_COLUMNS))
    return data


# sdd_metadata is set via command line
@pytest.fixture(scope="session")
def sdd_dtypes(sdd_metadata: str) -> Dict[str, str]:
    """Get expected sdd attributes from metadata JSON"""
    if sdd_metadata:
        sdd_meta_loc = Path(sdd_metadata)
    else:
        sdd_meta_loc = META_DIR
    with open(sdd_meta_loc / "sdd_dtype_map.json", "r") as f:
        dtype_map = json.load(f)
    return dtype_map


@pytest.fixture(scope="session")
def sdd_allowed_values(sdd_metadata: str) -> Dict[str, List[float]]:
    """Get allowed values, dict map of col name to list"""
    if sdd_metadata:
        sdd_meta_loc = Path(sdd_metadata)
    else:
        sdd_meta_loc = META_DIR
    with open(sdd_meta_loc / "sdd_allowed_values_map.json", "r") as f:
        value_map = json.load(f)

    return value_map


@pytest.fixture(scope="session")
def sdd_columns(sdd_metadata: str) -> Dict[str, List[str]]:
    """Get all different column lists that could want to be tested

    This uses dtypes from metadata to separate columns, logic may change
    as metadata structure is determined
    """
    if sdd_metadata:
        sdd_meta_loc = Path(sdd_metadata)
    else:
        sdd_meta_loc = META_DIR
    with open(sdd_meta_loc / "sdd_column_types.json", "r") as f:
        cols = json.load(f)
    return cols


def test_input_attributes(sdd_all: pd.DataFrame, sdd_dtypes: Dict[str, str]):
    """Does table have basic attributes as expected?

    Tests: No. rows, dtypes, column names
    """
    results = sdd_all
    exp_cols = sdd_dtypes.keys()
    # Convert string repr. of types to numpy.dtype for comparison
    exp_dtypes = {col: np.dtype(dtype) for col, dtype in sdd_dtypes.items()}
    exp_rows = NUM_ROWS  # Know exactly how many rows we should have

    assert list(exp_cols) == list(results.columns), "Expected columns not found"
    assert exp_rows == results.shape[0], "Not expected number of rows"
    assert exp_dtypes == results.dtypes.to_dict(), "Not expected data types"


# Currently cant pass fixtures as the list for parameterise
# TODO: Investigate pytest-lazy plugins plus similar
@pytest.mark.slow
# Parametrize loops over all columns
@pytest.mark.parametrize("column_index", list(range(0, 700)))
def test_discrete_values(
    sdd_all: pd.DataFrame,
    sdd_columns: Dict[str, List[str]],
    sdd_allowed_values: Dict[str, List[float]],
    column_index: int,
):
    """Do all discrete values fall within expected sets"""
    results = sdd_all

    # Use index list to get column name, can't pass exact list into
    # parametrize so use try/except
    try:
        column = sdd_columns["discrete"][column_index]
    except IndexError:
        return
    # Discrete values are integer response codes that should all be within
    # the allowed list in metadata
    exp_values = sdd_allowed_values[column]
    values_in_list = results[column].isin(exp_values)
    assert all(
        values_in_list
    ), f"Unexpected value in {column}: {dict(results.loc[~values_in_list, column])}"


@pytest.mark.slow
@pytest.mark.parametrize("column_index", list(range(0, 700)))
def test_continuous_values(
    sdd_all: pd.DataFrame,
    sdd_columns: Dict[str, List[str]],
    sdd_allowed_values: Dict[str, List[float]],
    column_index: int,
):
    """Do all continuous col values fall within expected range"""
    results = sdd_all
    try:
        column = sdd_columns["continuous"][column_index]
    except IndexError:
        return

    # Check if column has a min/max mapping to use as a range
    values = sdd_allowed_values[column]
    min_max = [val for val in values if val >= 0]
    if min_max:
        max_val = max(min_max)
        min_val = min(min_max) if min(min_max) != max_val else 0
    else:
        min_val = 0
        max_val = 500

    # Continuous values are float responses (i.e. units drank) that should be
    # in this range, or are coded as negatives missiing/unknown/other
    values_in_range = results[column].between(min_val, max_val) | results[column].isin(
        [-9, -8, -1]
    )

    assert all(
        values_in_range
    ), f"Unexpected values in {column}, {dict(results.loc[~values_in_range, column])}"


def test_pupilwt(sdd_all: pd.DataFrame):
    """Do all weights lie in expected ranges and sum to appropriate values"""
    assert (
        sdd_all["pupilwt"].between(0.01, 10).all()
    ), "Pupil weighting variable is not in range"

    assert sdd_all["pupilwt"].sum() == pytest.approx(
        NUM_ROWS
    ), "Pupil weighting does not sum to pupil count"


def test_null_vals(sdd_all: pd.DataFrame):
    assert any(sdd_all.isnull()), "Null values found in dataframe"


def test_unique_keys(sdd_all: pd.DataFrame):
    assert sdd_all["archsn"].is_unique, "ID column is not unique"
