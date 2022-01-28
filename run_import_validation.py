"""Main file for running the import data tests

Includes command line arguments for running the pytest data validation tests. Creates
a html report of the output
"""

import time
from pathlib import Path

import pytest


def run_import_tests(
    test_loc: Path,
    report_loc: Path,
    file_loc: Path,
    meta_loc: Path,
) -> None:
    """Run the set of import tests

    Args:
        test_loc: Optional, test file/folder to run
        report_loc: Optional, where to store the test report
        file_loc: Optional, where the data to test is located
        meta_loc: Optional, where the metadata to test against is located.

    Returns:
        None

    Raises:
        RuntimeError: If tests are not passing (i.e. non-zero return code)
    """
    run_time = time.strftime("%Y%m%d-%H%M%S")
    report_loc = report_loc / f"input_report_{run_time}.html"

    input_test_arguments = [
        # Which tests to run:
        str(test_loc),
        # Which file to run them on:
        f"--sdd_file={file_loc}",
        # What metadata to verify against:
        f"--sdd_metadata={meta_loc}",
        # Where to save the report:
        f"--html={report_loc}",
        "--self-contained-html",
    ]

    rc = pytest.main(input_test_arguments)
    if rc:
        raise RuntimeError(
            f"Tests failing, check latest output report: {report_loc.resolve()}"
        )


if __name__ == "__main__":
    # Run tests
    run_import_tests()
