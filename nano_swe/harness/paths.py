"""Filesystem layout inside the Daytona sandbox `episode.py` builds and runs against.

``REWARD_PATH`` must match the ``LOG_DIR`` written into ``GRADE_PY`` by
``nano_swe/swe_data/build_dataset.py`` — that script's own reward-writing logic runs as a
standalone file inside the sandbox, in whatever bare Python env the task's image ships (no
nano_swe install there), so it can't import this module and keep them in sync automatically.
"""

WORKSPACE_DIR = "/workspace"
TASK_DIR = f"{WORKSPACE_DIR}/task"
TASK_TESTS_DIR = f"{TASK_DIR}/tests"
INSTRUCTION_PATH = f"{TASK_DIR}/instruction.md"
DRIVER_PATH = f"{WORKSPACE_DIR}/driver.py"
TASK_TEST_FILES = ("test_patch.diff", "instance.json", "grade.py")

REPO_DIR = "/testbed"  # SWE-bench harness convention: repo checkout inside the prebuilt image
TESTBED_PYTHON = "/opt/miniconda3/envs/testbed/bin/python"  # SWE-bench harness convention

OH_VENV_DIR = "/opt/oh-venv"
OH_PYTHON = f"{OH_VENV_DIR}/bin/python"

REWARD_PATH = "/logs/verifier/reward.txt"
