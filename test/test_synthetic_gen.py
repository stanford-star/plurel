import subprocess
import sys
from pathlib import Path

from relbench.load import load_dataset

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "synthetic_gen.py"


def run(*flags: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *flags], capture_output=True, text=True)


def test_script_writes_one_loadable_database_per_seed(tmp_path):
    flags = ("--seed_offset", "3", "--num_proc", "2", "--out_dir", str(tmp_path))
    done = run(*flags, "--num_dbs", "2")
    assert done.returncode == 0, done.stderr
    names = sorted(line.split(":")[0] for line in done.stdout.splitlines())
    assert names == ["plurel-3", "plurel-4"]
    for seed in (3, 4):
        dataset = load_dataset(tmp_path / f"plurel-{seed}")
        assert dataset.val_timestamp < dataset.test_timestamp
        db = dataset.get_db()
        assert db.max_timestamp <= dataset.test_timestamp
        for table in db.table_dict.values():
            for column, parent in table.fkey_col_to_pkey_table.items():
                assert table.df[column].dropna().lt(len(db.table_dict[parent].df)).all()
    again = run(*flags, "--num_dbs", "1")
    assert again.returncode != 0 and "overwrite" in again.stderr
    assert run(*flags, "--num_dbs", "1", "--overwrite").returncode == 0
