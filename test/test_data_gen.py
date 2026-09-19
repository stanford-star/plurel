import glob
import os
import shutil

import pytest

from plurel.config import Choices, Config, DatabaseParams
from plurel.dataset import SyntheticDataset
from plurel.utils import set_random_seed

# All tests share scratch state via module-scoped fixtures, so they
# must run on the same xdist worker to avoid races.
pytestmark = pytest.mark.xdist_group("dev_run")

TAG = "pytest"
NUM_DBS = 10
SEED_OFFSET = 0
SEEDS = list(range(SEED_OFFSET, SEED_OFFSET + NUM_DBS))

HOME = os.path.expanduser("~")
SCRATCH_RELBENCH = os.path.join(HOME, "scratch", "relbench")

# Small tables so the test runs fast
SMALL_DB_PARAMS = DatabaseParams(
    num_tables_choices=Choices(kind="range", value=[3, 5]),
    num_rows_entity_table_choices=Choices(kind="range", value=[40, 80]),
    num_rows_activity_table_choices=Choices(kind="range", value=[100, 200]),
)


def _db_name(seed: int) -> str:
    return f"plurel-test-t{TAG}-s{seed}"


@pytest.fixture(scope="module")
def generated_dbs():
    """Generate all small DBs once, caching in relbench-3.0.0 format.
    Cleans up all created directories when the module is done."""
    dbs = {}
    for seed in SEEDS:
        db_name = _db_name(seed)
        cache_dir = os.path.join(SCRATCH_RELBENCH, db_name)
        set_random_seed(0)
        dataset = SyntheticDataset(
            seed=seed,
            config=Config(
                database_params=SMALL_DB_PARAMS,
                cache_dir=cache_dir,
            ),
        )
        # get_db() writes the dataset in relbench-3.0.0 format:
        # manifest.yaml + {cache_dir}/db/*.parquet
        dbs[seed] = dataset.get_db()

    yield dbs

    for seed in SEEDS:
        d = os.path.join(SCRATCH_RELBENCH, _db_name(seed))
        if os.path.exists(d):
            shutil.rmtree(d)


def test_all_dbs_generated(generated_dbs):
    assert len(generated_dbs) == NUM_DBS
    for seed, db in generated_dbs.items():
        assert db is not None
        assert len(db.table_dict) > 0


def test_relbench_3_0_0_format(generated_dbs):
    """Verify that get_db() wrote a self-describing relbench-3.0.0 dataset dir."""
    from relbench.manifest import DatasetManifest

    for seed in SEEDS:
        dataset_dir = os.path.join(SCRATCH_RELBENCH, _db_name(seed))
        db_dir = os.path.join(dataset_dir, "db")
        assert os.path.isdir(db_dir), f"Missing cache dir: {db_dir}"
        parquets = glob.glob(os.path.join(db_dir, "*.parquet"))
        assert len(parquets) > 0, f"No parquet files in {db_dir}"

        manifest_path = os.path.join(dataset_dir, "manifest.yaml")
        assert os.path.isfile(manifest_path), f"Missing manifest: {manifest_path}"
        manifest = DatasetManifest.load(manifest_path)
        assert manifest.name == _db_name(seed)
        assert manifest.val_timestamp and manifest.test_timestamp
        table_stems = {os.path.splitext(os.path.basename(p))[0] for p in parquets}
        assert set(manifest.tables) == table_stems


def test_relbench_load_dataset_roundtrip(generated_dbs):
    """Generated dataset dirs load with relbench 3.0.0's load_dataset."""
    from relbench.load import load_dataset

    seed = SEEDS[0]
    dataset_dir = os.path.join(SCRATCH_RELBENCH, _db_name(seed))
    rb = load_dataset(dataset_dir)
    db = rb.get_db()
    assert set(db.table_dict) == set(generated_dbs[seed].table_dict)
