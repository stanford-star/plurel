import numpy as np
import pandas as pd
import pytest

from plurel import (
    Categorical,
    Counts,
    ForeignKey,
    HSBMLink,
    Numeric,
    PrimaryKey,
    Range,
    Recorded,
    Schema,
    Summary,
    Table,
    Uniform,
)
from plurel.links import counted, matched

REGIONS = ("north", "south", "east")


def clinics(**key):
    return Schema(
        {
            "clinics": Table(
                {
                    "id": PrimaryKey(),
                    "region": Categorical({"north": 0.5, "south": 0.3, "east": 0.2}),
                    "capacity": Numeric(Uniform(10, 100)),
                }
            ),
            "patients": Table(
                {
                    "id": PrimaryKey(),
                    "region": Categorical(REGIONS),
                    "clinic_id": ForeignKey("clinics", **key),
                }
            ),
        }
    )


def joined(frames, child, key, parent):
    return frames[child].merge(
        frames[parent], how="left", left_on=key, right_on="id", suffixes=("", "_parent")
    )


def test_a_matched_key_points_at_eligible_parents_by_preference():
    schema = clinics(match={"region": "region"}, prefer={Recorded("capacity"): 0.05})
    frames = schema.sample({"clinics": 30, "patients": 6000}, seed=0)
    rows = joined(frames, "patients", "clinic_id", "clinics")
    assert (rows.region == rows.region_parent).all()
    load = frames["patients"].clinic_id.value_counts().reindex(range(30), fill_value=0)
    assert pd.Series(load.to_numpy()).corr(frames["clinics"].capacity, method="spearman") > 0.8
    again = schema.sample({"clinics": 30, "patients": 6000}, seed=0)
    assert all(frames[t].equals(again[t]) for t in frames)


def test_rows_no_parent_matches_point_at_nothing_only_on_a_nullable_key():
    rows = {"clinics": 2, "patients": 300}
    with pytest.raises(ValueError, match="match no row"):
        clinics(match={"region": "region"}).sample(rows, seed=0)
    frames = clinics(match={"region": "region"}, nullable=0.01).sample(rows, seed=0)
    linked = joined(frames, "patients", "clinic_id", "clinics")
    unlinked = linked.clinic_id.isna()
    absent = set(REGIONS) - set(frames["clinics"].region)
    assert absent and unlinked[linked.region.isin(absent)].all()
    assert (linked.region[~unlinked] == linked.region_parent[~unlinked]).all()


def test_keys_weigh_by_what_the_parent_reads_and_match_within_a_table():
    schema = Schema(
        {
            "cities": Table({"id": PrimaryKey(), "rich": Numeric(Uniform(0, 1))}),
            "shops": Table({"id": PrimaryKey(), "city_id": ForeignKey("cities")}),
            "visits": Table(
                {"shop_id": ForeignKey("shops", prefer={Recorded("city_id.rich"): 8.0})}
            ),
            "teams": Table({"id": PrimaryKey()}),
            "staff": Table(
                {
                    "id": PrimaryKey(),
                    "team_id": ForeignKey("teams"),
                    "boss_id": ForeignKey("staff", match={"team_id": "team_id"}),
                }
            ),
        }
    )
    rows = {"cities": 20, "shops": 200, "visits": 20000, "teams": 6, "staff": 300}
    frames = schema.sample(rows, seed=0)
    visits = frames["visits"].merge(frames["shops"], left_on="shop_id", right_on="id")
    visits = visits.merge(frames["cities"], left_on="city_id", right_on="id")
    assert visits.rich.mean() > frames["cities"].rich.mean() + 0.2
    bosses = frames["staff"].merge(frames["staff"], left_on="boss_id", right_on="id")
    assert len(bosses) == 300 and (bosses.team_id_x == bosses.team_id_y).all()


def test_counts_noise_alone_spreads_the_counts():
    def spread(noise):
        orders = ForeignKey("customers", counts=Counts(5.0, noise=noise))
        schema = Schema(
            {"customers": Table({"id": PrimaryKey()}), "orders": Table({"customer_id": orders})}
        )
        per = schema.sample({"customers": 4000}, seed=0)["orders"].customer_id.value_counts()
        return per.reindex(range(4000), fill_value=0).var()

    assert abs(spread(None) - 5.0) < 0.5 and spread(1.0) > 20.0


def test_keys_match_on_the_rows_other_keys_point_at():
    schema = Schema(
        {
            "merchants": Table({"id": PrimaryKey()}),
            "products": Table({"id": PrimaryKey(), "merchant_id": ForeignKey("merchants")}),
            "orders": Table(
                {
                    "merchant_id": ForeignKey("merchants", HSBMLink(popularity=Uniform(0.5, 2))),
                    "product_id": ForeignKey("products", match={"merchant_id": "merchant_id"}),
                }
            ),
        }
    )
    frames = schema.sample({"merchants": 20, "products": 400, "orders": 3000}, seed=0)
    sold = frames["orders"].merge(
        frames["products"], left_on="product_id", right_on="id", suffixes=("", "_product")
    )
    assert len(sold) == 3000 and (sold.merchant_id == sold.merchant_id_product).all()


def test_counts_follow_parent_columns_and_set_the_size():
    def schema(mean):
        orders = ForeignKey(
            "customers", counts=Counts(mean, {"tenure": 0.8, "segment": {"b": 1.0}})
        )
        return Schema(
            {
                "customers": Table(
                    {
                        "id": PrimaryKey(),
                        "tenure": Numeric(Uniform(0, 10)),
                        "segment": Categorical(("a", "b")),
                        "orders": Numeric(causes={Summary("orders", "customer_id"): 1.0}, noise=0),
                    }
                ),
                "orders": Table({"id": PrimaryKey(), "customer_id": orders}),
            }
        )

    for rows, total in (({"customers": 3000}, None), ({"customers": 3000, "orders": 4000}, 4000)):
        frames = schema(5.0).sample(rows, seed=0)
        per = frames["orders"].customer_id.value_counts().reindex(range(3000), fill_value=0)
        customers = frames["customers"].assign(n=per.to_numpy())
        assert np.array_equal(customers.n, customers.orders)
        assert (customers.n == 0).any() and customers.n.sum() == len(frames["orders"])
        if total is None:
            assert abs(len(frames["orders"]) / 15000 - 1) < 0.05
        else:
            assert len(frames["orders"]) == total
        means = customers.groupby("segment").n.mean()
        assert means["b"] > 2 * means["a"]
        assert customers.n.corr(customers.tenure, method="spearman") > 0.3
    realized = schema(Range(2.0, 6.0)).realize(0)
    assert realized.tables["orders"].fkeys["customer_id"].counts.mean != Range(2.0, 6.0)
    assert realized.sample({"customers": 10}, seed=0)["orders"].equals(
        realized.sample({"customers": 10}, seed=0)["orders"]
    )


def test_a_counted_table_can_be_the_parent_of_more_counts():
    schema = Schema(
        {
            "customers": Table({"id": PrimaryKey()}),
            "orders": Table(
                {"id": PrimaryKey(), "customer_id": ForeignKey("customers", counts=Counts(3.0))}
            ),
            "items": Table({"order_id": ForeignKey("orders", counts=Counts(2.0))}),
        }
    )
    frames = schema.sample({"customers": 500}, seed=1)
    assert frames["items"].order_id.max() < len(frames["orders"])
    assert abs(len(frames["items"]) / (2 * len(frames["orders"])) - 1) < 0.1


@pytest.mark.parametrize(
    "build, match",
    [
        (lambda: ForeignKey("p", HSBMLink(), match={"a": "a"}), "no link or values"),
        (lambda: ForeignKey("p", counts=Counts(2.0), nullable=0.1), "no match, prefer"),
        (lambda: Counts(0.0), "positive count"),
        (
            lambda: Table(
                {"a": ForeignKey("p", counts=Counts(1.0)), "b": ForeignKey("q", counts=Counts(1.0))}
            ),
            "set the table's size",
        ),
        (lambda: Table({"k": ForeignKey("p", match={"x": "x"}), "x": Numeric()}), "categorical"),
        (lambda: clinics(match={"region": "capacity"}), "no categorical"),
        (lambda: clinics(prefer={("capacity", "region"): 1.0}), "no products"),
        (lambda: clinics(prefer={"nothing": 1.0}), "unknown columns"),
        (
            lambda: Schema(
                {"t": Table({"id": PrimaryKey(), "k": ForeignKey("t", counts=Counts(1.0))})}
            ),
            "its own table",
        ),
    ],
)
def test_keys_reject_what_they_cannot_mean(build, match):
    with pytest.raises(ValueError, match=match):
        build()


def test_a_key_cannot_match_on_what_reads_across_it():
    with pytest.raises(ValueError, match="acyclic"):
        Schema(
            {
                "clinics": Table({"id": PrimaryKey(), "region": Categorical(REGIONS)}),
                "patients": Table(
                    {
                        "clinic_id": ForeignKey("clinics", match={"region": "region"}),
                        "region": Categorical(REGIONS, causes={"clinic_id.region": {"north": 1.0}}),
                    }
                ),
            }
        )
    with pytest.raises(ValueError, match="no key of"):
        Schema(
            {
                "a": Table({"id": PrimaryKey()}),
                "b": Table({"id": PrimaryKey(), "a_id": ForeignKey("a")}),
                "c": Table(
                    {"b_id": ForeignKey("b"), "x_id": ForeignKey("b", match={"b_id": "a_id"})}
                ),
            }
        )


def test_matched_and_counted_draw_within_groups_and_shares():
    rng = np.random.default_rng(0)
    parent = np.array([0, 1, 1, -1, 2, 3, 3])
    log_weights = np.array([1000.0, 0.0, np.log(3.0), 5.0, -np.inf, -2000.0, -2000.0 + np.log(3.0)])
    picks = matched(np.array([1] * 4000 + [3] * 4000 + [0, 2, -1, 4]), parent, log_weights, rng)
    assert set(picks[:4000].tolist()) == {1, 2} and abs((picks[:4000] == 2).mean() - 0.75) < 0.03
    assert (
        set(picks[4000:8000].tolist()) == {5, 6}
        and abs((picks[4000:8000] == 6).mean() - 0.75) < 0.03
    )
    assert picks[8000:].tolist() == [0, -1, -1, -1]
    owners = counted(np.array([1.0, 0.0, 3.0]), 2.0, 100, rng)
    assert len(owners) == 100 and 1 not in owners
    assert len(counted(np.empty(0), 2.0, None, rng)) == 0
    with pytest.raises(ValueError, match="no parent rows"):
        counted(np.empty(0), 2.0, 5, rng)
