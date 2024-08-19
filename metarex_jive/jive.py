from typing import List, Optional
import numpy as np
import pandas as pd
from pypika import Field, Query, Table, functions as F

from .column_map import ColumnMap


class JIVE:
    def __init__(self, table, dimensions, measures, conn):
        self.table = table
        self.dimensions = dimensions
        self.measures = measures
        self.conn = conn

    def query(self, conn=None):
        if not conn:
            conn = self.conn
        return get_all_moments(self.table, self.dimensions, self.measures)


def to_treatment_effects(
    moments: pd.DataFrame,
    test_id_cols: List[str],
    arm_id_col: str,
    weight_col: Optional[str] = None,
):
    reference_cells = moments.groupby(test_id_cols, as_index=False)[arm_id_col].min()
    reference_cell_data = moments.merge(reference_cells)
    treatment_effects = moments.merge(
        reference_cell_data, on=test_id_cols, suffixes=["|t", "|r"]
    ).pipe(lambda df: df[df[f"{arm_id_col}|t"] != df[f"{arm_id_col}|r"]])
    if weight_col:
        # Assume these are sample sizes and convert to harmonic means.
        treatment_effects[weight_col] = 4 / (
            1 / treatment_effects[f"{weight_col}|t"]
            + 1 / treatment_effects[f"{weight_col}|r"]
        )
    return treatment_effects


def _get_covariance_matrix_from_row(row, metrics, kind):
    cov = np.zeros((len(metrics) + 1, len(metrics) + 1))
    cov[0, 0] += 1

    if kind == "summary":
        for i, m1 in enumerate(metrics):
            cov[0, i + 1] = row[f"{m1}|avg"]  # Fill out first row
            cov[i + 1, 0] = row[f"{m1}|avg"]  # Fill out first col
            for j, m2 in enumerate(metrics):
                cov[i + 1, j + 1] = row[f"{m1}:{m2}|loo_cov"]
    elif kind == "treatment_effect":
        for i, m1 in enumerate(metrics):
            cov[0, i + 1] = row[f"{m1}|avg|t"] - row[f"{m1}|avg|r"]
            cov[i + 1, 0] = row[f"{m1}|avg|t"] - row[f"{m1}|avg|r"]
            for j, m2 in enumerate(metrics):
                cov[i + 1, j + 1] = (
                    row[f"{m1}:{m2}|loo_cov|t"]
                    + row[f"{m1}:{m2}|loo_cov|r"]
                    - np.outer(row[f"{m1}|avg|t"], row[f"{m2}|avg|r"])
                    - np.outer(row[f"{m1}|avg|r"], row[f"{m2}|avg|t"])
                )
    return cov


def _get_covariance_matrix(df, metrics, weight_col=None, kind="treatment_effect"):
    cov = np.zeros((len(metrics) + 1, len(metrics) + 1))
    for _, row in df.iterrows():
        w = row.get(weight_col) or 1
        cov += w * _get_covariance_matrix_from_row(row, metrics, kind=kind)
    return cov


def get_base_query(table, dimensions, measures):
    fields = [Field(d, table=table) for d in dimensions] + [
        Field(m, table=table) for m in measures
    ]
    query = Query.from_(table).select(*fields)
    for dimension in dimensions:
        query = query.where(Field(dimension, table=table).notnull())
    return query


def get_agg_query(table, dimensions, measures):
    base_query = get_base_query(table, dimensions, measures).as_("base_for_agg")
    return (
        Query.from_(base_query)
        .select(
            *[Field(d, table=base_query) for d in dimensions],
            *[F.Sum(Field(m)).as_(f"{m}|sum") for m in measures],
            *[F.Count(Field(m)).as_(f"{m}|count") for m in measures],
        )
        .groupby(*[Field(d) for d in dimensions])
    )


def get_joined_query(table, dimensions, measures):
    base_query = get_base_query(table, dimensions, measures)
    agg_query = get_agg_query(table, dimensions, measures).as_("agg")
    return (
        base_query.join(agg_query)
        .on_field(*dimensions)
        .select(
            *[
                (agg_query[f"{m}|sum"] - Field(m, table=table)).as_(f"{m}|loo_sum")
                for m in measures
            ],
            *[(agg_query[f"{m}|count"] - 1).as_(f"{m}|loo_count") for m in measures],
        )
    )


def get_all_moments(table, dimensions, measures):
    joined_query = get_joined_query(table, dimensions, measures)

    def loo_avg(measure):
        return (Field(f"{measure}|loo_sum") / Field(f"{measure}|loo_count")).as_(
            f"{measure}|loo_avg"
        )

    return (
        Query.from_(joined_query)
        .select(
            *[Field(d) for d in dimensions],
            *[F.Avg(Field(m)).as_(f"{m}|avg") for m in measures],
            *[
                F.Avg(Field(m1) * loo_avg(m2)).as_(f"{m1}:{m2}|loo_cov")
                for m1 in measures
                for m2 in measures
            ],
            F.Count("*").as_("row_count"),
        )
        .groupby(*[Field(d) for d in dimensions])
    )
