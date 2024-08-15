from pypika import Field, Query, Table, functions as F

from .column_map import ColumnMap


class JIVE:
    def __init__(self):
        pass

    def build_sql_query(self):
        pass

    def query(self, conn=None):
        if not conn:
            conn = self.conn
        pass


def get_base_query(table, dimensions, measures):
    fields = [Field(d) for d in dimensions] + [Field(m) for m in measures]
    return Query.from_(table).select(*fields)


def get_agg_query(table, dimensions, measures):
    base_query = get_base_query(table, dimensions, measures).as_("base")
    return (
        Query.from_(base_query)
        .select(
            *[Field(d) for d in dimensions],
            *[F.Sum(Field(m)).as_(f"{m}|sum") for m in measures],
            *[F.Count(Field(m)).as_(f"{m}|count") for m in measures],
        )
        .groupby(*[Field(d) for d in dimensions])
    )


def get_joined_query(table, dimensions, measures):
    base_query = get_base_query(table, dimensions, measures)
    agg_query = get_agg_query(table, dimensions, measures)
    fields = [Field(d) for d in dimensions] + [Field(m) for m in measures]
    return (
        base_query.join(agg_query)
        .on(*[base_query[d] == agg_query[d] for d in dimensions])
        .select(
            *fields,
            *[
                (agg_query[f"{m}|sum"] - base_query[m]).as_(f"{m}|loo_sum")
                for m in measures
            ],
            *[(agg_query[f"{m}|count"] - 1).as_(f"{m}|loo_count") for m in measures],
        )
    )


def get_all_moments(table, dimensions, measures):
    joined_query = get_joined_query(table, dimensions, measures)

    def loo_avg(measure):
        return (
            Field(f"{measure}|loo_sum") / Field(f"{measure}|loo_count")
        ).as_(f"{measure}|loo_avg")

    return (
        Query.from_(joined_query)
        .select(
            *[Field(d) for d in dimensions],
            *[F.Avg(Field(m)).as_(f"{m}_avg") for m in measures],
            *[
                F.Avg(Field(m) * loo_avg(m)).as_(f"{m}|loo_var")
                for m in measures
            ],
            *[
                F.Avg(Field(m1) * loo_avg(m2)).as_(f"{m1}:{m2}|loo_cov")
                for m1 in measures
                for m2 in measures
                if m1 != m2
            ],
            F.Count().as_("row_count"),
        )
        .groupby(*[Field(d) for d in dimensions])
    )
