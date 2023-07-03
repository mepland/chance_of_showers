################################################################################
# python imports
import hydra
import os
import glob
import datetime
from zoneinfo import ZoneInfo
import polars as pl


@hydra.main(version_base=None, config_path="..", config_name="config")
def etl(cfg):
    # setup variables
    package_path = cfg["general"]["package_path"]
    raw_data = cfg["daq"]["raw_data"]
    saved_data = cfg["daq"]["saved_data"]

    date_fmt = cfg["general"]["date_fmt"]
    time_fmt = cfg["general"]["time_fmt"]
    datetime_fmt = f"{date_fmt} {time_fmt}"

    # when t_start = t_start.replace(minute=t_start_minute, second=0, microsecond=0) fixed the issue of drifting seconds
    dt_end_of_drifting_seconds = datetime.datetime.strptime(
        cfg["daq"]["end_of_drifting_seconds"], datetime_fmt
    ).replace(tzinfo=ZoneInfo("UTC"))

    # when web threading was fixed, eliminating duplicate records from multiple threads
    dt_end_of_threading_duplicates = datetime.datetime.strptime(
        cfg["daq"]["end_of_threading_duplicates"], datetime_fmt
    ).replace(tzinfo=ZoneInfo("UTC"))

    # load raw csv files
    dfpl_list = []
    for f in glob.glob(os.path.join(package_path, raw_data, "*.csv")):
        try:
            dfpl = pl.scan_csv(f)
            dfpl = dfpl.with_columns(pl.lit(f.split("/")[-1]).alias("fname"))
            dfpl_list.append(dfpl)
        except:
            raise ValueError(f"Error loading file {f}")

    dfpl = pl.concat(dfpl_list)
    dfpl = (
        # convert date columns
        dfpl.with_columns(
            pl.col("datetime_utc")
            .str.to_datetime(datetime_fmt)
            .dt.replace_time_zone("UTC")
        ).with_columns(
            pl.col("datetime_utc")
            .dt.convert_time_zone("US/Eastern")
            .alias("datetime_est")
        )
        # Add more date columns
        .with_columns(
            pl.col("datetime_est").dt.weekday().alias("day_of_week_int"),
            pl.col("datetime_est").dt.to_string("%A").alias("day_of_week_str"),
            # pl.col('datetime_est').dt.to_string(time_hm_fmt).alias('time'),
        )
    )

    # remove any datetimes that have more than 1 row, caused by historical bug in DAQ threading
    dfpl_duplicate_datetime = (
        dfpl.groupby(by=["datetime_utc"]).agg(pl.count()).filter(1 < pl.col("count"))
    )
    dfpl = dfpl.join(
        dfpl_duplicate_datetime.select("datetime_utc"),
        on="datetime_utc",
        how="anti",
    )

    # make sure none of the duplicates are post threading fix
    dfpl_duplicate_datetime_post_threading_fix = dfpl_duplicate_datetime.filter(
        dt_end_of_threading_duplicates < pl.col("datetime_utc")
    )

    n_duplicate_datetime_post_threading_fix = (
        dfpl_duplicate_datetime_post_threading_fix.select(pl.count())
        .collect(streaming=True)
        .item()
    )
    if 0 < n_duplicate_datetime_post_threading_fix:
        duplicate_datetime_post_fix_min = (
            dfpl_duplicate_datetime_post_threading_fix.select(
                pl.col("datetime_utc").min()
            )
            .collect(streaming=True)
            .item()
            .strftime(datetime_fmt)
        )
        duplicate_datetime_post_fix_max = (
            dfpl_duplicate_datetime_post_threading_fix.select(
                pl.col("datetime_utc").max()
            )
            .collect(streaming=True)
            .item()
            .strftime(datetime_fmt)
        )
        raise ValueError(
            (
                f"Found {n_duplicate_datetime_post_threading_fix} datetimes with multiple entries"
                f" after {dt_end_of_threading_duplicates.strftime(datetime_fmt)} UTC!"
                f"\nDuplicates are between {duplicate_datetime_post_fix_min} and {duplicate_datetime_post_fix_max}."
            )
        )

    # check for drifting seconds after fix implemented at dt_end_of_drifting_seconds
    dfpl_drift_seconds_records = dfpl.filter(
        (dt_end_of_drifting_seconds < pl.col("datetime_utc"))
        & (pl.col("datetime_utc").dt.second() != 0)
    )
    n_rows_drift_seconds = (
        dfpl_drift_seconds_records.select(pl.count()).collect(streaming=True).item()
    )
    if 0 < n_rows_drift_seconds:
        print(dfpl_drift_seconds_records.collect(streaming=True))
        raise ValueError(
            f"Found {n_rows_drift_seconds = } after {dt_end_of_drifting_seconds.strftime(datetime_fmt)} UTC!"
        )

    # print(dfpl.fetch(1).limit(10))

    print(dfpl.collect(streaming=True).describe())
    # print(dfpl.select(pl.col("mean_pressure_value")).collect(streaming=True).describe())

    # dfpl.sink_parquet(path, maintain_order=False)
    dfpl.collect(streaming=True).write_parquet(
        os.path.join(package_path, saved_data, "etl_data.parquet")
    )


if __name__ == "__main__":
    etl()
