"""
"""
import sys
from typing import Optional
from time import localtime, strftime
from numpy import vectorize
import polars as pl

@vectorize
def is_float(x: str) -> int:
    """
    Note, "" will return True
    """

    #does x only consist of digits or decimal points
    a1 = sum(
            [False if char_.isdigit() or char_=="." else True for char_ in x]
            )==0
    if not a1:
        return False

    #does value have a decimal point
    if x.find(".")==-1:
        return False

    else:
        #is there only 1 decimal point
        if x.count(".") == 1:

            #do values after the decimal point consist of digits other than 0
            if x.split('.')[-1] == '0'*len(x.split('.')[-1]):
                return False
            else:
                return True

        else:
            return False


@vectorize
def is_int(x: str, check_is_float: bool=True,) -> int:
    """
    Must check is_float 1st, otherwise will true if float or int
    Therefore, check_is_float==False makes this effectively is_numeric

    Note, "" will return True
    """
    if check_is_float:
        if is_float([x])[0] == True:
            return False

    #does x only consist of digits or decimal points
    a1 = sum(
            [False if char_.isdigit() or char_=="." else True for char_ in x]
            )==0

    #only one or no decimal points
    a2 = x.count(".") <= 1

    return a1 and a2


def cast_datatypes(
        dat: pl.LazyFrame,
        column_map: dict[str,pl.DataType] = {},
        null_values: Optional[list[str]] = None,
        ) -> pl.LazyFrame:
    """
    LazyFrame must have all columns with datatype pl.Utf8
    """
    for col_ in dat.collect_schema().names():
        if null_values is not None:
            null_map = dict(zip(null_values, [None]*len(null_values)))
            dat = (
                    dat
                    .with_columns(
                        pl.col(col_).replace_strict(
                            null_map,
                            default = pl.col(col_),
                            )
                        )
                    )

        if col_ in column_map.keys():
            print(f"    {col_} to {column_map[col_]}")
            dat = (
                    dat
                    .with_columns(pl.col(col_).cast(column_map[col_], strict=True))
                    )
        else:
            test_col = (
                    dat
                    .select(pl.col(col_))
                    .filter(pl.col(col_).is_not_null())
                    .collect()
                    .get_column(col_)
                    .to_list()
                    )

            test_float = is_float(test_col)
            test_numeric = is_int(test_col, check_is_float=False) #already checked for int

            #if all values numeric, but at least one of float type
            if sum(test_numeric)==len(test_col) and sum(test_float)>0:
                print(f"    {col_} to flt64")
                dat = dat.with_columns(pl.col(col_).cast(pl.Float64, strict=True))

            else:
                #if numeric and no float values
                if sum(test_numeric)==len(test_col):
                    print(f"    {col_} to int64")
                    dat = dat.with_columns(
                            pl.col(col_).str.extract(
                                r"([^.]*)", 1
                                ).cast(pl.Int64, strict=True))
                else:
                    print(f"    {col_} to str")
                    dat = dat.with_columns(pl.col(col_).cast(pl.Utf8, strict=True))

    return dat
