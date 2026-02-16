from typing import Optional, Literal

import polars as pl
import numpy as np
from sklearn.compose import ColumnTransformer
from statsmodels import api as sm


class Match():
    """
    Note: ct is used to transform data after selecting columns
    """
    def __init__(self,
                 data: pl.LazyFrame,
                 vars_match: list[str],
                 var_exposure: str,
                 ct: Optional[ColumnTransformer] = None,
                 ):
        data = (
                data
                .select(
                    pl.col(vars_match + [var_exposure]
                           )
                    )
                )

        self.data_case: pl.LazyFrame = (
                data
                .filter(
                    pl.col(var_exposure) == 1,
                    )
                .select(
                    pl.all().exclude(var_exposure)
                    )
                .with_row_index(name="id")
                )

        self.data_control: pl.LazyFrame = (
                data
                .filter(
                    pl.col(var_exposure) == 0,
                    )
                .select(
                    pl.all().exclude(var_exposure)
                    )
                .with_row_index(name="id")
                )

        if ct is not None:
            self.data_case = ct.transform(self.data_case.collect()).lazy()
            self.data_control = ct.transform(self.data_control.collect()).lazy()

        self.vars_match = vars_match


    def nearest_neighbor(self,
                         order: Literal["largest",
                                        "smallest",
                                        "data",
                                        "closest",
                                        "farthest",
                                        "random",] = "largest",
                         ):
        ...


    def optimal_pair(self,):
        ...


    def full(self,):
        ...


    def exact(
            self,
            ratio: int = 1,
            ):
        subj_pairs: dict[
                str,
                list[str]
                ] = {str(subj):[] for subj in (self.data_case
                                          .select("id")
                                          .collect()
                                          .get_column("id")).unique()}
        subj_attr: dict[
                str,
                str,
                ] = {str(subj):"" for subj in (self.data_case
                                          .select("id")
                                          .collect()
                                          .get_column("id")).unique()}

        ## Define groups
        unpaired_controls = self.data_control
        paired_controls_ids: list[int] = []
        r_ = 0
        while r_ < ratio:
            # iterate over case subjects
            for i_ in subj_pairs.keys():
                subj_i: pl.DataFrame = (
                        self.data_case
                        .filter(pl.col("id")==int(i_))
                        .collect()
                        )

                if subj_attr[i_] == "":
                    subj_attr[i_] = (
                            subj_i
                            .select(
                                pl.col(self.vars_match).cast(pl.Utf8)
                                )
                            .select(
                                group=pl.concat_str(self.vars_match, separator=",",)
                                )
                            .get_column("group")
                            )[0]

                for col_ in self.vars_match:
                    match: pl.LazyFrame = (
                            unpaired_controls
                            .filter(
                                pl.col(col_) == subj_i.get_column(col_)[0]
                                )
                            )
                match_ids = (
                        match
                        .select(pl.col("id"))
                        .collect()
                        .get_column("id")
                        )
                if match_ids.shape[0] > 0:
                    paired_control_id = match_ids[0]
                    subj_pairs[str(i_)].append(str(paired_control_id))

                    paired_controls_ids.append(int(paired_control_id))
                    unpaired_controls = (
                            self.data_control
                            .filter(
                                ~pl.col("id").is_in(paired_controls_ids),
                                )
                            )

            r_+=1

        ## Concatenate groups with equal values
            # Ensure return original 1:n pairings, so can check for incomplete
            # matching
        match_sets: dict[
                str,
                list[list[str]]
                ] = {}
        for case_, controls_ in subj_pairs.items():
            group = subj_attr[case_]
            if group not in match_sets.keys():
                match_sets[group] = [[case_], controls_]
            else:
                match_sets[group][0].append(case_)
                match_sets[group][1] = match_sets[group][1] + controls_


        return match_sets, subj_pairs


class PropensityScores():
    """
    """
    def __init__(self,
                 data: pl.LazyFrame,
                 ) -> None:
        self.data = data

        self.model: Optional[sm.Logit] = None
        self.prop_scores: Optional[np.ndarray] = None
        self.vars_x: Optional[list[str]] = None
        self.var_y: Optional[str] = None


    def set_results(self,
                    model: Optional[sm.Logit],
                    prop_scores: Optional[np.ndarray],
                    vars_x: Optional[list[str]],
                    var_y: Optional[str],
                    ) -> None:
        self.model = model
        self.prop_scores = prop_scores
        self.vars_x = vars_x
        self.var_y = var_y


    def fit_scores_glm(self,
            y_col: str,
            x_cols: list[str],
            ) -> None:
        """
        """
        train_x = (
                self.data
                .select(
                    pl.col(x_cols)
                    )
                .collect()
                .to_numpy()
                )
        train_y = (
                self.data
                .select(
                    pl.col(y_col)
                    )
                .collect()
                .get_column(y_col)
                .to_numpy()
                )

        model=sm.Logit(train_y, train_x,).fit()
        prop_scores = model.predict(train_x)

        self.set_results(
                model = model,
                prop_scores = prop_scores,
                vars_x = x_cols,
                var_y = y_col,
                )

    def return_iptw(self,
                    estimand: Literal["ATT", "ATE"] = "ATE",
                    ) -> np.ndarray:
        """
        See 'How Matching Weights Are Computed' in below link:
        https://kosukeimai.github.io/MatchIt/reference/matchit.html
        """
        dat_ = pl.DataFrame({
            "prop_score": self.prop_scores,
            "y": self.data.select(pl.col(self.var_y)).collect().get_column(self.var_y).to_numpy(),
            })

        if estimand == "ATT":
            def estimand_func(prop, y):
                if y == 1:
                    iptw = 1
                else:
                    iptw = prop / (1-prop)

                return iptw

        elif estimand == "ATE":
            def estimand_func(prop, y):
                if y == 1:
                    iptw = 1 / prop
                else:
                    iptw = 1 / (1-prop)

                return iptw

        iptw = (
                dat_
                .fold(estimand_func)
                ).to_numpy()

        return iptw

