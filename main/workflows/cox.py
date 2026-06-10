from typing import Optional, Union
from pathlib import PurePath, Path

import numpy as np
from pandas import DataFrame as pdDataFrame
from lifelines import CoxPHFitter
from statsmodels.api import nonparametric
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.compose import ColumnTransformer
from plotly.graph_objects import Figure

try:
    from dataScienceLibrary.main.graphing.graphing import Visualisation
except ImportError:
    Visualisation = None


class Cox:
    """
    A class for performing various data transformations, model fitting, and
    diagnostic checks.

    Methods:
    --------
    get_dat(dat: pd.DataFrame) -> pd.DataFrame:
        Transforms the input DataFrame using the column transformer if it is not
        None.

    fit(dat: pd.DataFrame, coxmodel_fitpars: dict) -> None:
        Fits the model using the provided DataFrame and fitting parameters.

    checkoutdir(outdir: Union[str, PurePath, Path]) -> PurePath:
        Ensures the output directory exists, creating it if necessary, and
        returns the directory as a PurePath.

    fit_lowess(dat: pd.DataFrame, y: str, x: str, smooth: Optional[float],
    interval_sd: float = 1.0) -> pd.DataFrame:
        Fits a LOWESS (Locally Weighted Scatterplot Smoothing) model to the data
        and calculates confidence intervals.

    plot_lowess(dat: pd.DataFrame, x_var: str, y_var: str, figure: Figure,
    lowess_smooth: Optional[float]) -> Figure:
        Plots the LOWESS model and its confidence intervals on the given figure.

    plot_baseline(dat: pd.DataFrame, x_var: str, figure: Figure) -> Figure:
        Plots a baseline (zero) line on the given figure.

    check_proportionalhazards(dat: pd.DataFrame, lowess_smooth: Optional[float]
    = None, outdir: Union[str, PurePath, Path] = "checks/prop_hazards") -> None:
        Checks the proportional hazards assumption by plotting Schoenfeld
        residuals and their LOWESS smoothed values.

    check_multicolinearity(dat: pd.DataFrame, vars_to_eval: Optional[list[str]]
    = None, outdir: Union[str, PurePath, Path] = "checks/multicolinearity") ->
    None:
        Checks for multicollinearity among the specified variables by
        calculating the Variance Inflation Factor (VIF) and saves the results
        to a CSV file.

    check_continuouslinearity(dat: pd.DataFrame, lowess_smooth: Optional[float]
    = None, outdir: Union[str, PurePath, Path] = "checks/continuous_linearity")
    -> None:
        Checks the linearity of continuous variables by plotting deviance
        residuals and their LOWESS smoothed values.

    check_influentialobservations(dat: pd.DataFrame, outdir: Union[str,
    PurePath, Path] = "checks/influential_observations") -> None:
        Checks for influential observations by plotting delta-beta residuals
        and their limits.

    report() -> tuple[list[str], dict]:
        Generates a report summarizing the model's performance and significant
        predictors.
    """
    def __init__(self,
                 coxmodel_initpars: dict,
                 column_transformer: Optional[ColumnTransformer] = None,
                 graphing: Optional[Visualisation] = None,
                 ) -> None:
        """
        Initializes the class with optional parameters for data transformation,
        model fitting, and LOWESS smoothing.

        Parameters:
        column_transformer (Optional[ColumnTransformer]): An optional column
        transformer for data preprocessing. Default is None.
        model (Optional[Model]): An optional model for fitting and diagnostics.
        Default is None.
        lowess_smoothing_default (float): The default smoothing parameter for
        the LOWESS model. Default is 0.25.
        lowess_smoothing_values (Optional[list[float]]): A list of smoothing
        values for the LOWESS model. Default is None.
        duration_col (str): The name of the column representing the duration.
        Default is "duration".
        event_col (str): The name of the column representing the event.
        Default is "event".
        """
        self.model = CoxPHFitter(**coxmodel_initpars)
        self.summary: Optional[pdDataFrame] = None

        if graphing is None and Visualisation is not None:
            self.graphing = Visualisation()
            self.graphing.dict_lay.line_width = 4
            self.graphing.dict_lay.marker_opacity = 0.9
            self.graphing.dict_lay.line_opacity = 1
            self.graphing.dict_lay.legend_title_text = "Smooth Factor:"
            self.graphing.legends["Overall"] = {"Overall": "#7DD612",}
            self.graphing.legends_dash["Overall"] = {"Overall": "solid",}
        else:
            self.graphing = graphing

        self.column_transformer = column_transformer

        self.duration_col: Optional[str] = None
        self.event_col: Optional[str] = None

        self.lowess_smoothing_default = 0.66
        self.lowess_smoothing_values: Optional[list[Union[float, None]]] = [0.3,
                                                                            0.6,
                                                                            0.9]


    def get_dat(self, dat: pdDataFrame,) -> pdDataFrame:
        """
        Transforms the input DataFrame using the column transformer if it is not
        None.

        Parameters:
        dat (pd.DataFrame): The input DataFrame to be transformed.

        Returns:
        pd.DataFrame: The transformed DataFrame.
        """
        if self.column_transformer is not None:
            dat = self.column_transformer.transform(dat)
        return dat


    def fit(self,
            dat: pdDataFrame,
            coxmodel_fitpars: dict) -> None:
        """
        Fits the model using the provided DataFrame and fitting parameters.

        Parameters:
        dat (pd.DataFrame): The input DataFrame to be transformed and used for
        fitting.
        coxmodel_fitpars (dict): A dictionary containing the fitting parameters,
        including:
            - "duration_col" (str): The name of the column representing the
            duration.
            - "event_col" (str): The name of the column representing the event.

        Returns:
        None
        """
        dat = self.get_dat(dat)

        self.duration_col = coxmodel_fitpars["duration_col"]
        self.event_col = coxmodel_fitpars["event_col"]

        self.model.fit(df = dat, **coxmodel_fitpars)


    def checkoutdir(self, outdir: Union[str, PurePath, Path],) -> PurePath:
        """
        Ensures the output directory exists, creating it if necessary, and
        returns the directory as a PurePath.

        Parameters:
        outdir (Union[str, PurePath, Path]): The output directory path. Can be
        a string, PurePath, or Path object.

        Returns:
        PurePath: The output directory as a PurePath object.
        """
        outdir_ = Path(outdir)
        if not outdir_.exists():
            outdir_.mkdir(parents=True)

        return PurePath(outdir)


    def fit_lowess(self,
                   dat: pdDataFrame,
                   y: str,
                   x: str,
                   smooth: Optional[float],
                   interval_sd: float = 1.0,
                   ) -> pdDataFrame:
        """
        Fits a LOWESS (Locally Weighted Scatterplot Smoothing) model to the data
        and calculates confidence intervals.

        Parameters:
        dat (pd.DataFrame): The input DataFrame containing the data.
        y (str): The name of the dependent variable column.
        x (str): The name of the independent variable column.
        smooth (Optional[float]): The smoothing parameter for the LOWESS model.
        If None, a default value is used.
        interval_sd (float, optional): The number of standard deviations for the
        confidence interval. Default is 1.0.

        Returns:
        pd.DataFrame: A DataFrame containing the independent variable, the
        smoothed values, and the upper and lower confidence intervals.
        """
        if smooth is None:
            smooth = self.lowess_smoothing_default

        lowess_ = nonparametric.lowess(
                dat[y],
                dat[x],
                frac=smooth,
                return_sorted=False,
                )

        residuals = dat[y] - np.interp(dat[x], dat[x], lowess_)
        std_dev = np.std(residuals)

        return pdDataFrame({
            x: dat[x],
            "y_lowess": lowess_,
            "interval_upper_lowes": lowess_+(interval_sd * std_dev),
            "interval_lower_lowes": lowess_-(interval_sd * std_dev),
            })


    def plot_lowess(
            self,
            dat: pdDataFrame,
            x_var: str,
            y_var: str,
            figure: Figure,
            lowess_smooth: Optional[float],
            ) -> Figure:
        """
        Plots the LOWESS (Locally Weighted Scatterplot Smoothing) model and its
        confidence intervals on the given figure.

        Parameters:
        dat (pd.DataFrame): The input DataFrame containing the data.
        x_var (str): The name of the independent variable column.
        y_var (str): The name of the dependent variable column.
        figure (Figure): The figure object to plot on.
        lowess_smooth (Optional[float]): The smoothing parameter for the LOWESS
        model. If None, a default value is used.

        Returns:
        Figure: The figure object with the LOWESS plot and confidence intervals
        added.
        """
        if self.lowess_smoothing_values is not None and lowess_smooth is None:
            smooth_vals_ = self.lowess_smoothing_values
        else:
            smooth_vals_ = [lowess_smooth]

        for smooth_val_ in smooth_vals_:
            lowess_ = self.fit_lowess(dat,
                                      y=y_var,
                                      x=x_var,
                                      smooth=smooth_val_,)
            dat["lowess"] = lowess_["y_lowess"]
            dat["lowess_cname"] = [smooth_val_]*dat.shape[0]
            dat["interval_lower_lowes"] = lowess_["interval_lower_lowes"]
            dat["interval_upper_lowes"] = lowess_["interval_upper_lowes"]

            traces, _ = self.graphing.plot_scatter(
                    dat,
                    y_var = "lowess",
                    x_var = x_var,
                    c_name = "lowess_cname",
                    out_type = "trace",
                    withLine = True,
                    withPoints = False,
                    legend = True,
                    useObjectColours = True,
                    )
            for t_ in traces:
                figure.add_traces(t_)

            traces, _ = self.graphing.plot_scatter(
                    dat,
                    y_var = "interval_lower_lowes",
                    x_var = x_var,
                    c_name = "lowess_cname",
                    out_type = "trace",
                    withLine = True,
                    withPoints = False,
                    legend = False,
                    useObjectColours = True,
                    overrideDashType = "dash",
                    )
            for t_ in traces:
                figure.add_traces(t_)

            traces, _ = self.graphing.plot_scatter(
                    dat,
                    y_var = "interval_upper_lowes",
                    x_var = x_var,
                    c_name = "lowess_cname",
                    out_type = "trace",
                    withLine = True,
                    withPoints = False,
                    legend = False,
                    useObjectColours = True,
                    overrideDashType = "dash",
                    )
            for t_ in traces:
                figure.add_traces(t_)

        return figure


    def plot_baseline(
            self,
            dat: pdDataFrame,
            x_var: str,
            figure: Figure,
            ) -> Figure:
        """
        Plots a baseline (zero) line on the given figure.

        Parameters:
        dat (pd.DataFrame): The input DataFrame containing the data.
        x_var (str): The name of the independent variable column.
        figure (Figure): The figure object to plot on.

        Returns:
        Figure: The figure object with the baseline plot added.
        """
        dat["baseline"] = [0] * dat.shape[0]

        traces, _ = self.graphing.plot_scatter(
                dat,
                y_var = "baseline",
                x_var = x_var,
                out_type = "trace",
                withLine = True,
                withPoints = False,
                legend = False,
                overrideColour="black",
                )
        for t_ in traces:
            figure.add_traces(t_)

        return figure


    def check_proportionalhazards(
            self,
            dat: pdDataFrame,
            lowess_smooth: Optional[float] = None,
            outdir: Union[str, PurePath, Path] = "checks/prop_hazards",
            ):
        """
        Checks the proportional hazards assumption by plotting Schoenfeld
        residuals and their LOWESS smoothed values.

        Parameters:
        dat (pd.DataFrame): The input DataFrame to be transformed and used for
        residual computation.
        lowess_smooth (Optional[float]): The smoothing parameter for the LOWESS
        model. If None, a default value is used.
        outdir (Union[str, PurePath, Path], optional): The output directory
        where the plots will be saved. Default is "checks/prop_hazards".

        Returns:
        None
        """
        dat = self.get_dat(dat)

        outdir_: PurePath = self.checkoutdir(outdir)

        resid = self.model.compute_residuals(dat, "scaled_schoenfeld")
        resid["id"] = resid.index

        for var_ in [x for x in resid.columns if x!="id"]:
            figure = self.graphing.plot_scatter(
                    resid,
                    y_var=var_,
                    x_var="id",
                    meta_vars=[var_, "id"],
                    interactive=True,
                    toDisk=False,
                    out_type="interactive",
                    legend=False,
                    useObjectColours = True,
                    )

            figure = self.plot_lowess(
                    resid,
                    x_var = "id",
                    y_var = var_,
                    figure = figure,
                    lowess_smooth = lowess_smooth,
                    )

            figure = self.plot_baseline(
                    resid,
                    "id",
                    figure,
                    )

            dpi = self.graphing.dict_lay.dpi
            figure.write_image(
                    str(outdir_ / f"{var_}.html"),
                    engine="kaleido",
                    width=self.graphing.dict_lay.width*dpi,
                    height=self.graphing.dict_lay.height*dpi,
                    scale=1,
                    )


    def check_multicolinearity(
            self,
            dat: pdDataFrame,
            vars_to_eval: Optional[list[str]] = None,
            outdir: Union[str, PurePath, Path] = "checks/multicolinearity",
            ):
        """
        Checks for multicollinearity among the specified variables by
        calculating the Variance Inflation Factor (VIF) and saves the results
        to a CSV file.

        Parameters:
        dat (pd.DataFrame): The input DataFrame to be transformed and used for
        VIF calculation.
        vars_to_eval (Optional[list[str]]): A list of variable names to evaluate
        for multicolinearity. If None, all variables except the duration and
        event columns are evaluated.
        outdir (Union[str, PurePath, Path], optional): The output directory
        where the VIF results will be saved. Default is
        "checks/multicolinearity".

        Returns:
        None
        """
        dat = self.get_dat(dat)

        outdir_: PurePath = self.checkoutdir(outdir)

        if vars_to_eval is None:
            vars_to_eval = [x for x in dat.columns if x not in [
                self.duration_col,
                self.event_col,
                ]]

        dat = dat[vars_to_eval]

        # calculating VIF for each feature
        # https://www.geeksforgeeks.org/detecting-multicollinearity-with-vif-python/
        vif_data = pdDataFrame()
        vif_data["feature"] = dat.columns
        vif_data["VIF"] = [variance_inflation_factor(dat.values, i)
                                  for i in range(len(dat.columns))]

        vif_data.to_csv(outdir_ / "vif.csv")


    def check_continuouslinearity(
            self,
            dat: pdDataFrame,
            lowess_smooth: Optional[float] = None,
            outdir: Union[str, PurePath, Path] = "checks/continuous_linearity",
            ):
        """
        Checks the linearity of continuous variables by plotting deviance
        residuals and their LOWESS smoothed values.

        Parameters:
        dat (pd.DataFrame): The input DataFrame to be transformed and used for
        residual computation.
        lowess_smooth (Optional[float]): The smoothing parameter for the LOWESS
        model. If None, a default value is used.
        outdir (Union[str, PurePath, Path], optional): The output directory
        where the plots will be saved. Default is "checks/continuous_linearity".

        Returns:
        None
        """
        dat = self.get_dat(dat)

        outdir_: PurePath = self.checkoutdir(outdir)

        resid = self.model.compute_residuals(dat, "deviance")
        resid = dat[[x for x in dat.columns if x not in [
            self.duration_col,
            self.event_col,
            ]]].join(resid)

        for var_ in [x for x in resid.columns if x not in [
            self.duration_col,
            self.event_col,
            "deviance",
            ]]:
            resid_ = resid.sort_values(by=var_, axis=0,)

            figure = self.graphing.plot_scatter(
                    resid_,
                    y_var="deviance",
                    x_var=var_,
                    meta_vars=["deviance", var_],
                    interactive=True,
                    toDisk=False,
                    out_type="interactive",
                    legend=False,
                    useObjectColours = True,
                    )

            figure = self.plot_lowess(
                    resid_,
                    x_var = var_,
                    y_var = "deviance",
                    figure = figure,
                    lowess_smooth = lowess_smooth,
                    )

            figure = self.plot_baseline(
                    resid_,
                    var_,
                    figure,
                    )

            dpi = self.graphing.dict_lay.dpi
            figure.write_image(
                    str(outdir_ / f"{var_}.html"),
                    engine="kaleido",
                    width=self.graphing.dict_lay.width*dpi,
                    height=self.graphing.dict_lay.height*dpi,
                    scale=1,
                    )


    def check_influentialobservations(
            self,
            dat: pdDataFrame,
            outdir: Union[str, PurePath, Path] = "checks/influential_observations",
            ):
        """
        Checks for influential observations by plotting delta-beta residuals and
        their limits.

        Parameters:
        dat (pd.DataFrame): The input DataFrame to be transformed and used for
        residual computation.
        outdir (Union[str, PurePath, Path], optional): The output directory
        where the plots will be saved. Default is
        "checks/influential_observations".

        Returns:
        None
        """
        dat = self.get_dat(dat)

        outdir_: PurePath = self.checkoutdir(outdir)

        resid = self.model.compute_residuals(dat, "delta_beta")
        resid["id"] = resid.index

        limits_bound = 2/np.sqrt(resid.shape[0])
        resid["up_limit"] = [float(limits_bound)]*resid.shape[0]
        resid["lower_limit"] = [-float(limits_bound)]*resid.shape[0]

        for var_ in [x for x in resid.columns if x not in ["id",
                                                           "up_limit",
                                                           "lower_limit",]]:
            figure = self.graphing.plot_scatter(
                    resid,
                    y_var=var_,
                    x_var="id",
                    meta_vars=[var_, "id"],
                    interactive=True,
                    toDisk=False,
                    out_type="interactive",
                    legend=False,
                    useObjectColours = True,
                    )

            traces, _ = self.graphing.plot_scatter(
                    resid,
                    y_var = "up_limit",
                    x_var = "id",
                    out_type = "trace",
                    withLine = True,
                    withPoints = False,
                    legend = False,
                    overrideColour="blue",
                    overrideDashType="dash",
                    )
            for t_ in traces:
                figure.add_traces(t_)

            traces, _ = self.graphing.plot_scatter(
                    resid,
                    y_var = "lower_limit",
                    x_var = "id",
                    out_type = "trace",
                    withLine = True,
                    withPoints = False,
                    legend = False,
                    overrideColour="blue",
                    overrideDashType="dash",
                    )
            for t_ in traces:
                figure.add_traces(t_)

            figure = self.plot_baseline(
                    resid,
                    "id",
                    figure,
                    )

            yaxis_limits = list(figure.layout.yaxis.range)
            if yaxis_limits[0] > -limits_bound:
                yaxis_limits[0] = -limits_bound
            if yaxis_limits[1] < limits_bound:
                yaxis_limits[1] = limits_bound
            figure.update_layout(yaxis_range=yaxis_limits)

            dpi = self.graphing.dict_lay.dpi
            figure.write_image(
                    str(outdir_ / f"{var_}.html"),
                    engine="kaleido",
                    width=self.graphing.dict_lay.width*dpi,
                    height=self.graphing.dict_lay.height*dpi,
                    scale=1,
                    )


    def report(self,) -> tuple[list[str], dict]:
        """
        Generates a report summarizing the model's performance and significant predictors.

        Returns:
        tuple[list[str], dict]: A tuple containing:
            - A list of strings summarizing the model's performance, including
            concordance, log-likelihood ratio test, and significant/
            insignificant predictors.
            - A dictionary with detailed statistics, including:
                - "concordance": Harrell's C-index.
                - "ll_ratio_p": p-value of the log-likelihood ratio test.
                - "sig_pred": A dictionary of significant predictors with their
                hazard ratios and p-values.
                - "insig_pred": A list of insignificant predictors.
        """
        self.summary = self.model.summary
        stats = {
                "concordance": self.model._model.concordance_index_,
                "ll_ratio_p": self.model.log_likelihood_ratio_test().p_value,
                "sig_pred": dict(zip(
                    list(self.summary.index[self.summary.p < 0.05]),
                    zip(*[
                        list(self.summary["exp(coef)"][self.summary.p < 0.05]),
                        list(self.summary.p[self.summary.p < 0.05]),
                        ]),
                    )),
                "insig_pred": list(
                    self.summary.index[self.summary.p>=0.05]
                    ),
                }

        # ll-test statement
        if stats["ll_ratio_p"] < 0.05:
            ll_sig = "significant"
        else:
            ll_sig = "insignificant"
        ll_statement = f"Log-likelihood ratio test was {ll_sig} (p = {round(stats['ll_ratio_p'], 2)})."

        # concordance statement
        # https://www.medcalc.org/manual/cox-regression.php
        conc_explan = "(predicting among 2 patients, which will present with event 1st, is "
        if stats["concordance"] < 0.5:
            conc_explan += "worse than random)."
        elif stats["concordance"] >= 0.5 and stats["concordance"] <=0.6:
            conc_explan += "as good as random (or not much better))."
        elif stats["concordance"] >=0.6 and stats["concordance"] <= 0.8:
            conc_explan += "better than random)."
        else:
            conc_explan += "much better than random)."

        conc_statement = " ".join([
                f"Concordance (Harrell's C-index) was {round(stats['concordance'], 2)}",
                conc_explan,
                ])

        # variable statements
        if len(stats["sig_pred"]) != 0:
            sig_pred_statement = f"The following variables were deemed significant (p < {self.model.alpha}):"
            for i, pred_ in enumerate(stats["sig_pred"].keys()):
                sig_pred_statement += f" ({i+1}) {pred_} (hr = {round(stats['sig_pred'][pred_][0], 2)}),"
            # replace last "," with "."
            sig_pred_statement = sig_pred_statement[:-1] + "."
        else:
            sig_pred_statement = "No variables were deemed significant."

        if len(stats["insig_pred"]) != 0:
            insig_pred_statement = "The following variables were deemed insignificant:"
            for i, pred_ in enumerate(stats["insig_pred"]):
                insig_pred_statement += f" ({i+1}) {pred_},"
            # replace last "," with "."
            insig_pred_statement = insig_pred_statement[:-1] + "."
        else:
            insig_pred_statement = "No variables were deemed insignificant."


        report: list[str] = [
                conc_statement,
                ll_statement,
                sig_pred_statement,
                insig_pred_statement,
                ]


        return report, stats

