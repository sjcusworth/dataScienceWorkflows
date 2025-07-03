from os import remove
import unittest
import polars as pl
from scipy.stats import chi2_contingency, ttest_ind

from main.graphing.table1 import table1_polars


class Test_tb1(unittest.TestCase):
    def setUp(self,):
        self.FILE_TEMP_DATA = "test.parquet"

        self.data = pl.DataFrame({
            "a": ["a", "b", "a", "c", "a", "c", ""]*10,
            "b": ["b", "b", "b", "c", "", "a", "e"]*10,
            "c": ["a", "b", "c", "d", "e", "e", "e"]*10,
            "d": [1, None, 3, 1, 3, 4, 5]*10,
            "grp": ["a", "b", "a", "b", "a", "b", "b"]*10,
            })
        self.data.write_parquet(self.FILE_TEMP_DATA)

        self.catgs = ["a", "b", "c"]
        self.nums = ["d"]
        self.rnd = 2
        self.null_values = [""]
        self.grpCol = "grp"


    def check_single_catg(self,):
        dict_info: dict = {}
        for catg_ in self.catgs:
            dict_info[catg_] = {}
            for subCatg_ in [x for x in self.data.get_column(catg_).unique().to_list() if x not in self.null_values]:
                dict_info[catg_][subCatg_] = self.data.filter(
                        pl.col(catg_)==subCatg_
                        ).shape[0]

        with open("test.csv", "r") as file_test:
            file_test.readline() #skip header
            for line in file_test:
                line = line.split(",")
                if line[0] != "":
                    catg_ = line[0]
                if catg_ not in self.catgs:
                    continue
                subCatg_ = line[1]

                #Check count
                count_ = dict_info[catg_][subCatg_]
                self.assertEqual(line[3][:line[3].find(" ")], str(count_))

                #Check %
                n_total_ = sum([x for x in dict_info[catg_].values()])
                percent_ = line[3][line[3].find("(")+1:line[3].find("%")]
                self.assertEqual(percent_, str(round((count_/n_total_)*100, self.rnd)))


    def check_single_num(self,):
        dict_info: dict = {}
        for num_ in self.nums:
            dict_info[num_] = {}

            mean_ = round(self.data.get_column(num_).mean(), self.rnd)
            std_ = round(self.data.get_column(num_).std(), self.rnd)
            median_ = round(self.data.get_column(num_).median(), self.rnd)
            min_ = round(self.data.get_column(num_).min(), self.rnd)
            max_ = round(self.data.get_column(num_).max(), self.rnd)

            dict_info[num_]["mean std"] = f"{mean_} {std_}"
            dict_info[num_]["median min max"] = f"{median_} {min_} {max_}"

        with open("test.csv", "r") as file_test:
            file_test.readline() #skip header
            for line in file_test:
                line = line.rstrip()
                line = line.split(",")
                if line[0] != "":
                    num_ = line[0]
                if num_ not in self.nums:
                    continue
                subNum_ = line[1]
                self.assertEqual(line[3], dict_info[num_][subNum_])


    def check_single_missing(self,):
        dict_info = (
                self.data
                .select(
                    pl.col(self.catgs + self.nums)
                    )
                .with_columns(
                    pl.when(pl.col(pl.Utf8).is_in(self.null_values))
                    .then(None)
                    .otherwise(pl.col(pl.Utf8))
                    .name.keep()
                    )
                .null_count()
                .to_dict(as_series=False)
                )
        dict_info = {k:v[0] for k,v in dict_info.items()}
        total_n = self.data.shape[0]

        with open("test.csv", "r") as file_test:
            file_test.readline() #skip header
            for line in file_test:
                line = line.rstrip()
                line = line.split(",")
                if line[0] != "":
                    num_ = line[0]
                    # check count
                    self.assertEqual(line[2][:line[2].find(" ")],
                                     str(dict_info[num_]))
                    # check percent
                    self.assertEqual(
                            line[2][line[2].find("(")+1:line[2].find("%")],
                            str(round((dict_info[num_]/total_n)*100, self.rnd))
                            )


    def check_expCntrl_catg(self,):
        dict_info: dict = {}
        for group_ in self.data.get_column(self.grpCol).unique().to_list():
            dict_info[group_] = {}

            for catg_ in self.catgs:
                dict_info[group_][catg_] = {}
                for subCatg_ in [x for x in self.data.get_column(catg_).unique().to_list() if x not in self.null_values]:
                    dict_info[group_][catg_][subCatg_] = self.data.filter(
                            pl.col(catg_)==subCatg_,
                            pl.col(self.grpCol)==group_,
                            ).shape[0]

        with open("test.csv", "r") as file_test:
            header = file_test.readline() #skip header
            group_names = [x[x.find("_")+1:] for x in header.split(",") if x.startswith("Group")]
            file_test.readline() #2 headers in expCntrl

            for line in file_test:
                line = line.split(",")
                if line[0] != "":
                    catg_ = line[0]
                if catg_ not in self.catgs:
                    continue
                subCatg_ = line[1]

                for i, group_ in enumerate(group_names):
                    index_ = 3+i*2

                    #Check count
                    count_ = dict_info[group_][catg_][subCatg_]
                    self.assertEqual(line[index_][:line[index_].find(" ")], str(count_))

                    #Check %
                    n_total_ = sum([x for x in dict_info[group_][catg_].values()])
                    percent_ = line[index_][line[index_].find("(")+1:line[index_].find("%")]
                    self.assertEqual(percent_, str(round((count_/n_total_)*100, self.rnd)))


    def check_expCntrl_num(self,):
        dict_info: dict = {}
        for group_ in self.data.get_column(self.grpCol).unique().to_list():
            dict_info[group_] = {}

            for num_ in self.nums:
                dict_info[group_][num_] = {}
                dat_ = self.data.filter(
                        pl.col(self.grpCol)==group_
                        ).get_column(num_)

                mean_ = round(dat_.mean(), self.rnd)
                std_ = round(dat_.std(), self.rnd)
                median_ = round(dat_.median(), self.rnd)
                min_ = round(dat_.min(), self.rnd)
                max_ = round(dat_.max(), self.rnd)

                dict_info[group_][num_]["mean std"] = f"{mean_} {std_}"
                dict_info[group_][num_]["median min max"] = f"{median_} {min_} {max_}"

        with open("test.csv", "r") as file_test:
            header = file_test.readline() #skip header
            group_names = [x[x.find("_")+1:] for x in header.split(",") if x.startswith("Group")]
            file_test.readline() #2 headers in expCntrl

            for line in file_test:
                line = line.rstrip()
                line = line.split(",")
                if line[0] != "":
                    num_ = line[0]
                if num_ not in self.nums:
                    continue
                subNum_ = line[1]

                for i, group_ in enumerate(group_names):
                    index_ = 3+i*2
                    self.assertEqual(line[index_], dict_info[group_][num_][subNum_])


    def check_expCntrl_missing(self,):
        dict_info: dict = {}
        dict_n_total: dict = {}
        for group_ in self.data.get_column(self.grpCol).unique().to_list():
            dict_info[group_] = (
                    self.data
                    .filter(
                        pl.col(self.grpCol) == group_
                        )
                    .select(
                        pl.col(self.catgs + self.nums)
                        )
                    .with_columns(
                        pl.when(pl.col(pl.Utf8).is_in(self.null_values))
                        .then(None)
                        .otherwise(pl.col(pl.Utf8))
                        .name.keep()
                        )
                    .null_count()
                    .to_dict(as_series=False)
                    )
            dict_info[group_] = {k:v[0] for k,v in dict_info[group_].items()}
            dict_n_total[group_] = (
                    self.data
                    .filter(
                        pl.col(self.grpCol) == group_
                        )
                    .shape[0]
                    )

        with open("test.csv", "r") as file_test:
            header = file_test.readline() #skip header
            group_names = [x[x.find("_")+1:] for x in header.split(",") if x.startswith("Group")]
            file_test.readline() #2 headers in expCntrl

            for line in file_test:
                line = line.rstrip()
                line = line.split(",")
                if line[0] != "":
                    num_ = line[0]
                    for i, group_ in enumerate(group_names):
                        index_ = 2+i*2

                        # check count
                        self.assertEqual(line[index_][:line[index_].find(" ")],
                                         str(dict_info[group_][num_]))

                        # check percent
                        self.assertEqual(
                                line[index_][line[index_].find("(")+1:line[index_].find("%")],
                                str(round((dict_info[group_][num_]/dict_n_total[group_])*100, self.rnd))
                                )

    def check_expCntrl_pvalue(self,):
        dict_info = {}
        grps = self.data.get_column(self.grpCol).unique().to_list()

        for catg_ in self.catgs:
            dat_ = (
                    self.data
                    .select(pl.col(self.grpCol, catg_))
                    .with_columns(
                        pl.when(pl.col(catg_).is_in(self.null_values))
                        .then(None)
                        .otherwise(pl.col(catg_))
                        .name.keep()
                        )
                    .filter(pl.col(catg_).is_not_null())
                    .group_by(pl.col(self.grpCol, catg_))
                    .len()
                    .rename({
                        catg_: "subcatg",
                        "len": "count",
                        })
                    )
            dat_ = (
                    dat_
                    .filter(
                        pl.col(self.grpCol)==grps[0]
                        )
                    .select(pl.exclude(self.grpCol))
                    .join(
                        (
                            dat_
                            .filter(pl.col(self.grpCol)==grps[1])
                            .select(pl.exclude(self.grpCol))
                            ),
                        on=["subcatg"],
                        how="full", coalesce=True
                        )
                    .fill_null(strategy="zero")
                    .select(pl.exclude("subcatg"))
                    )
            dict_info[catg_] = chi2_contingency(dat_.to_numpy()).pvalue

        for num_ in self.nums:
            dat_ = (
                    self.data
                    .select(pl.col(self.grpCol, num_))
                    .filter(pl.col(num_).is_not_null())
                    )
            dat_1 = dat_.filter(pl.col(self.grpCol) == grps[0]).get_column(num_).to_list()
            dat_2 = dat_.filter(pl.col(self.grpCol) == grps[1]).get_column(num_).to_list()
            dict_info[num_] = ttest_ind(dat_1, dat_2).pvalue


        with open("test.csv", "r") as file_test:
            header = file_test.readline() #skip header
            group_names = [x[x.find("_")+1:] for x in header.split(",") if x.startswith("Group")]
            file_test.readline() #2 headers in expCntrl

            for line in file_test:
                line = line.rstrip()
                line = line.split(",")
                if line[0] != "":
                    num_ = line[0]
                    self.assertEqual(line[-1],
                                     str(round(dict_info[num_], self.rnd)))



    def test_single_group(self,):
        tb1 = table1_polars(
                fileDat=self.FILE_TEMP_DATA,
                catgs=self.catgs,
                nums=self.nums,
                null_values=self.null_values,
                delim=",",
                rnd=self.rnd,
                group_col=None,
                calc_sig=False,
                )
        tb1.write_csv("test.csv")

        self.check_single_catg()
        self.check_single_num()
        self.check_single_missing()

        remove("test.csv")


    def test_expCntrl(self,):
        tb1 = table1_polars(
                fileDat=self.FILE_TEMP_DATA,
                catgs=self.catgs,
                nums=self.nums,
                null_values=self.null_values,
                delim=",",
                rnd=self.rnd,
                group_col=self.grpCol,
                calc_sig=True,
                )
        tb1.write_csv("test.csv")

        self.check_expCntrl_catg()
        self.check_expCntrl_num()
        self.check_expCntrl_missing()
        self.check_expCntrl_pvalue()

        remove("test.csv")

    def tearDown(self,):
        remove(self.FILE_TEMP_DATA)


if __name__ == "__main__":
    ## NOTE: need to test for expCtrl but calc_sig=False
    unittest.main()
