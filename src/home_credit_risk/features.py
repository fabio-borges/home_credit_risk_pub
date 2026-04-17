import re
from typing import Callable

import pandas as pd
from pandas.core.groupby import DataFrameGroupBy

import home_credit_risk.data_util as data_util


def merge_features(base_df, features_df, default_value = None):
    base_df = base_df.merge(features_df, how='left', on=data_util.APP_ID_COL)
    if default_value is not None:
        mapping = {c: default_value for c in features_df.columns.tolist()}
        base_df.fillna(mapping, inplace=True)
    return base_df


class FeatureNamer:
    """
    This class builds generated feature names. As in the following patterns:

    prefix__cat_col__cat_col_value
    prefix__num_col__op
    prefix__num_col__op__cat_col__cat_col_value

    Where:

    prefix: either the initials of the table the feature comes from, ot that plus __t, where t is a time boundary.
    cat_col: categorical column
    cat_col_value: a possible value for the categorical column
    num_col: a numerical column
    op: one of {min, mean, median, max}

    Example:
        CCB_12_BALANCE_mean_STATUS_Active: the mean of the BALANCE numerical column of credit_card_balance table, with
        STATUS="Active" for the last 12 months.

    The categorical values are shortened and the column names are abbreviated according to the given dataframe of
    column abbreviations.
    """

    SEPARATOR = "__"

    def __init__(self,
                 table_name: str,
                 column_abbrevs_df: pd.DataFrame,
                 time_boundary: int = None):
        """
        Constructor.
        :param table_name: The table name
        :param column_abbrevs_df: A dataframe with columns [Table,Row,Abbreviation] for column name abbreviations.
        :param time_boundary: An arbitrary integer to differentiate same versions of the feature for different time frames.
        """
        self.table_name = table_name
        self.column_abbrevs_df = column_abbrevs_df
        self.prefix = "".join([c[0].upper() for c in table_name.split("_")])
        if time_boundary is not None:
            self.prefix = self.join(self.prefix, str(abs(time_boundary)))

    def abbreviate(self, column):
        return self.column_abbrevs_df.query("Table == @self.table_name and Row == @column")["Abbreviation"].item()

    @staticmethod
    def shorten_category_name(name):
        max_length = 10
        if type(name) != str:
            name = str(name)
        if len(name) < max_length:
            return name.replace(" ", "_")
        splits = re.split(r'[\s_\-():/+,]', name)
        if len(splits) < 2:
            return name[:max_length]
        stop_words = ["for", "of"]
        splits = filter(lambda x: len(x) > 0 and x.lower() not in stop_words, splits)

        return "_".join([s[:4] if s.lower() != "without" else "wout" for s in splits])

    def name_categories(self, categorical_col, categories) -> dict[str, str]:
        abbrev_column = self.abbreviate(categorical_col)
        names = [self.join(abbrev_column, self.shorten_category_name(c)) for c in categories]
        assert len(set(names)) == len(names), f"Short name duplication: {names}"
        return {c: n for c, n in zip(categories, names)}

    def name_numeric_col_aggregate(self, column: str, operation: str) -> str:
        abbrev_column = self.abbreviate(column)
        return self.build_name(abbrev_column, operation)

    def build_name(self, *name_components):
        return self.join(self.prefix, *name_components)

    @staticmethod
    def join(*name_components):
        return FeatureNamer.SEPARATOR.join(name_components)


class FeatureEliminator:
    """
    Eliminates Features based on missingness and variance. These are calculated over a given dataframe, after
    being merged into an applications dataframe containing at least id and target. That's because
    the missingness and variance are calculated per target, for a given set of applications (the training set).
    """

    def __init__(self,
                 applications_df: pd.DataFrame,
                 positive_class_var_threshold: float = 1e-4,
                 positive_class_missing_threshold: float = 0.99,
                 negative_class_var_threshold: float = 1e-4,
                 negative_class_missing_threshold: float = 0.99,
             ):
        """
        Constructor.
        :param applications_df: A dataframe of applications containing at least id and target.
        :param positive_class_var_threshold:
        :param positive_class_missing_threshold:
        :param negative_class_var_threshold:
        :param negative_class_missing_threshold:
        """
        self.applications_df = applications_df
        self.positive_class_var_threshold = positive_class_var_threshold
        self.positive_class_missing_threshold = positive_class_missing_threshold
        self.negative_class_var_threshold = negative_class_var_threshold
        self.negative_class_missing_threshold = negative_class_missing_threshold

    def eliminate_features(self, features_df: pd.DataFrame):
        print("Checking for features to eliminate...")
        merged_df = merge_features(self.applications_df, features_df)
        is_positive = merged_df[data_util.TARGET_COL] == 1
        positives_df = merged_df[is_positive]
        negatives_df = merged_df[~is_positive]
        weak_features = []
        for col_name in features_df.columns.tolist():
            positives = positives_df[col_name]
            if positives.dtype == 'category':
                continue
            negatives = negatives_df[col_name]
            if (self.has_high_missingness(negatives, positives)
                    or self.has_low_variance(negatives, positives)):
                weak_features.append(col_name)
                continue
        if len(weak_features) > 0:
            print(f"Eliminating {len(weak_features)}/{len(features_df.columns)} features...")
            features_df.drop(columns=weak_features, inplace=True)

    def has_high_missingness(self, negatives, positives) -> bool:
        return (positives.isnull().mean() >= self.positive_class_missing_threshold
                and negatives.isnull().mean() >= self.negative_class_missing_threshold)

    def has_low_variance(self, negatives, positives) -> bool:
        return (positives.var() <= self.positive_class_var_threshold
                and negatives.var() <= self.negative_class_var_threshold)


class DataAggregator:
    """
    Generates multiple features in the form of aggregations for a given historical table.
    There are 3 types of aggregations:
    1. Category counts: Done for each categorical column, and each category value on that column.
    2. Numerical column Aggregations: for each numerical column, and for each operation in (min, median, mean, max).
    3. Cross aggregations: Optional. If required, also generate numerical column aggregations over categorical groups.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 feature_namer: FeatureNamer,
                 feature_eliminator: FeatureEliminator,
                 cross_aggregate_all: bool = True,
                 cross_aggregate_categories: dict[str, list[str]] = None):
        """
        Constructor.
        :param data: The data frame holding the historical table data.
        :param feature_namer: This object generates the features names, which will be the columns on the aggregated dataframe.
        :param feature_eliminator: Excludes features.
        :param cross_aggregate_all: If all possible categorical groups should be used to generate cross-aggregated features.
        :param cross_aggregate_categories: A dictionary of the form {categorical_column:[category values]}, indicating
        which categories to use on the cross-aggregations, if cross_aggregate_all is False. Optional.
        """
        self.data = data
        self.cross_aggregate_all = cross_aggregate_all
        numeric_cols = data_util.get_numeric_cols(data)
        self.numeric_column_aggregations = {
            feature_namer.name_numeric_col_aggregate(col, op): (col, op)
            for col in numeric_cols for op in ['min', 'median', 'mean', 'max']}
        self.category_map = {
            col: feature_namer.name_categories(col, data[col].cat.categories.tolist())
            for col in data_util.get_categorical_cols(data)
        }
        self.feature_namer = feature_namer
        self.feature_eliminator = feature_eliminator
        if cross_aggregate_categories is None:
            cross_aggregate_categories = {}
        self.cross_aggregate_categories = cross_aggregate_categories

    def aggregate(self) -> pd.DataFrame:
        """
        Aggregates the dataframe passed into the constructor, returning a new dataframe containing the aggregated data.
        :return: A dataframe of features generated through data aggregations.
        """
        result_df = None
        for cat_col in self.category_map.keys():
            grouped_data = self.data.groupby([data_util.APP_ID_COL, cat_col], observed=False)
            
            counts_df = self._count_categories(grouped_data, cat_col)
            if result_df is not None:
                result_df = merge_features(result_df, counts_df, default_value=0)
            else:
                result_df = counts_df

            if self.cross_aggregate_all:
                result_df = self._add_numeric_col_aggregates(grouped_data, result_df, cat_col)
            elif cat_col in self.cross_aggregate_categories.keys():
                tmp_data = self.data[self.data[cat_col].isin(self.cross_aggregate_categories[cat_col])]
                grouped_data = tmp_data.groupby([data_util.APP_ID_COL, cat_col], observed=True)
                result_df = self._add_numeric_col_aggregates(grouped_data, result_df, cat_col)

        grouped_data = self.data.groupby([data_util.APP_ID_COL], observed=False)
        numeric_cols_agg_df = self._aggregate_numeric_columns(grouped_data)
        result_df = merge_features(result_df, numeric_cols_agg_df)

        return result_df

    def _add_numeric_col_aggregates(self,
                                    grouped_data: DataFrameGroupBy,
                                    base_data: pd.DataFrame,
                                    categorical_col: str = None):
        numeric_cols_agg_df = self._aggregate_numeric_columns(grouped_data, categorical_col)
        return merge_features(base_data, numeric_cols_agg_df)

    def _count_categories(self,
                          grouped_data: DataFrameGroupBy,
                          categorical_col: str) -> pd.DataFrame:
        count_data = (grouped_data
            .agg(count=(categorical_col, 'count'))
            .reset_index()
            .pivot(index=data_util.APP_ID_COL,
                   columns=categorical_col,
                   values="count")
        )
        cat_rename_map = self.category_map[categorical_col]
        feature_names = [self.feature_namer.build_name(cat_rename_map[v]) for v in count_data.columns.values]
        count_data.columns = feature_names
        self.feature_eliminator.eliminate_features(count_data)
        return count_data

    def _aggregate_numeric_columns(self,
                                   grouped_data: DataFrameGroupBy,
                                   categorical_col: str = None) -> pd.DataFrame:
        aggregate_df = grouped_data.agg(**self.numeric_column_aggregations)
        if categorical_col is not None:
            value_columns = [k for k in self.numeric_column_aggregations.keys()]
            aggregate_df = (aggregate_df
                .reset_index()
                .pivot(index=data_util.APP_ID_COL,
                       columns=categorical_col,
                       values=value_columns))
            cat_rename_map = self.category_map[categorical_col]
            # Since there are multiple value columns, the pivoted table will have a
            # multilevel index, with each column being the tuple (value column, category)
            feature_names = [self.feature_namer.join(c[0], cat_rename_map[c[1]])
                             for c in aggregate_df.columns.values]
            aggregate_df.columns = feature_names
        self.feature_eliminator.eliminate_features(aggregate_df)
        return aggregate_df


class FeatureGenerator:
    """
    Generates features for a given historical table.
    It loads the application IDs upon construction into 2 dataframes, on for the training set the other for the test set.
    Then generates and merges the generated features into those dataframes on each call to the generate() method.
    When all desired tables are processed, the save_generated_features() method may be called to save the generated
    features to disk, in 1 file for each data split (train/test).
    """

    def __init__(self, data_dir: str, column_abbrev_file_path: str):
        """
        Initializes the object by loading 2 dataframes containing the application IDs of the training and test sets
        separately. The training set dataframe also contains the targets.
        :param data_dir: The directory where the csv files of the tables are stored.
        :param column_abbrev_file_path: Path to the csv file of column abbreviations to be used when naming features.
        """
        self.data_dir = data_dir
        train_ids_targets_df = pd.read_csv(f"{data_dir}/application_train.csv",
                                           usecols=[data_util.APP_ID_COL, data_util.TARGET_COL])
        self.feature_eliminator = FeatureEliminator(train_ids_targets_df)
        self.features_df_train = train_ids_targets_df.drop(columns=[data_util.TARGET_COL])
        self.features_df_test = pd.read_csv(f"{data_dir}/application_test.csv",
                                            usecols=[data_util.APP_ID_COL])
        self.column_abbrevs_df = pd.read_csv(column_abbrev_file_path)

    def generate_features(self,
                          table_name: str,
                          cross_aggregate_all: bool,
                          time_boundary_column: str = None,
                          time_boundaries: tuple[int, ...] = (),
                          cross_aggregate_categories: dict[str, list[str]] = None,
                          preprocessor: Callable[[pd.DataFrame], pd.DataFrame] = None,
                          postprocessor: Callable[[pd.DataFrame, str], None] = None,
            ):
        """
        Generates features for a given table. The features are generated using a DataAggregator. New features may be
        added through an optional postprocessor, and a preprocessor may be used to transform the original data before
        aggregating. The historical tables contain data for both training and test. After the data is aggregated, it's
        merged into the training and test dataframes loaded in the constructor. So those dataframes grow incrementally
        on each call to this method.
        :param table_name: The table's name.
        :param cross_aggregate_all: Passed to the DataAggregator.
        :param time_boundary_column: Column to be used to place the record on the historical timeline. Optional.
        :param time_boundaries: Tuple of values to be used in conjunction with time_boundary_column. Optional.
        :param cross_aggregate_categories: Passed to the DataAggregator.
        :param preprocessor: A Callable acting on the original dataframe before the data aggregation. Optional.
        :param postprocessor: A Callable acting on aggregated data. Optional.
        :return: Nothing.
        """
        print(f"\nGenerating features for {table_name}...")

        data_file = f"{self.data_dir}/{table_name}.csv"
        print(f"Loading data from {data_file}...")
        data = data_util.load_table(data_file)
        if preprocessor is not None:
            print(f"Preprocessing...")
            data = preprocessor(data)

        for time_boundary in [None, *time_boundaries]:
            filtered_data = data
            if time_boundary is not None:
                print(f"Aggregating data for {time_boundary_column} >= {time_boundary}")
                filtered_data = data[data[time_boundary_column] >= time_boundary]
            else:
                print(f"Aggregating full table")
            feature_namer = FeatureNamer(table_name, self.column_abbrevs_df, time_boundary)
            aggregator = DataAggregator(filtered_data, feature_namer, self.feature_eliminator,
                                        cross_aggregate_all, cross_aggregate_categories)
            aggregated_data = aggregator.aggregate()
            if postprocessor is not None:
                print(f"Postprocessing...")
                postprocessor(aggregated_data, feature_namer.prefix)
            print(f"{table_name}: {aggregated_data.shape[1]} features generated. Merging...")
            self.features_df_train = merge_features(self.features_df_train, aggregated_data)
            self.features_df_test = merge_features(self.features_df_test, aggregated_data)

    def save_generated_features(self, base_file_name: str):
        """
        Generates 3 files:
        1. "{self.data_dir}/{base_file_name}_info.txt": List of feature names.
        2. "{self.data_dir}/{base_file_name}_train.parquet": Dataset with the generated features for the training set,
            plus the application ID and target.
        3. "{self.data_dir}/{base_file_name}_test.parquet": Dataset with the generated features for the test set,
            plus the application ID.
        :param base_file_name: The base file name.
        :return: Nothing
        """
        print(f"Saving generated features, dataset shapes: train={self.features_df_train.shape}, test={self.features_df_test.shape}")
        features = [c for c in self.features_df_train.columns.values
                    if c not in [data_util.APP_ID_COL, data_util.TARGET_COL]]
        # Write the list of features to {base_file_path}_info.txt
        files = (
            f"{self.data_dir}/{base_file_name}_info.txt",
            f"{self.data_dir}/{base_file_name}_train.parquet",
            f"{self.data_dir}/{base_file_name}_test.parquet"
        )
        with open(files[0], "w") as f:
            f.write("\n".join(features))
        self.features_df_train.to_parquet(files[1])
        self.features_df_test.to_parquet(files[2])
        print("Files saved:\n" + "\n".join(files))