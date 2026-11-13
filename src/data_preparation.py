# Standard library imports
import logging
from typing import Any, Dict

# Related third-party imports
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Canonical month names used to standardise the month columns
MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# Month name (lowercase) to month number
MONTH_NUMBER = {month.lower(): index + 1 for index, month in enumerate(MONTH_NAMES)}

# Number of days in each month, used to repair cross-month checkout durations
DAYS_IN_MONTH = {
    1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31,
}

# Assumed exchange rate used to convert USD room prices to SGD
USD_TO_SGD = 1.35


class DataPreparation:
    """
    A class used to clean and preprocess hotel booking no-show data.

    The raw data is a single CSV of customer booking records. Cleaning follows
    the EDA logic: target and identifier handling, month standardisation,
    numeric coercion, currency conversion, group-median price imputation,
    feature engineering of stay duration and booking lead time, column
    selection, and preprocessor construction.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing feature lists and the target column.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical and nominal features.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DataPreparation class with a configuration dictionary.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing feature lists
            and target column name.
        """
        self.config = config
        self.preprocessor = self._create_preprocessor()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the raw hotel booking DataFrame following EDA logic.

        Steps performed:
          1. Drop the single row where the target (no_show) is null.
          2. Drop booking_id (row identifier with no predictive value).
          3. Standardise booking_month, arrival_month and checkout_month to
             Title-case month names.
          4. Map word-form adult counts ('one', 'two') to integers and coerce
             num_adults to numeric.
          5. Parse the price string into a float and convert USD amounts to SGD
             at the assumed rate of 1 USD = 1.35 SGD, storing price_sgd.
          6. Fill missing room values with an explicit 'Unknown' category.
          7. Impute price_sgd with the median of the same (branch, room) group,
             falling back to the overall median.
          8. Engineer stay_length (repairing negative cross-month durations),
             booking_month_num, arrival_month_num and booking_lead_months.
          9. Drop checkout_month, checkout_day and the raw price column.
         10. Select only the columns declared in config.

        Args:
        -----
        df (pd.DataFrame): The raw booking DataFrame.

        Returns:
        --------
        pd.DataFrame: The cleaned and feature-engineered DataFrame.
        """
        logging.info("Starting data cleaning.")
        df_cleaned = df.copy()

        # Drop rows where the target is null, as the target cannot be imputed
        df_cleaned = df_cleaned.dropna(subset=[self.config["target_column"]])
        logging.info(f"Shape after dropping null target rows: {df_cleaned.shape}")

        # Drop booking_id as it is a row identifier with no predictive value
        df_cleaned = df_cleaned.drop(columns=["booking_id"])
        logging.info("Dropped booking_id column.")

        # Standardise month columns to Title-case month names
        month_cols = ["booking_month", "arrival_month", "checkout_month"]
        for col in month_cols:
            month_numbers = (
                df_cleaned[col].astype(str).str.strip().str.lower().map(MONTH_NUMBER)
            )
            df_cleaned[col] = month_numbers.map(
                {number: name for number, name in enumerate(MONTH_NAMES, start=1)}
            )

        # Map word-form adult counts to integers and coerce to numeric
        df_cleaned["num_adults"] = (
            df_cleaned["num_adults"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"one": 1, "two": 2})
        )
        df_cleaned["num_adults"] = pd.to_numeric(
            df_cleaned["num_adults"], errors="coerce"
        )

        # Parse the price string, e.g. 'SGD$ 492.98' or 'USD$ 1351.22'
        price_parts = df_cleaned["price"].str.extract(
            r"^(SGD|USD)\$\s*([\d,]+\.?\d*)$"
        )
        df_cleaned["price_sgd"] = pd.to_numeric(price_parts[1], errors="coerce")

        # Convert USD prices to SGD using the assumed exchange rate
        usd_mask = price_parts[0] == "USD"
        df_cleaned.loc[usd_mask, "price_sgd"] = (
            df_cleaned.loc[usd_mask, "price_sgd"] * USD_TO_SGD
        )
        logging.info(
            f"Price currency counts before conversion:\n{price_parts[0].value_counts()}"
        )

        # Fill missing room values with an explicit Unknown category
        df_cleaned["room"] = df_cleaned["room"].fillna("Unknown")
        logging.info(
            f"Null room values after filling with Unknown: "
            f"{df_cleaned['room'].isnull().sum()}"
        )

        # Impute price_sgd using the median of the same (branch, room) group,
        # falling back to the overall median for groups with no observed price
        group_median = df_cleaned.groupby(["branch", "room"])[
            "price_sgd"
        ].transform("median")
        df_cleaned["price_sgd"] = df_cleaned["price_sgd"].fillna(group_median)
        df_cleaned["price_sgd"] = df_cleaned["price_sgd"].fillna(
            df_cleaned["price_sgd"].median()
        )
        logging.info(
            f"Null price_sgd after group-median imputation: "
            f"{df_cleaned['price_sgd'].isnull().sum()}"
        )

        # Feature engineering: month numbers as 1-12 ordinals
        df_cleaned["booking_month_num"] = df_cleaned["booking_month"].map(
            {name: number for number, name in enumerate(MONTH_NAMES, start=1)}
        )
        df_cleaned["arrival_month_num"] = df_cleaned["arrival_month"].map(
            {name: number for number, name in enumerate(MONTH_NAMES, start=1)}
        )

        # Stay length: checkout day minus arrival day. Negative values arise when
        # checkout falls in the following month, so add the days in the arrival
        # month to recover a plausible duration. Floor at one night.
        df_cleaned["stay_length"] = (
            df_cleaned["checkout_day"] - df_cleaned["arrival_day"]
        )
        cross_month_mask = df_cleaned["stay_length"] < 0
        df_cleaned.loc[cross_month_mask, "stay_length"] = (
            df_cleaned.loc[cross_month_mask, "stay_length"]
            + df_cleaned.loc[cross_month_mask, "arrival_month_num"].map(DAYS_IN_MONTH)
        )
        df_cleaned["stay_length"] = df_cleaned["stay_length"].clip(lower=1)
        logging.info(
            f"Negative stay_length values before repair: {cross_month_mask.sum()}"
        )

        # Booking lead time in months, wrapped to a 0-11 range
        df_cleaned["booking_lead_months"] = (
            df_cleaned["arrival_month_num"] - df_cleaned["booking_month_num"]
        ) % 12

        # Drop columns fully captured by the engineered features
        df_cleaned = df_cleaned.drop(
            columns=["checkout_month", "checkout_day", "price", "booking_month"]
        )

        # Select only columns declared in config
        keep_cols = (
            self.config["numerical_features"]
            + self.config["nominal_features"]
            + [self.config["target_column"]]
        )
        df_cleaned = df_cleaned[[c for c in keep_cols if c in df_cleaned.columns]]

        logging.info(f"Final cleaned DataFrame shape: {df_cleaned.shape}")
        logging.info("Data cleaning completed.")
        return df_cleaned

    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Creates a preprocessor pipeline matching the EDA feature encoding choices:
          - Numerical: StandardScaler
          - Nominal: OneHotEncoder (drop='first', handle_unknown='ignore')

        Returns:
        --------
        sklearn.compose.ColumnTransformer
        """
        numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])

        nominal_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.config["numerical_features"]),
                ("nom", nominal_transformer, self.config["nominal_features"]),
            ],
            remainder="drop",  # drop anything not explicitly listed
            n_jobs=-1,
        )
        return preprocessor
