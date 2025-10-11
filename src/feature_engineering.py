"""Feature engineering utilities for Ghana Indigenous Intel pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data_loader import DatasetBundle
from .utils import configure_logging

LOGGER = configure_logging()

_INDICATOR_KEYWORDS = (
    "sun",
    "cloud",
    "wind",
    "moon",
    "heat",
    "tree",
    "animal",
    "star",
    "rain",
    "humidity",
)


@dataclass
class FeatureEngineer:
    """Generate enriched tabular features from indigenous indicators."""

    bundle: DatasetBundle
    max_pairwise_interactions: int = 50
    rare_indicator_threshold: float = 0.01
    rare_region_threshold: float = 0.02
    temporal_columns: List[str] = field(default_factory=list)
    indicator_columns: List[str] = field(default_factory=list)
    region_column: Optional[str] = None
    farmer_column: Optional[str] = None
    target_encode_columns: List[str] = field(default_factory=list)
    indicator_means_: Optional[pd.Series] = None
    indicator_std_: Optional[pd.Series] = None
    indicator_active_ratio_: Optional[pd.Series] = None
    indicator_missing_ratio_: Optional[pd.Series] = None
    indicator_missing_patterns_: Optional[pd.Series] = None
    region_categories_: List[str] = field(default_factory=list)
    primary_indicator_categories_: List[str] = field(default_factory=list)
    rare_indicator_flags_: Dict[str, bool] = field(default_factory=dict)
    rare_region_flags_: Dict[str, bool] = field(default_factory=dict)
    region_frequency_: Optional[pd.Series] = None
    region_indicator_stats_: Optional[pd.DataFrame] = None
    primary_indicator_frequency_: Optional[pd.Series] = None
    indicator_freq_: Optional[pd.Series] = None
    description_freq_: Optional[pd.Series] = None
    is_fitted: bool = False

    def fit(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> "FeatureEngineer":
        """Identify columns and fit statistics on the training dataframe."""
        LOGGER.info("Fitting feature engineer on dataframe shape=%s", df.shape)

        self.indicator_columns = self._identify_indicator_columns(df)
        LOGGER.info("Detected indicator columns: %s", self.indicator_columns)

        self.temporal_columns = self._identify_temporal_columns(df)
        LOGGER.info("Detected temporal columns: %s", self.temporal_columns)

        region_series: Optional[pd.Series] = None
        self.region_column = self._identify_region_column(df)
        if self.region_column:
            region_series = df[self.region_column].fillna("unknown_region").astype(str)
            self.region_categories_ = region_series.unique().tolist()
            self.region_frequency_ = region_series.value_counts(normalize=True)
            self.rare_region_flags_ = {
                region: freq < self.rare_region_threshold for region, freq in self.region_frequency_.items()
            }

        self.farmer_column = self._identify_farmer_column(df)
        if self.farmer_column:
            self.target_encode_columns = [self.farmer_column]

        if self.indicator_columns:
            indicators = df[self.indicator_columns].fillna(0.0).astype(float)
            self.indicator_means_ = indicators.mean()
            self.indicator_std_ = indicators.std(ddof=0).replace(0, 1.0)

            active_mask = (indicators > 0).astype(int)
            self.indicator_active_ratio_ = active_mask.mean()

            frequencies = active_mask.mean()
            self.rare_indicator_flags_ = {
                col: freq < self.rare_indicator_threshold for col, freq in frequencies.items()
            }

            primary_indicator = indicators.idxmax(axis=1).fillna("none")
            self.primary_indicator_categories_ = primary_indicator.astype(str).unique().tolist()
            self.primary_indicator_frequency_ = (
                primary_indicator.astype(str).value_counts(normalize=True)
            )

            # Enhanced missingness statistics
            missing_mask = indicators.isnull()
            self.indicator_missing_ratio_ = missing_mask.mean()
            self.indicator_missing_patterns_ = missing_mask.sum(axis=1).value_counts(normalize=True)

            if region_series is not None:
                region_df = pd.DataFrame({
                    "region": region_series,
                    "indicator_sum": indicators.sum(axis=1),
                    "indicator_nonzero": (indicators > 0).sum(axis=1),
                    "indicator_mean": indicators.mean(axis=1),
                    "indicator_std": indicators.std(axis=1),
                    "indicator_missing_count": missing_mask.sum(axis=1),
                })
                region_stats = region_df.groupby("region").agg({
                    "indicator_sum": ["mean", "std"],
                    "indicator_nonzero": ["mean"],
                    "indicator_mean": ["mean"],
                    "indicator_std": ["mean"],
                    "indicator_missing_count": ["mean"],
                })
                region_stats.columns = [
                    "_".join(filter(None, col)).strip("_") for col in region_stats.columns
                ]
                self.region_indicator_stats_ = region_stats.fillna(0.0)

        if "indicator" in df.columns:
            indicator_series = df["indicator"].fillna("__missing__").astype(str).str.lower()
            self.indicator_freq_ = indicator_series.value_counts(normalize=True)

        if "indicator_description" in df.columns:
            description_series = (
                df["indicator_description"].fillna("__missing__").astype(str).str.lower()
            )
            self.description_freq_ = description_series.value_counts(normalize=True)

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, drop_target: bool = True) -> pd.DataFrame:
        """Apply feature transformations to a dataframe."""
        if not self.is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted before calling transform().")

        features = df.copy()
        if drop_target:
            features = features.drop(columns=[self.bundle.target_column], errors="ignore")

        features = self._convert_temporal_columns(features)
        features = self._generate_indicator_features(features)
        features = self._generate_aggregation_features(features)
        features = self._generate_primary_indicator(features)
        features = self._generate_geographic_features(features)
        features = self._generate_region_statistics(features)
        features = self._generate_farmer_features(features)
        features = self._generate_indicator_text_features(features)
        features = self._generate_precision_boosting_features(features)

        # Ensure consistent column order and numeric types for downstream models
        features = self._finalize_features(features)
        return features

    def fit_transform(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        self.fit(df, target)
        return self.transform(df)

    # ------------------------------------------------------------------
    # Identification helpers
    def _identify_indicator_columns(self, df: pd.DataFrame) -> List[str]:
        found = []
        for col in df.columns:
            col_lower = col.lower()
            if col_lower == self.bundle.target_column.lower():
                continue
            if any(keyword in col_lower for keyword in _INDICATOR_KEYWORDS):
                if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
                    found.append(col)
        return sorted(set(found))

    def _identify_temporal_columns(self, df: pd.DataFrame) -> List[str]:
        temporal_cols: List[str] = []
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                temporal_cols.append(col)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                temporal_cols.append(col)
        return temporal_cols

    def _identify_region_column(self, df: pd.DataFrame) -> Optional[str]:
        for candidate in ("region", "state", "district", "area"):
            for col in df.columns:
                if col.lower() == candidate:
                    return col
        return None

    def _identify_farmer_column(self, df: pd.DataFrame) -> Optional[str]:
        for candidate in ("farmer_id", "observer_id", "collector_id"):
            for col in df.columns:
                if col.lower() == candidate:
                    return col
        return None

    # ------------------------------------------------------------------
    # Feature generators
    def _convert_temporal_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        temporal_data: Dict[str, pd.Series] = {}

        for col in self.temporal_columns:
            if col not in df.columns:
                continue
            temp = pd.to_datetime(df[col], errors="coerce")
            temporal_data[col] = temp

            df[f"{col}_hour"] = temp.dt.hour.fillna(-1).astype(int)
            df[f"{col}_dayofweek"] = temp.dt.dayofweek.fillna(-1).astype(int)
            df[f"{col}_month"] = temp.dt.month.fillna(-1).astype(int)

            # Cyclical encodings
            df[f"{col}_hour_sin"] = np.sin(2 * np.pi * df[f"{col}_hour"] / 24)
            df[f"{col}_hour_cos"] = np.cos(2 * np.pi * df[f"{col}_hour"] / 24)
            df[f"{col}_dow_sin"] = np.sin(2 * np.pi * df[f"{col}_dayofweek"] / 7)
            df[f"{col}_dow_cos"] = np.cos(2 * np.pi * df[f"{col}_dayofweek"] / 7)
            df[f"{col}_month_sin"] = np.sin(2 * np.pi * df[f"{col}_month"] / 12)
            df[f"{col}_month_cos"] = np.cos(2 * np.pi * df[f"{col}_month"] / 12)

            df[f"{col}_is_morning"] = df[f"{col}_hour"].between(5, 11).astype(int)
            df[f"{col}_is_afternoon"] = df[f"{col}_hour"].between(12, 16).astype(int)
            df[f"{col}_is_evening"] = df[f"{col}_hour"].between(17, 20).astype(int)
            df[f"{col}_is_night"] = (
                df[f"{col}_hour"].between(21, 23).astype(int) | df[f"{col}_hour"].between(0, 4).astype(int)
            )

        for col in self.temporal_columns:
            if col in df.columns:
                df = df.drop(columns=[col])

        if {"prediction_time", "time_observed"}.issubset(temporal_data.keys()):
            prediction_time = temporal_data["prediction_time"]
            time_observed = temporal_data["time_observed"]

            gap_hours = (prediction_time - time_observed).dt.total_seconds() / 3600
            df["prediction_observed_gap_hours"] = gap_hours.fillna(0.0)
            df["prediction_observed_gap_abs_hours"] = gap_hours.abs().fillna(0.0)
            df["prediction_time_missing"] = prediction_time.isna().astype(int)
            df["time_observed_missing"] = time_observed.isna().astype(int)
            df["observation_precedes_prediction"] = (time_observed <= prediction_time).fillna(False).astype(int)

        return df

    def _generate_indicator_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.indicator_columns:
            return df

        indicators = df[self.indicator_columns].copy().fillna(0.0)

        # Standardize indicators using training statistics
        if self.indicator_means_ is not None and self.indicator_std_ is not None:
            standardized = (indicators - self.indicator_means_) / self.indicator_std_
            standardized = standardized.add_prefix("std_")
            df = pd.concat([df, standardized], axis=1)

        df["indicator_sum"] = indicators.sum(axis=1)
        df["indicator_mean"] = indicators.mean(axis=1)
        df["indicator_std"] = indicators.std(axis=1)
        df["indicator_nonzero_count"] = (indicators > 0).sum(axis=1)
        df["indicator_nonnull_count"] = indicators.notnull().sum(axis=1)
        df["indicator_diversity_score"] = (
            df["indicator_nonzero_count"] / (len(self.indicator_columns) or 1)
        )

        # Enhanced indicator patterns (domain-specific feature engineering)
        indicator_patterns = {
            'moon_night_pattern': ['moon', 'night', 'star'],
            'morning_sun_pattern': ['sun', 'morning', 'heat'],
            'wind_cloud_pattern': ['wind', 'cloud', 'direction'],
            'animal_behavior_pattern': ['bird', 'animal', 'tree'],
            'seasonal_pattern': ['month', 'season', 'temperature']
        }

        for pattern_name, keywords in indicator_patterns.items():
            # Find columns matching pattern
            pattern_cols = []
            for col in self.indicator_columns:
                if any(kw in col.lower() for kw in keywords):
                    pattern_cols.append(col)

            if len(pattern_cols) >= 2:
                # Create pattern strength feature
                numeric_pattern_cols = [col for col in pattern_cols if pd.api.types.is_numeric_dtype(df[col])]
                if numeric_pattern_cols:
                    df[f'{pattern_name}_strength'] = df[numeric_pattern_cols].mean(axis=1)
                    df[f'{pattern_name}_variability'] = df[numeric_pattern_cols].std(axis=1)
                    df[f'{pattern_name}_max'] = df[numeric_pattern_cols].max(axis=1)

        # Time-based rainfall prediction features
        if 'hour' in df.columns:
            # Rainfall more likely at certain times
            df['is_peak_rain_time'] = df['hour'].isin([14, 15, 16, 17, 18]).astype(int)  # Afternoon
            df['is_night_forecast'] = (df['hour'] >= 18).astype(int)

        # Regional rainfall patterns
        if self.region_column and self.region_column in df.columns:
            # Different regions have different rainfall patterns
            region_rainfall_mapping = {
                'Central': 1.2,  # More rainfall
                'Eastern': 1.0,
                'Ashanti': 0.8   # Less rainfall
            }
            df['region_rainfall_factor'] = df[self.region_column].map(region_rainfall_mapping).fillna(1.0)

        # Temporal lag features for indicator columns
        if len(self.indicator_columns) > 0:
            for lag in [1, 2, 3]:
                for col in self.indicator_columns:
                    if col in df.columns:
                        df[f"{col}_lag_{lag}"] = df[col].shift(lag).fillna(0.0)

        # Indicator clustering based on semantic similarity
        indicator_groups = {
            'weather_indicators': ['cloud', 'wind', 'sun', 'moon', 'star'],
            'time_indicators': ['hour', 'day', 'month', 'season'],
            'nature_indicators': ['tree', 'animal', 'bird', 'humidity'],
        }

        for group_name, keywords in indicator_groups.items():
            group_cols = [col for col in self.indicator_columns if any(kw in col.lower() for kw in keywords)]
            if len(group_cols) > 1:
                df[f"{group_name}_sum"] = df[group_cols].sum(axis=1)
                df[f"{group_name}_mean"] = df[group_cols].mean(axis=1)
                df[f"{group_name}_std"] = df[group_cols].std(axis=1)
                df[f"{group_name}_active_count"] = (df[group_cols] > 0).sum(axis=1)

        # Interaction features
        pairwise_cols = list(combinations(self.indicator_columns, 2))
        interactions_to_create = pairwise_cols
        if len(pairwise_cols) > self.max_pairwise_interactions:
            manual_pairs = [
                pair
                for pair in pairwise_cols
                if set(pair) & {"cloud", "cloud_intensity", "cloud_cover", "cloud_density"}
                and set(pair) & {"wind", "wind_speed", "wind_strength"}
            ]
            manual_pairs += [
                pair
                for pair in pairwise_cols
                if set(pair) & {"moon", "moon_phase", "moon_visibility"}
                and set(pair) & {"heat", "temperature", "heat_index"}
            ]
            manual_pairs += [
                pair
                for pair in pairwise_cols
                if set(pair) & {"sun", "sun_intensity", "sunshine"}
                and set(pair) & {"cloud", "cloud_cover", "cloud_density"}
            ]
            manual_pairs = list(dict.fromkeys(manual_pairs))  # preserve order, remove duplicates
            remaining_slots = self.max_pairwise_interactions - len(manual_pairs)
            if remaining_slots > 0:
                manual_pairs += pairwise_cols[:remaining_slots]
            interactions_to_create = manual_pairs

        for col_a, col_b in interactions_to_create:
            name = f"{col_a}_x_{col_b}"
            df[name] = indicators[col_a] * indicators[col_b]

        df["indicator_cooccurrence"] = (
            (indicators > 0).astype(int).sum(axis=1)
        )

        if self.indicator_active_ratio_ is not None:
            mean_vector = indicators.copy()
            for col in indicators.columns:
                mean_vector[col] = indicators[col] - self.indicator_means_.get(col, 0.0)
            df["indicator_distance_from_mean"] = np.sqrt((mean_vector ** 2).sum(axis=1))

        # Rare indicator flags
        rare_flags = {}
        for col, is_rare in self.rare_indicator_flags_.items():
            if is_rare and col in indicators.columns:
                rare_flags[f"{col}_is_rare"] = (indicators[col] > 0).astype(int)
        if rare_flags:
            rare_df = pd.DataFrame(rare_flags, index=df.index)
            df = pd.concat([df, rare_df], axis=1)

        return df

    def _generate_precision_boosting_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features specifically designed to improve precision for minority classes."""
        if not self.indicator_columns:
            return df

        indicators = df[self.indicator_columns].copy().fillna(0.0)

        # 1. Minority class confidence indicators
        # Features that help distinguish between NORAIN and actual rain classes
        df['minority_class_confidence'] = (
            indicators.mean(axis=1) * indicators.std(axis=1)
        )

        # 2. Rain pattern strength
        # Specific combinations that indicate different rain intensities
        rain_indicators = [col for col in self.indicator_columns if 'rain' in col.lower()]
        if len(rain_indicators) >= 2:
            df['rain_pattern_strength'] = indicators[rain_indicators].sum(axis=1)
            df['rain_pattern_consistency'] = 1 - indicators[rain_indicators].std(axis=1) / (indicators[rain_indicators].std(axis=1) + 1e-8)

        # 3. Weather condition clustering
        # Group indicators into weather condition clusters
        weather_clusters = {
            'humidity_indicators': ['humidity', 'moisture', 'wet'],
            'wind_indicators': ['wind', 'breeze', 'direction'],
            'cloud_indicators': ['cloud', 'overcast', 'sky'],
            'temperature_indicators': ['heat', 'temperature', 'warm']
        }

        for cluster_name, keywords in weather_clusters.items():
            cluster_cols = [col for col in self.indicator_columns if any(kw in col.lower() for kw in keywords)]
            if len(cluster_cols) > 1:
                df[f'{cluster_name}_coherence'] = indicators[cluster_cols].corrwith(indicators[cluster_cols].mean(axis=1), method='spearman').mean()
                df[f'{cluster_name}_diversity'] = (indicators[cluster_cols] > 0).sum(axis=1) / len(cluster_cols)

        # 4. Temporal precision features
        # Features that help with time-based precision
        if 'hour' in df.columns:
            # Rain is more likely at certain times - create precision-boosting features
            df['optimal_rain_hour'] = df['hour'].isin([14, 15, 16, 17, 18]).astype(int)  # Peak rain hours
            df['night_rain_probability'] = (df['hour'] >= 18).astype(int) & (indicators.mean(axis=1) > indicators.mean(axis=1).quantile(0.7))

        # 5. Regional precision patterns
        if self.region_column and self.region_column in df.columns:
            # Different regions have different precision patterns for rain prediction
            region_series = df[self.region_column].fillna("unknown_region").astype(str)

            # High precision regions for minority classes
            high_precision_regions = ['Central', 'Eastern']  # Based on domain knowledge
            df['high_precision_region'] = region_series.isin(high_precision_regions).astype(int)

            # Regional indicator reliability scores
            for region in self.region_categories_:
                region_mask = (region_series == region)
                if region_mask.sum() > 10:  # Only for regions with enough data
                    region_indicator_mean = indicators[region_mask].mean()
                    df[f'{region}_indicator_reliability'] = region_mask.astype(int) * (indicators.mean(axis=1) / (region_indicator_mean + 1e-8))

        # 6. Indicator confidence scores
        # Create confidence scores for each indicator type
        if self.indicator_active_ratio_ is not None:
            for col in self.indicator_columns:
                if col in indicators.columns:
                    col_mean = self.indicator_means_.get(col, 0.0)
                    col_std = self.indicator_std_.get(col, 1.0)
                    # Higher confidence when value is significantly above mean
                    df[f'{col}_confidence'] = (indicators[col] - col_mean) / (col_std + 1e-8)

        # 7. Rain intensity indicators
        # Features that help distinguish between different rain intensities
        heavy_rain_keywords = ['heavy', 'intense', 'storm', 'downpour']
        light_rain_keywords = ['light', 'drizzle', 'mist', 'sprinkle']

        heavy_cols = [col for col in self.indicator_columns if any(kw in col.lower() for kw in heavy_rain_keywords)]
        light_cols = [col for col in self.indicator_columns if any(kw in col.lower() for kw in light_rain_keywords)]

        if heavy_cols:
            df['heavy_rain_signal'] = indicators[heavy_cols].sum(axis=1)
        if light_cols:
            df['light_rain_signal'] = indicators[light_cols].sum(axis=1)

        # 8. Cross-indicator validation
        # Features that validate consistency across related indicators
        if len(self.indicator_columns) > 3:
            # Correlation-based validation score
            indicator_corr = indicators.corr()
            df['indicator_consistency'] = indicator_corr.mean().mean()

            # High correlation groups
            high_corr_pairs = []
            for i in range(len(self.indicator_columns)):
                for j in range(i+1, len(self.indicator_columns)):
                    if indicator_corr.iloc[i, j] > 0.7:
                        high_corr_pairs.append((self.indicator_columns[i], self.indicator_columns[j]))

            for pair in high_corr_pairs[:5]:  # Limit to top 5 pairs
                col1, col2 = pair
                df[f'{col1}_{col2}_agreement'] = 1 - abs(indicators[col1] - indicators[col2]) / (indicators[[col1, col2]].max(axis=1) + 1e-8)

        # 9. Environmental condition combinations
        # Specific combinations that boost precision for minority classes
        environmental_combinations = [
            (['humidity', 'cloud'], 'humid_cloudy_combo'),
            (['wind', 'cloud'], 'windy_cloudy_combo'),
            (['heat', 'humidity'], 'hot_humid_combo'),
            (['moon', 'cloud'], 'night_cloudy_combo')
        ]

        for combo_cols, combo_name in environmental_combinations:
            matching_cols = [col for col in self.indicator_columns if any(kw in col.lower() for kw in combo_cols)]
            if len(matching_cols) >= 2:
                df[f'{combo_name}_strength'] = indicators[matching_cols].mean(axis=1) * (indicators[matching_cols] > 0).sum(axis=1)

        # 10. Uncertainty reduction features
        # Features that help reduce prediction uncertainty
        if 'indicator_sum' in df.columns:
            # Higher indicator activity reduces uncertainty
            df['prediction_certainty'] = df['indicator_sum'] / (df['indicator_sum'].max() + 1e-8)

        # 11. Temporal stability features
        # Features that capture stability over time windows
        for window in [3, 7]:  # 3 and 7 period windows
            for col in self.indicator_columns[:5]:  # Limit to first 5 indicators
                if col in df.columns:
                    rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
                    rolling_std = df[col].rolling(window=window, min_periods=1).std().fillna(0)
                    df[f'{col}_stability_{window}'] = 1 - (rolling_std / (rolling_mean + 1e-8))

        # 12. Domain-specific precision boosters
        # Features based on indigenous knowledge patterns
        indigenous_patterns = {
            'morning_dew_pattern': ['dew', 'morning', 'grass'],
            'evening_calm_pattern': ['calm', 'evening', 'wind'],
            'storm_warning_pattern': ['dark', 'cloud', 'wind', 'animal'],
        }

        for pattern_name, keywords in indigenous_patterns.items():
            pattern_cols = [col for col in self.indicator_columns if any(kw in col.lower() for kw in keywords)]
            if len(pattern_cols) >= 2:
                df[f'{pattern_name}_score'] = indicators[pattern_cols].sum(axis=1) / len(pattern_cols)
                df[f'{pattern_name}_active'] = (indicators[pattern_cols] > 0).sum(axis=1) >= 2

        return df

    def _generate_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.indicator_columns:
            return df
        indicators = df[self.indicator_columns].copy().fillna(0.0)
        df["indicator_max"] = indicators.max(axis=1)
        df["indicator_min"] = indicators.min(axis=1)
        df["indicator_range"] = df["indicator_max"] - df["indicator_min"]
        return df

    def _generate_primary_indicator(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.indicator_columns:
            return df
        indicators = df[self.indicator_columns].copy().fillna(-np.inf)
        primary = indicators.idxmax(axis=1).fillna("none").replace({None: "none"}).astype(str)
        dummy = pd.get_dummies(primary, prefix="primary_indicator")
        # Ensure train/test alignment using stored categories
        for category in self.primary_indicator_categories_:
            col = f"primary_indicator_{category}"
            if col not in dummy.columns:
                dummy[col] = 0
        dummy = dummy.reindex(columns=sorted(dummy.columns))
        df = pd.concat([df, dummy], axis=1)
        df["primary_indicator_label"] = primary

        if self.primary_indicator_frequency_ is not None:
            freq_map = self.primary_indicator_frequency_.to_dict()
            freq_values = primary.map(freq_map).fillna(0.0)
            df["primary_indicator_frequency"] = freq_values
            df["primary_indicator_is_rare"] = (freq_values < self.rare_indicator_threshold).astype(int)
        return df

    def _generate_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.region_column or self.region_column not in df.columns:
            return df
        region_series = df[self.region_column].fillna("unknown_region").astype(str)
        dummies = pd.get_dummies(region_series, prefix=self.region_column)
        for category in self.region_categories_:
            col = f"{self.region_column}_{category}"
            if col not in dummies.columns:
                dummies[col] = 0
        dummies = dummies.reindex(columns=sorted(dummies.columns))
        df = pd.concat([df, dummies], axis=1)

        if self.indicator_columns:
            for indicator in self.indicator_columns:
                for category in self.region_categories_:
                    region_col = f"{self.region_column}_{category}"
                    if region_col in df.columns and indicator in df.columns:
                        interaction_name = f"{indicator}_x_{region_col}"
                        df[interaction_name] = df[indicator] * df[region_col]

        if self.region_frequency_ is not None:
            freq_map = self.region_frequency_.to_dict()
            freq_values = region_series.map(freq_map).fillna(0.0)
            df[f"{self.region_column}_frequency"] = freq_values
            rare_map = self.rare_region_flags_ or {}
            df[f"{self.region_column}_is_rare"] = region_series.map(lambda r: int(rare_map.get(r, 0))).fillna(0).astype(int)
        return df

    def _generate_region_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.region_column or self.region_column not in df.columns:
            return df

        region_series = df[self.region_column].fillna("unknown_region").astype(str)
        if self.region_indicator_stats_ is None:
            return df

        stats_df = self.region_indicator_stats_
        for col in stats_df.columns:
            feature_name = f"{self.region_column}_{col}"
            lookup = stats_df[col].to_dict()
            df[feature_name] = region_series.map(lookup).fillna(stats_df[col].mean())
        return df

    def _generate_farmer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.farmer_column or self.farmer_column not in df.columns:
            return df
        df[self.farmer_column] = df[self.farmer_column].astype(str).fillna("unknown_farmer")
        df["farmer_indicator_count"] = (
            df.groupby(self.farmer_column)[self.indicator_columns]
            .transform(lambda x: (x > 0).sum())
            .sum(axis=1)
            if self.indicator_columns
            else 0
        )
        return df

    def _generate_indicator_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "indicator" in df.columns:
            raw_indicator = df["indicator"]
            indicator_missing = raw_indicator.isna().astype(int)
            indicator_series = raw_indicator.fillna("__missing__").astype(str).str.lower()

            df["indicator_is_missing"] = indicator_missing
            df["indicator_length"] = indicator_series.str.len().astype(int)
            df["indicator_word_count"] = indicator_series.str.split().apply(len).astype(int)

            if self.indicator_freq_ is not None:
                df["indicator_frequency"] = indicator_series.map(self.indicator_freq_).fillna(0.0)
            else:
                df["indicator_frequency"] = 0.0

            keyword_hits = np.zeros(len(df), dtype=int)
            for keyword in _INDICATOR_KEYWORDS:
                col_name = f"indicator_kw_{keyword}"
                matches = indicator_series.str.contains(keyword, na=False).astype(int)
                df[col_name] = matches
                keyword_hits += matches
            df["indicator_keyword_hits"] = keyword_hits

            missing_mask = indicator_missing == 1
            df.loc[missing_mask, "indicator_length"] = 0
            df.loc[missing_mask, "indicator_word_count"] = 0
            df.loc[missing_mask, "indicator_frequency"] = 0.0
            df.loc[missing_mask, "indicator_keyword_hits"] = 0
            for keyword in _INDICATOR_KEYWORDS:
                col_name = f"indicator_kw_{keyword}"
                if col_name in df.columns:
                    df.loc[missing_mask, col_name] = 0

        if "indicator_description" in df.columns:
            raw_desc = df["indicator_description"]
            desc_missing = raw_desc.isna().astype(int)
            desc_series = raw_desc.fillna("__missing__").astype(str).str.lower()

            df["indicator_description_is_missing"] = desc_missing
            df["indicator_description_length"] = desc_series.str.len().astype(int)
            df["indicator_description_word_count"] = desc_series.str.split().apply(len).astype(int)

            if self.description_freq_ is not None:
                df["indicator_description_frequency"] = desc_series.map(self.description_freq_).fillna(0.0)
            else:
                df["indicator_description_frequency"] = 0.0

            desc_keyword_hits = np.zeros(len(df), dtype=int)
            for keyword in _INDICATOR_KEYWORDS:
                col_name = f"indicator_description_kw_{keyword}"
                matches = desc_series.str.contains(keyword, na=False).astype(int)
                df[col_name] = matches
                desc_keyword_hits += matches
            df["indicator_description_keyword_hits"] = desc_keyword_hits

            missing_mask = desc_missing == 1
            df.loc[missing_mask, "indicator_description_length"] = 0
            df.loc[missing_mask, "indicator_description_word_count"] = 0
            df.loc[missing_mask, "indicator_description_frequency"] = 0.0
            df.loc[missing_mask, "indicator_description_keyword_hits"] = 0
            for keyword in _INDICATOR_KEYWORDS:
                col_name = f"indicator_description_kw_{keyword}"
                if col_name in df.columns:
                    df.loc[missing_mask, col_name] = 0

        return df

    def _finalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Replace boolean columns with integers
        bool_cols = df.select_dtypes(include="bool").columns
        df[bool_cols] = df[bool_cols].astype(int)

        # Drop identifier columns that should not be fed to models
        if self.bundle.id_column in df.columns:
            df = df.drop(columns=[self.bundle.id_column])

        # Remove redundant string columns (already represented by dummies) but retain target encoding columns
        object_cols = df.select_dtypes(include="object").columns
        protected_cols = set(self.target_encode_columns or [])
        drop_cols = [col for col in object_cols if col not in protected_cols]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        # Convert all remaining columns to numeric and fill missing values
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.fillna(0)

        return df


class TargetEncoder:
    """Multi-class target encoder with smoothing, applied fold-wise."""
    def __init__(
        self,
        columns: Sequence[str],
        n_classes: int,
        smoothing: float = 0.3,
        noise: float = 1e-3,
        min_samples_leaf: int = 1,
    ) -> None:
        self.columns = list(columns)
        self.n_classes = n_classes
        self.smoothing = smoothing
        self.noise = noise
        self.min_samples_leaf = min_samples_leaf
        self.prior_: np.ndarray | None = None
        self.mappings_: Dict[str, Dict[str, np.ndarray]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetEncoder":
        if not self.columns:
            return self
        y_array = y.to_numpy()
        self.prior_ = self._compute_prior(y_array)

        for col in self.columns:
            mapping = self._fit_single_column(X[col], y_array)
            self.mappings_[col] = mapping
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.columns:
            return X
        if self.prior_ is None:
            raise RuntimeError("TargetEncoder must be fitted before calling transform().")

        X_encoded = X.copy()
        for col in self.columns:
            encoded = self._encode_column(X[col], self.mappings_.get(col, {}))
            X_encoded = X_encoded.drop(columns=[col])
            for class_idx in range(self.n_classes):
                X_encoded[f"{col}_te_{class_idx}"] = encoded[:, class_idx]
        return X_encoded

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def transform_with_noise(self, X: pd.DataFrame, training: bool) -> pd.DataFrame:
        transformed = self.transform(X)
        if training and self.noise > 0:
            noise = np.random.normal(0, self.noise, size=transformed.shape)
            transformed += noise
        return transformed

    def _compute_prior(self, y: np.ndarray) -> np.ndarray:
        counts = np.bincount(y, minlength=self.n_classes).astype(float)
        prior = counts / counts.sum()
        if prior.sum() == 0:
            prior = np.full(self.n_classes, 1.0 / self.n_classes)
        return prior

    def _fit_single_column(self, series: pd.Series, y: np.ndarray) -> Dict[str, np.ndarray]:
        df = pd.DataFrame({"feature": series.astype(str).fillna("__nan__"), "target": y})
        summary = df.groupby("feature")["target"].value_counts().unstack(fill_value=0)
        summary = summary.reindex(columns=range(self.n_classes), fill_value=0)
        counts = summary.sum(axis=1)

        smoothed = {}
        for category, row in summary.iterrows():
            cat_count = counts.loc[category]
            weight = cat_count / (cat_count + self.smoothing)
            probs = row.to_numpy(dtype=float) / max(cat_count, 1)
            smoothed_vector = weight * probs + (1 - weight) * self.prior_
            smoothed[category] = smoothed_vector
        return smoothed

    def _encode_column(self, series: pd.Series, mapping: Dict[str, np.ndarray]) -> np.ndarray:
        series_str = series.astype(str).fillna("__nan__")
        encoded = np.tile(self.prior_, (len(series_str), 1))
        for idx, value in enumerate(series_str):
            if value in mapping:
                encoded[idx] = mapping[value]
        return encoded


def select_target_encoding_columns(df: pd.DataFrame, max_unique: int = 100) -> List[str]:
    """Identify high-cardinality categorical columns suitable for target encoding."""
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    target_encode_cols = []
    for col in categorical_cols:
        unique_count = df[col].nunique(dropna=False)
        if 1 < unique_count <= max_unique:
            target_encode_cols.append(col)
    return target_encode_cols


def create_features(
    df: pd.DataFrame,
    bundle: DatasetBundle,
    engineer: Optional[FeatureEngineer] = None,
    is_train: bool = True,
    target: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Series], FeatureEngineer]:
    """Generate model-ready features using the shared FeatureEngineer instance."""

    if engineer is None:
        engineer = FeatureEngineer(bundle=bundle)

    if is_train:
        features = engineer.fit_transform(df, target=target)
        target_series = df[bundle.target_column] if target is None else target
        return features, target_series, engineer

    features = engineer.transform(df, drop_target=True)
    return features, None, engineer


def apply_target_encoding(
    train_df: pd.DataFrame,
    train_target: pd.Series,
    valid_df: pd.DataFrame,
    encoder_cols: Sequence[str],
    n_classes: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, TargetEncoder]:
    """Apply fold-specific target encoding and return transformed splits."""
    if not encoder_cols:
        return train_df, valid_df, TargetEncoder([], n_classes)

    encoder = TargetEncoder(columns=encoder_cols, n_classes=n_classes)
    encoder.fit(train_df[encoder_cols], train_target)

    train_encoded = encoder.transform_with_noise(train_df[encoder_cols], training=True)
    valid_encoded = encoder.transform_with_noise(valid_df[encoder_cols], training=False)

    train_out = train_df.drop(columns=list(encoder_cols), errors="ignore")
    valid_out = valid_df.drop(columns=list(encoder_cols), errors="ignore")

    train_out = pd.concat([train_out.reset_index(drop=True), train_encoded.reset_index(drop=True)], axis=1)
    valid_out = pd.concat([valid_out.reset_index(drop=True), valid_encoded.reset_index(drop=True)], axis=1)

    return train_out, valid_out, encoder
