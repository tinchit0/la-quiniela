# Standard library imports
import os
import sys
import pickle
import sqlite3

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

# Project-specific imports
from pandas import DataFrame
from preprocessing import process_scores, calculate_match_results, normalize_dates, process_seasons
from io import load_matchday, load_historical_data
from features import inform_relatives_points, inform_win_lost_index, last5index, last_season_position
from validate import analyze_model_performance
import settings
import logging
import time



class QuinielaModel:
    """
    A model for processing and analyzing soccer match data, focused on match results and scoring.
    
    This class includes methods for preprocessing data, such as score parsing, result classification,
    and date normalization for soccer seasons spanning multiple years.
    """

    FEATURES = [
            "win_punct",
            "lost_punct",
            "points_relative_index",
            "last5_home",
            "last5_away",
            "last_season_points_home",
            "last_season_points_away",
        ]
    TARGET = "result"

    def __init__(self):
        """Initialize the QuinielaModel."""
        pass

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the input DataFrame by cleaning and transforming data.
        
        :param df: Input DataFrame containing match data
        :return: Preprocessed DataFrame with additional features
        :raises ValueError: If required columns are missing
        """
        processed_df = df.copy()
        processed_df = process_scores(processed_df)
        processed_df = calculate_match_results(processed_df)
        processed_df = normalize_dates(processed_df)
        processed_df = process_seasons(processed_df)
        return processed_df

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features for the input DataFrame.
        
        :param df: Input DataFrame containing match data
        :return: DataFrame with additional features
        """
        df_train = inform_relatives_points(df, df_train)
        logging.basicConfig(level=logging.INFO)
        
        start_time = time.time()
        logging.info("Calculating relative points")
        df_train = inform_relatives_points(df, df_train)
        logging.info(f"Relative points calculated in {time.time() - start_time:.2f} seconds")
        
        start_time = time.time()
        logging.info("Calculating win and loss index")
        df_train = inform_win_lost_index(df, df_train)
        logging.info(f"Win and loss index calculated in {time.time() - start_time:.2f} seconds")
        
        start_time = time.time()
        logging.info("Calculating last 5 index")
        df_train = last5index(df, df_train)
        logging.info(f"Last 5 index calculated in {time.time() - start_time:.2f} seconds")
        
        start_time = time.time()
        logging.info("Calculating last season position")
        df_train = last_season_position(df, df_train)
        logging.info(f"Last season position calculated in {time.time() - start_time:.2f} seconds")

    


    def train(self, df_train: pd.DataFrame):
        """Train the model on provided data."""

        x_train = df_train[self.FEATURES]
        y_train = df_train[self.TARGET]

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)

        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.1, random_state=40
        )

        clf = GradientBoostingClassifier()
        clf.fit(x_train, y_train)

        return clf, x_val, y_val

    def validate(self, clf, x_val: pd.DataFrame, y_val: pd.DataFrame):
        """Validate the model on provided data."""
        
        feature_importance = pd.DataFrame(
            {
                "feature": self.FEATURES,
                "importance": clf.feature_importances_,
            }
        )
        y_val_pred = clf.predict(x_val)
        feature_importance = feature_importance.sort_values("importance", ascending=False)
        analyze_model_performance(feature_importance, y_val, y_val_pred, clf)


    def predict(self, predict_data):
        """Generate predictions on the provided data."""
        return ["X" for _ in range(len(predict_data))]

    @classmethod
    def load(cls, filename):
        """Load model from file."""
        with open(filename, "rb") as f:
            model = pickle.load(f)
            assert isinstance(model, cls)
        return model

    def save(self, filename):
        """Save model to a file."""
        with open(filename, "wb") as f:
            pickle.dump(self, f)


if __name__ == "__main__":
    model = QuinielaModel()
    df = load_historical_data("2004/2005", 3)
    processed_df = model.preprocess(df)
    print(processed_df.head())
    model.train("2004/2005:2006/2007")
    model.save("my_quiniela.model")
    model = QuinielaModel.load("my_quiniela.model")
    print(model.predict(df))