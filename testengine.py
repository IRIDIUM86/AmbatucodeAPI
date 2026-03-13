"""
Manual unit test for the engine.py module.
"""

from engine import DataEngine

engine = DataEngine("DataReady.csv", "Item", "Transaction Date")

# def train_model(self, df: pd.DataFrame);

def train_model_unit_test():
    """
    Test the train_model method of the DataEngine class.
    """
    try:
        engine.train_model(engine.df)
        print("train_model_unit_test: PASSED")
    except Exception as e:
        print(f"train_model_unit_test: FAILED with error {e}")

# def predict_recommendation(self, df: pd.DataFrame):

def predict_recommendation_unit_test():
    """
    Test the predict_recommendation method of the DataEngine class.
    """
    try:
        recommendations = engine.predict_recommendation(engine.df)
        if isinstance(recommendations, list):
            print("predict_recommendation_unit_test: PASSED")
        else:
            print("predict_recommendation_unit_test: FAILED - Output is not a list")
    except Exception as e:
        print(f"predict_recommendation_unit_test: FAILED with error {e}")

train_model_unit_test()
predict_recommendation_unit_test()