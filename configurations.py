from pathlib import Path


# Directory containing the original datasets.
DATA_DIR = Path("data/Original")
# Directory for generated reports and figures.
OUTPUT_DIR = Path("output")
# Path to the training dataset.
TRAIN_PATH = DATA_DIR / "train.csv"
# Path to the test dataset.
TEST_PATH = DATA_DIR / "test.csv"
# Name of the prediction target column.
TARGET = "OutcomeType"
# Columns that cause data leakage
LEAKAGE_COLS = ['OutcomeSubtype']
# Unique identifiers that hold no predictive value
ID_COLS = ['AnimalID']
# Combined list of features to strictly exclude from model training
DROP_FEATURES = LEAKAGE_COLS + ID_COLS
# Seed for reproducible random operations.
RANDOM_STATE = 42

# Shared Plotly visual style and export settings.
PLOTLY_TEMPLATE = "plotly_white"
PLOTLY_CONFIG = {"displaylogo": False, "responsive": True}
PLOTLY_WIDTH = 950
PLOTLY_HEIGHT = 520
