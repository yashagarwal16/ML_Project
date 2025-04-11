# Importing required libraries
import os  # Used for interacting with the operating system, like file/folder handling
import sys  # Provides access to system-specific parameters and functions
from src.exception import CustomException  # Custom exception handling class (defined elsewhere)
from src.logger import logging  # Logging module to record logs
import pandas as pd  # Used to work with tabular data (like Excel/CSV files)

from sklearn.model_selection import train_test_split  # Used to split dataset into training and testing parts
from dataclasses import dataclass  # Provides a decorator to create data classes easily

from src.components.data_transformation import DataTransformation  # Importing DataTransformation class
from src.components.data_transformation import DataTransformationConfig  # Importing DataTransformationConfig class

# Step 1: Configuration class using dataclass
@dataclass
class DataIngestionConfig:
    # File paths to store train, test, and raw data
    train_data_path: str = os.path.join('artifacts', "train.csv")  # Path to save training data
    test_data_path: str = os.path.join('artifacts', "test.csv")    # Path to save testing data
    raw_data_path: str = os.path.join('artifacts', "data.csv")     # Path to save original/raw data

# Step 2: DataIngestion class that handles reading and splitting the data
class DataIngestion:
    def __init__(self):
        # Creating an instance of the configuration class
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")  # Log message
        
        try:
            # Step 3: Read data from the CSV file
            df = pd.read_csv('notebook\data\data\stud.csv')  # Reading data from source file
            logging.info('Read the dataset as dataframe')  # Log success

            # Step 4: Create folder if it doesn't exist to store output files
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Step 5: Save raw/original data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")  # Log split starting

            # Step 6: Split the dataset into training and testing sets (80% train, 20% test)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Step 7: Save the training data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            # Step 8: Save the testing data
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")  # Log completion

            # Step 9: Return paths to the train and test data files
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        # Step 10: If any error occurs, raise a custom exception
        except Exception as e:
            raise CustomException(e, sys)

# Step 11: This block runs when the script is executed directly
if __name__ == "__main__":
    obj = DataIngestion()  # Create object of DataIngestion class
    train_data, test_data = obj.initiate_data_ingestion()  # Call the method to perform ingestion

    data_transformation = DataTransformation()  # Create object of DataTransformation class
    data_transformation.initiate_data_transformation(train_data, test_data)  # Perform data transformation

    