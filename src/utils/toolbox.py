import pandas as pd
from colorama import Fore, Style

def load_csv_data(data_source: str) -> pd.DataFrame:
    try :
        data = pd.read_csv(data_source)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(
            Fore.RED + f"Could not find the CSV file at {data_source}. Please check the path/URL and try again." + Style.RESET_ALL)