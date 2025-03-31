import pandas as pd
import os
import sqlite3
from pathlib import Path
import io
import tkinter as tk
from tkinter import filedialog


class DataFrameReader:
    """
    A comprehensive class for reading various file formats into pandas DataFrames with automatic 
    format detection and file dialog selection by default.
    
    This class handles multiple file formats including CSV, Excel, JSON, Parquet, SQLite, 
    HDF5, and more. It identifies file types via extension or content examination
    and applies the appropriate reader. When instantiated without a file path, it will 
    automatically open a file selection dialog.
    
    Examples:
    ---------
    # Open with file dialog (default behavior)
    reader = DataFrameReader()  # Opens file dialog automatically
    df = reader.dataframe
    
    # Provide a specific file path
    reader = DataFrameReader("data.csv")
    df = reader.dataframe
    
    # Using class methods for convenience
    df = DataFrameReader.read_file()  # Opens file dialog
    df = DataFrameReader.read_file("data.unknown")  # Specific file
    
    # Create instance first, read multiple files sequentially
    reader = DataFrameReader()  # Opens file dialog
    # Later read another file
    reader.read("data.xlsx", sheet_name="Expenses")
    expenses_df = reader.dataframe
    
    # Access metadata about the loaded file
    reader = DataFrameReader()  # Opens file dialog
    print(f"Selected file: {reader.file_path}")
    print(f"File type detected: {reader.file_type}")
    print(f"Rows: {reader.dataframe.shape[0]}, Columns: {reader.dataframe.shape[1]}")
    """
    
    # Class attributes
    SUPPORTED_EXTENSIONS = {
        # CSV and text formats
        '.csv': 'csv',
        '.txt': 'text',
        '.tsv': 'text',
        '.data': 'text',
        '.dat': 'text',
        
        # Excel formats
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.xlsm': 'excel',
        '.xlsb': 'excel',
        
        # Other common formats
        '.json': 'json',
        '.parquet': 'parquet',
        '.pq': 'parquet',
        '.feather': 'feather',
        '.ftr': 'feather',
        '.pkl': 'pickle',
        '.pickle': 'pickle',
        
        # Database and storage formats
        '.db': 'sqlite',
        '.sqlite': 'sqlite',
        '.sqlite3': 'sqlite',
        '.h5': 'hdf',
        '.hdf5': 'hdf',
        '.hdf': 'hdf'
    }
    
    def __init__(self, file_or_df=None, dialog_title="Select a data file", **kwargs):
        """
        Initialize a DataFrameReader instance.
        
        If file_or_df is None, automatically opens a file selection dialog.
        
        Parameters:
        -----------
        file_or_df : str, Path, pandas.DataFrame, or None, default=None
            Path to the file to be read, a DataFrame, or None to open file dialog
        dialog_title : str, default="Select a data file"
            Title for the file selection dialog if opened
        **kwargs : dict
            Additional arguments to pass to the specific pandas reader function
        """
        # Initialize instance variables
        self.file_path = None
        self.dataframe = None
        self.file_type = None
        self.reader_options = {}
        
        # Process input if provided at initialization
        if file_or_df is None:
            # If no input is provided, open file dialog by default
            file_path = self._open_file_dialog(dialog_title)
            if file_path:  # Only proceed if a file was selected
                self.read(file_path, **kwargs)
        else:
            # If input is provided, process it normally
            self.read(file_or_df, **kwargs)
    
    def read(self, file_or_df=None, copy=False, dialog_title="Select a data file", **kwargs):
        """
        Read data from a file or DataFrame.
        
        If file_or_df is None, automatically opens a file selection dialog.
        
        Parameters:
        -----------
        file_or_df : str, Path, pandas.DataFrame, or None, default=None
            Path to the file to be read, a DataFrame, or None to open file dialog
        copy : bool, default=False
            Whether to return a copy if input is a DataFrame
        dialog_title : str, default="Select a data file"
            Title for the file selection dialog if opened
        **kwargs : dict
            Additional arguments to pass to the specific pandas reader function
            
        Returns:
        --------
        pandas.DataFrame
            The data read into a DataFrame
        
        Notes:
        ------
        This method also sets instance variables: dataframe, file_path, and file_type
        """
        # Store reader options
        self.reader_options = kwargs
        
        # Handle DataFrame input
        if isinstance(file_or_df, pd.DataFrame):
            self.dataframe = file_or_df.copy() if copy else file_or_df
            self.file_path = None
            self.file_type = "dataframe"
            return self.dataframe
        
        # Handle case when no file is specified - open dialog automatically
        if file_or_df is None:
            file_or_df = self._open_file_dialog(dialog_title)
            if not file_or_df:  # User canceled the dialog
                return None
        
        # Convert to Path for consistent handling
        file_path = Path(file_or_df)
        self.file_path = file_path
        
        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            # Determine file type and read
            self._determine_file_type(file_path)
            self._read_file_by_type(file_path, **kwargs)
            return self.dataframe
            
        except Exception as e:
            # Add context to the exception for easier debugging
            raise Exception(f"Error reading file {file_path}: {str(e)}")
    
    def _determine_file_type(self, file_path):
        """
        Determine the file type based on extension.
        
        Parameters:
        -----------
        file_path : Path
            Path to the file
            
        Returns:
        --------
        str
            The determined file type
        """
        # Get file extension (convert to lowercase for consistency)
        file_extension = file_path.suffix.lower()
        
        # Check if extension is in supported list
        if file_extension in self.SUPPORTED_EXTENSIONS:
            self.file_type = self.SUPPORTED_EXTENSIONS[file_extension]
        else:
            # For unknown extensions, set to 'unknown' and handle specially later
            self.file_type = 'unknown'
            
        return self.file_type
    
    def _read_file_by_type(self, file_path, **kwargs):
        """
        Read file based on determined type.
        
        Parameters:
        -----------
        file_path : Path
            Path to the file
        **kwargs : dict
            Additional arguments for the reader
        """
        # Select appropriate reader based on file type
        readers = {
            'csv': self._read_csv,
            'text': self._read_text,
            'excel': self._read_excel,
            'json': self._read_json,
            'parquet': self._read_parquet,
            'feather': self._read_feather,
            'pickle': self._read_pickle,
        }
        
        # Get reader function and apply it
        reader_func = readers.get(self.file_type)
        if reader_func:
            reader_func(file_path, **kwargs)
        else:
            raise ValueError(f"No reader available for file type: {self.file_type}")
            
    def _read_csv(self, file_path, **kwargs):
        """Read a CSV file."""
        temp_df = pd.read_csv(file_path, **kwargs)
        self.dataframe = self.clean_data(temp_df)
    
    def _read_text(self, file_path, **kwargs):
        """Read a text file with various potential delimiters."""
        temp_df = self._try_common_delimiters(file_path, **kwargs)
        self.dataframe = self.clean_data(temp_df)
    def _read_excel(self, file_path, **kwargs):
        """Read an Excel file."""
        temp_df= pd.read_excel(file_path, **kwargs)
        self.dataframe = self.clean_data(temp_df)
    
    def _read_json(self, file_path, **kwargs):
        """Read a JSON file."""
        temp_df = pd.read_json(file_path, **kwargs)
        self.dataframe = self.clean_data(temp_df)
    
    def _read_parquet(self, file_path, **kwargs):
        """Read a Parquet file."""
        temp_df = pd.read_parquet(file_path, **kwargs)
        self.dataframe = self.clean_data(temp_df)
    
    def _read_feather(self, file_path, **kwargs):
        """Read a Feather file."""
        temp_df = pd.read_feather(file_path, **kwargs)
        self.dataframe = self.clean_data(temp_df)
    
    def _read_pickle(self, file_path, **kwargs):
        """Read a Pickle file."""
        temp_df = pd.read_pickle(file_path, **kwargs)
        self.dataframe = self.clean_data(temp_df)

    
    def _try_common_delimiters(self, file_path, **kwargs):
        """
        Attempt to read a text file using common delimiters.
        
        Parameters:
        -----------
        file_path : Path
            Path to the file
        **kwargs : dict
            Additional arguments for pd.read_csv
            
        Returns:
        --------
        pandas.DataFrame
            The data read into a DataFrame
        """
        # Sample the first few lines to detect the delimiter
        with open(file_path, 'r', errors='replace') as f:
            sample = ''.join(f.readline() for _ in range(5))
            
        # Count occurrences of common delimiters in the sample
        delimiters = [(',', 0), ('\t', 0), (';', 0), ('|', 0)]
        for i, (delimiter, _) in enumerate(delimiters):
            delimiters[i] = (delimiter, sample.count(delimiter))
            
        # Sort delimiters by occurrence count (descending)
        delimiters.sort(key=lambda x: x[1], reverse=True)
            
        # Try each delimiter, starting with the most common
        for delimiter, count in delimiters:
            if count > 0:  # Only try delimiters that actually appear in the sample
                try:
                    df = pd.read_csv(file_path, delimiter=delimiter, **kwargs)
                    # If successful and has more than one column, return it
                    if len(df.columns) > 1:
                        return df
                except Exception:
                    continue
                    
        # If all specific delimiters fail, try pandas' default CSV reader
        try:
            return pd.read_csv(file_path, **kwargs)
        except Exception:
            # As a last resort, read as a single-column text file
            with open(file_path, 'r', errors='replace') as f:
                content = f.readlines()
            return pd.DataFrame({'text': content})
    
    def _open_file_dialog(self, title="Select a data file"):
        """
        Open a file dialog for selecting data files.
        
        Parameters:
        -----------
        title : str, default="Select a data file"
            Title for the file selection dialog window
            
        Returns:
        --------
        str or None
            Path to the selected file, or None if dialog was canceled
        """
        # Create a root window but hide it
        root = tk.Tk()
        root.withdraw()
            
        # Make the dialog stay on top of other windows
        root.attributes("-topmost", True)
            
        # Define file types to show in the dialog
        filetypes = [
            ("All Data Files", "*.csv;*.xlsx;*.xls;*.json;*.parquet;*.pq;*.feather;*.pkl;*.pickle;*.db;*.sqlite;*.h5;*.hdf5;*.txt;*.tsv"),
            ("CSV Files", "*.csv"),
            ("Excel Files", "*.xlsx;*.xls;*.xlsm;*.xlsb"),
            ("JSON Files", "*.json"),
            ("Parquet Files", "*.parquet;*.pq"),
            ("Feather Files", "*.feather;*.ftr"),
            ("Pickle Files", "*.pkl;*.pickle"),
            ("Database Files", "*.db;*.sqlite;*.sqlite3"),
            ("HDF5 Files", "*.h5;*.hdf5;*.hdf"),
            ("Text Files", "*.txt;*.tsv;*.data;*.dat"),
            ("All Files", "*.*")
        ]
            
        # Open the dialog and get selected file path
        file_path = filedialog.askopenfilename(
            title=title,
            filetypes=filetypes,
            initialdir=os.getcwd()  # Start in current directory
        )
            
        # Clean up the root window
        root.destroy()
            
        return file_path if file_path else None
  
    def clean_data(self, df):
        if df is None:
            raise ValueError("No data provided")

        # making sure columns exists
        req_cols = ["id","device_id", "login_time", "latitude", "longitude"]
        for col in req_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
            
        # converting login_time to datetime
        if not pd.api.types.is_datetime64_any_dtype(df["login_time"]):
            df["login_time"] = pd.to_datetime(df["login_time"])
        #removing null rows
        df = df.dropna(subset=["latitude", "longitude"])

        #making sure longitude and latitude are valid ranges
        df = df[(df["latitude"] >= -90) & (df["latitude"] <= 90) & (df["longitude"] >= -180) & (df["longitude"] <= 180)]

        #sorting data by id and time 
        df = df.sort_values(["id", "login_time"])

        return df

if __name__ == "__main__":
    # Example usage
    reader = DataFrameReader()
    df = reader.dataframe
    print(f"File path: {reader.file_path}")
    print(f"File type: {reader.file_type}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(df.head())