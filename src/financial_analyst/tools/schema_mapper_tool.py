import json
import os
from typing import Type

import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class SchemaMapperToolInput(BaseModel):
    """Input schema for SchemaMapperTool."""

    csv_path: str = Field(..., description="Path to the CSV file to analyze.")


class SchemaMapperTool(BaseTool):
    name: str = "schema_mapper_tool"
    description: str = (
        "Analyzes the first 10 rows of a transaction CSV file to identify its structure. "
        "Returns a JSON mapping with: date_column, description_column, amount_logic, "
        "merchant_extraction, and category_hints. Required parameter: csv_path."
    )
    args_schema: Type[BaseModel] = SchemaMapperToolInput

    def _run(self, csv_path: str) -> str:
        """
        Analyze the first 10 rows of a CSV to determine its schema.

        Args:
            csv_path: Path to the CSV file

        Returns:
            JSON string with mapping instructions
        """
        try:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")

            # Try different delimiters
            df = self._load_csv(csv_path)

            # Take only first 10 rows for analysis
            sample_df = df.head(10)

            # Analyze the structure
            mapping = self._analyze_structure(sample_df)

            return json.dumps(mapping, indent=2)

        except Exception as e:
            return json.dumps({"error": f"Error analyzing CSV: {str(e)}"})

    def _load_csv(self, csv_path: str) -> pd.DataFrame:
        """Load CSV with various delimiter attempts."""
        delimiters = [";", ",", "\t", "|"]

        for delimiter in delimiters:
            try:
                df = pd.read_csv(csv_path, sep=delimiter, nrows=100)
                # Check if we got multiple columns (not just one giant column)
                if len(df.columns) > 1:
                    return df
            except:
                continue

        # Fallback: let pandas auto-detect
        return pd.read_csv(csv_path, nrows=100)

    def _analyze_structure(self, df: pd.DataFrame) -> dict:
        """Analyze dataframe structure and create mapping."""
        columns = df.columns.tolist()

        # Identify date column
        date_column = self._find_date_column(df, columns)

        # Identify description/note column
        description_column = self._find_description_column(df, columns)

        # Identify amount logic (single column or debit/credit split)
        amount_logic = self._find_amount_logic(df, columns)

        # Merchant extraction pattern
        merchant_extraction = self._suggest_merchant_extraction(df, description_column)

        # Category hints
        category_hints = self._suggest_category_hints(df, columns)

        return {
            "date_column": date_column,
            "description_column": description_column,
            "amount_logic": amount_logic,
            "merchant_extraction": merchant_extraction,
            "category_hints": category_hints,
            "detected_columns": columns,
            "sample_rows": df.head(3).to_dict(orient="records"),
        }

    def _find_date_column(self, df: pd.DataFrame, columns: list) -> str:
        """Find the column containing dates."""
        date_keywords = ["date", "time", "datetime", "timestamp", "when", "day"]

        # First, check column names
        for col in columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in date_keywords):
                return col

        # Try to parse columns as dates
        for col in columns:
            try:
                pd.to_datetime(df[col], errors="coerce")
                # If most values are valid dates
                if df[col].notna().sum() > len(df) * 0.7:
                    return col
            except:
                continue

        return "date"  # default fallback

    def _find_description_column(self, df: pd.DataFrame, columns: list) -> str:
        """Find the column containing transaction descriptions."""
        desc_keywords = [
            "description",
            "memo",
            "note",
            "detail",
            "payee",
            "merchant",
            "vendor",
        ]

        for col in columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in desc_keywords):
                return col

        # Look for text columns with varying content
        for col in columns:
            if df[col].dtype == "object":
                unique_ratio = len(df[col].unique()) / len(df)
                if unique_ratio > 0.5:  # High variability suggests descriptions
                    return col

        return "description"  # default fallback

    def _find_amount_logic(self, df: pd.DataFrame, columns: list) -> str:
        """Determine how to calculate transaction amounts."""
        amount_keywords = ["amount", "value", "total", "price", "sum"]
        debit_keywords = ["debit", "withdrawal", "expense", "out", "spent"]
        credit_keywords = ["credit", "deposit", "income", "in", "received"]

        amount_cols = []
        debit_cols = []
        credit_cols = []

        for col in columns:
            col_lower = str(col).lower()

            # Check for amount column
            if any(keyword in col_lower for keyword in amount_keywords):
                if self._is_numeric_column(df[col]):
                    amount_cols.append(col)

            # Check for debit column
            if any(keyword in col_lower for keyword in debit_keywords):
                if self._is_numeric_column(df[col]):
                    debit_cols.append(col)

            # Check for credit column
            if any(keyword in col_lower for keyword in credit_keywords):
                if self._is_numeric_column(df[col]):
                    credit_cols.append(col)

        # Determine logic
        if amount_cols and not (debit_cols or credit_cols):
            return f"use_column:{amount_cols[0]}"
        elif debit_cols and credit_cols:
            return f"combine_columns:credit={credit_cols[0]},debit={debit_cols[0]}"
        elif amount_cols:
            return f"use_column:{amount_cols[0]}"

        # Fallback: look for any numeric column
        for col in columns:
            if self._is_numeric_column(df[col]):
                return f"use_column:{col}"

        return "use_column:amount"

    def _is_numeric_column(self, series: pd.Series) -> bool:
        """Check if a column contains numeric values."""
        try:
            pd.to_numeric(series, errors="coerce")
            non_null_count = series.notna().sum()
            return non_null_count > len(series) * 0.7
        except:
            return False

    def _suggest_merchant_extraction(self, df: pd.DataFrame, desc_column: str) -> str:
        """Suggest pattern for merchant extraction from description."""
        if desc_column not in df.columns:
            return "extract_first_word"

        # Analyze description patterns
        sample_descriptions = df[desc_column].dropna().head(10).tolist()

        # Common patterns:
        # - All caps words often represent merchants
        # - First few words before special characters
        # - Text before dashes or parentheses

        has_uppercase = any(
            any(word.isupper() for word in str(desc).split())
            for desc in sample_descriptions
        )

        has_separators = any(
            "-" in str(desc) or "(" in str(desc) for desc in sample_descriptions
        )

        if has_uppercase:
            return "extract_uppercase_words"
        elif has_separators:
            return "extract_before_separator"
        else:
            return "extract_first_two_words"

    def _suggest_category_hints(self, df: pd.DataFrame, columns: list) -> str:
        """Suggest patterns for categorization."""
        category_keywords = ["category", "type", "class", "group"]

        # Check if there's an explicit category column
        for col in columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in category_keywords):
                unique_categories = df[col].unique().tolist()
                return f"use_column:{col} (values: {', '.join(map(str, unique_categories[:5]))})"

        # Suggest keyword-based categorization
        hints = {
            "food": ["restaurant", "cafe", "food", "coffee", "dinner", "lunch"],
            "transport": ["fuel", "gas", "uber", "taxi", "parking"],
            "shopping": ["store", "shop", "amazon", "market"],
            "utilities": ["electric", "water", "internet", "phone"],
            "entertainment": ["movie", "cinema", "netflix", "spotify"],
        }

        return f"keyword_mapping:{json.dumps(hints)}"
