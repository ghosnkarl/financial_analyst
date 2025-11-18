import json
import os
import re
from datetime import datetime
from typing import Type

import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class BatchProcessorToolInput(BaseModel):
    """Input schema for BatchProcessorTool."""

    csv_path: str = Field(..., description="Path to the CSV file to process.")
    start_row: int = Field(
        default=0,
        description="Starting row index (inclusive). Defaults to 0 for full file processing.",
    )
    end_row: int = Field(
        default=-1,
        description="Ending row index (inclusive). Use -1 to process until end of file.",
    )
    schema_mapping: str = Field(
        default="{}",
        description="JSON string with schema mapping instructions. Will auto-detect if not provided.",
    )


class BatchProcessorTool(BaseTool):
    name: str = "batch_processor_tool"
    description: str = (
        "Processes a batch of transactions from a CSV file using schema mapping instructions. "
        "Transforms raw data into standardized JSON format with date, description, amount, "
        "merchant, category, and type (expense/income) fields. "
        "Required parameters: csv_path, start_row, end_row, schema_mapping."
    )
    args_schema: Type[BaseModel] = BatchProcessorToolInput

    def _run(
        self,
        csv_path: str,
        start_row: int = 0,
        end_row: int = -1,
        schema_mapping: str = "{}",
    ) -> str:
        """
        Process transactions using the provided schema mapping.

        Args:
            csv_path: Path to the CSV file
            start_row: Starting row index (0-based, inclusive). Defaults to 0.
            end_row: Ending row index (0-based, inclusive). Use -1 for all rows.
            schema_mapping: JSON string with mapping instructions. Auto-detects if empty.

        Returns:
            JSON array of processed transactions
        """
        try:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")

            # Parse schema mapping (or use empty dict for auto-detection)
            mapping = {}
            if (
                schema_mapping
                and schema_mapping.strip()
                and schema_mapping.strip() != "{}"
            ):
                try:
                    mapping = json.loads(schema_mapping)
                except json.JSONDecodeError as e:
                    # If JSON parsing fails, try to extract key information from the malformed string
                    mapping = self._extract_mapping_from_text(schema_mapping)
                    if not mapping:
                        return json.dumps(
                            {
                                "error": f"Invalid schema_mapping JSON: {str(e)}. Please provide valid JSON or use the Schema Mapper Tool first."
                            }
                        )

            # Load CSV
            df = self._load_csv(csv_path)

            # Determine end_row
            if end_row == -1:
                end_row = len(df) - 1

            # Extract the batch
            batch_df = df.iloc[start_row : end_row + 1]

            # Process each transaction
            processed_transactions = []
            for _, row in batch_df.iterrows():
                transaction = self._process_transaction(row, mapping)
                processed_transactions.append(transaction)

            # Save to output file
            output_path = "output/transactions.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(processed_transactions, f, indent=2)

            return f"Successfully processed {len(processed_transactions)} transactions and saved to {output_path}"

        except Exception as e:
            return json.dumps({"error": f"Error processing batch: {str(e)}"})

    def _load_csv(self, csv_path: str) -> pd.DataFrame:
        """Load CSV with various delimiter attempts."""
        delimiters = [";", ",", "\t", "|"]

        for delimiter in delimiters:
            try:
                df = pd.read_csv(csv_path, sep=delimiter)
                if len(df.columns) > 1:
                    return df
            except:
                continue

        return pd.read_csv(csv_path)

    def _process_transaction(self, row: pd.Series, mapping: dict) -> dict:
        """Process a single transaction row using the mapping."""
        # Extract date
        date = self._extract_date(row, mapping.get("date_column", "date"))

        # Extract description
        description = self._extract_description(
            row, mapping.get("description_column", "description")
        )

        # Calculate amount
        amount = self._calculate_amount(
            row, mapping.get("amount_logic", "use_column:amount")
        )

        # Extract merchant
        merchant = self._extract_merchant(
            description, row, mapping.get("merchant_extraction", "extract_first_word")
        )

        # Categorize
        category = self._categorize_transaction(
            description, row, mapping.get("category_hints", "")
        )

        # Extract type (expense or income)
        transaction_type = self._extract_type(row, mapping.get("type_column", "type"))

        return {
            "date": date,
            "description": description,
            "amount": amount,
            "merchant": merchant,
            "category": category,
            "type": transaction_type,
        }

    def _extract_date(self, row: pd.Series, date_column: str) -> str:
        """Extract and format date."""
        try:
            if date_column in row.index:
                date_val = row[date_column]
                if pd.notna(date_val):
                    # Try to parse as datetime
                    dt = pd.to_datetime(date_val)
                    return dt.strftime("%Y-%m-%d")
        except:
            pass

        return datetime.now().strftime("%Y-%m-%d")

    def _extract_description(self, row: pd.Series, desc_column: str) -> str:
        """Extract description/note."""
        if desc_column in row.index:
            desc = row[desc_column]
            if pd.notna(desc):
                return str(desc).strip()

        # Fallback: try common description columns
        for col in ["note", "memo", "description", "payee", "merchant"]:
            if col in row.index and pd.notna(row[col]):
                return str(row[col]).strip()

        return "No description"

    def _calculate_amount(self, row: pd.Series, amount_logic: str) -> float:
        """Calculate transaction amount based on logic."""
        try:
            if isinstance(amount_logic, str) and amount_logic.startswith("use_column:"):
                col_name = amount_logic.split(":", 1)[1]
                if col_name in row.index:
                    amount = pd.to_numeric(row[col_name], errors="coerce")
                    if pd.notna(amount):
                        return float(amount)

            elif isinstance(amount_logic, str) and amount_logic.startswith(
                "combine_columns:"
            ):
                # Parse: credit=col1,debit=col2
                parts = amount_logic.split(":", 1)[1]
                params = dict(part.split("=") for part in parts.split(","))

                credit_col = params.get("credit", "")
                debit_col = params.get("debit", "")

                credit_val = 0.0
                debit_val = 0.0

                if credit_col in row.index:
                    credit_val = pd.to_numeric(row[credit_col], errors="coerce")
                    credit_val = float(credit_val) if pd.notna(credit_val) else 0.0

                if debit_col in row.index:
                    debit_val = pd.to_numeric(row[debit_col], errors="coerce")
                    debit_val = float(debit_val) if pd.notna(debit_val) else 0.0

                # Debit is negative, credit is positive
                return credit_val - debit_val
        except:
            pass

        # Fallback: try to find any numeric column
        for col in row.index:
            try:
                val = pd.to_numeric(row[col], errors="coerce")
                if pd.notna(val):
                    return float(val)
            except:
                continue

        return 0.0

    def _extract_merchant(
        self, description: str, row: pd.Series, extraction_pattern: str
    ) -> str:
        """Extract merchant name from description."""
        # Check if there's a dedicated merchant/payee column
        for col in ["merchant", "payee", "vendor", "store"]:
            if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
                return str(row[col]).strip()

        # Apply extraction pattern
        if (
            isinstance(extraction_pattern, str)
            and extraction_pattern == "extract_first_word"
        ):
            words = description.split()
            return words[0] if words else "Unknown"

        elif (
            isinstance(extraction_pattern, str)
            and extraction_pattern == "extract_first_two_words"
        ):
            words = description.split()
            return (
                " ".join(words[:2])
                if len(words) >= 2
                else words[0] if words else "Unknown"
            )

        elif (
            isinstance(extraction_pattern, str)
            and extraction_pattern == "extract_uppercase_words"
        ):
            words = description.split()
            uppercase_words = [w for w in words if w.isupper() and len(w) > 2]
            return (
                " ".join(uppercase_words)
                if uppercase_words
                else words[0] if words else "Unknown"
            )

        elif (
            isinstance(extraction_pattern, str)
            and extraction_pattern == "extract_before_separator"
        ):
            # Extract text before -, (, or other separators
            match = re.match(r"^([^-\(\[]+)", description)
            if match:
                return match.group(1).strip()
            return description.split()[0] if description.split() else "Unknown"

        # Default: first word
        words = description.split()
        return words[0] if words else "Unknown"

    def _extract_type(self, row: pd.Series, type_column: str) -> str:
        """Extract transaction type (expense or income)."""
        # Check if there's a dedicated type column
        if type_column in row.index and pd.notna(row[type_column]):
            type_val = str(row[type_column]).strip().lower()
            # Normalize common variations
            if type_val in ["expense", "debit", "withdrawal", "payment", "out"]:
                return "expense"
            elif type_val in ["income", "credit", "deposit", "in"]:
                return "income"
            # Return the original value if it's already normalized
            if type_val in ["expense", "income"]:
                return type_val

        # Try other common type column names
        for col in ["type", "transaction_type", "kind", "category_type"]:
            if col in row.index and pd.notna(row[col]):
                type_val = str(row[col]).strip().lower()
                if type_val in ["expense", "debit", "withdrawal", "payment", "out"]:
                    return "expense"
                elif type_val in ["income", "credit", "deposit", "in"]:
                    return "income"

        # Fallback: assume expense if not specified
        return "expense"

    def _categorize_transaction(
        self, description: str, row: pd.Series, category_hints: str
    ) -> str:
        """Categorize transaction based on hints."""
        desc_lower = description.lower()

        # Check if there's a dedicated category column
        if isinstance(category_hints, str) and category_hints.startswith("use_column:"):
            col_info = category_hints.split(":", 1)[1]
            col_name = col_info.split("(")[0].strip()
            if col_name in row.index and pd.notna(row[col_name]):
                return str(row[col_name]).strip()

        # Apply keyword mapping
        if isinstance(category_hints, str) and category_hints.startswith(
            "keyword_mapping:"
        ):
            try:
                mapping_json = category_hints.split(":", 1)[1]
                keyword_map = json.loads(mapping_json)

                for category, keywords in keyword_map.items():
                    if any(keyword in desc_lower for keyword in keywords):
                        return category.title()
            except:
                pass

        # Check for category column in row
        for col in ["category", "type", "class"]:
            if col in row.index and pd.notna(row[col]):
                return str(row[col]).strip()

        # Default categorization based on common keywords
        if any(
            word in desc_lower
            for word in ["restaurant", "food", "cafe", "dinner", "lunch"]
        ):
            return "Food & Dining"
        elif any(
            word in desc_lower for word in ["fuel", "gas", "parking", "uber", "taxi"]
        ):
            return "Transportation"
        elif any(word in desc_lower for word in ["grocery", "market", "supermarket"]):
            return "Groceries"
        elif any(word in desc_lower for word in ["amazon", "store", "shop"]):
            return "Shopping"
        elif any(
            word in desc_lower
            for word in ["electric", "water", "internet", "phone", "utility"]
        ):
            return "Utilities"
        else:
            return "Other"

    def _extract_mapping_from_text(self, text: str) -> dict:
        """Extract mapping information from malformed JSON text."""
        mapping = {}

        # Try to extract key patterns even if JSON is malformed
        patterns = {
            "date_column": r'"date_column"\s*:\s*"([^"]*)"',
            "description_column": r'"description_column"\s*:\s*"([^"]*)"',
            "amount_logic": r'"amount_logic"\s*:\s*"([^"]*)"',
            "merchant_extraction": r'"merchant_extraction"\s*:\s*"([^"]*)"',
            "category_hints": r'"category_hints"\s*:\s*"([^"]*)"',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                mapping[key] = match.group(1)

        return mapping if mapping else None
