import json
import os
from typing import Type

import numpy as np
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class CsvAnalyzerToolInput(BaseModel):
    """Input schema for CsvAnalyzerTool."""

    csv_path: str = Field(
        default="output/transactions.json",
        description="Path to the transactions JSON file (default: output/transactions.json).",
    )


class CsvAnalyzerTool(BaseTool):
    name: str = "csv_analyzer_tool"
    description: str = (
        "Analyzes the processed transactions JSON file (output/transactions.json). "
        "Provides comprehensive analysis including: totals (expenses and income), "
        "averages and trends, grouping by category and merchant, recurring transactions, "
        "and anomaly detection. The JSON file contains standardized transactions with "
        "date, description, amount, merchant, and category fields."
    )
    args_schema: Type[BaseModel] = CsvAnalyzerToolInput

    def _run(self, csv_path: str = "output/transactions.json") -> str:
        """
        Analyze the transactions JSON file.

        Args:
            csv_path: Path to the JSON file (default: output/transactions.json)

        Returns:
            Comprehensive formatted analysis results as a markdown string
        """
        try:
            # Load and validate JSON file
            df = self._load_transactions_json(csv_path)
            col_map = {
                "amount_col": "amount",
                "date_col": "date",
                "category_col": "category",
                "merchant_col": "merchant",
                "description_col": "description",
            }
            return self._comprehensive_analysis(df, col_map)

        except Exception as e:
            return f"Error analyzing transactions: {str(e)}"

    def _load_transactions_json(self, file_path: str) -> pd.DataFrame:
        """Load and validate transactions JSON file. Returns DataFrame."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Transactions file not found: {file_path}")

        # Load JSON file
        with open(file_path, "r") as f:
            data = json.load(f)

        # Convert to DataFrame
        if not isinstance(data, list):
            raise ValueError("JSON file must contain an array of transactions")

        if len(data) == 0:
            raise ValueError("Transactions file is empty")

        df = pd.DataFrame(data)

        # Validate expected columns exist
        required_columns = ["date", "description", "amount", "merchant", "category"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        # Convert data types
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

        return df

    def _comprehensive_analysis(self, df: pd.DataFrame, col_map: dict) -> str:
        """Perform comprehensive analysis combining all methods."""
        results = []
        results.append("# Financial Transaction Analysis Report\n")

        # 1. Summary Statistics
        results.append(self._calculate_summary_stats(df, col_map))
        results.append("\n")

        # 2. Trends
        results.append(self._analyze_trends(df, col_map))
        results.append("\n")

        # 3. Category & Merchant Breakdown
        results.append(self._group_analysis(df, col_map))
        results.append("\n")

        # 4. Recurring Transactions
        results.append(self._identify_recurring_transactions(df, col_map))
        results.append("\n")

        # 5. Anomalies
        results.append(self._detect_anomalies(df, col_map))
        results.append("\n")

        return "\n".join(results)

    def _calculate_summary_stats(self, df: pd.DataFrame, col_map: dict) -> str:
        """Calculate totals, averages, and basic statistics."""
        results = ["## Summary Statistics\n"]

        if not col_map["amount_col"]:
            return "No amount column found in data."

        amounts = df[col_map["amount_col"]].dropna()

        results.append(f"- **Total Transactions**: {len(df):,}")
        results.append(f"- **Total Amount**: ${amounts.sum():,.2f}")
        results.append(f"- **Average Transaction**: ${amounts.mean():,.2f}")
        results.append(f"- **Median Transaction**: ${amounts.median():,.2f}")
        results.append(f"- **Min Transaction**: ${amounts.min():,.2f}")
        results.append(f"- **Max Transaction**: ${amounts.max():,.2f}")
        results.append(f"- **Standard Deviation**: ${amounts.std():,.2f}")

        # Separate income vs expenses if amounts can be negative
        if (amounts < 0).any():
            expenses = amounts[amounts < 0]
            income = amounts[amounts > 0]
            results.append(f"\n- **Total Expenses**: ${abs(expenses.sum()):,.2f}")
            results.append(f"- **Total Income**: ${income.sum():,.2f}")
            results.append(f"- **Net**: ${amounts.sum():,.2f}")

        return "\n".join(results)

    def _group_analysis(self, df: pd.DataFrame, col_map: dict) -> str:
        """Group transactions by category, date, and merchant."""
        results = ["## Category & Merchant Breakdown\n"]

        if not col_map["amount_col"]:
            return "No amount column found for grouping."

        # Group by category first
        if col_map["category_col"]:
            results.append("### By Category\n")
            category_groups = df.groupby(col_map["category_col"])[
                col_map["amount_col"]
            ].agg(["sum", "count"])
            category_groups = category_groups.sort_values("sum", ascending=False)
            for category, row in category_groups.iterrows():
                results.append(
                    f"- **{category}**: ${row['sum']:,.2f} ({int(row['count'])} transactions)"
                )

        # Group by merchant
        if col_map["merchant_col"]:
            results.append("\n### By Merchant (Top 10)\n")
            merchant_groups = df.groupby(col_map["merchant_col"])[
                col_map["amount_col"]
            ].agg(["sum", "count"])
            merchant_groups = merchant_groups.sort_values("sum", ascending=False)
            for merchant, row in merchant_groups.head(10).iterrows():
                results.append(
                    f"- **{merchant}**: ${row['sum']:,.2f} ({int(row['count'])} transactions)"
                )

        # Group by date (monthly)
        if col_map["date_col"]:
            results.append("\n### By Month\n")
            df_copy = df.copy()
            df_copy["month"] = df_copy[col_map["date_col"]].dt.to_period("M")
            monthly_groups = df_copy.groupby("month")[col_map["amount_col"]].agg(
                ["sum", "count"]
            )
            for month, row in monthly_groups.iterrows():
                results.append(
                    f"- **{month}**: ${row['sum']:,.2f} ({int(row['count'])} transactions)"
                )

        return "\n".join(results)

    def _identify_recurring_transactions(self, df: pd.DataFrame, col_map: dict) -> str:
        """Identify recurring transactions based on amount and merchant patterns."""
        results = ["## Recurring Transactions\n"]

        if not col_map["amount_col"] or not col_map["date_col"]:
            return (
                "Need both amount and date columns to identify recurring transactions."
            )

        # Sort by date
        df_sorted = df.sort_values(col_map["date_col"]).copy()

        # Group by merchant or description
        group_col = col_map["merchant_col"] or col_map["description_col"]

        if not group_col:
            return "Need merchant or description column to identify recurring transactions."

        recurring = []

        for name, group in df_sorted.groupby(group_col):
            if len(group) >= 3:  # At least 3 occurrences
                amounts = group[col_map["amount_col"]].values
                dates = group[col_map["date_col"]].values

                # Check if amounts are similar (within 10%)
                amount_std = np.std(amounts)
                amount_mean = np.mean(amounts)

                if amount_mean > 0 and amount_std / amount_mean < 0.1:
                    # Check date intervals
                    intervals = []
                    for i in range(1, len(dates)):
                        if pd.notna(dates[i]) and pd.notna(dates[i - 1]):
                            delta = (dates[i] - dates[i - 1]) / np.timedelta64(1, "D")
                            intervals.append(int(delta))

                    if intervals:
                        avg_interval = np.mean(intervals)
                        interval_std = np.std(intervals)

                        # If intervals are consistent (within 7 days)
                        if (
                            interval_std < 7
                            or len(set([i // 7 for i in intervals if i > 0])) == 1
                        ):
                            frequency = (
                                "weekly"
                                if 5 <= avg_interval <= 9
                                else (
                                    "bi-weekly"
                                    if 12 <= avg_interval <= 16
                                    else (
                                        "monthly"
                                        if 28 <= avg_interval <= 33
                                        else f"every {int(avg_interval)} days"
                                    )
                                )
                            )

                            recurring.append(
                                {
                                    "name": name,
                                    "amount": amount_mean,
                                    "count": len(group),
                                    "frequency": frequency,
                                }
                            )

        if recurring:
            recurring = sorted(recurring, key=lambda x: x["amount"], reverse=True)
            for item in recurring[:15]:
                results.append(
                    f"- **{item['name']}**: ${item['amount']:,.2f} ({item['frequency']}, {item['count']} occurrences)"
                )
        else:
            results.append("No recurring transactions detected.")

        return "\n".join(results)

    def _detect_anomalies(self, df: pd.DataFrame, col_map: dict) -> str:
        """Detect anomalies and unusual spending patterns."""
        results = ["## Anomaly Detection\n"]

        if not col_map["amount_col"]:
            return "No amount column found for anomaly detection."

        amounts = df[col_map["amount_col"]].dropna()

        # Use IQR method for outliers
        Q1 = amounts.quantile(0.25)
        Q3 = amounts.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[
            (df[col_map["amount_col"]] < lower_bound)
            | (df[col_map["amount_col"]] > upper_bound)
        ]

        if len(outliers) > 0:
            results.append(f"Found **{len(outliers)} unusual transactions**:\n")

            # Sort by absolute amount
            outliers_sorted = outliers.sort_values(
                col_map["amount_col"], ascending=False
            )

            for _, row in outliers_sorted.head(10).iterrows():
                amount = row[col_map["amount_col"]]
                details = []

                if col_map["merchant_col"] and pd.notna(row[col_map["merchant_col"]]):
                    details.append(f"{row[col_map['merchant_col']]}")
                if col_map["category_col"] and pd.notna(row[col_map["category_col"]]):
                    details.append(f"{row[col_map['category_col']]}")
                if col_map["date_col"] and pd.notna(row[col_map["date_col"]]):
                    details.append(f"{row[col_map['date_col']].strftime('%Y-%m-%d')}")

                detail_str = " | ".join(details) if details else "No additional details"
                results.append(f"- **${amount:,.2f}** - {detail_str}")
        else:
            results.append("No anomalies detected.")

        # Check for unusual spending velocity (if dates available)
        if col_map["date_col"]:
            df_sorted = df.sort_values(col_map["date_col"]).copy()
            df_sorted["date_only"] = df_sorted[col_map["date_col"]].dt.date
            daily_spending = df_sorted.groupby("date_only")[col_map["amount_col"]].sum()

            daily_mean = daily_spending.mean()
            daily_std = daily_spending.std()

            high_spending_days = daily_spending[
                daily_spending > daily_mean + 2 * daily_std
            ]

            if len(high_spending_days) > 0:
                results.append(f"\n### High Spending Days\n")
                for date, amount in high_spending_days.head(5).items():
                    results.append(
                        f"- **{date}**: ${amount:,.2f} (avg: ${daily_mean:,.2f})"
                    )

        return "\n".join(results)

    def _analyze_trends(self, df: pd.DataFrame, col_map: dict) -> str:
        """Analyze spending trends over time."""
        results = ["## Spending Trends\n"]

        if not col_map["amount_col"] or not col_map["date_col"]:
            return "Need both amount and date columns for trend analysis."

        df_sorted = df.sort_values(col_map["date_col"]).copy()
        df_sorted["month"] = df_sorted[col_map["date_col"]].dt.to_period("M")

        monthly_totals = df_sorted.groupby("month")[col_map["amount_col"]].sum()

        if len(monthly_totals) < 2:
            return "Not enough data for trend analysis (need at least 2 months)."

        results.append("### Monthly Spending\n")
        for month, total in monthly_totals.items():
            results.append(f"- **{month}**: ${total:,.2f}")

        # Calculate trend
        values = monthly_totals.values
        if len(values) >= 2:
            recent_avg = np.mean(values[-3:]) if len(values) >= 3 else values[-1]
            earlier_avg = np.mean(values[:3]) if len(values) >= 6 else values[0]

            change = (
                ((recent_avg - earlier_avg) / earlier_avg * 100)
                if earlier_avg != 0
                else 0
            )

            results.append(
                f"\n**Trend**: {'Increasing' if change > 5 else 'Decreasing' if change < -5 else 'Stable'}"
            )
            results.append(f"**Change**: {change:+.1f}%")

        # Category trends if available
        if col_map["category_col"]:
            results.append("\n### Top Growing Categories\n")
            category_monthly = (
                df_sorted.groupby([df_sorted["month"], col_map["category_col"]])[
                    col_map["amount_col"]
                ]
                .sum()
                .unstack(fill_value=0)
            )

            if len(category_monthly) >= 2:
                recent = category_monthly.iloc[-1]
                earlier = category_monthly.iloc[0]
                growth = ((recent - earlier) / earlier.replace(0, 1) * 100).sort_values(
                    ascending=False
                )

                for cat, pct in growth.head(5).items():
                    if recent[cat] > 0:
                        results.append(
                            f"- **{cat}**: {pct:+.1f}% (${recent[cat]:,.2f})"
                        )

        return "\n".join(results)
