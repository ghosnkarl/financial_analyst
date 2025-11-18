import json
import os
from typing import Type

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class VisualizationToolInput(BaseModel):
    """Input schema for VisualizationTool."""

    json_path: str = Field(
        default="output/transactions.json",
        description="Path to the transactions JSON file (default: output/transactions.json).",
    )


class VisualizationTool(BaseTool):
    name: str = "visualization_tool"
    description: str = (
        "Generates financial visualization charts from the processed transactions JSON file. "
        "Creates 4 PNG charts: spending by category, spending over time, top expenses, and income vs expenses. "
        "All charts are saved to the output directory with high quality (300 DPI)."
    )
    args_schema: Type[BaseModel] = VisualizationToolInput

    def _run(self, json_path: str = "output/transactions.json") -> str:
        """
        Generate visualization charts from transactions JSON.

        Args:
            json_path: Path to the JSON file (default: output/transactions.json)

        Returns:
            Success message with list of generated charts
        """
        errors = []

        try:
            # Load the transactions data
            with open(json_path, "r") as f:
                transactions = json.load(f)

            if not transactions:
                return "Error: Transactions file is empty"

            # Convert to DataFrame
            df = pd.DataFrame(transactions)

            # Ensure date column is datetime
            df["date"] = pd.to_datetime(df["date"])

            # Create output directory if it doesn't exist
            os.makedirs("output", exist_ok=True)

            charts_created = []

            # Normalize the type column if it exists
            if "type" in df.columns:
                df["type"] = df["type"].str.lower()
            else:
                # Fallback: Infer type from amount if not present
                df["type"] = df["amount"].apply(
                    lambda x: "expense" if x >= 0 else "income"
                )

            # Ensure amounts are positive for display
            df["amount"] = df["amount"].abs()

            # Chart 1: Spending by category
            try:
                expense_data = df[df["type"] == "expense"]
                if not expense_data.empty and "category" in df.columns:
                    category_spending = (
                        expense_data.groupby("category")["amount"]
                        .sum()
                        .sort_values(ascending=False)
                    )
                    plt.figure(figsize=(10, 6))
                    category_spending.plot(kind="bar", color="steelblue")
                    plt.title("Spending by Category", fontsize=14, fontweight="bold")
                    plt.xlabel("Category", fontsize=12)
                    plt.ylabel("Amount ($)", fontsize=12)
                    plt.xticks(rotation=45, ha="right")
                    plt.grid(axis="y", alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(
                        "output/spending_by_category.png", dpi=300, bbox_inches="tight"
                    )
                    plt.close()
                    charts_created.append("spending_by_category.png")
            except Exception as e:
                error_msg = f"Error creating spending by category chart: {e}"
                print(error_msg)
                errors.append(error_msg)

            # Chart 2: Spending over time
            try:
                expense_data = df[df["type"] == "expense"]
                if not expense_data.empty:
                    df["month"] = df["date"].dt.to_period("M")
                    monthly_spending = expense_data.groupby(
                        expense_data["date"].dt.to_period("M")
                    )["amount"].sum()
                    plt.figure(figsize=(12, 6))
                    monthly_spending.plot(
                        kind="line",
                        marker="o",
                        linewidth=2,
                        markersize=8,
                        color="steelblue",
                    )
                    plt.title("Spending Over Time", fontsize=14, fontweight="bold")
                    plt.xlabel("Month", fontsize=12)
                    plt.ylabel("Amount ($)", fontsize=12)
                    plt.xticks(rotation=45)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(
                        "output/spending_over_time.png", dpi=300, bbox_inches="tight"
                    )
                    plt.close()
                    charts_created.append("spending_over_time.png")
            except Exception as e:
                error_msg = f"Error creating spending over time chart: {e}"
                print(error_msg)
                errors.append(error_msg)

            # Chart 3: Top expenses
            try:
                expense_data = df[df["type"] == "expense"]
                if not expense_data.empty and "description" in df.columns:
                    top_expenses = expense_data.nlargest(10, "amount")
                    plt.figure(figsize=(10, 6))
                    plt.barh(
                        range(len(top_expenses)), top_expenses["amount"], color="coral"
                    )
                    plt.yticks(
                        range(len(top_expenses)),
                        top_expenses["description"],
                        fontsize=10,
                    )
                    plt.title("Top 10 Expenses", fontsize=14, fontweight="bold")
                    plt.xlabel("Amount ($)", fontsize=12)
                    plt.ylabel("Description", fontsize=12)
                    plt.grid(axis="x", alpha=0.3)
                    plt.tight_layout()
                    plt.savefig("output/top_expenses.png", dpi=300, bbox_inches="tight")
                    plt.close()
                    charts_created.append("top_expenses.png")
            except Exception as e:
                error_msg = f"Error creating top expenses chart: {e}"
                print(error_msg)
                errors.append(error_msg)

            # Chart 4: Total spending summary (since most data is expenses only)
            try:
                if not df.empty:
                    type_totals = df.groupby("type")["amount"].sum()

                    plt.figure(figsize=(8, 6))
                    colors = [
                        "green" if t == "income" else "crimson"
                        for t in type_totals.index
                    ]
                    type_totals.plot(kind="bar", color=colors, width=0.6)
                    plt.title("Income vs Expenses", fontsize=14, fontweight="bold")
                    plt.xlabel("Type", fontsize=12)
                    plt.ylabel("Amount ($)", fontsize=12)
                    plt.xticks(rotation=0)
                    plt.grid(axis="y", alpha=0.3)

                    # Add value labels on bars
                    for i, v in enumerate(type_totals.values):
                        plt.text(
                            i, v, f"${v:,.2f}", ha="center", va="bottom", fontsize=10
                        )

                    plt.tight_layout()
                    plt.savefig(
                        "output/income_vs_expenses.png", dpi=300, bbox_inches="tight"
                    )
                    plt.close()
                    charts_created.append("income_vs_expenses.png")
            except Exception as e:
                error_msg = f"Error creating income vs expenses chart: {e}"
                print(error_msg)
                errors.append(error_msg)

            if charts_created:
                result = f"Successfully created {len(charts_created)} charts: {', '.join(charts_created)}"
                if errors:
                    result += f"\n\nErrors encountered: {'; '.join(errors)}"
                return result
            else:
                return f"No charts were created. Errors: {'; '.join(errors) if errors else 'Unknown error'}"

        except FileNotFoundError:
            return f"Error: Transactions file not found at {json_path}"
        except json.JSONDecodeError:
            return f"Error: Invalid JSON format in {json_path}"
        except Exception as e:
            import traceback

            return f"Error generating visualizations: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
