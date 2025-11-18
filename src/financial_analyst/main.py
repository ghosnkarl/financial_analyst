#!/usr/bin/env python
import sys
import warnings

from financial_analyst.crew import FinancialAnalyst

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    """
    Run the crew.
    """
    import subprocess
    import os

    # Get CSV path from command line or use default
    csv_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "src/financial_analyst/data/transactions.csv"
    )

    inputs = {"csv_path": csv_path, "file_path": "output/transactions.json"}

    try:
        FinancialAnalyst().crew().kickoff(inputs=inputs)

        # Execute the generated visualization script if it exists
        chart_script = "output/generate_charts.py"
        if os.path.exists(chart_script):
            print("\n" + "=" * 80)
            print("Executing visualization script...")
            print("=" * 80 + "\n")
            result = subprocess.run(
                ["python", chart_script], capture_output=True, text=True
            )
            if result.returncode == 0:
                print("✓ Visualizations created successfully!")
                print(result.stdout)
            else:
                print("✗ Error creating visualizations:")
                print(result.stderr)

    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")
