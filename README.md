# Financial Analyst - AI-Powered Transaction Analysis

An automated financial analysis system using CrewAI that processes transaction data, generates visualizations, and creates comprehensive financial reports.

## Features

- **Transaction Processing**: Automatically standardizes transaction data from CSV files
- **Visual Analytics**: Generates 4 financial visualization charts:
  - Spending by category (pie chart)
  - Spending trends over time (line chart)
  - Top expenses (bar chart)
  - Monthly spending analysis (bar chart)
- **Financial Report**: Creates a detailed markdown report with insights and embedded charts

## Requirements

- Python 3.10+
- CrewAI
- Pandas
- Matplotlib
- Seaborn

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install crewai pandas matplotlib seaborn
```

## Usage

Run the analysis on your transaction data:

```bash
crewai run
```

Or specify a custom CSV file:

```bash
crewai run path/to/your/transactions.csv
```

## Project Structure

```
financial_analyst/
├── src/financial_analyst/
│   ├── crew.py              # Agent and task definitions
│   ├── main.py              # Entry point
│   ├── config/
│   │   ├── agents.yaml      # Agent configurations
│   │   └── tasks.yaml       # Task descriptions
│   ├── data/
│   │   └── transactions.csv # Sample transaction data
│   └── tools/               # Custom tools for data processing
├── output/                  # Generated files (JSON, charts, reports)
└── README.md
```

## How It Works

The system uses three AI agents:

1. **Transaction Processor Agent**: Reads CSV data, standardizes format, saves to JSON
2. **Visualization Code Generator Agent**: Creates Python scripts to generate charts
3. **Data Analyst Agent**: Analyzes data and creates comprehensive markdown report

The workflow is:

1. Process transactions → `output/transactions.json`
2. Generate visualization script → `output/generate_charts.py`
3. Execute script → Create 4 PNG charts in `output/`
4. Analyze and report → `output/analyze_transactions.md`

## Output Files

- `output/transactions.json` - Standardized transaction data
- `output/generate_charts.py` - Generated visualization code
- `output/spending_by_category.png` - Category spending pie chart
- `output/spending_over_time.png` - Time series line chart
- `output/top_expenses.png` - Top transactions bar chart
- `output/income_vs_expenses.png` - Monthly spending chart
- `output/analyze_transactions.md` - Full financial analysis report

## Configuration

Edit the configuration files to customize agent behavior:

- `src/financial_analyst/config/agents.yaml` - Agent roles and goals
- `src/financial_analyst/config/tasks.yaml` - Task descriptions and requirements

## LLM Configuration

The project uses Ollama models by default:

- `phi4:14b` - Main analysis agent

Modify `src/financial_analyst/crew.py` to use different models or providers.

## Troubleshooting

**Charts not generating?**

- Check that matplotlib is installed
- Verify `output/transactions.json` exists
- Manually run `python output/generate_charts.py` to see errors

**Analysis fails?**

- Ensure your CSV has the required columns
- Check CSV encoding (UTF-8 recommended)
- Review error messages in the terminal output
