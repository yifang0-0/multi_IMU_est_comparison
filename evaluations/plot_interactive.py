"""Generate interactive, zoomable HTML plots from time series CSV files."""

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import argparse


def plot_timeseries_interactive(csv_path, output_dir="plots/interactive"):
    """Create an interactive HTML plot from a time series CSV file."""
    df = pd.read_csv(csv_path)
    name = Path(csv_path).stem  # e.g., "Subject08_knee_vqf_olsson"

    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=df['time'], y=df['estimated_deg'],
                               name='Estimated', line=dict(color='#3498db')))
    fig.add_trace(go.Scattergl(x=df['time'], y=df['gt_deg'],
                               name='Ground Truth', line=dict(color='#2ecc71')))

    fig.update_layout(title=name, xaxis_title='Time (s)', yaxis_title='Angle (°)',
                      hovermode='x unified')

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.write_html(f"{output_dir}/{name}.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate interactive HTML plots from time series CSVs")
    parser.add_argument("--input-dir", default="results/time_series", help="Directory containing CSV files")
    parser.add_argument("--output-dir", default="plots/interactive", help="Output directory for HTML files")
    args = parser.parse_args()

    csv_files = list(Path(args.input_dir).glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    for csv in csv_files:
        plot_timeseries_interactive(csv, args.output_dir)
        print(f"Created {args.output_dir}/{csv.stem}.html")

    print(f"Done! Generated {len(csv_files)} interactive plots in {args.output_dir}/")
