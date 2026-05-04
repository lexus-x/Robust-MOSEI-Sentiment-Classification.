#!/usr/bin/env python
"""Generate a premium interactive dashboard (Graphifyy) from experiment results."""

import argparse
import json
from pathlib import Path
import pandas as pd
import sys

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="outputs/main_run/aggregate_results.csv")
    parser.add_argument("--summary", default="outputs/main_run/run_summary.csv")
    parser.add_argument("--acceptance", default="outputs/main_run/acceptance_summary.json")
    parser.add_argument("--output", default="outputs/main_run/dashboard.html")
    return parser.parse_args()

def generate_html(results_df, summary_df, acceptance_data):
    # Prepare data for Chart.js
    # 1. Condition Comparison (Robust vs Vanilla)
    conditions = ["clean", "missing_audio", "missing_vision", "missing_audio_vision", "mild_jitter"]
    
    def get_mean_f1(df, experiment):
        subset = df[df["experiment"] == experiment]
        # Ensure we convert to list of floats (not numpy types) for JS compatibility
        return [float(subset[subset["condition"] == c]["weighted_f1"].mean()) for c in conditions]

    vanilla_f1 = get_mean_f1(results_df, "xmodal_transformer")
    robust_f1 = get_mean_f1(results_df, "xmodal_transformer_robust")

    # 2. Ablation Study (Seed 13)
    ablation_experiments = ["xmodal_transformer_robust", "minus_gating", "minus_modality_dropout", "minus_jitter_augmentation"]
    ablation_labels = ["Full Robust", "No Gating", "No Dropout", "No Jitter Aug"]
    ablation_subset = summary_df[summary_df["seed"] == 13].set_index("experiment")
    
    ablation_clean = [float(ablation_subset.loc[e, "clean_weighted_f1"]) for e in ablation_experiments]
    ablation_perturbed = [float(ablation_subset.loc[e, "avg_perturbed_weighted_f1"]) for e in ablation_experiments]

    # Verdict Class
    verdict_class = "verdict-pass" if acceptance_data.get("meets_perturbed_direction_criterion") else "verdict-fail"
    verdict_text = "HYPOTHESIS SUPPORTED" if acceptance_data.get("meets_perturbed_direction_criterion") else "HYPOTHESIS REJECTED"
    if not acceptance_data.get("meets_clean_criterion_for_all_seeds"):
        verdict_text += " (WITH TRADEOFF)"
        verdict_class = "verdict-warning"

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Graphifyy | MultiMod Robustness Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg: #030712;
            --card-bg: rgba(17, 24, 39, 0.7);
            --border: rgba(255, 255, 255, 0.1);
            --accent-teal: #2dd4bf;
            --accent-indigo: #818cf8;
            --text-main: #f9fafb;
            --text-muted: #9ca3af;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            background-color: var(--bg);
            background-image: 
                radial-gradient(circle at 0% 0%, rgba(45, 212, 191, 0.05) 0%, transparent 50%),
                radial-gradient(circle at 100% 100%, rgba(129, 140, 248, 0.05) 0%, transparent 50%);
            color: var(--text-main);
            font-family: 'Outfit', sans-serif;
            line-height: 1.5;
            min-height: 100vh;
            padding: 2rem;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 3rem;
        }}

        h1 {{
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(to right, var(--accent-teal), var(--accent-indigo));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
        }}

        .verdict-badge {{
            padding: 0.5rem 1.5rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.875rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            border: 1px solid var(--border);
        }}

        .verdict-pass {{ background: rgba(16, 185, 129, 0.1); color: var(--success); border-color: rgba(16, 185, 129, 0.2); }}
        .verdict-warning {{ background: rgba(245, 158, 11, 0.1); color: var(--warning); border-color: rgba(245, 158, 11, 0.2); }}
        .verdict-fail {{ background: rgba(239, 68, 68, 0.1); color: var(--danger); border-color: rgba(239, 68, 68, 0.2); }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 2rem;
            margin-bottom: 2rem;
        }}

        .card {{
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border);
            border-radius: 1.5rem;
            padding: 2rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            transition: transform 0.3s ease, border-color 0.3s ease;
        }}

        .card:hover {{
            transform: translateY(-4px);
            border-color: rgba(255, 255, 255, 0.2);
        }}

        .card h2 {{
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            color: var(--accent-teal);
        }}

        .full-width {{
            grid-column: span 2;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        .stat-item {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border);
            border-radius: 1rem;
            padding: 1.5rem;
            text-align: center;
        }}

        .stat-value {{
            font-size: 1.75rem;
            font-weight: 800;
            display: block;
            margin-bottom: 0.25rem;
        }}

        .stat-label {{
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }}

        canvas {{
            width: 100% !important;
            height: 350px !important;
        }}

        .footer {{
            margin-top: 4rem;
            text-align: center;
            color: var(--text-muted);
            font-size: 0.875rem;
        }}

        @media (max-width: 900px) {{
            .grid {{ grid-template-columns: 1fr; }}
            .full-width {{ grid-column: span 1; }}
            .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Graphifyy.</h1>
            <div class="verdict-badge {verdict_class}">{verdict_text}</div>
        </header>

        <div class="stats-grid">
            <div class="stat-item">
                <span class="stat-value">{summary_df[summary_df['experiment']=='xmodal_transformer_robust']['clean_weighted_f1'].mean():.4f}</span>
                <span class="stat-label">Robust Mean F1 (Clean)</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">{summary_df[summary_df['experiment']=='xmodal_transformer_robust']['avg_perturbed_weighted_f1'].mean():.4f}</span>
                <span class="stat-label">Robust Mean F1 (Perturbed)</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">+{acceptance_data['seeds_with_better_avg_perturbed_f1']}/3</span>
                <span class="stat-label">Seeds Better on Stress</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">{len(results_df)}</span>
                <span class="stat-label">Total Exp Data Points</span>
            </div>
        </div>

        <div class="grid">
            <div class="card full-width">
                <h2>Transformer Pair Performance by Condition</h2>
                <canvas id="conditionChart"></canvas>
            </div>

            <div class="card">
                <h2>Ablation Analysis (Seed 13)</h2>
                <canvas id="ablationChart"></canvas>
            </div>

            <div class="card">
                <h2>Robustness Gain Breakdown</h2>
                <canvas id="gainChart"></canvas>
            </div>
        </div>

        <div class="footer">
            Generated by Antigravity | Robust MOSEI Project Analysis
        </div>
    </div>

    <script>
        const ctxCondition = document.getElementById('conditionChart').getContext('2d');
        new Chart(ctxCondition, {{
            type: 'bar',
            data: {{
                labels: ['Clean', 'Missing Audio', 'Missing Vision', 'No Audio+Vision', 'Mild Jitter'],
                datasets: [
                    {{
                        label: 'Vanilla Transformer',
                        data: {vanilla_f1},
                        backgroundColor: 'rgba(156, 163, 175, 0.5)',
                        borderColor: '#9ca3af',
                        borderWidth: 1
                    }},
                    {{
                        label: 'Robust Transformer',
                        data: {robust_f1},
                        backgroundColor: 'rgba(45, 212, 191, 0.6)',
                        borderColor: '#2dd4bf',
                        borderWidth: 1
                    }}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ labels: {{ color: '#f9fafb', font: {{ family: 'Outfit' }} }} }}
                }},
                scales: {{
                    y: {{ 
                        min: 0.5,
                        grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                        ticks: {{ color: '#9ca3af' }}
                    }},
                    x: {{ 
                        grid: {{ display: false }},
                        ticks: {{ color: '#9ca3af' }}
                    }}
                }}
            }}
        }});

        const ctxAblation = document.getElementById('ablationChart').getContext('2d');
        new Chart(ctxAblation, {{
            type: 'line',
            data: {{
                labels: {ablation_labels},
                datasets: [
                    {{
                        label: 'Clean F1',
                        data: {ablation_clean},
                        borderColor: '#818cf8',
                        tension: 0.4,
                        fill: false
                    }},
                    {{
                        label: 'Avg Perturbed F1',
                        data: {ablation_perturbed},
                        borderColor: '#2dd4bf',
                        tension: 0.4,
                        fill: false
                    }}
                ]
            }},
            options: {{
                plugins: {{
                    legend: {{ labels: {{ color: '#f9fafb', font: {{ family: 'Outfit' }} }} }}
                }},
                scales: {{
                    y: {{ 
                        grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                        ticks: {{ color: '#9ca3af' }}
                    }},
                    x: {{ ticks: {{ color: '#9ca3af' }} }}
                }}
            }}
        }});

        const ctxGain = document.getElementById('gainChart').getContext('2d');
        const gains = {robust_f1}.map((v, i) => v - {vanilla_f1}[i]);
        new Chart(ctxGain, {{
            type: 'doughnut',
            data: {{
                labels: ['Clean Gain', 'Audio Gain', 'Vision Gain', 'Dual Gain', 'Jitter Gain'],
                datasets: [{{
                    data: gains.map(g => Math.abs(g)),
                    backgroundColor: [
                        'rgba(240, 45, 45, 0.4)',
                        'rgba(45, 212, 191, 0.6)',
                        'rgba(129, 140, 248, 0.6)',
                        'rgba(16, 185, 129, 0.6)',
                        'rgba(245, 158, 11, 0.6)'
                    ],
                    borderColor: 'rgba(255, 255, 255, 0.1)'
                }}]
            }},
            options: {{
                plugins: {{
                    legend: {{ position: 'right', labels: {{ color: '#f9fafb', font: {{ family: 'Outfit' }} }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    return html_template

def main():
    args = parse_args()
    
    # Load data
    results_df = pd.read_csv(args.results)
    summary_df = pd.read_csv(args.summary)
    with open(args.acceptance, 'r') as f:
        acceptance_data = json.load(f)
        
    # Generate HTML
    html_content = generate_html(results_df, summary_df, acceptance_data)
    
    # Write to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding='utf-8')
    
    print(f"WOW! Graphifyy dashboard generated at: {output_path}")

if __name__ == "__main__":
    main()
