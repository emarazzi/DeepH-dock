import click
from pathlib import Path
from deepx_dock._cli.registry import register


@register(
    cli_name="features",
    cli_help="Analyze DeepH graph file and extract training-relevant features.",
    cli_args=[
        click.argument(
            "graph_path",
            type=click.Path(exists=True),
        ),
        click.option(
            "--output",
            "-o",
            type=click.Path(),
            default=None,
            help="Output JSON file path for analysis results.",
        ),
        click.option(
            "--consider-parity",
            is_flag=True,
            help="Consider parity when calculating irreps.",
        ),
        click.option(
            "--quiet",
            "-q",
            is_flag=True,
            help="Only output JSON, suppress console output.",
        ),
    ],
)
def analyze_graph_features(
    graph_path: str | Path,
    output: str | Path | None,
    consider_parity: bool,
    quiet: bool,
):
    """Analyze DeepH graph file and extract training-relevant features.

    GRAPH_PATH: Path to graph file (.npz for memory storage) or directory
    containing disk storage files (.db + .npz).

    This command extracts:
    - Graph type and storage type
    - Elements and orbital types
    - Structure statistics (nodes, edges, entries)
    - Recommended batch_size based on edge constraint
    - Suggested irreps for model configuration

    Examples:
        dock analyze graph features ./graph/data.memory.npz
        dock analyze graph features ./graph/ -o analysis.json
    """
    from deepx_dock.analyze.graph.analyze_graph import analyze_graph

    analyze_graph(
        graph_path=Path(graph_path),
        output=Path(output) if output else None,
        consider_parity=consider_parity,
        quiet=quiet,
    )
