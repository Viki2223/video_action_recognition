from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, \
                          TimeRemainingColumn, MofNCompleteColumn

shared_console = Console(force_terminal=True,
                         force_interactive=True,
                         color_system="truecolor")

PROGRESS_COLUMNS = [
    TextColumn("[bold blue]{task.fields[video]}", justify="right"),
    BarColumn(bar_width=None),
    MofNCompleteColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
    TimeRemainingColumn(),
    TextColumn("{task.fields[status]}", style="dim"),
]

def create_batch_progress():
    return Progress(*PROGRESS_COLUMNS,
                    console=shared_console,
                    transient=False,
                    refresh_per_second=10,
                    expand=True)