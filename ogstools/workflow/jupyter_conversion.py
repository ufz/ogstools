import os
import sys
import tempfile
import warnings
from pathlib import Path

from nbconvert import HTMLExporter
from nbconvert.preprocessors import TagRemovePreprocessor
from traitlets.config import Config

# suppress output of version incompatibility between sphinx-gallery and jupytext
# it works anyways
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    sys.stdout = open(os.devnull, "w")  # noqa: SIM115, PTH123
    import jupytext

    sys.stdout = sys.__stdout__
import papermill as pm


def jupyter_to_html(input_path: Path, show_input: bool = False) -> str:
    """
    Converts a Jupyter notebook to HTML format.

    :param input_path:  The path to the Jupyter notebook file.
    :param show_input:  If True, includes input cells in the HTML output.
                        Defaults to False, hiding input cells.

    returns:    The generated HTML representation of the Jupyter notebook.
    """
    conf = Config()
    conf.TagRemovePreprocessor.remove_cell_tags = ("remove_cell",)
    hide_input_tags = ["injected-parameters"]
    if not show_input:
        hide_input_tags += ["remove_input"]
    conf.TagRemovePreprocessor.remove_input_tags = hide_input_tags
    exporter = HTMLExporter(config=conf)
    exporter.register_preprocessor(TagRemovePreprocessor(config=conf), True)
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    report_html, _ = exporter.from_filename(str(input_path))
    return report_html


def jupytext_to_jupyter(
    template_path: Path,
    output_name: Path,
    params: dict,
    prepare_only: bool = False,
    show_progress: bool = False,
) -> None:
    """
    Convert a Jupytext-formatted notebook to a Jupyter notebook and execute it.

    :param template_path:   The path to the Jupytext-formatted notebook.
    :param output_name:     The desired path for the resulting Jupyter notebook.
    :param params:          Parameters passed to the notebook during execution.
    :param prepare_only:    If True, don't execute the notebook.
                            Defaults to False, executing the notebook.
    :param show_progress:   If True, display a progress bar during execution.
                            Defaults to False.

    Returns:
        None: The function does not return a value, but generates the Jupyter notebook.
    """
    nb = jupytext.read(template_path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ipynb") as temp:
        jupytext.write(nb, temp.name)
        pm.execute_notebook(
            input_path=temp.name,
            output_path=output_name,
            parameters=params,
            prepare_only=prepare_only,
            progress_bar=show_progress,
        )
        Path(temp.name).unlink()
