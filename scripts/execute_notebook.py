"""Execute one notebook, saving completed cells even if a later cell fails."""
import os
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

path = Path(sys.argv[1]).resolve()
root = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLBACKEND', 'module://matplotlib_inline.backend_inline')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
nb = nbformat.read(path, as_version=4)

def save_cell(cell, cell_index, **kwargs):
    nbformat.write(nb, path)
    print(f'{path.name}: completed cell {cell_index}', flush=True)
    for output in cell.get('outputs', []):
        if output.output_type == 'stream':
            print(output.text[-4000:], flush=True)

client = NotebookClient(nb, timeout=None, kernel_name='road-accident-m4',
                        resources={'metadata': {'path': str(root)}}, on_cell_executed=save_cell)
try:
    client.execute()
finally:
    nbformat.write(nb, path)
