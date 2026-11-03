import copy
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

source = Path(r"C:\Users\ACER\Code\TMG301\Integrated_NLP_Amazon_Reviews.ipynb")
target = Path(r"C:\Users\ACER\Code\TMG301\Integrated_NLP_Amazon_Reviews_executed.ipynb")
nb = nbformat.read(source, as_version=4)

# The local Python environment already contains the required packages.
# Keep the install cell in the student notebook, but skip reinstalling locally.
for cell in nb.cells:
    if cell.cell_type == "code" and "!pip -q install datasets" in "".join(cell.source):
        cell.source = "print('Dependencies already available in the local runtime.')\n"
        break

client = NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(source.parent)}})
client.execute()
nbformat.write(nb, target)
print(target)
