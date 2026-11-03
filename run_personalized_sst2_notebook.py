from pathlib import Path

import nbformat
from nbclient import NotebookClient


source = Path(r"C:\Users\ACER\Downloads\Phan_Ba_Song_Toan_DE200247_SST2_complete.ipynb")
target = Path(r"C:\Users\ACER\Downloads\Phan_Ba_Song_Toan_DE200247_SST2_complete_executed.ipynb")
nb = nbformat.read(source, as_version=4)
client = NotebookClient(
    nb,
    timeout=3600,
    kernel_name="python3",
    resources={"metadata": {"path": str(source.parent)}},
)
client.execute()
nbformat.write(nb, target)
print(target)
