from pathlib import Path

import nbformat
from nbclient import NotebookClient


source = Path(r"C:\Users\ACER\Code\TMG301\SST2_DistilBERT_Sentiment_Analysis.ipynb")
target = Path(r"C:\Users\ACER\Code\TMG301\SST2_DistilBERT_Sentiment_Analysis_executed.ipynb")
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
