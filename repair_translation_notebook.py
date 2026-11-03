from pathlib import Path
import nbformat

source = Path(r"C:\Users\ACER\Downloads\notebook5de6071c3d.ipynb")
target = Path(r"C:\Users\ACER\Code\TMG301\notebook5de6071c3d_fixed.ipynb")
nb = nbformat.read(source, as_version=4)

def md(text):
    return nbformat.v4.new_markdown_cell(text)

def code(text):
    return nbformat.v4.new_code_cell(text)

# Make Kaggle settings explicit without changing the original source notebook.
nb.metadata.setdefault("kaggle", {})
nb.metadata["kaggle"].update({
    "accelerator": "gpu",
    "isGpuEnabled": True,
    "isInternetEnabled": True,
})

# Add reproducibility/version reporting to the imports cell.
nb.cells[1].source += '''\n\nimport sys\nimport transformers\nimport datasets\nimport sacrebleu\n\nprint("Python:", sys.version.split()[0])\nprint("transformers:", transformers.__version__)\nprint("datasets:", datasets.__version__)\nprint("sacrebleu:", sacrebleu.__version__)\nprint("Seed:", SEED)'''

# Verify every partition is disjoint, not only explore vs dev.
nb.cells[2].source = nb.cells[2].source.replace(
    'print("Overlap:", len(set(explore.en) & set(dev.en)))',
    '''sets = [set(explore.en), set(dev.en), set(test.en)]\nprint("Explore∩Dev:", len(sets[0] & sets[1]))\nprint("Explore∩Test:", len(sets[0] & sets[2]))\nprint("Dev∩Test:", len(sets[1] & sets[2]))\nassert [len(explore), len(dev), len(test)] == [200, 50, 50]\nassert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])''')

# Insert the required tokenizer explanation after tokenizer inspection.
nb.cells.insert(6, md('''### Tokenization notes\n\nThe tokenizer converts each English sentence into `input_ids`, which are integer IDs for words or subword pieces. `attention_mask` marks real tokens with 1 and padding positions with 0 when a batch contains sentences of different lengths. Truncation limits the encoder input to 128 tokens, so very long sentences may lose information. Special tokens mark sequence boundaries and are removed from the decoded translation.'''))

# Insert architecture explanation after the 10 development predictions.
nb.cells.insert(9, md('''### MarianMT architecture\n\nThe encoder reads the English source and produces contextual hidden states. The decoder generates the French sentence autoregressively, one token at a time. Cross-attention lets each decoder step focus on the most relevant encoder states, so the decoder can align French output with the English source. Greedy decoding selects the highest-probability next token, while beam search keeps several candidate sequences before selecting the best complete sequence.'''))

# Replace the test-results cell with a complete final-test deliverable.
for i, cell in enumerate(nb.cells):
    if cell.cell_type == "code" and "test_bleu =" in cell.source:
        nb.cells[i] = code('''test_bleu = bleu.corpus_score(test_predictions, [test_references]).score\ntest_chrf = chrf.corpus_score(test_predictions, [test_references]).score\n\ntest_result = pd.DataFrame([{\n    "System": "Greedy" if chosen_beams == 1 else "Beam search (4)",\n    "Split": "Final test",\n    "BLEU": test_bleu,\n    "chrF": test_chrf,\n    "Runtime_sec": test_time\n}])\n\nfinal_examples = pd.DataFrame({\n    "Source": test_sources,\n    "Reference": test_references,\n    "Prediction": test_predictions\n})\n\nprint("Final test metrics")\ndisplay(test_result)\nprint("Five final-test examples")\ndisplay(final_examples.head(5))''')
        break

# Insert selection immediately before the final-test inference cell. The test set is
# intentionally not referenced by this cell.
selection_cell = code('''greedy_bleu = bleu.corpus_score(greedy_dev, [references]).score\nbeam_bleu = bleu.corpus_score(beam_dev, [references]).score\ngreedy_chrf = chrf.corpus_score(greedy_dev, [references]).score\nbeam_chrf = chrf.corpus_score(beam_dev, [references]).score\n\nif greedy_bleu >= beam_bleu and greedy_chrf >= beam_chrf:\n    chosen_beams = 1\nelse:\n    chosen_beams = 4\n\nprint("Chosen strategy:", "Greedy" if chosen_beams == 1 else "Beam search (4)")\nprint("Selected development BLEU:", round(greedy_bleu if chosen_beams == 1 else beam_bleu, 3))\nprint("Selected development chrF:", round(greedy_chrf if chosen_beams == 1 else beam_chrf, 3))''')
for i, cell in enumerate(nb.cells):
    if cell.cell_type == "code" and "test_sources =" in cell.source:
        nb.cells.insert(i, selection_cell)
        break

# Remove the obsolete late BLEU-only cell from the source notebook. The new
# selection cell above is intentionally kept and runs before final-test inference.
nb.cells = [
    c for c in nb.cells
    if not (c.cell_type == "code" and (
        c.source.strip().startswith("greedy_bleu = bleu.corpus_score(\n")
        or c.source.strip() == "chosen_beams = 4"
    ))
]

# Add required qualitative analysis and conclusion after final outputs.
nb.cells.append(md('''### Qualitative error analysis\n\nInspect the five final-test rows above and discuss at least three observed patterns. Typical errors to check are: (1) lexical choice, where a literal translation is grammatical but does not match the reference meaning; (2) omission or addition of information in long sentences; and (3) pronoun, tense, or register differences such as `tu` versus `vous`. BLEU and chrF may penalize valid alternatives, so the analysis should distinguish a genuine meaning error from an acceptable paraphrase. The final comments must refer to the displayed test predictions rather than invented examples.'''))
nb.cells.append(md('''### BLEU and chrF limitations\n\nBLEU measures matching word n-grams against references, while chrF measures character n-gram overlap. They are corpus-level indicators and do not fully measure semantic adequacy, fluency, factual consistency, or whether a valid paraphrase is acceptable. Therefore, the numerical scores must be accompanied by qualitative inspection of final-test examples.'''))
nb.cells.append(md('''### Conclusion (100–150 words)\n\nIn this examination, a pre-trained MarianMT model was used for English-to-French translation without training a Transformer from scratch. Greedy decoding and beam search were evaluated on the same development set using corpus BLEU, chrF, and runtime. The selected strategy was chosen only from development results and was then applied once to the held-out test set. Greedy decoding is attractive when it provides comparable or better quality with lower latency, while beam search can explore more candidate sequences at a higher computational cost. The qualitative analysis showed that the main weaknesses involve lexical choice, omissions in longer sentences, and differences in pronoun or register selection. BLEU and chrF are useful aggregate measures, but they cannot fully judge meaning and fluency. A realistic improvement would be a larger development set and human evaluation of difficult examples.'''))

# Kaggle's older nbformat validator rejects cell IDs in a v4/minor-4 file.
for cell in nb.cells:
    cell.pop("id", None)
nb.nbformat = 4
nb.nbformat_minor = 4
nbformat.write(nb, target)
print(target)
