## How to Compile the Thesis

### Requirements
Install a full TeX distribution:
- **Windows:** [MiKTeX](https://miktex.org/) or [TeX Live](https://tug.org/texlive/)
- **Linux:** `sudo apt install texlive-full`
- **Mac:** [MacTeX](https://www.tug.org/mactex/)

### Quick compile (3 passes required for cross-references)

```bash
cd docs/thesis/

pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Or use latexmk (recommended — handles all passes automatically)

```bash
cd docs/thesis/
latexmk -pdf -interaction=nonstopmode main.tex
```

### Clean build artefacts

```bash
latexmk -C
```

### Output
The compiled thesis will be at `docs/thesis/main.pdf`.

---

### Placeholder figure
The title page references `figures/placeholder_logo.pdf`.
Either create a `figures/` folder with a logo PDF, or remove the
`\includegraphics` line from `main.tex` if no logo is needed.

### Filling in results
Search for `---` in the chapter files and replace with your actual
experimental results. Also replace `\todo{...}` markers.

### File structure

```
docs/thesis/
├── main.tex                  ← Master document (compile this)
├── references.bib            ← BibTeX bibliography (37 references)
├── chapters/
│   ├── ch1_introduction.tex
│   ├── ch2_literature.tex
│   ├── ch3_methodology.tex
│   ├── ch4_results.tex
│   ├── ch5_discussion.tex
│   ├── ch6_conclusion.tex
│   └── appendix.tex
└── figures/                  ← Place figure files here
```
