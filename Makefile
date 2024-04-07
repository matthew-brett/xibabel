DOC_DIR=docs
IPYNBS:=$(patsubst %.Rmd,%.ipynb,$(wildcard $(DOC_DIR)/*.Rmd))

docs: $(IPYNBS)
	jupytext --execute --to ipynb $(DOC_DIR)/*.Rmd
	mkdocs build

clean:
	rm -rf $(IPYNBS)

%.ipynb: %.Rmd
	# No references, beamer output.
	jupytext --execute --to ipynb $<
