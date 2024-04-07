DOC_DIR=docs
IPYNBS:=$(patsubst %.Rmd,%.ipynb,$(wildcard $(DOC_DIR)/*.Rmd))

docs: $(IPYNBS)
	mkdocs build

clean:
	rm -rf $(IPYNBS)

%.ipynb: %.Rmd
	jupytext --execute --to ipynb $<

gh-pages: docs
	ghp-import -fop site
