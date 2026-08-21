REPORT := jetson_cuda_migration_report

.PHONY: all clean

all: $(REPORT).pdf

$(REPORT).pdf: $(REPORT).tex
	latexmk -xelatex -interaction=nonstopmode -halt-on-error $(REPORT).tex

clean:
	latexmk -C $(REPORT).tex
