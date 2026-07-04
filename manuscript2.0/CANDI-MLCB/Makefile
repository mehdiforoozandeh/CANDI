# CANDI MLCB submission: main paper + standalone supplement (two PDFs).
#
# The two documents cross-reference each other (main -> Suppl. Fig S1/S2,
# supplement -> Fig 4) via the xr package, which reads the other document's
# .aux file. So the aux files must exist before refs resolve: build main,
# then supplement, then main again.
#
#   make            -> build/main.pdf and build/supplementary.pdf
#   make main       -> main paper only
#   make supp       -> supplement only
#   make clean      -> remove build artifacts

OUT   = build
LMK   = latexmk -interaction=nonstopmode -file-line-error -pdf -outdir=$(OUT)

.PHONY: all main supp clean
all:
	$(LMK) main.tex
	$(LMK) supplementary.tex
	$(LMK) main.tex

main:
	$(LMK) main.tex

supp:
	$(LMK) supplementary.tex

clean:
	latexmk -C -outdir=$(OUT) main.tex supplementary.tex
