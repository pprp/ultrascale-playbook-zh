
# tlmgr install soul adjustbox babel-german background bidi collectbox csquotes everypage filehook footmisc footnotebackref framed fvextra letltxmacro ly1 mdframed mweights needspace pagecolor sourcecodepro sourcesanspro titling ucharcat unicode-math upquote xecjk xurl zref
export PATH=$PATH:/usr/local/texlive/2024basic/bin/universal-darwin
export PATH=$PATH:/Library/TeX/texbin
pandoc metadata.yaml "docs/The UltraScale Playbook.md" \
     -o ultrascale-playbook-zh.pdf \
     --pdf-engine=xelatex \
     --top-level-division=chapter \
     --highlight-style=tango \
     -V colorlinks=true \
     -V linkcolor=blue \
     --wrap=auto