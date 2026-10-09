PY      ?= python
WEB     := web

.PHONY: install web data themes characters jutsu persona all app dev test clean

install:            ## Python package (editable) + spaCy models + web deps
	$(PY) -m pip install -e ".[dev]"
	$(PY) -m spacy download en_core_web_trf
	cd $(WEB) && npm install

web:                ## Build the React site into web/dist
	cd $(WEB) && npm run build

data:               ## Re-scrape jutsu articles from the Naruto wiki
	sharingan scrape

themes:      ; sharingan themes
characters:  ; sharingan characters
jutsu:       ; sharingan jutsu train
persona:     ; sharingan persona memory

all:                ## Every offline stage
	sharingan all

app: web            ## Production: API + built site on :8000
	sharingan app

dev:                ## Hot reload: API on :8000, Vite on :5173
	( sharingan app --reload & ) && cd $(WEB) && npm run dev

test:
	$(PY) -m pytest

clean:
	rm -rf $(WEB)/dist outputs/themes/_checkpoint.parquet .pytest_cache
