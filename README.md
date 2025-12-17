# openalex-authorgraph

> **Are you looking for suggested reviewers for your manuscript? Or, perhaps, researchers in your field who can comment on your work without conflict of interest?**

This `openalex-authorgraph` repository is a compact Python wrapper around the [OpenAlex REST API](https://docs.openalex.org/#about-the-api). I primarily wrote this package as a simple tool to find "relevant but unaffiliated" authors for a given author (and for experience with REST APIs).

**The intended usage is to parse the OpenAlex database for authors that have cited the target author but have never collaborated with them.** The source code is found in `authorparse.py` and the implementation meant for the user is in the `AuthorParse.ipynb` Jupyter notebook.

## Quick links

- OpenAlex documentation: https://docs.openalex.org/
- OpenAlex API overview: https://docs.openalex.org/#about-the-api

This codebase works with three main OpenAlex entity types:

- [Works](https://docs.openalex.org/api-entities/works/work-object) — publications and their metadata (authors appear in a work's
	`authorships` array).
- [Authors](https://docs.openalex.org/api-entities/authors/author-object) — person-level records; may be identified by an OpenAlex author ID
	(e.g. `https://openalex.org/A12345`) or an ORCID (e.g. `0000-0001-2345-6789`).
- [Authorships](https://docs.openalex.org/api-entities/works/work-object/authorship-object) — top-level field of a work record that lists modified author objects.

See the OpenAlex docs for canonical, up-to-date field lists and examples.

## Installation

Create a conda environment from the included `environment.yml`, or install
dependencies manually.

Conda (recommended):

```bash
conda env create -f environment.yml
conda activate openalex-authorgraph
```

Or install with pip into an existing environment:

```bash
pip install -r requirements.txt
```

## Usage (quickstart)

First, define your contact email and the ORCID of the target author:

```python
import authorparse as ap
ap.USER_EMAIL = '<your_email>@example.com'  # for polite API usage + access to reliable data rates
orcid = '0000-0001-2345-6789'  # replace with target author's ORCID
```

Your email will not be used for any purpose other than to identify your requests to the OpenAlex API. Doing so gets you access to better rate limits and more reliable data access. You are free to leave the email blank (`ap.USER_EMAIL = ""`), but please be polite for the folks at OpenAlex!

I recommend using `AuthorParse.ipynb` as a starting point for exploring the functionality.

## Functions and behavior

[ NEED TO CHECK THIS ]

All functions are implemented in `authorparse.py`. Below are the most-used
utilities and a short description of what they do:

- `standardize_author_ids(authors)` — normalize inputs (ORCID, ORCID URL,
	OpenAlex ID/URL, or dict) into canonical OpenAlex author URLs.
- `standardize_work_ids(works)` — normalize DOIs, DOI URLs, OpenAlex work IDs/URLs
	into canonical OpenAlex work URLs.
- `fetch_authors(author_ids, fields=None)` — batch-fetch author records from
	the OpenAlex `/authors` endpoint. Accepts mixed identifier types; returns
	results in the same order as inputs.
- `fetch_works(work_ids, fields=None)` — batch-fetch work records from the
	`/works` endpoint (accepts DOIs or OpenAlex work IDs).
- `fetch_works_from_author(author_id)` — list all works by an author.
- `fetch_authors_from_work(work_id)` — return the authors (full records)
	credited on a specific work (reads the work's `authorships` array and
	resolves author IDs).
- `fetch_authors_from_works(work_ids, keep_parent_ids=False)` — wrapper to
	fetch authors for many works; can return a mapping from work → authors.
- `fetch_citing_works_from_work(work_id)` and related helpers — find works that
	cite a given work or set of works.
- `get_coauthors(author_id, output='set'|'dict'|'df', keep_works=False)` —
	high-level coauthor network builder. Returns either a set of coauthor IDs,
	a dict mapping coauthor ID → record (with `linking_works` and `occurrences`),
	or a pandas DataFrame.
- `get_citing_authors(author_id, output='set'|'dict'|'df', keep_works=False)` —
	high-level builder for authors who cite the target author.

The functions are documented inline in `authorparse.py` with usage and
parameter descriptions. If you need to adapt behavior (for example how missing
author IDs are handled), see the `fetch_authors_from_work` implementation.

## License

MIT (see LICENSE)


