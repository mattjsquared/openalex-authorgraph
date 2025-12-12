import numpy as np
import pandas as pd
import requests, csv, time, copy, warnings

from functools import partial as functools_partial
from tqdm import tqdm
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from IPython.display import Markdown, display


# --- Helpers: HTTP GET with exponential backoff on "429" errors & automated GET pagination ---
@contextmanager
def _conditional_session(
  session: requests.Session = None
) -> requests.Session:
  """
  Context‐manager that yields a requests.Session for HTTP requests.

  If an existing Session is provided, that session is reused (and not closed).
  Otherwise, a new Session is created at entry and automatically closed on exit.

  Args:
    session (requests.Session, optional): An existing requests Session to reuse. 
      If None, a new Session will be created.

  Yields:
    requests.Session: The Session object to use within the context.

  Example:
    with _conditional_session() as sess:
      resp = sess.get(url)
  """
  if session is None:
    s = requests.Session()
    try:
      yield s
    finally:
      s.close()
  else:
    # assume user wants to manage its lifetime
    yield session


def request_with_backoff(
  url: str,
  params: dict = None,
  session: requests.Session = None,
  max_retries: int = 10,
  max_wait: float = 10
) -> dict:
  """
  Perform an HTTP GET with automatic retry logic for rate‐limit and occasional 404 fallbacks.

  This function will:
    1. Attach a `mailto` parameter for polite API usage.
    2. On HTTP 404 responses against an `/authors` endpoint, retry once against `/people` alias.
    3. On HTTP 429 responses (rate limit), retry with exponential backoff until
       either `max_retries` is reached or `max_wait` seconds have elapsed.
    4. Raise for any other HTTP error status.

  Args:
    url (str): Full OpenAlex API URL (e.g. "https://api.openalex.org/works").
    params (dict, optional): Query parameters to include; a `mailto` key will be added.
    session (requests.Session, optional): Existing Session to reuse; if None, a new one is created.
    max_retries (int): Max number of 429‐retry attempts.
    max_wait (float): Max total time (in seconds) to spend retrying before giving up.

  Returns:
    dict: The JSON‐decoded response body.

  Raises:
    requests.exceptions.HTTPError: If a non‐retryable error occurs (anything other than handled 404/429),
      or if 429 retries/time‐budget are exhausted.
  """
  retries = 0
  pause = 0.1
  if params is None:
    params = {}
  params.update({'mailto': USER_EMAIL})
  with _conditional_session(session) as _session:
    tstart = time.time()
    while True:
      r = _session.get(url, params=params)
      # If searching for authors and we hit a 404, retry with the "people" url
      if (r.status_code == 404) and ('authors' in url):
        url = f"{BASE_URL}people"
        continue
      # If we hit a rate limit and still can retry within time budget
      if (r.status_code == 429) and (retries < max_retries) and ((time.time() - tstart) <= max_wait):
        time.sleep(pause)
        retries += 1
        pause *= 2  # exponential increase
        continue
      # Either success or non-retryable error
      r.raise_for_status()
      break
  return r.json()


def _paginate_request(
  url: str,
  params: dict = None,
  session: requests.Session = None,
  per_page: int = 200,
  max_pages: int = None
):
  """
  Lazily retrieve paginated results from an OpenAlex endpoint using cursor‐based pagination.

  This generator will:
    1. Initialize the cursor to "*" and include a per‐page limit.
    2. Repeatedly call `request_with_backoff()` to fetch each page.
    3. Yield the full JSON response for each page (including both "results" and "meta").
    4. Stop when the returned page has no `meta.next_cursor` or when `max_pages` is reached.

  Args:
    url (str): Full API endpoint URL (e.g. "https://api.openalex.org/works").
    params (dict, optional): Base query parameters; will be copied and augmented.
    session (requests.Session, optional): Session for HTTP calls; if None, one is created.
    per_page (int): Number of items to fetch per page (added as "per-page" param).
    max_pages (int, optional): Maximum number of pages to retrieve; if None, fetch until exhausted.

  Yields:
    dict: Each page’s parsed JSON payload, containing:
      - "results": list of entity records for that page
      - "meta":   metadata including "count", "next_cursor", etc.
  """
  count = 0
  _params = (params or {}).copy()
  _params.update({"per-page": per_page, "cursor": "*"})
  with _conditional_session(session) as _session:
    while True:
      page = request_with_backoff(url, params=_params, session=_session)
      yield page
      next_cursor = page.get("meta", {}).get("next_cursor")
      count += 1
      if not next_cursor or (max_pages is not None and count >= max_pages):
        break
      _params["cursor"] = next_cursor


def request_collated_pages(
  url: str,
  params: dict = None,
  session: requests.Session = None,
  per_page: int = 200,
  max_pages: int = None
) -> dict:
  """
  Retrieve all pages from an OpenAlex endpoint and group their payloads.

  This function will:
    1. Use the cursor‑based pagination generator `_paginate_request` to fetch each page.
    2. Collect each page’s “results” lists into a single list of lists.
    3. Collect each page’s “meta” dictionaries into a single list of page metadata.

  Args:
    url (str): The base OpenAlex API URL to page through (e.g. "https://api.openalex.org/works").
    params (dict, optional): Query parameters to include on every request (augmented per page).
    session (requests.Session, optional): An existing Session to reuse for HTTP calls; if None, a new Session is created.
    per_page (int): Number of items to request per page.
    max_pages (int, optional): If set, limits the number of pages fetched; if None, fetches until no more pages.

  Returns:
    dict:
      - "results" (List[List[dict]]): A list where each element is the `"results"` list from one page.
      - "meta"    (List[dict])      : A list of the `"meta"` dict for each fetched page.

  Example:
    pages = request_collated_pages("https://api.openalex.org/works", params={"filter": "author.id:A123"}, per_page=100)
    all_results = pages["results"]        # [[...page1...], [...page2...], ...]
    all_meta    = pages["meta"]           # [{...meta1...}, {...meta2...}, ...]
  """
  with _conditional_session(session) as _session:
    pages = list(_paginate_request(url, params=params, session=_session, per_page=per_page, max_pages=max_pages))
  collated = {
    "results": [p["results"] for p in pages],
    "meta":    [p["meta"]    for p in pages],
  }
  return collated


def request_entities(
  url: str,
  params: dict = None,
  session: requests.Session = None,
  per_page: int = 200,
  max_pages: int = None,
  include_meta: bool = False
) -> dict:
  """
  Retrieve all records from a paginated OpenAlex endpoint, optionally summarizing pagination metadata.

  This function will:
    1. Fetch every page of results via `request_collated_pages`.
    2. Flatten the per‑page “results” lists into a single list.
    3. If `include_meta` is True, aggregate shared metadata across pages.

  Args:
    url (str): Base API endpoint (e.g. "https://api.openalex.org/works").
    params (dict, optional): Query parameters applied to every page request.
    session (requests.Session, optional): Session to reuse for HTTP requests; if None, a new one is created.
    per_page (int): Number of items to request per page.
    max_pages (int, optional): Maximum number of pages to fetch; if None, continues until no more pages.
    include_meta (bool): If True, include aggregate metadata under the “meta” key.

  Returns:
    dict:
      - "results" (List[dict]): All items from every page, concatenated.
      - "meta" (dict, optional): Present only if `include_meta` is True, containing:
          • "count":             Total items available (from first page’s metadata).
          • "groups_count":      Optional grouping count (from first page’s metadata).
          • "page_count":        Number of pages retrieved.
          • "db_response_time_total": Sum of `db_response_time_ms` across all pages.

  Example:
    resp = request_entities(
      "https://api.openalex.org/works",
      params={"filter": "author.id:A123"},
      include_meta=True
    )
    all_works = resp["results"]
    info = resp["meta"]
  """
  with _conditional_session(session) as _session:
    pages = request_collated_pages(url, params=params, session=_session, per_page=per_page, max_pages=max_pages)
  # Flatten only the per‐page result lists, not the dict keys:
  flat = [item for page_results in pages["results"] for item in page_results]
  out = {"results": flat}
  if include_meta:
    first_meta = pages["meta"][0]
    page_count = len(pages["meta"])
    total_db = sum(m.get("db_response_time_ms", 0) for m in pages["meta"])
    out["meta"] = {
      "count": first_meta.get("count"),
      "groups_count": first_meta.get("groups_count"),
      "page_count": page_count,
      "db_response_time_total": total_db,
      }
  return out


# --- Field filtering utility ---
def _filter_fields(
  obj: dict,
  fields
) -> dict:
  """
  Reduce a JSON object to only the specified top-level fields.

  Args:
    obj (dict): The original JSON object.
    fields (str or list of str, optional): Field names to keep.
      If None or empty, returns the original object.

  Returns:
    dict: Filtered JSON containing only requested fields.
  """
  if not fields:
    return obj
  # Normalize single string to list
  if isinstance(fields, str):
    fields = [fields]
  return {f: obj.get(f) for f in fields}


def _fields2params(
  fields
) -> dict:
  """
  Format OpenAlex field filter(s) as HTTP GET parameters.

  Args:
    fields (str or list of str, optional): Field names to keep.
      If None or empty, returns empty parameters.

  Returns:
    dict: `fields` formatted for HTTP GET request.
  """
  params = {}
  if (fields is None) or (len(fields) == 0):
    return params
  if isinstance(fields, str):
    params['select'] = fields if (',' in fields) else [fields]
  else:
    params['select'] = ','.join(fields)
  return params
  


# --- ID normalization ---
def standardize_author_ids(
  authors,
  session: requests.Session = None
) -> list:
  """
  Convert a mix of ORCID identifiers, OpenAlex author IDs/URLs, or author dicts
  into a list of canonical OpenAlex author URLs.

  This function will:
    1. Accept a single author (str or dict) or a list of them.
    2. Identify which inputs look like ORCIDs (contain exactly 3 hyphens).
    3. Batch‑fetch the corresponding OpenAlex author URLs for those ORCIDs.
    4. Preserve any inputs that are already OpenAlex URLs/IDs unchanged.
    5. Return a list of OpenAlex author URLs in the same order as the inputs.

  Args:
    authors (str, dict, or list[str or dict]):
      - If dict, must contain an 'id' key with the OpenAlex URL.
      - If str, may be an ORCID (e.g. "0000-0002-1234-5678"), an ORCID URL,
        or an OpenAlex author URL/ID (e.g. "A123456").
      - May also be a list of such values.
    session (requests.Session, optional):
      An existing HTTP session to reuse for the batch lookup. If None,
      a temporary session is created and closed automatically.

  Returns:
    list[str]: Canonical OpenAlex author URLs, in the same order as the inputs.

  Example:
    # Single ORCID string
    standardize_author_ids("0000-0002-3318-5801")
    # -> ["https://openalex.org/A5040982088"]

    # Mixed list of ORCID and OpenAlex IDs
    standardize_author_ids([
      "0000-0002-3318-5801",
      "https://openalex.org/A123456"
    ])
    # -> ["https://openalex.org/A5040982088", "https://openalex.org/A123456"]
  """
  # normalize to a list of strings
  if isinstance(authors, dict):
    return [authors['id']]
  if isinstance(authors, str):
    authors = [authors]
  # extract the bare tokens (either ORCID strings or OA IDs)
  tokens = [a.rstrip('/').split('/')[-1] for a in authors]
  # pick out just the ORCID‐looking tokens (4 hyphens)
  is_orcid = lambda t: t.count('-') == 3
  orcids = [t for t in tokens if is_orcid(t)]
  # batch‐lookup the ORCIDs in one call
  mapping = {}
  if orcids:
    params = _fields2params(['orcid', 'id'])
    params['filter'] = 'orcid:' + '|'.join(orcids)
    with _conditional_session(session) as _session:
      r = request_entities(f"{BASE_URL}authors", params=params, session=_session)
    # build quick lookup from short‐form ORCID → full OA URL
    mapping = {auth['orcid'].split('/')[-1]: auth['id'] for auth in r.get('results', [])}
  # rebuild preserving order; non-ORCID tokens stay as originally given
  result = [mapping.get(tok, auth) for tok, auth in zip(tokens, authors)]
  return result


def standardize_work_ids(
  works,
  session: requests.Session = None
) -> list:
  """
  Convert a mix of DOIs, OpenAlex work IDs/URLs, or work dicts into canonical OpenAlex work URLs.

  This function will:
    1. Accept a single work identifier (str or dict) or a list of them.
    2. Identify which inputs look like DOIs (the part after “.org/” starts with "10.").
    3. Batch‑fetch the corresponding OpenAlex work URLs for those DOIs in a single API call.
    4. Preserve any inputs that are already OpenAlex work URLs/IDs unchanged.
    5. Return a list of OpenAlex work URLs in the same order as the inputs.

  Args:
    works (str, dict, or list[str or dict]):
      - If dict, must contain an 'id' key with the OpenAlex URL.
      - If str, may be a DOI (e.g. "10.7717/peerj.4375"), a DOI URL
        (e.g. "https://doi.org/10.7717/peerj.4375"), or an OpenAlex work URL/ID
        (e.g. "W123456" or "https://openalex.org/W123456").
      - May also be a list of such values.
    session (requests.Session, optional):
      An existing HTTP session to reuse for the batch lookup. If None,
      a temporary session is created and closed automatically.

  Returns:
    list[str]: Canonical OpenAlex work URLs, in the same order as the inputs.

  Examples:
    # Single DOI string
    standardize_work_ids("10.7717/peerj.4375")
    # -> ["https://openalex.org/W4375"]

    # Mixed list of DOI and OpenAlex IDs
    standardize_work_ids([
      "10.7717/peerj.4375",
      "https://openalex.org/W123456"
    ])
    # -> ["https://openalex.org/W4375", "https://openalex.org/W123456"]
  """
  # normalize to a list of strings
  if isinstance(works, dict):
    return [works['id']]
  if isinstance(works, str):
    works = [works]
  # extract the bare tokens (either DOIs or OA IDs)
  tokens = [w for w in works]
  # pick out just the DOI‐looking tokens
  is_doi = lambda t: t.rstrip('/').split('.org/')[-1].startswith('10.')
  dois = [t for t in tokens if is_doi(t)]
  # batch‐lookup the DOIs in one call
  mapping = {}
  if dois:
    params = _fields2params(['doi', 'id'])
    params['filter'] = 'doi:' + '|'.join(dois)
    with _conditional_session(session) as _session:
      r = request_entities(f"{BASE_URL}works", params=params, session=_session)
    # build quick lookup from short‐form DOI → full OA URL
    mapping = {wk['doi']: wk['id'] for wk in r.get('results', []) if wk.get('doi')}
  # rebuild preserving order; non-ORCID tokens stay as originally given
  result = [mapping.get(tok, wk) for tok, wk in zip(tokens, works)]
  return result


# --- Direct Entity Fetchers ---
def fetch_authors(
  author_ids,
  fields = None,
  session: requests.Session = None
) -> list:
  """
  Retrieve one or more OpenAlex author records in a single batched API call.

  This function accepts a single author identifier or a list of them, where each identifier
  can be:
    - An OpenAlex author URL (e.g. "https://openalex.org/A123456")
    - An OpenAlex author ID (e.g. "A123456")
    - An ORCID (e.g. "0000-0002-3318-5801") or ORCID URL (e.g. "https://orcid.org/0000-0002-3318-5801")
    - A dict representing a partial author JSON with at least an "id" key

  The identifiers are first normalized to canonical OpenAlex URLs, then fetched
  in one request using the `ids.openalex` filter. The returned list is ordered
  to correspond exactly to the order of the input identifiers.

  Args:
    author_ids (str, dict, or list[str or dict]):
      Single author identifier or list of identifiers/dicts as described above.
    fields (str or list[str], optional):
      Fields to include in each author record (returned under `"select"`). If None,
      all top‑level fields are returned.
    session (requests.Session, optional):
      An existing HTTP session to reuse for all requests. If None, a temporary session
      will be created and closed automatically.

  Returns:
    list[dict]:
      A list of author JSON objects, in the same order as the provided `author_ids`.

  Raises:
    RuntimeError:
      If any requested author ID could not be retrieved (i.e., the API did not return it).
  """
  params = _fields2params(fields)
  params.update({'per-page': 200, 'cursor': '*'})
  url = f"{BASE_URL}authors"
  with _conditional_session(session) as _session:
    author_ids = standardize_author_ids(author_ids, session=_session)
    params['filter'] = f"ids.openalex:{'|'.join(author_ids)}"
    data = request_entities(url, params=params, session=_session)
  data = data['results']
  author_map = {auth['id']: auth for auth in data}
  data = [author_map.get(aid) for aid in author_ids]
  if any(auth is None for auth in data):
    missing_id = [aid for aid, auth in zip(author_ids, data) if auth is None]
    raise RuntimeError("≥1 author was not retrieved:\n"+"\n".join(f"{mid}" for mid in missing_id))
  return data


def fetch_works(
  work_ids,
  fields = None,
  session: requests.Session = None
) -> list:
  """
  Retrieve one or more OpenAlex work records in a single batched API call.

  This function accepts a single work identifier or a list of them, where each identifier
  can be:
    - An OpenAlex work URL (e.g. "https://openalex.org/W123456")
    - An OpenAlex work ID (e.g. "W123456")
    - A DOI string (e.g. "10.7717/peerj.4375") or DOI URL
      (e.g. "https://doi.org/10.7717/peerj.4375")
    - A dict containing at least an "id" key (and optionally "doi")

  All identifiers are first normalized to canonical OpenAlex work URLs via
  `standardize_work_ids()`, then fetched in one API call using the `ids.openalex`
  filter. The returned list is ordered to correspond exactly to the order of the
  input identifiers.

  Args:
    work_ids (str, dict, or list[str or dict]):
      Single work identifier or list of identifiers/dicts as described above.
    fields (str or list[str], optional):
      Fields to include in each work record (mapped to the "select" parameter).
      If None, all top‑level fields are returned.
    session (requests.Session, optional):
      An existing HTTP session to reuse for all requests. If None, a temporary session
      will be created and closed automatically.

  Returns:
    list[dict]:
      A list of work JSON objects, in the same order as the provided `work_ids`.

  Raises:
    RuntimeError:
      If any requested work ID could not be retrieved (i.e., the API did not return
      a record for one or more of the input identifiers).
  """
  params = _fields2params(fields)
  url = f"{BASE_URL}works"
  with _conditional_session(session) as _session:
    work_ids = standardize_work_ids(work_ids)
    params['filter'] = f"ids.openalex:{'|'.join(work_ids)}"
    data = request_entities(url, params=params, session=_session)
  data = data['results']
  work_map = {wk['id']: wk for wk in data}
  data = [work_map.get(wid) for wid in work_ids]
  if any(wk is None for wk in data):
    missing = [wid for wid, wk in zip(work_ids, data) if wk is None]
    raise RuntimeError("≥1 work was not retrieved:\n"+"\n".join(f"{msng}" for msng in missing))
  return data


def fetch_works_from_author(
  author_id,
  fields = None,
  session: requests.Session = None
) -> list:
  """
  Retrieve every work record for a specified OpenAlex author in a single batched request.

  This function:
    1. Normalizes the input `author_id` (which may be an ORCID, OpenAlex ID, URL, or dict)
       into the canonical OpenAlex author URL.
    2. Uses the OpenAlex `/works` endpoint with an `author.id:` filter to page through
       all works authored by that individual.
    3. Returns the complete list of work JSON objects, optionally filtered to only
       include the requested fields.

  Args:
    author_id (str or dict):
      An author identifier, which may be:
        - An OpenAlex author URL (e.g. "https://openalex.org/A123456")
        - An OpenAlex author ID (e.g. "A123456")
        - An ORCID URL or code (e.g. "https://orcid.org/0000-0001-2345-6789" or "0000-0001-2345-6789")
        - A dict containing at least an `"id"` or `"orcid"` key
    fields (str or list[str], optional):
      One or more top‑level work attributes to include in each returned record.
      Passed directly to the API’s `select` parameter. If None, all fields are returned.
    session (requests.Session, optional):
      An existing `requests.Session` to reuse for HTTP calls. If None, a temporary
      session is created and closed automatically.

  Returns:
    list[dict]:
      A list of OpenAlex work JSON objects authored by the specified author,
      in no particular order beyond the API’s internal pagination sequence.

  Raises:
    HTTPError:
      If the underlying HTTP request fails for any non‑rate‑limit error.
  """
  params = _fields2params(fields)
  url = f"{BASE_URL}works"
  with _conditional_session(session) as _session:
    author_id = standardize_author_ids(author_id, session=_session)[0]
    params['filter'] = f"author.id:{author_id}"
    data = request_entities(url, params=params, session=_session)['results']
  return data


def fetch_citing_works_from_work(
  work_id,
  fields = None,
  session: requests.Session = None
) -> list:
  """
  Retrieve all works that cite a specific OpenAlex work.

  This function:
    1. Normalizes the input `work_id` (which may be a DOI, OpenAlex ID, URL, or dict)
       into the canonical OpenAlex work URL.
    2. Queries the OpenAlex `/works` endpoint using the `cites:` filter to page through
       every work that references the given work.
    3. Collects and returns the full list of citing work JSON objects, optionally filtered
       to include only the specified fields.

  Args:
    work_id (str or dict):
      A work identifier, which may be:
        - An OpenAlex work URL (e.g. "https://openalex.org/W123456")
        - An OpenAlex work ID (e.g. "W123456")
        - A DOI URL or code (e.g. "https://doi.org/10.1000/xyz123" or "10.1000/xyz123")
        - A dict containing at least an `"id"` or `"doi"` key
    fields (str or list[str], optional):
      One or more top‑level work attributes to include in each returned record.
      Passed directly to the API’s `select` parameter. If None, all fields are returned.
    session (requests.Session, optional):
      An existing `requests.Session` to reuse for HTTP calls. If None, a temporary
      session is created and closed automatically.

  Returns:
    list[dict]:
      A list of OpenAlex work JSON objects that cite the specified work,
      in no particular order beyond the API’s internal pagination sequence.

  Raises:
    HTTPError:
      If any underlying HTTP request fails for a non‑rate‑limit error.
  """
  params = _fields2params(fields)
  params['per-page'] = 200
  url = f"{BASE_URL}works"
  with _conditional_session(session) as _session:
    work_id = standardize_work_ids(work_id, session=_session)[0]
    params['filter'] = 'cites:' + work_id
    data = request_entities(url, params=params, session=_session)['results']
  return data


# --- Compound/Relationship Fetchers ---
def fetch_authors_from_work(
  work_id,
  fields = None,
  session: requests.Session = None
) -> list:
  """
  Retrieve all authors credited on a specific OpenAlex work.

  This function:
    1. Normalizes the input `work_id` (which may be a DOI, OpenAlex ID, URL, or dict)
       into the canonical OpenAlex work URL.
    2. Fetches the work’s basic record (including its `authorships` array) via the
       `/works` endpoint.
    3. Extracts each author’s OpenAlex URL/ID from the `authorships` entries.
    4. Batch‑fetches the full author records (optionally filtered to `fields`) via
       the `/authors` endpoint.

  Args:
    work_id (str or dict):
      A work identifier, which may be:
        - An OpenAlex work URL (e.g. "https://openalex.org/W123456")
        - An OpenAlex work ID (e.g. "W123456")
        - A DOI URL or code (e.g. "https://doi.org/10.1000/xyz123" or "10.1000/xyz123")
        - A dict containing at least an `"id"` or `"doi"` key
    fields (str or list[str], optional):
      One or more top‑level author attributes to include in each returned record.
      Passed directly to the API’s `select` parameter. If None, all author fields are returned.
    session (requests.Session, optional):
      An existing `requests.Session` to reuse for HTTP calls. If None, a temporary
      session is created and closed automatically.

  Returns:
    list[dict]:
      A list of OpenAlex author JSON objects corresponding to the work’s authors,
      in the same order as they appear in the work’s `authorships` list.
      Returns an empty list (and issues a warning) if the work has no `authorships`.

  Raises:
    RuntimeError:
      If any author lookup fails (e.g., an expected author ID cannot be retrieved).
    HTTPError:
      If any underlying HTTP request fails for a non‑rate‑limit error.
  """
  with _conditional_session(session) as _session:
    work_id = standardize_work_ids(work_id, session=_session)[0]
    work = fetch_works(work_id, fields=['id', 'authorships'], session=_session)[0]
    if not work.get('authorships'):
      warnings.warn(f"No authorships: work ID {work_id}")
      return []
    author_ids = [auth.get('author').get('id') for auth in work.get('authorships')]
    authors = fetch_authors(author_ids, fields=fields, session=_session)
  return authors


def fetch_authors_from_works(
  work_ids,
  fields = None,
  session: requests.Session = None,
  keep_parent_ids: bool = False
):
  """
  Retrieve authors for one or more OpenAlex works, optionally keyed by work ID.

  This function:
    1. Normalizes each entry in `work_ids` to its canonical OpenAlex work URL.
    2. For each work, fetches its authorships via `fetch_authors_from_work`.
    3. Aggregates all author records into a single flat list, or, if
       `keep_parent_ids=True`, returns a dict mapping each work’s URL to
       its list of author JSON objects.

  Args:
    work_ids (str or list[str] or dict or list[dict]):
      A single work identifier or a collection thereof. Each item may be:
        - An OpenAlex work URL (e.g. "https://openalex.org/W123456")
        - An OpenAlex work ID (e.g. "W123456")
        - A DOI URL or code (e.g. "https://doi.org/10.1000/xyz123" or "10.1000/xyz123")
        - A dict containing at least an `"id"` or `"doi"` key
    fields (str or list[str], optional):
      Top‑level author fields to include in each returned record. Passed through
      to the API’s `select` parameter; if None, the full author record is returned.
    session (requests.Session, optional):
      An existing HTTP session to reuse for all requests. If None, a temporary
      session is created and closed internally.
    keep_parent_ids (bool):
      - If False (default), returns a flat `list[dict]` of all authors across
        the specified works.
      - If True, returns a `dict` whose keys are each work’s OpenAlex URL and
        whose values are the `list[dict]` of that work’s authors.

  Returns:
    list[dict] or dict[str, list[dict]]:
      Depending on `keep_parent_ids`, either:
        - A flat list of author JSON objects (duplicates possible if authors
          appear on multiple works), or
        - A mapping from each work’s URL to its list of author JSON objects.

  Raises:
    HTTPError:
      If any underlying API call fails with a non‑rate‑limit error.
    RuntimeError:
      If author lookup for any work fails to retrieve the expected data.
  """
  authors = {} if keep_parent_ids else []
  with _conditional_session(session) as _session:
    work_ids = standardize_work_ids(work_ids, session=_session)
    _fetcher = functools_partial(fetch_authors_from_work, fields=fields, session=_session)
    for wid in tqdm(work_ids, desc="Fetching authors from works"):
      if keep_parent_ids:
        authors[wid] = _fetcher(wid)
      else:
        authors.extend(_fetcher(wid))
  return authors


def fetch_coauthors_from_author(
  author_id,
  fields = None,
  session: requests.Session = None,
  keep_parent_ids: bool = False
):
  """
  Retrieve coauthors for a given author, across all of their works.

  This function:
    1. Normalizes the input `author_id` to its canonical OpenAlex URL.
    2. Fetches all works authored by that author.
    3. For each work, retrieves its list of authors.
    4. Aggregates these coauthor records into either:
       - A flat list of coauthor JSON objects (if `keep_parent_ids=False`), or
       - A dict mapping each work’s URL to its list of coauthor JSON objects
         (if `keep_parent_ids=True`).

  Args:
    author_id (str or dict):
      An ORCID, OpenAlex author URL/ID, or a dict containing an "id" or "orcid".
    fields (str or list[str], optional):
      Specific top-level author fields to include in each returned record.
      Passed through to the API’s `select` parameter; if None, the full record
      is returned.
    session (requests.Session, optional):
      An existing HTTP session to reuse for all requests. If None, a new session
      is opened and closed internally.
    keep_parent_ids (bool, default=False):
      - If False: returns a flat list of all coauthor JSON dicts.
      - If True: returns a dict where each key is a work’s OpenAlex URL and each
        value is the list of coauthor JSON dicts for that work.

  Returns:
    list[dict] or dict[str, list[dict]]:
      Depending on `keep_parent_ids`, either a flat list of coauthor records
      (duplicates possible if the same coauthor appears on multiple works), or
      a mapping from work URLs to their respective coauthor lists.

  Raises:
    HTTPError:
      If any underlying API request fails with a non‑rate‑limit error.
    RuntimeError:
      If a work or author lookup does not return the expected data.
  """
  _looper = functools_partial(tqdm, desc="Fetching target author's coauthors")
  with _conditional_session(session) as _session:
    author_id = standardize_author_ids(author_id, session=_session)[0]
    works = fetch_works_from_author(author_id, fields=['id'], session=_session)
    work_ids = [wk['id'] for wk in works]
    coauthors = fetch_authors_from_works(work_ids, fields=fields, session=_session, keep_parent_ids=keep_parent_ids)
  return coauthors


def fetch_citing_works_from_works(
  work_ids,
  fields = None,
  session: requests.Session = None,
  keep_parent_ids: bool = False
):
  """
  Retrieve works that cite any of the given works.

  This function:
    1. Normalizes each input in `work_ids` to its canonical OpenAlex work URL.
    2. For each work URL, fetches all works that cite it.
    3. Aggregates the citing-work records into either:
       - A flat list of work JSON objects (if `keep_parent_ids=False`), or
       - A dict mapping each original work’s URL to its list of its citing works
         (if `keep_parent_ids=True`).

  Args:
    work_ids (str or list[str] or dict or list[dict]):
      One or more OpenAlex work URLs/IDs, DOIs, or work JSON dicts containing an 'id'.
    fields (str or list[str], optional):
      Specific top-level work fields to include in each returned record.
      Passed through to the API’s `select` parameter; if None, the full record is returned.
    session (requests.Session, optional):
      An existing HTTP session to reuse for all requests. If None, a new session is
      opened and closed internally.
    keep_parent_ids (bool, default=False):
      - If False: returns a flat list of all citing-work JSON dicts.
      - If True: returns a dict where each key is an original work’s OpenAlex URL and
        each value is the list of works that cite that work.

  Returns:
    list[dict] or dict[str, list[dict]]:
      Depending on `keep_parent_ids`, either a flat list of citing-work records, or
      a mapping from each input work URL to its respective list of citing works.

  Raises:
    HTTPError:
      If any underlying API request fails with an HTTP error other than rate limiting.
    RuntimeError:
      If a lookup for a work or its citing works does not return the expected data.
  """
  with _conditional_session(session) as _session:
    work_ids = standardize_work_ids(work_ids, session=_session)
    citing_works = {} if keep_parent_ids else []
    _fetcher = functools_partial(fetch_citing_works_from_work, fields=fields, session=_session)
    for wid in tqdm(work_ids, desc="Fetching works that cite target work"):
      if keep_parent_ids:
        citing_works[wid] = _fetcher(wid)
      else:
        citing_works.extend(_fetcher(wid))
  return citing_works


def fetch_citing_works_from_author(
  author_id,
  fields = None,
  session: requests.Session = None,
  keep_parent_ids: bool = False
):
  """
  Retrieve all works that cite any publication by a given author.

  This function:
    1. Normalizes the `author_id` to its canonical OpenAlex URL.
    2. Fetches all works authored by that author (only IDs).
    3. For each of those works, fetches all works that cite it.
    4. Aggregates results into either:
       - A flat list of citing-work JSON dicts (if `keep_parent_ids=False`), or
       - A dict mapping each original work’s URL to its list of its citing works
         (if `keep_parent_ids=True`).

  Args:
    author_id (str or dict):
      An ORCID, OpenAlex author URL/ID, or an author JSON dict containing an 'id'.
    fields (str or list[str], optional):
      Specific top-level fields of each citing work to include (passed to the API’s
      `select` parameter). If None, the API returns the full work record.
    session (requests.Session, optional):
      If provided, reuses this HTTP session for all requests; otherwise a new session
      is created and closed internally.
    keep_parent_ids (bool, default=False):
      - False: returns a flat list of all citing-work records.
      - True: returns a dict mapping each authored work’s URL to its list of citing works.

  Returns:
    list[dict] or dict[str, list[dict]]:
      Depending on `keep_parent_ids`, either:
        - A flat list of works that cite any of the author’s works.
        - A mapping from each authored work’s URL to its list of citing works.

  Raises:
    HTTPError:
      If any API request fails with a non-retryable HTTP error.
    RuntimeError:
      If expected data is missing (e.g., unable to retrieve works or citations).
  """
  with _conditional_session(session) as _session:
    author_id = standardize_author_ids(author_id, session=_session)[0]
    works = fetch_works_from_author(author_id, fields=['id'], session=_session)
    work_ids = [wk['id'] for wk in works]
    citing_works = fetch_citing_works_from_works(work_ids, fields=fields, session=_session, keep_parent_ids=keep_parent_ids)
  return citing_works


def fetch_citing_works_from_authors(
  author_ids,
  fields = None,
  session: requests.Session = None,
  keep_parent_ids: bool = False
):
  """
  Retrieve all works that cite any publications by one or more authors.

  This function:
    1. Normalizes each entry in `author_ids` to its canonical OpenAlex author URL.
    2. For each author, fetches all works they have published.
    3. For each of those works, fetches all works that cite it.
    4. Aggregates results into either:
       - A flat list of all citing-work records (if `keep_parent_ids=False`), or
       - A dict mapping each author’s URL to the list of works that cite any of their works
         (if `keep_parent_ids=True`).

  Args:
    author_ids (str or list[str] or dict or list[dict]):
      An ORCID, OpenAlex author URL/ID, or author JSON dict (or a list thereof).
    fields (str or list[str], optional):
      Specific top-level fields of each citing work to include (passed to the API’s
      `select` parameter). If None, the API returns the full work records.
    session (requests.Session, optional):
      If provided, reuses this HTTP session for all requests; otherwise a new session
      is created and closed internally.
    keep_parent_ids (bool, default=False):
      - False: returns a flat list of all works citing any of the authors’ works.
      - True: returns a dict mapping each author’s URL to its list of citing works.

  Returns:
    list[dict] or dict[str, list of dict]:
      Depending on `keep_parent_ids`, either a flat list of citing-work records,
      or a mapping from each author URL to its list of citing works.

  Raises:
    HTTPError:
      If any API request fails with a non-retryable HTTP error.
    RuntimeError:
      If expected data is missing or if no works/citations can be retrieved.
  """
  _looper = functools_partial(tqdm, desc="Fetching authors' citing works")
  citing_works = {} if keep_parent_ids else []
  with _conditional_session(session) as _session:
    author_ids = standardize_author_ids(author_ids, session=_session)
    _fetcher = functools_partial(fetch_citing_works_from_author, fields=fields, session=_session, keep_parent_ids=False) # TODO: add multi-tiered parent IDs?
    for aid in _looper(author_ids):
      if keep_parent_ids:
        citing_works[aid] = _fetcher(aid)
      else:
        citing_works.extend(_fetcher(aid))
  return citing_works


def fetch_citing_authors_from_work(
  work_id,
  fields = None,
  session: requests.Session = None,
  keep_parent_ids: bool = False
):
  """
  Retrieve all authors whose works cite a specified publication.

  This function:
    1. Normalizes `work_id` to its canonical OpenAlex URL.
    2. Fetches all works that cite the given work (only the work IDs by default).
    3. For each citing work, extracts its list of authors.
    4. Aggregates those authors into either:
       - A flat list of author records (if `keep_parent_ids=False`), or
       - A dict mapping the cited work’s URL to the list of its citing authors
         (if `keep_parent_ids=True`).

  Args:
    work_id (str or dict):
      An OpenAlex work URL/ID, DOI URL/ID, or partial work JSON dict containing `'id'`.
    fields (str or list[str], optional):
      Specific top-level fields of each author to include (passed to the API’s
      `select` parameter). If None, the full author records are returned.
    session (requests.Session, optional):
      If provided, reuses this HTTP session for all requests; otherwise a new session
      is created and closed internally.
    keep_parent_ids (bool, default=False):
      - False: returns a flat list of author records across all citing works.
      - True: returns a dict with the single key being the cited work’s URL and the
        value being its list of citing-author records.

  Returns:
    list[dict] or dict[str, list of dict]:
      Depending on `keep_parent_ids`, either a flat list of author JSON dicts,
      or a mapping `{ work_id: [author_dict, …] }`.

  Raises:
    HTTPError:
      If any underlying API request fails with a non-retryable HTTP error.
    RuntimeError:
      If expected data (e.g. authorships) is missing or cannot be retrieved.
  """
  with _conditional_session(session) as _session:
    work_id = standardize_work_ids(work_id, session=_session)[0]
    citing_works = fetch_citing_works_from_work(work_id, fields=['id'], session=_session)
    cwork_ids = [cwk['id'] for cwk in citing_works]
    citing_authors = fetch_authors_from_works(cwork_ids, fields=fields, session=_session, keep_parent_ids=keep_parent_ids)
  return citing_authors


def fetch_citing_authors_from_works(
  work_ids,
  fields = None,
  session: requests.Session = None,
  keep_parent_ids: bool = False
):
  """
  Retrieve all authors whose works cite any publication in a given list.

  This function:
    1. Normalizes each entry in `work_ids` to canonical OpenAlex URLs.
    2. Fetches all works citing each target work.
    3. Collects the author lists for those citing works.
    4. Returns either:
       - A flat list of author records (if `keep_parent_ids=False`), or
       - A dict mapping each target work’s URL to the list of its citing authors
         (if `keep_parent_ids=True`).

  Args:
    work_ids (str or list[str] or dict or list[dict]):
      A single or list of OpenAlex work URLs/IDs, DOI URLs/IDs, or partial
      work JSON dicts containing `'id'`.
    fields (str or list[str], optional):
      Field or list of fields to include for each citing author (passed to the
      API `select` parameter). If None, full author records are returned.
    session (requests.Session, optional):
      If provided, uses this HTTP session for all requests; otherwise creates
      and closes a new session internally.
    keep_parent_ids (bool, default=False):
      - False: returns a flat list of author JSON dicts across all citing works.
      - True: returns a dict `{ work_id: [author_dict, …] }` for each input work.

  Returns:
    list[dict] or dict[str, list[dict]]:
      Depending on `keep_parent_ids`, either a flat list of author records,
      or a mapping from each original work ID to its list of citing-author records.

  Raises:
    HTTPError:
      If any API request fails irrecoverably.
    RuntimeError:
      If expected data is missing or cannot be retrieved.
  """
  with _conditional_session(session) as _session:
    work_ids = standardize_work_ids(work_ids, session=_session)
    citing_works = fetch_citing_works_from_works(work_ids, fields=['id'], session=_session, keep_parent_ids=False) # TODO: enable two-tiered parent ID–saving?
    citing_work_ids = [cwk['id'] for cwk in citing_works]
    citing_authors = fetch_authors_from_works(citing_work_ids, fields=fields, session=_session, keep_parent_ids=keep_parent_ids)
  return citing_authors


def fetch_citing_authors_from_author(
  author_id,
  fields = None,
  session: requests.Session = None,
  keep_parent_ids: bool = False
):
  """
  Retrieve all authors who have cited any publication by a given author.

  This function:
    1. Normalizes the input `author_id` to a canonical OpenAlex author URL.
    2. Fetches all works authored by that author.
    3. Retrieves all works that cite each of those works.
    4. Gathers the author lists for those citing works.
    5. Returns either:
       - A flat list of citing-author records (if `keep_parent_ids=False`), or
       - A mapping from each citing-work URL back to its list of citing-author records
         (if `keep_parent_ids=True`).

  Args:
    author_id (str or dict):
      ORCID, OpenAlex author URL/ID, or an author JSON dict containing an `'id'` key.
    fields (str or list[str], optional):
      Field or list of fields to include for each citing author (passed to the API `select` parameter).
      If None, full author records are returned.
    session (requests.Session, optional):
      If provided, uses this HTTP session for all requests; otherwise creates
      and closes a new session internally.
    keep_parent_ids (bool, default=False):
      - False: returns a flat list of author JSON dicts across all citing works.
      - True: returns a dict `{ work_id: [author_dict, …] }` mapping each citing-work URL
        to its list of author records.

  Returns:
    list[dict] or dict[str, list[dict]]:
      Depending on `keep_parent_ids`, either a flat list of author records,
      or a mapping from each citing-work ID to its list of citing-author records.

  Raises:
    HTTPError:
      If any API request fails after retries.
    RuntimeError:
      If any expected data cannot be retrieved or is missing.
  """
  with _conditional_session(session) as _session:
    author_id = standardize_author_ids(author_id, session=_session)[0]
    works = fetch_works_from_author(author_id, fields=['id'], session=_session, keep_parent_ids=False) # TODO: add multi-tiered parent IDs?
    work_ids = [wk['id'] for wk in works]
    citing_works = fetch_citing_works_from_works(work_ids, fields=['id'], session=_session, keep_parent_ids=False) # TODO: add multi-tiered parent IDs?
    citing_work_ids = [cwk['id'] for cwk in citing_works]
    citing_authors = fetch_authors_from_works(citing_work_ids, fields=fields, session=_session, keep_parent_ids=keep_parent_ids)
  return citing_authors


# --- Conversion Utilities ---
def _to_set(
  entities: list
) -> set:
  """
  Build a set of unique display names from a list of OpenAlex entity records.

  Iterates through each entity dict in the input list, extracts the value
  associated with the 'display_name' key, and returns a set of these values,
  thereby deduplicating any repeated names.

  Args:
    entities (list of dict): A list where each element is an OpenAlex entity
      JSON dict containing at least the 'display_name' key mapping to a string.

  Returns:
    set of str: The unique display names found in the input entities.

  Raises:
    KeyError: If any entity in the list does not include the 'display_name' key.
  """
  return set(e['display_name'] for e in entities)


def _to_dict(
  entities: list
) -> dict:
  """
  Build a mapping from each entity’s unique ID to its display name.

  Processes a list of OpenAlex entity records, each of which must include
  the keys 'id' (a unique URI string) and 'display_name' (the human-readable
  name). Returns a dictionary where each key is an entity ID and each value
  is the corresponding display name.

  Args:
    entities (list of dict): A list of entity JSON dicts, each containing:
      - 'id' (str): The unique identifier (e.g. "https://openalex.org/A12345").
      - 'display_name' (str): The name to display for that entity.

  Returns:
    dict[str, str]: A dictionary mapping entity IDs to their display names.

  Raises:
    KeyError: If any entity dict is missing the 'id' or 'display_name' key.
  """
  return {e['id']: e['display_name'] for e in entities}


def _to_dataframe(
  entities: list,
  df_cols = None,
  override_defaults: bool = False
) -> pd.DataFrame:
  """
  Construct a pandas DataFrame from a sequence of OpenAlex entity records.

  This utility converts each entity (a JSON-like dict) into a row in a DataFrame.
  By default, every row will include the columns:
    - 'id'             : the entity’s unique identifier (e.g. URL)
    - 'display_name'   : the human-readable name
    - 'occurrences'    : a numeric count attached to the entity

  Additional fields may be included or renamed via `df_cols`:
    - If `df_cols` is a string, that single field is added as a new column.
    - If `df_cols` is a list, each named field is added as its own column.
    - If `df_cols` is a dict, its keys are JSON fields to extract and its
      values are the desired column names in the DataFrame.

  If `override_defaults` is True, the default columns ('id', 'display_name',
  'occurrences') are omitted, and **only** the fields specified in `df_cols`
  will appear.

  Args:
    entities (list of dict): Each dict represents an OpenAlex entity and may
      contain arbitrary keys. At minimum, default mode expects 'id',
      'display_name', and 'occurrences'.
    df_cols (str, list, or dict, optional): Specifies extra fields to include:
      - str: single field name to pull in
      - list of str: multiple field names
      - dict: mapping from JSON key → desired column name
      If None, no extra fields beyond the defaults are added.
    override_defaults (bool): If True, drop the default columns and include
      only those specified in `df_cols`. In override mode, `df_cols` **must**
      be non-empty.

  Returns:
    pandas.DataFrame: A DataFrame whose columns consist of the default columns
      (unless overridden) plus any additional columns requested via `df_cols`.

  Raises:
    ValueError: If `override_defaults` is True but `df_cols` is None or empty.
  """
  rows = []
  for e in entities:
    if override_defaults:
      if (not df_cols) or (len(df_cols) == 0):
        raise ValueError("`override_defaults` is True, so `df_cols` must include at least one attribute.")
      row = {}
    else:
      row = {'id': e.get('id'), 'display_name': e.get('display_name'), 'occurrences': e.get('occurrences')}
    if df_cols:
      if isinstance(df_cols, str):
        row[df_cols] = e.get(df_cols)
      elif hasattr(df_cols, '__iter__') and not isinstance(df_cols, dict):
        for f in df_cols:
          row[f] = e.get(f)
      elif isinstance(df_cols, dict):
        for k, v in df_cols.items():
          row[v] = e.get(k)
    rows.append(row)
  return pd.DataFrame(rows)


# --- Primary Functions ---
def _format_output(
  entities,
  output: str,
  **kwargs
):
# def _format_output(
#     entities: list[dict],
#     output: str,
#     **kwargs
# ) -> Union[set, dict, "pandas.DataFrame"]:
  """
  Convert a list of entity records into the specified output format.

  Parameters
  ----------
  entities : list of dict
    A list of JSON-like entity records to format.
  output : str
    Specifies the desired output type. Must be one of:
    - 'set': return a set of entity identifiers (via `_to_set`)
    - 'dict': return a dict mapping entity identifiers to records (via `_to_dict`)
    - 'df' : return a pandas DataFrame (via `_to_dataframe`)
  **kwargs : optional
    Additional keyword arguments forwarded to `_to_dataframe()` when `output='df'`. Supported keys:
    - df_cols (str, list of str, or dict): fields to include or mapping of JSON keys to column names.
      Required if `output='df'`.
    - override_defaults (bool): if True, omit the default columns (`id`, `display_name`, `occurrences`)
      and only include those specified in `df_cols`.

  Returns
  -------
  set or dict or pandas.DataFrame
    - set: a set of entity IDs
    - dict: a mapping of entity IDs to their record dicts
    - pandas.DataFrame: a DataFrame containing the requested columns

  Raises
  ------
  ValueError
    If `output` is not one of 'set', 'dict', or 'df'.
  """
  if output == 'set':
    return _to_set(entities)
  elif output == 'dict':
    return _to_dict(entities)
  elif output == 'df':
    return _to_dataframe(entities, **kwargs)
  else:
    raise ValueError(f"Unsupported output type: {output}")


def get_coauthors(
  author_id,
  output: str = 'set', 
  fields = None,
  session: requests.Session = None,
  keep_works: bool = False,
  work_fields = None,
  **kwargs
):
  """
  Retrieve and format the coauthor network for a given author.

  Parameters
  ----------
  author_id : str or dict
    The target author identifier. Can be:
    - ORCID string (e.g., "0000-0001-2345-6789")
    - OpenAlex URL or ID (e.g., "https://openalex.org/A1234567890" or "A1234567890")
    - A JSON-like dict representing an OpenAlex author record.
  output : {'set', 'dict', 'df'}, default 'set'
    The desired return format:
    - 'set': return a set of coauthor IDs.
    - 'dict': return a dict mapping coauthor IDs to author record dicts,
      each augmented with:
        • 'linking_works' (list of work IDs or work dicts)
        • 'occurrences' (int count of shared works)
    - 'df' : return a pandas.DataFrame with one row per coauthor,
      including columns for requested fields plus 'linking_works' and 'occurrences'.
  fields : str or list of str, optional
    Fields to fetch for each coauthor record from the API. If None, defaults
    to all available fields.
  session : requests.Session, optional
    An existing `requests.Session` to reuse for HTTP requests. If None,
    a temporary session will be created and closed automatically.
  keep_works : bool, default False
    Whether to include the list of linking works for each coauthor under
    the key 'linking_works'. If False, only counts are recorded.
  work_fields : str or list of str, optional
    Fields to fetch when retrieving works to build the coauthorship links.
    Used only if `keep_works=True`. If None, only work IDs are fetched.
  **kwargs : dict, optional
    Additional keyword arguments forwarded to `_format_output()` when
    `output='df'`. Supported keys include:
    - `df_cols`: fields or mapping of JSON keys to DataFrame columns (required
      if `output='df'`)
    - `override_defaults`: bool flag to override default columns in DataFrame.

  Returns
  -------
  set or dict or pandas.DataFrame
    - set: a set of coauthor IDs.
    - dict: mapping from coauthor ID to author record dict with
      'linking_works' and 'occurrences'.
    - pandas.DataFrame: rows of coauthor information, including
      one column per requested field plus 'linking_works' and 'occurrences'.

  Raises
  ------
  ValueError
    If `output` is not one of 'set', 'dict', or 'df', or if required
    DataFrame parameters (e.g., `df_cols`) are missing when `output='df'`.
  """
  with _conditional_session(session) as _session:
    author_id = standardize_author_ids(author_id, session=_session)
    _work_fields = work_fields if (keep_works and work_fields) else ['id']
    works = fetch_works_from_author(author_id, fields=_work_fields, session=_session)
    works = {wk['id']: wk for wk in works}
    work_ids = list(works.keys())
    coauthors_per_work = fetch_authors_from_works(work_ids, fields=fields, session=_session, keep_parent_ids=True)
  coauthors = {}
  for wid in tqdm(work_ids, desc="Building coauthor-work link network"):
    wk = works[wid] if keep_works else wid
    for cauth in coauthors_per_work[wid]:
      caid = cauth['id']
      if coauthors.get(caid):
        coauthors[caid]['linking_works'].append(wk)
        coauthors[caid]['occurrences'] += 1
      else:
        coauthors[caid] = cauth
        coauthors[caid]['linking_works'] = [wk]
        coauthors[caid]['occurrences'] = 1
  return _format_output(list(coauthors.values()), output, **kwargs)


def get_citing_authors(
  author_id,
  output: str = 'set',
  fields = None,
  session: requests.Session = None,
  keep_works: bool = False,
  work_fields = None,
  **kwargs
):
  """
  Retrieve and format the authors who cite a given target author.

  Parameters
  ----------
  author_id : str or dict
    Identifier for the target author. Can be:
    - ORCID string (e.g., "0000-0001-2345-6789")
    - OpenAlex URL or ID (e.g., "https://openalex.org/A1234567890" or "A1234567890")
    - A JSON-like dict representing an OpenAlex author record.
  output : {'set', 'dict', 'df'}, default 'set'
    Desired output format:
    - 'set': return a set of citing-author IDs.
    - 'dict': return a dict mapping each citing-author ID to its record dict, 
      augmented with:
        • 'linking_works': list of work IDs or work dicts that cite the target author
        • 'occurrences': number of citing works
    - 'df' : return a pandas.DataFrame with one row per citing author,
      including requested fields plus 'linking_works' and 'occurrences'.
  fields : str or list of str, optional
    Fields to fetch for each citing-author record. If None, defaults to all available fields.
  session : requests.Session, optional
    An existing requests.Session for connection reuse. If None, a temporary session is created.
  keep_works : bool, default False
    If True, include the list of linking works for each citing author under 'linking_works'.
    If False, only counts are recorded.
  work_fields : str or list of str, optional
    Fields to fetch when retrieving citing works. Used only if `keep_works=True`. 
    If None, only work IDs are fetched.
  **kwargs : dict, optional
    Additional keyword arguments forwarded to `_format_output()` when `output='df'`. Supported keys:
    - `df_cols`: fields or mapping of JSON keys to DataFrame columns (required if `output='df'`)
    - `override_defaults`: bool flag to override default columns in the DataFrame.

  Returns
  -------
  set or dict or pandas.DataFrame
    - set: a set of IDs of citing authors.
    - dict: mapping from citing-author ID to its record dict with
      'linking_works' and 'occurrences'.
    - pandas.DataFrame: rows of citing-author data, including requested fields,
      'linking_works', and 'occurrences'.

  Raises
  ------
  ValueError
    If `output` is not one of 'set', 'dict', or 'df', or if required DataFrame
    parameters (e.g., `df_cols`) are missing when `output='df'`.
  """
  with _conditional_session(session) as _session:
    author_id = standardize_author_ids(author_id, session=_session)
    _work_fields = work_fields if (keep_works and work_fields) else ['id']
    citing_works = fetch_citing_works_from_author(author_id, fields=_work_fields, session=_session)
    citing_works = {cwk['id']: cwk for cwk in citing_works}
    citing_work_ids = list(citing_works.keys())
    citing_authors_per_work = fetch_authors_from_works(citing_work_ids, fields=fields, session=_session, keep_parent_ids=True)
  citing_authors = {}
  for cwid in tqdm(citing_work_ids, desc="Building citing author–work link network"):
    cwk = citing_works[cwid] if keep_works else cwid
    for cauth in citing_authors_per_work[cwid]:
      caid = cauth['id']
      if citing_authors.get(caid):
        citing_authors[caid]['linking_works'].append(cwk)
        citing_authors[caid]['occurrences'] += 1
      else:
        citing_authors[caid] = cauth
        citing_authors[caid]['linking_works'] = [cwk]
        citing_authors[caid]['occurrences'] = 1
  return _format_output(list(citing_authors.values()), output, **kwargs)


def remove_author_list(
  authors,
  authors_subtract
):
  """
  Subtract one set of authors from another based on their IDs.

  Parameters
  ----------
  authors : pandas.DataFrame
    DataFrame of author records to filter. Must include an 'id' column.
  authors_subtract : pandas.DataFrame
    DataFrame of author records to remove. Must include an 'id' column.

  Returns
  -------
  pandas.DataFrame
    A new DataFrame containing only those rows from `authors` whose 'id'
    values are not present in `authors_subtract['id']`. The original index
    and column order of `authors` are preserved.

  Raises
  ------
  NotImplementedError
    If either `authors` or `authors_subtract` is not a pandas.DataFrame.

  Examples
  --------
  >>> import pandas as pd
  >>> authors = pd.DataFrame([{'id': 'A1', 'name': 'Alice'},
  ...                         {'id': 'B2', 'name': 'Bob'}])
  >>> to_remove = pd.DataFrame([{'id': 'B2', 'name': 'Bob'}])
  >>> remove_author_list(authors, to_remove)
     id   name
  0  A1  Alice
  """
  if all([isinstance(x, pd.DataFrame) for x in [authors, authors_subtract]]):
    return authors.loc[~authors["id"].isin(authors_subtract["id"])]
  else:
    raise NotImplementedError("Currently the only implemented object type is pd.DataFrame.")


def extract_dict_key_to_column(
  df: pd.DataFrame,
  dict_col: str,
  key: str,
  new_col: str = None,
  drop_origin: bool = False
) -> pd.DataFrame:
  """
  Extract a value for a given key from dictionaries in one DataFrame column
  into its own new column.

  Parameters
  ----------
  df : pandas.DataFrame
    Input DataFrame containing a column of dicts.
  dict_col : str
    Name of the column whose entries are dictionaries (or possibly other types).
  key : str
    The key to extract from each dictionary in `dict_col`.
  new_col : str, optional
    Name for the new column to hold extracted values.
    If None (default), the new column will be named the same as `key`.
  drop_origin : bool, default False
    If True, drop the original `dict_col` column after extraction.

  Returns
  -------
  pandas.DataFrame
    A DataFrame with the new column (`new_col`) containing the values
    extracted from each dict. Entries where the dict is missing,
    not a dict, or does not contain `key` will be NaN/None.
    If `drop_origin` is True, the original `dict_col` is removed.

  Examples
  --------
  >>> import pandas as pd
  >>> df = pd.DataFrame({'info': [{'x': 10}, {'x': 20}, None, {'y': 5}]})
  >>> extract_dict_key_to_column(df, 'info', 'x', new_col='value')
       info  value
  0  {'x': 10}   10
  1  {'x': 20}   20
  2     None   None
  3  {'y': 5}   None
  """
  if new_col is None:
    new_col = key
  # Use .apply so that missing or non‐dict entries safely give NaN
  df[new_col] = df[dict_col].apply(lambda d: d.get(key) if isinstance(d, dict) else None)
  if drop_origin:
    return df.drop(columns=[dict_col])
  return df


def list_by_sorted_hindex(
  df: pd.DataFrame,
  ttl: str = "Authors sorted by h-index",
  max_count: int = 25,
  min_occurrences: int = 1
) -> None:
  """
  Print a Markdown table of authors ranked by their h-index.

  Parameters
  ----------
  df : pandas.DataFrame
    DataFrame containing at least the following columns:
    - 'display_name' (str): the author’s name
    - 'h_index' (int): the author’s h-index
    - 'occurrences' (int): number of works linking to the target author
    The DataFrame must be sorted in descending order by 'h_index' and
    should have a monotonically increasing index (used as the rank).
  ttl : str, default "Authors sorted by h-index"
    Markdown header title printed above the table.
  max_count : int, default 25
    Maximum number of authors (rows) to display.
  min_occurrences : int, default 1
    Minimum value of 'occurrences' required for an author to be included.

  Returns
  -------
  None
    Prints the table via IPython.display.Markdown and returns None.

  Examples
  --------
  >>> # Assume df is sorted by h_index descending and has 'display_name', 'h_index', 'occurrences'
  >>> list_by_sorted_hindex(df, ttl="Top Coauthors", max_count=10, min_occurrences=2)
  """
  # Initialize
  df_height = len(df)
  max_count = min(df_height, max_count) if max_count else df_height
  if max_count < 1:
    raise ValueError(f"`max_count` ({max_count}) cannot be < 0.")
  min_occurrences = min_occurrences if min_occurrences else 1
  if min_occurrences < 1:
    raise ValueError(f"`min_occurrences` ({min_occurrences}) cannot be < 0.")
  enforce_min = lambda c: max(len(c), 3)
  md_str = f"### {ttl}\n\n"
  # Parse dataframe from formatting info
  max_rank = len(df)
  col1 = f"{'Rank':{len(str(max_rank))}}"
  col1_w = enforce_min(col1)
  col1_sep = '-' * col1_w
  max_name_length = df['display_name'].str.len().max()
  col2 = f"{'Author':{max_name_length}}"
  col2_w = enforce_min(col2)
  col2_sep = '-' * col2_w
  max_hindex = df['h_index'].max()
  col3 = f"{'h-index':{len(str(max_hindex))}}"
  col3_w = enforce_min(col3)
  col3_sep = '-' * col3_w
  max_occurrences = df['occurrences'].max()
  col4 = f"{'occurrences':{len(str(max_occurrences))}}"
  col4_w = enforce_min(col4)
  col4_sep = '-' * col4_w
  # Define Markdown table
  md_str += f"| {col1:{col1_w}} | {col2:{col2_w}} | {col3:{col3_w}} | {col4:{col4_w}} |\n"
  md_str += f"| {col1_sep:{col1_w}} | {col2_sep:{col2_w}} | {col3_sep:{col3_w}} | {col4_sep:{col4_w}} |\n"
  count = 1
  for idx, auth in df.iterrows():
    if count > max_count:
      break
    if auth['occurrences'] < min_occurrences:
      continue
    md_str += f"| {idx:{col1_w}} | {auth['display_name']:{col2_w}} | {auth['h_index']:{col3_w}} | {auth['occurrences']:{col4_w}} |\n"
    count += 1
  md_str = md_str[:-1] # remove final '\n'
  display(Markdown(md_str))
  return

