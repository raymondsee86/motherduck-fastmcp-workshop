"""Your MCP server for San Francisco 311 data.

FastMCP Gateway runs this file. Its entrypoint setting is `server.py:mcp`.
"""

import logging
import re
from collections import Counter, defaultdict
from statistics import median
from typing import Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from sf311.data import BY_ID, LOCAL_NAMES, NEIGHBORHOODS, REQUESTS, SERVICES, Request, closest
from sf311.kit import caller_email

mcp = FastMCP("sf-311")
log = logging.getLogger("sf311")
log.setLevel(logging.INFO)
if not log.handlers:
    # Without a handler, INFO lines never reach the deployment's server logs.
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.addHandler(_handler)

MAX_ROWS = 50
MIN_CLOSED_FOR_MEDIAN = 10
READ_ONLY = {"readOnlyHint": True}
HOUSE_NUMBER = re.compile(r"^(\d+)\s+([^,]+)")

GROUPS = {
    "neighborhood": lambda r: r.neighborhood,
    "service": lambda r: r.service,
    "agency": lambda r: r.agency,
    "source": lambda r: r.source,
    "weekday": lambda r: r.opened.strftime("%A"),
    "status": lambda r: r.status,
}

CountGroup = Literal["neighborhood", "service", "agency", "source", "weekday", "status"]
MedianGroup = Literal["neighborhood", "agency"]


def _check_name(value: str | None, choices: list[str], kind: str) -> None:
    if value is None or value in choices:
        return
    suggestions = closest(value, choices)
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise ToolError(f"Unknown {kind} {value!r}.{hint} Call dataset_info for the exact {kind} names.")


def _mask_address(address: str) -> str:
    """Cut a street address down to its block: "994 HARRISON ST, ..." becomes "900 block of HARRISON ST".

    An address with no house number, such as an intersection, is returned as it is.
    """
    match = HOUSE_NUMBER.match(address)
    if not match:
        return address
    block = int(match.group(1)) // 100 * 100
    return f"{block} block of {match.group(2).strip()}"


def _public(request: Request) -> dict:
    """Return a request as a dict that is safe to show the model, with the address masked to the block."""
    row = request.as_dict()
    row["address"] = _mask_address(request.address)
    return row


def _matching(service: str | None = None, neighborhood: str | None = None, status: str | None = None) -> list[Request]:
    _check_name(service, SERVICES, "service")
    _check_name(neighborhood, NEIGHBORHOODS, "neighborhood")
    return [
        r
        for r in REQUESTS
        if (service is None or r.service == service)
        and (neighborhood is None or r.neighborhood == neighborhood)
        and (status is None or r.status == status)
    ]


@mcp.tool(annotations=READ_ONLY)
def dataset_info() -> dict:
    """Describe the 311 dataset this server answers questions about.

    Call this first to learn the date range and the exact service and
    neighborhood names that other tools accept.
    """
    return {
        "description": "San Francisco 311 service requests filed in one week.",
        "first_opened": min(request.opened for request in REQUESTS).isoformat(),
        "last_opened": max(request.opened for request in REQUESTS).isoformat(),
        "requests": len(REQUESTS),
        "services": SERVICES,
        "neighborhoods": NEIGHBORHOODS,
        "local_neighborhood_names": len(LOCAL_NAMES),
    }


@mcp.tool(annotations=READ_ONLY)
def search_requests(
    service: str | None = None,
    neighborhood: str | None = None,
    status: Literal["Open", "Closed"] | None = None,
    limit: int = 20,
) -> list[dict]:
    """List individual 311 requests that match the filters, newest first.

    Use this to see examples: locations, subtypes, open and close times.
    Addresses are given only to the block, such as "900 block of HARRISON ST";
    exact house numbers are never available.
    Returns at most 50 rows, so never use it to count or to compute times:
    use count_requests for "how many" questions and median_close_hours for
    "how long" questions. Use get_request when you already have an ID.

    service and neighborhood must be exact names from dataset_info.
    """
    if not 1 <= limit <= MAX_ROWS:
        raise ToolError(f"limit must be between 1 and {MAX_ROWS}. For totals, use count_requests instead.")
    rows = sorted(_matching(service, neighborhood, status), key=lambda r: r.opened, reverse=True)
    return [_public(r) for r in rows[:limit]]


@mcp.tool(annotations=READ_ONLY)
def count_requests(
    service: str | None = None,
    neighborhood: str | None = None,
    group_by: CountGroup | None = None,
) -> int | dict[str, int]:
    """Count 311 requests this week, optionally broken down by one field.

    Without group_by, returns one number. With group_by, returns a dict of
    group to count, largest first, at most 50 groups; requests with a blank
    value for that field are left out. Use this for "how many", "which is
    most common", and "busiest day" questions. For time to close, use
    median_close_hours instead.

    service and neighborhood must be exact names from dataset_info.
    """
    rows = _matching(service, neighborhood)
    if group_by is None:
        return len(rows)
    key = GROUPS[group_by]
    counts = Counter(key(r) for r in rows if key(r))
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return dict(ranked[:MAX_ROWS])


@mcp.tool(annotations=READ_ONLY)
def median_close_hours(
    service: str | None = None,
    group_by: MedianGroup | None = None,
) -> float | None | dict[str, float]:
    """Median hours from opened to closed for closed 311 requests, rounded to 0.1.

    Only closed requests count; open ones are ignored. Without group_by,
    returns one number, or null when nothing matching has closed. With
    group_by, returns a dict of group to median hours, fastest first,
    leaving out groups with fewer than 10 closed requests. Use this for
    "how long" and "which is fastest or slowest" questions. For volumes,
    use count_requests.

    service must be an exact name from dataset_info.
    """
    rows = [r for r in _matching(service) if r.hours_to_close is not None]
    if group_by is None:
        return round(median(r.hours_to_close for r in rows), 1) if rows else None
    key = GROUPS[group_by]
    hours: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if key(r):
            hours[key(r)].append(r.hours_to_close)
    medians = {
        group: round(median(values), 1) for group, values in hours.items() if len(values) >= MIN_CLOSED_FOR_MEDIAN
    }
    return dict(sorted(medians.items(), key=lambda item: (item[1], item[0]))[:MAX_ROWS])


@mcp.tool(annotations=READ_ONLY)
def get_request(request_id: str) -> dict:
    """Return every field of one 311 request by its request_id.

    Use this when you already have an ID, for example from search_requests.
    To find requests by service or neighborhood, use search_requests.
    The address is given only to the block, such as "900 block of HARRISON ST".
    """
    log.info("get_request request_id=%s caller=%s", request_id, caller_email() or "local")
    request = BY_ID.get(request_id.strip())
    if request is None:
        raise ToolError(f"No request with ID {request_id!r}. Use search_requests to find real IDs.")
    return _public(request)


if __name__ == "__main__":
    mcp.run()
