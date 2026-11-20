#!/usr/bin/env python
# coding: utf-8
"""
Create STAC Items for Sentinel-2 TerraMind embedding GeoTIFFs.

The container listing is paged (Swift ``marker`` pagination) and one STAC Item
is written per GeoTIFF. Items follow the STAC Embedding extension as laid out
in catalog_guide.ipynb (section 3.2):
https://github.com/jonas-hurst/store-and-catalog-earth-embeddings

Only file headers are read: GDAL opens each GeoTIFF over /vsicurl/ and reads
the geotransform, CRS, band count and data type. Raster blocks are never
decoded.

The parent STAC Collection the Items point at is written alongside them, with
its extents aggregated over the Items (section 3.3 of the notebook).

Usage:
    python create_stac_items.py --output-dir out --max-pages 3

The --output-dir, --page-size, --max-pages and --workers options default to the
environment variables output_dir, page_size, max_pages and workers. This lets
c3_create_operator expose them as component inputs.
"""

import argparse
import json
import logging
import os
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial

# pip install gdal==3.10.3.post1 --extra-index-url https://gitlab.com/api/v4/projects/61637378/packages/pypi/simple
from osgeo import gdal, osr

log = logging.getLogger(__name__)

BASE_URL = "https://s3.waw4-1.cloudferro.com/swift/v1/embed2scale-terramind"
COLLECTION_ID = "terramind_embeddings"
EMBEDDING_SUFFIX = "_embed2scale_terramind_embeddings"
S2_ITEMS_URL = (
    "https://stac.dataspace.copernicus.eu/v1/collections/sentinel-2-l2a/items"
)
STAC_VERSION = "1.1.0"
EMBEDDINGS_EXTENSION = "https://stac-extensions.github.io/embeddings/v1.0.0/schema.json"
PROJECTION_EXTENSION = "https://stac-extensions.github.io/projection/v2.1.0/schema.json"
STAC_EXTENSIONS = [EMBEDDINGS_EXTENSION, PROJECTION_EXTENSION]

COLLECTION_TITLE = "TerraMind-Sentinel-2-tokenizer-embeddings"
COLLECTION_DESCRIPTION = (
    "A STAC Collection of Sentinel embeddings, produced with the TerraMind Tokenizer."
)
COLLECTION_KEYWORDS = ["terramind", "embeddings"]
COLLECTION_PROVIDERS = [
    {
        "url": "https://embed2scale.eu",
        "name": "Embed2Scale",
        "roles": ["producer", "processor"],
        "description": "EU Horizon Embed2Scale Project, GA Number 101131841",
    }
]
# Embedding properties promoted to the Collection when every Item agrees on them.
EMB_PROPERTY_KEYS = ("emb:type", "emb:dimensions", "emb:chip_size", "emb:stride")

# Sentinel-2 acquisition time inside the product name, e.g. S2A_MSIL2A_20170102T175732_
_ACQUISITION_RE = re.compile(r"\d{8}T\d{6}")


def _list_pages(base_url, page_size, max_pages):
    """Yield (page_number, entries) for successive pages of the container listing."""
    marker = None
    page = 0
    while max_pages is None or page < max_pages:
        query = {"format": "json", "limit": page_size}
        if marker:
            query["marker"] = marker
        with urllib.request.urlopen(
            f"{base_url}/?{urllib.parse.urlencode(query)}", timeout=60
        ) as resp:
            entries = json.load(resp)
        if not entries:
            return
        page += 1
        yield page, entries
        if len(entries) < page_size:
            return
        marker = entries[-1]["name"]


def _read_header(url):
    """Read raster header fields over /vsicurl/ without decoding any pixels."""
    ds = gdal.Open(f"/vsicurl/{url}")
    wkt = ds.GetProjection()
    if not wkt:
        raise ValueError(f"No embedded CRS in {url}")

    srs = osr.SpatialReference(wkt=wkt)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)
    wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    to_wgs84 = osr.CoordinateTransformation(srs, wgs84)

    gt = ds.GetGeoTransform()
    minx, maxy = gt[0], gt[3]
    maxx = minx + gt[1] * ds.RasterXSize
    miny = maxy + gt[5] * ds.RasterYSize
    corners = [
        to_wgs84.TransformPoint(x, y)[:2]
        for x, y in [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
    ]
    lons = [lon for lon, _ in corners]
    lats = [lat for _, lat in corners]

    return {
        "epsg": srs.GetAuthorityCode(None),
        "bbox": [min(lons), min(lats), max(lons), max(lats)],
        "grid_m": abs(gt[1]),
        "bands": ds.RasterCount,
        "dtype": gdal.GetDataTypeName(ds.GetRasterBand(1).DataType).lower(),
        "patch_size": ds.GetMetadataItem("patch_size"),
    }


def _bbox_polygon(bbox):
    minx, miny, maxx, maxy = bbox
    ring = [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]
    return {"type": "Polygon", "coordinates": [ring]}


def _build_item(base_url, name):
    """Build the STAC Item dict for one embedding GeoTIFF in the container."""
    stem = name.removesuffix(".tif")
    header = _read_header(f"{base_url}/{name}")

    match = _ACQUISITION_RE.search(stem)
    if match is None:
        raise ValueError(f"No acquisition time in file name {name}")
    acquired = datetime.strptime(match.group(0), "%Y%m%dT%H%M%S")

    properties = {
        "datetime": acquired.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gsd": header["grid_m"],
        "data_type": header["dtype"],
        "title": "Terramind Embedding",
        "description": "Terramind embedding from Sentinel-2-L2A imagery.",
        "license": "CC-BY-4.0",
        "emb:type": "chip",
        "emb:dimensions": [header["bands"]],
    }
    if header["epsg"]:
        properties["proj:code"] = f"EPSG:{header['epsg']}"
    if header["patch_size"]:
        # 256 px chips at 10 m equal the 2560 m pixel pitch, so chips do not overlap.
        patch = int(header["patch_size"])
        properties["emb:chip_size"] = [patch, patch]
        properties["emb:stride"] = [patch, patch]

    s2_id = stem.removesuffix(EMBEDDING_SUFFIX)
    return {
        "stac_version": STAC_VERSION,
        "stac_extensions": STAC_EXTENSIONS,
        "type": "Feature",
        "id": stem,
        "collection": COLLECTION_ID,
        "links": [
            {"rel": "self", "href": f"./{stem}.json", "type": "application/json"},
            {
                "rel": "derived_from",
                "href": f"{S2_ITEMS_URL}/{s2_id}",
                "type": "application/json",
                "title": "Sentinel-2 image from which this embedding was computed.",
            },
            {
                "rel": "collection",
                "href": f"./{COLLECTION_ID}.json",
                "type": "application/json",
            },
        ],
        "bbox": header["bbox"],
        "geometry": _bbox_polygon(header["bbox"]),
        "properties": properties,
        "assets": {
            "embeddings": {
                "href": f"{base_url}/{name}",
                "title": "Terramind Embeddings",
                "type": "image/tiff; application=geotiff",
                "roles": ["embeddings"],
            }
        },
    }


def _write_item(base_url, name, output_dir):
    """Build and write one Item. Returns the Item, or None if it failed."""
    try:
        item = _build_item(base_url, name)
    except Exception:
        log.exception("Skipping %s", name)
        return None
    path = os.path.join(output_dir, f"{item['id']}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(item, fh, indent=4)
    return item


def _common_value(items, key):
    """Return the value every Item shares for ``key``, or None otherwise.

    None is also returned if any Item lacks ``key``.
    """
    if any(key not in item["properties"] for item in items):
        return None
    seen = {json.dumps(item["properties"][key], sort_keys=True) for item in items}
    if len(seen) != 1:
        return None
    return json.loads(seen.pop())


def _write_collection(items, output_dir):
    """Write the parent Collection, with extents aggregated over ``items``."""
    lons = [c for item in items for c in (item["bbox"][0], item["bbox"][2])]
    lats = [c for item in items for c in (item["bbox"][1], item["bbox"][3])]
    times = sorted(item["properties"]["datetime"] for item in items)

    collection = {
        "id": COLLECTION_ID,
        "stac_version": STAC_VERSION,
        # The projection extension is left out: the Items span many UTM zones,
        # so there is no single proj:code to state at Collection level.
        "stac_extensions": [EMBEDDINGS_EXTENSION],
        "type": "Collection",
        "title": COLLECTION_TITLE,
        "description": COLLECTION_DESCRIPTION,
        "license": "CC-BY-4.0",
        "keywords": COLLECTION_KEYWORDS,
        "providers": COLLECTION_PROVIDERS,
        "extent": {
            "spatial": {"bbox": [[min(lons), min(lats), max(lons), max(lats)]]},
            "temporal": {"interval": [[times[0], times[-1]]]},
        },
        "links": [
            {
                "rel": "self",
                "href": f"./{COLLECTION_ID}.json",
                "type": "application/json",
            },
            *(
                {
                    "rel": "item",
                    "href": f"./{item['id']}.json",
                    "type": "application/json",
                }
                for item in items
            ),
        ],
        "item_assets": {
            "embeddings": {
                "type": "image/tiff; application=geotiff",
                "roles": ["embeddings"],
                "title": "TerraMind Embeddings",
            }
        },
    }

    for key in ("data_type", *EMB_PROPERTY_KEYS):
        value = _common_value(items, key)
        if value is not None:
            collection[key] = value

    path = os.path.join(output_dir, f"{COLLECTION_ID}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(collection, fh, indent=4)
    return path


def run(base_url=BASE_URL, output_dir="out", page_size=1000, max_pages=None, workers=8):
    """
    Write one STAC Item per GeoTIFF on the first ``max_pages`` listing pages,
    plus the Collection grouping them.

    Returns
    -------
    tuple[int, int]
        ``(written, failed)`` item counts.
    """
    gdal.UseExceptions()
    osr.UseExceptions()
    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    gdal.SetConfigOption("GDAL_HTTP_TIMEOUT", "60")
    base_url = base_url.rstrip("/")
    os.makedirs(output_dir, exist_ok=True)

    written = failed = 0
    items = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for page, entries in _list_pages(base_url, page_size, max_pages):
            names = [
                e["name"]
                for e in entries
                if e["name"].endswith(EMBEDDING_SUFFIX + ".tif")
            ]
            results = list(
                pool.map(partial(_write_item, base_url, output_dir=output_dir), names)
            )
            page_items = [item for item in results if item is not None]
            items.extend(page_items)
            written += len(page_items)
            failed += len(results) - len(page_items)
            log.info(
                "Page %d: %d objects, %d GeoTIFFs, %d written, %d failed",
                page,
                len(entries),
                len(names),
                len(page_items),
                len(results) - len(page_items),
            )

    if items:
        log.info("Collection written to %s", _write_collection(items, output_dir))
    return written, failed


def main():
    parser = argparse.ArgumentParser(
        description="Create STAC Items for TerraMind embedding GeoTIFFs."
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("output_dir", "out"),
        help="Directory for the <id>.json Items.",
    )
    parser.add_argument(
        "--base-url",
        default=BASE_URL,
        help="Container URL serving the GeoTIFFs and their JSON listing.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=os.environ.get("page_size", 1000),
        help="Objects per listing page.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=os.environ.get("max_pages"),
        help="Stop after this many listing pages (default: all pages).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.environ.get("workers", 8),
        help="Concurrent header reads per page.",
    )
    args, unknown = parser.parse_known_args()
    # CLAIMED passes inputs as name=value; the setup code already put them in
    # the environment (read as defaults above), so only reject anything else.
    unknown = [a for a in unknown if not re.fullmatch(r"\w+=.*", a)]
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")

    logging.basicConfig(level=logging.INFO)
    written, failed = run(
        args.base_url, args.output_dir, args.page_size, args.max_pages, args.workers
    )
    log.info(
        "Done: %d STAC Items written to %s, %d failed",
        written,
        args.output_dir,
        failed,
    )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
