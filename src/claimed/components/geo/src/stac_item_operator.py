#!/usr/bin/env python
# coding: utf-8
"""
STAC Item Operator

Converts a GeoTIFF embedding into a STAC Item JSON by populating a
provided template with spatial, temporal, and asset metadata extracted
directly from the file.
"""

import copy
import json
import logging
import os
import re
from datetime import datetime, timezone

from osgeo import gdal, osr

# ---------------------------------------------------------------------------
# Sentinel-2 MGRS tile origin lookup  (EPSG:32613 easting/northing, metres)
# ---------------------------------------------------------------------------
# Each entry: MGRS_TILE_ID → (epsg, ul_easting, ul_northing, pixel_size_m)
# Values sourced from the official ESA Sentinel-2 tiling grid.
# Extend this dict as additional tiles are encountered.
_S2_TILE_ORIGINS: dict[str, tuple[int, float, float, int]] = {
    "T13TEF": (32613, 399960.0, 4000020.0, 10),
}

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Filename pattern for embedding pixel windows:
# …_T<tile>_…__<col_start>-<col_end>_<row_start>-<row_end>_embedding…
_TILE_RE   = re.compile(r'_(T[0-9]{2}[A-Z]{3})_')
_WINDOW_RE = re.compile(r'__(\d+)-(\d+)_(\d+)-(\d+)_embedding')


def _parse_spatial_from_filename(filename: str) -> tuple:
    """
    Derive CRS, bounding box, and GeoJSON geometry from a Sentinel-2
    embedding filename when the TIFF has no embedded projection.

    The filename is expected to contain:
      - An MGRS tile ID of the form  ``_T<zone><letters>_``
        (e.g. ``T13TEF``).
      - A pixel-window suffix of the form
        ``__<col_start>-<col_end>_<row_start>-<row_end>_embedding``.

    Coordinates are computed in the tile's native UTM CRS and then
    reprojected to WGS 84 for the returned bbox and geometry, because
    STAC requires geographic (lon/lat) coordinates in those fields.

    Parameters
    ----------
    filename : str
        Basename (or full path) of the embedding TIFF.

    Returns
    -------
    tuple[list, dict, int]
        ``(bbox_wgs84, geometry_wgs84, epsg)`` where *bbox_wgs84* is
        ``[minx, miny, maxx, maxy]`` in WGS 84 degrees and *epsg* is
        the integer EPSG code of the tile's UTM zone.

    Raises
    ------
    ValueError
        If the tile ID is not found in the lookup table, or if the
        filename does not match the expected pattern.
    """
    tile_m = _TILE_RE.search(filename)
    win_m  = _WINDOW_RE.search(filename)
    if not tile_m or not win_m:
        raise ValueError(
            f"Cannot parse spatial metadata from filename '{filename}'. "
            "Expected pattern: …_T<tile>_…__<col_start>-<col_end>_"
            "<row_start>-<row_end>_embedding…"
        )

    tile_id = tile_m.group(1)
    if tile_id not in _S2_TILE_ORIGINS:
        raise ValueError(
            f"Tile '{tile_id}' is not in the _S2_TILE_ORIGINS lookup table. "
            "Add its (epsg, ul_easting, ul_northing, pixel_size_m) entry."
        )

    epsg, ul_x, ul_y, px = _S2_TILE_ORIGINS[tile_id]
    col_start, col_end = int(win_m.group(1)), int(win_m.group(2))
    row_start, row_end = int(win_m.group(3)), int(win_m.group(4))

    # UTM corners (y decreases southward from the upper-left origin)
    utm_minx = ul_x + col_start * px
    utm_maxx = ul_x + col_end   * px
    utm_maxy = ul_y - row_start * px
    utm_miny = ul_y - row_end   * px

    # Reproject the four corners to WGS 84 (STAC bbox must be lon/lat)
    srs_utm = osr.SpatialReference()
    srs_utm.ImportFromEPSG(epsg)
    srs_wgs = osr.SpatialReference()
    srs_wgs.ImportFromEPSG(4326)
    srs_wgs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    to_wgs = osr.CoordinateTransformation(srs_utm, srs_wgs)

    corners_utm = [
        (utm_minx, utm_miny),
        (utm_maxx, utm_miny),
        (utm_maxx, utm_maxy),
        (utm_minx, utm_maxy),
    ]
    corners_wgs = []
    for cx, cy in corners_utm:
        lon, lat, _ = to_wgs.TransformPoint(cx, cy)
        corners_wgs.append((lon, lat))

    lons = [c[0] for c in corners_wgs]
    lats = [c[1] for c in corners_wgs]
    bbox_wgs84 = [min(lons), min(lats), max(lons), max(lats)]

    ring = corners_wgs + [corners_wgs[0]]  # close the ring
    geometry_wgs84 = {
        "type": "Polygon",
        "coordinates": [[[lon, lat] for lon, lat in ring]],
    }

    return bbox_wgs84, geometry_wgs84, epsg


def _compute_bbox_and_geometry(
    gt: tuple, width: int, height: int
) -> tuple:
    """
    Derive a bounding box and GeoJSON Polygon from a GDAL GeoTransform.

    Parameters
    ----------
    gt : tuple
        GDAL GeoTransform – (originX, pixelW, 0, originY, 0, pixelH).
    width : int
        Raster width in pixels.
    height : int
        Raster height in pixels.

    Returns
    -------
    tuple[list, dict]
        ``(bbox, geometry)`` where *bbox* is ``[minx, miny, maxx, maxy]``
        and *geometry* is a GeoJSON Polygon dict.
    """
    minx = gt[0]
    maxy = gt[3]
    maxx = minx + gt[1] * width
    miny = maxy + gt[5] * height   # gt[5] is negative → miny < maxy

    bbox = [minx, miny, maxx, maxy]
    geometry = {
        "type": "Polygon",
        "coordinates": [[
            [minx, miny],
            [maxx, miny],
            [maxx, maxy],
            [minx, maxy],
            [minx, miny],
        ]],
    }
    return bbox, geometry


def _extract_tiff_metadata(tiff_path: str) -> dict:
    """
    Extract spatial and band metadata from a GeoTIFF using GDAL.

    For standard georeferenced TIFFs the CRS, bbox, and geometry are read
    directly from the file.  For Sentinel-2 embedding TIFFs that carry no
    embedded projection, they are derived from the MGRS tile ID and pixel
    window encoded in the filename via :func:`_parse_spatial_from_filename`.

    Returns a dict with:
        bbox     – [minx, miny, maxx, maxy] in WGS 84 (lon/lat)
        geometry – GeoJSON Polygon in WGS 84
        epsg     – integer EPSG code of the tile's native UTM CRS
        bands    – number of raster bands
        dtype    – GDAL data-type name of band 1

    Raises
    ------
    ValueError
        If GDAL cannot open the file, the CRS cannot be determined from
        the file or filename, or the tile is not in the lookup table.
    """
    ds = gdal.Open(tiff_path)
    if ds is None:
        raise ValueError(f"GDAL could not open '{tiff_path}'")

    try:
        wkt = ds.GetProjection()
        srs = osr.SpatialReference(wkt=wkt)
        epsg = None
        has_crs = bool(wkt) and (srs.IsProjected() or srs.IsGeographic())

        if has_crs:
            # Prefer authority code; fall back to None (still georeferenced).
            code = srs.GetAuthorityCode(None)
            if code:
                epsg = int(code)

            # Always reproject to WGS 84 for STAC bbox/geometry.
            srs_wgs = osr.SpatialReference()
            srs_wgs.ImportFromEPSG(4326)
            srs_wgs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

            gt = ds.GetGeoTransform()
            w, h = ds.RasterXSize, ds.RasterYSize

            # Compute corners in native CRS, then transform to WGS 84.
            native_minx = gt[0]
            native_maxy = gt[3]
            native_maxx = native_minx + gt[1] * w
            native_miny = native_maxy + gt[5] * h

            corners_native = [
                (native_minx, native_miny),
                (native_maxx, native_miny),
                (native_maxx, native_maxy),
                (native_minx, native_maxy),
            ]

            to_wgs = osr.CoordinateTransformation(srs, srs_wgs)
            corners_wgs = []
            for cx, cy in corners_native:
                lon, lat, _ = to_wgs.TransformPoint(cx, cy)
                corners_wgs.append((lon, lat))

            lons = [c[0] for c in corners_wgs]
            lats = [c[1] for c in corners_wgs]
            bbox = [min(lons), min(lats), max(lons), max(lats)]
            ring = corners_wgs + [corners_wgs[0]]
            geometry = {
                "type": "Polygon",
                "coordinates": [[[lon, lat] for lon, lat in ring]],
            }
        else:
            # Embedding TIFF with no embedded CRS: derive from filename.
            log.debug(
                "No embedded CRS in '%s'; deriving spatial metadata from filename.",
                tiff_path,
            )
            bbox, geometry, epsg = _parse_spatial_from_filename(
                os.path.basename(tiff_path)
            )

        bands = ds.RasterCount
        dtype = gdal.GetDataTypeName(ds.GetRasterBand(1).DataType)
    finally:
        del ds

    return {
        "bbox":     bbox,
        "geometry": geometry,
        "epsg":     epsg,
        "bands":    bands,
        "dtype":    dtype,
    }


def _datetime_from_filename(filename: str) -> str:
    """
    Try to parse an ISO-8601 datetime string from a filename such as
    ``S2A_MSIL2A_20170102T175732_…``.

    Returns an ISO-8601 string (UTC) or the current UTC time if no date
    is found.
    """
    # Sentinel-2 style: YYYYMMDDTHHMMSS
    m = re.search(r'(\d{8})T(\d{6})', filename)
    if m:
        date_str, time_str = m.group(1), m.group(2)
        dt = datetime(
            int(date_str[0:4]), int(date_str[4:6]), int(date_str[6:8]),
            int(time_str[0:2]), int(time_str[2:4]), int(time_str[4:6]),
            tzinfo=timezone.utc,
        )
        return dt.isoformat()

    # Plain date YYYYMMDD
    m = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
    if m:
        dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                      tzinfo=timezone.utc)
        return dt.isoformat()

    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(tiff: str, stac_template: dict, cos_url: str) -> dict:
    """
    Build a STAC Item JSON for a GeoTIFF embedding from a template.

    Parameters
    ----------
    tiff : str
        Absolute or relative path to the GeoTIFF embedding file.
    stac_template : dict
        A STAC Item JSON template (loaded from e.g. ``stac_examples/``).
        The following fields are populated / overwritten from the tiff:
            - ``id``                     → filename stem (no extension)
            - ``bbox``                   → derived from geotransform (or filename)
            - ``geometry``               → bounding-box polygon in WGS 84
            - ``properties.datetime``    → parsed from the filename
            - ``properties.proj:epsg``   → EPSG code of the tile's native CRS
                                           (omitted when the code cannot be identified)
            - ``assets.embeddings.href`` → full COS/S3 URL (``cos_url/filename``)

        All other template fields are preserved unchanged.

    Returns
    -------
    dict
        The populated STAC Item as a plain Python dict (also written to disk).

    Side-effects
    ------------
    Writes ``<stem>.json`` next to the tiff file (same directory).
    """
    tiff = os.path.abspath(tiff)
    meta = _extract_tiff_metadata(tiff)

    stem      = os.path.splitext(os.path.basename(tiff))[0]
    tiff_dir  = os.path.dirname(tiff)
    json_path = os.path.join(tiff_dir, f"{stem}.json")

    item = copy.deepcopy(stac_template)

    # Core STAC fields
    item["id"]       = stem
    item["bbox"]     = meta["bbox"]
    item["geometry"] = meta["geometry"]

    # Properties – keep everything in the template, just update datetime + CRS
    item.setdefault("properties", {})
    item["properties"]["datetime"] = _datetime_from_filename(stem)
    if meta["epsg"] is not None:
        item["properties"]["proj:epsg"] = meta["epsg"]
    else:
        item["properties"].pop("proj:epsg", None)

    # Asset href – full COS/S3 URL pointing to the TIFF in the bucket
    item.setdefault("assets", {}).setdefault("embeddings", {})
    item["assets"]["embeddings"]["href"] = f"{cos_url}/{os.path.basename(tiff)}"

    # Persist
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(item, fh, indent=4)

    log.info("STAC Item written to %s", json_path)
    return item


# ---------------------------------------------------------------------------
# CLI entry-point (optional convenience)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create a STAC Item JSON for a TIFF embedding.")
    parser.add_argument("tiff",          help="Path to the GeoTIFF embedding.")
    parser.add_argument("stac_template", help="Path to the STAC Item JSON template.")
    parser.add_argument("cos_url", help="URL to the COS bucket where the STAC Item will be stored.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.stac_template, encoding="utf-8") as fh:
        template = json.load(fh)

    result = run(args.tiff, template, args.cos_url)
    print(json.dumps(result, indent=4))
