import os
from typing import Dict, Optional

import geopandas as gpd

from extractosm.utils import _parse_osm_tags


def get_osm_features(
    bounding_box: tuple[float, float, float, float],
    tags: Dict[str, bool],
    crs: str = "EPSG:3857",
    osm_pbf_path: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Get OSM features within a bounding box.

    This function retrieves OpenStreetMap (OSM) features based on the specified
    tags and bounding box. The retrieved features are returned as a GeoDataFrame
    in long format, with one row per feature.

    Args:
        bounding_box (tuple): A tuple of (west, south, east, north) coordinates.
        tags (Dict[str, bool] or Dict[str, list/str/None]): A dictionary of OSM tags to filter features.
            Values can be:
               - True: accept any value for that key
               - list/tuple/set: accept only listed values
               - str/int: accept that exact value
               - None: key existence is enough
        crs (str): The coordinate reference system for the output GeoDataFrame.
        osm_pbf_path (str, optional): Path to a local .osm.pbf file. If provided and exists,
            pyogrio will be used to read the file. Otherwise falls back to osmnx.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the OSM features within the
        bounding box.
    """

    # Validate bounding box length
    if not (isinstance(bounding_box, (tuple, list)) and len(bounding_box) == 4):
        raise ValueError(
            "bounding_box must be a tuple or list of exactly 4 elements: (west, south, east, north)"
        )

    if not osm_pbf_path or not os.path.exists(osm_pbf_path):
        raise ValueError(
            f"OSM PBF file not found: {osm_pbf_path}. Provide a valid path."
        )

    gdf = gpd.read_file(
        osm_pbf_path,
        layer="points",
        bbox=bounding_box,
    )

    if "other_tags" in gdf.columns:
        gdf["tags"] = gdf["other_tags"].apply(_parse_osm_tags)
    else:
        # If "other_tags" is missing, fill "tags" with empty dicts
        gdf["tags"] = [{} for _ in range(len(gdf))]
    for col in tags.keys():
        gdf[col] = gdf["tags"].apply(lambda d: d.get(col))

    # filter rows based on tags
    def row_matches_tags(row, tags):
        # Check if a row matches one of the specified tags
        for key, value in tags.items():
            tag_value = row.get(key)
            if value is True:
                if tag_value is not None:
                    return True
            elif isinstance(value, (list, tuple, set)):
                if tag_value in value:
                    return True
            else:
                if tag_value == value:
                    return True
        return False

    gdf = gdf[gdf.apply(lambda row: row_matches_tags(row, tags), axis=1)]

    # Normalize tag columns and reshape to long format (one row per tag key)
    id_names = ["osm_type", "osm_id", "geometry"]
    tag_names_requested = [k.lower() for k in tags.keys()]

    existing_tag_cols = [c for c in tag_names_requested if c in gdf.columns]
    # If none of the requested tag columns exist (handler may have stored them differently), try to pick any tag-like columns
    if not existing_tag_cols:
        # treat all non-id, non-geometry columns as tag columns
        existing_tag_cols = [c for c in gdf.columns if c not in id_names]

    # ensure columns for id_names exist
    for col in id_names:
        if col not in gdf.columns:
            gdf[col] = None

    # Convert to requested CRS and reduce geometry to centroid for non-point geometries
    # First reproject
    try:
        gdf = gdf.to_crs(crs)
    except Exception:
        # If gdf is already in the desired crs or reprojection fails, ignore
        pass

    # Convert polygons/lines to centroids
    gdf["geometry"] = gdf.geometry.representative_point()

    # Melt to long format for requested/existing tag columns
    gdf_long = gdf.melt(
        id_vars=id_names,
        value_vars=existing_tag_cols,
        var_name="variable",
        value_name="value",
    )

    gdf_long = gdf_long[gdf_long["value"].notna()].reset_index(drop=True)

    # Convert back to GeoDataFrame (melt converts to DataFrame)
    gdf_long = gpd.GeoDataFrame(gdf_long, geometry="geometry", crs=gdf.crs)

    return gdf_long
