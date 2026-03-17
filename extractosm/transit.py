import os
import warnings
from collections import defaultdict
from typing import Optional, cast

import geopandas as gpd
import osmium
import osmium.geom
import pandas as pd
import pyogrio
from shapely import wkb
import pyarrow  # noqa: F401 - for checking if pyarrow is installed when saving GeoParquet

from extractosm.utils import _parse_osm_tags


def _infer_transit_mode(tags: dict) -> Optional[str]:
    """
    Infer transit mode from OSM tags.

    Args:
        tags (dict): Dictionary of OSM tags from a node.

    Returns:
        str or None: Transit mode ("train", "bus", "tram", "metro", "mixed") or None if unknown.
    """
    # Direct indicators from railway tag
    if tags.get("railway") in ("station", "halt"):
        return "train"
    if tags.get("railway") == "tram_stop":
        return "tram"
    if tags.get("amenity") == "bus_station":
        return "bus"

    # Check boolean tags for multiple modes
    has_train = tags.get("train") == "yes"
    has_bus = tags.get("bus") == "yes"
    has_tram = tags.get("tram") == "yes"
    has_subway = tags.get("subway") == "yes"

    modes = []
    if has_train:
        modes.append("train")
    if has_bus:
        modes.append("bus")
    if has_tram:
        modes.append("tram")
    if has_subway:
        modes.append("metro")

    if len(modes) == 1:
        return modes[0]
    elif len(modes) > 1:
        return "mixed"

    # Check network tag for hints
    network = tags.get("network", "").lower()
    if any(keyword in network for keyword in ["train", "sncf", "ter", "rail"]):
        return "train"
    if "tram" in network:
        return "tram"
    if "bus" in network:
        return "bus"
    if any(keyword in network for keyword in ["metro", "subway", "underground"]):
        return "metro"

    return None  # Unknown


def extract_transit_stops(
    bounding_box: tuple[float, float, float, float],
    osm_pbf_path: str,
    crs: str = "EPSG:4326",
    include_route_ids: bool = False,
) -> gpd.GeoDataFrame:
    """
    Extract public transport stops and stations from OSM data.

    This function retrieves transit infrastructure nodes from OpenStreetMap data
    within the specified bounding box. It includes:
    - Stop positions (exact vehicle stopping points)
    - Stations (railway, metro, bus stations)
    - Platforms (passenger waiting areas)
    - Railway infrastructure (stations, halts, tram stops)

    The function checks for nodes with any of these OSM tag combinations:
    - public_transport=stop_position (exact vehicle stopping point)
    - public_transport=station (main station building/area)
    - public_transport=platform (passenger waiting platforms)
    - railway=station (railway stations)
    - railway=halt (small railway stops)
    - railway=tram_stop (tram stops)
    - amenity=bus_station (bus stations/terminals)

    Args:
        bounding_box (tuple): Tuple of (west, south, east, north) coordinates in EPSG:4326.
        osm_pbf_path (str): Path to local .osm.pbf file.
        crs (str): Coordinate reference system for output GeoDataFrame. Default "EPSG:4326".
        include_route_ids (bool): If True, add a route_ids column showing
            which routes serve each stop. Default False.


    Returns:
        gpd.GeoDataFrame: GeoDataFrame with columns:
            - osm_id (int): OSM node ID
            - osm_type (str): Always "node"
            - name (str): Stop/station name (may be None)
            - stop_type (str): Type of stop ("stop_position", "station", "railway_station",
                "railway_halt", "tram_stop", "platform", "bus_station")
            - transit_mode (str): Inferred mode ("train", "bus", "tram", "metro", "mixed", or None)
            - railway (str): Value of railway tag (may be None)
            - public_transport (str): Value of public_transport tag (may be None)
            - stop_area_id (int): OSM stop_area relation ID (None if not in any stop_area)
            - stop_area_name (str): Stop area name (None if not in any stop_area)
            - route_ids (list[int]): List of route OSM IDs serving this stop (only if include_route_ids=True)
            - route_count (int): Number of routes serving this stop (only if include_route_ids=True)
            - geometry (Point): Point location

    Raises:
        ValueError: If bounding_box is not a tuple/list of exactly 4 elements.

    Examples:
        >>> bbox = (6.13, 46.19, 6.15, 46.21)
        >>> stops = extract_transit_stops(bbox, osm_pbf_path="geneva-greater-area.osm.pbf")
        >>> print(f"Found {len(stops)} transit stops")
        >>>
        >>> # Filter for only railway stations
        >>> railway_stations = stops[stops["stop_type"].str.contains("railway")]
        >>> print(f"Found {len(railway_stations)} railway stations")
        >>>
        >>> # Get all train stops (includes stations and stop_positions)
        >>> train_stops = stops[stops["transit_mode"] == "train"]
        >>> print(f"Found {len(train_stops)} train stops")

        >>> # Check which stops belong to stop_areas
        >>> stops_with_areas = stops[stops['stop_area_id'].notna()]
        >>> print(f"{len(stops_with_areas)} of {len(stops)} stops belong to stop_areas")

        >>> # Analyze stops by stop_area
        >>> stops.groupby('stop_area_name').agg({'osm_id': 'count', 'name': list})

    Notes:
        - Most stops won't belong to a stop_area (stop_area_id will be None)
        - Stop areas are more common in major stations and interchanges
        - A stop_area groups multiple platforms, stop_positions, and related infrastructure
        - Performance: ~10-20 seconds for 68MB file (includes node extraction + filtering)
    """
    # Validate bounding box
    if not (isinstance(bounding_box, (tuple, list)) and len(bounding_box) == 4):
        raise ValueError(
            "bounding_box must be a tuple or list of exactly 4 elements: (west, south, east, north)"
        )

    if not osm_pbf_path or not os.path.exists(osm_pbf_path):
        raise ValueError(
            f"OSM PBF file not found: {osm_pbf_path}. Provide a valid path"
        )

    # Extract stops using pyosmium with spatial filtering
    class StopHandler(osmium.SimpleHandler):
        def __init__(self, bbox: Optional[tuple[float, float, float, float]] = None):
            super().__init__()
            self.stops: list[dict] = []
            self.wkb_factory = osmium.geom.WKBFactory()
            self.bbox = bbox  # (west, south, east, north)

        def node(self, n):
            if self.bbox is not None:
                west, south, east, north = self.bbox
                if not (
                    west <= n.location.lon <= east and south <= n.location.lat <= north
                ):
                    return  # Skip nodes outside bounding box immediately

            # Check tags in priority order (most common first for early exit)
            stop_type = None

            # Most common: public_transport=stop_position
            pt_value = n.tags.get("public_transport")
            if pt_value == "stop_position":
                stop_type = "stop_position"
            elif pt_value == "station":
                stop_type = "station"
            elif pt_value == "platform":
                stop_type = "platform"

            # If not found, check railway tags
            if stop_type is None:
                railway_value = n.tags.get("railway")
                if railway_value == "station":
                    stop_type = "railway_station"
                elif railway_value == "halt":
                    stop_type = "railway_halt"
                elif railway_value == "tram_stop":
                    stop_type = "tram_stop"

            # Finally check amenity
            if stop_type is None:
                if n.tags.get("amenity") == "bus_station":
                    stop_type = "bus_station"

            # Skip if no relevant tags found
            if stop_type is None:
                return

            try:
                # Create Point geometry from node
                geom_wkb = self.wkb_factory.create_point(n)
                geom = wkb.loads(geom_wkb, hex=True)

                # Now create tags dict for transit mode inference
                tags_dict = dict(n.tags)
                transit_mode = _infer_transit_mode(tags_dict)

                self.stops.append(
                    {
                        "osm_id": n.id,
                        "osm_type": "node",
                        "name": n.tags.get("name"),
                        "stop_type": stop_type,
                        "transit_mode": transit_mode,
                        "railway": n.tags.get("railway"),
                        "public_transport": n.tags.get("public_transport"),
                        "geometry": geom,
                    }
                )
            except Exception:
                pass  # Skip nodes with invalid geometries

    # Process the OSM file with spatial filtering
    handler = StopHandler(bbox=bounding_box)
    handler.apply_file(osm_pbf_path)

    # Create GeoDataFrame
    if not handler.stops:
        # Return empty GeoDataFrame with correct schema
        gdf = gpd.GeoDataFrame(
            columns=[
                "osm_id",
                "osm_type",
                "name",
                "stop_type",
                "transit_mode",
                "railway",
                "public_transport",
                "geometry",
            ],
            crs="EPSG:4326",
            geometry="geometry",
        )
    else:
        gdf = gpd.GeoDataFrame(handler.stops, crs="EPSG:4326", geometry="geometry")

    # Convert to requested CRS
    if crs != "EPSG:4326":
        try:
            gdf = gdf.to_crs(crs)
        except Exception:
            pass  # If conversion fails, keep in WGS84

    # Reset index
    gdf = gdf.reset_index(drop=True)

    # Add route IDs if requested
    if include_route_ids:
        # Get route-stop mapping
        route_stop_mapping = get_route_stop_mapping(osm_pbf_path)

        # Invert mapping: stop_id -> [route_id, route_id, ...]
        stop_route_mapping: dict[int, list[int]] = defaultdict(list)
        for route_id, stop_ids in route_stop_mapping.items():
            for stop_id in stop_ids:
                stop_route_mapping[stop_id].append(route_id)

        # Add route_ids and route_count columns
        gdf["route_ids"] = gdf["osm_id"].apply(
            lambda stop_id: stop_route_mapping.get(stop_id, [])
        )

    # Add stop_area information (always included)
    if not gdf.empty:
        # Get stop→stop_area mapping
        stop_to_sa = get_stop_to_stop_area_mapping(osm_pbf_path=osm_pbf_path)

        # Get full stop_area details
        stop_areas = get_stop_areas(osm_pbf_path=osm_pbf_path)

        # Add stop_area columns
        gdf["stop_area_id"] = gdf["osm_id"].apply(lambda sid: stop_to_sa.get(sid))
        gdf["stop_area_name"] = gdf["stop_area_id"].apply(
            lambda said: (
                stop_areas[said]["name"] if said and said in stop_areas else None
            )
        )

    return gdf


def extract_all_transit_stops(
    osm_pbf_path: str,
    crs: str = "EPSG:4326",
    output_path: Optional[str] = None,
    include_route_ids: bool = False,
) -> gpd.GeoDataFrame:
    """
    Extract ALL transit stops from an OSM PBF file (no bounding box filter).

    This function is designed for offline/cron job extraction of transit stops
    from large OSM PBF files. It processes the entire file and optionally saves
    to GeoParquet format for fast reloading in production.

    Performance: Extracts all stops from a 68MB file in approximately 30-60 seconds.
    The resulting GeoParquet file can be loaded in <100ms and clipped in <1ms.

    Args:
        osm_pbf_path (str): Path to local .osm.pbf file or filename in package data.
        crs (str): Coordinate reference system for output GeoDataFrame. Default "EPSG:4326".
        output_path (str, optional): If provided, saves result to GeoParquet file at this path.
            Recommended for production use. Example: "data/all_stops.geoparquet"
        include_route_ids (bool): If True, add route_ids and route_count columns showing
            which routes serve each stop. Default False. Note: This adds processing time.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with all transit stops, same schema as extract_transit_stops():
            - osm_id (int): OSM node ID
            - osm_type (str): Always "node"
            - name (str): Stop/station name (may be None)
            - stop_type (str): Type of stop
            - transit_mode (str): Inferred mode (train, bus, tram, metro, mixed, or None)
            - railway (str): Value of railway tag (may be None)
            - public_transport (str): Value of public_transport tag (may be None)
            - geometry (Point): Point location

    Raises:
        ValueError: If OSM PBF file not found.
        ImportError: If output_path is specified but pyarrow is not installed.

    Examples:
        >>> # Extract all stops and save to GeoParquet (offline/cron job)
        >>> all_stops = extract_all_transit_stops(
        ...     osm_pbf_path="geneva-greater-area.osm.pbf",
        ...     output_path="data/all_stops.geoparquet"
        ... )
        >>> print(f"Extracted {len(all_stops)} stops, saved to GeoParquet")
        >>>
        >>> # In production, load pre-extracted data (fast: <100ms)
        >>> import geopandas as gpd
        >>> all_stops = gpd.read_parquet("data/all_stops.geoparquet")
    """
    if not osm_pbf_path or not os.path.exists(osm_pbf_path):
        raise ValueError(
            f"OSM PBF file not found: {osm_pbf_path}. Provide a valid path."
        )

    # Check if pyarrow is available if output_path is specified
    # Extract ALL stops using extract_transit_stops with a global bbox
    # Use a bbox that covers the entire world to get all stops
    global_bbox = (-180.0, -90.0, 180.0, 90.0)
    gdf = extract_transit_stops(
        global_bbox,
        crs=crs,
        osm_pbf_path=osm_pbf_path,
        include_route_ids=include_route_ids,
    )

    # Save to GeoParquet if output path specified
    if output_path is not None:
        try:
            gdf.to_parquet(output_path)
            warnings.warn(
                f"Saved {len(gdf)} transit stops to {output_path}",
                UserWarning,
                stacklevel=2,
            )
        except Exception as e:
            warnings.warn(
                f"Failed to save GeoParquet: {e}",
                UserWarning,
                stacklevel=2,
            )

    return gdf


def clip_transit_stops(
    stops: gpd.GeoDataFrame,
    bounding_box: tuple[float, float, float, float],
    crs: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Clip transit stops GeoDataFrame to a bounding box (fast spatial filtering).

    This function uses GeoPandas' built-in spatial indexing (R-tree) for extremely
    fast clipping (<1ms for typical queries). It's designed to work with pre-extracted
    stops from extract_all_transit_stops() in production environments.

    Performance: <1ms to clip 10,000+ stops to a small bounding box using spatial index.

    Args:
        stops (gpd.GeoDataFrame): GeoDataFrame of transit stops (from extract_all_transit_stops
            or extract_transit_stops). Must have a 'geometry' column with Point geometries.
        bounding_box (tuple): Tuple of (west, south, east, north) coordinates.
            Coordinates should be in the same CRS as the stops GeoDataFrame.
        crs (str, optional): If provided, converts the result to this CRS after clipping.
            If None, keeps the same CRS as input stops.

    Returns:
        gpd.GeoDataFrame: Clipped GeoDataFrame containing only stops within the bounding box.
            Same schema as input stops GeoDataFrame.

    Raises:
        ValueError: If bounding_box is not a tuple/list of exactly 4 elements.

    Examples:
        >>> # Production workflow: Load pre-extracted stops once at startup
        >>> import geopandas as gpd
        >>> all_stops = gpd.read_parquet("data/all_stops.geoparquet")  # <100ms
        >>>
        >>> # Clip to different bounding boxes as needed (instant)
        >>> bbox1 = (6.13, 46.19, 6.15, 46.21)  # Geneva
        >>> stops1 = clip_transit_stops(all_stops, bbox1)  # <1ms
        >>>
        >>> bbox2 = (5.20, 46.19, 5.25, 46.21)  # Bourg-en-Bresse
        >>> stops2 = clip_transit_stops(all_stops, bbox2)  # <1ms
        >>>
        >>> # Convert CRS during clipping
        >>> stops_web_mercator = clip_transit_stops(
        ...     all_stops, bbox1, crs="EPSG:3857"
        ... )
    """
    # Validate bounding box
    if not (isinstance(bounding_box, (tuple, list)) and len(bounding_box) == 4):
        raise ValueError(
            "bounding_box must be a tuple or list of exactly 4 elements: (west, south, east, north)"
        )

    # Extract bbox coordinates
    west, south, east, north = bounding_box

    # Use GeoPandas spatial indexing for fast clipping (R-tree based)
    # The .cx indexer uses the spatial index automatically
    clipped = stops.cx[west:east, south:north].copy()  # type: ignore[misc]

    # Convert to requested CRS if specified
    if crs is not None and crs != stops.crs:
        try:
            clipped = clipped.to_crs(crs)
        except Exception:
            pass  # If conversion fails, keep original CRS

    # Reset index
    clipped = clipped.reset_index(drop=True)

    return clipped


def extract_transit_routes(
    bounding_box: tuple[float, float, float, float],
    osm_pbf_path: str,
    route_types: list[str] | None = None,
    crs: str = "EPSG:4326",
    include_stop_ids: bool = False,
    group_by: str = "route",
) -> gpd.GeoDataFrame:
    """
    Get fixed-route transit systems within a bounding box.

    This function retrieves OpenStreetMap route relations for public transport
    services (train, bus, tram, etc.) based on the specified route types and
    bounding box. The retrieved routes are returned as a GeoDataFrame with
    route metadata and geometries. Routes can be grouped by route_master
    (service level) or kept as individual variants (directional).

    Args:
        bounding_box (tuple): A tuple of (west, south, east, north) coordinates.
        route_types (list[str], optional): List of route types to extract.
            Default is ["train", "bus", "tram", "subway", "trolleybus", "light_rail"].
            Supported values include: "train", "bus", "tram", "subway", "light_rail",
            "trolleybus", "ferry", "monorail", etc. See OSM route types for full list.
        crs (str): The coordinate reference system for the output GeoDataFrame.
            Default is "EPSG:4326".
        osm_pbf_path (str): Path to a local .osm.pbf file. If provided
            and exists, pyogrio will be used to read the file. Otherwise falls
            back to osmnx.
        include_stop_ids (bool): If True, adds stop_ids (list of stop OSM IDs)
            and stop_count (number of stops) columns to the output GeoDataFrame.
            Default is False.
        group_by (str): How to group routes in the output. Default is "route".
            - "route": One row per route variant (e.g., "Bus 7 A→B" and "Bus 7 B→A" are separate)
            - "route_master": One row per service (e.g., single "Bus 7" row combining all variants)
            - None: No grouping (may include duplicate routes from overlapping relations)

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing transit routes.

        For group_by='route' (default):
            - osm_id: OSM relation ID
            - osm_type: OSM element type (typically "relation")
            - route: Route type (bus, train, tram, etc.)
            - name: Route name
            - ref: Route reference/number
            - from: Starting station/stop name
            - to: Ending station/stop name
            - network: Transit network name
            - operator: Operating company
            - website: Route website URL
            - route_master_id: OSM route_master relation ID (None if not in route_master)
            - route_master_ref: Route master reference number (e.g., "7")
            - route_master_name: Route master name (e.g., "Bus 7")
            - geometry: Route path (LineString) or centroid (Point)
            - stop_ids: (if include_stop_ids=True) List of stop OSM IDs on this route
            - stop_count: (if include_stop_ids=True) Number of stops on this route

        For group_by='route_master':
            - osm_id: OSM route_master relation ID
            - route_master_ref: Route reference number (e.g., "7")
            - route_master_name: Route name (e.g., "Bus 7")
            - route: Route type (bus, tram, trolleybus, etc.)
            - network: Transit network name
            - operator: Operating company
            - website: Website URL
            - variant_count: Number of route variants (directions)
            - variant_route_ids: list[int] of member route relation IDs
            - geometry: MultiLineString (union of all variant geometries)
            - stop_ids: (if include_stop_ids=True) Union of stops across all variants
            - stop_count: (if include_stop_ids=True) Total unique stops

    Raises:
        ValueError: If bounding_box is not a tuple/list of exactly 4 elements,
            or if route_types is an empty list.

    Examples:
        >>> # Get all train, bus, and tram routes in Geneva
        >>> bbox = (6.13, 46.19, 6.15, 46.21)
        >>> routes = extract_transit_routes(bbox, osm_pbf_path="geneva.osm.pbf")
        >>> print(routes[['route', 'name', 'ref']].head())

        >>> # Get only bus routes with simplified geometry
        >>> routes = extract_transit_routes(
        ...     bbox,
        ...     route_types=["bus"],
        ... )

        >>> # Get routes with stop relationships
        >>> routes = extract_transit_routes(
        ...     bbox,
        ...     osm_pbf_path="geneva.osm.pbf",
        ...     include_stop_ids=True,
        ... )
        >>> print(routes[['route', 'name', 'stop_count']].head())

        >>> # Group by route_master for service-level view
        >>> routes = extract_transit_routes(
        ...     bbox,
        ...     osm_pbf_path="lausanne.osm.pbf",
        ...     group_by="route_master",
        ... )
        >>> print(f"Found {len(routes)} transit services")

        >>> # Keep route variants but analyze by route_master
        >>> routes = extract_transit_routes(bbox, "lausanne.osm.pbf")
        >>> routes.groupby('route_master_ref').agg({'osm_id': 'count', 'name': 'first'})
    """
    # Set default route types
    if route_types is None:
        route_types = ["train", "bus", "tram", "subway", "trolleybus", "light_rail"]

    # Validate bounding box length
    if not (isinstance(bounding_box, (tuple, list)) and len(bounding_box) == 4):
        raise ValueError(
            "bounding_box must be a tuple or list of exactly 4 elements: (west, south, east, north)"
        )

    # Validate route_types
    if not route_types or not isinstance(route_types, list):
        raise ValueError("route_types must be a non-empty list of route type strings")

    if osm_pbf_path and not os.path.exists(osm_pbf_path):
        raise ValueError(
            f"OSM PBF file not found: {osm_pbf_path}. Provide a valid path"
        )

    # Try reading from multilinestrings layer first (where route relations are stored)
    gdf_raw = pyogrio.read_dataframe(
        osm_pbf_path,
        layer="multilinestrings",
        bbox=bounding_box,
    )
    gdf = cast(gpd.GeoDataFrame, gdf_raw)

    if not gdf.empty:
        # Convert osm_id from string to int64 for consistency with route_stop_mapping
        if "osm_id" in gdf.columns:
            gdf["osm_id"] = gdf["osm_id"].astype("int64")
        # Parse other_tags to extract route metadata
        if "other_tags" in gdf.columns:
            gdf["tags"] = gdf["other_tags"].apply(_parse_osm_tags)
        else:
            gdf["tags"] = [{} for _ in range(len(gdf))]

        # Extract route type from tags
        gdf["route"] = gdf["tags"].apply(lambda d: d.get("route"))

        # Filter for route relations with matching route types
        # Note: PBF files may not have type="route" as a direct tag
        # We filter by route tag directly
        gdf = gdf[gdf["route"].isin(route_types)].copy()

        # Extract metadata fields from tags
        metadata_fields = [
            "ref",
            "from",
            "to",
            "network",
            "operator",
            "website",
        ]
        for field in metadata_fields:
            gdf[field] = gdf["tags"].apply(lambda d: d.get(field))

        # Keep only relevant columns
        expected_cols = (
            ["osm_type", "osm_id", "route", "name"] + metadata_fields + ["geometry"]
        )
        for col in expected_cols:
            if col not in gdf.columns:
                gdf[col] = None
        gdf = gdf[expected_cols]

    # If no routes found, return empty GeoDataFrame with expected schema
    if gdf.empty:
        return gpd.GeoDataFrame(
            columns=[
                "osm_type",
                "osm_id",
                "route",
                "name",
                "ref",
                "from",
                "to",
                "network",
                "operator",
                "website",
                "geometry",
            ],
            crs=crs,
            geometry="geometry",
        )

    # Reproject to target CRS
    try:
        gdf = gdf.to_crs(crs)
    except Exception:
        # If reprojection fails or GeoDataFrame is already in target CRS, continue
        pass

    # Reset index
    gdf = gdf.reset_index(drop=True)

    # Add route_master information (always included)
    if not gdf.empty:
        # Get route→route_master mapping
        route_to_rm = get_route_to_route_master_mapping(
            osm_pbf_path=osm_pbf_path,
            route_types=route_types,
        )

        # Get full route_master details
        route_masters = get_route_masters(
            osm_pbf_path=osm_pbf_path,
            route_types=route_types,
        )

        # Add route_master columns
        gdf["route_master_id"] = gdf["osm_id"].apply(lambda rid: route_to_rm.get(rid))
        gdf["route_master_ref"] = gdf["route_master_id"].apply(
            lambda rmid: (
                route_masters[rmid]["ref"] if rmid and rmid in route_masters else None
            )
        )
        gdf["route_master_name"] = gdf["route_master_id"].apply(
            lambda rmid: (
                route_masters[rmid]["name"] if rmid and rmid in route_masters else None
            )
        )

    # Add stop IDs if requested (before grouping, so we can aggregate stops)
    if include_stop_ids and not gdf.empty:
        # Get route-stop mapping (extracts from entire PBF file)
        route_stop_mapping = get_route_stop_mapping(
            osm_pbf_path=osm_pbf_path,
            route_types=route_types,
        )

        # Create a mapping from route_id to list of stop_ids
        route_to_stops: dict[int, list[int]] = {}
        for route_id, stop_ids in route_stop_mapping.items():
            route_to_stops[route_id] = stop_ids

        # Add stop_ids column
        gdf["stop_ids"] = gdf["osm_id"].apply(lambda rid: route_to_stops.get(rid, []))
        gdf["stop_count"] = gdf["stop_ids"].apply(len)

    # Apply grouping if requested
    if group_by == "route_master" and not gdf.empty:
        # Filter to only routes that have a route_master
        grouped_routes = gdf[gdf["route_master_id"].notna()].copy()
        ungrouped_routes = gdf[gdf["route_master_id"].isna()].copy()

        if not grouped_routes.empty:
            # Group by route_master_id
            agg_dict = {
                "route_master_ref": "first",
                "route_master_name": "first",
                "route": "first",  # route type (bus, tram, etc.)
                "network": "first",
                "operator": "first",
                "website": "first",
                "osm_id": list,  # collect all route IDs → rename to variant_route_ids
                "geometry": lambda geoms: geoms.union_all(),  # union all geometries
            }

            # If stop_ids exist, aggregate them too
            if "stop_ids" in grouped_routes.columns:
                agg_dict["stop_ids"] = lambda stops: list(
                    set(sum(stops.tolist(), []))
                )  # union of all stop lists

            grouped_gdf = grouped_routes.groupby("route_master_id", as_index=False).agg(
                agg_dict
            )

            # Rename columns
            grouped_gdf = grouped_gdf.rename(
                columns={
                    "osm_id": "variant_route_ids",
                    "route_master_id": "osm_id",  # route_master_id becomes the primary ID
                }
            )

            # Add variant_count
            grouped_gdf["variant_count"] = grouped_gdf["variant_route_ids"].apply(len)

            # Recalculate stop_count if stop_ids exist
            if "stop_ids" in grouped_gdf.columns:
                grouped_gdf["stop_count"] = grouped_gdf["stop_ids"].apply(len)

            # Reorder columns
            base_cols = [
                "osm_id",
                "route_master_ref",
                "route_master_name",
                "route",
                "network",
                "operator",
                "website",
                "variant_count",
                "variant_route_ids",
            ]
            if "stop_ids" in grouped_gdf.columns:
                base_cols.extend(["stop_ids", "stop_count"])
            base_cols.append("geometry")

            # Keep only columns that exist
            base_cols = [col for col in base_cols if col in grouped_gdf.columns]
            grouped_gdf = grouped_gdf[base_cols]

            # Combine with ungrouped routes (if any)
            if not ungrouped_routes.empty:
                # Add placeholder columns to ungrouped routes to match schema
                ungrouped_routes["variant_count"] = 1
                ungrouped_routes["variant_route_ids"] = ungrouped_routes[
                    "osm_id"
                ].apply(lambda x: [x])
                # Reorder ungrouped routes columns to match grouped
                ungrouped_routes = ungrouped_routes[base_cols]
                gdf = pd.concat([grouped_gdf, ungrouped_routes], ignore_index=True)
            else:
                gdf = grouped_gdf

            # Convert back to GeoDataFrame
            gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=crs)
            gdf = gdf.reset_index(drop=True)

    elif group_by is None:
        # No grouping - return as-is (may have duplicates)
        pass
    # else: group_by == "route" (default) - already in this format

    return gdf


def extract_all_transit_routes(
    osm_pbf_path: str,
    route_types: list[str] | None = None,
    crs: str = "EPSG:4326",
    output_path: Optional[str] = None,
    include_stop_ids: bool = False,
    group_by: str = "route",
) -> gpd.GeoDataFrame:
    """
    Extract ALL transit routes from OSM PBF file and save to GeoParquet.

    This function extracts all transit route relations from an OSM PBF file
    without spatial filtering (uses global bounding box). Designed for offline
    extraction in production workflows where routes are pre-extracted once and
    then clipped on-demand for fast queries.

    Args:
        osm_pbf_path (str): Path to .osm.pbf file (local path or package data file name).
        route_types (list[str], optional): List of route types to extract.
            Default is ["train", "bus", "tram", "subway", "trolleybus", "light_rail"].
            Supported values: "train", "bus", "tram", "subway", "light_rail",
            "trolleybus", "ferry", "monorail", etc.
        crs (str): Output coordinate reference system. Default is "EPSG:4326".
        output_path (str, optional): Path to save GeoParquet file. If provided,
            the extracted routes will be saved to this path with a warning message.
        include_stop_ids (bool): If True, adds stop_ids (list of stop OSM IDs)
            columns to the output GeoDataFrame.
            Default is False.
        group_by (str): How to group route variants. Options:
            - "route" (default): One row per route relation (individual directional variants)
            - "route_master": One row per route_master (service-level grouping, e.g., "Bus 7")
            - None: No grouping (may contain duplicate geometries)
            Default is "route".

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing all transit routes with columns:
            - osm_id: OSM relation ID
            - osm_type: OSM element type
            - route: Route type (bus, train, tram, etc.)
            - name: Route name
            - ref: Route reference/number
            - from: Starting station/stop name
            - to: Ending station/stop name
            - network: Transit network name
            - operator: Operating company
            - website: Route website URL
            - geometry: Route path (LineString/MultiLineString)
            - stop_ids: (if include_stop_ids=True) List of stop OSM IDs on this route

    Examples:
        >>> # Extract all routes from PBF file
        >>> routes = extract_all_transit_routes(
        ...     osm_pbf_path="geneva-greater-area.osm.pbf",
        ...     output_path="transit_routes.geoparquet"
        ... )
        >>> print(f"Extracted {len(routes)} routes")
        Extracted 250 routes

        >>> # Extract only buses and trams
        >>> routes = extract_all_transit_routes(
        ...     osm_pbf_path="geneva-greater-area.osm.pbf",
        ...     route_types=["bus", "tram"]
        ... )

        >>> # Extract routes with stop relationships
        >>> routes = extract_all_transit_routes(
        ...     osm_pbf_path="geneva-greater-area.osm.pbf",
        ...     include_stop_ids=True,
        ... )

    See Also:
        - clip_transit_routes(): Fast clipping of pre-extracted routes
    """
    # Set default route types
    if route_types is None:
        route_types = ["train", "bus", "tram", "subway", "trolleybus", "light_rail"]

    # Use global bounding box to extract ALL routes
    global_bbox = (-180.0, -90.0, 180.0, 90.0)

    # Extract routes using existing extract_transit_routes function
    gdf = extract_transit_routes(
        bounding_box=global_bbox,
        route_types=route_types,
        crs=crs,
        osm_pbf_path=osm_pbf_path,
        include_stop_ids=include_stop_ids,
        group_by=group_by,
    )

    # Save to GeoParquet if output path specified
    if output_path is not None:
        gdf.to_parquet(output_path)
        import warnings

        warnings.warn(
            f"Saved {len(gdf)} transit routes to {output_path}",
            UserWarning,
            stacklevel=2,
        )

    return gdf


def clip_transit_routes(
    routes: gpd.GeoDataFrame,
    bounding_box: tuple[float, float, float, float],
    crs: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Clip transit routes to bounding box using spatial index (instant).

    This function efficiently clips a pre-loaded routes GeoDataFrame to a
    bounding box using GeoPandas' built-in spatial indexing (R-tree). Designed
    for production use where routes are extracted once offline and clipped
    on-demand for each request.

    Args:
        routes (gpd.GeoDataFrame): Pre-loaded GeoDataFrame of all transit routes
            (typically from extract_all_transit_routes() or read_parquet()).
        bounding_box (tuple): Bounding box as (west, south, east, north) in
            same CRS as the routes GeoDataFrame.
        crs (str, optional): Target CRS to convert result to. If None, uses
            the CRS of the input routes GeoDataFrame.

    Returns:
        gpd.GeoDataFrame: Clipped routes GeoDataFrame with same schema as input.

    Raises:
        ValueError: If bounding_box is not a tuple/list of exactly 4 elements.

    Examples:
        >>> # Load pre-extracted routes
        >>> all_routes = gpd.read_parquet("transit_routes.geoparquet")
        >>> print(f"Loaded {len(all_routes)} routes")
        Loaded 250 routes

        >>> # Clip to Geneva bbox (instant)
        >>> geneva_bbox = (6.13, 46.19, 6.15, 46.21)
        >>> geneva_routes = clip_transit_routes(all_routes, geneva_bbox)
        >>> print(f"Clipped to {len(geneva_routes)} routes")
        Clipped to 45 routes

        >>> # Clip and convert CRS
        >>> routes_3857 = clip_transit_routes(
        ...     all_routes,
        ...     geneva_bbox,
        ...     crs="EPSG:3857"
        ... )

    """
    # Validate bounding box
    if not (isinstance(bounding_box, (tuple, list)) and len(bounding_box) == 4):
        raise ValueError(
            "bounding_box must be a tuple or list of exactly 4 elements: "
            "(west, south, east, north)"
        )

    # Clip routes to bounding box (uses spatial index automatically)
    west, south, east, north = bounding_box
    clipped = routes.clip((west, south, east, north))

    # Convert to target CRS if specified
    if crs is not None and str(clipped.crs) != crs:
        clipped = clipped.to_crs(crs)

    return clipped


def get_route_stop_mapping(
    osm_pbf_path: str,
    route_types: list[str] | None = None,
) -> dict[int, list[int]]:
    """
    Extract mapping of transit routes to their member stops from OSM relations.

    This function parses route relations from OSM data and builds a mapping
    of which stops (nodes) belong to which routes. Only nodes that are members
    of the route relations are included.

    Args:
        route_types (list[str], optional): List of route types to extract (e.g., ["train", "bus", "tram"]).
            Default is ["train", "bus", "tram", "subway", "trolleybus", "light_rail"].
        osm_pbf_path (str): Path to local .osm.pbf file.

    Returns:
        dict[int, list[int]]: Dictionary mapping route OSM relation ID to list of stop node OSM IDs.
            Example: {123456: [789, 790, 791], 123457: [792, 793]}

    Examples:
        >>> mapping = get_route_stop_mapping(
        ...     route_types=["bus", "tram"],
        ...     osm_pbf_path="geneva-greater-area.osm.pbf"
        ... )
        >>> route_id = list(mapping.keys())[0]
        >>> print(f"Route {route_id} has {len(mapping[route_id])} stops")
    """
    # Set defaults
    if route_types is None:
        route_types = ["train", "bus", "tram", "subway", "trolleybus", "light_rail"]

    if not osm_pbf_path or not os.path.exists(osm_pbf_path):
        raise ValueError(
            f"OSM PBF file not found: {osm_pbf_path}. Provide a valid path."
        )

    # Extract route-stop mapping using pyosmium
    class RouteHandler(osmium.SimpleHandler):
        def __init__(self, route_types):
            super().__init__()
            self.route_types = set(route_types)
            self.route_stops = defaultdict(list)

        def relation(self, r):
            # Check if it's a route relation of the right type
            if r.tags.get("type") != "route":
                return

            route_type = r.tags.get("route")
            if route_type not in self.route_types:
                return

            # Extract member nodes (stops)
            for member in r.members:
                if member.type == "n":  # node
                    self.route_stops[r.id].append(member.ref)

    # Process the OSM file
    handler = RouteHandler(route_types)
    handler.apply_file(osm_pbf_path, locations=False, idx="sparse_file_array")

    # Convert to regular dict
    result = dict(handler.route_stops)

    return result


def save_route_stop_mapping(mapping: dict[int, list[int]], output_path: str) -> None:
    """
    Save route→stop mapping to Parquet file for fast loading.

    This function converts a route-to-stops mapping dictionary into a DataFrame
    and saves it in Parquet format using PyArrow for efficient storage and fast
    loading. The Parquet format provides better compression and faster I/O than
    JSON while preserving the list structure.

    Args:
        mapping (dict[int, list[int]]): Dictionary mapping route OSM relation IDs
            to lists of stop node OSM IDs. Typically from get_route_stop_mapping().
        output_path (str): Path to save the Parquet file (e.g., "route_stops.parquet").

    Returns:
        None

    Examples:
        >>> # Get mapping from OSM data
        >>> mapping = get_route_stop_mapping(
        ...     osm_pbf_path="geneva-greater-area.osm.pbf"
        ... )
        >>> # Save to Parquet
        >>> save_route_stop_mapping(mapping, "route_stop_mapping.parquet")
        >>> print(f"Saved mapping for {len(mapping)} routes")
        Saved mapping for 250 routes

    Notes:
        - Parquet format: ~2-3MB for 250 routes (compressed)
        - Load time: ~100-500ms with load_route_stop_mapping()
        - Schema: {route_id: int64, stop_ids: list<int64>}
        - Uses PyArrow engine for maximum compatibility

    See Also:
        - load_route_stop_mapping(): Load mapping from Parquet file
        - get_route_stop_mapping(): Extract mapping from OSM PBF file
        - extract_and_save_all_transit_data(): Convenience function for full workflow
    """

    # Convert dict to DataFrame
    # Format: [{route_id: int, stop_ids: [int, int, ...]}, ...]
    records = [
        {"route_id": route_id, "stop_ids": stop_ids}
        for route_id, stop_ids in mapping.items()
    ]
    df = pd.DataFrame(records)

    # Save to Parquet with PyArrow engine
    # PyArrow handles list<int64> type automatically
    df.to_parquet(output_path, engine="pyarrow", index=False)


def load_route_stop_mapping(input_path: str) -> dict[int, list[int]]:
    """
    Load route→stop mapping from Parquet file.

    This function loads a previously saved route-to-stops mapping from Parquet
    format and converts it back to a dictionary for efficient lookup.

    Args:
        input_path (str): Path to the Parquet file (e.g., "route_stop_mapping.parquet").

    Returns:
        dict[int, list[int]]: Dictionary mapping route OSM relation IDs to
            lists of stop node OSM IDs.

    Raises:
        FileNotFoundError: If the input file does not exist.

    Examples:
        >>> # Load pre-saved mapping
        >>> mapping = load_route_stop_mapping("route_stop_mapping.parquet")
        >>> print(f"Loaded mapping for {len(mapping)} routes")
        Loaded mapping for 250 routes

        >>> # Use in production workflow
        >>> route_id = 123456
        >>> stop_ids = mapping.get(route_id, [])
        >>> print(f"Route {route_id} has {len(stop_ids)} stops")
        Route 123456 has 15 stops

    Notes:
        - Load time: ~100-500ms (much faster than JSON)
        - File size: ~2-3MB for 250 routes (compressed)
        - Recommended: Load once at server startup, reuse in memory

    See Also:
        - save_route_stop_mapping(): Save mapping to Parquet file
    """

    # Load from Parquet
    df = pd.read_parquet(input_path, engine="pyarrow")

    # Convert back to dict
    # DataFrame has columns: route_id (int64), stop_ids (list<int64>)
    mapping = dict(zip(df["route_id"], df["stop_ids"]))

    return mapping


def get_route_masters(
    osm_pbf_path: str,
    route_types: list[str] | None = None,
) -> dict[int, dict]:
    """
    Extract route_master relations from OSM data.

    A route_master relation groups all directional variants of a transit service
    (e.g., "Bus 7" contains both "Bus 7: A→B" and "Bus 7: B→A" routes).

    Args:
        osm_pbf_path (str): Path to local .osm.pbf file.
        route_types (list[str], optional): List of route types to extract.
            Default is ["train", "bus", "tram", "subway", "trolleybus", "light_rail"].

    Returns:
        dict[int, dict]: Dictionary mapping route_master_id (int) to dict with:
            - 'name': str - Route master name (e.g., "Bus 7")
            - 'ref': str - Route reference number (e.g., "7")
            - 'route_master': str - Type (bus, tram, trolleybus, etc.)
            - 'network': str - Transit network name
            - 'operator': str - Operating company
            - 'member_route_ids': list[int] - Member route relation IDs

    Examples:
        >>> route_masters = get_route_masters("lausanne.osm.pbf")
        >>> rm = route_masters[3210864]  # Bus 7 route_master
        >>> print(f"{rm['name']}: {len(rm['member_route_ids'])} variants")
        Bus 7: 2 variants

        >>> # Count route_masters by type
        >>> from collections import Counter
        >>> types = [rm['route_master'] for rm in route_masters.values()]
        >>> print(Counter(types))
        Counter({'bus': 400, 'tram': 50, 'train': 30})
    """
    # Set defaults
    if route_types is None:
        route_types = ["train", "bus", "tram", "subway", "trolleybus", "light_rail"]

    if not osm_pbf_path or not os.path.exists(osm_pbf_path):
        raise ValueError(
            f"OSM PBF file not found: {osm_pbf_path}. Provide a valid path."
        )

    # Extract route_masters using pyosmium
    class RouteMasterHandler(osmium.SimpleHandler):
        def __init__(self, route_types):
            super().__init__()
            self.route_types = set(route_types)
            self.route_masters = {}

        def relation(self, r):
            # Check if it's a route_master relation
            if r.tags.get("type") != "route_master":
                return

            route_master_type = r.tags.get("route_master")
            if route_master_type not in self.route_types:
                return

            # Extract member route relation IDs
            member_route_ids = []
            for member in r.members:
                if member.type == "r":  # relation
                    member_route_ids.append(member.ref)

            # Store route_master info
            self.route_masters[r.id] = {
                "name": r.tags.get("name"),
                "ref": r.tags.get("ref"),
                "route_master": route_master_type,
                "network": r.tags.get("network"),
                "operator": r.tags.get("operator"),
                "member_route_ids": member_route_ids,
            }

    # Process the OSM file
    handler = RouteMasterHandler(route_types)
    handler.apply_file(osm_pbf_path, locations=False, idx="sparse_file_array")

    return handler.route_masters


def get_stop_areas(
    osm_pbf_path: str,
) -> dict[int, dict]:
    """
    Extract stop_area relations from OSM data.

    A stop_area relation groups all elements of a transit stop (platforms,
    stop_positions, shelters, etc.) into a single logical stop.

    Args:
        osm_pbf_path (str): Path to local .osm.pbf file.

    Returns:
        dict[int, dict]: Dictionary mapping stop_area_id (int) to dict with:
            - 'name': str - Stop area name
            - 'network': str - Transit network name
            - 'operator': str - Operating company
            - 'member_stop_ids': list[int] - Node IDs with role=stop
            - 'member_platform_ids': list[int] - Node/way IDs with role=platform

    Examples:
        >>> stop_areas = get_stop_areas("lausanne.osm.pbf")
        >>> sa = stop_areas[4830941]  # Béthusy stop area
        >>> print(f"{sa['name']}: {len(sa['member_stop_ids'])} stops")
        Béthusy: 3 stops

        >>> # Find stop_areas with most stops
        >>> sorted_sa = sorted(
        ...     stop_areas.items(),
        ...     key=lambda x: len(x[1]['member_stop_ids']),
        ...     reverse=True
        ... )
        >>> print(f"Largest stop area: {sorted_sa[0][1]['name']}")
    """
    if not osm_pbf_path or not os.path.exists(osm_pbf_path):
        raise ValueError(
            f"OSM PBF file not found: {osm_pbf_path}. Provide a valid path."
        )

    # Extract stop_areas using pyosmium
    class StopAreaHandler(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.stop_areas = {}

        def relation(self, r):
            # Check if it's a stop_area relation
            if r.tags.get("public_transport") != "stop_area":
                return

            # Extract member stops and platforms
            member_stop_ids = []
            member_platform_ids = []

            for member in r.members:
                role = member.role
                # Collect stops (node members with stop role)
                if "stop" in role and member.type == "n":
                    member_stop_ids.append(member.ref)
                # Collect platforms (node/way members with platform role)
                elif "platform" in role:
                    member_platform_ids.append(member.ref)

            # Store stop_area info
            self.stop_areas[r.id] = {
                "name": r.tags.get("name"),
                "network": r.tags.get("network"),
                "operator": r.tags.get("operator"),
                "member_stop_ids": member_stop_ids,
                "member_platform_ids": member_platform_ids,
            }

    # Process the OSM file
    handler = StopAreaHandler()
    handler.apply_file(osm_pbf_path, locations=False, idx="sparse_file_array")

    return handler.stop_areas


def get_route_to_route_master_mapping(
    osm_pbf_path: str,
    route_types: list[str] | None = None,
) -> dict[int, int]:
    """
    Create mapping from route_id to its parent route_master_id.

    Args:
        osm_pbf_path (str): Path to local .osm.pbf file.
        route_types (list[str], optional): List of route types.
            Default is ["train", "bus", "tram", "subway", "trolleybus", "light_rail"].

    Returns:
        dict[int, int]: Dictionary mapping route_id to route_master_id.

    Examples:
        >>> mapping = get_route_to_route_master_mapping("lausanne.osm.pbf")
        >>> route_master_id = mapping.get(33560)  # route 33560
        >>> print(f"Route 33560 belongs to route_master {route_master_id}")
        Route 33560 belongs to route_master 3210864

        >>> # Count how many routes have a route_master
        >>> print(f"{len(mapping)} routes belong to a route_master")
        215 routes belong to a route_master
    """
    # Get all route_masters
    route_masters = get_route_masters(osm_pbf_path, route_types)

    # Invert: for each route_master, map its member routes to the route_master_id
    route_to_rm = {}
    for rm_id, rm_data in route_masters.items():
        for route_id in rm_data["member_route_ids"]:
            route_to_rm[route_id] = rm_id

    return route_to_rm


def get_stop_to_stop_area_mapping(
    osm_pbf_path: str,
) -> dict[int, int]:
    """
    Create mapping from stop_id to its parent stop_area_id.

    Args:
        osm_pbf_path (str): Path to local .osm.pbf file.

    Returns:
        dict[int, int]: Dictionary mapping stop_id to stop_area_id.

    Examples:
        >>> mapping = get_stop_to_stop_area_mapping("lausanne.osm.pbf")
        >>> stop_area_id = mapping.get(347549332)  # stop node
        >>> print(f"Stop 347549332 belongs to stop_area {stop_area_id}")
        Stop 347549332 belongs to stop_area 4830941

        >>> # Count how many stops belong to a stop_area
        >>> print(f"{len(mapping)} stops belong to a stop_area")
        30 stops belong to a stop_area
    """
    # Get all stop_areas
    stop_areas = get_stop_areas(osm_pbf_path)

    # Invert: for each stop_area, map its member stops to the stop_area_id
    stop_to_sa = {}
    for sa_id, sa_data in stop_areas.items():
        for stop_id in sa_data["member_stop_ids"]:
            stop_to_sa[stop_id] = sa_id

    return stop_to_sa


def extract_transit_network(
    osm_pbf_path: str,
    route_types: list[str] | None = None,
    include_networks: Optional[list[str]] = None,
    exclude_networks: Optional[list[str]] = None,
    group_by: str = "route",
    include_all_route_stops: bool = True,
    crs: str = "EPSG:4326",
    include_stop_ids: bool = False,
    include_route_ids: bool = False,
    output_dir: Optional[str] = None,
) -> dict:
    """
    Extract transit routes and stops from OSM PBF file.

    This function extracts transit routes and stops from an OSM PBF file, applies
    optional filtering by route type and network, and returns the results as GeoDataFrames.

    Args:
        osm_pbf_path (str): Path to local .osm.pbf file.
        route_types (list[str], optional): List of route types to include (e.g., ["train", "bus"]). Default is ["train", "bus", "tram", "subway", "trolleybus", "light_rail"].
        include_networks (list[str], optional): List of network names to include. If None, includes all networks. Default is None.
        exclude_networks (list[str], optional): List of network names to exclude. If None, excludes no networks. Default is None.
        group_by_ref (bool): If True, groups routes by (ref, network) and combines bidirectional routes. Default is True.
        include_all_route_stops (bool): If True, includes all stops that are members of the extracted routes, even if they fall outside the bounding box
            when using extract_transit_routes(). Default is True.
        crs (str): Coordinate reference system for output GeoDataFrames. Default is "EPSG:4326".
        include_stop_ids (bool): If True, adds stop_ids column to routes GeoDataFrame. Default is False.
        include_route_ids (bool): If True, adds route_ids column to stops GeoDataFrame. Default is False.
        output_dir (str, optional): If provided, saves intermediate GeoParquet files for routes
    """

    # TODO: switch to a bbox-based approach, make it optional as a parameter

    routes = extract_all_transit_routes(
        osm_pbf_path,
        route_types=route_types,
        crs=crs,
        include_stop_ids=include_stop_ids,
    )
    stops = extract_all_transit_stops(
        osm_pbf_path, crs=crs, include_route_ids=include_route_ids
    )

    # Validate conflicting network filters
    if include_networks and exclude_networks:
        include_set = {n.lower() for n in include_networks}
        exclude_set = {n.lower() for n in exclude_networks}
        overlap = include_set & exclude_set
        if overlap:
            raise ValueError(
                f"Networks cannot be in both include and exclude lists: {overlap}"
            )

    # Handle empty inputs
    if routes.empty or stops.empty:
        return {
            "routes": routes.iloc[:0].copy(),  # Empty GDF with same schema
            "stops": stops.iloc[:0].copy(),
        }

    filtered_routes = routes.copy()

    #  Apply network filtering
    if include_networks is not None or exclude_networks is not None:
        # Handle routes with missing network field
        filtered_routes["_network_lower"] = (
            filtered_routes["network"].fillna("").str.strip().str.lower()
        )

        if include_networks is not None:
            include_set = {net.strip().lower() for net in include_networks}
            filtered_routes = filtered_routes[
                filtered_routes["_network_lower"].isin(include_set)
            ]

        if exclude_networks is not None:
            exclude_set = {net.strip().lower() for net in exclude_networks}
            filtered_routes = filtered_routes[
                ~filtered_routes["_network_lower"].isin(exclude_set)
            ]

        # Remove temporary column
        filtered_routes = filtered_routes.drop(columns=["_network_lower"])

    # Grouping disabled
    final_routes = filtered_routes

    # Get stops to return
    if include_all_route_stops and not final_routes.empty:
        # Get all route IDs (handling both grouped and ungrouped routes)
        all_route_ids = set()
        route_stop_mapping = get_route_stop_mapping(
            osm_pbf_path, route_types=route_types
        )

        if "osm_ids" in final_routes.columns:
            # Grouped routes: flatten list of osm_ids
            for osm_ids in final_routes["osm_ids"]:
                if isinstance(osm_ids, list):
                    all_route_ids.update(osm_ids)
                else:
                    all_route_ids.add(osm_ids)
        else:
            # Ungrouped routes: just use osm_id column
            all_route_ids = set(final_routes["osm_id"])

        # Get all stop IDs from these routes using mapping
        all_stop_ids = set()
        for route_id in all_route_ids:
            if route_id in route_stop_mapping:
                all_stop_ids.update(route_stop_mapping[route_id])

        # Filter stops to those in all_stop_ids
        final_stops = stops[stops["osm_id"].isin(list(all_stop_ids))].copy()
    else:
        # Original behavior: only stops within isochrone
        final_stops = stops.copy()

    if output_dir is not None:
        routes_output_path = os.path.join(output_dir, "routes.geoparquet")
        final_routes.to_parquet(routes_output_path)
        print(f"Saved {len(final_routes)} routes to {routes_output_path}")

        stops_output_path = os.path.join(output_dir, "stops.geoparquet")
        final_stops.to_parquet(stops_output_path)
        print(f"Saved {len(final_stops)} stops to {stops_output_path}")

    return {
        "routes": final_routes,  # Unclipped
        "stops": final_stops,  # All stops on routes or original stops if include_all_route_stops=False
    }
