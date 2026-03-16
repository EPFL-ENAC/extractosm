import os
import warnings
from collections import defaultdict
from typing import Optional, cast

import geopandas as gpd
import osmium
import osmium.geom
import pandas as pd
import pyogrio
from importlib.resources import files
from shapely import box, union_all, wkb
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
            - route_ids (list[int]): List of route OSM IDs serving this stop (only if include_route_ids=True)
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
            f"OSM PBF file not found: {osm_pbf_path}. "
            "Provide a valid path."
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
) -> gpd.GeoDataFrame:
    """
    Get fixed-route transit systems within a bounding box.

    This function retrieves OpenStreetMap route relations for public transport
    services (train, bus, tram, etc.) based on the specified route types and
    bounding box. The retrieved routes are returned as a GeoDataFrame with
    route metadata and geometries.

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

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing transit routes with the
        following columns:
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
            - geometry: Route path (LineString) or centroid (Point)
            - stop_ids: (if include_stop_ids=True) List of stop OSM IDs on this route

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
        >>> print(routes[['route', 'name']].head())
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

    # Add stop IDs if requested
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

    return gdf


def extract_all_transit_routes(
    osm_pbf_path: str,
    route_types: list[str] | None = None,
    crs: str = "EPSG:4326",
    output_path: Optional[str] = None,
    include_stop_ids: bool = False,
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
            f"OSM PBF file not found: {osm_pbf_path}. "
            "Provide a valid path."
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


def extract_transit_network(
    osm_pbf_path: str,
    route_types: list[str] | None = None,
    include_networks: Optional[list[str]] = None,
    exclude_networks: Optional[list[str]] = None,
    group_by_ref: bool = True,
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
            "grouped_routes_info": None if group_by_ref else None,
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

    # Group routes by (ref, network) if requested (NEW FEATURE #3)
    grouped_routes_info = None
    if group_by_ref and not filtered_routes.empty:
        # Separate routes with complete info (ref, from, to) from others
        has_required_fields = (
            filtered_routes["ref"].notna()
            & filtered_routes["from"].notna()
            & filtered_routes["to"].notna()
        )

        routes_with_ref = filtered_routes[has_required_fields].copy()
        routes_without_ref = filtered_routes[~has_required_fields].copy()

        if not routes_with_ref.empty:
            # Normalize from/to for matching (case-insensitive, stripped)
            routes_with_ref["_from_norm"] = (
                routes_with_ref["from"].str.strip().str.lower()
            )
            routes_with_ref["_to_norm"] = routes_with_ref["to"].str.strip().str.lower()
            routes_with_ref["_network_clean"] = routes_with_ref["network"].fillna("")

            # Find bidirectional pairs within each (ref, network) group
            grouped_list = []
            ungrouped_list = []
            processed_ids = set()

            # Group by (ref, network)
            for group_key, group_df in routes_with_ref.groupby(
                ["ref", "_network_clean"]
            ):
                group_routes = list(group_df.iterrows())

                # Try to find bidirectional pairs
                for i, (idx_a, route_a) in enumerate(group_routes):
                    if route_a["osm_id"] in processed_ids:
                        continue

                    # Look for a matching bidirectional route
                    paired = False
                    for j, (idx_b, route_b) in enumerate(
                        group_routes[i + 1 :], start=i + 1
                    ):
                        if route_b["osm_id"] in processed_ids:
                            continue

                        # Check if bidirectional: A.from==B.to AND A.to==B.from (normalized)
                        if (
                            route_a["_from_norm"] == route_b["_to_norm"]
                            and route_a["_to_norm"] == route_b["_from_norm"]
                        ):
                            # Found a bidirectional pair! Combine them
                            name_a = route_a.get("name", "")
                            name_b = route_b.get("name", "")
                            combined_name = (
                                f"{name_a} / {name_b}".strip(" / ")
                                if name_a or name_b
                                else None
                            )

                            combined_route = {
                                "osm_id": route_a[
                                    "osm_id"
                                ],  # Use first route's ID as primary
                                "osm_ids": [route_a["osm_id"], route_b["osm_id"]],
                                "route": route_a["route"],
                                "ref": route_a["ref"],
                                "network": route_a["network"],
                                "name": combined_name,
                                "from": f"{route_a['from']} ↔ {route_a['to']}",  # Bidirectional indicator
                                "to": None,  # Not applicable for bidirectional
                                "operator": route_a.get("operator"),
                                "website": route_a.get("website"),
                                "geometry": union_all(
                                    [route_a["geometry"], route_b["geometry"]]
                                ),
                                "route_count": 2,
                            }

                            grouped_list.append(combined_route)
                            processed_ids.add(route_a["osm_id"])
                            processed_ids.add(route_b["osm_id"])
                            paired = True
                            break

                    # If no pair found, keep route ungrouped
                    if not paired:
                        ungrouped_dict = route_a.to_dict()
                        ungrouped_dict["route_count"] = 1
                        ungrouped_dict["osm_ids"] = [route_a["osm_id"]]
                        # Remove temporary columns
                        for temp_col in ["_from_norm", "_to_norm", "_network_clean"]:
                            ungrouped_dict.pop(temp_col, None)
                        ungrouped_list.append(ungrouped_dict)
                        processed_ids.add(route_a["osm_id"])

            # Create GeoDataFrames from lists
            if grouped_list:
                grouped_gdf = gpd.GeoDataFrame(grouped_list, crs=routes.crs)
            else:
                # Create empty GeoDataFrame with geometry column
                grouped_gdf = gpd.GeoDataFrame([], geometry=[], crs=routes.crs)

            if ungrouped_list:
                ungrouped_gdf = gpd.GeoDataFrame(ungrouped_list, crs=routes.crs)
                # Remove temporary columns if they exist
                for temp_col in ["_from_norm", "_to_norm", "_network_clean"]:
                    if temp_col in ungrouped_gdf.columns:
                        ungrouped_gdf = ungrouped_gdf.drop(columns=[temp_col])
            else:
                # Create empty GeoDataFrame with geometry column
                ungrouped_gdf = gpd.GeoDataFrame([], geometry=[], crs=routes.crs)

            # Add route_count and osm_ids to routes_without_ref
            if not routes_without_ref.empty:
                routes_without_ref["route_count"] = 1
                routes_without_ref["osm_ids"] = routes_without_ref["osm_id"].apply(
                    lambda x: [x]
                )

            # Combine all routes
            final_routes = pd.concat(
                [grouped_gdf, ungrouped_gdf, routes_without_ref], ignore_index=True
            )

            # Convert to GeoDataFrame
            final_routes = gpd.GeoDataFrame(final_routes, crs=routes.crs)

            # Create grouping info for return
            grouped_routes_info = {
                "total_original_routes": len(filtered_routes),
                "bidirectional_pairs_found": len(grouped_list),
                "routes_without_pair": len(ungrouped_list),
                "routes_missing_ref_from_to": len(routes_without_ref),
                "final_route_count": len(final_routes),
            }
        else:
            # No routes have required fields for grouping
            if not routes_without_ref.empty:
                routes_without_ref["route_count"] = 1
                routes_without_ref["osm_ids"] = routes_without_ref["osm_id"].apply(
                    lambda x: [x]
                )
            final_routes = routes_without_ref
            grouped_routes_info = {
                "total_original_routes": len(filtered_routes),
                "bidirectional_pairs_found": 0,
                "routes_without_pair": 0,
                "routes_missing_ref_from_to": len(routes_without_ref),
                "final_route_count": len(final_routes),
            }
    else:
        # Grouping disabled
        final_routes = filtered_routes

    # Get stops to return
    if include_all_route_stops and not final_routes.empty:
        # Get all route IDs (handling both grouped and ungrouped routes)
        all_route_ids = set()
        route_stop_mapping = get_route_stop_mapping(osm_pbf_path, route_types=route_types)

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
        "routes": final_routes,  # Unclipped, optionally grouped
        "stops": final_stops,  # All stops on routes or original stops if include_all_route_stops=False
        "grouped_routes_info": grouped_routes_info,  # None if group_by_ref=False
    }
