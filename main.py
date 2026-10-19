from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Dict, List, Tuple, Set
import h3
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from shapely.geometry import Polygon

CENTER_LAT = 56.0
CENTER_LON = 38.0
RADIUS_KM = 7.0
RES = 12


def compute_level_cell(h3_index: str) -> Tuple[int, int]:
    n = int(h3_index, 16)
    q = n // 512
    level = (q % 74) - 120
    cell_id = (q % 100) + 1
    return level, cell_id


def generate_dataset() -> List[Tuple[str, int, int]]:
    center = (CENTER_LAT, CENTER_LON)
    center_cell = h3.latlng_to_cell(CENTER_LAT, CENTER_LON, RES)

    edge_km = h3.average_hexagon_edge_length(RES, unit="km")
    k = int(RADIUS_KM / (1.5 * edge_km)) + 20

    dataset: List[Tuple[str, int, int]] = []
    for h in h3.grid_disk(center_cell, k):
        lat, lng = h3.cell_to_latlng(h)
        if h3.great_circle_distance((lat, lng), center, unit="km") < RADIUS_KM:
            level, cell_id = compute_level_cell(h)
            dataset.append((h, level, cell_id))
    return dataset


def build_parent_indexes(
    dataset: List[Tuple[str, int, int]]
) -> Dict[int, Dict[str, List[List]]]:
    indexes: Dict[int, Dict[str, List[List]]] = {
        r: defaultdict(list) for r in range(RES + 1)
    }
    for h, level, cell_id in dataset:
        entry = [h, level, cell_id]
        indexes[RES][h].append(entry)
        for r in range(RES):
            parent = h3.cell_to_parent(h, r)
            indexes[r][parent].append(entry)
    return indexes


def build_cell_polygons(dataset: List[Tuple[str, int, int]]) -> Dict[str, Polygon]:
    cache: Dict[str, Polygon] = {}
    for h, _, _ in dataset:
        boundary = h3.cell_to_boundary(h)
        cache[h] = Polygon([(lng, lat) for lat, lng in boundary])
    return cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    dataset = generate_dataset()
    app.state.dataset = dataset
    app.state.dataset_by_h3 = {h: (h, level, cell_id) for h, level, cell_id in dataset}
    app.state.parent_indexes = build_parent_indexes(dataset)
    app.state.cell_polygons = build_cell_polygons(dataset)
    app.state.avg_cache = {}
    yield


app = FastAPI(title="H3 Backend Test", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def parse_border(border: str) -> List[Tuple[float, float]]:
    parts = border.split(",")
    if len(parts) < 3:
        raise ValueError("border must contain at least 3 points")

    coords: List[Tuple[float, float]] = []
    for p in parts:
        p = p.strip()
        if "/" not in p:
            raise ValueError("each border point must be lat/lon")
        lat_s, lon_s = p.split("/", 1)
        coords.append((float(lat_s), float(lon_s)))
    return coords


def polygon_candidates(coords: List[Tuple[float, float]]) -> Set[str]:
    ring = coords[:]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return h3.polygon_to_cells(h3.LatLngPoly(ring), RES)


def filter_fully_inside(
    candidates: Set[str],
    coords: List[Tuple[float, float]],
    dataset_by_h3: dict,
    cell_polygons: Dict[str, Polygon],
) -> List[List]:
    border_poly = Polygon([(lon, lat) for lat, lon in coords])
    if not border_poly.is_valid:
        border_poly = border_poly.buffer(0)
    if border_poly.is_empty:
        return []

    result: List[List] = []
    for h in candidates:
        entry = dataset_by_h3.get(h)
        if not entry:
            continue
        cell_poly = cell_polygons.get(h)
        if cell_poly is None:
            continue
        if border_poly.contains(cell_poly):
            result.append([entry[0], entry[1], entry[2]])
    return result


def build_kml(entries: List[List]) -> str:
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        '<Document>',
    ]

    for h, level, cell_id in entries:
        boundary = h3.cell_to_boundary(h)
        coords = [(lng, lat) for lat, lng in boundary]
        if coords[0] != coords[-1]:
            coords.append(coords[0])

        coord_str = " ".join(f"{lng},{lat},0" for lng, lat in coords)
        parts.append(
            f"<Placemark>"
            f"<name>{h}</name>"
            f"<description>level={level}, cell_id={cell_id}</description>"
            f"<Polygon>"
            f"<outerBoundaryIs>"
            f"<LinearRing>"
            f"<coordinates>{coord_str}</coordinates>"
            f"</LinearRing>"
            f"</outerBoundaryIs>"
            f"</Polygon>"
            f"</Placemark>"
        )

    parts.append("</Document></kml>")
    return "\n".join(parts)


@app.get("/hex")
def get_hex(request: Request, hex_index: str = Query(..., alias="hex")):
    if not h3.is_valid_cell(hex_index):
        raise HTTPException(status_code=400, detail="Invalid H3 index")

    q_res = h3.get_resolution(hex_index)
    if q_res > RES:
        return []

    parent_indexes = request.app.state.parent_indexes
    return parent_indexes[q_res].get(hex_index, [])


@app.get("/avg")
def get_avg(request: Request, resolution: int = Query(..., ge=0, le=12)):
    cache = request.app.state.avg_cache
    if resolution in cache:
        return cache[resolution]

    parent_indexes = request.app.state.parent_indexes
    groups: Dict[Tuple[str, int], List[int]] = defaultdict(list)

    for parent, entries in parent_indexes[resolution].items():
        for _, level, cell_id in entries:
            groups[(parent, cell_id)].append(level)

    result = []
    for (parent, cell_id), levels in groups.items():
        levels.sort()
        n = len(levels)
        if n % 2 == 1:
            med = levels[n // 2]
        else:
            med = (levels[n // 2 - 1] + levels[n // 2]) // 2
        result.append([parent, med, cell_id])

    cache[resolution] = result
    return result


@app.get("/bbox")
def get_bbox(request: Request, border: str = Query(...)):
    try:
        coords = parse_border(border)
        candidates = polygon_candidates(coords)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return filter_fully_inside(
        candidates,
        coords,
        request.app.state.dataset_by_h3,
        request.app.state.cell_polygons,
    )


@app.get("/bbox_kml")
def get_bbox_kml(request: Request, border: str = Query(...)):
    try:
        coords = parse_border(border)
        candidates = polygon_candidates(coords)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    entries = filter_fully_inside(
        candidates,
        coords,
        request.app.state.dataset_by_h3,
        request.app.state.cell_polygons,
    )
    kml = build_kml(entries)

    return Response(
        content=kml,
        media_type="application/vnd.google-earth.kml+xml",
        headers={"Content-Disposition": "attachment; filename=bbox.kml"},
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)