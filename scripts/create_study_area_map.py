"""Create the publication study-area map with QGIS/PyQGIS.

Run from the repository root with the QGIS Python environment, for example:

    & "C:\\Program Files\\QGIS 3.44.7\\bin\\python-qgis.bat" `
      scripts/create_study_area_map.py

The script derives the full level-6 HydroBASINS catchment upstream of the
selected GloFAS grid point rather than copying an illustrative basin outline.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

from osgeo import gdal
from PIL import Image
from qgis.core import (
    Qgis,
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsFeature,
    QgsFeatureRequest,
    QgsField,
    QgsFillSymbol,
    QgsGeometry,
    QgsGraduatedSymbolRenderer,
    QgsLayoutExporter,
    QgsLayoutItemLabel,
    QgsLayoutItemMap,
    QgsLayoutItemMapGrid,
    QgsLayoutItemPicture,
    QgsLayoutItemPolyline,
    QgsLayoutItemScaleBar,
    QgsLayoutItemShape,
    QgsLayoutMeasurement,
    QgsLayoutPoint,
    QgsLayoutSize,
    QgsLineSymbol,
    QgsMarkerSymbol,
    QgsPointXY,
    QgsPrintLayout,
    QgsProject,
    QgsRectangle,
    QgsRendererRange,
    QgsSingleSymbolRenderer,
    QgsTextBufferSettings,
    QgsTextFormat,
    QgsUnitTypes,
    QgsVectorFileWriter,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QMetaType, QPointF, Qt
from qgis.PyQt.QtGui import QColor, QFont, QPolygonF

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIRECTORY = REPOSITORY_ROOT / "data" / "external" / "map_sources"
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "Diagram"

CRS = "EPSG:4326"
HARDINGE = (89.03, 24.07)
GLOFAS_POINT = (89.025, 24.075)
ERA5_DOMAIN = QgsRectangle(88.0, 20.7, 92.6, 26.6)
GLOFAS_DOMAIN = QgsRectangle(88.9, 23.9, 89.2, 24.2)
REGIONAL_EXTENT = QgsRectangle(60.0, 0.0, 100.0, 35.0)
DETAIL_EXTENT = QgsRectangle(87.0, 20.0, 95.0, 27.0)

COLORS = {
    "boundary": "#6B7280",
    "country": "#FCFCFC",
    "ocean": "#F7FBFF",
    "basin_fill": "#DDEED7",
    "basin_line": "#6EA66A",
    "bangladesh": "#FBE8A6",
    "river": "#2C86C7",
    "era5": "#2563EB",
    "glofas": "#F97316",
    "hardinge": "#D73027",
    "text": "#111827",
    "grid": "#D7DEE8",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIRECTORY,
        help="Directory created by download_study_area_map_data.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for the map project and exports",
    )
    return parser.parse_args()


def require_layer(path: Path, name: str) -> QgsVectorLayer:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing map layer: {path}. Run "
            "scripts/download_study_area_map_data.py first."
        )
    layer = QgsVectorLayer(str(path), name, "ogr")
    if not layer.isValid():
        raise RuntimeError(f"QGIS could not load {path}")
    return layer


def memory_layer(geometry: str, name: str, fields: list[QgsField]) -> QgsVectorLayer:
    layer = QgsVectorLayer(f"{geometry}?crs={CRS}", name, "memory")
    provider = layer.dataProvider()
    provider.addAttributes(fields)
    layer.updateFields()
    return layer


def add_feature(
    layer: QgsVectorLayer,
    geometry: QgsGeometry,
    attributes: list[object],
) -> None:
    feature = QgsFeature(layer.fields())
    feature.setGeometry(geometry)
    feature.setAttributes(attributes)
    if not layer.dataProvider().addFeature(feature):
        raise RuntimeError(f"Could not add feature to {layer.name()}")


def write_gpkg_layer(
    layer: QgsVectorLayer,
    gpkg_path: Path,
    layer_name: str,
    *,
    create_file: bool,
) -> None:
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.layerName = layer_name
    options.fileEncoding = "UTF-8"
    options.actionOnExistingFile = (
        QgsVectorFileWriter.CreateOrOverwriteFile
        if create_file
        else QgsVectorFileWriter.CreateOrOverwriteLayer
    )
    result = QgsVectorFileWriter.writeAsVectorFormatV3(
        layer,
        str(gpkg_path),
        QgsProject.instance().transformContext(),
        options,
    )
    if result[0] != QgsVectorFileWriter.NoError:
        raise RuntimeError(f"Failed to write {layer_name}: {result}")


def load_gpkg_layer(gpkg_path: Path, layer_name: str) -> QgsVectorLayer:
    return require_layer(
        Path(f"{gpkg_path}|layername={layer_name}"),
        layer_name.replace("_", " ").title(),
    )


def derive_upstream_catchment(
    basins: QgsVectorLayer,
) -> tuple[QgsVectorLayer, QgsGeometry, dict[str, object]]:
    features_by_id: dict[int, QgsFeature] = {}
    upstream_by_downstream: dict[int, list[int]] = defaultdict(list)
    target_id: int | None = None
    target_feature: QgsFeature | None = None
    target_geometry = QgsGeometry.fromPointXY(QgsPointXY(*GLOFAS_POINT))

    for feature in basins.getFeatures():
        basin_id = int(feature["HYBAS_ID"])
        downstream_id = int(feature["NEXT_DOWN"])
        features_by_id[basin_id] = feature
        if downstream_id > 0:
            upstream_by_downstream[downstream_id].append(basin_id)
        if feature.geometry().intersects(target_geometry):
            target_id = basin_id
            target_feature = feature

    if target_id is None or target_feature is None:
        raise RuntimeError("No HydroBASINS polygon contains the GloFAS point")

    selected: set[int] = {target_id}
    queue: deque[int] = deque([target_id])
    while queue:
        downstream = queue.popleft()
        for upstream in upstream_by_downstream.get(downstream, []):
            if upstream not in selected:
                selected.add(upstream)
                queue.append(upstream)

    geometries = [features_by_id[basin_id].geometry() for basin_id in selected]
    dissolved = QgsGeometry.unaryUnion(geometries)
    if dissolved.isEmpty():
        raise RuntimeError("The derived upstream catchment is empty")
    if not dissolved.isGeosValid():
        dissolved = dissolved.makeValid()

    layer = memory_layer(
        "MultiPolygon",
        "Upstream catchment at Hardinge",
        [
            QgsField("TARGET_ID", QMetaType.LongLong),
            QgsField("SUBBASINS", QMetaType.Int),
            QgsField("UP_AREA", QMetaType.Double),
        ],
    )
    add_feature(
        layer,
        dissolved,
        [target_id, len(selected), float(target_feature["UP_AREA"])],
    )
    metadata = {
        "target_hybas_id": target_id,
        "target_subbasin_area_km2": float(target_feature["SUB_AREA"]),
        "reported_upstream_area_km2": float(target_feature["UP_AREA"]),
        "selected_level6_subbasins": len(selected),
        "derived_extent": [
            dissolved.boundingBox().xMinimum(),
            dissolved.boundingBox().yMinimum(),
            dissolved.boundingBox().xMaximum(),
            dissolved.boundingBox().yMaximum(),
        ],
    }
    return layer, dissolved, metadata


def derive_bangladesh(countries: QgsVectorLayer) -> QgsVectorLayer:
    layer = memory_layer(
        "MultiPolygon",
        "Bangladesh",
        [QgsField("NAME", QMetaType.QString)],
    )
    request = QgsFeatureRequest().setFilterExpression(
        '"ADMIN" = \'Bangladesh\' OR "ADM0_A3" = \'BGD\''
    )
    features = list(countries.getFeatures(request))
    if len(features) != 1:
        raise RuntimeError(f"Expected one Bangladesh polygon, found {len(features)}")
    add_feature(layer, features[0].geometry(), ["Bangladesh"])
    return layer


def derive_rivers(
    rivers: QgsVectorLayer,
    catchment: QgsGeometry,
) -> tuple[QgsVectorLayer, QgsVectorLayer, dict[str, int]]:
    fields = [
        QgsField("HYRIV_ID", QMetaType.Int),
        QgsField("UPLAND_SKM", QMetaType.Double),
        QgsField("DIS_AV_CMS", QMetaType.Double),
        QgsField("ORD_STRA", QMetaType.Int),
    ]
    context = memory_layer("MultiLineString", "Ganges river network", fields)
    detail = memory_layer("MultiLineString", "Regional rivers", fields)
    detail_clip = QgsGeometry.fromRect(DETAIL_EXTENT)

    request = QgsFeatureRequest()
    request.setFilterRect(REGIONAL_EXTENT)
    request.setFilterExpression('"UPLAND_SKM" >= 500')
    for feature in rivers.getFeatures(request):
        geometry = feature.geometry()
        if geometry.isEmpty():
            continue
        attributes = [
            int(feature["HYRIV_ID"]),
            float(feature["UPLAND_SKM"]),
            float(feature["DIS_AV_CMS"]),
            int(feature["ORD_STRA"]),
        ]
        upland_area = attributes[1]

        if geometry.intersects(detail_clip):
            clipped = geometry.intersection(detail_clip)
            if not clipped.isEmpty():
                clipped.convertToMultiType()
                add_feature(detail, clipped, attributes)

        if upland_area >= 2500 and geometry.intersects(catchment):
            clipped = geometry.intersection(catchment)
            if not clipped.isEmpty():
                clipped.convertToMultiType()
                add_feature(context, clipped, attributes)

    return context, detail, {
        "context_river_reaches": context.featureCount(),
        "detail_river_reaches": detail.featureCount(),
    }


def derive_points_and_domains() -> dict[str, QgsVectorLayer]:
    hardinge = memory_layer(
        "Point",
        "Hardinge Bridge and NASA POWER point",
        [QgsField("NAME", QMetaType.QString)],
    )
    add_feature(
        hardinge,
        QgsGeometry.fromPointXY(QgsPointXY(*HARDINGE)),
        ["Hardinge Bridge analysis point and NASA POWER point"],
    )
    glofas_point = memory_layer(
        "Point",
        "Selected GloFAS grid point",
        [QgsField("NAME", QMetaType.QString)],
    )
    add_feature(
        glofas_point,
        QgsGeometry.fromPointXY(QgsPointXY(*GLOFAS_POINT)),
        ["Selected GloFAS grid point"],
    )
    era5 = memory_layer(
        "Polygon",
        "ERA5-Land averaging domain",
        [QgsField("NAME", QMetaType.QString)],
    )
    add_feature(era5, QgsGeometry.fromRect(ERA5_DOMAIN), ["ERA5-Land domain"])
    glofas_domain = memory_layer(
        "Polygon",
        "GloFAS download domain",
        [QgsField("NAME", QMetaType.QString)],
    )
    add_feature(
        glofas_domain,
        QgsGeometry.fromRect(GLOFAS_DOMAIN),
        ["GloFAS download domain"],
    )
    return {
        "hardinge_point": hardinge,
        "glofas_grid_point": glofas_point,
        "era5_domain": era5,
        "glofas_download_domain": glofas_domain,
    }


def style_fill(
    layer: QgsVectorLayer,
    fill: str | None,
    outline: str,
    *,
    width: float = 0.25,
    opacity: float = 1.0,
    outline_style: str = "solid",
) -> None:
    properties = {
        "color": fill or "255,255,255,0",
        "outline_color": outline,
        "outline_width": str(width),
        "outline_style": outline_style,
    }
    if fill is None:
        properties["style"] = "no"
    symbol = QgsFillSymbol.createSimple(properties)
    symbol.setOpacity(opacity)
    layer.setRenderer(QgsSingleSymbolRenderer(symbol))


def style_rivers(layer: QgsVectorLayer, *, detailed: bool) -> None:
    ranges: list[QgsRendererRange] = []
    definitions = (
        [(500, 2000, 0.12), (2000, 10000, 0.20), (10000, 100000, 0.33), (100000, 2e6, 0.52)]
        if detailed
        else [(2500, 10000, 0.15), (10000, 100000, 0.25), (100000, 2e6, 0.45)]
    )
    for lower, upper, width in definitions:
        symbol = QgsLineSymbol.createSimple(
            {"line_color": COLORS["river"], "line_width": str(width)}
        )
        ranges.append(QgsRendererRange(lower, upper, symbol, f"{lower:g}-{upper:g}"))
    layer.setRenderer(QgsGraduatedSymbolRenderer("UPLAND_SKM", ranges))


def style_layers(layers: dict[str, QgsVectorLayer]) -> None:
    style_fill(layers["countries"], COLORS["country"], COLORS["boundary"], width=0.22)
    style_fill(
        layers["upstream_catchment"],
        COLORS["basin_fill"],
        COLORS["basin_line"],
        width=0.35,
        opacity=0.72,
    )
    style_fill(
        layers["bangladesh"],
        COLORS["bangladesh"],
        "#9A8A55",
        width=0.30,
        opacity=0.78,
    )
    style_rivers(layers["context_rivers"], detailed=False)
    style_rivers(layers["detail_rivers"], detailed=True)
    style_fill(
        layers["era5_domain"],
        None,
        COLORS["era5"],
        width=0.55,
        outline_style="dash",
    )
    style_fill(
        layers["glofas_download_domain"],
        None,
        COLORS["glofas"],
        width=0.65,
    )
    layers["hardinge_point"].setRenderer(
        QgsSingleSymbolRenderer(
            QgsMarkerSymbol.createSimple(
                {
                    "name": "star",
                    "color": COLORS["hardinge"],
                    "outline_color": "#111111",
                    "outline_width": "0.35",
                    "size": "4.3",
                }
            )
        )
    )
    layers["glofas_grid_point"].setRenderer(
        QgsSingleSymbolRenderer(
            QgsMarkerSymbol.createSimple(
                {
                    "name": "circle",
                    "color": "#FFFFFF",
                    "outline_color": "#111111",
                    "outline_width": "0.45",
                    "size": "3.2",
                }
            )
        )
    )


def text_format(
    size: float,
    *,
    bold: bool = False,
    color: str | None = None,
    halo_size: float = 0.0,
) -> QgsTextFormat:
    font = QFont("Arial")
    font.setBold(bold)
    formatting = QgsTextFormat()
    formatting.setFont(font)
    formatting.setSize(size)
    formatting.setColor(QColor(color or COLORS["text"]))
    if halo_size > 0:
        buffer = QgsTextBufferSettings()
        buffer.setEnabled(True)
        buffer.setSize(halo_size)
        buffer.setColor(QColor("#FFFFFF"))
        formatting.setBuffer(buffer)
    return formatting


def add_label(
    layout: QgsPrintLayout,
    text: str,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    size: float = 8,
    bold: bool = False,
    color: str | None = None,
    align: Qt.AlignmentFlag = Qt.AlignLeft,
    halo_size: float = 0.0,
    background: str | None = None,
    frame_color: str = "#CBD5E1",
    margin: float = 0.0,
) -> QgsLayoutItemLabel:
    label = QgsLayoutItemLabel(layout)
    label.setText(text)
    label.setTextFormat(
        text_format(size, bold=bold, color=color, halo_size=halo_size)
    )
    label.setHAlign(align)
    label.setVAlign(Qt.AlignVCenter)
    label.setMargin(margin)
    if background is not None:
        label.setBackgroundEnabled(True)
        label.setBackgroundColor(QColor(background))
        label.setFrameEnabled(True)
        label.setFrameStrokeColor(QColor(frame_color))
        label.setFrameStrokeWidth(QgsLayoutMeasurement(0.18))
    layout.addLayoutItem(label)
    label.attemptMove(QgsLayoutPoint(x, y, QgsUnitTypes.LayoutMillimeters))
    label.attemptResize(QgsLayoutSize(width, height, QgsUnitTypes.LayoutMillimeters))
    return label


def add_line(
    layout: QgsPrintLayout,
    points: list[tuple[float, float]],
    *,
    color: str = "#111111",
    width: float = 0.28,
    style: str = "solid",
) -> QgsLayoutItemPolyline:
    polygon = QPolygonF([QPointF(x, y) for x, y in points])
    line = QgsLayoutItemPolyline(polygon, layout)
    line.setSymbol(
        QgsLineSymbol.createSimple(
            {"line_color": color, "line_width": str(width), "line_style": style}
        )
    )
    layout.addLayoutItem(line)
    return line


def add_rectangle(
    layout: QgsPrintLayout,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str | None = None,
    outline: str = "#111111",
    line_width: float = 0.25,
    line_style: str = "solid",
) -> QgsLayoutItemShape:
    shape = QgsLayoutItemShape(layout)
    shape.setShapeType(QgsLayoutItemShape.Rectangle)
    properties = {
        "color": fill or "255,255,255,0",
        "outline_color": outline,
        "outline_width": str(line_width),
        "outline_style": line_style,
    }
    if fill is None:
        properties["style"] = "no"
    shape.setSymbol(QgsFillSymbol.createSimple(properties))
    layout.addLayoutItem(shape)
    shape.attemptMove(QgsLayoutPoint(x, y, QgsUnitTypes.LayoutMillimeters))
    shape.attemptResize(QgsLayoutSize(width, height, QgsUnitTypes.LayoutMillimeters))
    return shape


def add_ellipse(
    layout: QgsPrintLayout,
    x: float,
    y: float,
    diameter: float,
    *,
    fill: str = "#FFFFFF",
    outline: str = "#111111",
) -> QgsLayoutItemShape:
    shape = QgsLayoutItemShape(layout)
    shape.setShapeType(QgsLayoutItemShape.Ellipse)
    shape.setSymbol(
        QgsFillSymbol.createSimple(
            {"color": fill, "outline_color": outline, "outline_width": "0.45"}
        )
    )
    layout.addLayoutItem(shape)
    shape.attemptMove(QgsLayoutPoint(x, y, QgsUnitTypes.LayoutMillimeters))
    shape.attemptResize(
        QgsLayoutSize(diameter, diameter, QgsUnitTypes.LayoutMillimeters)
    )
    return shape


def add_map_grid(
    map_item: QgsLayoutItemMap,
    *,
    interval_x: float,
    interval_y: float,
) -> None:
    grid = QgsLayoutItemMapGrid("Coordinate grid", map_item)
    grid.setEnabled(True)
    grid.setStyle(QgsLayoutItemMapGrid.Solid)
    grid.setIntervalX(interval_x)
    grid.setIntervalY(interval_y)
    grid.setLineSymbol(
        QgsLineSymbol.createSimple(
            {"line_color": COLORS["grid"], "line_width": "0.16"}
        )
    )
    grid.setFrameStyle(QgsLayoutItemMapGrid.NoFrame)
    grid.setAnnotationEnabled(True)
    grid.setAnnotationFormat(QgsLayoutItemMapGrid.DecimalWithSuffix)
    grid.setAnnotationPrecision(0)
    grid.setAnnotationTextFormat(text_format(6.7, color="#374151"))
    grid.setAnnotationPosition(
        QgsLayoutItemMapGrid.OutsideMapFrame,
        QgsLayoutItemMapGrid.Left,
    )
    grid.setAnnotationPosition(
        QgsLayoutItemMapGrid.OutsideMapFrame,
        QgsLayoutItemMapGrid.Bottom,
    )
    grid.setAnnotationDisplay(QgsLayoutItemMapGrid.HideAll, QgsLayoutItemMapGrid.Top)
    grid.setAnnotationDisplay(QgsLayoutItemMapGrid.HideAll, QgsLayoutItemMapGrid.Right)
    grid.setAnnotationFrameDistance(1.0)
    map_item.grids().addGrid(grid)


def add_map(
    layout: QgsPrintLayout,
    x: float,
    y: float,
    width: float,
    height: float,
    extent: QgsRectangle,
    layers: list[QgsVectorLayer],
    *,
    grid_x: float,
    grid_y: float,
) -> QgsLayoutItemMap:
    item = QgsLayoutItemMap(layout)
    item.setBackgroundEnabled(True)
    item.setBackgroundColor(QColor(COLORS["ocean"]))
    item.setFrameEnabled(True)
    item.setFrameStrokeColor(QColor("#111111"))
    item.setFrameStrokeWidth(QgsLayoutMeasurement(0.35))
    item.setCrs(QgsCoordinateReferenceSystem(CRS))
    item.setLayers(layers)
    item.setKeepLayerSet(True)
    layout.addLayoutItem(item)
    item.attemptMove(QgsLayoutPoint(x, y, QgsUnitTypes.LayoutMillimeters))
    item.attemptResize(QgsLayoutSize(width, height, QgsUnitTypes.LayoutMillimeters))
    item.setExtent(extent)
    add_map_grid(item, interval_x=grid_x, interval_y=grid_y)
    return item


def map_coordinate_to_layout(
    item: QgsLayoutItemMap,
    x_mm: float,
    y_mm: float,
    width_mm: float,
    height_mm: float,
    longitude: float,
    latitude: float,
) -> tuple[float, float]:
    extent = item.extent()
    x = x_mm + (longitude - extent.xMinimum()) / extent.width() * width_mm
    y = y_mm + (extent.yMaximum() - latitude) / extent.height() * height_mm
    return x, y


def add_manual_legend(layout: QgsPrintLayout) -> None:
    y1, y2 = 172.0, 187.0
    add_rectangle(layout, 15, y1 + 2, 10, 5, fill=COLORS["country"], outline=COLORS["boundary"])
    add_label(layout, "Country boundary", 28, y1, 34, 9, size=7.2)
    add_line(layout, [(67, y1 + 4.5), (78, y1 + 4.5)], color=COLORS["river"], width=0.45)
    add_label(layout, "Ganges network and major tributaries", 81, y1, 55, 9, size=7.2)
    add_rectangle(layout, 143, y1 + 2, 10, 5, fill=COLORS["basin_fill"], outline=COLORS["basin_line"])
    add_label(layout, "Upstream Ganges catchment at Hardinge", 156, y1, 57, 9, size=7.2)
    add_rectangle(layout, 220, y1 + 2, 10, 5, fill=COLORS["bangladesh"], outline="#9A8A55")
    add_label(layout, "Bangladesh", 233, y1, 30, 9, size=7.2)

    add_label(layout, "★", 14, y2, 10, 9, size=13, bold=True, color=COLORS["hardinge"], align=Qt.AlignCenter)
    add_label(layout, "Hardinge Bridge / NASA POWER point", 28, y2, 51, 9, size=7.2)
    add_ellipse(layout, 86, y2 + 2.5, 4, fill="#FFFFFF", outline="#111111")
    add_label(layout, "Selected GloFAS grid point", 93, y2, 44, 9, size=7.2)
    add_rectangle(layout, 143, y2 + 2, 10, 6, fill=None, outline=COLORS["era5"], line_width=0.55, line_style="dash")
    add_label(layout, "ERA5-Land averaging domain", 156, y2, 49, 9, size=7.2)
    add_rectangle(layout, 212, y2 + 2, 10, 6, fill=None, outline=COLORS["glofas"], line_width=0.60)
    add_label(layout, "GloFAS download domain", 225, y2, 42, 9, size=7.2)


def build_layout(
    project: QgsProject,
    layers: dict[str, QgsVectorLayer],
) -> QgsPrintLayout:
    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName("Fig1 Study Area and Data Coverage")
    page = layout.pageCollection().page(0)
    page.setPageSize(QgsLayoutSize(330, 220, QgsUnitTypes.LayoutMillimeters))

    left_x, right_x, map_y = 13.0, 171.0, 18.0
    # Both extents have an 8:7 aspect ratio, so QGIS does not resize either
    # map frame while preserving the requested geographic coverage.
    map_width, map_height = 146.0, 127.75
    regional_layers = [
        layers["hardinge_point"],
        layers["context_rivers"],
        layers["bangladesh"],
        layers["upstream_catchment"],
        layers["countries"],
    ]
    detail_layers = [
        layers["hardinge_point"],
        layers["glofas_grid_point"],
        layers["glofas_download_domain"],
        layers["era5_domain"],
        layers["detail_rivers"],
        layers["bangladesh"],
        layers["countries"],
    ]
    regional = add_map(
        layout,
        left_x,
        map_y,
        map_width,
        map_height,
        REGIONAL_EXTENT,
        regional_layers,
        grid_x=5,
        grid_y=5,
    )
    detail = add_map(
        layout,
        right_x,
        map_y,
        map_width,
        map_height,
        DETAIL_EXTENT,
        detail_layers,
        grid_x=1,
        grid_y=1,
    )

    add_label(layout, "a", 4, 4, 10, 10, size=13, bold=True)
    add_label(layout, "Regional context", left_x, 5, map_width, 10, size=11, align=Qt.AlignCenter)
    add_label(layout, "b", 162, 4, 10, 10, size=13, bold=True)
    add_label(layout, "Detailed study area", right_x, 5, map_width, 10, size=11, align=Qt.AlignCenter)

    country_labels = {
        "PAKISTAN": (71.7, 29.0),
        "INDIA": (78.5, 22.2),
        "NEPAL": (84.0, 28.2),
        "BHUTAN": (90.4, 27.8),
        "CHINA": (87.2, 32.3),
        "MYANMAR": (94.1, 22.3),
    }
    for text, (longitude, latitude) in country_labels.items():
        x, y = map_coordinate_to_layout(
            regional,
            left_x,
            map_y,
            map_width,
            map_height,
            longitude,
            latitude,
        )
        add_label(
            layout,
            text,
            x - 11,
            y - 3,
            22,
            6,
            size=7.2,
            bold=True,
            align=Qt.AlignCenter,
            halo_size=0.75,
        )
    bay_x, bay_y = map_coordinate_to_layout(
        regional, left_x, map_y, map_width, map_height, 88.0, 19.2
    )
    add_label(
        layout,
        "Bay of Bengal",
        bay_x - 17,
        bay_y - 3,
        34,
        6,
        size=7.4,
        color="#2563A8",
        align=Qt.AlignCenter,
        halo_size=0.7,
    )

    bd_x, bd_y = map_coordinate_to_layout(
        detail, right_x, map_y, map_width, map_height, 90.1, 24.5
    )
    add_label(
        layout,
        "BANGLADESH",
        bd_x - 18,
        bd_y - 3,
        36,
        7,
        size=9,
        bold=True,
        align=Qt.AlignCenter,
        halo_size=0.9,
    )

    hardinge_x, hardinge_y = map_coordinate_to_layout(
        detail, right_x, map_y, map_width, map_height, *HARDINGE
    )
    era5_x, era5_y = map_coordinate_to_layout(
        detail,
        right_x,
        map_y,
        map_width,
        map_height,
        ERA5_DOMAIN.xMaximum(),
        ERA5_DOMAIN.yMaximum(),
    )
    hardinge_label_x = 269.0
    add_line(
        layout,
        [
            (hardinge_x + 2, hardinge_y - 1),
            (252, hardinge_y - 1),
            (hardinge_label_x - 5, 80),
        ],
        width=0.28,
    )
    add_label(
        layout,
        "Hardinge Bridge analysis point\nand NASA POWER point\n(24.07° N, 89.03° E)",
        hardinge_label_x,
        69,
        43,
        24,
        size=7.0,
        halo_size=0.8,
    )
    add_line(
        layout,
        [
            (hardinge_x + 2, hardinge_y + 3),
            (246, 92),
            (hardinge_label_x - 5, 109),
        ],
        color="#374151",
        width=0.28,
    )
    add_label(
        layout,
        "Selected GloFAS grid point\n(cell centre: 24.075° N,\n89.025° E)",
        hardinge_label_x,
        99,
        43,
        22,
        size=7.0,
        halo_size=0.8,
    )
    add_line(
        layout,
        [(era5_x, era5_y), (276, 39)],
        color=COLORS["era5"],
        width=0.35,
    )
    add_label(
        layout,
        "ERA5-Land averaging domain\n(regional box, not the full\nupstream catchment)",
        280,
        28,
        34,
        24,
        size=7.0,
        color=COLORS["era5"],
        halo_size=0.75,
    )

    add_manual_legend(layout)

    north_arrow = QgsLayoutItemPicture(layout)
    north_arrow.setPicturePath(
        str(
            Path(QgsApplication.prefixPath())
            / "svg"
            / "arrows"
            / "NorthArrow_04.svg"
        )
    )
    north_arrow.setLinkedMap(detail)
    layout.addLayoutItem(north_arrow)
    north_arrow.attemptMove(QgsLayoutPoint(274, 171, QgsUnitTypes.LayoutMillimeters))
    north_arrow.attemptResize(QgsLayoutSize(11, 17, QgsUnitTypes.LayoutMillimeters))
    add_label(layout, "N", 274, 167, 11, 6, size=7.5, bold=True, align=Qt.AlignCenter)

    scale_bar = QgsLayoutItemScaleBar(layout)
    scale_bar.setStyle("Single Box")
    scale_bar.setLinkedMap(detail)
    scale_bar.setUnits(QgsUnitTypes.DistanceKilometers)
    scale_bar.setNumberOfSegments(4)
    scale_bar.setNumberOfSegmentsLeft(0)
    scale_bar.setUnitsPerSegment(50)
    scale_bar.setUnitLabel("km")
    scale_bar.setHeight(3.5)
    scale_bar.setTextFormat(text_format(6.5))
    scale_bar.setFillSymbol(
        QgsFillSymbol.createSimple({"color": "#111111", "outline_style": "no"})
    )
    scale_bar.setAlternateFillSymbol(
        QgsFillSymbol.createSimple({"color": "#FFFFFF", "outline_style": "no"})
    )
    scale_bar.setLineSymbol(
        QgsLineSymbol.createSimple({"color": "#111111", "width": "0.25"})
    )
    layout.addLayoutItem(scale_bar)
    scale_bar.attemptMove(QgsLayoutPoint(285, 177, QgsUnitTypes.LayoutMillimeters))
    scale_bar.update()

    add_label(
        layout,
        "Boundaries: Natural Earth 5.1.1 · Catchment and rivers: HydroBASINS/HydroRIVERS v1 · CRS: WGS 84",
        13,
        207,
        304,
        7,
        size=6.4,
        color="#4B5563",
        align=Qt.AlignCenter,
    )
    # The figure is a publication graphic, not a georeferenced raster product.
    # Clearing the reference map prevents GDAL from trying to update PNG
    # georeferencing metadata after export (PNG is a create-only GDAL driver).
    layout.setReferenceMap(None)
    return layout


def export_layout(layout: QgsPrintLayout, output_directory: Path) -> dict[str, str]:
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = output_directory / "Fig1_Study_Area_and_Data_Coverage"
    exporter = QgsLayoutExporter(layout)

    image_settings = QgsLayoutExporter.ImageExportSettings()
    image_settings.dpi = 600
    image_settings.generateWorldFile = False
    png_path = stem.with_suffix(".png")
    png_path.unlink(missing_ok=True)
    # QGIS 3.44 asks GDAL to reopen the newly created PNG in update mode to
    # attach optional metadata. The PNG driver is create-only, so silence that
    # known non-fatal GDAL message while still checking QGIS's export result.
    gdal.PushErrorHandler("CPLQuietErrorHandler")
    try:
        result = exporter.exportToImage(str(png_path), image_settings)
    finally:
        gdal.PopErrorHandler()
    if result != QgsLayoutExporter.Success:
        raise RuntimeError(f"PNG export failed with code {result}")

    tiff_path = stem.with_suffix(".tiff")
    tiff_path.unlink(missing_ok=True)
    result = exporter.exportToImage(str(tiff_path), image_settings)
    if result != QgsLayoutExporter.Success:
        raise RuntimeError(f"TIFF export failed with code {result}")
    # A multi-panel publication figure cannot have one valid map transform.
    # QGIS automatically embeds its reference map's CRS in TIFF exports, so
    # rewrite the pixels without that misleading geospatial metadata.
    clean_tiff_path = tiff_path.with_name(f"{tiff_path.stem}.clean.tiff")
    clean_tiff_path.unlink(missing_ok=True)
    with Image.open(tiff_path) as image:
        image.load()
        image.save(
            clean_tiff_path,
            format="TIFF",
            compression="tiff_lzw",
            dpi=(600, 600),
        )
    clean_tiff_path.replace(tiff_path)

    pdf_settings = QgsLayoutExporter.PdfExportSettings()
    pdf_settings.dpi = 600
    pdf_settings.forceVectorOutput = True
    pdf_path = stem.with_suffix(".pdf")
    pdf_path.unlink(missing_ok=True)
    result = exporter.exportToPdf(str(pdf_path), pdf_settings)
    if result != QgsLayoutExporter.Success:
        raise RuntimeError(f"PDF export failed with code {result}")

    svg_settings = QgsLayoutExporter.SvgExportSettings()
    svg_settings.forceVectorOutput = True
    svg_path = stem.with_suffix(".svg")
    svg_path.unlink(missing_ok=True)
    result = exporter.exportToSvg(str(svg_path), svg_settings)
    if result != QgsLayoutExporter.Success:
        raise RuntimeError(f"SVG export failed with code {result}")

    return {
        "png": str(png_path),
        "tiff": str(tiff_path),
        "pdf": str(pdf_path),
        "svg": str(svg_path),
    }


def main() -> None:
    arguments = parse_arguments()
    data_directory = arguments.data_dir.resolve()
    output_directory = arguments.output_dir.resolve()

    application = QgsApplication([], False)
    application.initQgis()
    try:
        project = QgsProject.instance()
        project.clear()
        project.setCrs(QgsCoordinateReferenceSystem(CRS))
        project.setEllipsoid("WGS84")
        project.setFilePathStorage(Qgis.FilePathType.Relative)

        countries_path = (
            data_directory
            / "ne_10m_admin_0_countries"
            / "ne_10m_admin_0_countries.shp"
        )
        basins_path = (
            data_directory / "hybas_as_lev06_v1c" / "hybas_as_lev06_v1c.shp"
        )
        rivers_path = (
            data_directory
            / "HydroRIVERS_v10_as_shp"
            / "HydroRIVERS_v10_as_shp"
            / "HydroRIVERS_v10_as.shp"
        )
        countries = require_layer(countries_path, "Country boundaries")
        basins = require_layer(basins_path, "HydroBASINS level 6")
        rivers = require_layer(rivers_path, "HydroRIVERS Asia")

        upstream, catchment_geometry, catchment_metadata = derive_upstream_catchment(
            basins
        )
        bangladesh = derive_bangladesh(countries)
        context_rivers, detail_rivers, river_metadata = derive_rivers(
            rivers, catchment_geometry
        )
        derived_layers = {
            "upstream_catchment": upstream,
            "bangladesh": bangladesh,
            "context_rivers": context_rivers,
            "detail_rivers": detail_rivers,
            **derive_points_and_domains(),
        }

        derived_directory = data_directory / "derived"
        derived_directory.mkdir(parents=True, exist_ok=True)
        gpkg_path = derived_directory / "study_area_map_layers.gpkg"
        gpkg_path.unlink(missing_ok=True)
        first = True
        for layer_name, layer in derived_layers.items():
            write_gpkg_layer(layer, gpkg_path, layer_name, create_file=first)
            first = False

        layers: dict[str, QgsVectorLayer] = {"countries": countries}
        for layer_name in derived_layers:
            layer = QgsVectorLayer(
                f"{gpkg_path}|layername={layer_name}",
                layer_name.replace("_", " ").title(),
                "ogr",
            )
            if not layer.isValid():
                raise RuntimeError(f"Could not reload {layer_name} from {gpkg_path}")
            layers[layer_name] = layer

        style_layers(layers)
        for layer in reversed(list(layers.values())):
            project.addMapLayer(layer)

        layout = build_layout(project, layers)
        project.layoutManager().addLayout(layout)
        output_directory.mkdir(parents=True, exist_ok=True)
        project_path = output_directory / "Fig1_Study_Area_and_Data_Coverage.qgz"
        project.setFileName(str(project_path))
        if not project.write():
            raise RuntimeError(f"Could not write QGIS project to {project_path}")

        exports = export_layout(layout, output_directory)
        metadata = {
            "hardinge_analysis_point": {"longitude": HARDINGE[0], "latitude": HARDINGE[1]},
            "glofas_grid_point": {
                "longitude": GLOFAS_POINT[0],
                "latitude": GLOFAS_POINT[1],
            },
            "era5_domain": [88.0, 20.7, 92.6, 26.6],
            "glofas_download_domain": [88.9, 23.9, 89.2, 24.2],
            "catchment": catchment_metadata,
            "rivers": river_metadata,
            "sources": {
                "countries": "Natural Earth Admin-0 countries 1:10m v5.1.1",
                "catchment": "HydroBASINS Asia level 6 v1c",
                "rivers": "HydroRIVERS Asia v1.0",
            },
            "project": str(project_path),
            "exports": exports,
        }
        metadata_path = output_directory / "Fig1_Study_Area_and_Data_Coverage.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata, indent=2))
    finally:
        QgsProject.instance().clear()
        # On Windows/QGIS 3.44, exitQgis() can re-enter GDAL cleanup after a
        # high-resolution PNG export and terminate Python with an access
        # violation. The standalone process releases QGIS resources safely on
        # interpreter shutdown, after the project has been cleared here.


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - command-line failure reporting
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
